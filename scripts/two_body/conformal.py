# Uncertainty for --conformal, shared by the in-sequence (per-timestep) and whole-trajectory
# classifiers: temperature scaling (Guo et al. 2017) for calibrated per-class probabilities, then
# class-conditional (Mondrian) split conformal prediction sets with
# P(true class in set | true class = c) >= 1 - alpha for EVERY class c. Post-hoc on trained models:
# no retraining, no effect on any other reported number.
#
# Everything here takes [N,T,C] scores and [N,T] labels. A whole-trajectory classifier is the T = 1
# case: pass scores[:, None] and labels[:, None], and each "frame" is one trajectory.
#
# The printed block is parsed by gmat/data/seqClassification/aggregateManuscript.py (and, through it,
# classification/aggregateManuscriptTotal.py) -- keep reportConformal's row format in sync with
# CONF_ROW_RE there. Two log-parser constraints shape it: never follow a percentage with "(a/b)"
# (displaySeqLogData's accuracy regexes), and never print a bare integer token (displayLogData's
# confusion-matrix parser reads any line of integers after "Confusion Matrix" as a matrix row).
import numpy as np
import torch
from torch import nn, optim

HEADER = "Uncertainty (temperature scaling + Mondrian conformal)"


def printHeader(name):
    """'<model> [<approach>] Uncertainty (...)' -- the line the aggregators key each block on."""
    print(f"\n{name} {HEADER}")


def fitTemperature(logits, labels, pad_idx=-100):
    """One scalar T minimizing UNWEIGHTED NLL of softmax(logits / T) on valid frames. Unweighted on
    purpose: training may use class-weighted loss, but calibration targets the true frequencies.
    A single T cannot undo that weighting's prior shift -- the Mondrian quantiles absorb it.
    Strong-Wolfe line search: without it LBFGS stalls near T = 1 when the optimum is decades away
    (ridge scores), checked against a grid search."""
    m = labels != pad_idx
    z, y = logits[m], labels[m]
    log_t = torch.zeros(1, dtype=z.dtype, requires_grad=True)
    opt = optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def ece(probs, y, n_bins=15):
    """Top-label expected calibration error. probs [M,C], y [M] (numpy)."""
    conf, pred = probs.max(1), probs.argmax(1)
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return sum(abs((pred[b == k] == y[b == k]).mean() - conf[b == k].mean()) * (b == k).mean()
               for k in range(n_bins) if (b == k).any())


def calibratedProbs(z_cal, y_cal, z_eval, y_eval, label, pad_idx=-100):
    """Fit T on the calibration split's scores [N,T,C] against labels [N,T]; returns temperature-
    scaled probabilities for both splits. Scores need not be logits: softmax(z / T) with a fitted T
    is a valid calibration map for any per-class score (log-probs, ridge outputs)."""
    zc, ze = torch.as_tensor(z_cal, dtype=torch.float64), torch.as_tensor(z_eval, dtype=torch.float64)
    yc, ye = (torch.as_tensor(np.asarray(y, dtype=np.int64)) for y in (y_cal, y_eval))
    T = fitTemperature(zc, yc, pad_idx)
    m = ye != pad_idx
    ece_raw = ece(torch.softmax(ze[m], -1).numpy(), ye[m].numpy())
    ece_t = ece(torch.softmax(ze[m] / T, -1).numpy(), ye[m].numpy())
    print(f"  {label}: T = {T:.3f}, eval ECE {ece_raw:.2%} -> {ece_t:.2%}")
    return torch.softmax(zc / T, -1).numpy(), torch.softmax(ze / T, -1).numpy()


def mondrianThresholds(probs, y, alpha, pad_idx=-100):
    """Per-class conformal thresholds q_c on the score 1 - p_c. probs [N,T,C], y [N,T].
    Timesteps within a trajectory are dependent, so the finite-sample correction counts calibration
    TRAJECTORIES containing class c, not frames (frames would overstate the guarantee). At T = 1
    that is the textbook split-conformal correction."""
    C = probs.shape[-1]
    q = np.full(C, np.inf)  # class absent from calibration -> always in the set (no evidence to exclude)
    for c in range(C):
        s = 1.0 - probs[y == c][:, c]
        n_traj = int((y == c).any(axis=1).sum())
        if n_traj:
            q[c] = np.quantile(s, min(1.0, (1 - alpha) * (1 + 1 / n_traj)), method="higher")
    return q


def predictionSets(probs, q):
    """[..., C] boolean membership: class c is in the set iff 1 - p_c <= q_c."""
    return (1.0 - probs) <= q


def reportConformal(cal_probs, cal_y, eval_probs, eval_y, class_names, alpha, pad_idx=-100, unit="frames"):
    """One row per true class. 'asserted at' is the calibrated probability a class needs to enter a
    set (1 - q_c). Every eval frame of that class lands in exactly one outcome: certain (set is
    {true class}), ambiguous (true class plus others), abstain (empty set -- no class cleared its
    bar) or wrong (non-empty set without the true class). covered = certain + ambiguous is the
    >= 1 - alpha guarantee; the other columns are what distinguishes one model from another.
    'precision' runs the other direction: of the frames whose set is exactly {c}, the fraction that
    truly are c -- "when the model asserts c alone, it is right this often"."""
    q = mondrianThresholds(cal_probs, cal_y, alpha, pad_idx)
    m = eval_y != pad_idx
    sets, y = predictionSets(eval_probs[m], q), eval_y[m]
    size = sets.sum(1)
    hit = sets[np.arange(len(y)), y]
    outcomes = np.stack([hit, hit & (size == 1), hit & (size > 1), size == 0, ~hit & (size > 0)], axis=1)
    widths = (9, 9, 11, 9, 8)
    fmt = lambda k: "".join(f"{v:>{w}.1%}" for v, w in zip(outcomes[k].mean(0), widths))
    single = size == 1
    prec = lambda named: f"{(hit & named).sum() / named.sum():>11.1%}" if named.any() else f"{'-':>11}"

    print(f"  Conformal sets at {1 - alpha:.0%} per-class coverage, n={len(y)} eval {unit}")
    print(f"  {'class':<12}{'asserted at':>14}{'precision':>11}{'covered':>9}{'certain':>9}{'ambiguous':>11}{'abstain':>9}{'wrong':>8}")
    for c, name in enumerate(class_names):
        k = y == c
        bar = f"P >= {1 - q[c]:.2%}" if np.isfinite(q[c]) else "always"
        print(f"  {name:<12}{bar:>14}{prec(single & sets[:, c])}" + (fmt(k) if k.any() else "   no eval frames"))
    print(f"  {'all':<12}{'':>14}{prec(single)}" + fmt(slice(None)))
    return q
