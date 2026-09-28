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
#
# saveScores writes the raw inputs of each report to disk, so scripts/two_body/uncertaintyReport.py can
# recompute everything here -- plus per-prediction p-values and bootstrap confidence intervals -- with
# no retraining.
import os
import re

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


def calibratedProbs(z_cal, y_cal, z_eval, y_eval, label, pad_idx=-100, verbose=True):
    """Fit T on the calibration split's scores [N,T,C] against labels [N,T]; returns temperature-
    scaled probabilities for both splits. Scores need not be logits: softmax(z / T) with a fitted T
    is a valid calibration map for any per-class score (log-probs, ridge outputs)."""
    zc, ze = torch.as_tensor(z_cal, dtype=torch.float64), torch.as_tensor(z_eval, dtype=torch.float64)
    yc, ye = (torch.as_tensor(np.asarray(y, dtype=np.int64)) for y in (y_cal, y_eval))
    T = fitTemperature(zc, yc, pad_idx)
    if verbose:
        m = ye != pad_idx
        ece_raw = ece(torch.softmax(ze[m], -1).numpy(), ye[m].numpy())
        ece_t = ece(torch.softmax(ze[m] / T, -1).numpy(), ye[m].numpy())
        print(f"  {label}: T = {T:.3f}, eval ECE {ece_raw:.2%} -> {ece_t:.2%}")
    return torch.softmax(zc / T, -1).numpy(), torch.softmax(ze / T, -1).numpy()


def stageLabels(y_joint, mode, pad_idx=-100):
    """Joint 4-class labels -> the label view a model of `mode` was trained on: joint as is; stage1
    thrust yes/no; stage2 thrust type 0..2 with non-thrust frames pad_idx. pad_idx frames stay
    pad_idx in every view. (seqData._deriveStage1Stage2, without its torch/qutils imports.)"""
    y = np.asarray(y_joint)
    if mode == "joint":
        return y
    out = (y > 0).astype(np.int64) if mode == "stage1" else np.where(y > 0, y - 1, pad_idx)
    return np.where(y == pad_idx, pad_idx, out)


def composedProbs(scores, y_cal, y_eval, pad_idx=-100, verbose=True):
    """scores: {'joint': (z_cal, z_eval)} or {'stage1': (...), 'stage2': (...)}, each [N,T,C];
    y_cal/y_eval: [N,T] joint labels. Each mode is temperature-scaled against its own label view,
    and a cascade's calibrated stages compose into joint probabilities P(NoThrust) = P1(no),
    P(type) = P1(yes) * P2(type | thrust), so every form gets sets over the same 4 classes.
    Returns (cal_probs, eval_probs), [N,T,4]."""
    probs = {mode: calibratedProbs(zc, stageLabels(y_cal, mode, pad_idx), ze, stageLabels(y_eval, mode, pad_idx),
                                   mode, pad_idx, verbose)
             for mode, (zc, ze) in scores.items()}
    if "joint" in probs:
        return probs["joint"]
    compose = lambda p1, p2: np.concatenate([p1[..., :1], p1[..., 1:] * p2], axis=-1)
    return tuple(compose(p1, p2) for p1, p2 in zip(probs["stage1"], probs["stage2"]))


def saveScores(out_dir, model, approach, scores, y_cal, y_eval, alpha, pad_idx=-100, valid_from=0):
    """One report's raw inputs -> <out_dir>/<model>[_<approach>].npz, for uncertaintyReport.py.
    scores as composedProbs takes them (T = 1 for a whole-trajectory classifier); y_cal/y_eval the
    UNMASKED [N,T] joint labels; frames t < valid_from have no prediction (the classic models' first
    hankel_L-1 frames). Raw scores rather than probabilities, so the temperature can be refitted."""
    os.makedirs(out_dir, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9]+", "_", f"{model} {approach}").strip("_")
    arrays = {f"{mode}_{split}": np.asarray(z, dtype=np.float32)
              for mode, zs in scores.items() for split, z in zip(("cal", "eval"), zs)}
    np.savez_compressed(os.path.join(out_dir, name + ".npz"), model=model, approach=approach,
                        y_cal=np.asarray(y_cal, dtype=np.int64), y_eval=np.asarray(y_eval, dtype=np.int64),
                        alpha=alpha, pad_idx=pad_idx, valid_from=valid_from, **arrays)


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


def conformalPValues(cal_probs, cal_y, eval_probs):
    """Per-class Mondrian p-values [..., C]: (1 + #{calibration frames of class c whose score 1 - p_c
    is >= this sample's}) / (n_c + 1) -- how typical this sample would be as a member of class c.
    A class absent from calibration gets 1 (mondrianThresholds keeps it in every set).
    Frames are counted individually, while mondrianThresholds' finite-sample correction counts
    trajectories, so at the set boundary a p-value can disagree with the reported set by about one
    calibration rank; the sets (thresholds) remain the definition. At T = 1 both count trajectories."""
    out = np.ones(eval_probs.shape)
    for c in range(eval_probs.shape[-1]):
        s = np.sort(1.0 - cal_probs[cal_y == c][:, c])
        if len(s):
            n_ge = len(s) - np.searchsorted(s, 1.0 - eval_probs[..., c], side="left")
            out[..., c] = (1 + n_ge) / (len(s) + 1)
    return out


def perPrediction(cal_probs, cal_y, eval_probs, q):
    """Per-sample uncertainty for every eval frame (or trajectory at T = 1), as {name: array}:
    probs [...,C] calibrated probabilities; sets [...,C] the prediction set at thresholds q;
    pvalues [...,C]; credibility = the largest p-value (how typical the most plausible label is --
    low means unlike anything in calibration); confidence = 1 - the second largest (how firmly the
    runner-up is ruled out)."""
    pv = conformalPValues(cal_probs, cal_y, eval_probs)
    top2 = np.sort(pv, axis=-1)[..., -2:]
    return {"probs": eval_probs, "sets": predictionSets(eval_probs, q), "pvalues": pv,
            "credibility": top2[..., 1], "confidence": 1.0 - top2[..., 0]}


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
