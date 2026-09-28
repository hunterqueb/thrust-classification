# Confidence intervals for classification metrics: a hierarchical percentile bootstrap over seeds
# and eval trajectories. Shared by the whole-trajectory aggregator (CIs straight from the logged
# confusion matrices) and uncertaintyReport.py (CIs from the scores --conformal saves). numpy only,
# so the aggregators can import it without torch.
#
# A replicate resamples the seeds (training runs) with replacement, then -- independently for each
# draw -- that seed's eval TRAJECTORIES with replacement, stratified by a per-trajectory stratum (its
# thrust type) so every class keeps its support. The replicate's value is the mean over its drawn
# seeds, the same seed-mean the tables report as the point estimate, so the interval covers "a new
# training run and a new eval set of the same size". With 3 seeds the seed level is coarse (10
# distinct multisets); most of the width comes from the trajectory level.
#
# Trajectories, not frames: frames within one trajectory are correlated, and resampling them
# independently would understate the width. Everything is additive -- each trajectory is a count
# vector (its flattened confusion matrix plus whatever else a statistic needs), a replicate is a
# weighted sum of those vectors, and a statistic maps summed counts to metrics -- so B replicates of
# one seed are a single matrix product.
import warnings

import numpy as np

CLASS_KEYS = ["no_thrust", "chemical", "electric", "impulsive"]   # displayLogData's column prefixes


def _div(a, b):
    """a / b with 0 where b == 0: sklearn's zero_division=0, so point estimates match its reports."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return np.divide(a, b, out=np.zeros(a.shape), where=b > 0)


def expandConfusion(cm):
    """[C,C] counts (rows true) -> (units [n, C*C], strata [n]): one one-hot unit per sample, so a
    whole-trajectory confusion matrix bootstraps exactly like per-trajectory counts would."""
    cm = np.asarray(cm, dtype=np.int64)
    C = cm.shape[0]
    cells = np.repeat(np.arange(C * C), cm.reshape(-1))
    units = np.zeros((len(cells), C * C))
    units[np.arange(len(cells)), cells] = 1.0
    return units, cells // C


def confusionMetrics(cm, class_keys=CLASS_KEYS, prefix=""):
    """cm [..., C, C] (rows true) -> {name: [...]}: accuracy, per-class <key>_precision/_recall/_f1,
    their macro averages (mean of per-class values, as classification_report's 'macro avg'), and
    min_thrust_class_recall over classes 1..C-1."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    true, pred = cm.sum(-1), cm.sum(-2)
    per = {"precision": _div(tp, pred), "recall": _div(tp, true), "f1": _div(2 * tp, true + pred)}
    out = {f"{prefix}accuracy": _div(tp.sum(-1), cm.sum((-2, -1)))}
    for name, v in per.items():
        out[f"{prefix}macro_{name}"] = v.mean(-1)
        for i, k in enumerate(class_keys):
            out[f"{prefix}{k}_{name}"] = v[..., i]
    out[f"{prefix}min_thrust_class_recall"] = per["recall"][..., 1:].min(-1)
    return out


def confusionStat(counts, class_keys=CLASS_KEYS):
    """hierarchicalCI statistic for units that are flattened confusion matrices."""
    C = len(class_keys)
    return confusionMetrics(counts[:, :C * C].reshape(-1, C, C), class_keys)


# --- Per-trajectory units for uncertaintyReport.py ------------------------------------------------
# Layout of one unit (C classes):
#   [0, C*C)          point-prediction confusion matrix (the predictions the log reports)
#   [C*C, 2C*C)       confusion matrix of samples whose conformal set is a single class (that class
#                     is the prediction) -- selective classification
#   [2C*C, +C)        conformal samples per true class
#   [2C*C+C, +C)      ... whose set contains the true class (covered)
#   [2C*C+2C, +C)     ... whose set is empty (abstain)
#   [2C*C+3C]         summed set sizes
def unitWidth(C):
    return 2 * C * C + 3 * C + 1


def trajectoryUnits(y_point, pred, y_conf, sets, pad_idx=-100):
    """Per-trajectory count vectors [N, unitWidth(C)].
    y_point/pred [N,T]: labels and the reported point predictions; pad_idx in y_point = frame not
    scored. y_conf [N,T]: labels for the conformal part (pad_idx = no conformal prediction).
    sets [N,T,C] bool: conformal prediction sets. A whole-trajectory classifier is T = 1."""
    C = sets.shape[-1]
    U = np.zeros((len(y_point), unitWidth(C)))
    m = y_point != pad_idx
    np.add.at(U, (np.nonzero(m)[0], y_point[m] * C + pred[m]), 1.0)

    m = y_conf != pad_idx
    ii, y, s = np.nonzero(m)[0], y_conf[m], sets[m]
    size = s.sum(1)
    hit = s[np.arange(len(y)), y]
    one, empty = size == 1, size == 0
    base = 2 * C * C
    np.add.at(U, (ii[one], C * C + y[one] * C + s[one].argmax(1)), 1.0)
    np.add.at(U, (ii, base + y), 1.0)
    np.add.at(U, (ii[hit], base + C + y[hit]), 1.0)
    np.add.at(U, (ii[empty], base + 2 * C + y[empty]), 1.0)
    np.add.at(U, (ii, np.full(len(ii), base + 3 * C)), size.astype(float))
    return U


def trajectoryStrata(y, pad_idx=-100):
    """A trajectory's stratum: its highest label over scored frames, i.e. its thrust type (0 when it
    never thrusts). Whole-trajectory: the label itself."""
    return np.where(y == pad_idx, -1, y).max(axis=1)


def unitStat(counts, class_keys=CLASS_KEYS):
    """hierarchicalCI statistic for trajectoryUnits: the point metrics, the same metrics restricted to
    single-class conformal sets (selective_*), per-class conformal outcome rates (coverage, certain,
    ambiguous, abstain, wrong; the reportConformal columns) plus their pooled 'all' versions,
    acceptance_rate (fraction given a single-class set) and mean_set_size."""
    C = len(class_keys)
    cc, base = C * C, 2 * C * C
    single = counts[:, cc:2 * cc].reshape(-1, C, C)
    n, cov, abst = (counts[:, base + i * C:base + (i + 1) * C] for i in range(3))
    certain = np.diagonal(single, axis1=-2, axis2=-1)
    out = confusionMetrics(counts[:, :cc].reshape(-1, C, C), class_keys)
    out.update(confusionMetrics(single, class_keys, prefix="selective_"))
    rates = {"coverage": cov, "certain": certain, "ambiguous": cov - certain, "abstain": abst,
             "wrong": n - cov - abst}
    for name, v in rates.items():
        for i, k in enumerate(class_keys):
            out[f"{name}_{k}"] = _div(v[:, i], n[:, i])
        out[f"{name}_all"] = _div(v.sum(1), n.sum(1))
    out["acceptance_rate"] = _div(single.sum((1, 2)), n.sum(1))
    out["mean_set_size"] = _div(counts[:, base + 3 * C], n.sum(1))
    return out


# --- The bootstrap --------------------------------------------------------------------------------
def _weights(strata, m, rng):
    """[m, n] resampling counts: n draws with replacement, stratified so each stratum keeps its size."""
    W = np.zeros((m, len(strata)))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=m)
    return W


def hierarchicalCI(units, strata, stat, n_boot=2000, level=0.95, seed=0):
    """units/strata: one [n_s, K] count array and one [n_s] stratum array per seed (training run).
    stat: [R, K] summed counts -> {name: [R]}.
    Returns {name: (point, lo, hi)}: point is the mean over seeds of stat on each seed's full eval
    set; (lo, hi) the central `level` percentile interval of the replicate seed-means. NaN values are
    skipped in both means."""
    rng = np.random.default_rng(seed)
    S = len(units)
    per_seed = [stat(u.sum(0, keepdims=True)) for u in units]
    names = list(per_seed[0])
    draw = rng.integers(S, size=(n_boot, S))            # which seed fills each (replicate, slot)
    vals = {k: np.full((n_boot, S), np.nan) for k in names}
    for s in range(S):
        where = np.nonzero(draw == s)
        if len(where[0]):
            r = stat(_weights(np.asarray(strata[s]), len(where[0]), rng) @ units[s])
            for k in names:
                vals[k][where] = r[k]
    q = 100 * (1 - level) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices, e.g. a class never present
        return {k: (float(np.nanmean([p[k][0] for p in per_seed])),
                    *(float(v) for v in np.nanpercentile(np.nanmean(vals[k], axis=1), [q, 100 - q])))
                for k in names}


def _selfcheck():
    from sklearn.metrics import precision_recall_fscore_support
    rng = np.random.default_rng(1)
    cm = rng.integers(0, 60, size=(4, 4))
    cm[3] = 0                                   # a class with no support ...
    cm[:, 2] = 0                                # ... and one never predicted: zero_division paths
    units, strata = expandConfusion(cm)
    assert units.shape == (cm.sum(), 16) and np.array_equal(np.bincount(strata, minlength=4), cm.sum(1))
    y, p = strata, units.argmax(1) % 4
    P, R, F, _ = precision_recall_fscore_support(y, p, labels=range(4), zero_division=0)
    got = confusionMetrics(cm)
    assert np.allclose(got["macro_precision"], P.mean()) and np.allclose(got["macro_recall"], R.mean())
    assert np.allclose(got["macro_f1"], F.mean()) and np.allclose(got["electric_f1"], F[2])
    assert np.allclose(got["accuracy"], (y == p).mean())

    # Perfect classifier: zero-width interval at 1.
    perfect = [expandConfusion(np.diag([50, 40, 30, 20])) for _ in range(3)]
    ci = hierarchicalCI([u for u, _ in perfect], [s for _, s in perfect], confusionStat, n_boot=200)
    assert ci["macro_f1"] == (1.0, 1.0, 1.0), ci["macro_f1"]

    # Correlated frames: each trajectory is right or wrong as a whole. Resampling trajectories must
    # give a wider interval than (wrongly) resampling its frames as if independent.
    N, T = 200, 30
    y = np.repeat(np.arange(4), N // 4)[:, None].repeat(T, 1)
    ok = rng.random(N) < 0.8
    pred = np.where(ok[:, None], y, (y + 1) % 4)
    sets = np.eye(4, dtype=bool)[pred]
    traj = trajectoryUnits(y, pred, y, sets)
    frame = trajectoryUnits(y.reshape(-1, 1), pred.reshape(-1, 1), y.reshape(-1, 1), sets.reshape(-1, 1, 4))
    width = lambda u, st: (lambda _, lo, hi: hi - lo)(*hierarchicalCI([u], [st], unitStat, n_boot=500)["accuracy"])
    w_traj, w_frame = width(traj, y[:, 0]), width(frame, y.reshape(-1))
    assert w_traj > 3 * w_frame, (w_traj, w_frame)

    # unitStat's conformal rates against a direct count, including empty and multi-class sets.
    sets = rng.random((N, T, 4)) < 0.35
    s = unitStat(trajectoryUnits(y, pred, y, sets).sum(0, keepdims=True))
    hit = np.take_along_axis(sets, y[..., None], -1)[..., 0]
    size = sets.sum(-1)
    assert np.isclose(s["coverage_all"][0], hit.mean()) and np.isclose(s["abstain_all"][0], (size == 0).mean())
    assert np.isclose(s["certain_all"][0], (hit & (size == 1)).mean())
    assert np.isclose(s["wrong_all"][0], (~hit & (size > 0)).mean())
    assert np.isclose(s["acceptance_rate"][0], (size == 1).mean()) and np.isclose(s["mean_set_size"][0], size.mean())
    assert np.isclose(s["accuracy"][0], (y == pred).mean())
    # pad_idx frames are excluded from both parts.
    y_pad = y.copy()
    y_pad[:, :4] = -100
    s = unitStat(trajectoryUnits(y_pad, pred, y_pad, sets).sum(0, keepdims=True))
    assert np.isclose(s["coverage_all"][0], hit[:, 4:].mean()) and np.isclose(s["accuracy"][0], (y == pred)[:, 4:].mean())
    assert np.array_equal(trajectoryStrata(y_pad), y[:, 0])
    print("bootstrapCI selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
