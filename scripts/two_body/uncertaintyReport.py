#!/usr/bin/env python3
"""Per-prediction uncertainty and bootstrap confidence intervals from the scores --conformal saves.

Both training scripts, run with --conformal ALPHA --save, write each model's raw calibration/eval
scores to <logLoc>/scores/<log stem>/<model>[_<approach>].npz (conformal.saveScores). This script
recomputes everything from those files, so changing alpha, the CI level or a metric never needs a
retrain. Run from the repo root:

    python scripts/two_body/uncertaintyReport.py --task seq      # in-sequence (per-timestep)
    python scripts/two_body/uncertaintyReport.py --task total    # whole-trajectory

Per file: temperature scaling and Mondrian conformal sets exactly as the log's conformal block
(conformal.composedProbs / mondrianThresholds), plus the point predictions the log reports --
argmax for joint models; stage 1 then stage 2 for cascades, whose unpredicted leading frames (the
classic models' first hankel_L-1) count as No Thrust, as combineCascadePredictions scores them.

Writes to <task root>/manuscript_tables/ (the aggregators' default --out-dir):
    [total_]uncertainty_ci_long.csv    per (cell, model, approach, metric): seed-mean point + CI
    [total_]uncertainty_per_seed.csv   the per-seed values the points average; the aggregators
                                       cross-check its macro_f1 against the parsed logs
Metrics: bootstrapCI.unitStat -- point accuracy/P/R/F1, the same restricted to single-class
conformal sets (selective_*), acceptance_rate, per-class conformal outcome rates, mean_set_size.
CIs: bootstrapCI.hierarchicalCI (seeds, then trajectories within each seed, stratified by thrust
type), conditional on each run's calibration split -- the situation of a deployed model.

--dump-predictions also writes per-sample uncertainty (calibrated probabilities, p-values, set,
outcome, credibility, confidence) to scores/<log stem>/predictions/<file>.csv.gz.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "gmat" / "data" / "seqClassification"))
from bootstrapCI import CLASS_KEYS, hierarchicalCI, trajectoryStrata, trajectoryUnits, unitStat  # noqa: E402
from conformal import composedProbs, mondrianThresholds, perPrediction  # noqa: E402

TASKS = {"seq": (REPO / "gmat" / "data" / "seqClassification", ""),
         "total": (REPO / "gmat" / "data" / "classification", "total_")}
CELL = ["train", "test", "feat", "propMin", "model", "eval_stage"]


def analyse(path: Path, dump: bool = False) -> dict:
    """One saved report -> {model, approach, alpha, units [N,K], strata [N]}."""
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    pad, vf, alpha = int(d["pad_idx"]), int(d["valid_from"]), float(d["alpha"])
    y_cal, y_eval = d["y_cal"], d["y_eval"]
    scores = {m: (d[f"{m}_cal"].astype(np.float64), d[f"{m}_eval"].astype(np.float64))
              for m in ("joint", "stage1", "stage2") if f"{m}_cal" in d}
    mask = lambda y: np.where(np.arange(y.shape[1]) < vf, pad, y)
    yc, ye = mask(y_cal), mask(y_eval)

    pc, pe = composedProbs(scores, yc, ye, pad, verbose=False)
    q = mondrianThresholds(pc, yc, alpha, pad)
    u = perPrediction(pc, yc, pe, q)

    if "joint" in scores:            # scored on predicted frames only
        pred, y_point = scores["joint"][1].argmax(-1), ye
    else:                            # the cascade scores every frame
        pred = np.where(scores["stage1"][1].argmax(-1) == 0, 0, scores["stage2"][1].argmax(-1) + 1)
        pred[:, :vf] = 0
        y_point = y_eval
    if dump:
        _dumpPredictions(path, ye, pred, u, pad)
    return {"model": str(d["model"]), "approach": str(d["approach"]), "alpha": alpha,
            "units": trajectoryUnits(y_point, pred, ye, u["sets"], pad),
            "strata": trajectoryStrata(y_eval, pad)}


def _dumpPredictions(path: Path, y, pred, u, pad):
    n, t = np.nonzero(y != pad)
    yt, sets = y[n, t], u["sets"][n, t]
    size, hit = sets.sum(1), sets[np.arange(len(yt)), yt]
    df = pd.DataFrame({"traj": n, "t": t, "y_true": yt, "y_pred": pred[n, t]})
    for i, k in enumerate(CLASS_KEYS):
        df[f"p_{k}"] = u["probs"][n, t, i]
    for i, k in enumerate(CLASS_KEYS):
        df[f"pval_{k}"] = u["pvalues"][n, t, i]
    df["set"] = ["|".join(k for k, s in zip(CLASS_KEYS, row) if s) for row in sets]
    df["set_size"] = size
    df["outcome"] = np.select([hit & (size == 1), hit, size == 0], ["certain", "ambiguous", "abstain"], "wrong")
    df["credibility"], df["confidence"] = u["credibility"][n, t], u["confidence"][n, t]
    out = path.parent / "predictions" / (path.stem + ".csv.gz")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6g")


def collect(root: Path, dump: bool) -> list[dict]:
    from aggregateManuscript import E2E, JOINT, SEED_RE, _config
    stage = {"Joint": JOINT, "Cascade": E2E, "": "whole_trajectory"}
    runs = []
    for run_dir in sorted(root.glob("*/*min-*/scores/*")):
        stem = run_dir.name
        if not run_dir.is_dir() or not SEED_RE.search(stem) or "OnePass_" in stem:
            continue
        cfg = _config(stem, run_dir.relative_to(root).as_posix())
        files = sorted(run_dir.glob("*.npz"))
        print(f"  {run_dir.relative_to(root).as_posix()}: {len(files)} model(s)")
        for f in files:
            r = analyse(f, dump)
            runs.append({**cfg, "model": r.pop("model"), "eval_stage": stage[r.pop("approach")], **r})
    return runs


def summarise(runs: list[dict], n_boot: int, level: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(runs)
    ci, per = [], []
    for key, g in df.groupby(CELL, sort=True):
        g = g.sort_values("seed")
        cell = dict(zip(CELL, key))
        if g.alpha.nunique() > 1:
            print(f"  ! seeds of {cell} were run at different alphas {sorted(g.alpha.unique())}")
        for met, (pt, lo, hi) in hierarchicalCI(list(g.units), list(g.strata), unitStat,
                                                n_boot=n_boot, level=level).items():
            ci.append({**cell, "metric": met, "point": pt, "lo": lo, "hi": hi, "level": level,
                       "n_boot": n_boot, "n_seeds": len(g), "alpha": g.alpha.iloc[0]})
        for seed, alpha, units in zip(g.seed, g.alpha, g.units):
            per += [{**cell, "seed": seed, "alpha": alpha, "metric": met, "value": float(v[0])}
                    for met, v in unitStat(units.sum(0, keepdims=True)).items()]
    return pd.DataFrame(ci), pd.DataFrame(per)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(TASKS), help="seq: in-sequence, total: whole-trajectory")
    ap.add_argument("--root", type=Path, help="override the task's log root")
    ap.add_argument("--out-dir", type=Path, help="default: <root>/manuscript_tables")
    ap.add_argument("--ci-level", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--dump-predictions", action="store_true", help="per-sample CSVs (large for 100 min)")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return _selfcheck()
    if not a.task:
        ap.error("--task is required")
    root, prefix = TASKS[a.task]
    root = a.root or root
    out = a.out_dir or root / "manuscript_tables"

    runs = collect(root, a.dump_predictions)
    if not runs:   # not an error: the sweep scripts call this unconditionally under CONFORMAL
        print(f"  (no scores under {root}/*/*min-*/scores/ -- run the sweep with CONFORMAL=<alpha>; nothing written)")
        return
    ci, per = summarise(runs, a.n_boot, a.ci_level)
    out.mkdir(parents=True, exist_ok=True)
    ci.to_csv(out / f"{prefix}uncertainty_ci_long.csv", index=False)
    per.to_csv(out / f"{prefix}uncertainty_per_seed.csv", index=False)
    print(f"wrote {out / f'{prefix}uncertainty_ci_long.csv'}  ({ci.groupby(CELL).ngroups} cells, "
          f"{len(runs)} model runs)")
    print(f"wrote {out / f'{prefix}uncertainty_per_seed.csv'}")


def _selfcheck() -> None:
    import tempfile

    from conformal import predictionSets, saveScores
    rng = np.random.default_rng(0)
    C, alpha = 4, 0.1

    def logits(y, sharp=2.0):
        return rng.normal(size=y.shape + (C,)) + sharp * np.eye(C)[y]

    # T = 1: set membership and p-value > alpha agree up to about one calibration rank.
    y_cal, y_eval = rng.integers(0, C, (2000, 1)), rng.integers(0, C, (4000, 1))
    pc, pe = composedProbs({"joint": (logits(y_cal), logits(y_eval))}, y_cal, y_eval, verbose=False)
    q = mondrianThresholds(pc, y_cal, alpha)
    u = perPrediction(pc, y_cal, pe, q)
    disagree = (u["sets"] != (u["pvalues"] > alpha)).mean()
    assert disagree < 0.005, disagree
    assert np.array_equal(u["sets"], predictionSets(pe, q))
    assert np.all((u["credibility"] >= 0) & (u["credibility"] <= 1) & (u["confidence"] <= 1))

    # Round trip through saveScores: a classic-style cascade with valid_from, and its joint twin.
    N, T, vf = 60, 12, 4
    y_cal, y_eval = (np.where(np.arange(T) >= rng.integers(0, T, (n, 1)), rng.integers(1, C, (n, 1)), 0)
                     for n in (N, N))
    s2 = lambda y: logits(np.maximum(y - 1, 0), 1.0)[..., :3]
    s1 = lambda y: logits((y > 0).astype(int), 1.0)[..., :2]
    cas = {"stage1": (s1(y_cal), s1(y_eval)), "stage2": (s2(y_cal), s2(y_eval))}
    joint = {"joint": (logits(y_cal, 1.0), logits(y_eval, 1.0))}
    with tempfile.TemporaryDirectory() as td:
        saveScores(td, "LightGBM", "Cascade", cas, y_cal, y_eval, alpha, valid_from=vf)
        saveScores(td, "LightGBM", "Joint", joint, y_cal, y_eval, alpha, valid_from=vf)
        rc = analyse(Path(td) / "LightGBM_Cascade.npz", dump=True)
        rj = analyse(Path(td) / "LightGBM_Joint.npz")
        dumped = pd.read_csv(Path(td) / "predictions" / "LightGBM_Cascade.csv.gz")
    sc, sj = (unitStat(r["units"].sum(0, keepdims=True)) for r in (rc, rj))
    # Cascade: every frame scored, the first vf predicted No Thrust.
    pred = np.where(cas["stage1"][1].argmax(-1) == 0, 0, cas["stage2"][1].argmax(-1) + 1)
    pred[:, :vf] = 0
    assert np.isclose(sc["accuracy"][0], (pred == y_eval).mean())
    # Joint: only predicted frames scored.
    assert np.isclose(sj["accuracy"][0], (joint["joint"][1].argmax(-1) == y_eval)[:, vf:].mean())
    assert rc["model"] == "LightGBM" and rc["approach"] == "Cascade" and rc["alpha"] == alpha
    assert np.array_equal(rc["strata"], y_eval.max(1))
    # Conformal part over predicted frames only, and the dump holds exactly those.
    assert len(dumped) == N * (T - vf) and dumped.t.min() == vf
    assert np.isclose(sc["coverage_all"][0], dumped.outcome.isin(["certain", "ambiguous"]).mean())
    assert np.isclose(sc["acceptance_rate"][0], (dumped.set_size == 1).mean())

    # Whole pipeline: a fake 3-seed sweep cell (plus a OnePass smoke run, which must be ignored)
    # -> collect -> summarise -> the aggregator's table.
    from aggregateManuscript import CI_SELECTIVE, E2E, JOINT, ci_table
    with tempfile.TemporaryDirectory() as td:
        cell = Path(td) / "leo" / "30min-1500" / "scores"
        for stem in [f"30min1500Energy_J2Energy_OE_EvalTest_Seed{s}" for s in range(3)] + \
                    ["30min1500Energy_J2Energy_OE_OnePass_EvalTest_Seed0"]:
            saveScores(cell / stem, "LSTM", "Joint", joint, y_cal, y_eval, alpha)
            saveScores(cell / stem, "LSTM", "Cascade", cas, y_cal, y_eval, alpha)
        ci, per = summarise(collect(Path(td), False), n_boot=200, level=0.9)
    assert set(ci.eval_stage) == {JOINT, E2E} and set(ci.n_seeds) == {3} and set(per.seed) == {0, 1, 2}
    assert set(ci.feat) == {"phys"} and set(ci.propMin) == {30} and set(ci.alpha) == {alpha}
    tex = ci_table(ci, "leo", "leo", "phys", [(JOINT, 30, "Joint"), (E2E, 30, "Cascade")], "W", "tab:x", 3,
                   metrics=CI_SELECTIVE)
    assert tex and "\\textbf{LSTM}" in tex and "--" not in tex.split("\\midrule")[1].split("\\bottomrule")[0]
    print("uncertaintyReport selfcheck ok")


if __name__ == "__main__":
    main()
