#!/usr/bin/env python3
"""Aggregate runManuscriptTotal.sh (whole-trajectory classification) -> mean +/- std over seeds -> LaTeX.

Run from gmat/data/classification, after the sweep's parse step:

    python displayLogData.py . --group-dir leo/ --group-name manuscript   # (per train orbit)
    python aggregateManuscriptTotal.py --out-dir manuscript_tables

The whole-trajectory counterpart of seqClassification/aggregateManuscript.py, built from its pieces
-- stem parsing, Seed<n>/OnePass filtering, mean/std and paired-delta cells, model order, table
layout -- so the two tasks' tables line up row for row. generateLatexTableCompact.py is NOT usable
on this sweep: it keeps the single best row per (model, window), which here would be a max over
seeds, test orbits and the phys/ladder arms (it only knows OE vs Cartesian).

One approach here (one label per trajectory), so the in-sequence script's cascade-only outputs
(t5_cascade, cascade_vs_joint, best_cascade) have no counterpart, and t7_complexity is skipped
because displayLogData.py does not emit Big-O. Every file and \\label is prefixed total_ so these
can sit next to the in-sequence tables in one manuscript.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seqClassification"))
from aggregateManuscript import (_disp, _pivot, agg, emit, feat_label,  # noqa: E402
                                 load, paired_delta, rpf1_table)

# The training script's "Entering <X> Training Loop" names -> the in-sequence tables' names.
MODEL_NAME = {"Decision Trees (LightGBM)": "LightGBM", "Mamba": "MAMBA",
              "Transformer": "TRANSFORMER", "1D-CNN (InceptionTime)": "CNN"}
STAGE = "whole_trajectory"   # constant eval_stage, so the in-sequence helpers' KEYS still apply
APPROACH = {STAGE: "Whole-trajectory"}


def load_total(pattern: str) -> pd.DataFrame:
    df = load(pattern, rename_models=MODEL_NAME)
    df["eval_stage"] = STAGE
    # displayLogData.py names the sklearn report's rows <label>_<metric>: "macro avg" -> macro_avg_*
    df = df.rename(columns={f"macro_avg_{m}": f"macro_{m}" for m in ("precision", "recall", "f1")})
    thrust = [f"{c}_recall" for c in ("chemical", "electric", "impulsive")]
    df["min_thrust_class_recall"] = df[thrust].min(axis=1, skipna=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("manuscript_tables"))
    ap.add_argument("--headline-prop", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3, help="expected seeds per cell; warns if short")
    ap.add_argument("--decimals", type=int, default=2, help="decimals in the per-orbit R/P/F1 tables")
    ap.add_argument("--no-color", action="store_true", help="drop \\cellcolor from those tables")
    ap.add_argument("--color-lo", type=float, default=0.25)
    ap.add_argument("--color-hi", type=float, default=0.75)
    a = ap.parse_args()

    ev = load_total("parsed_data/*/_group/csv/summary_manuscript.csv")
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    ev.assign(feat_label=ev["feat"].map(feat_label)).to_csv(out / "total_eval_long.csv", index=False)
    print(f"wrote {out / 'total_eval_long.csv'}  ({len(ev)} rows, "
          f"{ev['seed'].nunique()} seeds, {ev['model'].nunique()} models)")

    ind = ev[ev.train == ev.test]
    hp, n = a.headline_prop, a.seeds
    ms = f"mean $\\pm$ std over {n} seeds"

    # T1 -- headline, in-distribution, phys features.
    t = agg(ind[(ind.propMin == hp) & (ind.feat == "phys")], "macro_f1", n_expected=n)
    emit(_pivot(t, "model", "train"), out / "total_t1_main.tex",
         f"In-distribution whole-trajectory macro-F1 ({hp} min windows, {feat_label('phys')} "
         f"features), {ms}.", "tab:total_main")

    # T2 -- feature ablation, plus one paired delta per ablation step.
    for metric, tag in (("macro_f1", "f1"), ("min_thrust_class_recall", "minrec")):
        t = agg(ind[ind.propMin == hp], metric, n_expected=n)
        emit(_pivot(_disp(t), "model", ["train", "feat"]), out / f"total_t2_features_{tag}.tex",
             f"Feature-set ablation, in-distribution whole-trajectory {metric.replace('_', ' ')} "
             f"({hp} min), {ms}.", f"tab:total_feat_{tag}")
        for hi, lo in (("phys", "eci"), ("ladder", "phys")):
            d = paired_delta(ind[ind.propMin == hp], metric, hi, lo)
            emit(_pivot(d, "model", "train"), out / f"total_t2_features_{tag}_delta_{hi}_vs_{lo}.tex",
                 f"Paired {feat_label(hi)} $-$ {feat_label(lo)} delta in whole-trajectory "
                 f"{metric.replace('_', ' ')} ({hp} min), differenced within each seed then averaged.",
                 f"tab:total_featdelta_{tag}_{hi}")

    # T3 -- window length.
    t = agg(ind[ind.feat == "phys"], "macro_f1", n_expected=n)
    emit(_pivot(t, "model", ["train", "propMin"]), out / "total_t3_window.tex",
         f"Effect of propagation window on in-distribution whole-trajectory macro-F1 "
         f"({feat_label('phys')}), {ms}.", "tab:total_window")

    # T4 -- cross-regime transfer, in-distribution columns as the reference.
    xr = ev[(ev.propMin == hp) & (ev.feat == "phys") & ev.train.isin(["leo", "geo"])].copy()
    xr["arm"] = xr.train + r"$\rightarrow$" + xr.test
    for metric, tag in (("macro_f1", "f1"), ("min_thrust_class_recall", "minrec")):
        t = agg(xr, metric, keys=["arm", "model", "eval_stage"], n_expected=n)
        emit(_pivot(t, "model", "arm"), out / f"total_t4_transfer_{tag}.tex",
             f"Cross-regime transfer, whole-trajectory {metric.replace('_', ' ')} ({hp} min, "
             f"{feat_label('phys')}).", f"tab:total_transfer_{tag}")

    # T6 -- per-class, the two hardest classes.
    cols = [c for c in ("electric_recall", "electric_f1", "impulsive_recall", "impulsive_f1")
            if c in ind.columns]
    parts = [agg(ind[ind.propMin == hp], c, keys=["feat", "train", "model", "eval_stage"],
                 n_expected=n).assign(col=c) for c in cols]
    emit(_pivot(_disp(pd.concat(parts, ignore_index=True)), ["model", "train"], ["col", "feat"]),
         out / "total_t6_perclass.tex",
         f"Per-class whole-trajectory recall and F1 for the two hardest classes, in-distribution "
         f"({hp} min).", "tab:total_perclass")

    # Per train->test arm x feature set: R/P/F1 at every window, same layout as seq_*_jc.tex.
    print()
    pairs = sorted(set(zip(ev.train.astype(str), ev.test.astype(str))), key=lambda p: (p[0] != p[1], p))
    for train, test in pairs:
        for feat in sorted(ev.feat.unique()):
            slug = f"total_{train}" if train == test else f"total_{train}_to_{test}"
            tex = rpf1_table(ev, train, test, feat, n, decimals=a.decimals, use_color=not a.no_color,
                             color_lo=a.color_lo, color_hi=a.color_hi, approaches=APPROACH,
                             what="Whole-trajectory macro Recall (R), Precision (P), and F1",
                             label=f"{slug}_{feat}")
            if tex is not None:
                (out / f"{slug}_{feat}.tex").write_text(tex)
                print(f"wrote {out / f'{slug}_{feat}.tex'}")


if __name__ == "__main__":
    main()
