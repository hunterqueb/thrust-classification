#!/usr/bin/env python3
"""Total-sequence (whole-trajectory) vs in-sequence (joint 4-class) R/P/F1, side by side.

One table per train->test arm x feature set, in the seq_*_jc.tex layout: whole-trajectory on the
left, in-sequence joint on the right. The cascade is left out. Run after both sweeps' parse steps:

    python aggregateManuscriptCombined.py --out-dir manuscript_tables
    python aggregateManuscriptCombined.py --ci          # adds a [lo, hi] row under each model

--ci intervals are a hierarchical percentile bootstrap (seeds, then eval trajectories within each
seed, stratified by thrust type; scripts/two_body/bootstrapCI.py) on both halves:
  Total Sequence  straight from each log's confusion matrix -- its cells count trajectories, so
                  nothing beyond the logs is needed.
  In Sequence     from manuscript_tables/uncertainty_ci_long.csv, written by
                  `scripts/two_body/uncertaintyReport.py --task seq` from the scores a
                  CONFORMAL=<alpha> ./runManuscriptInSeq.sh sweep saves. The logged confusion matrix
                  counts FRAMES, which are correlated within a trajectory, so bootstrapping it would
                  understate the width; until that CSV exists those cells print "--".
"""
import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLASS_DIR = HERE.parent / "classification"
sys.path.insert(0, str(CLASS_DIR))
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "two_body"))
from aggregateManuscript import (JOINT, RPF1, _config, complexity_legend, feat_label,  # noqa: E402
                                 load, manuscript_stem, model_order, rpf1_table)
from aggregateManuscriptTotal import MODEL_NAME, STAGE, load_total  # noqa: E402
from bootstrapCI import confusionStat, expandConfusion, hierarchicalCI  # noqa: E402
from displayLogData import RE_VAL_ACC, iter_models, parse_confusion_matrix  # noqa: E402
from displaySeqLogData import BIG_O_BY_MODEL, _model_key  # noqa: E402

APPROACHES = {STAGE: "Total Sequence", JOINT: "In Sequence"}
SEQ_CI_CSV = HERE / "manuscript_tables" / "uncertainty_ci_long.csv"
CELL = ["train", "test", "feat", "propMin", "model", "eval_stage"]

# Whole-trajectory classic models see the flattened trajectory once, not a Hankel window per
# timestep, so their cost drops the per-timestep factor. Deep backbones still run over all T steps
# (then pool), so they share the in-sequence Big-O.
TOTAL_BIG_O = {"LIGHTGBM": r"$\mathcal{O}(T d + E\,C\,D)$",
               "MINIROCKET": r"$\mathcal{O}(K\,(T w + C))$"}


def complexity_table(tot: pd.DataFrame, cx: pd.DataFrame, train: str, feat: str) -> pd.DataFrame:
    """Model x {Total Sequence, In Sequence (joint)}: Big-O, params, memory -- the old t7 layout.
    In-distribution only; params/memory are a plain mean over seeds and windows."""
    t = tot[(tot.train == train) & (tot.test == train) & (tot.feat == feat)]
    s = cx[(cx.train == train) & (cx.test == train) & (cx.feat == feat) & (cx.Approach == "Joint")]
    t = t.groupby("model", observed=True)[["params", "memory_mb"]].mean()
    s = s.groupby("Model", observed=True)[["Params", "Memory_MB"]].mean()

    def _int(v: float) -> str:
        return "--" if pd.isna(v) else f"{int(round(v)):,}".replace(",", "{,}")

    def _mb(v: float) -> str:
        return "--" if pd.isna(v) else f"{v:.3f}"

    rows = []
    for m in model_order(set(t.index) | set(s.index)):
        seq_o = BIG_O_BY_MODEL.get(_model_key(m), ("", "n/a"))[1]
        if m in t.index:
            rows.append((m, TOTAL_BIG_O.get(_model_key(m), seq_o), APPROACHES[STAGE],
                         _int(t.at[m, "params"]), _mb(t.at[m, "memory_mb"])))
        if m in s.index:
            rows.append((m, seq_o, APPROACHES[JOINT], _int(s.at[m, "Params"]), _mb(s.at[m, "Memory_MB"])))
    # Complexity as an index level, like the old t7: to_latex prints it once when both approaches share it.
    return pd.DataFrame(rows, columns=["Model", "Complexity", "Approach", "Params", "Memory (MB)"]
                        ).set_index(["Model", "Complexity", "Approach"])


def total_ci(trains, n_boot: int, level: float) -> dict:
    """(*CELL, metric) -> (lo, hi) for the whole-trajectory runs, from each log's confusion matrix.
    Same log set as load_total: every manuscript_stem log under the train orbit's directory."""
    runs = {}
    for train in trains:
        for log in sorted((CLASS_DIR / train).rglob("*.log")):
            if not manuscript_stem(log.stem):
                continue
            cfg = _config(log.stem, log.relative_to(CLASS_DIR).as_posix())
            for model, blk in iter_models(log.read_text(errors="ignore")):
                cm = parse_confusion_matrix(blk) if RE_VAL_ACC.search(blk) else None
                if cm is None or cm.shape != (4, 4):
                    continue
                key = (cfg["train"], cfg["test"], cfg["feat"], cfg["propMin"],
                       MODEL_NAME.get(model, model), STAGE)
                runs.setdefault(key, []).append(expandConfusion(cm))
    out = {}
    for key, us in runs.items():
        ci = hierarchicalCI([u for u, _ in us], [s for _, s in us], confusionStat, n_boot, level)
        out.update({key + (m,): ci[m][1:] for m, _ in RPF1})
    return out


def seq_ci(level: float) -> dict:
    """(*CELL, metric) -> (lo, hi) for the in-sequence joint runs, from uncertaintyReport.py."""
    if not SEQ_CI_CSV.exists():
        print(f"  ! {SEQ_CI_CSV} missing -- In Sequence CIs print '--'. Run the in-sequence sweep with "
              f"CONFORMAL=<alpha>, then scripts/two_body/uncertaintyReport.py --task seq")
        return {}
    d = pd.read_csv(SEQ_CI_CSV)
    d = d[(d.eval_stage == JOINT) & np.isclose(d.level, level)]
    if d.empty:
        print(f"  ! no joint rows at level {level} in {SEQ_CI_CSV} -- rerun uncertaintyReport.py --ci-level {level}")
    return {tuple(r[c] for c in CELL) + (r.metric,): (r.lo, r.hi) for r in d.itertuples()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=HERE / "manuscript_tables")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--no-color", action="store_true", help="drop \\cellcolor")
    ap.add_argument("--color-lo", type=float, default=0.25)
    ap.add_argument("--color-hi", type=float, default=0.75)
    ap.add_argument("--ci", action="store_true", help="add bootstrap CI rows (see module docstring)")
    ap.add_argument("--ci-level", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    # Both loaders glob relative to their own sweep directory.
    with contextlib.chdir(CLASS_DIR):
        tot = load_total("parsed_data/*/_group/csv/summary_manuscript.csv")
    with contextlib.chdir(HERE):
        seq = load("parsed_data/*/_group/csv/eval_manuscript.csv")
    seq = seq[seq.eval_stage == JOINT]
    # Different model categories per frame; rpf1_table re-derives the order itself.
    ev = pd.concat([tot.astype({"model": str}), seq.astype({"model": str})], ignore_index=True)

    ci, ci_note = {}, ""
    if a.ci:
        ci = {**total_ci(sorted(tot.train.unique()), a.n_boot, a.ci_level), **seq_ci(a.ci_level)}
        ci_note = (f"Brackets: {a.ci_level:.0%} hierarchical bootstrap interval over seeds and "
                   f"evaluation trajectories.").replace("%", "\\%")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    arms = set(zip(tot.train, tot.test, tot.feat)) & set(zip(seq.train, seq.test, seq.feat))
    for train, test, feat in sorted(arms, key=lambda p: (p[0] != p[1], p)):
        slug = f"combined_{train}" if train == test else f"combined_{train}_to_{test}"
        lookup = (lambda m, st, pm, met, k=(train, test, feat): ci.get((*k, pm, m, st, met))) if a.ci else None
        tex = rpf1_table(ev, train, test, feat, a.seeds, decimals=a.decimals,
                         use_color=not a.no_color, color_lo=a.color_lo, color_hi=a.color_hi,
                         approaches=APPROACHES,
                         what="Macro Recall (R), Precision (P), and F1 for whole-trajectory "
                              "classification (left) and the per-timestep classification "
                              "(right)",
                         label=f"{slug}_{feat}", ci=lookup, ci_note=ci_note)
        if tex is not None:
            (a.out_dir / f"{slug}_{feat}.tex").write_text(
                tex.replace("\\begin{table}[t]", "\\begin{table}[H]", 1)
                   .replace(" seeds; per-seed spread in eval\\_long.csv.", " seeds.", 1)
                   .replace(" Cell colour: red\\,=\\,low, green\\,=\\,high.", "", 1))
            print(f"wrote {a.out_dir / f'{slug}_{feat}.tex'}")

    # Model size and asymptotic cost per train orbit, headline feature set, as in the old t7.
    with contextlib.chdir(HERE):
        cx = load("parsed_data/*/_group/csv/complexity_manuscript.csv")
    for train in sorted(set(tot.train) & set(cx.train)):
        ct = complexity_table(tot, cx, train, "phys")
        symbols = complexity_legend(ct.index.get_level_values("Complexity"))
        if ct.empty:
            continue
        out = a.out_dir / f"combined_complexity_{train}.tex"
        out.write_text(ct.to_latex(escape=False, multirow=False, position="H", caption=(
             f"Model size and asymptotic inference cost, whole-trajectory (Total Sequence) vs "
             f"per-timestep joint 4-class (In Sequence), {train.upper()} ({feat_label('phys')} "
             f"features). Gradient-boosted trees report no parameter count (--). {symbols}."),
             label=f"tab:combined_complexity_{train}"))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
