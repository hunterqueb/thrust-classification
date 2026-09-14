#!/usr/bin/env python3
"""Aggregate the manuscript sweep -> mean +/- std over seeds -> LaTeX tables.

Run from gmat/data/seqClassification, after runManuscriptSweep.sh's parse step:

    python displaySeqLogData.py . --group-dir leo/ --group-name manuscript   # (per train orbit)
    python aggregateManuscript.py --out-dir manuscript_tables

Consumes displaySeqLogData.py's group CSVs. Only logs whose stem ends in Seed<n> are kept, which
is what excludes the pre-existing non-sweep logs (ResidLadder*, plain Energy_J2Energy_OE, ...)
that --group-dir also rglob'd out of the same directories.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from displaySeqLogData import _suffix   # strips "^\d+min\d+_?"; see its comment for why

SEED_RE = re.compile(r"Seed(\d+)$")
TEST_RE = re.compile(r"Test_([A-Za-z][A-Za-z0-9\-/]*)_")
PROP_RE = re.compile(r"^(\d+)min")
KEYS = ["feat", "train", "test", "propMin", "model", "eval_stage"]

JOINT = "joint_4class"
E2E = "cascade_end_to_end"


def _config(stem: str, relpath: str) -> dict:
    """Config back out of the log name. Train orbit comes from the relpath (logLoc uses --orbit, so
    a leo->geo run lives under leo/); everything else from the stem's strAdd tail."""
    suf = _suffix(stem)                      # e.g. "Energy_J2Energy_OE_Test_geo_Seed1"
    train = relpath.replace("\\", "/").split("/")[0]
    t = TEST_RE.search(suf)
    m = SEED_RE.search(suf)
    return {
        "train": train,
        "test": t.group(1) if t else train,
        "propMin": int(PROP_RE.match(stem).group(1)),
        # "OE_" is the marker for the phys arm; the eci arm carries no feature tokens at all.
        "feat": "phys" if re.search(r"(^|_)OE(_|$)", suf) else "eci",
        "seed": int(m.group(1)) if m else -1,
    }


def load(pattern: str) -> pd.DataFrame:
    paths = sorted(Path(".").glob(pattern))
    if not paths:
        raise SystemExit(f"no CSVs matched {pattern!r} -- run the sweep's parse step first")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df["log_stem"].str.contains(r"Seed\d+$", regex=True)].copy()
    # OnePass runs are 1-epoch smoke tests. _config maps them onto the SAME cell as the real run
    # with those settings (the OnePass_ token is not part of the config it extracts), so without
    # this they would silently average into manuscript numbers.
    n_before = len(df)
    df = df[~df["log_stem"].str.contains("OnePass_")].copy()
    if len(df) < n_before:
        print(f"  (excluded {n_before - len(df)} rows from --one-pass smoke-test logs)")
    if df.empty:
        raise SystemExit(f"{pattern!r} matched files, but none carried a Seed<n> stem")
    cfg = pd.DataFrame([_config(s, r) for s, r in zip(df["log_stem"], df["log_relpath"])],
                       index=df.index)
    return pd.concat([df, cfg], axis=1)


def agg(df: pd.DataFrame, metric: str, keys=KEYS, n_expected: int | None = None) -> pd.DataFrame:
    """mean +/- std over seeds, pre-formatted into a LaTeX cell string."""
    a = df.groupby(keys, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
    a["cell"] = [
        "--" if n == 0 else (f"{m:.3f}" if n < 2 else f"{m:.3f} $\\pm$ {s:.3f}")
        for m, s, n in zip(a["mean"], a["std"], a["count"])
    ]
    # groupby.mean() skips NaN silently (a class with zero predictions gives NaN recall), so a short
    # count is the ONLY signal that a seed dropped out of this cell.
    # A count ABOVE n_expected means two different log stems collapsed onto one cell -- some stem
    # carries a token _config does not extract, so runs that are not replicates are being averaged.
    if n_expected:
        for _, r in a[(a["count"] < n_expected) | (a["count"] > n_expected)].iterrows():
            how = "only" if r["count"] < n_expected else "!! MORE THAN"
            print(f"  ! {metric}: {how} {r['count']}/{n_expected} seeds for "
                  f"{'/'.join(str(r[k]) for k in keys)}")
    return a


def paired_delta(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """phys minus eci, differenced WITHIN each seed before averaging. Both arms see the same split
    at a given seed, so pairing removes split variance -- much tighter than mean-minus-mean."""
    keys = [k for k in KEYS if k != "feat"] + ["seed"]
    w = df.pivot_table(index=keys, columns="feat", values=metric, aggfunc="mean")
    if not {"eci", "phys"} <= set(w.columns):
        return pd.DataFrame()
    w = w.dropna(subset=["eci", "phys"])
    w["d"] = w["phys"] - w["eci"]
    out = w.reset_index().groupby([k for k in keys if k != "seed"])["d"].agg(["mean", "std", "count"]).reset_index()
    out["cell"] = [f"{m:+.3f} $\\pm$ {s:.3f}" if n >= 2 else f"{m:+.3f}"
                   for m, s, n in zip(out["mean"], out["std"], out["count"])]
    return out


def emit(piv: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    if piv.empty:
        print(f"  (skip {path.name}: no rows)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(piv.to_latex(escape=False, na_rep="--", caption=caption, label=label))
    print(f"wrote {path}")


def _pivot(a, index, columns):
    return a.pivot_table(index=index, columns=columns, values="cell", aggfunc="first")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("manuscript_tables"))
    ap.add_argument("--headline-prop", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3, help="expected seeds per cell; warns if short")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return _selfcheck()

    ev = load("parsed_data/*/_group/csv/eval_manuscript.csv")
    cmp_ = load("parsed_data/*/_group/csv/comparison_manuscript.csv")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    # Tidy long form -- when a reviewer asks for a number that is not in a table, slice this instead
    # of editing the script.
    ev.to_csv(a.out_dir / "eval_long.csv", index=False)
    print(f"wrote {a.out_dir / 'eval_long.csv'}  ({len(ev)} rows, "
          f"{ev['seed'].nunique()} seeds, {ev['model'].nunique()} models)")

    ind = ev[ev.train == ev.test]
    hp, both = a.headline_prop, [JOINT, E2E]

    # T1 -- headline, in-distribution, best feature set.
    t = agg(ind[(ind.propMin == hp) & (ind.feat == "phys") & ind.eval_stage.isin(both)],
            "macro_f1", n_expected=a.seeds)
    emit(_pivot(t, "model", ["train", "eval_stage"]), a.out_dir / "t1_main.tex",
         f"In-distribution per-timestep macro-F1 ({hp} min windows, OE + energy + $J_2$ features), "
         f"mean $\\pm$ std over {a.seeds} seeds.", "tab:main")

    # T2 -- feature ablation. macro_f1 AND min_thrust_class_recall: an accuracy-only table hides
    # total failure on the rarest class (Electric recall has been 0.0009 at 77.7% accuracy).
    for metric, tag in (("macro_f1", "f1"), ("min_thrust_class_recall", "minrec")):
        t = agg(ind[(ind.propMin == hp) & ind.eval_stage.isin(both)], metric, n_expected=a.seeds)
        emit(_pivot(t, ["model", "eval_stage"], ["train", "feat"]),
             a.out_dir / f"t2_features_{tag}.tex",
             f"Feature-set ablation, in-distribution {metric.replace('_', ' ')} "
             f"({hp} min), mean $\\pm$ std over {a.seeds} seeds.", f"tab:feat_{tag}")
        d = paired_delta(ind[(ind.propMin == hp) & ind.eval_stage.isin(both)], metric)
        emit(_pivot(d, ["model", "eval_stage"], "train"),
             a.out_dir / f"t2_features_{tag}_delta.tex",
             f"Paired phys $-$ eci delta in {metric.replace('_', ' ')} ({hp} min), differenced "
             f"within each seed then averaged.", f"tab:featdelta_{tag}")

    # T3 -- window length.
    t = agg(ind[(ind.feat == "phys") & ind.eval_stage.isin(both)], "macro_f1", n_expected=a.seeds)
    emit(_pivot(t, ["model", "eval_stage"], ["train", "propMin"]), a.out_dir / "t3_window.tex",
         f"Effect of propagation window on in-distribution macro-F1 (OE + energy + $J_2$), "
         f"mean $\\pm$ std over {a.seeds} seeds.", "tab:window")

    # T4 -- cross-regime transfer. In-distribution columns are the reference the drop is measured
    # against. combined/leo-meo-geo is excluded from the sweep entirely (93% LEO; 1390/1500 ICs
    # byte-identical to the standalone LEO set) -- say so in the caption.
    xr = ev[(ev.propMin == hp) & (ev.feat == "phys") & ev.eval_stage.isin(both) &
            ev.train.isin(["leo", "geo"])].copy()
    xr["arm"] = xr.train + r"$\rightarrow$" + xr.test
    for metric, tag in (("macro_f1", "f1"), ("min_thrust_class_recall", "minrec")):
        t = agg(xr, metric, keys=["arm", "model", "eval_stage"], n_expected=a.seeds)
        emit(_pivot(t, ["model", "eval_stage"], "arm"), a.out_dir / f"t4_transfer_{tag}.tex",
             f"Cross-regime transfer, {metric.replace('_', ' ')} ({hp} min, OE + energy + $J_2$). "
             f"The mixed combined/leo-meo-geo dataset is excluded: it is 93\\% LEO and 1390 of its "
             f"1500 ICs are identical to the standalone LEO set.", f"tab:transfer_{tag}")

    # T5 -- cascade error attribution. This is what justifies the cascade: loss has been almost
    # entirely in detection, not typing (stage-2 conditional 96.75% vs 77.67% end-to-end).
    st = ind[(ind.propMin == hp) & (ind.feat == "phys")]
    parts = []
    for stage, metric, name in (
            ("cascade_stage1_standalone", "macro_f1", "Stage1 F1"),
            ("cascade_stage2_standalone", "macro_f1", "Stage2 F1"),
            (E2E, "stage2_conditional_accuracy", "Stage2 cond. acc"),
            (E2E, "stage1_only_accuracy", "Stage1 only acc"),
            (E2E, "macro_f1", "End-to-end F1")):
        p = agg(st[st.eval_stage == stage], metric, keys=["train", "model"], n_expected=a.seeds)
        parts.append(p.assign(col=name))
    emit(_pivot(pd.concat(parts, ignore_index=True), ["train", "model"], "col"),
         a.out_dir / "t5_cascade.tex",
         f"Cascade error attribution, in-distribution ({hp} min, OE + energy + $J_2$).",
         "tab:cascade")

    # T6 -- per-class. Electric is the scientific claim; Impulsive is the rarest.
    cols = [c for c in ("electric_recall", "electric_f1", "impulsive_recall", "impulsive_f1")
            if c in ind.columns]
    parts = [agg(ind[(ind.propMin == hp) & ind.eval_stage.isin(both)], c,
                 keys=["feat", "train", "model", "eval_stage"], n_expected=a.seeds).assign(col=c)
             for c in cols]
    emit(_pivot(pd.concat(parts, ignore_index=True), ["model", "eval_stage", "train"], ["col", "feat"]),
         a.out_dir / "t6_perclass.tex",
         f"Per-class recall and F1 for the two hardest classes, in-distribution ({hp} min).",
         "tab:perclass")

    _best_cascade(ev, cmp_, a)


def _best_cascade(ev: pd.DataFrame, cmp_: pd.DataFrame, a) -> None:
    """Two separate questions -- do not conflate them."""
    print("\n=== Does cascade beat joint? (win counts + mean signed delta) ===")
    col = "Better_By_Macro_F1"
    if col in cmp_.columns:
        wins = cmp_.groupby(["feat", "Model"])[col].value_counts().unstack(fill_value=0)
        delta = cmp_.groupby(["feat", "Model"])["Macro_F1_Delta_Cascade_minus_Joint"].mean()
        out = wins.join(delta.rename("mean_delta"))
        print(out.to_string())
        out.to_csv(a.out_dir / "cascade_vs_joint.csv")
        # A model that wins 15/15 by 0.002 is a different claim from one that wins 9/15 by 0.06.
        print(f"wrote {a.out_dir / 'cascade_vs_joint.csv'}")
    else:
        print(f"  (no {col} column in comparison CSVs)")

    print("\n=== Which cascade model is best? (not in comparison.csv -- it never compares models) ===")
    e2e = ev[ev.eval_stage == E2E]
    cells = [c for c in KEYS if c != "model"]
    per = e2e.groupby(cells + ["model"])["macro_f1"].mean().reset_index()
    win = per.loc[per.groupby(cells)["macro_f1"].idxmax(), "model"].value_counts()
    rank = (e2e.groupby("model")[["macro_f1", "min_thrust_class_recall", "inference_time_s"]]
            .mean().sort_values(["macro_f1", "min_thrust_class_recall"], ascending=[False, False]))
    rank["cells_won"] = win.reindex(rank.index).fillna(0).astype(int)
    rank["of_cells"] = per.groupby(cells).ngroups
    print(rank.to_string())
    rank.to_csv(a.out_dir / "best_cascade.csv")
    print(f"wrote {a.out_dir / 'best_cascade.csv'}")


def _selfcheck() -> None:
    assert _config("30min1500Energy_J2Energy_OE_Test_geo_Seed1", "leo/30min-1500/x.log") == {
        "train": "leo", "test": "geo", "propMin": 30, "feat": "phys", "seed": 1}
    assert _config("100min1500EvalTest_Seed2", "geo/100min-1500/x.log") == {
        "train": "geo", "test": "geo", "propMin": 100, "feat": "eci", "seed": 2}
    assert _config("10min1500Energy_J2Energy_OE_EvalTest_Seed0", "meo/10min-1500/x.log") == {
        "train": "meo", "test": "meo", "propMin": 10, "feat": "phys", "seed": 0}
    # The one that matters: pre-existing non-sweep logs carry no seed, so the load() filter drops
    # them -- and their OE token must not fool the feat detector into claiming they are sweep rows.
    assert SEED_RE.search(_suffix("30min1500Energy_J2Energy_ResidLadder3_OE")) is None
    assert SEED_RE.search(_suffix("30min1500Energy_OE")) is None
    # A OnePass smoke log parses to the SAME config as the real run it shadows -- which is exactly
    # why load() drops it by stem rather than trusting _config to tell them apart.
    assert (_config("10min1500Energy_J2Energy_OE_OnePass_EvalTest_Seed0", "leo/x/y.log")
            == _config("10min1500Energy_J2Energy_OE_EvalTest_Seed0", "leo/x/y.log"))
    print("selfcheck ok")


if __name__ == "__main__":
    main()
