#!/usr/bin/env python3
"""Aggregate the manuscript sweep -> mean +/- std over seeds -> LaTeX tables.

Run from gmat/data/seqClassification, after runManuscriptSweep.sh's parse step:

    python displaySeqLogData.py . --group-dir leo/ --group-name manuscript   # (per train orbit)
    python aggregateManuscript.py --out-dir manuscript_tables

Consumes displaySeqLogData.py's group CSVs (eval_, comparison_, and -- for the t7 model-size /
inference-time / Big-O table -- complexity_manuscript.csv). Only logs whose stem ends in Seed<n> are kept, which
is what excludes the pre-existing non-sweep logs (ResidLadder*, plain Energy_J2Energy_OE, ...)
that --group-dir also rglob'd out of the same directories.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from displaySeqLogData import _suffix   # strips "^\d+min\d+_?"; see its comment for why
from displaySeqLogData import COMPLEXITY_SYMBOLS

# Reuse the existing tables' colour ramp and cell formatter rather than re-deriving them, so the
# sequence-classification tables render identically to the ones already in gmat/data/tables/ and a
# retune in one place carries to both.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classification"))
from generateLatexTableCompact import format_cell, val_to_color  # noqa: E402,F401

SEED_RE = re.compile(r"Seed(\d+)$")
TEST_RE = re.compile(r"Test_([A-Za-z][A-Za-z0-9\-/]*)_")
PROP_RE = re.compile(r"^(\d+)min")
KEYS = ["feat", "train", "test", "propMin", "model", "eval_stage"]

JOINT = "joint_4class"
E2E = "cascade_end_to_end"

# Display names for the feature arms. The internal keys stay "eci"/"phys"/"ladder" -- they are what
# _config derives from the log stem, what every filter here matches on, and what the .tex/.pdf
# filenames and \label{} identifiers use, so renaming them would break slicing and cross-references
# for a cosmetic gain. This map is applied only where a human reads the value: captions, table
# column headers, figure titles, and the summary CSVs. Order is the ablation order (6 -> 8 -> 11
# channels), not alphabetical, and is carried into pivots via an ordered Categorical.
FEAT_LABEL = {"eci": "ECI",
              "phys": "OE + Energy",
              "ladder": "OE + Acceleration Residual"}
FEAT_DISPLAY_ORDER = [FEAT_LABEL[k] for k in ("eci", "phys", "ladder")]


def feat_label(feat: str) -> str:
    """Display name, falling through unchanged for an arm this map does not know about."""
    return FEAT_LABEL.get(str(feat), str(feat))


def _disp(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with `feat` swapped for its display name, as an ordered Categorical so pivot_table
    keeps the ablation order instead of sorting the new labels alphabetically (which would put
    "OE + Acceleration Residual" before "OE + Energy" and both before "ECI")."""
    if "feat" not in df.columns:
        return df
    out = df.copy()
    labels = [f for f in FEAT_DISPLAY_ORDER if f in set(out["feat"].map(feat_label))]
    extra = sorted({feat_label(f) for f in out["feat"]} - set(labels))
    out["feat"] = pd.Categorical(out["feat"].map(feat_label),
                                  categories=labels + extra, ordered=True)
    return out

# Row order for every table: classic/shallow baselines first, then deep sequence models, each
# group alphabetical (case-insensitive -- names are mixed case, e.g. LightGBM vs MAMBA).
# PCA+MLP and MiniRocket sit with the classic group: both are Hankel-window / fixed-transform
# baselines in this codebase, not trained sequence models. HYBRID is MiniRocket stage 1 + CNN
# stage 2, so it goes with the deep group.
CLASSIC_MODELS = frozenset({"CatBoost", "Extra Trees", "LightGBM", "MiniRocket",
                            "PCA+MLP", "Random Forest", "XGBoost"})
DEEP_MODELS = frozenset({"CNN", "HYBRID", "LSTM", "MAMBA", "TRANSFORMER"})


def model_order(names) -> list:
    """Classic first, then deep, then anything unrecognised -- each alphabetical. An unknown name
    is reported rather than silently folded into a group, so a newly added backbone is visible."""
    names = sorted({str(n) for n in names if pd.notna(n)}, key=str.lower)
    classic = [n for n in names if n in CLASSIC_MODELS]
    deep = [n for n in names if n in DEEP_MODELS]
    other = [n for n in names if n not in CLASSIC_MODELS and n not in DEEP_MODELS]
    if other:
        print(f"  ! unrecognised model(s) placed last: {other} "
              f"(add to CLASSIC_MODELS/DEEP_MODELS in {Path(__file__).name})")
    return classic + deep + other


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
        # Order matters: the ladder arm is phys PLUS --residual-ladder, so it also carries OE_.
        # Checking OE first would silently fold the two arms together.
        "feat": ("ladder" if "ResidLadder" in suf
                 else "phys" if re.search(r"(^|_)OE(_|$)", suf) else "eci"),
        "seed": int(m.group(1)) if m else -1,
    }


def load(pattern: str, rename_models: dict | None = None) -> pd.DataFrame:
    """rename_models maps log model names onto this script's (used by the whole-trajectory
    aggregator, classification/aggregateManuscriptTotal.py) before the model order is fixed."""
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
    out = pd.concat([df, cfg], axis=1)
    # Ordered Categorical is the single lever: groupby and pivot_table both sort by category order,
    # so every table inherits classic-then-deep without each call site restating it. The eval CSV
    # calls the column "model", the comparison CSV "Model".
    for col in ("model", "Model"):
        if col in out.columns:
            if rename_models:
                out[col] = out[col].replace(rename_models)
            out[col] = pd.Categorical(out[col], categories=model_order(out[col].unique()),
                                       ordered=True)
    return out


def agg(df: pd.DataFrame, metric: str, keys=KEYS, n_expected: int | None = None) -> pd.DataFrame:
    """mean +/- std over seeds, pre-formatted into a LaTeX cell string."""
    a = df.groupby(keys, dropna=False, observed=True)[metric].agg(["mean", "std", "count"]).reset_index()
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


def paired_delta(df: pd.DataFrame, metric: str, hi: str = "phys", lo: str = "eci") -> pd.DataFrame:
    """hi minus lo, differenced WITHIN each seed before averaging. Both arms see the same split at
    a given seed, so pairing removes split variance -- much tighter than mean-minus-mean."""
    keys = [k for k in KEYS if k != "feat"] + ["seed"]
    w = df.pivot_table(index=keys, columns="feat", values=metric, aggfunc="mean",
                        observed=True)
    if not {lo, hi} <= set(w.columns):
        return pd.DataFrame()
    w = w.dropna(subset=[lo, hi])
    w["d"] = w[hi] - w[lo]
    out = (w.reset_index().groupby([k for k in keys if k != "seed"], observed=True)["d"]
             .agg(["mean", "std", "count"]).reset_index())
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
    # paired_delta returns an empty frame when an arm is absent (e.g. a partial sweep, or --FEATS
    # limited to one arm). pivot_table raises KeyError('cell') on that, so absorb it here rather
    # than at each call site -- emit() already no-ops on an empty frame.
    if a.empty or "cell" not in a.columns:
        return pd.DataFrame()
    return a.pivot_table(index=index, columns=columns, values="cell", aggfunc="first",
                         observed=True)


APPROACH_LABEL = {JOINT: "Joint", E2E: "Cascade"}
RPF1 = (("macro_recall", r"\textbf{R}"),
        ("macro_precision", r"\textbf{P}"),
        ("macro_f1", r"\textbf{F1}"))


def rpf1_table(ev: pd.DataFrame, train: str, test: str, feat: str, seeds: int,
                decimals: int = 2, use_color: bool = True,
                color_lo: float = 0.25, color_hi: float = 0.75,
                approaches: dict | None = None, what: str | None = None,
                label: str | None = None) -> str | None:
    """One table per orbit group: Joint vs Cascade macro R/P/F1 across every propagation window.

    train == test gives the in-distribution table; train != test gives the out-of-distribution
    transfer table in the identical layout, so the two can be read side by side.

    Layout mirrors gmat/data/tables/class_*_feat.tex -- two spanning header groups over 3-metric
    blocks per time domain, bold on the column max, cell colour on the value. Cells are the MEAN
    over seeds with no +/-: 18 coloured columns with error terms is unreadable, and the per-seed
    spread is in eval_long.csv.

    approaches/what/label let the whole-trajectory aggregator (classification/
    aggregateManuscriptTotal.py) reuse this layout with its single approach; the defaults are
    this script's Joint/Cascade table.
    """
    approaches = approaches or APPROACH_LABEL
    what = what or ("Per-timestep macro Recall (R), Precision (P), and F1 for the joint 4-class "
                    "model and the end-to-end cascade")
    label = label or f"{arm_slug(train, test)}_{feat}_jc"
    d = ev[(ev.train == train) & (ev.test == test) & (ev.feat == feat)
           & ev.eval_stage.isin(approaches)]
    if d.empty:
        return None
    props = sorted(d.propMin.unique())
    # Column order must match the header groups built below: approach outer, time inner, metric last.
    blocks = [(st, pm) for st in approaches for pm in props]

    piv = (d.groupby(["model", "eval_stage", "propMin"], observed=True)[[m for m, _ in RPF1]]
             .mean().reset_index())
    models = model_order(piv.model.unique())

    def cell(model, stage, pm, metric):
        r = piv[(piv.model == model) & (piv.eval_stage == stage) & (piv.propMin == pm)]
        return float("nan") if r.empty else r.iloc[0][metric]

    # Bold = best per (approach, time, metric) column, matching the reference tables' rule.
    best = {(st, pm, m): max((cell(x, st, pm, m) for x in models), default=float("nan"))
            for st, pm in blocks for m, _ in RPF1}

    ncol = len(blocks) * len(RPF1)
    spec = "l " + " ".join(["ccc"] * len(blocks))
    per_approach = len(props) * len(RPF1)

    top, tcm, mid, mcm, c = [], [], [], [], 2
    for st in approaches:
        top.append(f"\\multicolumn{{{per_approach}}}{{c}}{{\\textbf{{{approaches[st]}}}}}")
        tcm.append(f"\\cmidrule(lr){{{c}-{c + per_approach - 1}}}")
        c += per_approach
    c = 2
    for _st, pm in blocks:
        mid.append(f"\\multicolumn{{{len(RPF1)}}}{{c}}{{\\textbf{{{pm} Minute}}}}")
        mcm.append(f"\\cmidrule(lr){{{c}-{c + len(RPF1) - 1}}}")
        c += len(RPF1)

    rows = []
    for model in models:
        cells = [format_cell(cell(model, st, pm, m),
                             is_bold=(not pd.isna(cell(model, st, pm, m))
                                      and cell(model, st, pm, m) >= best[(st, pm, m)]),
                             use_color=use_color, decimals=decimals,
                             color_lo=color_lo, color_hi=color_hi)
                 for st, pm in blocks for m, _ in RPF1]
        rows.append(f"\\textbf{{{model}}} & " + " & ".join(cells) + r" \\")

    hdr = " & ".join(lbl for _ in blocks for _, lbl in RPF1)
    return (
        "% Requires \\usepackage[table]{xcolor} in LaTeX preamble.\n"
        "\\begin{table}[t]\n\\centering\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\renewcommand{\\arraystretch}{1.1}\n\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n"
        f" & {' & '.join(top)} \\\\\n{''.join(tcm)}\n"
        f" & {' & '.join(mid)} \\\\\n{''.join(mcm)}\n"
        f"\\textbf{{Model}} & {hdr} \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}%\n}\n"
        f"\\caption{{{what}, {_arm_phrase(train, test)}, {feat_label(feat)} "
        f"features, "
        f"at {'/'.join(str(p) for p in props)} minutes. Mean over {seeds} seeds; per-seed spread in "
        f"eval\\_long.csv. Bold = best per approach/time/metric. Cell colour: red\\,=\\,low, "
        f"green\\,=\\,high.}}\n"
        f"\\label{{tab:{label}}}\n\\end{{table}}\n"
    )


COMPLEXITY_APPROACHES = ["Joint", "Cascade"]


def complexity_table(cx: pd.DataFrame, train: str, feat: str) -> pd.DataFrame:
    """Model x {Joint, Cascade}: parameters, memory, and the asymptotic inference cost.

    In-distribution only -- a transfer arm runs the same trained models, so its size and cost are
    not new information. Params/memory are fixed by the architecture and input width, so they are
    shown as a plain mean over seeds and windows. Measured inference time is left out of the table
    (the logged timers are not scoped identically for joint and cascade); it stays in
    complexity_manuscript.csv.

    Cascade params/memory are Stage 1 + Stage 2 (displaySeqLogData.build_complexity)."""
    d = cx[(cx.train == train) & (cx.test == train) & (cx.feat == feat)].copy()
    if d.empty:
        return pd.DataFrame()
    d["Approach"] = pd.Categorical(d["Approach"], categories=COMPLEXITY_APPROACHES, ordered=True)
    keys = ["Model", "Approach"]
    g = d.groupby(keys, observed=True)

    def _int(v: float) -> str:
        return "--" if pd.isna(v) else f"{int(round(v)):,}".replace(",", "{,}")

    out = pd.DataFrame({
        "Params": g["Params"].mean().map(_int),
        "Memory (MB)": g["Memory_MB"].mean().map(lambda v: "--" if pd.isna(v) else f"{v:.3f}"),
    })
    # Joint and cascade share one Big-O (the cascade is ~2x the same cost), so it is an index level
    # beside Model rather than a column: to_latex sparsifies repeated index labels, which prints it
    # once per model group.
    out["Complexity"] = g["Big_O_LaTeX"].first()
    return out.set_index("Complexity", append=True).reorder_levels(["Model", "Complexity", "Approach"])


def complexity_legend(big_o_latex) -> str:
    """Caption legend restricted to the symbols the given Big-O expressions actually use, in
    COMPLEXITY_SYMBOLS order -- so e.g. p (PCA components) is only defined when PCA+MLP is in the
    table. LaTeX commands (\\mathcal{O}, \\,) are stripped first so their letters are not mistaken
    for symbols."""
    used = set()
    for s in big_o_latex:
        used.update(re.findall(r"[A-Za-z]+", re.sub(r"\\[A-Za-z]+(\{[^}]*\})?|\\,", " ", str(s))))
    entries = [e.strip() for e in COMPLEXITY_SYMBOLS.split(",")]
    return ", ".join(e.replace("=", r"\,=\,", 1) for e in entries if e.split("=", 1)[0] in used)


def arm_slug(train: str, test: str) -> str:
    """Filename/label stem. In-distribution keeps the original single-orbit name so existing
    references to tab:seq_leo_phys_jc do not break."""
    return f"seq_{train}" if train == test else f"seq_{train}_to_{test}"


def _arm_phrase(train: str, test: str) -> str:
    if train == test:
        return f"{train.upper()} train and test"
    return f"trained on {train.upper()}, tested out-of-distribution on {test.upper()}"


# Two-series categorical pair, Okabe-Ito. Validated with the dataviz palette checker against the
# light chart surface: lightness band PASS, chroma floor PASS, CVD separation dE 21.9 (protan) /
# 30.9 (tritan) PASS, normal-vision dE 31.2 PASS, contrast >= 3:1 PASS. Do not substitute
# eyeballed colours -- re-run the validator if these change.
BOX_COLORS = {JOINT: "#0072B2", E2E: "#D55E00"}


def boxplot_pdf(ev: pd.DataFrame, out_dir: Path, metric: str = "macro_f1",
                 stem: str = "macro_f1_boxplots") -> list[Path]:
    """Box plots of `metric` for every propagation window x orbit regime: **one PDF per feature
    arm** (`<stem>_<feat>.pdf`), each a single figure with one subplot per train->test arm and two
    dodged boxes (Joint / Cascade) per window. Returns the paths written.

    Separate files rather than one multi-page PDF so each can be \\includegraphics'd as its own
    manuscript figure without page-extraction.

    Each box pools models x seeds. That deliberately mixes two sources of spread -- architecture
    and split -- so the box reads as "what this configuration achieves across the model zoo", not
    as a seed-variance error bar. The individual points are overlaid so the reader can see n and
    where the zoo actually sits; a wide box with a tight cluster plus one outlier is a different
    story from a genuinely diffuse one, and the strip is what distinguishes them.
    """
    import matplotlib
    matplotlib.use("Agg")            # headless: this runs over SSH / in the sweep script
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    d = ev[ev.eval_stage.isin(APPROACH_LABEL) & ev[metric].notna()]
    if d.empty:
        print(f"  (skip {stem}: no {metric} rows)")
        return []
    stages = [st for st in APPROACH_LABEL if st in set(d.eval_stage)]
    arms = sorted(set(zip(d.train.astype(str), d.test.astype(str))),
                  key=lambda a: (a[0] != a[1], a))
    rng = np.random.default_rng(0)   # fixed jitter, so the figure is reproducible

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for feat in sorted(d.feat.unique()):
            df = d[d.feat == feat]
            props = sorted(df.propMin.unique())
            fig, axes = plt.subplots(1, len(arms), figsize=(2.9 * len(arms) + 0.8, 3.7),
                                      sharey=True, squeeze=False)
            n_seen = set()
            for ax, (tr, te) in zip(axes[0], arms):
                sub = df[(df.train == tr) & (df.test == te)]
                for j, st in enumerate(stages):
                    # Dodge the two approaches around each window's tick.
                    off = (j - (len(stages) - 1) / 2) * 0.34
                    groups = [sub[(sub.propMin == pm) & (sub.eval_stage == st)][metric].values
                              for pm in props]
                    n_seen.update(len(g) for g in groups if len(g))
                    pos = [i + off for i in range(len(props))]
                    col = BOX_COLORS.get(st, "#666666")
                    bp = ax.boxplot(groups, positions=pos, widths=0.30, patch_artist=True,
                                    showfliers=False, manage_ticks=False,
                                    medianprops=dict(color=col, lw=1.8),
                                    whiskerprops=dict(color=col, lw=1.0),
                                    capprops=dict(color=col, lw=1.0))
                    for box in bp["boxes"]:
                        box.set(facecolor=col, alpha=0.22, edgecolor=col, lw=1.2)
                    for i, g in enumerate(groups):     # strip overlay: show n and the clustering
                        if len(g):
                            ax.plot(pos[i] + rng.uniform(-0.07, 0.07, len(g)), g, "o",
                                    ms=2.6, color=col, alpha=0.65, mew=0, zorder=3)
                ax.set_xticks(range(len(props)))
                ax.set_xticklabels([f"{pm}" for pm in props])
                ax.set_xlabel("window (min)")
                ax.set_title(_arm_title(tr, te), fontsize=10)
                ax.set_ylim(0, 1)
                ax.yaxis.grid(True, lw=0.5, color="#dddddd")
                ax.set_axisbelow(True)                 # recessive grid, behind the marks
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if ax is not axes[0][0]:
                    # sharey already suppresses the inner tick labels; the spine is then pure
                    # non-data ink repeating a scale the leftmost axis already carries.
                    ax.spines["left"].set_visible(False)
                    ax.tick_params(axis="y", length=0)
            axes[0][0].set_ylabel(metric.replace("_", " "))
            # Legend always present for >= 2 series -- identity is never colour-alone.
            fig.legend(handles=[Patch(facecolor=BOX_COLORS[st], alpha=0.22,
                                      edgecolor=BOX_COLORS[st], label=APPROACH_LABEL[st])
                                for st in stages],
                       loc="lower center", ncol=len(stages), frameon=False,
                       bbox_to_anchor=(0.5, -0.02))
            n = f"{min(n_seen)}-{max(n_seen)}" if len(n_seen) > 1 else str(max(n_seen, default=0))
            fig.suptitle(f"{metric.replace('_', ' ')} by window and regime - "
                         f"{feat_label(feat)} features "
                         f"(each box: {n} points = models x seeds)", fontsize=11)
            fig.tight_layout(rect=(0, 0.05, 1, 0.94))
            path = out_dir / f"{stem}_{feat}.pdf"
            fig.savefig(path)
            plt.close(fig)
            written.append(path)
            print(f"wrote {path}")
    return written


def _arm_title(train: str, test: str) -> str:
    return train.upper() if train == test else f"{train.upper()} $\\rightarrow$ {test.upper()}"


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
    ap.add_argument("--no-plots", action="store_true", help="skip the box-plot PDF")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return _selfcheck()

    ev = load("parsed_data/*/_group/csv/eval_manuscript.csv")
    cmp_ = load("parsed_data/*/_group/csv/comparison_manuscript.csv")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    # Tidy long form -- when a reviewer asks for a number that is not in a table, slice this instead
    # of editing the script.
    ev.assign(feat_label=ev["feat"].map(feat_label)).to_csv(
        a.out_dir / "eval_long.csv", index=False)
    print(f"wrote {a.out_dir / 'eval_long.csv'}  ({len(ev)} rows, "
          f"{ev['seed'].nunique()} seeds, {ev['model'].nunique()} models)")

    ind = ev[ev.train == ev.test]
    hp, both = a.headline_prop, [JOINT, E2E]

    # T1 -- headline, in-distribution, best feature set.
    t = agg(ind[(ind.propMin == hp) & (ind.feat == "phys") & ind.eval_stage.isin(both)],
            "macro_f1", n_expected=a.seeds)
    emit(_pivot(t, "model", ["train", "eval_stage"]), a.out_dir / "t1_main.tex",
         f"In-distribution per-timestep macro-F1 ({hp} min windows, {feat_label('phys')} "
         f"features), "
         f"mean $\\pm$ std over {a.seeds} seeds.", "tab:main")

    # T2 -- feature ablation. macro_f1 AND min_thrust_class_recall: an accuracy-only table hides
    # total failure on the rarest class (Electric recall has been 0.0009 at 77.7% accuracy).
    for metric, tag in (("macro_f1", "f1"), ("min_thrust_class_recall", "minrec")):
        t = agg(ind[(ind.propMin == hp) & ind.eval_stage.isin(both)], metric, n_expected=a.seeds)
        emit(_pivot(_disp(t), ["model", "eval_stage"], ["train", "feat"]),
             a.out_dir / f"t2_features_{tag}.tex",
             f"Feature-set ablation, in-distribution {metric.replace('_', ' ')} "
             f"({hp} min), mean $\\pm$ std over {a.seeds} seeds.", f"tab:feat_{tag}")
        # One delta per ablation STEP, so a gain is attributable to the feature that caused it
        # rather than to the whole bundle.
        for hi, lo in (("phys", "eci"), ("ladder", "phys")):
            d = paired_delta(ind[(ind.propMin == hp) & ind.eval_stage.isin(both)], metric, hi, lo)
            emit(_pivot(d, ["model", "eval_stage"], "train"),
                 a.out_dir / f"t2_features_{tag}_delta_{hi}_vs_{lo}.tex",
                 f"Paired {feat_label(hi)} $-$ {feat_label(lo)} delta in "
                 f"{metric.replace('_', ' ')} ({hp} min), differenced within each seed then "
                 f"averaged.", f"tab:featdelta_{tag}_{hi}")

    # T3 -- window length.
    t = agg(ind[(ind.feat == "phys") & ind.eval_stage.isin(both)], "macro_f1", n_expected=a.seeds)
    emit(_pivot(t, ["model", "eval_stage"], ["train", "propMin"]), a.out_dir / "t3_window.tex",
         f"Effect of propagation window on in-distribution macro-F1 ({feat_label('phys')}), "
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
             f"Cross-regime transfer, {metric.replace('_', ' ')} ({hp} min, "
             f"{feat_label('phys')}). "
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
         f"Cascade error attribution, in-distribution ({hp} min, {feat_label('phys')}).",
         "tab:cascade")

    # T6 -- per-class. Electric is the scientific claim; Impulsive is the rarest.
    cols = [c for c in ("electric_recall", "electric_f1", "impulsive_recall", "impulsive_f1")
            if c in ind.columns]
    parts = [agg(ind[(ind.propMin == hp) & ind.eval_stage.isin(both)], c,
                 keys=["feat", "train", "model", "eval_stage"], n_expected=a.seeds).assign(col=c)
             for c in cols]
    emit(_pivot(_disp(pd.concat(parts, ignore_index=True)),
                ["model", "eval_stage", "train"], ["col", "feat"]),
         a.out_dir / "t6_perclass.tex",
         f"Per-class recall and F1 for the two hardest classes, in-distribution ({hp} min).",
         "tab:perclass")

    # T7 -- model size, inference time and asymptotic cost, joint vs. cascade. One table per train
    # orbit at the headline feature set. Optional input: group CSVs parsed before displaySeqLogData
    # emitted complexity_<group>.csv simply do not have it.
    cx_pattern = "parsed_data/*/_group/csv/complexity_manuscript.csv"
    if any(Path(".").glob(cx_pattern)):
        cx = load(cx_pattern)
        for train in sorted(cx.train.unique()):
            ct = complexity_table(cx, train, "phys")
            symbols = (complexity_legend(ct.index.get_level_values("Complexity"))
                       if not ct.empty else "")
            emit(ct,
                 a.out_dir / f"t7_complexity_{train}.tex",
                 f"Model size and asymptotic inference cost, {train.upper()} "
                 f"({feat_label('phys')} features). Cascade parameters and memory are Stage 1 + "
                 f"Stage 2; both stages run on every frame. "
                 f"Gradient-boosted trees report no parameter count (--). "
                 f"{symbols}.",
                 f"tab:complexity_{train}")
    else:
        print(f"  (skip t7_complexity: no {cx_pattern} -- re-run displaySeqLogData.py --group-dir)")

    # Per-orbit Joint-vs-Cascade R/P/F1 across every window -- one table per orbit group, per
    # feature arm (the arm has to be fixed within a table: approach x time x metric already fills
    # all 18 columns of the reference layout).
    # Every observed train->test pair: train == test is the in-distribution table, train != test
    # the out-of-distribution transfer table in the identical layout.
    print()
    pairs = sorted(set(zip(ev.train.astype(str), ev.test.astype(str))),
                   key=lambda p: (p[0] != p[1], p))
    for train, test in pairs:
        for feat in sorted(ev.feat.unique()):
            tex = rpf1_table(ev, train, test, feat, a.seeds, decimals=a.decimals,
                             use_color=not a.no_color, color_lo=a.color_lo, color_hi=a.color_hi)
            if tex is None:
                continue
            out = a.out_dir / f"{arm_slug(train, test)}_{feat}_jc.tex"
            out.write_text(tex)
            print(f"wrote {out}")

    if not a.no_plots:
        print()
        boxplot_pdf(ev, a.out_dir, "macro_f1")

    _best_cascade(ev, cmp_, a)


def _best_cascade(ev: pd.DataFrame, cmp_: pd.DataFrame, a) -> None:
    """Two separate questions -- do not conflate them."""
    print("\n=== Does cascade beat joint? (win counts + mean signed delta) ===")
    col = "Better_By_Macro_F1"
    if col in cmp_.columns:
        cd = _disp(cmp_)
        wins = cd.groupby(["feat", "Model"], observed=True)[col].value_counts().unstack(fill_value=0)
        delta = cd.groupby(["feat", "Model"], observed=True)["Macro_F1_Delta_Cascade_minus_Joint"].mean()
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
    per = e2e.groupby(cells + ["model"], observed=True)["macro_f1"].mean().reset_index()
    win = per.loc[per.groupby(cells)["macro_f1"].idxmax(), "model"].value_counts()
    # This one stays ranked by score on purpose -- it exists to answer "which cascade is best",
    # so a fixed row order would defeat it.
    rank = (e2e.groupby("model", observed=True)[["macro_f1", "min_thrust_class_recall", "inference_time_s"]]
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
    # The ladder arm is phys PLUS --residual-ladder, so it also carries OE_ -- it must not be
    # folded into phys, or the two arms average together and the ablation step vanishes.
    assert _config("30min1500Energy_J2Energy_ResidLadder3_OE_EvalTest_Seed2",
                   "geo/30min-1500/x.log") == {
        "train": "geo", "test": "geo", "propMin": 30, "feat": "ladder", "seed": 2}
    assert _config("30min1500Energy_J2Energy_ResidLadder5_OE_Test_leo_Seed1",
                   "geo/30min-1500/x.log") == {
        "train": "geo", "test": "leo", "propMin": 30, "feat": "ladder", "seed": 1}
    # The one that matters: pre-existing non-sweep logs carry no seed, so the load() filter drops
    # them -- and their OE token must not fool the feat detector into claiming they are sweep rows.
    assert SEED_RE.search(_suffix("30min1500Energy_J2Energy_ResidLadder3_OE")) is None
    assert SEED_RE.search(_suffix("30min1500Energy_OE")) is None
    # A OnePass smoke log parses to the SAME config as the real run it shadows -- which is exactly
    # why load() drops it by stem rather than trusting _config to tell them apart.
    assert (_config("10min1500Energy_J2Energy_OE_OnePass_EvalTest_Seed0", "leo/x/y.log")
            == _config("10min1500Energy_J2Energy_OE_EvalTest_Seed0", "leo/x/y.log"))
    assert model_order(["MAMBA", "LightGBM", "CNN", "XGBoost", "LSTM"]) == [
        "LightGBM", "XGBoost", "CNN", "LSTM", "MAMBA"], model_order(
        ["MAMBA", "LightGBM", "CNN", "XGBoost", "LSTM"])
    # Case-insensitive within a group, and an unknown name lands last rather than in either group.
    assert model_order(["TRANSFORMER", "CatBoost", "Extra Trees", "cnn"]) == [
        "CatBoost", "Extra Trees", "TRANSFORMER", "cnn"]
    assert model_order([]) == []
    # rpf1_table is the only non-trivial layout logic here; exercise the full 3-window shape that
    # the real data cannot reach until the sweep finishes.
    rows = [{"train": "leo", "test": "leo", "feat": "phys", "model": m, "eval_stage": st,
             "propMin": pm, "seed": sd, "macro_recall": 0.9, "macro_precision": 0.8,
             "macro_f1": 0.85 if m != "LightGBM" else 0.4}
            for m in ("CNN", "LSTM", "LightGBM") for st in (JOINT, E2E)
            for pm in (10, 30, 100) for sd in (0, 1, 2)]
    tex = rpf1_table(pd.DataFrame(rows), "leo", "leo", "phys", seeds=3)
    assert tex is not None
    # Row order is classic-then-deep, not best-first: LightGBM precedes CNN despite a worse score.
    order = [ln.split("}")[0].split("{")[-1] for ln in tex.splitlines()
             if ln.startswith("\\textbf{") and " & " in ln and "Model" not in ln]
    assert order == ["LightGBM", "CNN", "LSTM"], order
    # 2 approaches x 3 windows x 3 metrics = 18 data columns, + the model column.
    assert "\\begin{tabular}{l ccc ccc ccc ccc ccc ccc}" in tex, tex[:400]
    body = [ln for ln in tex.splitlines() if ln.startswith("\\textbf{CNN}")]
    assert len(body) == 1 and body[0].count("&") == 18, body
    assert tex.count("\\cmidrule(lr){2-10}") == 1 and tex.count("\\cmidrule(lr){11-19}") == 1
    assert "10 Minute" in tex and "30 Minute" in tex and "100 Minute" in tex
    assert rpf1_table(pd.DataFrame(rows), "geo", "geo", "phys", seeds=3) is None   # absent orbit
    # OOD arm: same layout, distinct filename/label, caption says which way the transfer runs.
    ood = [{**r, "test": "geo"} for r in rows]
    tex_ood = rpf1_table(pd.DataFrame(ood), "leo", "geo", "phys", seeds=3)
    assert tex_ood is not None
    assert "\\begin{tabular}{l ccc ccc ccc ccc ccc ccc}" in tex_ood
    assert "tested out-of-distribution on GEO" in tex_ood
    assert arm_slug("leo", "geo") == "seq_leo_to_geo" and arm_slug("leo", "leo") == "seq_leo"
    assert "tab:seq_leo_to_geo_phys_jc" in tex_ood
    # boxplot_pdf: one page per feature arm, and it must survive an arm with a missing cell.
    import tempfile
    box_rows = [{**r, "feat": f} for r in rows for f in ("eci", "phys")]
    box_rows += [{**r, "test": "geo", "feat": "eci"} for r in rows]
    box_rows = [r for r in box_rows                      # drop a cell: tests the empty-group path
                if not (r["feat"] == "phys" and r["propMin"] == 100 and r["model"] == "CNN")]
    with tempfile.TemporaryDirectory() as td:
        made = boxplot_pdf(pd.DataFrame(box_rows), Path(td), "macro_f1", stem="b")
        assert [q.name for q in made] == ["b_eci.pdf", "b_phys.pdf"], made
        for q in made:
            assert q.stat().st_size > 1000
            raw = q.read_bytes()
            n_pages = raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")
            assert n_pages == 1, (q.name, n_pages)       # one figure per file, not multi-page
        # An all-NaN metric must no-op rather than emit an empty figure.
        nan_rows = [{**r, "macro_f1": float("nan")} for r in box_rows]
        assert boxplot_pdf(pd.DataFrame(nan_rows), Path(td), "macro_f1", stem="c") == []
        assert not (Path(td) / "c_eci.pdf").exists()
    assert feat_label("phys") == "OE + Energy"
    assert feat_label("ladder") == "OE + Acceleration Residual"
    assert feat_label("eci") == "ECI" and feat_label("unknown_arm") == "unknown_arm"
    # _disp keeps the ablation order, not the alphabetical order of the new labels.
    dd = _disp(pd.DataFrame({"feat": ["ladder", "eci", "phys"]}))
    assert list(dd["feat"].cat.categories) == [
        "ECI", "OE + Energy", "OE + Acceleration Residual"], list(dd["feat"].cat.categories)
    # Captions carry the display name; \label{} identifiers keep the raw key so cross-references
    # and filenames stay stable.
    ladder_rows = [{**r, "feat": "ladder"} for r in rows]
    tex_l = rpf1_table(pd.DataFrame(ladder_rows), "leo", "leo", "ladder", seeds=3)
    assert "OE + Acceleration Residual features" in tex_l, tex_l[-400:]
    assert "tab:seq_leo_ladder_jc" in tex_l
    assert "OE + Energy features" in rpf1_table(pd.DataFrame(rows), "leo", "leo", "phys", seeds=3)
    # complexity_table: Joint before Cascade, one time column per window, NaN params -> "--",
    # and the transfer arm (test != train) is excluded.
    cx_rows = [{"train": "leo", "test": te, "feat": "phys", "Model": m, "Approach": ap,
                "propMin": pm, "seed": sd,
                "Params": float("nan") if m == "LightGBM" else 1000.0 * (2 if ap == "Cascade" else 1),
                "Memory_MB": 0.5, "Inference_us_per_frame": 10.0 + sd,
                "Big_O_LaTeX": r"$\mathcal{O}(T)$"}
               for te in ("leo", "geo") for m in ("LightGBM", "CNN") for ap in ("Cascade", "Joint")
               for pm in (10, 30) for sd in (0, 1, 2)]
    ct = complexity_table(pd.DataFrame(cx_rows), "leo", "phys")
    big_o = r"$\mathcal{O}(T)$"
    assert list(ct.index) == [("CNN", big_o, "Joint"), ("CNN", big_o, "Cascade"),
                              ("LightGBM", big_o, "Joint"), ("LightGBM", big_o, "Cascade")], list(ct.index)
    assert list(ct.columns) == ["Params", "Memory (MB)"], list(ct.columns)
    assert ct.loc[("CNN", big_o, "Cascade"), "Params"] == "2{,}000"
    assert ct.loc[("LightGBM", big_o, "Joint"), "Params"] == "--"
    # One Big-O per model group in the rendered table, not one per row.
    assert ct.to_latex(escape=False).count(big_o) == 2
    # Legend lists only the symbols in use, in legend order; p appears only with PCA+MLP.
    lstm, pca = r"$\mathcal{O}(L\,T\,h\,(h+d))$", r"$\mathcal{O}(T\,(W d\,p + p\,h + h\,C))$"
    assert complexity_legend([lstm]) == (r"T\,=\,timesteps, d\,=\,input features, "
                                         r"h\,=\,hidden width, L\,=\,layers"), complexity_legend([lstm])
    assert "PCA" not in complexity_legend([lstm]) and "PCA" in complexity_legend([lstm, pca])
    assert "O\\,=" not in complexity_legend([lstm, pca])
    assert complexity_table(pd.DataFrame(cx_rows), "geo", "phys").empty
    print("selfcheck ok")


if __name__ == "__main__":
    main()
