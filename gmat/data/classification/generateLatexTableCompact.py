#!/usr/bin/env python3
"""
Build compact LaTeX classification-results tables with improvements over generateLatexTable.py:

  1. F1-only output by default — cuts 18-column tables down to 6 columns.
     Use --metrics rpf1 to restore Recall / Precision / F1.
  2. Cell background color gradient (red→green) via xcolor.
     Requires \\usepackage[table]{xcolor} in the LaTeX preamble.
     Disable with --no-color.
  3. 2 decimal-place precision (override with --decimals N).
  4. Cartesian results replaced with a compact LaTeX paragraph by default
     (add --include-cart to also emit the full Cartesian table).
  5. OE table written first; rows sorted by descending mean F1.

Outputs (mirroring generateLatexTable.py):
  <out_prefix>_oe.tex   — OE table only
  <out_prefix>_cart.tex — Cartesian table or summary paragraph
  <out_prefix>_both.tex — OE + cart/note concatenated

Usage: from /gmat/data/classification

  # Single train set, F1-only + color (default)
  python generateLatexTableCompact.py \
      --csv parsed_data/vleo/_group/csv/summary_group.csv --train-label VLEO \
      --out-prefix class_vleo

  # Two train sets
  python generateLatexTableCompact.py \
      --csv parsed_data/vleo/_group/csv/summary_group.csv --train-label VLEO \
      --csv parsed_data/leo/_group/csv/summary_group.csv  --train-label LEO \
      --out-prefix class_vleo_leo --metrics rpf1

  # Cartesian vs OE side-by-side from a single CSV (--combine-features)
  python generateLatexTableCompact.py \
      --csv parsed_data/vleo/_group/csv/summary_group.csv --train-label VLEO \
      --out-prefix class_vleo --combine-features

  # combine-features with multiple CSVs produces blocks: VLEO Cart | VLEO OE | LEO Cart | LEO OE
  python generateLatexTableCompact.py \
      --csv parsed_data/vleo/_group/csv/summary_group.csv --train-label VLEO \
      --csv parsed_data/leo/_group/csv/summary_group.csv  --train-label LEO \
      --out-prefix class_vleo_leo --combine-features

  # All metrics, no color, full Cartesian table
  python generateLatexTableCompact.py \
      --csv parsed_data/vleo/_group/csv/summary_group.csv --train-label VLEO \
      --csv parsed_data/combined/leo-meo-geo/_group/csv/summary_group.csv --train-label LEO-MEO-GEO \
      --out-prefix class_vleo_combined --metrics rpf1 --no-color --include-cart

  # Legacy two-CSV style (unchanged from generateLatexTable.py)
  python generateLatexTableCompact.py \
      --vleo parsed_data/vleo/_group/csv/summary_group.csv \
      --leo  parsed_data/leo/_group/csv/summary_group.csv  \
      --out-prefix class_vleo_leo_legacy

  # GEO train set only
  python generateLatexTableCompact.py \\
      --csv parsed_data/geo/_group/csv/summary_group.csv --train-label GEO \\
      --out-prefix class_geo

Generated using Claude
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import pandas as pd


# ─── Constants ────────────────────────────────────────────────────────────────

TIME_SET = (10, 30, 100)

# Metric column names (internal) and their LaTeX header strings
METRICS_F1  : Tuple[str, ...] = ("macro_f1",)
METRICS_ALL : Tuple[str, ...] = ("macro_recall", "macro_precision", "macro_f1")

METRIC_HEADERS: Dict[str, str] = {
    "macro_recall":    r"\textbf{R}",
    "macro_precision": r"\textbf{P}",
    "macro_f1":        r"\textbf{F1}",
}


# ─── Colour helpers ───────────────────────────────────────────────────────────

def _lerp_int(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def val_to_color(val: float, lo: float = 0.25, hi: float = 0.75) -> str:
    """
    Map a float in [lo, hi] to an HTML hex colour string.
    Below lo  → coral-red  (#FF6B6B)
    At midpoint → amber   (#FFD43B)
    At hi      → green    (#51CF66)
    """
    if pd.isna(val):
        return "FFFFFF"
    t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    # Colour stops: red(0) → amber(0.5) → green(1)
    # Red:   (255, 107, 107)
    # Amber: (255, 212,  59)
    # Green: ( 81, 207, 102)
    if t <= 0.5:
        s = t * 2.0
        r = _lerp_int(255, 255, s)
        g = _lerp_int(107, 212, s)
        b = _lerp_int(107,  59, s)
    else:
        s = (t - 0.5) * 2.0
        r = _lerp_int(255,  81, s)
        g = _lerp_int(212, 207, s)
        b = _lerp_int( 59, 102, s)
    return f"{r:02X}{g:02X}{b:02X}"


# ─── DataFrame helpers ────────────────────────────────────────────────────────

def get_model_order(df: pd.DataFrame) -> List[str]:
    """Return models sorted by descending mean macro_f1; ties broken alphabetically."""
    if df.empty:
        return []
    score = (
        df.groupby("model_display", as_index=True)["macro_f1"]
        .mean()
        .sort_values(ascending=False)
    )
    return sorted(score.index.tolist(), key=lambda m: (-score[m], m))


def extract_minutes(stem: str) -> Optional[int]:
    if not isinstance(stem, str):
        return None
    m = re.search(r"(\d+)\s*min", stem, flags=re.I)
    return int(m.group(1)) if m else None


def is_oe_log(stem: str) -> bool:
    if not isinstance(stem, str):
        return False
    return re.search(r"(^|[^A-Za-z])OE([^A-Za-z]|$)", stem, flags=re.I) is not None


def normalize_models(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {"Decision Trees": "DT", "Mamba": "S4"}
    g = df.copy()
    g["model_display"] = g["model"].map(mapping).fillna(g["model"])
    return g


def coerce_float_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    g = df.copy()
    for c in cols:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    return g


def prepare_frame(df: pd.DataFrame, train_set: str) -> pd.DataFrame:
    g = df.copy()
    g["time_min"] = g["log_stem"].apply(extract_minutes)
    g = normalize_models(g)
    g["is_oe"] = g["log_stem"].apply(is_oe_log)
    needed = ["macro_avg_precision", "macro_avg_recall", "macro_avg_f1"]
    g = coerce_float_cols(g, needed)
    keep = ["model_display", "time_min", "is_oe"] + needed
    for k in keep:
        if k not in g.columns:
            raise ValueError(f"Missing expected column in CSV: {k!r}")
    g = g[keep].dropna(subset=["time_min"])
    g["time_min"] = g["time_min"].astype(int)
    g = g[g["time_min"].isin(TIME_SET)]
    g["train_set"] = train_set
    g = g.rename(columns={
        "macro_avg_precision": "macro_precision",
        "macro_avg_recall":    "macro_recall",
        "macro_avg_f1":        "macro_f1",
    })
    return g


def select_best(df: pd.DataFrame, criterion: str) -> pd.DataFrame:
    """For each (train_set, time_min, model_display) keep the row with highest criterion."""
    if df.empty:
        return df
    idx = df.groupby(["train_set", "time_min", "model_display"])[criterion].idxmax()
    return df.loc[idx].reset_index(drop=True)


# ─── Cell formatting ──────────────────────────────────────────────────────────

def format_cell(
    val: float,
    is_bold: bool,
    use_color: bool,
    decimals: int,
    color_lo: float,
    color_hi: float,
) -> str:
    if pd.isna(val):
        return "---"
    num_str = f"{val:.{decimals}f}"
    inner = f"\\textbf{{{num_str}}}" if is_bold else num_str
    if use_color:
        hex_col = val_to_color(val, lo=color_lo, hi=color_hi)
        return f"\\cellcolor[HTML]{{{hex_col}}}{inner}"
    return inner


# ─── Bold-mask builder ────────────────────────────────────────────────────────

def build_bold_masks(
    best_pivot: pd.DataFrame,
    metrics: Tuple[str, ...],
) -> Dict[Tuple[str, int, str], Dict[str, bool]]:
    """
    For each (train_set, time_min) block, mark which cells hold the column maximum.
    Bold = best per train × time × metric.
    """
    masks: Dict[Tuple[str, int, str], Dict[str, bool]] = {}
    for (_, _), block in best_pivot.groupby(level=["train_set", "time_min"], sort=False):
        maxes = {col: block[col].max(skipna=True) for col in metrics}
        for (train, t, model), row in block.iterrows():
            masks[(train, t, model)] = {
                col: (pd.notna(row[col]) and abs(row[col] - maxes[col]) < 1e-12)
                for col in metrics
            }
    return masks


# ─── Table builder ────────────────────────────────────────────────────────────

def make_latex_table(
    best: pd.DataFrame,
    caption: str,
    label: str,
    train_sets: List[str],
    metrics: Tuple[str, ...],
    decimals: int,
    use_color: bool,
    color_lo: float,
    color_hi: float,
    block_suffix: str = " Train",
) -> str:
    if best.empty:
        raise ValueError("No rows selected for this table.")

    n_metrics     = len(metrics)        # 1 (F1-only) or 3 (R/P/F1)
    cols_per_train = 3 * n_metrics      # one group per time window

    best_pivot = best.set_index(["train_set", "time_min", "model_display"]).sort_index()
    masks      = build_bold_masks(best_pivot, metrics)
    model_order = get_model_order(best)

    # ── Body rows ──────────────────────────────────────────────────────────
    rows: List[str] = []
    for model in model_order:
        cells: List[str] = [f"\\textbf{{{model}}}"]
        for train in train_sets:
            for t in TIME_SET:
                key = (train, t, model)
                if key in best_pivot.index:
                    row = best_pivot.loc[key]
                    m   = masks.get(key, {col: False for col in metrics})
                    for col in metrics:
                        cells.append(
                            format_cell(row[col], m[col], use_color, decimals, color_lo, color_hi)
                        )
                else:
                    cells += ["---"] * n_metrics
        rows.append(" & ".join(cells) + r" \\")
    body = "\n".join(rows)

    # ── Column spec ────────────────────────────────────────────────────────
    # Within each train block: 3 time windows, each n_metrics wide.
    # Preserve the original visual spacing between time groups when n_metrics > 1.
    time_group = "c" * n_metrics
    block_colspec = " ".join([time_group] * 3)        # e.g. "c c c" or "ccc ccc ccc"
    colspec = "l " + " ".join([block_colspec] * len(train_sets))

    # ── Header rows ────────────────────────────────────────────────────────
    top_headers:   List[str] = []
    top_cmidrules: List[str] = []
    mid_headers:   List[str] = []
    mid_cmidrules: List[str] = []
    metric_headers: List[str] = [r"\textbf{Model}"]

    for i, train in enumerate(train_sets):
        col_start = 2 + i * cols_per_train
        col_end   = col_start + cols_per_train - 1
        top_headers.append(
            f"\\multicolumn{{{cols_per_train}}}{{c}}{{\\textbf{{{train}{block_suffix}}}}}"
        )
        top_cmidrules.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")

        for j, t in enumerate(TIME_SET):
            t_start = col_start + j * n_metrics
            t_end   = t_start + n_metrics - 1
            mid_headers.append(
                f"\\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{{t} Minute}}}}"
            )
            mid_cmidrules.append(f"\\cmidrule(lr){{{t_start}-{t_end}}}")
            for col in metrics:
                metric_headers.append(METRIC_HEADERS[col])

    top_row    = " & " + " & ".join(top_headers)  + r" \\"
    mid_row    = " & " + " & ".join(mid_headers)   + r" \\"
    metric_row = " & ".join(metric_headers)         + r" \\"

    # ── Assemble LaTeX ─────────────────────────────────────────────────────
    color_note = (
        "% Requires \\usepackage[table]{xcolor} in LaTeX preamble.\n"
        if use_color else ""
    )

    return (
        color_note
        + "\\begin{table}[t]\n"
        + "\\centering\n"
        + "\\setlength{\\tabcolsep}{3pt}\n"
        + "\\renewcommand{\\arraystretch}{1.1}\n\n"
        + "\\resizebox{\\textwidth}{!}{%\n"
        + "\\begin{tabular}{" + colspec + "}\n"
        + "\\toprule\n"
        + top_row + "\n"
        + "".join(top_cmidrules) + "\n"
        + mid_row + "\n"
        + "".join(mid_cmidrules) + "\n"
        + metric_row + "\n"
        + "\\midrule\n"
        + body + "\n"
        + "\\bottomrule\n"
        + "\\end{tabular}%\n"
        + "}\n"
        + "\\caption{" + caption + "}\n"
        + "\\label{" + label + "}\n"
        + "\\end{table}"
    )


# ─── Cartesian summary paragraph ─────────────────────────────────────────────

def make_cart_note(best_cart: pd.DataFrame, train_labels: List[str]) -> str:
    """
    Emit a brief LaTeX \\paragraph{} summarising the near-random Cartesian results
    so the paper can reference them without a full table.
    """
    if best_cart.empty:
        return "% No Cartesian data found after filtering."

    mean_f1 = best_cart["macro_f1"].mean()
    best_row = best_cart.loc[best_cart["macro_f1"].idxmax()]
    best_model = best_row["model_display"]
    best_train = best_row["train_set"]
    best_time  = int(best_row["time_min"])
    max_f1     = best_row["macro_f1"]

    train_text = (
        train_labels[0] if len(train_labels) == 1
        else ", ".join(train_labels[:-1]) + f" and {train_labels[-1]}"
    )

    return (
        "% ── Cartesian results: full table replaced with summary paragraph ──────────────\n"
        "% All Cartesian models clustered near the 4-class random baseline (~0.25).\n"
        "% Use --include-cart to emit the full table instead.\n\n"
        "\\paragraph{Cartesian features}"
        f" Cartesian features yielded near-random performance"
        f" (mean macro-F1 $\\approx {mean_f1:.2f}$) across all models,"
        f" {train_text} train sets, and observation windows (10/30/100\\,min)."
        f" The best individual result was {best_model} on the {best_train} train"
        f" set at {best_time}\\,min (macro-F1~$= {max_f1:.2f}$),"
        f" only marginally above the four-class random baseline ($\\approx 0.25$)."
        " Full Cartesian results are provided in the supplementary material."
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compact LaTeX classification-results tables (F1-only + cell color).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input (same style as generateLatexTable.py) ────────────────────────
    ap.add_argument("--csv", action="append", type=Path,
                    help="Path to a train-set summary CSV. Repeat for multiple train sets.")
    ap.add_argument("--train-label", action="append",
                    help="Label for each --csv in order (e.g. VLEO, LEO, GEO).")
    ap.add_argument("--vleo", type=Path, help="Legacy: VLEO summary CSV")
    ap.add_argument("--leo",  type=Path, help="Legacy: LEO summary CSV")
    ap.add_argument("--train1", default="VLEO", help="Legacy label for --vleo")
    ap.add_argument("--train2", default="LEO",  help="Legacy label for --leo")
    ap.add_argument("--out-prefix", required=True, type=Path,
                    help="Output file prefix (no extension)")
    ap.add_argument("--select-by",
                    choices=["macro_f1", "macro_precision", "macro_recall"],
                    default="macro_f1",
                    help="Selection criterion for best run per (train, time, model). Default: macro_f1")

    # ── New options ────────────────────────────────────────────────────────
    ap.add_argument("--metrics", choices=["f1", "rpf1"], default="f1",
                    help="Metrics per time window: 'f1' = F1 only (default), "
                         "'rpf1' = Recall + Precision + F1")
    ap.add_argument("--decimals", type=int, default=2,
                    help="Decimal places in table cells (default: 2)")
    ap.add_argument("--no-color", action="store_true",
                    help="Disable cell background color gradient")
    ap.add_argument("--include-cart", action="store_true",
                    help="Include the full Cartesian table; default is a summary paragraph only")
    ap.add_argument("--color-lo", type=float, default=0.25,
                    help="Value mapped to full red (default: 0.25 = random baseline)")
    ap.add_argument("--color-hi", type=float, default=0.75,
                    help="Value mapped to full green (default: 0.75)")
    ap.add_argument("--oe-only", action="store_true",
                    help="Only generate the OE table; skip Cartesian results entirely")
    ap.add_argument("--combine-features", action="store_true",
                    help="Put Cartesian and OE side-by-side in one table instead of "
                         "separate tables. For a single CSV the blocks are labelled "
                         "'Cartesian' and 'OE'; for multiple CSVs they become "
                         "'{label} Cart' and '{label} OE'.")
    args = ap.parse_args()

    # ── Resolve train-set inputs ───────────────────────────────────────────
    if args.csv:
        train_paths = args.csv
        if args.train_label:
            if len(args.train_label) != len(train_paths):
                raise ValueError("Provide exactly one --train-label per --csv.")
            train_labels = args.train_label
        else:
            train_labels = [f"TRAIN{i + 1}" for i in range(len(train_paths))]
    else:
        if not (args.vleo and args.leo):
            raise ValueError("Provide --csv / --train-label pairs, or both --vleo and --leo.")
        train_paths  = [args.vleo, args.leo]
        train_labels = [args.train1, args.train2]

    # ── Load and split data ────────────────────────────────────────────────
    frames      = [prepare_frame(pd.read_csv(p), lbl)
                   for p, lbl in zip(train_paths, train_labels)]
    oe_frames   = [df[df["is_oe"]]  for df in frames]
    cart_frames = [df[~df["is_oe"]] for df in frames]

    metrics   = METRICS_ALL if args.metrics == "rpf1" else METRICS_F1
    use_color = not args.no_color

    train_text = (
        train_labels[0] if len(train_labels) == 1
        else ", ".join(train_labels[:-1]) + f" and {train_labels[-1]}"
    )
    metric_desc = {
        METRICS_F1:  "macro F1",
        METRICS_ALL: "macro Precision (P), Recall (R), and F1",
    }[metrics]

    color_caption_note = " Cell color: red\\,=\\,low, green\\,=\\,high." if use_color else ""
    bold_note = "Bold = best per train/time" + ("/metric." if len(metrics) > 1 else ".")

    stem = str(args.out_prefix.with_suffix(""))

    # ── combine-features: Cartesian | OE blocks in one table ──────────────
    if args.combine_features:
        single = len(train_labels) == 1
        combined_blocks: List[pd.DataFrame] = []
        block_labels:    List[str]          = []

        for frame, label in zip(frames, train_labels):
            cart_label = "Cartesian"       if single else f"{label} Cart"
            oe_label   = "OE"              if single else f"{label} OE"
            cart_frame = frame[~frame["is_oe"]].copy()
            oe_frame   = frame[ frame["is_oe"]].copy()
            cart_frame["train_set"] = cart_label
            oe_frame["train_set"]   = oe_label
            combined_blocks += [cart_frame, oe_frame]
            block_labels    += [cart_label, oe_label]

        df_combined   = pd.concat(combined_blocks, ignore_index=True)
        best_combined = select_best(df_combined, args.select_by)

        feat_cap_train = f"{train_text} train set{'s' if len(train_labels) > 1 else ''}"
        feat_tex = make_latex_table(
            best_combined,
            caption=(
                f"Summary of Cartesian and OE {metric_desc} for {feat_cap_train} "
                f"at 10/30/100 minutes. {bold_note}{color_caption_note}"
            ),
            label=f"tab:{args.out_prefix.name}_feat",
            train_sets=block_labels,
            metrics=metrics,
            decimals=args.decimals,
            use_color=use_color,
            color_lo=args.color_lo,
            color_hi=args.color_hi,
            block_suffix="",
        )

        feat_path = Path(stem + "_feat.tex")
        both_path = Path(stem + "_both.tex")
        feat_path.write_text(feat_tex + "\n", encoding="utf-8")
        both_path.write_text(feat_tex + "\n", encoding="utf-8")
        print(f"Wrote feature-combined -> {feat_path.resolve()}")
        print(f"Wrote both             -> {both_path.resolve()}")
        return

    # ── OE table ───────────────────────────────────────────────────────────
    df_oe   = pd.concat(oe_frames, ignore_index=True)
    best_oe = select_best(df_oe, args.select_by) if not df_oe.empty else df_oe

    if best_oe.empty:
        oe_tex = "% No OE rows found after filtering; check your CSVs."
    else:
        oe_tex = make_latex_table(
            best_oe,
            caption=(
                f"Summary of OE {metric_desc} for {train_text} train sets "
                f"at 10/30/100 minutes. {bold_note}{color_caption_note}"
            ),
            label=f"tab:{args.out_prefix.name}_oe",
            train_sets=train_labels,
            metrics=metrics,
            decimals=args.decimals,
            use_color=use_color,
            color_lo=args.color_lo,
            color_hi=args.color_hi,
        )

    # ── Cartesian: full table or summary paragraph ─────────────────────────
    df_cart   = pd.concat(cart_frames, ignore_index=True)
    best_cart = select_best(df_cart, args.select_by) if not df_cart.empty else df_cart

    if args.include_cart and not best_cart.empty:
        cart_tex = make_latex_table(
            best_cart,
            caption=(
                f"Summary of Cartesian {metric_desc} for {train_text} train sets "
                f"at 10/30/100 minutes. {bold_note}{color_caption_note}"
            ),
            label=f"tab:{args.out_prefix.name}_cart",
            train_sets=train_labels,
            metrics=metrics,
            decimals=args.decimals,
            use_color=use_color,
            color_lo=args.color_lo,
            color_hi=args.color_hi,
        )
    else:
        cart_tex = make_cart_note(best_cart, train_labels)

    # ── Write outputs ──────────────────────────────────────────────────────
    oe_path = Path(stem + "_oe.tex")
    oe_path.write_text(oe_tex + "\n", encoding="utf-8")
    print(f"Wrote OE table   -> {oe_path.resolve()}")

    if not args.oe_only:
        cart_path = Path(stem + "_cart.tex")
        both_path = Path(stem + "_both.tex")
        cart_path.write_text(cart_tex + "\n", encoding="utf-8")
        both_path.write_text(oe_tex + "\n\n" + cart_tex + "\n", encoding="utf-8")
        print(f"Wrote cart/note  -> {cart_path.resolve()}")
        print(f"Wrote both       -> {both_path.resolve()}")


if __name__ == "__main__":
    main()
