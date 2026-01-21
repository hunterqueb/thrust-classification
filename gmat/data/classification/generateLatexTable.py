#!/usr/bin/env python3
"""
Build TWO LaTeX tables (OE + Cartesian) in ONE run from your summary CSVs.

Assumptions:
- Each CSV row has a log name in `log_stem` that indicates whether metrics are
  OE or Cartesian. We classify rows as OE if the log_stem contains the token "OE"
  (case-insensitive). Everything else is treated as Cartesian.
- Macro metrics are in columns:
    macro_avg_precision, macro_avg_recall, macro_avg_f1
- Times are encoded in the log name (e.g., '10min', '30min', '100min').
- Train sets are split by using two CSVs: one for VLEO, one for LEO.

Output:
- <out_prefix>_oe.tex   : LaTeX table for OE metrics
- <out_prefix>_cart.tex : LaTeX table for Cartesian metrics
- <out_prefix>_both.tex : Both tables concatenated (OE first, then Cartesian)

Selection rule:
- For each (train_set, time_min, model) pick the row with max `macro_avg_f1`
  (configurable with --select-by {macro_f1|macro_precision|macro_recall}).

Model normalization:
- "Decision Trees" -> "DT", "Mamba" -> "S4", "LSTM" -> "LSTM"

Usage: from /gmat/data/classification
python generateLatexTable.py --vleo parsed_data/vleo/_group/csv/summary_vleo-low.csv --leo parsed_data/leo/_group/csv/summary_leo-low.csv --out-prefix class_sum_macro_low
python generateLatexTable.py --vleo parsed_data/vleo/_group/csv/summary_vleo-high.csv --leo parsed_data/leo/_group/csv/summary_leo-high.csv --out-prefix class_sum_macro_high

"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import pandas as pd


# ------------------------- Utilities -------------------------

TIME_SET = (10, 30, 100)
MODEL_ORDER = ("DT", "LSTM", "S4")


def extract_minutes(stem: str) -> Optional[int]:
    """Extract integer minutes from strings like '10min...' or '100 min ...'."""
    if not isinstance(stem, str):
        return None
    m = re.search(r'(\d+)\s*min', stem, flags=re.I)
    return int(m.group(1)) if m else None


def is_oe_log(stem: str) -> bool:
    """
    Classify OE vs Cartesian from log_stem using token 'OE' (case-insensitive).
    Example OE names: '...OE_Norm_Noise...', '..._OE_...'
    """
    if not isinstance(stem, str):
        return False
    # Token match: non-letter boundaries around 'OE' to reduce false positives.
    return re.search(r'(^|[^A-Za-z])OE([^A-Za-z]|$)', stem, flags=re.I) is not None


def normalize_models(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Decision Trees": "DT",
        "Mamba": "S4",
        "LSTM": "LSTM",
    }
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
    """
    Add time_min, model_display, is_oe; coerce macro columns; filter time.
    """
    g = df.copy()
    g["time_min"] = g["log_stem"].apply(extract_minutes)
    g = normalize_models(g)
    g["is_oe"] = g["log_stem"].apply(is_oe_log)

    needed = ["macro_avg_precision", "macro_avg_recall", "macro_avg_f1"]
    g = coerce_float_cols(g, needed)

    keep = ["model_display", "time_min", "is_oe"] + needed
    for k in keep:
        if k not in g.columns:
            raise ValueError(f"Missing expected column in CSV: {k}")

    g = g[keep].dropna(subset=["time_min"])
    g["time_min"] = g["time_min"].astype(int)
    g = g[g["time_min"].isin(TIME_SET)]
    g["train_set"] = train_set
    # Standardize names for downstream
    g = g.rename(columns={
        "macro_avg_precision": "macro_precision",
        "macro_avg_recall": "macro_recall",
        "macro_avg_f1": "macro_f1",
    })
    return g


def select_best(df: pd.DataFrame, criterion: str) -> pd.DataFrame:
    """
    For each (train_set, time_min, model_display), select row with highest criterion.
    """
    if df.empty:
        return df
    idx = df.groupby(["train_set", "time_min", "model_display"])[criterion].idxmax()
    return df.loc[idx].reset_index(drop=True)


def format_cell(val: float, is_bold: bool) -> str:
    if pd.isna(val):
        return "---"
    s = f"{val:.4f}"
    return f"\\textbf{{{s}}}" if is_bold else s


def build_bold_masks(best_pivot: pd.DataFrame) -> Dict[Tuple[str, int, str], Dict[str, bool]]:
    """
    For each train_set and time_min, compute max for R, P, F across models and
    mark which cells should be bold.
    """
    masks: Dict[Tuple[str, int, str], Dict[str, bool]] = {}
    # Defensive: determine available (train, time) pairs dynamically
    available_pairs = set(best_pivot.index.droplevel("model_display").unique())
    for train in ("VLEO", "LEO"):
        for t in TIME_SET:
            if (train, t) not in available_pairs:
                continue
            block = best_pivot.xs((train, t), level=("train_set", "time_min"), drop_level=False)
            max_R = block["macro_recall"].max(skipna=True)
            max_P = block["macro_precision"].max(skipna=True)
            max_F = block["macro_f1"].max(skipna=True)
            for (_, _, model) in block.index:
                row = block.loc[(train, t, model)]
                r, p, f = row["macro_recall"], row["macro_precision"], row["macro_f1"]
                masks[(train, t, model)] = {
                    "R": (pd.notna(r) and abs(r - max_R) < 1e-12),
                    "P": (pd.notna(p) and abs(p - max_P) < 1e-12),
                    "F": (pd.notna(f) and abs(f - max_F) < 1e-12),
                }
    return masks


def make_latex_table(best: pd.DataFrame, caption: str, label: str) -> str:
    """
    Horizontal format:
      VLEO block (10/30/100 × R,P,F1) | LEO block (10/30/100 × R,P,F1)
    Bold = best per train_set × time_min × metric.
    """
    if best.empty:
        raise ValueError("No rows selected for this table.")

    best_pivot = best.set_index(["train_set", "time_min", "model_display"]).sort_index()
    masks = build_bold_masks(best_pivot)

    rows: List[str] = []
    for model in MODEL_ORDER:
        cells: List[str] = [f"\\textbf{{{model}}}"]
        # VLEO side
        for t in TIME_SET:
            key = ("VLEO", t, model)
            if key in best_pivot.index:
                r = best_pivot.loc[key, "macro_recall"]
                p = best_pivot.loc[key, "macro_precision"]
                f = best_pivot.loc[key, "macro_f1"]
                m = masks.get(key, {"R": False, "P": False, "F": False})
                cells += [format_cell(r, m["R"]), format_cell(p, m["P"]), format_cell(f, m["F"])]
            else:
                cells += ["---", "---", "---"]
        cells.append(" ")
        # LEO side
        for t in TIME_SET:
            key = ("LEO", t, model)
            if key in best_pivot.index:
                r = best_pivot.loc[key, "macro_recall"]
                p = best_pivot.loc[key, "macro_precision"]
                f = best_pivot.loc[key, "macro_f1"]
                m = masks.get(key, {"R": False, "P": False, "F": False})
                cells += [format_cell(r, m["R"]), format_cell(p, m["P"]), format_cell(f, m["F"])]
            else:
                cells += ["---", "---", "---"]
        rows.append(" & ".join(cells) + " \\\\")

    body = "\n".join(rows)

    latex = r"""
\begin{table}[t]
\centering
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.1}

\resizebox{\textwidth}{!}{%
\begin{tabular}{lccc ccc ccc l ccc ccc ccc}
\toprule
\multicolumn{10}{c}{\textbf{VLEO Train}} &
\multicolumn{10}{c}{\textbf{LEO Train}} \\
\cmidrule(lr){1-10}\cmidrule(lr){11-20}
 & \multicolumn{3}{c}{\textbf{10 Minute}} & \multicolumn{3}{c}{\textbf{30 Minute}} & \multicolumn{3}{c}{\textbf{100 Minute}} &
 & \multicolumn{3}{c}{\textbf{10 Minute}} & \multicolumn{3}{c}{\textbf{30 Minute}} & \multicolumn{3}{c}{\textbf{100 Minute}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}%
\cmidrule(lr){12-14}\cmidrule(lr){15-17}\cmidrule(lr){18-20}
\textbf{Model} & \textbf{R} & \textbf{P} & \textbf{F1} & \textbf{R} & \textbf{P} & \textbf{F1} & \textbf{R} & \textbf{P} & \textbf{F1} &
 & \textbf{R} & \textbf{P} & \textbf{F1} & \textbf{R} & \textbf{P} & \textbf{F1} & \textbf{R} & \textbf{P} & \textbf{F1} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}%
}
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\end{table}
"""
    return latex.strip()

# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vleo", required=True, type=Path, help="Path to summary_vleo-high.csv")
    ap.add_argument("--leo",  required=True, type=Path, help="Path to summary_leo-high.csv")
    ap.add_argument("--out-prefix", required=True, type=Path, help="Output prefix for .tex files")
    ap.add_argument("--select-by",
                    choices=["macro_f1", "macro_precision", "macro_recall"],
                    default="macro_f1",
                    help="Selection criterion for the best run per (train,time,model). Default: macro_f1")
    args = ap.parse_args()

    # Load CSVs
    vleo_raw = pd.read_csv(args.vleo)
    leo_raw  = pd.read_csv(args.leo)

    # Prepare frames (adds time_min, model_display, is_oe, macro_* standardized)
    vleo_all = prepare_frame(vleo_raw, "VLEO")
    leo_all  = prepare_frame(leo_raw,  "LEO")

    # Split into OE vs Cartesian by log name token
    vleo_oe   = vleo_all[vleo_all["is_oe"] == True]
    vleo_cart = vleo_all[vleo_all["is_oe"] == False]
    leo_oe    = leo_all[leo_all["is_oe"] == True]
    leo_cart  = leo_all[leo_all["is_oe"] == False]


    # Build Cartesian table
    df_cart = pd.concat([vleo_cart, leo_cart], ignore_index=True)
    best_cart = select_best(df_cart, criterion=args.select_by) if not df_cart.empty else df_cart
    if best_cart.empty:
        cart_tex = "% No Cartesian rows found after filtering; check your CSVs."
    else:
        cart_tex = make_latex_table(
            best_cart,
            caption=("Summary of Cartesian macro Precision (P), Recall (R), and F1 for VLEO and LEO "
                     "train sets at 10/30/100 minutes. Bold = best per train/time/metric."),
            label=f"tab:{args.out_prefix}_cart",
        )

    # Build OE table
    df_oe = pd.concat([vleo_oe, leo_oe], ignore_index=True)
    best_oe = select_best(df_oe, criterion=args.select_by) if not df_oe.empty else df_oe
    if best_oe.empty:
        oe_tex = "% No OE rows found after filtering; check your CSVs."
    else:
        oe_tex = make_latex_table(
            best_oe,
            caption=("Summary of OE macro Precision (P), Recall (R), and F1 for VLEO and LEO "
                     "train sets at 10/30/100 minutes. Bold = best per train/time/metric."),
            label=f"tab:{args.out_prefix}_oe",
        )

    # Write outputs
    out_oe   = args.out_prefix.with_suffix("")  # ensure we can add suffixes
    out_cart = args.out_prefix.with_suffix("")
    out_both = args.out_prefix.with_suffix("")

    oe_path   = Path(str(out_oe)   + "_oe.tex")
    cart_path = Path(str(out_cart) + "_cart.tex")
    both_path = Path(str(out_both) + "_both.tex")

    # oe_path.write_text(oe_tex + "\n", encoding="utf-8")
    # cart_path.write_text(cart_tex + "\n", encoding="utf-8")
    both_path.write_text(cart_tex + "\n\n" + oe_tex + "\n", encoding="utf-8")

    # print(f"Wrote OE table   -> {oe_path.resolve()}")
    # print(f"Wrote CART table -> {cart_path.resolve()}")
    print(f"Wrote BOTH file  -> {both_path.resolve()}")


if __name__ == "__main__":
    main()
