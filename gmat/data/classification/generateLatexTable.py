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
python generateLatexTable.py --vleo parsed_data/vleo/_group/csv/summary_group.csv --leo parsed_data/leo/_group/csv/summary_group.csv --out-prefix class_vleo_leo

python generateLatexTable.py \
  --csv parsed_data/vleo/_group/csv/summary_group.csv \
  --train-label VLEO \
  --csv parsed_data/geo/_group/csv/summary_group.csv \
  --train-label GEO \
  --out-prefix class_vleo_geo

python generateLatexTable.py \
  --csv parsed_data/vleo/_group/csv/summary_group.csv \
  --train-label VLEO \
  --out-prefix class_vleo

Generated using ChatGPT
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import pandas as pd


# ------------------------- Utilities -------------------------


TIME_SET = (10, 30, 100)


def get_model_order(df: pd.DataFrame) -> List[str]:
    """
    Determine model order from available data.
    Order by descending mean macro_f1; break ties alphabetically for stability.
    """
    if df.empty:
        return []
    score = (
        df.groupby("model_display", as_index=True)["macro_f1"]
        .mean()
        .sort_values(ascending=False)
    )
    return sorted(score.index.tolist(), key=lambda m: (-score[m], m))


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
    for (_, _), block in best_pivot.groupby(level=["train_set", "time_min"], sort=False):
        max_R = block["macro_recall"].max(skipna=True)
        max_P = block["macro_precision"].max(skipna=True)
        max_F = block["macro_f1"].max(skipna=True)
        for (train, t, model), row in block.iterrows():
            r, p, f = row["macro_recall"], row["macro_precision"], row["macro_f1"]
            masks[(train, t, model)] = {
                "R": (pd.notna(r) and abs(r - max_R) < 1e-12),
                "P": (pd.notna(p) and abs(p - max_P) < 1e-12),
                "F": (pd.notna(f) and abs(f - max_F) < 1e-12),
            }
    return masks


def make_latex_table(best: pd.DataFrame, caption: str, label: str, train_sets: List[str]) -> str:
    """
    Horizontal format:
      One block per train set (10/30/100 × R,P,F1).
    Bold = best per train_set × time_min × metric.
    """
    if best.empty:
        raise ValueError("No rows selected for this table.")

    best_pivot = best.set_index(["train_set", "time_min", "model_display"]).sort_index()
    masks = build_bold_masks(best_pivot)
    model_order = get_model_order(best)

    rows: List[str] = []
    for model in model_order:
        cells: List[str] = [f"\\textbf{{{model}}}"]
        for train in train_sets:
            for t in TIME_SET:
                key = (train, t, model)
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

    block_colspec = " ".join(["ccc ccc ccc"] * len(train_sets))
    colspec = f"l {block_colspec}".strip()

    top_headers: List[str] = []
    top_cmidrules: List[str] = []
    mid_headers: List[str] = []
    metric_headers: List[str] = ["\\textbf{Model}"]
    metric_cmidrules: List[str] = []

    for i, train in enumerate(train_sets):
        start = 2 + i * 9
        end = start + 8
        top_headers.append(f"\\multicolumn{{9}}{{c}}{{\\textbf{{{train} Train}}}}")
        top_cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")

        for j, t in enumerate(TIME_SET):
            t_start = start + j * 3
            t_end = t_start + 2
            mid_headers.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{t} Minute}}}}")
            metric_cmidrules.append(f"\\cmidrule(lr){{{t_start}-{t_end}}}")
            metric_headers += ["\\textbf{R}", "\\textbf{P}", "\\textbf{F1}"]

    top_row = " & " + " & ".join(top_headers) + r" \\"
    mid_row = " & " + " & ".join(mid_headers) + r" \\"
    metric_row = " & ".join(metric_headers) + r" \\"

    latex = r"""
\begin{table}[t]
\centering
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.1}

\resizebox{\textwidth}{!}{%
\begin{tabular}{""" + colspec + r"""}
\toprule
""" + top_row + """
""" + "".join(top_cmidrules) + """
""" + mid_row + """
""" + "".join(metric_cmidrules) + """
""" + metric_row + r"""
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
    ap.add_argument("--csv", action="append", type=Path,
                    help="Path to a train-set summary CSV. Repeat to include multiple train sets.")
    ap.add_argument("--train-label", action="append",
                    help="Label for each --csv in the same order (e.g., VLEO, LEO, HEO).")
    ap.add_argument("--vleo", type=Path, help="Legacy: Path to VLEO summary CSV")
    ap.add_argument("--leo",  type=Path, help="Legacy: Path to LEO summary CSV")
    ap.add_argument("--train1", default="VLEO", help="Legacy label for --vleo")
    ap.add_argument("--train2", default="LEO", help="Legacy label for --leo")
    ap.add_argument("--out-prefix", required=True, type=Path, help="Output prefix for .tex files")
    ap.add_argument("--select-by",
                    choices=["macro_f1", "macro_precision", "macro_recall"],
                    default="macro_f1",
                    help="Selection criterion for the best run per (train,time,model). Default: macro_f1")
    args = ap.parse_args()

    # Resolve train-set inputs
    train_paths: List[Path]
    train_labels: List[str]
    if args.csv:
        train_paths = args.csv
        if args.train_label:
            if len(args.train_label) != len(train_paths):
                raise ValueError("When using --csv, provide exactly one --train-label per CSV.")
            train_labels = args.train_label
        else:
            train_labels = [f"TRAIN{i + 1}" for i in range(len(train_paths))]
    else:
        if not (args.vleo and args.leo):
            raise ValueError("Provide either one-or-more --csv arguments, or both --vleo and --leo.")
        train_paths = [args.vleo, args.leo]
        train_labels = [args.train1, args.train2]

    # Load and prepare all train sets
    prepared_frames: List[pd.DataFrame] = []
    for path, label in zip(train_paths, train_labels):
        prepared_frames.append(prepare_frame(pd.read_csv(path), label))

    # Split into OE vs Cartesian by log name token
    oe_frames = [df[df["is_oe"] == True] for df in prepared_frames]
    cart_frames = [df[df["is_oe"] == False] for df in prepared_frames]

    train_set_text = (
        train_labels[0]
        if len(train_labels) == 1
        else ", ".join(train_labels[:-1]) + f" and {train_labels[-1]}"
    )

    # Build Cartesian table
    df_cart = pd.concat(cart_frames, ignore_index=True)
    best_cart = select_best(df_cart, criterion=args.select_by) if not df_cart.empty else df_cart
    if best_cart.empty:
        cart_tex = "% No Cartesian rows found after filtering; check your CSVs."
    else:
        cart_tex = make_latex_table(
            best_cart,
            caption=(f"Summary of Cartesian macro Precision (P), Recall (R), and F1 for {train_set_text} "
                     "train sets at 10/30/100 minutes. Bold = best per train/time/metric."),
            label=f"tab:{args.out_prefix}_cart",
            train_sets=train_labels,
        )

    # Build OE table
    df_oe = pd.concat(oe_frames, ignore_index=True)
    best_oe = select_best(df_oe, criterion=args.select_by) if not df_oe.empty else df_oe
    if best_oe.empty:
        oe_tex = "% No OE rows found after filtering; check your CSVs."
    else:
        oe_tex = make_latex_table(
            best_oe,
            caption=(f"Summary of OE macro Precision (P), Recall (R), and F1 for {train_set_text} "
                     "train sets at 10/30/100 minutes. Bold = best per train/time/metric."),
            label=f"tab:{args.out_prefix}_oe",
            train_sets=train_labels,
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
