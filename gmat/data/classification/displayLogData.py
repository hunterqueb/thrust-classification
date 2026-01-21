from __future__ import annotations

"""Smart training-log parser (v4.0)
• Adds a Validation-Loss plot over epochs (one line per model).
• Extends epoch parsing to capture Training Loss and Validation Loss.
• Keeps existing outputs/back-compat; epochs CSV now includes loss columns.
• NEW: --group-dir mode to aggregate a single directory of logs into ONE summary CSV.



Usage:
python displayLogData.py . --force
python displayLogData.py . --group-dir vleo/ --group-name vleo-low
python displayLogData.py . --group-dir vleo/ --group-name vleo-low --emit-per-log
"""

from pathlib import Path
import argparse
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ───── Regexes ─────
RE_MODEL_SPLIT = re.compile(r"Entering\s+(.+?)\s+Training Loop", re.S)
RE_EPOCH_LINE  = re.compile(r"^Epoch\s*\[(\d+)/(\d+)\]")
RE_TRAIN_LOSS  = re.compile(r"Training Loss:\s*([\d.]+)")
RE_VAL_ACC     = re.compile(r"Validation Accuracy:\s*([\d.]+)%")
RE_VAL_LOSS    = re.compile(r"Validation Loss:\s*([\d.]+)")
RE_PARAMS      = re.compile(r"Total parameters:\s*(\d+)")
RE_MEMORY      = re.compile(r"Total memory \(MB\):\s*([\d.]+)")
RE_TIME        = re.compile(r"Elapsed time is\s*([\d.]+)\s*seconds")

RE_CLASS_REPORT = re.compile(
    r"Classification Report:\s*[\r\n]+(.*?)(?=\n(?:Epoch|Per-Class|Entering|Early stopping|=|$))",
    re.S,
)
RE_CONF_MATRIX = re.compile(
    r"Confusion Matrix.*?:\s*[\r\n]+(.*?)(?=\n(?:Classification Report:|Epoch|Per-Class|Entering|Early stopping|=|$))",
    re.S | re.I,
)

@dataclass
class Summary:
    model: str
    max_val_accuracy: float
    final_val_accuracy: float
    best_epoch: int
    epochs_trained: int
    final_val_loss: float
    params: int
    memory_mb: float
    training_time_s: float
    early_stopping: bool
    lr_reductions: int
    class_metrics: Dict[str, float]
    # extras for grouping
    log_stem: str | None = None
    log_relpath: str | None = None

    def to_flat(self) -> Dict[str, Any]:
        d = asdict(self)
        class_metrics = d.pop("class_metrics", {})
        d.update(class_metrics)
        return d

# ───── Helpers ─────
def parse_confusion_matrix(block: str) -> np.ndarray | None:
    matches = RE_CONF_MATRIX.findall(block)
    if not matches:
        return None
    raw_lines = matches[-1].strip().splitlines()

    rows: List[List[int]] = []
    for ln in raw_lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.lower().startswith("p_"):
            continue
        tokens = re.split(r"\s+", ln)
        if tokens and re.match(r"^[tT]_\d+", tokens[0]):
            tokens = tokens[1:]
        nums = [int(tok) for tok in tokens if re.fullmatch(r"-?\d+", tok)]
        if nums:
            rows.append(nums)

    return np.array(rows, dtype=int) if rows else None


def save_confusion_matrix(cm: np.ndarray, png: Path, title: str) -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm[np.isnan(cm_norm)] = 0.0

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    n = cm.shape[0]
    ticks = range(n)
    label = ["Chemical", "Electric", "Impulsive", "No Thrust"]
    plt.xticks(ticks, label)
    plt.yticks(ticks, label)

    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center",
                     va="center", fontsize=9, color="white" if cm_norm[i, j] > 0.5 else "black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(png)
    plt.close()

def parse_classification_block(block: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for raw_line in block.strip().splitlines():
        ln = raw_line.strip()
        if not ln or ln.lower().startswith("accuracy") or "precision" in ln.lower():
            continue
        parts = re.split(r"\s+", ln)
        if len(parts) < 5:
            continue
        try:
            support  = int(float(parts[-1]))
            f1       = float(parts[-2])
            recall   = float(parts[-3])
            precision= float(parts[-4])
        except ValueError:
            continue
        label = "_".join(parts[:-4]).lower()
        rows.append(
            {"label": label, "precision": precision, "recall": recall, "f1": f1, "support": support}
        )
    return pd.DataFrame(rows, columns=["label", "precision", "recall", "f1", "support"])


def iter_models(text: str) -> Iterator[tuple[str, str]]:
    parts = RE_MODEL_SPLIT.split(text)
    for i in range(1, len(parts), 2):
        yield parts[i].strip(), parts[i + 1]


def epoch_trace(block: str) -> pd.DataFrame:
    """Extract per-epoch metrics. Captures Training Loss, Validation Loss, and Validation Accuracy."""
    rows: List[Dict[str, Any]] = []
    current_epoch: int | None = None
    train_loss: float | None = None
    for raw_ln in block.splitlines():
        ln = raw_ln.strip()

        m_ep = RE_EPOCH_LINE.match(ln)
        if m_ep:
            current_epoch = int(m_ep.group(1))
            train_loss = None
            continue

        if current_epoch is None:
            continue

        m_tr = RE_TRAIN_LOSS.search(ln)
        if m_tr:
            try:
                train_loss = float(m_tr.group(1))
            except ValueError:
                train_loss = None
            continue

        m_val_loss = RE_VAL_LOSS.search(ln)
        m_acc      = RE_VAL_ACC.search(ln)
        if m_val_loss or m_acc:
            row: Dict[str, Any] = {"Epoch": current_epoch}
            if train_loss is not None:
                row["Training Loss"] = train_loss
            if m_val_loss:
                try:
                    row["Validation Loss"] = float(m_val_loss.group(1))
                except ValueError:
                    pass
            if m_acc:
                try:
                    row["Validation Accuracy"] = float(m_acc.group(1))
                except ValueError:
                    pass
            rows.append(row)
            current_epoch = None
            train_loss = None

    return pd.DataFrame(rows)


def summarize(model: str, block: str) -> Tuple[Summary, pd.DataFrame]:
    accs = [float(x) for x in RE_VAL_ACC.findall(block)]
    if not accs:
        raise ValueError(f"No validation accuracy for {model}")
    max_acc    = max(accs)
    best_epoch = accs.index(max_acc) + 1
    final_acc  = accs[-1]

    losses     = [float(x) for x in RE_VAL_LOSS.findall(block)]
    final_loss = losses[-1] if losses else float("nan")

    params = int(RE_PARAMS.search(block).group(1)) if RE_PARAMS.search(block) else 0
    mem    = float(RE_MEMORY.search(block).group(1)) if RE_MEMORY.search(block) else float("nan")
    time_s = float(RE_TIME.search(block).group(1)) if RE_TIME.search(block) else float("nan")

    class_df      = pd.DataFrame(columns=["label", "precision", "recall", "f1", "support"])
    class_metrics: Dict[str, float] = {}
    reports = RE_CLASS_REPORT.findall(block)
    if reports:
        class_df = parse_classification_block(reports[-1])
        if not class_df.empty:
            class_df.insert(0, "Model", model)
            for _, r in class_df.iterrows():
                for m in ("precision", "recall", "f1"):
                    class_metrics[f"{r['label']}_{m}"] = r[m]

    summary = Summary(
        model,
        max_acc,
        final_acc,
        best_epoch,
        len(accs),
        final_loss,
        params, mem, time_s,
        "Early stopping" in block,
        block.lower().count("reducing learning rate"),
        class_metrics,
    )
    return summary, class_df

# ───── Plot helpers ─────
def save_acc_plot(df: pd.DataFrame, png: Path) -> None:
    plt.figure(figsize=(8, 5))
    for model, grp in df.groupby("Model"):
        plt.plot(grp["Epoch"], grp["Validation Accuracy"], marker="o", label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy vs. Epoch")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(png)
    plt.close()


def save_loss_plot(df: pd.DataFrame, png: Path) -> None:
    """Plot Validation Loss vs. Epoch for all models."""
    plt.figure(figsize=(8, 5))
    for model, grp in df.groupby("Model"):
        gg = grp.dropna(subset=["Validation Loss"])
        if gg.empty:
            continue
        plt.plot(gg["Epoch"], gg["Validation Loss"], marker="o", label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs. Epoch")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(png)
    plt.close()

# ───── I/O ─────
_SUFFIX_RE = re.compile(r"([A-Za-z]+(?:_[A-Za-z]+)*)$")
def _suffix(stem: str) -> str:
    m = _SUFFIX_RE.search(stem)
    return m.group(1) if m else "unsuffixed"

def process_log(path: Path, root: Path, force: bool = False, emit_outputs: bool = True) -> Tuple[List[Summary], pd.DataFrame | None, pd.DataFrame | None]:
    """
    parsed_data/
        <orbit-dir>/
            <run-dir>/
                <suffix>/
                    csv/
                        summary_<stem>.csv
                        epochs_<stem>.csv
                        class_report_<stem>.csv
                    plots/
                        acc_plot_<stem>.png
                        loss_plot_<stem>.png
                        confmats/
                            confmat_<stem>_<model>.png
    """
    stem      = path.stem
    suffix    = _suffix(stem)
    rel_dir   = path.parent.relative_to(root)
    base_dir  = Path("parsed_data") / rel_dir / suffix
    csv_dir   = base_dir / "csv"
    plot_dir  = base_dir / "plots"
    cm_dir    = plot_dir / "confmats"
    if emit_outputs:
        for d in (csv_dir, cm_dir):
            d.mkdir(parents=True, exist_ok=True)

    summary_csv = (csv_dir / f"summary_{stem}.csv") if emit_outputs else None
    epochs_csv  = (csv_dir / f"epochs_{stem}.csv") if emit_outputs else None
    class_csv   = (csv_dir / f"class_report_{stem}.csv") if emit_outputs else None
    acc_png     = (plot_dir / f"acc_plot_{stem}.png") if emit_outputs else None
    loss_png    = (plot_dir / f"loss_plot_{stem}.png") if emit_outputs else None

    if emit_outputs and (not force and summary_csv.exists() and epochs_csv.exists()
            and class_csv.exists() and acc_png.exists() and loss_png.exists()):
        print(f"skip {path}")
        return [], None, None

    text = path.read_text(errors="ignore")
    summaries: List[Summary] = []
    epochs_dfs: List[pd.DataFrame] = []
    class_dfs: List[pd.DataFrame] = []

    for model, blk in iter_models(text):
        summ, cls_df = summarize(model, blk)
        # annotate for grouping
        summ.log_stem = stem
        summ.log_relpath = str(path.relative_to(root))
        summaries.append(summ)

        if cls_df is not None and not cls_df.empty:
            class_dfs.append(cls_df)

        ep_df = epoch_trace(blk)
        if ep_df is not None and not ep_df.empty:
            ep_df.insert(0, "Model", model)
            epochs_dfs.append(ep_df)

        cm = parse_confusion_matrix(blk)
        if emit_outputs and cm is not None:
            cm_png = cm_dir / f"confmat_{stem}_{model.replace(' ', '_')}.png"
            title = f"{model} – Confusion Matrix (Final Val Acc: {summ.final_val_accuracy:.2f}%)"
            save_confusion_matrix(cm, cm_png, title)

    if not emit_outputs:
        # Return parsed frames to caller for aggregation
        ep_all = pd.concat(epochs_dfs, ignore_index=True) if epochs_dfs else None
        cls_all = pd.concat(class_dfs, ignore_index=True) if class_dfs else None
        return summaries, ep_all, cls_all

    # Write per-log artifacts
    pd.DataFrame([s.to_flat() for s in summaries]).to_csv(summary_csv, index=False)

    if epochs_dfs:
        ep_all = pd.concat(epochs_dfs, ignore_index=True)
        cols = ["Model", "Epoch", "Training Loss", "Validation Loss", "Validation Accuracy"]
        present = [c for c in cols if c in ep_all.columns]
        ep_all = ep_all[present]
        ep_all.to_csv(epochs_csv, index=False)
    else:
        ep_all = pd.DataFrame(columns=["Model", "Epoch", "Validation Accuracy"])
        ep_all.to_csv(epochs_csv, index=False)
        print(f"{path} → no epoch data; wrote empty epochs.csv")

    if class_dfs:
        pd.concat(class_dfs, ignore_index=True).to_csv(class_csv, index=False)
    else:
        pd.DataFrame(columns=["Model","label","precision","recall","f1","support"]).to_csv(
            class_csv, index=False
        )

    if not ep_all.empty:
        save_acc_plot(ep_all, acc_png)
        save_loss_plot(ep_all, loss_png)

    print(f"processed → {summary_csv}")
    return summaries, ep_all, None

# ───── Group processing ─────
def process_group_dir(group_dir: Path, root: Path, force: bool, group_name: str | None, emit_per_log: bool) -> Path:
    """
    Parse all *.log under group_dir and write ONE combined summary CSV:
    parsed_data/<group_rel>/_group/csv/summary_<group_name or 'group'>.csv
    """
    if not group_dir.is_dir():
        raise ValueError(f"--group-dir must be a directory: {group_dir}")

    logs = list(group_dir.rglob("*.log"))
    print(f"[group] found {len(logs)} logs under {group_dir}")
    if not logs:
        raise ValueError(f"No logs found in group dir: {group_dir}")

    group_rel = group_dir.relative_to(root)
    base_dir  = Path("parsed_data") / group_rel / "_group"
    csv_dir   = base_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"summary_{group_name}.csv" if group_name else "summary_group.csv"
    out_csv  = csv_dir / out_name

    # Parse all logs (optionally also emit per-log artifacts)
    all_summ_rows: List[Dict[str, Any]] = []
    for lg in logs:
        summaries, _, _ = process_log(lg, root, force=force, emit_outputs=emit_per_log)
        # summaries can be empty if skipped; handle it
        for s in summaries:
            row = s.to_flat()
            row["log_stem"] = s.log_stem
            row["log_relpath"] = s.log_relpath
            all_summ_rows.append(row)

    if not all_summ_rows:
        # Possible if everything was skipped and emit_per_log=True; still write empty CSV for group
        pd.DataFrame().to_csv(out_csv, index=False)
        print(f"[group] wrote empty summary: {out_csv}")
        return out_csv

    # Column ordering: stable, useful first
    first_cols = [
        "log_stem", "log_relpath", "model",
        "final_val_accuracy", "max_val_accuracy", "best_epoch", "epochs_trained",
        "final_val_loss", "params", "memory_mb", "training_time_s",
        "early_stopping", "lr_reductions",
    ]
    df = pd.DataFrame(all_summ_rows)
    cols = first_cols + [c for c in df.columns if c not in first_cols]
    df = df.reindex(columns=cols)
    df.to_csv(out_csv, index=False)
    print(f"[group] wrote combined summary → {out_csv}")
    return out_csv

# ───── CLI ─────
def main() -> None:
    ap = argparse.ArgumentParser(description="Parse training logs → CSV + plots (per-log or grouped)")
    ap.add_argument("root", type=Path, help="Root folder that contains logs (used for relative paths).")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--journal", action="store_true", help="(Unused placeholder for now).")

    # Group mode
    ap.add_argument("--group-dir", type=Path,
                    help="Process only this directory and emit ONE combined summary CSV for all *.log within.")
    ap.add_argument("--group-name", type=str, default=None,
                    help="Optional name for the combined summary file (summary_<name>.csv).")
    ap.add_argument("--emit-per-log", action="store_true",
                    help="When using --group-dir, also emit the usual per-log CSVs/plots.")

    args = ap.parse_args()

    # Validate root
    if not args.root.exists():
        raise SystemExit(f"root not found: {args.root}")
    if args.group_dir is not None:
        # Ensure group_dir is inside root
        try:
            _ = args.group_dir.resolve().relative_to(args.root.resolve())
        except Exception:
            raise SystemExit(f"--group-dir must be inside ROOT: {args.group_dir} not under {args.root}")

        process_group_dir(args.group_dir, args.root, force=args.force,
                          group_name=args.group_name, emit_per_log=args.emit_per_log)
        return

    # Default: per-log recursion as before
    logs = list(args.root.rglob("*.log"))
    print(f"found {len(logs)} logs under {args.root}")
    for lg in logs:
        process_log(lg, args.root, force=args.force, emit_outputs=True)

if __name__ == "__main__":
    main()
