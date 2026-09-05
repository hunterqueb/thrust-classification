from __future__ import annotations

"""Sequence-classification log parser (v1.0)

Parses logs produced by scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py, which
trains and reports on TWO approaches per backbone (LSTM/MAMBA/... and classic-ML families like
LightGBM) for per-timestep 4-class thrust typing:
  joint    a single 4-class model (No Thrust/Chemical/Electric/Impulsive)
  cascade  a binary Stage 1 detector (No Thrust/Thrust) feeding a 3-class Stage 2 type
           classifier (Chemical/Electric/Impulsive), combined at inference into the same
           4-class label space so it is directly comparable to the joint model.

Emits, per log, mirroring gmat/data/classification/displayLogData.py's layout:
  parsed_data/<orbit>/<run-dir>/<suffix>/csv/
      runs_<stem>.csv         one row per training run (backbone x component)
      epochs_<stem>.csv       one row per epoch per training run
      eval_<stem>.csv         one row per evaluated report (joint / cascade stage1 / cascade
                               end-to-end / stage1-solo / stage2-solo)
      comparison_<stem>.csv   one row per backbone: joint vs. cascade head-to-head
  parsed_data/<orbit>/<run-dir>/<suffix>/plots/
      epoch_f1_<stem>.png, epoch_loss_<stem>.png
      comparison_<stem>.png   grouped bar chart: joint vs. cascade accuracy & macro-F1
      confmats/confmat_<stem>_<backbone>_<eval_stage>.png

Usage:
python displaySeqLogData.py . --force
python displaySeqLogData.py . --group-dir leo/ --group-name leo
"""

from pathlib import Path
import argparse
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ───── Regexes ─────
RE_ENTER = re.compile(r"Entering\s+(.+?)\s+Training(?:\s+Loop)?\b")

RE_EPOCH_TRAIN = re.compile(r"^Epoch\s*\[(\d+)/(\d+)\]\s*Train Loss:\s*([\d.]+)")
RE_VAL_EVENT = re.compile(r"Val Event P\(macro[^)]*\):\s*([\d.]+)\s*\|\s*R:\s*([\d.]+)\s*\|\s*F1:\s*([\d.]+)")
RE_VAL_LOSS_LINE = re.compile(r"^Val Loss:\s*([\d.]+)")

RE_PARAMS = re.compile(r"Total parameters:\s*(\d+|NaN)")
RE_MEMORY = re.compile(r"Total memory \(MB\):\s*([\d.]+)")
RE_TRAIN_ELAPSED = re.compile(r"^[ \t]*Elapsed time is\s*([\d.]+)\s*seconds\.?\s*$", re.M)
RE_INFERENCE_TIME = re.compile(r"Inference Time\s+Elapsed time is\s*([\d.]+)\s*seconds")

RE_OVERALL_ACC = re.compile(r"Accuracy:\s*([\d.]+)%\s*\((\d+)/(\d+)\)")
RE_PERCLASS_ACC_BLOCK = re.compile(r"Per-Class Accuracy:\s*\r?\n((?:[ \t]+.+\r?\n)+)")
RE_PERCLASS_LINE = re.compile(r"^\s*(.+?):\s*([\d.]+)%\s*\((\d+)/(\d+)\)\s*$", re.M)
RE_CLASS_REPORT = re.compile(
    r"Classification Report:\s*\r?\n(.*?)\r?\n[ \t]*(?:\r?\n|\Z)Confusion Matrix", re.S
)
RE_CONF_MATRIX_BLOCK = re.compile(
    r"Confusion Matrix \(rows = true, cols = predicted\):\s*\r?\n(.*?)(?:\r?\n[ \t]*\r?\n|\Z)", re.S
)

RE_STAGE2_COND_ACC = re.compile(
    r"Stage-2 type accuracy conditioned on correct stage-1 detection:\s*([\d.]+)%\s*\((\d+)\s*frames\)"
)
RE_STAGE1_ONLY_ACC = re.compile(r"Stage-1 \(detector-only\) accuracy:\s*([\d.]+)%")

# Header anchors used to pull the right report out of a cascade Stage 2 block, which can contain
# two or three (Accuracy/Per-Class Accuracy/Classification Report/Confusion Matrix) quadruples
# depending on backbone family and script version: a standalone Stage 2 report (present once the
# training script prints one for both neural and classic-ML backbones; absent in older logs),
# the Cascade Stage 1 standalone report, and the Cascade End-to-End report. Anchoring on each
# report's own header -- rather than counting quadruples positionally -- stays correct regardless
# of which subset is present.
RE_STAGE2_STANDALONE_HDR = re.compile(r"Stage 2 \(Type Classifier\) Standalone Validation")
RE_CASCADE_STAGE1_HDR = re.compile(r"---\s*Cascade Stage 1 \(Detector\) Standalone Metrics\s*---")
RE_CASCADE_E2E_HDR = re.compile(r"---\s*Cascade End-to-End \(Stage1 -> Stage2 combined\) Metrics\s*---")

APPROACH_BY_COMPONENT = {
    "Joint": "joint",
    "Stage1": "cascade",
    "Stage2": "cascade",
    "Stage1_solo": "stage1_solo",
    "Stage2_solo": "stage2_solo",
    "Unknown": "unknown",
}


# ───── Dataclasses ─────
@dataclass
class RunSummary:
    model: str
    component: str          # Joint | Stage1 | Stage2 | Stage1_solo | Stage2_solo
    approach: str            # joint | cascade | stage1_solo | stage2_solo
    epochs_trained: int
    best_epoch: int          # epoch (1-indexed) of minimum validation loss; -1 if no epochs (classic ML)
    min_val_loss: float
    final_val_loss: float
    max_event_f1: float
    final_event_f1: float
    params: float            # NaN for classic-ML/GBDT models (no meaningful parameter count)
    memory_mb: float
    training_time_s: float
    early_stopping: bool
    lr_reductions: int
    log_stem: str | None = None
    log_relpath: str | None = None


@dataclass
class EvalSummary:
    model: str
    component: str           # Joint | Stage1 | Stage2 | Cascade
    approach: str
    eval_stage: str          # joint_4class | cascade_stage1_standalone | cascade_end_to_end |
                              # stage1_solo | stage2_solo
    accuracy: float
    n_correct: int
    n_total: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    min_thrust_class_recall: float  # worst recall among non-background classes -- low value with
                                     # high accuracy flags a model that mostly predicts "No Thrust"
    inference_time_s: float
    stage1_only_accuracy: float | None = None
    stage2_conditional_accuracy: float | None = None
    stage2_conditional_frames: int | None = None
    class_metrics: Dict[str, float] = field(default_factory=dict)
    log_stem: str | None = None
    log_relpath: str | None = None

    def to_flat(self) -> Dict[str, Any]:
        d = asdict(self)
        class_metrics = d.pop("class_metrics", {})
        d.update(class_metrics)
        return d


# ───── Label parsing helpers ─────
def classify_component(label: str) -> Tuple[str, str]:
    """label = text captured between 'Entering ' and ' Training[ Loop]'. Returns (backbone,
    component) where component in {Joint, Stage1, Stage2, Stage1_solo, Stage2_solo, Unknown}."""
    l = label.strip()

    # classic-ML / PCA+MLP solo-mode form: "<Name> (joint|stage1|stage2)"
    m = re.match(r"^(.*?)\s*\((joint|stage1|stage2)\)\s*$", l, re.I)
    if m:
        backbone = m.group(1).strip()
        mode = m.group(2).lower()
        component = {"joint": "Joint", "stage1": "Stage1_solo", "stage2": "Stage2_solo"}[mode]
        return backbone, component

    # whole-trajectory MiniRocket joint special-case: "MiniRocket (whole-trajectory, Joint 4-class)"
    m = re.match(r"^(.*?)\s*\(.*Joint.*\)\s*$", l, re.I)
    if m:
        return m.group(1).strip(), "Joint"

    m = re.match(r"^(.+?)\s+Stage\s*1\b\s*(\(.*\))?$", l, re.I)
    if m:
        backbone = m.group(1).strip()
        solo = bool(m.group(2) and "standalone" in m.group(2).lower())
        return backbone, ("Stage1_solo" if solo else "Stage1")

    m = re.match(r"^(.+?)\s+Stage\s*2\b\s*(\(.*\))?$", l, re.I)
    if m:
        backbone = m.group(1).strip()
        solo = bool(m.group(2) and "standalone" in m.group(2).lower())
        return backbone, ("Stage2_solo" if solo else "Stage2")

    m = re.match(r"^(.+?)\s+Joint\b", l, re.I)
    if m:
        return m.group(1).strip(), "Joint"

    return l, "Unknown"


def iter_blocks(text: str) -> Iterator[Tuple[str, str, str, str]]:
    """Yields (raw_label, backbone, component, block_text) for each 'Entering ... Training[ Loop]'
    marker. block_text runs up to (not including) the next such marker -- so a cascade's Stage 2
    block also carries the Cascade Evaluation text that follows it in the log, since that text
    isn't preceded by its own 'Entering' marker."""
    matches = list(RE_ENTER.finditer(text))
    for i, m in enumerate(matches):
        label = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block_text = text[start:end]
        backbone, component = classify_component(label)
        yield label, backbone, component, block_text


# ───── Epoch trace ─────
def epoch_trace_seq(block: str) -> pd.DataFrame:
    """Extracts per-epoch Training Loss, Val Event P/R/F1 (macro), and Val Loss. Per-class P/R/F1
    arrays are intentionally not parsed here -- the post-training classification report already
    gives per-class detail at convergence, which is what the eval CSV captures."""
    rows: List[Dict[str, Any]] = []
    cur: Dict[str, Any] | None = None
    for raw in block.splitlines():
        ln = raw.strip()

        m_ep = RE_EPOCH_TRAIN.match(ln)
        if m_ep:
            if cur is not None:
                rows.append(cur)
            cur = {"Epoch": int(m_ep.group(1)), "Training Loss": float(m_ep.group(3))}
            continue

        if cur is None:
            continue

        m_ev = RE_VAL_EVENT.search(ln)
        if m_ev:
            cur["Val Event Precision"] = float(m_ev.group(1))
            cur["Val Event Recall"] = float(m_ev.group(2))
            cur["Val Event F1"] = float(m_ev.group(3))
            continue

        m_vl = RE_VAL_LOSS_LINE.match(ln)
        if m_vl:
            cur["Val Loss"] = float(m_vl.group(1))
            continue

    if cur is not None:
        rows.append(cur)

    cols = ["Epoch", "Training Loss", "Val Event Precision", "Val Event Recall", "Val Event F1", "Val Loss"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    return df.reindex(columns=[c for c in cols if c in df.columns])


# ───── Classification report / confusion matrix parsing ─────
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
            support = int(float(parts[-1]))
            f1 = float(parts[-2])
            recall = float(parts[-3])
            precision = float(parts[-4])
        except ValueError:
            continue
        label = "_".join(parts[:-4]).lower()
        rows.append({"label": label, "precision": precision, "recall": recall, "f1": f1, "support": support})
    return pd.DataFrame(rows, columns=["label", "precision", "recall", "f1", "support"])


def parse_confusion_matrix_lines(text: str) -> np.ndarray | None:
    rows: List[List[int]] = []
    for ln in text.strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.lower().startswith("p_"):
            continue
        tokens = re.split(r"\s+", ln)
        nums = [int(tok) for tok in tokens if re.fullmatch(r"-?\d+", tok)]
        if nums:
            rows.append(nums)
    return np.array(rows, dtype=int) if rows else None


def _extract_section(text: str) -> Dict[str, Any] | None:
    """Extracts the FIRST (Accuracy / Per-Class Accuracy / Classification Report / Confusion
    Matrix) quadruple found in text, or None if any piece is missing. A Joint or solo-stage block
    always has exactly one such quadruple, so this suffices there; for a cascade Stage 2 block
    (which can carry two or three quadruples), use find_eval_section_after to anchor on a specific
    report's own header first."""
    m_acc = RE_OVERALL_ACC.search(text)
    m_pca = RE_PERCLASS_ACC_BLOCK.search(text)
    m_cr = RE_CLASS_REPORT.search(text)
    m_cm = RE_CONF_MATRIX_BLOCK.search(text)
    if not (m_acc and m_pca and m_cr and m_cm):
        return None
    per_class_acc = {lbl.strip(): float(pct) for lbl, pct, _c, _t in RE_PERCLASS_LINE.findall(m_pca.group(1))}
    return {
        "accuracy": float(m_acc.group(1)),
        "n_correct": int(m_acc.group(2)),
        "n_total": int(m_acc.group(3)),
        "per_class_accuracy": per_class_acc,
        "labels": list(per_class_acc.keys()),
        "class_df": parse_classification_block(m_cr.group(1)),
        "confusion_matrix": parse_confusion_matrix_lines(m_cm.group(1)),
    }


def find_eval_section_after(block: str, header: re.Pattern) -> Dict[str, Any] | None:
    """Finds header in block, then extracts the report quadruple immediately following it (the
    first one found in the remaining text) -- correct regardless of how many other reports follow
    later in the same block."""
    m = header.search(block)
    if not m:
        return None
    return _extract_section(block[m.end():])


# ───── Summary builders ─────
def build_run_summary(backbone: str, component: str, approach: str, block: str,
                       log_stem: str, log_relpath: str) -> Tuple[RunSummary, pd.DataFrame]:
    ep_df = epoch_trace_seq(block)

    if not ep_df.empty and "Val Loss" in ep_df.columns and ep_df["Val Loss"].notna().any():
        best_idx = ep_df["Val Loss"].idxmin()
        best_epoch = int(ep_df.loc[best_idx, "Epoch"])
        min_val_loss = float(ep_df["Val Loss"].min())
        final_val_loss = float(ep_df["Val Loss"].iloc[-1])
    else:
        best_epoch = -1
        min_val_loss = float("nan")
        final_val_loss = float("nan")

    if not ep_df.empty and "Val Event F1" in ep_df.columns and ep_df["Val Event F1"].notna().any():
        max_event_f1 = float(ep_df["Val Event F1"].max())
        final_event_f1 = float(ep_df["Val Event F1"].iloc[-1])
    else:
        max_event_f1 = float("nan")
        final_event_f1 = float("nan")

    epochs_trained = int(ep_df["Epoch"].max()) if not ep_df.empty else 0

    m_params = RE_PARAMS.search(block)
    if m_params is None or m_params.group(1) == "NaN":
        params = float("nan")
    else:
        params = float(m_params.group(1))
    m_mem = RE_MEMORY.search(block)
    memory_mb = float(m_mem.group(1)) if m_mem else float("nan")
    m_time = RE_TRAIN_ELAPSED.search(block)
    training_time_s = float(m_time.group(1)) if m_time else float("nan")

    run = RunSummary(
        model=backbone, component=component, approach=approach,
        epochs_trained=epochs_trained, best_epoch=best_epoch,
        min_val_loss=min_val_loss, final_val_loss=final_val_loss,
        max_event_f1=max_event_f1, final_event_f1=final_event_f1,
        params=params, memory_mb=memory_mb, training_time_s=training_time_s,
        early_stopping=("Early stopping" in block),
        lr_reductions=block.lower().count("reducing learning rate"),
        log_stem=log_stem, log_relpath=log_relpath,
    )
    return run, ep_df


def eval_summary_from_section(backbone: str, component: str, approach: str, eval_stage: str,
                               section: Dict[str, Any], inference_time_s: float,
                               stage1_only_accuracy: float | None = None,
                               stage2_conditional_accuracy: float | None = None,
                               stage2_conditional_frames: int | None = None,
                               log_stem: str | None = None, log_relpath: str | None = None) -> EvalSummary:
    class_df = section["class_df"]

    def _row(label: str):
        r = class_df[class_df["label"] == label]
        return r.iloc[0] if not r.empty else None

    macro = _row("macro_avg")
    weighted = _row("weighted_avg")
    macro_p = float(macro["precision"]) if macro is not None else float("nan")
    macro_r = float(macro["recall"]) if macro is not None else float("nan")
    macro_f1 = float(macro["f1"]) if macro is not None else float("nan")
    weighted_f1 = float(weighted["f1"]) if weighted is not None else float("nan")

    non_bg = class_df[~class_df["label"].isin(["no_thrust", "macro_avg", "weighted_avg"])]
    min_recall = float(non_bg["recall"].min()) if not non_bg.empty else float("nan")

    class_metrics: Dict[str, float] = {}
    for _, r in class_df.iterrows():
        if r["label"] in ("macro_avg", "weighted_avg"):
            continue
        for m in ("precision", "recall", "f1"):
            class_metrics[f"{r['label']}_{m}"] = r[m]
    for lbl, pct in section["per_class_accuracy"].items():
        key = re.sub(r"\s+", "_", lbl.strip().lower())
        class_metrics[f"{key}_accuracy"] = pct

    return EvalSummary(
        model=backbone, component=component, approach=approach, eval_stage=eval_stage,
        accuracy=section["accuracy"], n_correct=section["n_correct"], n_total=section["n_total"],
        macro_precision=macro_p, macro_recall=macro_r, macro_f1=macro_f1, weighted_f1=weighted_f1,
        min_thrust_class_recall=min_recall,
        inference_time_s=inference_time_s,
        stage1_only_accuracy=stage1_only_accuracy,
        stage2_conditional_accuracy=stage2_conditional_accuracy,
        stage2_conditional_frames=stage2_conditional_frames,
        class_metrics=class_metrics,
        log_stem=log_stem, log_relpath=log_relpath,
    )


# ───── Comparison table ─────
def build_comparison(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for model, grp in eval_df.groupby("model"):
        row: Dict[str, Any] = {"Model": model}
        joint = grp[grp["eval_stage"] == "joint_4class"]
        end2end = grp[grp["eval_stage"] == "cascade_end_to_end"]
        stage1 = grp[grp["eval_stage"] == "cascade_stage1_standalone"]
        stage2 = grp[grp["eval_stage"] == "cascade_stage2_standalone"]
        s1solo = grp[grp["eval_stage"] == "stage1_solo"]
        s2solo = grp[grp["eval_stage"] == "stage2_solo"]

        if not joint.empty:
            j = joint.iloc[0]
            row["Joint_Accuracy"] = j["accuracy"]
            row["Joint_Macro_F1"] = j["macro_f1"]
            row["Joint_Weighted_F1"] = j["weighted_f1"]
            row["Joint_Min_Thrust_Recall"] = j["min_thrust_class_recall"]
            row["Joint_Inference_Time_s"] = j["inference_time_s"]

        if not end2end.empty:
            e = end2end.iloc[0]
            row["Cascade_EndToEnd_Accuracy"] = e["accuracy"]
            row["Cascade_EndToEnd_Macro_F1"] = e["macro_f1"]
            row["Cascade_EndToEnd_Weighted_F1"] = e["weighted_f1"]
            row["Cascade_EndToEnd_Min_Thrust_Recall"] = e["min_thrust_class_recall"]
            row["Cascade_Stage2_Conditional_Accuracy"] = e["stage2_conditional_accuracy"]
            row["Cascade_Inference_Time_s"] = e["inference_time_s"]

        if not stage1.empty:
            row["Cascade_Stage1_Detector_Accuracy"] = stage1.iloc[0]["accuracy"]
            row["Cascade_Stage1_Detector_Macro_F1"] = stage1.iloc[0]["macro_f1"]
        if not stage2.empty:
            row["Cascade_Stage2_Standalone_Accuracy"] = stage2.iloc[0]["accuracy"]
            row["Cascade_Stage2_Standalone_Macro_F1"] = stage2.iloc[0]["macro_f1"]
        if not s1solo.empty:
            row["Stage1_Solo_Accuracy"] = s1solo.iloc[0]["accuracy"]
        if not s2solo.empty:
            row["Stage2_Solo_Accuracy"] = s2solo.iloc[0]["accuracy"]

        if "Joint_Accuracy" in row and "Cascade_EndToEnd_Accuracy" in row:
            row["Accuracy_Delta_Cascade_minus_Joint"] = row["Cascade_EndToEnd_Accuracy"] - row["Joint_Accuracy"]
            row["Macro_F1_Delta_Cascade_minus_Joint"] = row["Cascade_EndToEnd_Macro_F1"] - row["Joint_Macro_F1"]
            row["Better_By_Accuracy"] = (
                "Cascade" if row["Accuracy_Delta_Cascade_minus_Joint"] > 0
                else "Joint" if row["Accuracy_Delta_Cascade_minus_Joint"] < 0 else "Tie"
            )
            row["Better_By_Macro_F1"] = (
                "Cascade" if row["Macro_F1_Delta_Cascade_minus_Joint"] > 0
                else "Joint" if row["Macro_F1_Delta_Cascade_minus_Joint"] < 0 else "Tie"
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ───── Plot helpers ─────
def save_confusion_matrix(cm: np.ndarray, labels: List[str], png: Path, title: str) -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm[np.isnan(cm_norm)] = 0.0

    n = cm.shape[0]
    side = max(4.5, 1.3 * n + 2)
    plt.figure(figsize=(side, side - 0.7))
    im = plt.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    ticks = range(n)
    plt.xticks(ticks, labels, rotation=30, ha="right")
    plt.yticks(ticks, labels)

    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                      fontsize=9, color="white" if cm_norm[i, j] > 0.5 else "black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(png)
    plt.close()


def save_epoch_f1_plot(df: pd.DataFrame, png: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    for (model, comp), grp in df.groupby(["Model", "Component"]):
        gg = grp.dropna(subset=["Val Event F1"]) if "Val Event F1" in grp.columns else grp.iloc[0:0]
        if gg.empty:
            continue
        plt.plot(gg["Epoch"], gg["Val Event F1"], marker="o", markersize=3, label=f"{model} {comp}")
    plt.xlabel("Epoch")
    plt.ylabel("Val Event F1 (macro)")
    plt.title("Validation Event F1 (macro) vs. Epoch")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(png)
    plt.close()


def save_epoch_loss_plot(df: pd.DataFrame, png: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    for (model, comp), grp in df.groupby(["Model", "Component"]):
        gg = grp.dropna(subset=["Val Loss"]) if "Val Loss" in grp.columns else grp.iloc[0:0]
        if gg.empty:
            continue
        plt.plot(gg["Epoch"], gg["Val Loss"], marker="o", markersize=3, label=f"{model} {comp}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs. Epoch")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(png)
    plt.close()


def save_comparison_plot(df: pd.DataFrame, png: Path) -> None:
    if df.empty:
        return
    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2.4 * len(models)), 5))

    acc_joint = df["Joint_Accuracy"] if "Joint_Accuracy" in df.columns else pd.Series([np.nan] * len(models))
    acc_casc = df["Cascade_EndToEnd_Accuracy"] if "Cascade_EndToEnd_Accuracy" in df.columns else pd.Series([np.nan] * len(models))
    axes[0].bar(x - width / 2, acc_joint, width, label="Joint")
    axes[0].bar(x + width / 2, acc_casc, width, label="Cascade (end-to-end)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Overall Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    f1_joint = df["Joint_Macro_F1"] if "Joint_Macro_F1" in df.columns else pd.Series([np.nan] * len(models))
    f1_casc = df["Cascade_EndToEnd_Macro_F1"] if "Cascade_EndToEnd_Macro_F1" in df.columns else pd.Series([np.nan] * len(models))
    axes[1].bar(x - width / 2, f1_joint, width, label="Joint")
    axes[1].bar(x + width / 2, f1_casc, width, label="Cascade (end-to-end)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Macro-averaged F1 (class-balance aware)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Joint vs. Cascade Classifier Comparison")
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)


def save_stage_cascade_comparison_plot(eval_df: pd.DataFrame, run_df: pd.DataFrame, png: Path) -> None:
    """Compares cascade Stage 1 (detector) models against each other across backbones, and cascade
    Stage 2 (type classifier) models against each other across backbones -- i.e. within-stage,
    not joint vs. cascade. Both stages get their own standalone Accuracy/Macro-F1 report from the
    training script (see mambaTimeSeriesSeqClassificationGMATThrusts.py's '... Standalone
    Validation' prints, added for every backbone family including classic-ML/GBDT). Logs from
    before that print existed won't have a standalone Stage 2 report, so this falls back to the
    accuracy conditioned on Stage 1 detecting correctly and the macro-F1 Stage 2 reached on its
    own 3-class validation split during training."""
    empty_sf = pd.DataFrame(columns=["model", "accuracy", "macro_f1"])
    stage1 = eval_df[eval_df["eval_stage"] == "cascade_stage1_standalone"][["model", "accuracy", "macro_f1"]] \
        if not eval_df.empty else empty_sf.copy()
    stage1 = stage1.sort_values("accuracy", ascending=False)

    stage2_standalone = eval_df[eval_df["eval_stage"] == "cascade_stage2_standalone"][["model", "accuracy", "macro_f1"]] \
        if not eval_df.empty else empty_sf.copy()
    end2end = eval_df[eval_df["eval_stage"] == "cascade_end_to_end"][["model", "stage2_conditional_accuracy"]] \
        if not eval_df.empty else pd.DataFrame(columns=["model", "stage2_conditional_accuracy"])
    stage2_train_f1 = run_df[(run_df["component"] == "Stage2") & (run_df["approach"] == "cascade")][["model", "max_event_f1"]] \
        if not run_df.empty else pd.DataFrame(columns=["model", "max_event_f1"])

    stage2 = end2end.merge(stage2_standalone, on="model", how="outer").merge(stage2_train_f1, on="model", how="outer")
    stage2["Accuracy"] = stage2.get("accuracy", pd.Series(dtype=float)).combine_first(
        stage2.get("stage2_conditional_accuracy", pd.Series(dtype=float)))
    stage2["Macro_F1"] = stage2.get("macro_f1", pd.Series(dtype=float)).combine_first(
        stage2.get("max_event_f1", pd.Series(dtype=float)))
    stage2 = stage2.sort_values("Accuracy", ascending=False)

    if stage1.empty and stage2.empty:
        return

    n_cols = max(len(stage1), len(stage2), 1)
    fig, axes = plt.subplots(2, 2, figsize=(max(9, 2.2 * n_cols), 9))

    def _bar(ax, models, values, ylabel, title, color):
        x = np.arange(len(models))
        ax.bar(x, values.fillna(0), color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        for xi, v in zip(x, values):
            label = f"{v:.2f}" if pd.notna(v) else "N/A"
            ax.annotate(label, (xi, v if pd.notna(v) else 0), ha="center", va="bottom", fontsize=8)

    _bar(axes[0, 0], stage1["model"], stage1["accuracy"], "Accuracy (%)",
         "Stage 1 (Detector) -- Standalone Accuracy", "#4C72B0")
    _bar(axes[0, 1], stage1["model"], stage1["macro_f1"], "Macro F1",
         "Stage 1 (Detector) -- Macro F1", "#4C72B0")
    _bar(axes[1, 0], stage2["model"], stage2["Accuracy"], "Accuracy (%)",
         "Stage 2 (Type Classifier) -- Standalone Accuracy", "#DD8452")
    _bar(axes[1, 1], stage2["model"], stage2["Macro_F1"], "Macro F1",
         "Stage 2 (Type Classifier) -- Macro F1", "#DD8452")

    fig.suptitle("Cascade Stage 1 vs. Stage 1, and Stage 2 vs. Stage 2, Across Backbones")
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)


# ───── I/O ─────
# Log stems are built by the training script as f"{propMin}min{systems}{strAdd}" (see logStem in
# mambaTimeSeriesSeqClassificationGMATThrusts.py), where strAdd is the underscore-joined list of
# active flags -- "30min1500Energy_OE", "30min1500Energy_J2Energy_OE",
# "30min1500Energy_OE_PhysLoss0.1". The suffix naming the output directory is everything after the
# "<propMin>min<systems>" prefix, so matching that prefix and keeping the remainder is what gives
# each flag combination its own directory.
#
# Doing it by prefix rather than by a letters-only pattern anchored to the END of the stem matters
# because several flag names contain digits, and the old pattern mis-grouped every one of them:
#   30min1500Energy_J2Energy_OE  -> "Energy_OE"  (the 2 broke the token, and the trailing letters
#                                   happened to spell a real suffix, so J2-energy runs landed in
#                                   the SAME directory as plain --energy ones)
#   30min1500Energy_OE_PhysLoss0.1 -> "unsuffixed" (stem ends in a digit; nothing matched at all,
#                                     so every such run shared one bucket -- also SmoothGap2,
#                                     Sinusoids3, VelNoise0.01)
#   100min800Norm_Noise_Train_640_Test_vleo -> "Test_vleo" (truncated at the 640)
_STEM_PREFIX_RE = re.compile(r"^\d+min\d+_?")


def _suffix(stem: str) -> str:
    rest = _STEM_PREFIX_RE.sub("", stem, count=1)
    if rest == stem:
        # Not a "<propMin>min<systems>..." stem. Keep the whole name rather than guessing at where
        # the flags start, so an unrecognised scheme still gets its own directory instead of
        # silently sharing one with something else.
        return stem or "unsuffixed"
    return rest or "unsuffixed"


def process_log(path: Path, root: Path, force: bool = False, emit_outputs: bool = True
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stem = path.stem
    suffix = _suffix(stem)
    rel_dir = path.parent.relative_to(root)
    base_dir = Path("parsed_data") / rel_dir / suffix
    csv_dir = base_dir / "csv"
    plot_dir = base_dir / "plots"
    cm_dir = plot_dir / "confmats"
    if emit_outputs:
        for d in (csv_dir, cm_dir):
            d.mkdir(parents=True, exist_ok=True)

    runs_csv = csv_dir / f"runs_{stem}.csv"
    epochs_csv = csv_dir / f"epochs_{stem}.csv"
    eval_csv = csv_dir / f"eval_{stem}.csv"
    comparison_csv = csv_dir / f"comparison_{stem}.csv"
    f1_png = plot_dir / f"epoch_f1_{stem}.png"
    loss_png = plot_dir / f"epoch_loss_{stem}.png"
    comparison_png = plot_dir / f"comparison_{stem}.png"
    stage_comparison_png = plot_dir / f"stage_comparison_{stem}.png"

    if emit_outputs and not force and all(
        p.exists() for p in (runs_csv, epochs_csv, eval_csv, comparison_csv, f1_png, loss_png)
    ):
        print(f"skip {path}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    text = path.read_text(errors="ignore")
    log_relpath = str(path.relative_to(root))

    run_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    epoch_frames: List[pd.DataFrame] = []

    for label, backbone, component, block in iter_blocks(text):
        approach = APPROACH_BY_COMPONENT.get(component, "unknown")
        run, ep_df = build_run_summary(backbone, component, approach, block, stem, log_relpath)
        run_rows.append(asdict(run))
        if not ep_df.empty:
            ep_df = ep_df.copy()
            ep_df.insert(0, "Component", component)
            ep_df.insert(0, "Model", backbone)
            epoch_frames.append(ep_df)

        inf_m = RE_INFERENCE_TIME.search(block)
        inf_time = float(inf_m.group(1)) if inf_m else float("nan")

        def _save_cm(section, tag, title):
            if emit_outputs and section["confusion_matrix"] is not None:
                png = cm_dir / f"confmat_{stem}_{backbone}_{tag}.png"
                save_confusion_matrix(section["confusion_matrix"], section["labels"], png, title)

        if component == "Joint":
            section = _extract_section(block)
            if section:
                ev = eval_summary_from_section(backbone, "Joint", "joint", "joint_4class", section,
                                                inf_time, log_stem=stem, log_relpath=log_relpath)
                eval_rows.append(ev.to_flat())
                _save_cm(section, "joint", f"{backbone} Joint (Acc: {section['accuracy']:.2f}%)")

        elif component == "Stage2":
            # A Stage 2 block also carries the trailing Cascade Evaluation text (no 'Entering'
            # marker precedes it). It may hold two or three report quadruples depending on
            # backbone family/script version, so each is pulled out by its own header rather than
            # by position -- see find_eval_section_after.
            stage2_standalone = find_eval_section_after(block, RE_STAGE2_STANDALONE_HDR)
            stage1_standalone = find_eval_section_after(block, RE_CASCADE_STAGE1_HDR)
            end_to_end = find_eval_section_after(block, RE_CASCADE_E2E_HDR)

            if stage2_standalone:
                ev0 = eval_summary_from_section(backbone, "Stage2", "cascade", "cascade_stage2_standalone",
                                                 stage2_standalone, float("nan"),
                                                 log_stem=stem, log_relpath=log_relpath)
                eval_rows.append(ev0.to_flat())
                _save_cm(stage2_standalone, "cascade_stage2",
                         f"{backbone} Cascade Stage2 Type Classifier (Acc: {stage2_standalone['accuracy']:.2f}%)")

            if stage1_standalone:
                ev1 = eval_summary_from_section(backbone, "Stage1", "cascade", "cascade_stage1_standalone",
                                                 stage1_standalone, float("nan"),
                                                 log_stem=stem, log_relpath=log_relpath)
                eval_rows.append(ev1.to_flat())
                _save_cm(stage1_standalone, "cascade_stage1",
                         f"{backbone} Cascade Stage1 Detector (Acc: {stage1_standalone['accuracy']:.2f}%)")

            if end_to_end:
                m_cond = RE_STAGE2_COND_ACC.search(block)
                m_s1only = RE_STAGE1_ONLY_ACC.search(block)
                stage2_cond_acc = float(m_cond.group(1)) if m_cond else float("nan")
                stage2_cond_frames = int(m_cond.group(2)) if m_cond else None
                stage1_only_acc = float(m_s1only.group(1)) if m_s1only else float("nan")
                ev2 = eval_summary_from_section(backbone, "Cascade", "cascade", "cascade_end_to_end",
                                                 end_to_end, inf_time,
                                                 stage1_only_accuracy=stage1_only_acc,
                                                 stage2_conditional_accuracy=stage2_cond_acc,
                                                 stage2_conditional_frames=stage2_cond_frames,
                                                 log_stem=stem, log_relpath=log_relpath)
                eval_rows.append(ev2.to_flat())
                _save_cm(end_to_end, "cascade_end_to_end",
                         f"{backbone} Cascade End-to-End (Acc: {end_to_end['accuracy']:.2f}%)")

        elif component == "Stage1_solo":
            section = _extract_section(block)
            if section:
                ev = eval_summary_from_section(backbone, "Stage1", "stage1_solo", "stage1_solo", section,
                                                inf_time, log_stem=stem, log_relpath=log_relpath)
                eval_rows.append(ev.to_flat())
                _save_cm(section, "stage1_solo", f"{backbone} Stage1 Solo (Acc: {section['accuracy']:.2f}%)")

        elif component == "Stage2_solo":
            section = _extract_section(block)
            if section:
                ev = eval_summary_from_section(backbone, "Stage2", "stage2_solo", "stage2_solo", section,
                                                inf_time, log_stem=stem, log_relpath=log_relpath)
                eval_rows.append(ev.to_flat())
                _save_cm(section, "stage2_solo", f"{backbone} Stage2 Solo (Acc: {section['accuracy']:.2f}%)")

    run_df = pd.DataFrame(run_rows)
    eval_df = pd.DataFrame(eval_rows)
    epoch_cols = ["Model", "Component", "Epoch", "Training Loss", "Val Event Precision",
                  "Val Event Recall", "Val Event F1", "Val Loss"]
    ep_all = pd.concat(epoch_frames, ignore_index=True) if epoch_frames else pd.DataFrame(columns=epoch_cols)
    if not ep_all.empty:
        ep_all = ep_all.reindex(columns=[c for c in epoch_cols if c in ep_all.columns])

    comparison_df = build_comparison(eval_df)

    if not emit_outputs:
        return run_df, eval_df, comparison_df

    run_df.to_csv(runs_csv, index=False)
    ep_all.to_csv(epochs_csv, index=False)
    eval_df.to_csv(eval_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)

    if not ep_all.empty:
        save_epoch_f1_plot(ep_all, f1_png)
        save_epoch_loss_plot(ep_all, loss_png)
    if not comparison_df.empty:
        save_comparison_plot(comparison_df, comparison_png)
    save_stage_cascade_comparison_plot(eval_df, run_df, stage_comparison_png)

    print(f"processed -> {runs_csv}")
    if not comparison_df.empty:
        print(f"\nJoint vs. Cascade comparison ({stem}):")
        cols = [c for c in ["Model", "Joint_Accuracy", "Joint_Macro_F1", "Joint_Min_Thrust_Recall",
                             "Cascade_EndToEnd_Accuracy", "Cascade_EndToEnd_Macro_F1",
                             "Cascade_EndToEnd_Min_Thrust_Recall", "Better_By_Macro_F1"]
                if c in comparison_df.columns]
        print(comparison_df[cols].to_string(index=False))

    return run_df, eval_df, comparison_df


# ───── Group processing ─────
def process_group_dir(group_dir: Path, root: Path, force: bool, group_name: str | None,
                       emit_per_log: bool) -> Path:
    if not group_dir.is_dir():
        raise ValueError(f"--group-dir must be a directory: {group_dir}")

    logs = list(group_dir.rglob("*.log"))
    print(f"[group] found {len(logs)} logs under {group_dir}")
    if not logs:
        raise ValueError(f"No logs found in group dir: {group_dir}")

    group_rel = group_dir.relative_to(root)
    base_dir = Path("parsed_data") / group_rel / "_group"
    csv_dir = base_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{group_name}" if group_name else ""
    eval_out = csv_dir / f"eval{suffix}.csv"
    comparison_out = csv_dir / f"comparison{suffix}.csv"

    all_eval: List[pd.DataFrame] = []
    all_comparison: List[pd.DataFrame] = []
    for lg in logs:
        _run_df, eval_df, comparison_df = process_log(lg, root, force=force, emit_outputs=emit_per_log)
        if not eval_df.empty:
            all_eval.append(eval_df)
        if not comparison_df.empty:
            comparison_df = comparison_df.copy()
            comparison_df.insert(0, "log_relpath", str(lg.relative_to(root)))
            comparison_df.insert(0, "log_stem", lg.stem)
            all_comparison.append(comparison_df)

    eval_all = pd.concat(all_eval, ignore_index=True) if all_eval else pd.DataFrame()
    comparison_all = pd.concat(all_comparison, ignore_index=True) if all_comparison else pd.DataFrame()
    eval_all.to_csv(eval_out, index=False)
    comparison_all.to_csv(comparison_out, index=False)
    print(f"[group] wrote combined eval -> {eval_out}")
    print(f"[group] wrote combined comparison -> {comparison_out}")
    return comparison_out


# ───── CLI ─────
def main() -> None:
    ap = argparse.ArgumentParser(description="Parse sequence-classification training logs -> CSV + plots")
    ap.add_argument("root", type=Path, help="Root folder that contains logs (used for relative paths).")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs.")

    ap.add_argument("--group-dir", type=Path,
                     help="Process only this directory and emit ONE combined eval/comparison CSV for all *.log within.")
    ap.add_argument("--group-name", type=str, default=None,
                     help="Optional name suffix for the combined output files.")
    ap.add_argument("--emit-per-log", action="store_true",
                     help="When using --group-dir, also emit the usual per-log CSVs/plots.")

    args = ap.parse_args()

    if not args.root.exists():
        raise SystemExit(f"root not found: {args.root}")
    if args.group_dir is not None:
        try:
            _ = args.group_dir.resolve().relative_to(args.root.resolve())
        except Exception:
            raise SystemExit(f"--group-dir must be inside ROOT: {args.group_dir} not under {args.root}")

        process_group_dir(args.group_dir, args.root, force=args.force,
                           group_name=args.group_name, emit_per_log=args.emit_per_log)
        return

    logs = list(args.root.rglob("*.log"))
    print(f"found {len(logs)} logs under {args.root}")
    for lg in logs:
        process_log(lg, args.root, force=args.force, emit_outputs=True)


if __name__ == "__main__":
    main()
