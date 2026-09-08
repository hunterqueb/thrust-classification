# Standalone analysis tool for choosing --smooth-max-gap (see smoothPerTimestepGrid,
# _reportEventLevelWithSmoothing in mambaTimeSeriesSeqClassificationGMATThrusts.py).
#
# Gap-closing is pure post-processing on an already-computed [N,T] prediction grid, so sweeping it
# by re-running the main script once per gap value would be both wasteful (retraining a fresh model
# every time) AND invalid (train_model has no fixed seed, so different gap values would silently be
# scored against DIFFERENT trained models rather than the same one). This script instead trains
# ONE joint 4-class model, captures its raw validation-split prediction grid once via
# _predictPerTimestepNeural, and sweeps smoothPerTimestepGrid directly on that fixed grid --
# cheap (no retraining per gap value) and apples-to-apples (every gap value is scored against
# identical underlying predictions).
#
# Prints a Gap x Class table of event-level recall/precision (see _eventLevelReport) and, with
# --plot, saves a per-class recall/precision-vs-gap figure. What to look for: Electric's
# recall/precision should climb and then plateau as the gap size approaches its ~10-minute burst
# duration; Chemical's recall should stay flat throughout (the gap-closing design's core safety
# property -- it never erases a predicted-positive run, so an isolated true Chemical detection is
# never at risk); the best gap is typically right where Electric's curve plateaus, before larger
# gaps start bridging genuinely separate events (visible as Chemical/Impulsive precision dropping).
#
# $ python scripts/two_body/sweepSmoothGap.py --orbit leo --propMin 30 --systems 1500 --backbone lstm --max-gap 15
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Sweep --smooth-max-gap on one trained model's fixed prediction grid.")
parser.add_argument("--orbit", type=str, default="leo")
parser.add_argument("--propMin", type=int, default=30)
parser.add_argument("--systems", type=int, default=1500)
parser.add_argument("--backbone", type=str, default="lstm", choices=["lstm", "mamba", "transformer", "cnn"])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--max-gap", type=int, default=15, dest="max_gap", help="Sweep max_gap from 0 to this value inclusive")
parser.add_argument("--seed", type=int, default=0, help="Split seed (data) and init seed (model weights)")
parser.add_argument("--no-standardize", dest="standardize", action="store_false",
                     help="Disable z-scoring features (on by default here -- unlike the main script's own "
                          "off-by-default, this tool has no prior results to stay reproducible with, and "
                          "runSeqClassificationExperiment.py's tuning campaign measured --standardize's "
                          "effect on this exact task at event F1 0.00 -> 0.94, so leaving it off by default "
                          "would make every other number in this sweep meaningless)")
parser.set_defaults(standardize=True)
parser.add_argument("--oversample", action="store_true",
                     help="Oversample rarer per-timestep classes in train_loader (see main script's --oversample)")
parser.add_argument("--loss-scheme", type=str, default="inverse", choices=["effective", "inverse"],
                     dest="loss_scheme", help="Class-weighting scheme for CE loss (default matches the main "
                                               "script's own default -- combining --oversample with "
                                               "'effective' stacks two imbalance corrections and can "
                                               "overcorrect into an all-NoThrust collapse)")
parser.add_argument("--plot", action="store_true", help="Save a recall/precision-vs-gap figure per class")
parser.add_argument("--verbose", action="store_true", help="Show per-epoch training output")
args = parser.parse_args()

# Import the main script for its pipeline. It parses argv at import time, so swap in a synthetic
# argv matching ITS parser first (see runSeqClassificationExperiment.py for the same dance).
_real_argv = sys.argv
sys.argv = ["mambaTimeSeriesSeqClassificationGMATThrusts.py",
            "--orbit", args.orbit, "--propMin", str(args.propMin), "--systems", str(args.systems),
            "--mode", "joint", "--no-classic"]
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mambaTimeSeriesSeqClassificationGMATThrusts as M
finally:
    sys.argv = _real_argv

import contextlib
import io

import numpy as np
import torch
import yaml


@contextlib.contextmanager
def quiet(enabled):
    """train_model prints per-epoch lines outside its verbose guard."""
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def main():
    with open("data.yaml") as f:
        data_config = yaml.safe_load(f)
    yaml_config = {
        "useOE": False, "useNorm": False, "useNoise": False, "useEnergy": False, "useEnergyRate": False,
        "numSinusoids": 0, "usePhysicsLoss": False,
        "prop_time": args.propMin, "orbit": args.orbit, "systems": args.systems,
        "test_dataset": args.orbit, "test_systems": args.systems,
    }
    (train_loader, val_loader, test_loader, train_data, *_rest) = M.prepareInSequenceThrustClassificationDatasets(
        yaml_config, data_config, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        batch_size=16, seed=args.seed, standardize=args.standardize, oversample=args.oversample)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_size = train_data.shape[2]
    hidden_size = int(input_size * 8)
    model = M.build_model(args.backbone, 4, input_size, hidden_size, num_layers=1)

    print(f"Training {args.backbone.upper()} joint 4-class model on {args.orbit}/{args.propMin}min-{args.systems} "
          f"for {args.epochs} epochs (seed={args.seed}, oversample={args.oversample}, loss_scheme={args.loss_scheme})...")
    with quiet(not args.verbose):
        # restore_metric='event_f1': checkpointing on val loss measurably selects a worse-F1 model
        # under this dataset's imbalance (see mambaTimeSeriesSeqClassificationGMATThrusts.py's
        # train_model docstring) -- the sweep should read the same model a real run would keep.
        M.train_model(model, train_loader, val_loader, num_epochs=args.epochs, num_classes=4,
                       mode="joint", verbose=False, restore_best=True, restore_metric="event_f1",
                       loss_scheme=args.loss_scheme)

    y_true_grid, _, pred_grid = M._predictPerTimestepNeural(model, val_loader, M.device)

    # Diagnostic: count actual interior NoThrust gap lengths in the RAW grid, so a flat sweep
    # table is distinguishable from "nothing to close" vs. a bug -- without this, both look
    # identical from the table alone.
    gap_lengths = []
    for row in pred_grid:
        for (s, e) in M._findSegments(row, 0):
            if s == 0 or e == row.shape[0] - 1:
                continue
            gap_lengths.append(e - s + 1)
    if gap_lengths:
        import collections
        hist = collections.Counter(gap_lengths)
        print(f"Interior NoThrust gaps in raw predictions: {len(gap_lengths)} total, "
              f"length histogram (length: count) = {dict(sorted(hist.items()))}")
    else:
        print("Interior NoThrust gaps in raw predictions: NONE -- this model's predictions never "
              "flicker off mid-event in this validation split, so no --smooth-max-gap value can "
              "change anything for it (a flat sweep table is therefore expected, not a bug).")

    print(f"\nSweeping gap 0..{args.max_gap} on the fixed prediction grid ({pred_grid.shape[0]} trajectories, "
          f"{pred_grid.shape[1]} timesteps)...\n")
    gap_values = list(range(0, args.max_gap + 1))
    classes = M.JOINT_CLASS_NAMES[1:]  # skip NoThrust, matching _eventLevelReport's own convention
    curves = {c: {"recall": [], "precision": []} for c in classes}

    for g in gap_values:
        smoothed = M.smoothPerTimestepGrid(pred_grid, g)
        metrics = M._eventLevelReport(y_true_grid, smoothed, M.JOINT_CLASS_NAMES, print_report=False)
        for c in classes:
            curves[c]["recall"].append(metrics[c]["recall"])
            curves[c]["precision"].append(metrics[c]["precision"])

    col_w = 13
    header = f"{'Gap':>4} | " + " | ".join(f"{c + ' R/P':^{col_w}}" for c in classes)
    print(header)
    print("-" * len(header))
    for i, g in enumerate(gap_values):
        row = f"{g:>4} | " + " | ".join(
            f"{curves[c]['recall'][i] * 100:5.1f}/{curves[c]['precision'][i] * 100:5.1f}".center(col_w)
            for c in classes)
        print(row)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4), sharex=True, squeeze=False)
        for ax, c in zip(axes[0], classes):
            ax.plot(gap_values, [r * 100 for r in curves[c]["recall"]], label="Recall", marker="o")
            ax.plot(gap_values, [p * 100 for p in curves[c]["precision"]], label="Precision", marker="s")
            ax.set_title(c)
            ax.set_xlabel("smooth-max-gap")
            ax.set_ylabel("%")
            ax.set_ylim(-5, 105)
            ax.legend()
            ax.grid(alpha=0.3)
        fig.suptitle(f"{args.backbone.upper()} joint -- {args.orbit}/{args.propMin}min-{args.systems} "
                     f"-- event-level recall/precision vs. smooth-max-gap")
        fig.tight_layout()

        plot_dir = f"gmat/data/seqClassification/{args.orbit}/{args.propMin}min-{args.systems}/plots"
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"smoothGapSweep_{args.backbone}_{args.orbit}_{args.propMin}min{args.systems}.png")
        fig.savefig(save_path, dpi=150)
        print(f"\nSaved sweep plot -> {save_path}")


if __name__ == "__main__":
    main()
