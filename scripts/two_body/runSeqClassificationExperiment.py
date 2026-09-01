# Staged tuning experiment for in-sequence (per-timestep) thrust classification.
#
# A full grid over features x imbalance x capacity x optimizer is thousands of fits, and a fit
# costs ~4 minutes (LSTM) to ~8 minutes (InceptionTime) per 100 epochs on CPU at leo/30min. This
# runs the factors in stages instead, ordered by measured effect size, fixing each stage's winner
# before starting the next. Roughly 145 fits, ~14h CPU, instead of ~3000.
#
# The ordering is not arbitrary -- it comes from effects measured on this dataset:
#     feature conditioning (--standardize)   event F1  0.00 -> 0.94
#     capacity                               event F1  0.59 -> 0.86
#     init-seed noise alone                            +/- 0.13
# A factor whose effect is smaller than the noise floor cannot be resolved, which is why stage 0
# measures that floor before anything is swept.
#
#   0 noise      One config, N_init x N_split fits. Yields sigma_init and sigma_split. Every later
#                stage promotes a winner only if it beats the runner-up by more than
#                2*sqrt(sigma_init^2 + sigma_split^2). Skipping this is how a 145-fit campaign
#                ends up reporting seed noise as findings.
#   1 features   {ECI, OE} x {none, energy, energyRate, both}. One-factor-at-a-time is adequate:
#                the effects here are large relative to the floor.
#   2 imbalance  loss-scheme x focal-gamma x oversample, FULL FACTORIAL -- these interact by
#                construction (--oversample and --loss-scheme both compensate for the same class
#                skew), so a one-at-a-time pass would misattribute their joint effect.
#   3 capacity   hidden_factor / n_filters per backbone. Same grid the standalone sweep script
#                uses; this stage exists so its winner feeds stages 4-6.
#   4 optimizer  lr x num_layers x batch_size. All three are hardcoded in the main script's main()
#                (lr=1e-3, num_layers=1, batch_size=16) and have never been tuned.
#   5 arch       Backbone comparison at MATCHED parameter count, so an architecture result cannot
#                be a parameter-budget result wearing a disguise.
#   6 confirm    Winner re-fit on split seeds never used during tuning, then a single test-set
#                evaluation. Most apparent gains die here; that is the point of the stage.
#
# Rules enforced in code rather than left to discipline:
#   - Test data is untouched until stage 6. Stages 0-5 read validation only.
#   - Within a stage the IC split is fixed and only the init seed varies, so a comparison measures
#     the factor rather than which orbits landed in which split. Stage 0 and stage 6 are the
#     deliberate exceptions -- they vary the split precisely to measure/confirm its effect.
#   - Ties promote the CHEAPER configuration, not the higher mean. Across ~145 fits the argmax is
#     reliably noise; the one-standard-error habit is what keeps the campaign honest.
#   - Every fit appends to results.csv immediately and completed fits are skipped on re-run, so an
#     interrupted campaign resumes by re-invoking the same command.
#
# Per-class F1 is recorded alongside the macro, always. Impulsive is one frame per trajectory --
# an onset-detection problem structurally unlike Chemical/Electric's sustained burns -- and a
# macro average lets a model that never predicts it at all look respectable.
#
# $ python scripts/two_body/runSeqClassificationExperiment.py \
#     --orbit leo --propMin 30 --systems 1500 --mode stage1
# $ python scripts/two_body/runSeqClassificationExperiment.py --stages 3 4 5   # resume midway
import argparse
import contextlib
import csv
import io
import itertools
import json
import os
import sys

parser = argparse.ArgumentParser(description="Staged tuning experiment for per-timestep thrust classification.")
parser.add_argument("--orbit", type=str, default="leo")
parser.add_argument("--propMin", type=int, default=30)
parser.add_argument("--systems", type=int, default=1500)
parser.add_argument("--mode", type=str, default="stage1", choices=["joint", "stage1", "stage2"],
                    help="Tune on stage1 by default: it has the most positive frames and isolates "
                         "detection from typing. Tuning on 'joint' lets Electric's ~78%% frame share "
                         "dominate the gradient and conflates two different decisions.")
parser.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6],
                    help="Which stages to run. Later stages read earlier winners from best.json, so "
                         "a partial run requires that earlier stages completed at some point.")
parser.add_argument("--backbone", type=str, default="lstm", choices=["lstm", "mamba", "transformer", "cnn"],
                    help="Backbone used for stages 0-2 and 4 (the cheapest that clears the floor check). "
                         "Stages 3/5 sweep backbones explicitly.")
parser.add_argument("--arch-backbones", type=str, nargs="+", default=["lstm", "mamba", "cnn"],
                    help="Backbones compared in stages 3 and 5")
parser.add_argument("--init-seeds", type=int, default=5, help="Weight-init seeds per configuration")
parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="IC-split seeds for stage 0's variance estimate. Element 0 is the split used "
                         "for every tuning stage.")
parser.add_argument("--holdout-split-seeds", type=int, nargs="+", default=[7, 8, 9],
                    help="Splits reserved for stage 6, never used during tuning")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--hidden-factors", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
parser.add_argument("--cnn-filters", type=int, nargs="+", default=[4, 8, 16, 32, 64])
parser.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
parser.add_argument("--layers", type=int, nargs="+", default=[1, 2])
parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64])
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

# Import the main script for its pipeline. It parses argv at import time, so swap in a synthetic
# argv matching ITS parser first (see the sweep script for the same dance).
_real_argv = sys.argv
sys.argv = ["mambaTimeSeriesSeqClassificationGMATThrusts.py",
            "--orbit", args.orbit, "--propMin", str(args.propMin), "--systems", str(args.systems),
            "--mode", "joint", "--no-classic"]
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mambaTimeSeriesSeqClassificationGMATThrusts as M
finally:
    sys.argv = _real_argv

import numpy as np
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support

# Line-buffer stdout so a campaign redirected to a log file reports progress as it happens rather
# than in one block at the end -- these runs last hours and are normally watched through a tail.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

MODE_NUM_CLASSES = {"joint": 4, "stage1": 2, "stage2": 3}
MODE_CLASS_NAMES = {"joint": M.JOINT_CLASS_NAMES, "stage1": M.STAGE1_CLASS_NAMES,
                    "stage2": M.STAGE2_CLASS_NAMES}
PAD_IDX = -100
NUM_CLASSES = MODE_NUM_CLASSES[args.mode]

CSV_FIELDS = ["stage", "config_id", "split_seed", "init_seed", "backbone", "params",
              "val_event_macro_f1", "per_class_f1", "train_seconds", "config_json"]

# The configuration carried between stages. Each stage overwrites only the keys it tunes, so the
# campaign narrows one factor group at a time from a documented starting point.
DEFAULT_CONFIG = {
    "useOE": False, "useEnergy": False, "useEnergyRate": False,
    "useNorm": False, "useNoise": False, "standardize": True,
    "loss_scheme": "inverse", "focal_gamma": 0.0, "oversample": False,
    "backbone": args.backbone, "hidden_factor": 8, "n_filters": 32, "cnn_depth": 6,
    "lr": 1e-3, "num_layers": 1, "batch_size": 16,
}


# ---------------------------------------------------------------------------
# Data preparation, cached by (feature flags, batch size, split seed). The OE conversion is a
# Python double loop over every (IC, timestep) and costs ~6s per call at 1500x30 -- with ~145 fits
# that is real time, and stages 2/4 vary nothing that affects the features at all.
# ---------------------------------------------------------------------------
_data_cache = {}


def prepareData(cfg, split_seed):
    # oversample belongs in the key even though it changes no feature value: it swaps the
    # train_loader's sampler for a WeightedRandomSampler. Omitting it would let stage 2's
    # oversample=True configurations silently reuse a cached non-oversampled loader and report
    # that oversampling does nothing.
    key = (cfg["useOE"], cfg["useEnergy"], cfg["useEnergyRate"], cfg["useNorm"],
           cfg["useNoise"], cfg["standardize"], cfg["batch_size"], cfg["oversample"], split_seed)
    if key in _data_cache:
        return _data_cache[key]

    with open("data.yaml") as f:
        data_config = yaml.safe_load(f)
    yaml_config = {
        "useOE": cfg["useOE"], "useNorm": cfg["useNorm"], "useNoise": cfg["useNoise"],
        "useEnergy": cfg["useEnergy"], "useEnergyRate": cfg["useEnergyRate"],
        "prop_time": args.propMin, "orbit": args.orbit, "systems": args.systems,
        "test_dataset": args.orbit, "test_systems": args.systems,
    }
    with quiet(not args.verbose):
        out = M.prepareInSequenceThrustClassificationDatasets(
            yaml_config, data_config, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            batch_size=cfg["batch_size"], seed=split_seed, supress_print=True,
            oversample=cfg["oversample"], standardize=cfg["standardize"])
    # Cache is bounded: only a handful of distinct feature/split combinations ever occur.
    _data_cache[key] = out
    return out


@contextlib.contextmanager
def quiet(enabled):
    """train_model prints per-epoch lines outside its verbose guard; without this the campaign log
    is buried under tens of thousands of them."""
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def makeModel(cfg, input_size, num_classes):
    if cfg["backbone"] == "cnn":
        return M.InceptionTimeSequenceClassifier(input_size, num_classes,
                                                 n_filters=cfg["n_filters"], depth=cfg["cnn_depth"])
    hidden_size = input_size * cfg["hidden_factor"]
    return M.build_model(cfg["backbone"], num_classes, input_size, hidden_size, cfg["num_layers"])


def eventMacroF1(y_true, y_pred):
    event_labels, _ = M._default_label_sets(args.mode, NUM_CLASSES)
    if len(y_true) == 0:
        return 0.0, np.zeros(len(event_labels))
    _, _, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=event_labels,
                                                       average="macro", zero_division=0)
    _, _, f_per, _ = precision_recall_fscore_support(y_true, y_pred, labels=event_labels,
                                                     average=None, zero_division=0)
    return float(f_macro), f_per


def runFit(cfg, split_seed, init_seed, evaluate_on="val"):
    """One fit. Returns (params, macro_f1, per_class_f1, seconds). evaluate_on='test' is reachable
    only from stage 6 -- every other stage reads validation, so the test split stays unseen while
    selection is happening."""
    (train_loader, val_loader, test_loader, train_data, *_rest) = prepareData(cfg, split_seed)
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)

    model = makeModel(cfg, train_data.shape[2], NUM_CLASSES)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eval_loader = test_loader if evaluate_on == "test" else val_loader

    with quiet(not args.verbose):
        seconds = M.train_model(
            model, train_loader, val_loader, num_epochs=args.epochs, num_classes=NUM_CLASSES,
            mode=args.mode, pad_idx=PAD_IDX, verbose=False,
            loss_scheme=cfg["loss_scheme"], focal_gamma=cfg["focal_gamma"],
            # Checkpoint on the same metric this campaign ranks configurations by. Restoring the
            # lowest-val-loss epoch instead measurably picks a worse-F1 model under this dataset's
            # imbalance, which would put every stage's promotion decision at odds with its metric.
            lr=cfg["lr"], restore_best=True, restore_metric="event_f1")
        res = M.validateInSequenceClassifier(
            model, eval_loader, mode=args.mode, num_classes=NUM_CLASSES, device=M.device,
            pad_idx=PAD_IDX, class_names=MODE_CLASS_NAMES[args.mode],
            print_report=False, return_predictions=True)

    f1, per_class = eventMacroF1(res["y_true"], res["y_pred"])
    return params, f1, per_class, seconds


# ---------------------------------------------------------------------------
# Result store. Append-only CSV keyed by (stage, config_id, split_seed, init_seed) so re-running
# skips completed fits.
# ---------------------------------------------------------------------------
class Results:
    def __init__(self, path):
        self.path = path
        self.done = set()
        if os.path.exists(path):
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    self.done.add((r["stage"], r["config_id"], int(r["split_seed"]), int(r["init_seed"])))
        else:
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def has(self, stage, cid, split_seed, init_seed):
        return (str(stage), cid, split_seed, init_seed) in self.done

    def add(self, stage, cid, split_seed, init_seed, cfg, params, f1, per_class, seconds):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "stage": stage, "config_id": cid, "split_seed": split_seed, "init_seed": init_seed,
                "backbone": cfg["backbone"], "params": params,
                "val_event_macro_f1": f"{f1:.6f}",
                "per_class_f1": json.dumps([round(float(x), 6) for x in per_class]),
                "train_seconds": f"{seconds:.2f}" if seconds is not None else "",
                "config_json": json.dumps(cfg, sort_keys=True),
            })
        self.done.add((str(stage), cid, split_seed, init_seed))

    def scores(self, stage, cid):
        out = []
        with open(self.path, newline="") as f:
            for r in csv.DictReader(f):
                if r["stage"] == str(stage) and r["config_id"] == cid:
                    out.append(float(r["val_event_macro_f1"]))
        return np.array(out)

    def paramsFor(self, stage, cid):
        with open(self.path, newline="") as f:
            for r in csv.DictReader(f):
                if r["stage"] == str(stage) and r["config_id"] == cid:
                    return int(r["params"])
        return 0


def runGrid(stage, candidates, results, split_seeds=None, init_seeds=None, evaluate_on="val"):
    """candidates: [(config_id, cfg), ...]. Runs every (config, split, init) combination not
    already recorded and returns {config_id: array_of_scores}."""
    split_seeds = split_seeds if split_seeds is not None else [args.split_seeds[0]]
    init_seeds = init_seeds if init_seeds is not None else list(range(args.init_seeds))
    total = len(candidates) * len(split_seeds) * len(init_seeds)
    print(f"  {len(candidates)} configs x {len(split_seeds)} split x {len(init_seeds)} init "
          f"= {total} fits")

    for cid, cfg in candidates:
        for ss in split_seeds:
            for iseed in init_seeds:
                if results.has(stage, cid, ss, iseed):
                    continue
                params, f1, per_class, secs = runFit(cfg, ss, iseed, evaluate_on=evaluate_on)
                results.add(stage, cid, ss, iseed, cfg, params, f1, per_class, secs)
                print(f"    {cid:<38} split={ss} init={iseed} F1={f1:.4f} "
                      f"per-class={[round(float(x), 3) for x in per_class]}")
    return {cid: results.scores(stage, cid) for cid, _ in candidates}


def pickWinner(stage, scores, candidates, results, threshold, cheaper_key):
    """Promotes the best mean only when it clears the runner-up by more than the stage-0 noise
    threshold; otherwise promotes the cheapest configuration among those statistically tied with
    the leader. cheaper_key ranks 'cheap' for the tie case (params, or a complexity proxy)."""
    means = {cid: (s.mean() if len(s) else 0.0) for cid, s in scores.items()}
    ordered = sorted(means.items(), key=lambda kv: -kv[1])
    best_cid, best_mean = ordered[0]

    tied = [cid for cid, m in means.items() if m >= best_mean - threshold]
    cfg_by_id = dict(candidates)
    if len(tied) > 1:
        chosen = min(tied, key=lambda c: cheaper_key(c, cfg_by_id[c], results, stage))
        note = (f"{len(tied)} configs within the {threshold:.4f} noise threshold of the leader "
                f"({best_cid}, {best_mean:.4f}); promoting the cheapest")
    else:
        chosen, note = best_cid, "clear winner (beats the field by more than the noise threshold)"

    print(f"\n  ranking:")
    for cid, m in ordered:
        mark = "  <-- promoted" if cid == chosen else ("  (tied)" if cid in tied else "")
        sd = scores[cid].std(ddof=1) if len(scores[cid]) > 1 else 0.0
        print(f"    {cid:<38} {m:.4f} +/- {sd:.4f}{mark}")
    print(f"  {note}")
    return chosen, cfg_by_id[chosen]


def byParams(cid, cfg, results, stage):
    return results.paramsFor(stage, cid)


def byComplexity(cid, cfg, results, stage):
    """Fewer enabled feature/imbalance knobs is 'cheaper' -- prefers the simpler pipeline when two
    are statistically indistinguishable."""
    return sum([cfg["useOE"], cfg["useEnergy"], cfg["useEnergyRate"], cfg["useNorm"],
                cfg["oversample"], cfg["focal_gamma"] > 0, cfg["loss_scheme"] == "effective"])


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage0_noise(cfg, results):
    """Init-seed and split-seed variance for one fixed configuration. sigma_init is the spread
    across weight inits at a fixed split; sigma_split is the spread of per-split means. They add in
    quadrature into the threshold every later stage compares against."""
    print(f"\n{'='*80}\nStage 0 -- noise floor\n{'='*80}")
    cands = [("baseline", cfg)]
    runGrid(0, cands, results, split_seeds=args.split_seeds)

    per_split = []
    with open(results.path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["stage"] == "0"]
    for ss in args.split_seeds:
        vals = [float(r["val_event_macro_f1"]) for r in rows if int(r["split_seed"]) == ss]
        if vals:
            per_split.append(np.array(vals))

    # A std needs >=2 samples. With --init-seeds 1 no split contributes one, and averaging an
    # empty list yields NaN -- which would propagate into the threshold and make every later
    # `mean >= best - threshold` test False, silently degrading every promotion to a plain argmax
    # and discarding the entire one-standard-error discipline. Fail loudly instead.
    init_stds = [v.std(ddof=1) for v in per_split if len(v) > 1]
    sigma_init = float(np.mean(init_stds)) if init_stds else float("nan")
    split_means = np.array([v.mean() for v in per_split])
    sigma_split = float(split_means.std(ddof=1)) if len(split_means) > 1 else float("nan")

    if not init_stds:
        print(f"\n  [ERROR] --init-seeds is {args.init_seeds}; at least 2 are required to estimate "
              f"init variance. Without it there is no noise threshold, every stage would promote "
              f"a bare argmax, and the campaign would report seed noise as findings.")
        return None
    if len(split_means) < 2:
        print(f"\n  [warn] only one split seed given, so split variance is unmeasured and the "
              f"threshold covers init noise alone. It will be too permissive. Pass at least two "
              f"--split-seeds for a defensible threshold.")
        sigma_split = 0.0

    threshold = 2.0 * float(np.sqrt(sigma_init ** 2 + sigma_split ** 2))

    print(f"\n  sigma_init  (weight init, fixed split) = {sigma_init:.4f}")
    print(f"  sigma_split (across IC splits)         = {sigma_split:.4f}")
    print(f"  promotion threshold 2*sqrt(si^2+ss^2)  = {threshold:.4f}")
    print(f"  -> differences smaller than {threshold:.4f} event-F1 are not resolvable with "
          f"{args.init_seeds} seeds and will not be promoted.")
    if sigma_split > 2 * max(sigma_init, 1e-9):
        print(f"  [note] split variance dominates init variance. Single-split comparisons in later "
              f"stages are correspondingly weak evidence; widen --init-seeds or report stage 6 "
              f"across more holdout splits.")
    return threshold


def stage1_features(cfg, results, threshold):
    print(f"\n{'='*80}\nStage 1 -- feature representation\n{'='*80}")
    cands = []
    for use_oe in (False, True):
        for en, rate in ((False, False), (True, False), (False, True), (True, True)):
            c = dict(cfg, useOE=use_oe, useEnergy=en, useEnergyRate=rate)
            cid = f"{'OE' if use_oe else 'ECI'}+{'E' if en else ''}{'R' if rate else ''}" or "base"
            cands.append((cid.rstrip("+"), c))
    scores = runGrid(1, cands, results)
    _, best = pickWinner(1, scores, cands, results, threshold, byComplexity)
    return best


def stage2_imbalance(cfg, results, threshold):
    print(f"\n{'='*80}\nStage 2 -- class-imbalance handling (full factorial)\n{'='*80}")
    cands = []
    for scheme, gamma, over in itertools.product(("inverse", "effective"), (0.0, 2.0), (False, True)):
        c = dict(cfg, loss_scheme=scheme, focal_gamma=gamma, oversample=over)
        cands.append((f"{scheme}_g{gamma:g}_{'os' if over else 'noos'}", c))
    scores = runGrid(2, cands, results)
    _, best = pickWinner(2, scores, cands, results, threshold, byComplexity)
    return best


def stage3_capacity(cfg, results, threshold):
    print(f"\n{'='*80}\nStage 3 -- capacity, per backbone\n{'='*80}")
    winners = {}
    for backbone in args.arch_backbones:
        cands = []
        if backbone == "cnn":
            for f in args.cnn_filters:
                if (f * 4) % 8:
                    continue
                cands.append((f"cnn_f{f}", dict(cfg, backbone=backbone, n_filters=f)))
        else:
            for hf in args.hidden_factors:
                cands.append((f"{backbone}_h{hf}", dict(cfg, backbone=backbone, hidden_factor=hf)))
        print(f"\n  -- {backbone} --")
        scores = runGrid(3, cands, results)
        cid, best = pickWinner(3, scores, cands, results, threshold, byParams)
        winners[backbone] = (cid, best, scores[cid].mean())
    return winners


def stage4_optimizer(cfg, results, threshold):
    print(f"\n{'='*80}\nStage 4 -- optimizer (lr / depth / batch)\n{'='*80}")
    cands = []
    for lr, nl, bs in itertools.product(args.lrs, args.layers, args.batch_sizes):
        c = dict(cfg, lr=lr, num_layers=nl, batch_size=bs)
        cands.append((f"lr{lr:g}_l{nl}_b{bs}", c))
    scores = runGrid(4, cands, results)
    _, best = pickWinner(4, scores, cands, results, threshold,
                         lambda cid, c, r, s: (c["num_layers"], c["batch_size"]))
    return best


def stage5_architecture(stage3_winners, results, threshold):
    """Compares backbones at matched parameter count. Reports the comparison as unusable when the
    grids do not overlap within 2x, rather than presenting nearest-available as if it were
    matched."""
    print(f"\n{'='*80}\nStage 5 -- architecture at matched capacity\n{'='*80}")
    if len(stage3_winners) < 2:
        print("  fewer than two backbones swept; nothing to compare")
        return None

    target = int(np.median([results.paramsFor(3, cid) for cid, _, _ in stage3_winners.values()]))
    print(f"  target parameter count (median of stage-3 winners): {target:,}\n")
    print(f"  {'backbone':<12} {'config':<20} {'params':>10} {'mean F1':>9}  note")
    best = None
    for backbone, (cid, cfg_b, mean_f1) in stage3_winners.items():
        params = results.paramsFor(3, cid)
        ratio = params / max(1, target)
        note = "" if 0.5 <= ratio <= 2.0 else f"NOT MATCHED ({ratio:.2f}x)"
        print(f"  {backbone:<12} {cid:<20} {params:>10,} {mean_f1:>9.4f}  {note}")
        if best is None or mean_f1 > best[2]:
            best = (backbone, cfg_b, mean_f1)
    print(f"\n  leading backbone: {best[0]} ({best[2]:.4f})")
    print(f"  Rows flagged NOT MATCHED cannot support an architecture claim -- extend that "
          f"backbone's capacity grid until the parameter ranges overlap.")
    return best[1]


def stage6_confirm(cfg, results, threshold, tuned_val_mean):
    """Re-fits the selected configuration on splits never used during tuning, then evaluates the
    test set exactly once. A drop larger than the noise threshold means the campaign selected on
    quirks of the tuning split rather than on a real effect."""
    print(f"\n{'='*80}\nStage 6 -- confirmation on held-out splits\n{'='*80}")
    cands = [("winner", cfg)]
    scores = runGrid(6, cands, results, split_seeds=args.holdout_split_seeds,
                     init_seeds=list(range(args.init_seeds)))
    holdout = scores["winner"]
    print(f"\n  tuning-split val mean : {tuned_val_mean:.4f}")
    print(f"  holdout-split val mean: {holdout.mean():.4f} +/- "
          f"{holdout.std(ddof=1) if len(holdout) > 1 else 0.0:.4f}")
    drop = tuned_val_mean - holdout.mean()
    if drop > threshold:
        print(f"  [warn] {drop:.4f} drop exceeds the {threshold:.4f} noise threshold -- the "
              f"selection is partly an artifact of the tuning split. Treat the tuned numbers as "
              f"optimistic and prefer the holdout figure when reporting.")
    else:
        print(f"  drop {drop:.4f} is within the noise threshold; the selection holds up.")

    print(f"\n  --- single test-set evaluation (first use of test data) ---")
    params, f1, per_class, _ = runFit(cfg, args.holdout_split_seeds[0], 0, evaluate_on="test")
    names = MODE_CLASS_NAMES[args.mode]
    event_labels, _ = M._default_label_sets(args.mode, NUM_CLASSES)
    print(f"  test event macro-F1: {f1:.4f}  ({params:,} params)")
    for lbl, v in zip(event_labels, per_class):
        print(f"    {names[lbl]:<12} F1={v:.4f}")
    return f1, per_class


def main():
    tag = f"{args.orbit}_{args.propMin}min{args.systems}_{args.mode}"
    out_dir = args.out or os.path.join("plots", "experiment", tag)
    os.makedirs(out_dir, exist_ok=True)
    results = Results(os.path.join(out_dir, "results.csv"))
    best_path = os.path.join(out_dir, "best.json")

    state = {"config": dict(DEFAULT_CONFIG), "threshold": None, "stage3": None,
             "tuned_val_mean": None}
    if os.path.exists(best_path):
        with open(best_path) as f:
            state.update(json.load(f))
        print(f"Resuming from {best_path}")

    def save():
        with open(best_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    print(f"{'='*80}\nStaged tuning -- {args.orbit} {args.propMin}min/{args.systems}sys, "
          f"mode={args.mode}\n{'='*80}")
    print(f"Tuning split seed: {args.split_seeds[0]}   holdout splits: {args.holdout_split_seeds}")
    print(f"Test data is not read until stage 6.")

    cfg = state["config"]

    if 0 in args.stages:
        state["threshold"] = stage0_noise(cfg, results)
        save()
    if state["threshold"] is None or not np.isfinite(float(state["threshold"])):
        print("\nNo usable noise threshold. Run stage 0 with --init-seeds >= 2 and at least two "
              "--split-seeds before running any tuning stage.")
        return 1
    threshold = float(state["threshold"])

    if 1 in args.stages:
        cfg = state["config"] = stage1_features(cfg, results, threshold)
        save()
    if 2 in args.stages:
        cfg = state["config"] = stage2_imbalance(cfg, results, threshold)
        save()
    if 3 in args.stages:
        winners = stage3_capacity(cfg, results, threshold)
        state["stage3"] = {b: [cid, c, float(m)] for b, (cid, c, m) in winners.items()}
        best_b = max(winners.items(), key=lambda kv: kv[1][2])
        cfg = state["config"] = best_b[1][1]
        state["tuned_val_mean"] = float(best_b[1][2])
        save()
    if 4 in args.stages:
        cfg = state["config"] = stage4_optimizer(cfg, results, threshold)
        save()
    if 5 in args.stages and state.get("stage3"):
        w = {b: (v[0], v[1], v[2]) for b, v in state["stage3"].items()}
        chosen = stage5_architecture(w, results, threshold)
        if chosen:
            # keep stage-4 optimizer settings, adopt the winning architecture
            cfg = state["config"] = dict(chosen, lr=cfg["lr"], num_layers=cfg["num_layers"],
                                         batch_size=cfg["batch_size"])
            save()
    if 6 in args.stages:
        tuned = state.get("tuned_val_mean") or 0.0
        f1, per_class = stage6_confirm(cfg, results, threshold, float(tuned))
        state["test_event_macro_f1"] = f1
        state["test_per_class_f1"] = [float(x) for x in per_class]
        save()

    print(f"\n{'='*80}\nFinal configuration\n{'='*80}")
    for k, v in sorted(state["config"].items()):
        print(f"  {k:<16} {v}")
    print(f"\nresults.csv and best.json in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
