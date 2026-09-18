# parse at the beginning before long imports
# script usage
#
# In-sequence (per-timestep) thrust classification. At every timestep of a trajectory, classify
# whether thrust is occurring and, if so, which of Chemical/Electric/Impulsive it is. Loads all
# 4 classes (Chemical/Electric/ImpBurn/NoThrust) from {data.yaml classification path}/{orbit}/
# {propMin}min-{systems}/statesArray{Class}.npz, using each file's per-timestep 'thrustingTime'
# array as ground truth (Impulsive is forward-filled from the burn instant to the end of the
# propagation window, since the orbit stays altered afterward; a class file missing
# 'thrustingTime' -- as ImpBurn/NoThrust currently are, pending an upstream GMAT-Thrust-Data fix
# -- degrades to all-background labels with a printed warning instead of crashing).
#
# Two approaches are trained and compared side by side in one run:
#   joint    a single 4-class per-timestep model (0=NoThrust,1=Chemical,2=Electric,3=Impulsive)
#   cascade  a binary "is thrust occurring" detector (stage 1) feeding a 3-class Chemical/
#            Electric/Impulsive type classifier (stage 2, trained only on thrusting timesteps),
#            combined at inference into the same 4-class label space as the joint model so the
#            two approaches' classification reports are directly comparable.
# --mode {all,joint,cascade,stage1,stage2} selects which of these run (default: all).
#
# LSTM and Mamba backbones always run (--no-lstm/--no-mamba disable them); --transformer/--cnn
# opt in to per-timestep Transformer / 1D-CNN (InceptionTime) backbones as additional comparisons
# in the same run. --hybrid opts in to a mixed cascade -- a whole-trajectory MiniRocket stage-1
# detector ("does this ~30-minute window contain thrust anywhere", broadcast across every
# timestep of a trajectory) paired with a CNN (InceptionTime) stage-2 per-timestep type
# classifier -- and only participates in cascade/stage-solo modes (no single joint 4-class
# 'hybrid' model exists).
#
# Classic ML / GBDT baselines (LightGBM on by default via --no-classic to disable; --xgboost/
# --catboost/--rf/--extratrees to opt in) and --mlp (PCA+MLP) operate on Hankel-windowed rows
# (a trailing per-timestep context window, via buildHankelWindowRowsPerTimestep) rather than
# full [B,T,C] sequences -- tree/linear models have no global-pooling requirement, unlike
# MiniRocket, so per-timestep windowed rows are a natural fit for them. --minirocket runs
# MiniRocket the same way: end-to-end per-timestep sequence classification (joint AND cascade)
# over short trailing windows, one shared kernel transform feeding three ridge heads. See
# runMiniRocketPerTimestep for why it is windowed rather than whole-trajectory.
#
# Class imbalance: --loss-scheme/--cb-beta/--focal-gamma reweight the loss (see
# _infer_class_weights); --oversample instead reweights *sampling* -- the neural train_loader
# (LSTM/Mamba/Transformer/CNN) draws whole training trajectories with replacement, biased toward
# trajectories containing rarer per-timestep classes (see _computeOversamplingWeights). Whole
# trajectories, not individual timesteps, are the unit of resampling, since shuffling timesteps
# within a sequence would break the temporal context these backbones need. The two knobs are
# independent and can be combined; classic-ML/PCA+MLP/MiniRocket/hybrid-stage-1 train on raw
# arrays outside train_loader and are unaffected by --oversample.
#
# --j2-energy switches the --energy/--energyRate channels to the J2-INCLUSIVE specific energy
# (qutils.orbital.orbitalEnergyJ2). The Keplerian v^2/2 - mu/r is not conserved under J2 -- energy
# shuttles between it and the J2 potential term every orbit -- so its rate measures that exchange,
# not the perturbing forces you care about. On leo that exchange is ~55x an electric thruster's
# signature and ~1000x atmospheric drag, and removing it takes per-frame Electric-vs-NoThrust ROC
# AUC from 0.52 (chance) to 0.83. Off by default only to keep previously logged --energy results
# reproducible; recommended on for any low-thrust work. See analyzeThrustSeparability.py.
#
# --sinusoids K replaces every feature channel with K dominant sinusoidal components extracted per
# trajectory via FFT (see _decomposeSinusoids): the channel's K largest-magnitude non-DC frequency
# bins, each reconstructed as its own real sinusoid signal, ordered by descending magnitude. C
# channels become C*K -- e.g. raw ECI states (6 channels) with --sinusoids 3 yields 18. Off (0) by
# default; applied after --OE/--energy/--energyRate (whatever channels those leave behind are what
# gets decomposed) and before --standardize.
#
# --physics-loss-weight W (see _computePhysicsResidualTensors, physicsConsistencyLoss) adds an
# auxiliary loss term (--mode joint/stage2) that nudges the model's EXISTING Chemical-vs-Electric
# logit margin toward a physics-derived pseudo-target -- no new head/params. Chemical thrust
# accelerations are typically on the order of LEO's J2 perturbation and Electric on the order of
# the much smaller combined J3-J6 perturbation, so which analytic scale the measured residual sits
# closer to is itself evidence of the type. That coincidence is LEO-specific (J2/J3-J6 fall off
# steeply with altitude while thruster accel does not, so at GEO both thrust types vastly exceed
# all zonal harmonics) -- to stay orbit-regime-agnostic, J2/J3-J6 accelerations are normalized by
# local two-body gravity (mu/r^2) into dimensionless ratios that gate the term's per-timestep
# weight, large in LEO and naturally collapsing toward 0 in GEO, with no regime-specific branching
# in the code. Off (0.0) by default.
#
# --smooth-max-gap N (see smoothPerTimestepGrid, _reportEventLevelWithSmoothing) is a separate,
# purely post-hoc knob -- no retraining, no loss changes. It closes short interior NoThrust gaps
# between two thrusting predictions (bounded on both sides) before event-level reporting/plotting,
# targeting patchy detection within one real burst. It only ever CLOSES gaps and never erases short
# positive runs, so it cannot destroy a correctly-detected short burst the way a majority-vote
# window or minimum-run-length filter could. Applied uniformly to every model family's [N,T]
# prediction grid via one shared function; the raw report always runs first and is never replaced,
# only supplemented, so the payoff is directly comparable. Off (0) by default.
#
# Measured effect (leo/30min-1500, one full training run per backbone): LSTM produced ZERO interior
# gaps at any config tried and CNN only 8 (all length 1), so the flag does nothing for either --
# their local/recurrent context already yields contiguous predictions. Transformer produced 373
# gaps (max length 3) and gained ~0.9pp Impulsive event recall at --smooth-max-gap 3, with Chemical
# and Electric bit-identical. Use scripts/two_body/sweepSmoothGap.py to check any other config: it
# trains once and sweeps gap values on that one fixed prediction grid, and reports the interior-gap
# histogram so a flat sweep is distinguishable from a bug.
#
# $ python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
# --systems 1500 --propMin 30 --orbit vleo --mode all
import argparse

# MiniRocket per-timestep constants. Defined here (not beside the runner) because --minirocket-kernels
# advertises the default in its help text and the strAdd stem block compares against it, both of which
# execute at import time, long before the function definitions below.
MINIROCKET_DEFAULT_KERNELS = 2500   # sktime rounds down to a multiple of 84 -> 2436
MINIROCKET_WINDOW = 9               # MiniRocket's kernels are length 9; shorter windows are rejected
MINIROCKET_CHUNK = 8192             # rows per transform/solve chunk -- caps peak memory, see _ridgeFit

parser = argparse.ArgumentParser()
parser.add_argument('--no-lstm', dest="use_lstm", action='store_false', help='Use LSTM model')
parser.add_argument('--no-mamba', dest="use_mamba", action='store_false', help='Use Mamba model')
parser.add_argument("--systems", type=int, default=1500, help="Number of random systems to access")
parser.add_argument("--propMin", type=int, default=10, help="Minimum propagation time in minutes")
parser.add_argument("--orbit", type=str, default="leo", help="Orbit type: vleo, leo")
parser.add_argument("--test", type=str, default=None, help="Orbit type for test set: vleo, leo, OR the same as --orbit and an integer number of random systems to use for testing")
parser.add_argument("--testSys", type=int, default=1500, help="Number of systems to use for testing if --test is a different string than --orbit")
parser.add_argument("--OE", action='store_true', help="Use OE elements instead of ECI states")
parser.add_argument("--noise", action='store_true', help="Add noise to the data")
parser.add_argument("--velNoise", type=float, default=1e-3, help="std of noise to add to velocity terms")
parser.add_argument("--norm", action='store_true', help="Normalize the semi-major axis by Earth's radius")
parser.add_argument("--one-pass", dest="one_pass", action='store_true', help="Use one pass learning.")
parser.add_argument("--save", dest="save_to_log", action="store_true", help="output console printout to log file in the same location as datasets")
parser.add_argument("--energy", dest="use_energy", action="store_true", help="Use energy as a feature.")
parser.add_argument("--energyRate", dest="use_energy_rate", action="store_true",
                     help="Use per-timestep orbital-energy rate of change (finite difference "
                          "along time) as an additional feature; implies computing energy "
                          "internally even without --energy. Energy is ~constant under "
                          "two-body coasting, so its rate is a direct residual signal for "
                          "thrust occurring.")
parser.add_argument("--j2-energy", dest="use_j2_energy", action="store_true",
                     help="Compute the --energy/--energyRate feature channels from the J2-INCLUSIVE "
                          "specific energy (qutils.orbital.orbitalEnergyJ2) instead of the Keplerian "
                          "v^2/2 - mu/r. Strongly recommended for low-thrust work: the Keplerian energy "
                          "is not conserved under J2, so its rate is dominated by J2 potential exchange "
                          "rather than by any perturbing force. Measured on leo/100min-1500 (see "
                          "scripts/two_body/analyzeThrustSeparability.py), that exchange has 55x the "
                          "amplitude of an electric thruster's signature, and removing it takes "
                          "per-frame Electric-vs-NoThrust ROC AUC from 0.52 (chance) to 0.83, and the "
                          "trajectory-integrated AUC from 0.53 to 0.94 at 30 min / 1.00 at 100 min. "
                          "Off by default only so previously logged --energy/--energyRate results stay "
                          "reproducible; the physics-loss pathway (--physics-loss-weight) always uses "
                          "the J2-inclusive form regardless, since it has no such legacy.")
parser.add_argument("--residual-ladder", dest="use_residual_ladder", action="store_true",
                     help="Append 5 hierarchical-residual-decomposition channels to the feature set "
                          "(see _computeResidualLadder). Encodes the domain fact that chemical thrust "
                          "is O(J2) while electric thrust is only O(J3-J6) by measuring the energy "
                          "residual under three successively richer dynamics truncations (two-body, "
                          "+J2, +J2..J6) alongside the J2 and J3-J6 reference rung heights, all "
                          "g0-normalized (so LEO and GEO share one scale) and log10'd (so 'the "
                          "residual sits at the J2 rung' is a subtraction rather than a 4-decade "
                          "ratio). Computed from the raw dimensional ECI snapshot, so it is unaffected "
                          "by --OE/--norm and composes with any of them. This is the feature-side "
                          "counterpart to --physics-loss-weight, which asserts the same fact as a "
                          "training penalty and then discards it at inference. Strongly recommended "
                          "with --standardize: these channels sit around -6..-2 while ECI channels are "
                          "O(1e3) and --norm channels are O(1). Note --sinusoids, if also set, "
                          "FFT-decomposes these channels along with every other one, which discards "
                          "the per-timestep transient this feature exists to expose.")
parser.add_argument("--residual-ladder-full", dest="use_residual_ladder_full", action="store_true",
                     help="With --residual-ladder, emit all 5 ladder channels (every residual rung "
                          "plus both reference heights) instead of the default lean 3 (deepest rung "
                          "+ both reference heights). Measured per-frame on leo/30min-1500, the two "
                          "extra rungs buy ~0.003 Electric-vs-NoThrust AUC and rungs 1-2 are 0.944 "
                          "correlated, so the lean set is the default -- but that ablation is "
                          "memoryless and the backbones here are sequence models, which may be able "
                          "to use rung 0 as an orbital-phase reference for the local noise floor. "
                          "Run runResidLadderAB.sh to test that. No effect without --residual-ladder.")
parser.add_argument("--sinusoids", type=int, default=0,
                     help="If > 0, replace each feature channel (ECI/OE, plus --energy/--energyRate "
                          "channels when enabled) with this many dominant sinusoidal components "
                          "extracted per trajectory via FFT (see _decomposeSinusoids), expanding C "
                          "channels into C*<sinusoids>. E.g. raw ECI states (6 channels) with "
                          "--sinusoids 3 yields 18 input channels: each channel's 3 most dominant "
                          "(non-DC) frequency components, reconstructed as their own real sinusoid "
                          "signal and ordered by descending magnitude, so component 0 is that "
                          "channel's single strongest oscillation for that trajectory, component 1 "
                          "the next, etc. 0 (default) leaves channels untouched.")
parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data to use for training")
parser.add_argument("--mode", type=str, default="all", choices=["all", "joint", "cascade", "stage1", "stage2"],
                     help="'joint': single 4-class per-timestep model. 'cascade': binary detector + "
                          "3-class type classifier, combined and compared against the joint model. "
                          "'stage1'/'stage2': train/evaluate only that cascade stage standalone. "
                          "'all' (default): joint and cascade both.")
parser.add_argument("--transformer", dest="use_transformer", action="store_true", help="Enable per-timestep Transformer model comparison (disabled by default)")
parser.add_argument("--cnn", dest="use_cnn", action="store_true", help="Enable per-timestep 1D-CNN (InceptionTime) model comparison (disabled by default)")
parser.add_argument("--hybrid", dest="use_hybrid", action="store_true", help="Enable 'hybrid' cascade comparison: whole-trajectory MiniRocket stage-1 detector + CNN (InceptionTime) stage-2 type classifier (disabled by default; cascade/stage-solo modes only, no joint form)")
parser.add_argument("--no-classic", dest="use_classic", action="store_false", help="Disable the LightGBM per-timestep classic-ML comparison (enabled by default)")
parser.add_argument("--xgboost", dest="use_xgboost", action="store_true", help="Enable XGBoost per-timestep classic-ML comparison (disabled by default)")
parser.add_argument("--catboost", dest="use_catboost", action="store_true", help="Enable CatBoost per-timestep classic-ML comparison (disabled by default)")
parser.add_argument("--rf", dest="use_random_forest", action="store_true", help="Enable Random Forest per-timestep classic-ML comparison (disabled by default)")
parser.add_argument("--extratrees", dest="use_extra_trees", action="store_true", help="Enable Extra Trees per-timestep classic-ML comparison (disabled by default)")
parser.add_argument("--pca", type=int, default=None, help="If set, PCA-reduce the Hankel-window features to this many components for the --mlp comparison (default: keep 95%% variance)")
parser.add_argument("--mlp", dest="use_mlp", action="store_true", help="Enable PCA+MLP per-timestep comparison on Hankel-windowed rows (disabled by default)")
parser.add_argument("--minirocket", dest="use_minirocket", action="store_true", help="Enable per-timestep MiniRocket comparison over trailing windows -- supports joint, cascade and stage-solo modes like the other per-timestep backbones (disabled by default)")
parser.add_argument("--minirocket-kernels", dest="minirocket_kernels", type=int, default=MINIROCKET_DEFAULT_KERNELS,
                     help=f"MiniRocket kernel count for --minirocket (default {MINIROCKET_DEFAULT_KERNELS}; sktime rounds this down to a multiple of 84)")
parser.add_argument("--loss-scheme", type=str, default="inverse", choices=["effective", "inverse"],
                     help="Per-timestep class weighting for CrossEntropy/focal loss. 'effective' (default): "
                          "class-balanced weights from the effective number of samples (Cui et al. 2019), which "
                          "scales more gently than raw inverse frequency across the dataset's variable 2:1-99:1 "
                          "NoThrust imbalance ratios. 'inverse': plain N/count_c weighting (previous default).")
parser.add_argument("--cb-beta", type=float, default=0.999, help="Beta for --loss-scheme effective (closer to 1 = more aggressive rebalancing of rare classes)")
parser.add_argument("--focal-gamma", type=float, default=0.0, help="If > 0, use focal loss with this gamma (on top of --loss-scheme class weights) instead of plain weighted CrossEntropy, to focus gradient on hard/misclassified timesteps rather than just rare ones")
parser.add_argument("--standardize", action="store_true",
                     help="Z-score every feature channel using TRAIN-split statistics only (val/test "
                          "are transformed with the train mean/std). Off by default so prior results "
                          "reproduce, but effectively required with --OE: those channels carry the "
                          "semi-major axis (~6.7e3 km), the orbital period (~5.5e3 s) and the "
                          "eccentricity (~1e-5) side by side, and that spread saturates the first "
                          "layer of every backbone -- training collapses to predicting NoThrust "
                          "everywhere and event F1 goes to exactly 0.")
parser.add_argument("--oversample", action="store_true",
                     help="Random-oversample the neural per-timestep training DataLoader (LSTM/Mamba/Transformer/CNN "
                          "train_loader; classic-ML/PCA+MLP/MiniRocket/hybrid stage 1 train on raw arrays and are "
                          "unaffected). Whole training trajectories are resampled with replacement, weighted toward "
                          "trajectories containing rarer per-timestep classes, since reordering individual timesteps "
                          "would break the temporal context LSTM/Mamba need. Val/test are never resampled. "
                          "Complementary to --loss-scheme (both can be combined, or --loss-scheme's effect reduced "
                          "via --cb-beta if double-compensation is a concern).")
parser.add_argument("--physics-loss-weight", type=float, default=0.0, dest="physics_loss_weight",
                     help="If > 0, adds an auxiliary Chemical-vs-Electric physics-consistency loss term on top of "
                          "the existing CE/focal loss, for --mode joint/stage2 (no-op for stage1, which has no "
                          "per-type logits, and for classic-ML/PCA+MLP/MiniRocket/hybrid, which don't go through "
                          "train_model at all). Reuses the model's EXISTING Chemical-vs-Electric logit margin (no "
                          "new head/parameters) as a binary prediction, trained via BCEWithLogitsLoss toward "
                          "whether the per-timestep energy-rate-derived thrust-accel-scale estimate looks closer "
                          "(in log-space) to the analytic J2 zonal-harmonic acceleration scale (Chemical-like) or "
                          "the combined J3-J6 scale (Electric-like). Gated to Chemical/Electric-labeled frames, "
                          "weighted per-timestep by a dimensionless, purely-geometric gate (J2/J3-J6 accel "
                          "normalized by local two-body gravity) that self-attenuates at high altitude -- where "
                          "thruster accel dwarfs ALL zonal harmonics and the cue stops being informative -- with "
                          "no regime-specific branching. A companion Thrust-vs-NoThrust ('detect') term was "
                          "removed after scripts/two_body/analyzeThrustSeparability.py measured its premise as "
                          "false; see the note in _load_and_label. 0.0 (default): fully off, no behavior change, "
                          "no extra compute.")
parser.add_argument("--smooth-max-gap", type=int, default=0, dest="smooth_max_gap",
                     help="If > 0, closes short interior gaps in per-timestep predictions before "
                          "event-level reporting/plotting (see smoothPerTimestepGrid): a run of "
                          "NoThrust-predicted frames of this length or shorter, bounded by a "
                          "thrusting prediction on BOTH sides, is filled in (type forward-filled "
                          "from the frame before the gap). Purely post-hoc -- no retraining, no "
                          "loss changes -- applied uniformly to every model family's [N,T] "
                          "prediction grid (neural joint/cascade, classic-ML/GBDT, PCA+MLP; a "
                          "no-op for --minirocket, whose broadcast whole-trajectory prediction has "
                          "no temporal gaps to close). Deliberately only ever CLOSES gaps, never "
                          "erases short positive runs, so it cannot destroy a correctly-detected "
                          "short burst the way a majority-vote window or minimum-run-length filter "
                          "could. Measured: no effect at all on LSTM or CNN (they produce ~no "
                          "interior gaps); ~0.9pp Impulsive event recall on Transformer at N=3, "
                          "with Chemical/Electric unchanged -- see sweepSmoothGap.py. "
                          "The raw (unsmoothed) event-level report and "
                          "plot always run first and are never replaced; a --smooth-max-gap > 0 "
                          "adds a second, separately labeled report+plot on the gap-closed grid, "
                          "so the payoff is always visible side by side. 0 (default): fully off.")
parser.add_argument("--seed", type=int, default=None,
                     help="Seed the IC train/val/test group split AND torch/numpy init, making a run "
                          "reproducible and making two runs that differ only in a feature flag a PAIRED "
                          "comparison on the same split. Without it every invocation draws a fresh split, "
                          "so A/B differences are confounded by split variance (which has moved these "
                          "numbers before). Tagged into the log stem as Seed<n>, LAST, so repeated seeds "
                          "of one flag set land in separate logs instead of silently overwriting each "
                          "other -- strAdd does not encode --mode/--loss-scheme/--standardize/backbones, "
                          "so without this token all seeds of a cell would compute the same path. "
                          "Default None: fresh entropy, and no token, so previously logged runs keep "
                          "reproducing their existing stems.")
parser.add_argument("--eval-test", dest="eval_test", action="store_true",
                     help="Evaluate on the held-out test split even when the test orbit equals the train "
                          "orbit. By default an in-distribution run reports on the VALIDATION split, which "
                          "is also what early stopping and best-checkpoint restore select on -- so those "
                          "numbers are optimistically biased, and are not comparable against a cross-orbit "
                          "run, which does report on a genuinely held-out split. The IC-disjoint 15% test "
                          "split already exists in that branch and is simply unused. Turn this on for any "
                          "table that puts in-distribution and cross-regime results side by side. Tagged "
                          "as EvalTest_ so it cannot collide with existing logs.")

parser.set_defaults(use_lstm=True)
parser.set_defaults(use_mamba=True)
parser.set_defaults(OE=False)
parser.set_defaults(noise=False)
parser.set_defaults(norm=False)
parser.set_defaults(one_pass=False)
parser.set_defaults(save_to_log=False)
parser.set_defaults(use_energy=False)
parser.set_defaults(use_energy_rate=False)
parser.set_defaults(use_j2_energy=False)
parser.set_defaults(use_transformer=False)
parser.set_defaults(use_cnn=False)
parser.set_defaults(use_hybrid=False)
parser.set_defaults(use_classic=True)
parser.set_defaults(use_xgboost=False)
parser.set_defaults(use_catboost=False)
parser.set_defaults(use_random_forest=False)
parser.set_defaults(use_extra_trees=False)
parser.set_defaults(use_mlp=False)
parser.set_defaults(use_minirocket=False)

args = parser.parse_args()
use_lstm = args.use_lstm
use_mamba = args.use_mamba
numMinProp = args.propMin
numRandSys = args.systems
orbitType = args.orbit
if args.test is None:
    args.test = args.orbit
    args.testSys = numRandSys
testSet = args.test
testSys = args.testSys
useOE = args.OE
useNoise = args.noise
useNorm = args.norm
useOnePass = args.one_pass
save_to_log = args.save_to_log
useEnergy = args.use_energy
useEnergyRate = args.use_energy_rate
useJ2Energy = args.use_j2_energy
useResidualLadder = args.use_residual_ladder
useResidualLadderFull = args.use_residual_ladder_full
numSinusoids = args.sinusoids
velNoise = args.velNoise
train_ratio = args.train_ratio
runMode = args.mode
use_transformer = args.use_transformer
use_cnn = args.use_cnn
use_hybrid = args.use_hybrid
use_classic = args.use_classic
use_xgboost = args.use_xgboost
use_catboost = args.use_catboost
use_random_forest = args.use_random_forest
use_extra_trees = args.use_extra_trees
use_mlp = args.use_mlp
use_minirocket = args.use_minirocket
minirocketKernels = args.minirocket_kernels
lossScheme = args.loss_scheme
cbBeta = args.cb_beta
focalGamma = args.focal_gamma
useOversample = args.oversample
useStandardize = args.standardize
physicsLossWeight = args.physics_loss_weight
usePhysicsLoss = physicsLossWeight > 0
smoothMaxGap = args.smooth_max_gap
runSeed = args.seed
evalTest = args.eval_test
if args.pca is not None and args.pca > 0:
    pca_n_components = args.pca
else:
    pca_n_components = 0.95

import os
import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import matplotlib
matplotlib.use("Agg")  # headless-safe: this script is typically run over SSH / redirected to a log file
import matplotlib.pyplot as plt

from qutils.tictoc import timer
from qutils.ml.utils import getDevice, printModelParmSize
from qutils.ml.classifer import apply_noise
from qutils.ml.mamba import Mamba, MambaConfig

if runSeed is not None:
    # Seeding the split alone (via _icGroupSplit's default_rng) would leave weight init, dropout and
    # the WeightedRandomSampler draw unseeded -- enough for a valid variance estimate across seeds,
    # but not enough to regenerate a specific table row. This is the first legal spot: torch/numpy
    # are imported after argument parsing.
    torch.manual_seed(runSeed)
    np.random.seed(runSeed)

device = getDevice()

# --residual-ladder channel layout, in the order _computeResidualLadder computes them. Used for the
# feature-count printout and for anyone reading a saved feature array back.
RESIDUAL_LADDER_ALL_CHANNEL_NAMES = (
    "log10(|dE_kep/dt| / (|v| g0))",      # rung 0: residual after removing two-body only
    "log10(|dE_J2/dt| / (|v| g0))",       # rung 1: ... after two-body + J2
    "log10(|dE_J2toJ6/dt| / (|v| g0))",   # rung 2: ... after two-body + J2..J6
    "log10(a_J2 / g0)",                   # reference rung height: J2
    "log10(a_J3toJ6 / g0)",               # reference rung height: J3-J6
)

# LEAN (default) keeps only the DEEPEST residual rung plus the two reference rung heights.
#
# Measured on leo/30min-1500, per-frame Electric-vs-NoThrust test AUC over an IC-split
# (1000 train / 500 test ICs), logistic regression / HistGradientBoosting:
#
#   rung 2 alone      0.8175 / 0.8127     three rungs (0,1,2)  0.8168 / 0.8193
#   rung 2 + refs     0.8187 / 0.8274     all five             0.8182 / 0.8301
#
# Stacking three residual rungs instead of one buys ~0.007 AUC, and rungs 1 and 2 are 0.944
# correlated -- they are near-duplicates, because the coasting floor barely moves between them
# (1.02x; the leftover is non-zonal, see the docs). What actually earns its channels is the pair
# of REFERENCE heights: rung 2 alone 0.8127 -> rung 2 + refs 0.8274 under the nonlinear model.
# So the lean set drops the two redundant rungs, not the comparison baselines, for 0.003 AUC.
#
# The multi-rung form is kept behind --residual-ladder-full: rung-to-rung drops are the diagnostic
# that told us the residual floor is not zonal, and that may not hold in another regime or against
# another force model. Caveat: the ablation above is per-frame and memoryless, while the backbones
# here are sequence models -- rung 0 is dominated by the J2 along-track projection, a smooth
# function of orbital phase that a sequence model could in principle use as a phase reference for
# the local floor. Untested; --residual-ladder-full vs. default is the A/B (see runResidLadderAB.sh).
RESIDUAL_LADDER_LEAN_INDICES = (2, 3, 4)


def residualLadderChannelNames(full=False):
    if full:
        return RESIDUAL_LADDER_ALL_CHANNEL_NAMES
    return tuple(RESIDUAL_LADDER_ALL_CHANNEL_NAMES[i] for i in RESIDUAL_LADDER_LEAN_INDICES)


def residualLadderChannelCount(full=False):
    return len(residualLadderChannelNames(full))


strAdd = ""
if useEnergy:
    strAdd = strAdd + "Energy_"
if useEnergyRate:
    strAdd = strAdd + "EnergyRate_"
if useJ2Energy:
    strAdd = strAdd + "J2Energy_"
if useResidualLadder:
    # Tagged with the explicit channel count rather than a bare "ResidLadder_": the A/B this flag
    # exists for compares 3 vs 5 channels, so the count has to be in the log stem or the two arms
    # land in the same parsed_data directory and silently overwrite each other (exactly the
    # collision displaySeqLogData.py's _suffix was fixed for). Note logs already on disk tagged
    # "ResidLadder_" predate this switch and are 5-channel runs.
    strAdd = strAdd + f"ResidLadder{residualLadderChannelCount(useResidualLadderFull)}_"
if useOE:
    strAdd = strAdd + "OE_"
if numSinusoids > 0:
    strAdd = strAdd + f"Sinusoids{numSinusoids}_"
if useNorm:
    strAdd = strAdd + "Norm_"
if useNoise:
    strAdd = strAdd + "Noise_"
if useOnePass:
    strAdd = strAdd + "OnePass_"
if train_ratio != 0.7:
    strAdd = strAdd + f"Train_{int(4*train_ratio*numRandSys)}_"
if testSet != orbitType:
    strAdd = strAdd + "Test_" + testSet + "_"
if velNoise != 1e-3:
    strAdd = strAdd + f"VelNoise{velNoise}_"
if physicsLossWeight > 0:
    strAdd = strAdd + f"PhysLoss{physicsLossWeight}_"
if smoothMaxGap > 0:
    strAdd = strAdd + f"SmoothGap{smoothMaxGap}_"
if use_minirocket and minirocketKernels != MINIROCKET_DEFAULT_KERNELS:
    # Non-default kernel counts otherwise resolve to the same stem as the default run, and --save
    # opens the log 'w' -- a sweep over kernel counts would silently truncate down to one log.
    strAdd = strAdd + f"MRKernels{minirocketKernels}_"
if evalTest and testSet == orbitType:
    # Only meaningful in-distribution: cross-orbit runs already evaluate on the held-out test split,
    # and they carry Test_<orbit>_ above, so tagging them too would split one arm across two names.
    strAdd = strAdd + "EvalTest_"
if runSeed is not None:
    # LAST, deliberately. The sweep script mirrors this stem by string concatenation, so a token that
    # is always final means the stem has no conditional tail; and "Seed\d+$" then doubles as the
    # aggregator's "is this a sweep log" filter, excluding pre-existing logs in the same directories.
    strAdd = strAdd + f"Seed{runSeed}_"

if strAdd.endswith("_"):
    strAdd = strAdd[:-1]

print(f"Training with {int(4*train_ratio*numRandSys)} systems")

logLoc = "gmat/data/seqClassification/" + str(orbitType) + "/" + str(numMinProp) + "min-" + str(numRandSys) + "/"
logStem = str(numMinProp) + "min" + str(numRandSys) + strAdd
logFileLoc = logLoc + logStem + '.log'

# Per-timestep sequence-prediction plots (see plotSequencePrediction) are generated regardless of
# --save, since they're a standalone visual artifact rather than part of the redirected log.
plotLoc = logLoc + "plots/"
if not os.path.exists(plotLoc):
    os.makedirs(plotLoc)

if save_to_log:
    from contextlib import redirect_stdout, redirect_stderr

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 10000)
    pd.set_option('display.expand_frame_repr', False)

    import warnings
    warnings.filterwarnings("ignore")

    if not os.path.exists(logLoc):
        os.makedirs(logLoc)
    print("saving log output to {}".format(logFileLoc))


# ---------------------------------------------------------------------------
# Class/label conventions shared by data loading, training, and reporting
# ---------------------------------------------------------------------------
CLASS_ORDER = ["Chemical", "Electric", "ImpBurn", "NoThrust"]
JOINT_LABELS = {"NoThrust": 0, "Chemical": 1, "Electric": 2, "ImpBurn": 3}
JOINT_CLASS_NAMES = ["No Thrust", "Chemical", "Electric", "Impulsive"]
STAGE1_CLASS_NAMES = ["No Thrust", "Thrust"]
STAGE2_CLASS_NAMES = ["Chemical", "Electric", "Impulsive"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _getThrustingTime(npz_dict, class_name, N, T, warn_if_missing=True):
    if 'thrustingTime' in npz_dict:
        return npz_dict['thrustingTime']
    if warn_if_missing:
        print(f"[prepareInSequenceThrustClassificationDatasets] WARNING: '{class_name}' has no "
              f"'thrustingTime' key -- defaulting to all-background per-timestep labels. This is "
              f"expected until the upstream GMAT-Thrust-Data dataset fix lands; per-timestep "
              f"labels for {class_name} will be wrong until then.")
    return np.zeros((N, T, 1))


def _decomposeSinusoids(states, num_sinusoids):
    """states: [N,T,C] -> [N,T,C*num_sinusoids]. Every (IC, channel) time series is decomposed
    independently: an FFT along the time axis picks the `num_sinusoids` frequency bins with the
    largest magnitude (excluding the DC bin, which is just that channel's mean over the window, not
    an oscillation), and each selected bin is reconstructed as its own real sinusoid signal
    A*cos(2*pi*m*t/T + phase) of length T. The K reconstructed component signals REPLACE the single
    raw channel, so C channels become C*num_sinusoids -- e.g. 6 ECI channels with num_sinusoids=3
    yields 18, grouped as [chan0_comp0, chan0_comp1, chan0_comp2, chan1_comp0, ...].

    Components are ordered by descending magnitude per (IC, channel), so component 0 is always that
    trajectory's single most dominant oscillation for that channel, component 1 the next, etc.,
    regardless of which absolute frequency bin that turns out to be -- the network is fed a stable
    "1st/2nd/3rd dominant mode" ordering rather than fixed frequency bins that fit some trajectories
    (e.g. NoThrust, near-periodic) and not others (e.g. a burn mid-window)."""
    N, T, C = states.shape
    max_components = T // 2  # non-DC rfft bins: 1..T//2 (T//2+1 bins total incl. DC, and Nyquist if T even)
    if num_sinusoids > max_components:
        print(f"[_decomposeSinusoids] WARNING: requested {num_sinusoids} sinusoids but only "
              f"{max_components} non-DC frequency bins are available for T={T}; clamping.")
        num_sinusoids = max_components

    freqs = np.fft.rfft(states, axis=1)   # [N, T//2+1, C] complex
    mag = np.abs(freqs)
    mag[:, 0, :] = -1.0                   # exclude the DC bin from selection

    order = np.argsort(-mag, axis=1)[:, :num_sinusoids, :]         # [N,K,C] bin indices, descending magnitude
    amp = np.take_along_axis(mag, order, axis=1)                    # [N,K,C]
    phase = np.angle(np.take_along_axis(freqs, order, axis=1))      # [N,K,C]

    # A real signal's rfft coefficients need a factor of 2/T to reconstruct amplitude, except the
    # Nyquist bin (T even only), which has no conjugate partner and needs 1/T.
    nyquist_bin = T // 2 if T % 2 == 0 else -1
    scale = np.where(order == nyquist_bin, 1.0, 2.0) / T            # [N,K,C]

    t = np.arange(T).reshape(1, T, 1, 1)
    m = order.reshape(N, 1, num_sinusoids, C)
    a = (amp * scale).reshape(N, 1, num_sinusoids, C)
    p = phase.reshape(N, 1, num_sinusoids, C)

    components = a * np.cos(2 * np.pi * m * t / T + p)              # [N,T,K,C]
    return components.transpose(0, 1, 3, 2).reshape(N, T, C * num_sinusoids)


def _applyTransforms(states_by_class, useOE, useNorm, useNoise, useEnergy, useEnergyRate, numSinusoids, pos_noise_std, vel_noise_std, usePhysicsLoss=False, useJ2Energy=False, useResidualLadder=False, useResidualLadderFull=False):
    if useNoise:
        for c in CLASS_ORDER:
            states_by_class[c] = apply_noise(states_by_class[c], pos_noise_std, vel_noise_std)

    # orbitalEnergy() is a Cartesian quantity: it reads Y[:,0:3] as position and Y[:,3:6] as
    # velocity. --OE overwrites states_by_class with orbital elements below, so snapshot the raw
    # ECI states first and compute energy from those. Feeding orbital elements to it instead
    # evaluates 0.5*||(Omega,omega,M0)||^2 - mu/||(a,e,i)||, a quantity that correlates only ~0.25
    # with true orbital energy and destroys the whole premise of the feature (energy is ~constant
    # under two-body coasting, so departures from constant are the thrust residual).
    eci_by_class = None
    if useOE and (useEnergy or useEnergyRate):
        eci_by_class = {c: states_by_class[c].copy() for c in CLASS_ORDER}

    # Physics-loss inputs (--physics-loss-weight) and the residual ladder (--residual-ladder) both
    # need raw dimensional (km, km/s) ECI Cartesian state regardless of what --OE/--norm do to
    # states_by_class below -- unlike eci_by_class above, this snapshot is taken unconditionally
    # (not gated on useOE) so it stays correct even under --norm alone. It is taken AFTER
    # apply_noise, so under --noise the ladder measures the residual of the noisy trajectory the
    # model actually sees rather than a clean one the model has no access to.
    phys_eci_by_class = None
    if usePhysicsLoss or useResidualLadder:
        phys_eci_by_class = {c: states_by_class[c].copy() for c in CLASS_ORDER}

    oe_by_class = None
    if useOE:
        from qutils.orbital import ECI2OE
        oe_by_class = {}
        for c in CLASS_ORDER:
            s = states_by_class[c]
            n_ic, T = s.shape[0], s.shape[1]
            oe = np.zeros((n_ic, T, 7))
            for i in range(n_ic):
                for j in range(T):
                    oe[i, j, :] = ECI2OE(s[i, j, 0:3], s[i, j, 3:6])
            if useNorm:
                R = 6378.1363
                oe[:, :, 0] = oe[:, :, 0] / R
            oe_by_class[c] = oe
            states_by_class[c] = oe[:, :, 0:6]
    elif useNorm:
        from qutils.orbital import dim2NonDim6
        for c in CLASS_ORDER:
            s = states_by_class[c]
            for i in range(s.shape[0]):
                s[i, :, :] = dim2NonDim6(s[i, :, :])
            states_by_class[c] = s

    if useEnergy or useEnergyRate:
        from qutils.orbital import orbitalEnergy
        energy_by_class = {}
        rate_by_class = {}
        # norming_energy is a single scalar shared across all 4 classes (taken from the first
        # class processed), recomputed fresh on every _applyTransforms call -- when --test names a
        # different orbit/system count than --orbit, train and test energy (and energy rate) get
        # normalized by DIFFERENT scalars, a pre-existing train/test scale inconsistency under
        # --norm --energy --test <different orbit>, not introduced or fixed here.
        norming_energy = None
        for c in CLASS_ORDER:
            # eci_by_class is set only under --OE; otherwise states_by_class is still Cartesian
            # (dimensional, or non-dimensionalized by dim2NonDim6 under --norm) and is used as-is.
            s = states_by_class[c] if eci_by_class is None else eci_by_class[c]
            n_ic, T = s.shape[0], s.shape[1]
            energy = np.zeros((n_ic, T, 1))
            if useJ2Energy:
                # J2-inclusive specific energy: the Keplerian form below is NOT conserved under
                # J2, so its rate measures J2 potential exchange (~55x the amplitude of a
                # low-thrust electric signature) far more than it measures any thrust. See
                # orbitalEnergyJ2 and scripts/two_body/analyzeThrustSeparability.py. Vectorized
                # over the whole [N,T,6] block, unlike orbitalEnergy's per-row loop.
                from qutils.orbital import orbitalEnergyJ2
                energy[:, :, 0] = orbitalEnergyJ2(s)
            else:
                for i in range(n_ic):
                    energy[i, :, 0] = orbitalEnergy(s[i, :, :])
            if useNorm:
                if norming_energy is None:
                    norming_energy = energy[0, 0, 0]
                energy[:, :, 0] = energy[:, :, 0] / norming_energy
            if useEnergyRate:
                # diff is linear, so computing the rate after normalization above is equivalent to
                # normalizing a raw-energy rate -- order doesn't change the numeric result. prepend
                # gives t=0 a rate of exactly 0 (no prior point) in one call.
                rate_by_class[c] = np.diff(energy, axis=1, prepend=energy[:, :1, :])
            energy_by_class[c] = energy

        extra_channels = {}
        for c in CLASS_ORDER:
            parts = ([energy_by_class[c]] if useEnergy else []) + \
                    ([rate_by_class[c]] if useEnergyRate else [])
            extra_channels[c] = np.concatenate(parts, axis=2)

        # Energy/energy-rate are ADDITIONAL channels, never a replacement. The previous else-branch
        # assigned extra_channels directly, so --energy or --energyRate without --OE silently
        # dropped all six ECI channels and trained on the single energy scalar alone.
        base = oe_by_class if oe_by_class is not None else states_by_class
        for c in CLASS_ORDER:
            states_by_class[c] = np.concatenate((base[c], extra_channels[c]), axis=2)

    if useResidualLadder:
        # Hierarchical residual decomposition (--residual-ladder): 5 additional channels appended
        # to whatever --OE/--energy/--energyRate leave behind, never a replacement (same policy as
        # the energy channels above). Computed from the raw dimensional ECI snapshot, so these
        # channels are identical with and without --OE/--norm -- unlike --energy, which silently
        # changes meaning under --norm (see norming_energy above). See _computeResidualLadder.
        for c in CLASS_ORDER:
            ladder = _computeResidualLadder(phys_eci_by_class[c], full=useResidualLadderFull)
            states_by_class[c] = np.concatenate((states_by_class[c], ladder), axis=2)

    if numSinusoids > 0:
        # Decomposes whatever channels the rest of _applyTransforms leaves behind (raw ECI, OE,
        # and/or the energy/energy-rate channels above) -- runs last so it always sees the final
        # per-timestep feature set, not the pre-OE/pre-energy Cartesian states.
        for c in CLASS_ORDER:
            states_by_class[c] = _decomposeSinusoids(states_by_class[c], numSinusoids)

    return states_by_class, phys_eci_by_class


PHYS_LOSS_DT_SECONDS = 60.0  # GMAT generation scripts' fixed propagation step (see
                              # gmat/scripts/generateSpacecraftThrusts.py / generateSpacecraftEThrusts.py: dt = 60.0)


def _computePhysicsResidualTensors(eci_states, dt_seconds=PHYS_LOSS_DT_SECONDS):
    """eci_states: [N,T,6] raw ECI Cartesian (km, km/s), pre-OE/pre-norm (see phys_eci_by_class in
    _applyTransforms). Returns (h2, h36, g0, residual_accel), each [N,T]:
      g0             = mu/|r|^2          two-body gravitational accel magnitude (km/s^2)
      h2             = j2AccelMag(r)/g0  dimensionless J2-to-two-body ratio (orbit-invariant
                                          formula; VALUE naturally shrinks with altitude)
      h36            = j3to6AccelMag(r)/g0  dimensionless combined-J3-J6-to-two-body ratio
      residual_accel = |dE/dt| / |v|     thrust-accel-scale estimate: a finite-difference energy
                       residual divided by the real sample interval (not left as a raw per-index
                       delta, since this pathway compares against absolute physical accelerations)
                       and converted power->accel via /|v|.

    E here is orbitalEnergyJ2 -- the J2-INCLUSIVE specific energy -- NOT the Keplerian
    v^2/2 - mu/r that --energy/--energyRate use by default.

    mu is fixed to qutils.orbital.MU_EARTH_JGM2 throughout (energy, g0, h2, h36) for internal
    self-consistency of the log-ratio comparison in _load_and_label.
    """
    from qutils.orbital import MU_EARTH_JGM2, orbitalEnergyJ2, twoBodyAccel, j2AccelMag, j3to6AccelMag

    r = eci_states[..., 0:3]                                          # [N,T,3]
    v_mag = np.linalg.norm(eci_states[..., 3:6], axis=-1)             # [N,T]

    energy = orbitalEnergyJ2(eci_states, mu=MU_EARTH_JGM2)            # [N,T]
    energy_rate = np.diff(energy, axis=1, prepend=energy[:, :1]) / dt_seconds
    residual_accel = np.abs(energy_rate) / np.maximum(v_mag, 1e-9)    # [N,T], km/s^2

    g0 = twoBodyAccel(r, mu=MU_EARTH_JGM2)
    h2 = j2AccelMag(r, mu=MU_EARTH_JGM2) / g0
    h36 = j3to6AccelMag(r, mu=MU_EARTH_JGM2) / g0
    return h2, h36, g0, residual_accel


# Floor applied to every dimensionless ratio before the log. The quantities of interest sit around
# 1e-6..1e-3 (see the docstring), so 1e-12 is ~6 decades below anything meaningful: it exists only
# to keep an incidental near-zero energy-rate crossing from becoming a -inf/-30 spike that would
# dominate --standardize's mean/std for the whole channel.
RESIDUAL_LADDER_FLOOR = 1e-12


def _computeResidualLadder(eci_states, dt_seconds=PHYS_LOSS_DT_SECONDS, full=False):
    """Hierarchical residual decomposition. eci_states: [N,T,6] raw ECI Cartesian (km, km/s),
    pre-OE/pre-norm (see phys_eci_by_class in _applyTransforms). Returns [N,T,C] float32, where
    C is 3 for the default lean set (deepest rung + both reference heights) and 5 with full=True
    (every rung). All five are always computed -- the switch only selects which are returned, so
    the two forms are guaranteed bit-identical on the channels they share. See
    RESIDUAL_LADDER_LEAN_INDICES for the ablation behind that default.

    The idea: dE/dt under a given dynamics truncation measures exactly those accelerations the
    truncation leaves out. So evaluating the energy residual at successively richer truncations
    gives a ladder of noise floors, and a thrust reveals itself at the rung where it first stands
    above the floor:

      rung 0 (two-body):    floor is J2 potential exchange, ~2.7e-6 km/s^2 in LEO
      rung 1 (+J2):         floor drops ~55x to ~9.9e-8 km/s^2 -- J3-J6, tesserals, drag
      rung 2 (+J2..J6):     floor is whatever is left (tesseral/sectoral terms, drag, noise)

    That is a direct encoding of the domain fact this feature exists for: chemical thrust is
    O(J2) so it is visible at every rung, while electric thrust is O(J3-J6) so it only clears the
    floor at rungs 1-2. Rather than asserting that as a loss penalty (--physics-loss-weight,
    which is discarded at inference), it hands the model the measurements the assertion is about
    and lets it use them at every forward pass.

    All five channels are dimensionless (divided by g0 = mu/r^2) and log10-scaled:

      * dimensionless, so the same channel means the same thing in LEO and GEO -- the raw
        accelerations differ by ~3 decades between regimes but the ratios do not, which is what
        makes this regime-agnostic in the same sense as the --physics-loss-weight gate.
      * log10, because the prior is about ORDERS OF MAGNITUDE. In the log domain "the residual
        sits at the J2 rung" is a subtraction, so a linear layer can express it; in the linear
        domain it is a ratio spanning 4+ decades that a standardized channel cannot resolve.

    Channels 3-4 are the reference rung heights a_J2/g0 and a_J3toJ6/g0. They are deterministic
    functions of r and so carry no new information in principle, but they are what the residual
    rungs must be COMPARED against, and in log space channel_k - channel_3 is exactly "how many
    decades is the residual above the J2 rung" -- the comparison the domain fact is stated in.

    mu/Re/J_n are fixed to the qutils JGM2 constants throughout, matching
    _computePhysicsResidualTensors, so the two pathways cannot disagree about rung heights.
    """
    from qutils.orbital import (MU_EARTH_JGM2, orbitalEnergyZonal, twoBodyAccel,
                                j2AccelMag, j3to6AccelMag)

    r = eci_states[..., 0:3]                                          # [N,T,3]
    v_mag = np.linalg.norm(eci_states[..., 3:6], axis=-1)             # [N,T]
    g0 = twoBodyAccel(r, mu=MU_EARTH_JGM2)                            # [N,T]
    # |dE/dt| / |v| converts specific power to a thrust-acceleration scale; the further /g0 makes
    # it dimensionless. Folded into one denominator so it is clamped once.
    denom = np.maximum(v_mag * g0, 1e-30)

    T = eci_states.shape[1]
    rungs = []
    for degrees in ((), (2,), (2, 3, 4, 5, 6)):
        E = orbitalEnergyZonal(eci_states, degrees=degrees, mu=MU_EARTH_JGM2)   # [N,T]
        rate = np.zeros_like(E)
        if T > 1:
            rate[:, 1:] = np.diff(E, axis=1) / dt_seconds
            # t=0 has no prior sample. Copy t=1's rate rather than np.diff's usual prepend-zero:
            # an exact 0 here would floor to log10(1e-12) and put a -12 outlier in every single
            # trajectory's first frame, skewing --standardize's per-channel statistics.
            rate[:, 0] = rate[:, 1]
        rungs.append(np.abs(rate) / denom)

    rungs.append(j2AccelMag(r, mu=MU_EARTH_JGM2) / g0)
    rungs.append(j3to6AccelMag(r, mu=MU_EARTH_JGM2) / g0)

    ladder = np.stack(rungs, axis=-1)                                 # [N,T,5]
    if not full:
        ladder = ladder[..., list(RESIDUAL_LADDER_LEAN_INDICES)]      # [N,T,3]
    return np.log10(np.maximum(ladder, RESIDUAL_LADDER_FLOOR)).astype(np.float32)


def _load_and_label(loc, useOE, useNorm, useNoise, useEnergy, useEnergyRate, numSinusoids, pos_noise_std, vel_noise_std, usePhysicsLoss=False, useJ2Energy=False, useResidualLadder=False, useResidualLadderFull=False):
    states_by_class = {}
    thrusting_by_class = {}
    for class_name in CLASS_ORDER:
        npz = np.load(f"{loc}/statesArray{class_name}.npz")
        states = npz[f"statesArray{class_name}"]
        N, T = states.shape[0], states.shape[1]
        tt = _getThrustingTime(npz, class_name, N, T, warn_if_missing=(class_name != "NoThrust"))
        states_by_class[class_name] = states
        thrusting_by_class[class_name] = tt

    n_ic_per_class = [states_by_class[c].shape[0] for c in CLASS_ORDER]
    T_per_class = [states_by_class[c].shape[1] for c in CLASS_ORDER]
    assert len(set(T_per_class)) == 1, f"Timestep count mismatch across classes: {dict(zip(CLASS_ORDER, T_per_class))}"

    states_by_class, phys_eci_by_class = _applyTransforms(
        states_by_class, useOE, useNorm, useNoise, useEnergy, useEnergyRate, numSinusoids,
        pos_noise_std, vel_noise_std, usePhysicsLoss, useJ2Energy, useResidualLadder,
        useResidualLadderFull)

    states_cat = np.concatenate([states_by_class[c] for c in CLASS_ORDER], axis=0)

    joint_labels_list = [thrusting_by_class[c].squeeze(-1).astype(np.int64) * JOINT_LABELS[c] for c in CLASS_ORDER]
    y_joint = np.concatenate(joint_labels_list, axis=0)

    phys_target_ce, phys_gate_ce = None, None
    if usePhysicsLoss:
        h2_parts, h36_parts, g0_parts, resid_parts = [], [], [], []
        for c in CLASS_ORDER:
            h2_c, h36_c, g0_c, resid_c = _computePhysicsResidualTensors(phys_eci_by_class[c])
            h2_parts.append(h2_c); h36_parts.append(h36_c)
            g0_parts.append(g0_c); resid_parts.append(resid_c)
        h2 = np.concatenate(h2_parts, axis=0)
        h36 = np.concatenate(h36_parts, axis=0)
        g0 = np.concatenate(g0_parts, axis=0)
        residual_accel = np.concatenate(resid_parts, axis=0)

        log_resid = np.log(np.maximum(residual_accel, 1e-30))
        log_aJ2 = np.log(np.maximum(h2 * g0, 1e-30))
        log_aJ36 = np.log(np.maximum(h36 * g0, 1e-30))
        # Chemical-vs-Electric term (--mode joint/stage2): physics-only pseudo-target, independent
        # of the true label -- does the empirical thrust-accel-scale estimate look closer (in
        # log-space) to the J2 scale (Chemical-like, target=1) or the combined J3-J6 scale
        # (Electric-like, target=0)?
        pseudo_target_ce = (np.abs(log_resid - log_aJ2) < np.abs(log_resid - log_aJ36)).astype(np.float32)

        # Only Chemical/Electric-labeled frames get any physics-loss contribution; NoThrust and
        # Impulsive are gated out entirely (gate=0 there).
        is_chem_or_elec = (y_joint == JOINT_LABELS["Chemical"]) | (y_joint == JOINT_LABELS["Electric"])
        gate_ce = np.where(is_chem_or_elec, (h2 + h36).astype(np.float32), 0.0).astype(np.float32)

        if is_chem_or_elec.any():
            print(f"[physics-loss] chem/elec gate (h2+h36) over {int(is_chem_or_elec.sum())} "
                  f"Chemical/Electric frames: mean={gate_ce[is_chem_or_elec].mean():.3e}, "
                  f"max={gate_ce.max():.3e} (naturally shrinks toward 0 at higher altitude -- LEO "
                  f"~1e-3 vs. GEO several orders of magnitude smaller, since J2/J3-J6 fall off as "
                  f"~1/r^4..1/r^8)")

        # A Thrust-vs-NoThrust ('detect') term used to live here, targeting the NoThrust/Electric
        # boundary via 'is residual_accel above the J3-J6 floor'. It was REMOVED after
        # scripts/two_body/analyzeThrustSeparability.py measured the quantities it assumed:
        #   - measured residual_accel on pure COASTING frames is ~2.7e-6 km/s^2, while a_J3-J6 is
        #     ~3.8e-8 -- so that pseudo-target was true on essentially every frame, teaching the
        #     model "everything is thrust".
        #   - the background is a ZERO-MEAN J2 oscillation (median signed dE/dt on NoThrust is
        #     exactly 0.0) of amplitude ~2.7e-6, and electric thrust contributes a constant-sign
        #     bias of only ~+2.0e-7 -- a 1:13 ratio, giving per-frame Electric-vs-NoThrust
        #     ROC AUC ~0.52 (chance). No threshold rescues an uninformative feature: recalibrating
        #     it merely flips the target to 0 on true Electric frames, which would supervise the
        #     model to MISS Electric.
        # The signal that does separate them is the SIGNED, orbit-integrated energy change
        # (AUC 0.652 over one full orbit, rising with window length), which is a trajectory-level
        # quantity and does not fit this per-timestep loss. Chemical needs no such help: it sits
        # ~12x above the background and is already perfectly separable (AUC 1.000).
        phys_target_ce, phys_gate_ce = pseudo_target_ce, gate_ce

    return states_cat, y_joint, n_ic_per_class, phys_target_ce, phys_gate_ce


def _deriveStage1Stage2(y_joint, pad_idx=-100):
    """y_joint: [...,] any shape of 4-class per-timestep labels (0=NoThrust,1=Chemical,
    2=Electric,3=Impulsive) -> (y_stage1, y_stage2). y_stage1 is the binary 'is thrust
    occurring' label; y_stage2 remaps Chemical/Electric/Impulsive to 0/1/2 and masks every
    non-thrusting position to pad_idx, so a type classifier is never supervised on background
    frames. Shared by the DataLoader-building path and the classic-ML/PCA+MLP row-based paths."""
    y_stage1 = (y_joint > 0).astype(np.int64)
    y_stage2 = np.full_like(y_joint, pad_idx)
    for src, dst in ((1, 0), (2, 1), (3, 2)):
        y_stage2[y_joint == src] = dst
    return y_stage1, y_stage2


def _icGroupSplit(n_ic_per_class, train_ratio, val_ratio, test_ratio, seed=None):
    """One IC-index permutation applied identically to every class block, so the same underlying
    initial condition (IC index i, shared across all 4 class npz files since each generator run
    reseeds the same RNG) always lands in the same split regardless of which thrust-type file it
    appears in -- prevents leaking near-duplicate pre-thrust dynamics across train/val/test.
    Assumes every class shares the same number of ICs."""
    assert len(set(n_ic_per_class)) == 1, f"_icGroupSplit assumes equal IC counts per class, got {n_ic_per_class}"
    n_ic = n_ic_per_class[0]

    n_train = int(np.floor(train_ratio * n_ic))
    n_val = int(np.floor(val_ratio * n_ic))
    n_test = n_ic - n_train - n_val
    assert n_test > 0, "Ratios leave no ICs for test; reduce train/val."

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_ic)
    train_ic = perm[:n_train]
    val_ic = perm[n_train:n_train + n_val]
    test_ic = perm[n_train + n_val:]

    groups = np.tile(np.arange(n_ic, dtype=np.int64), len(n_ic_per_class))
    train_mask = np.isin(groups, train_ic)
    val_mask = np.isin(groups, val_ic)
    test_mask = np.isin(groups, test_ic)
    return train_mask, val_mask, test_mask


def _computeOversamplingWeights(y_joint, num_classes):
    """y_joint: [N,T] training-split joint labels -> weights[N] for a WeightedRandomSampler.
    Random-oversamples whole trajectories (rather than individual timesteps, which would break
    the temporal context LSTM/Mamba need) by weighting each trajectory toward the rarest
    per-timestep class it contains: weight_i = max over classes c present in trajectory i of
    (1 / count_c), so a trajectory touching a rare class (e.g. a single Chemical burst minute) is
    drawn more often regardless of how much NoThrust background padding surrounds it, while
    NoThrust-only trajectories keep the baseline weight."""
    counts = np.bincount(y_joint.reshape(-1), minlength=num_classes).astype(np.float64)
    inv_freq = counts.sum() / np.clip(counts, 1.0, None)
    weights = np.array([inv_freq[np.unique(row)].max() for row in y_joint], dtype=np.float64)
    return weights


def _standardizeSplits(train_data, val_data, test_data, supress_print=False):
    """Z-scores each feature channel using TRAIN-split statistics only, applied identically to val
    and test. Returns (train, val, test, mu, sigma).

    Fitting on train alone is what keeps this from leaking: computing val/test statistics from
    their own splits would let information about held-out orbits reach the model through the
    scaling constants. Channels with zero variance in train (a constant feature) are left alone
    rather than divided by ~0."""
    C = train_data.shape[2]
    flat = train_data.reshape(-1, C)
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0)
    degenerate = sigma < 1e-12
    sigma = np.where(degenerate, 1.0, sigma)

    if not supress_print:
        if degenerate.any():
            print(f"Standardize: channels {np.flatnonzero(degenerate).tolist()} are constant in "
                  f"train; left unscaled.")
        print(f"Standardize: per-channel train mean range [{mu.min():.4g}, {mu.max():.4g}], "
              f"std range [{sigma.min():.4g}, {sigma.max():.4g}] -> all channels z-scored")

    out = tuple((d - mu) / sigma for d in (train_data, val_data, test_data))
    return out + (mu, sigma)


def prepareInSequenceThrustClassificationDatasets(
    yaml_config, data_config,
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
    pos_noise_std=1e-3, vel_noise_std=1e-3,
    batch_size=16, pad_idx=-100, seed=None,
    supress_print=False, return_meta=False, oversample=False, standardize=False,
):
    """Loads all 4 thrust-type classes (Chemical, Electric, ImpBurn, NoThrust) and builds
    per-timestep labels for three views of the same data:
      - joint:  4-class per-timestep label (0=NoThrust,1=Chemical,2=Electric,3=Impulsive)
      - stage1: binary per-timestep 'is thrust occurring' label (0/1)
      - stage2: 3-class per-timestep thrust-type label (0=Chemical,1=Electric,2=Impulsive),
                masked to pad_idx on every non-thrusting timestep so a type classifier is never
                supervised on background frames.
    Assumes equal ICs per class and that IC index i refers to the same underlying orbit across
    all 4 class files. Per-timestep 'thrustingTime' ground truth is defensively loaded -- a class
    file missing the key degrades to all-background labels with a printed warning rather than
    raising.

    oversample: if True, train_loader draws whole training trajectories with replacement via a
    WeightedRandomSampler (see _computeOversamplingWeights) instead of a plain shuffle, so
    trajectories containing rarer per-timestep classes are seen more often each epoch. val_loader
    and test_loader are never resampled.

    standardize: if True, z-score every feature channel using TRAIN-split statistics only (val and
    test are transformed with the train mean/std, never their own). Off by default so existing
    results stay reproducible, but strongly recommended for any flag combination that leaves
    channels on wildly different scales -- notably --OE, whose channels span the semi-major axis
    (~6.7e3 km) and eccentricity (~1e-5) simultaneously, an 8-order-of-magnitude spread that
    saturates LSTM/Mamba gates on the first layer and collapses training to majority-class
    prediction. See _standardizeSplits.

    yaml_config['numSinusoids']: if > 0, replaces every feature channel with this many dominant
    sinusoidal components extracted per trajectory via FFT (see _decomposeSinusoids), applied as
    the last step of _applyTransforms -- so it decomposes whatever channels --OE/--energy/
    --energyRate leave behind. C channels become C*numSinusoids (e.g. 6 ECI channels with
    numSinusoids=3 yields 18), and standardize (if also on) fits on those expanded channels.
    """
    useOE = yaml_config['useOE']
    useNorm = yaml_config['useNorm']
    useNoise = yaml_config['useNoise']
    useEnergy = yaml_config['useEnergy']
    useEnergyRate = yaml_config['useEnergyRate']
    numSinusoids = yaml_config['numSinusoids']
    usePhysicsLoss = yaml_config.get('usePhysicsLoss', False)
    useJ2Energy = yaml_config.get('useJ2Energy', False)
    useResidualLadder = yaml_config.get('useResidualLadder', False)
    useResidualLadderFull = yaml_config.get('useResidualLadderFull', False)

    numMinProp = yaml_config['prop_time']
    train_set = yaml_config['orbit']
    systems = yaml_config['systems']
    test_set = yaml_config['test_dataset']
    test_systems = yaml_config['test_systems']

    dataLoc = data_config['seqClassification'] + train_set + "/" + str(numMinProp) + "min-" + str(systems)
    dataLoc_test = data_config['seqClassification'] + test_set + "/" + str(numMinProp) + "min-" + str(test_systems)

    if not supress_print:
        print(f"Training data location: {dataLoc}")
        print(f"Test data location: {dataLoc_test}")

    states, y_joint, n_ic_per_class, phys_target_ce, phys_gate_ce = _load_and_label(
        dataLoc, useOE, useNorm, useNoise, useEnergy, useEnergyRate, numSinusoids, pos_noise_std, vel_noise_std,
        usePhysicsLoss, useJ2Energy, useResidualLadder, useResidualLadderFull
    )
    if phys_target_ce is None:
        phys_target_ce = np.zeros_like(y_joint, dtype=np.float32)
        phys_gate_ce = np.zeros_like(y_joint, dtype=np.float32)

    train_mask, val_mask, test_mask = _icGroupSplit(n_ic_per_class, train_ratio, val_ratio, test_ratio, seed=seed)

    train_data, train_joint = states[train_mask], y_joint[train_mask]
    train_phys_target_ce, train_phys_gate_ce = phys_target_ce[train_mask], phys_gate_ce[train_mask]
    val_data, val_joint = states[val_mask], y_joint[val_mask]
    val_phys_target_ce, val_phys_gate_ce = phys_target_ce[val_mask], phys_gate_ce[val_mask]

    if test_set != train_set or test_systems != systems:
        states_t, y_joint_t, n_ic_per_class_t, phys_target_ce_t, phys_gate_ce_t = _load_and_label(
            dataLoc_test, useOE, useNorm, useNoise, useEnergy, useEnergyRate, numSinusoids, pos_noise_std, vel_noise_std,
            usePhysicsLoss, useJ2Energy, useResidualLadder, useResidualLadderFull
        )
        if phys_target_ce_t is None:
            phys_target_ce_t = np.zeros_like(y_joint_t, dtype=np.float32)
            phys_gate_ce_t = np.zeros_like(y_joint_t, dtype=np.float32)
        _, _, test_mask_t = _icGroupSplit(n_ic_per_class_t, train_ratio, val_ratio, test_ratio, seed=seed)
        test_data, test_joint = states_t[test_mask_t], y_joint_t[test_mask_t]
        test_phys_target_ce, test_phys_gate_ce = phys_target_ce_t[test_mask_t], phys_gate_ce_t[test_mask_t]
    else:
        test_data, test_joint = states[test_mask], y_joint[test_mask]
        test_phys_target_ce, test_phys_gate_ce = phys_target_ce[test_mask], phys_gate_ce[test_mask]

    # Standardization sits after the split (so statistics come from train only) and before the
    # loaders/raw arrays are handed out, keeping the neural and classic-ML paths on identical
    # features -- the Hankel-window baselines consume train_data/val_data/test_data directly.
    if standardize:
        train_data, val_data, test_data, _, _ = _standardizeSplits(
            train_data, val_data, test_data, supress_print=supress_print)

    train_stage1, train_stage2 = _deriveStage1Stage2(train_joint, pad_idx)
    val_stage1, val_stage2 = _deriveStage1Stage2(val_joint, pad_idx)
    test_stage1, test_stage2 = _deriveStage1Stage2(test_joint, pad_idx)

    if not supress_print:
        print(f"train_data {train_data.shape}  val_data {val_data.shape}  test_data {test_data.shape}")

    def _make_loader(data, yj, y1, y2, y_pt_ce, y_pg_ce, shuffle, sampler=None):
        ds = TensorDataset(
            torch.from_numpy(data),
            torch.from_numpy(yj).long(),
            torch.from_numpy(y1).long(),
            torch.from_numpy(y2).long(),
            torch.from_numpy(y_pt_ce).double(),
            torch.from_numpy(y_pg_ce).double(),
        )
        if sampler is not None:
            return DataLoader(ds, batch_size=batch_size, sampler=sampler, pin_memory=True)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    train_sampler = None
    if oversample:
        weights = _computeOversamplingWeights(train_joint, num_classes=len(JOINT_LABELS))
        train_sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)
        if not supress_print:
            print(f"Oversampling enabled: train_loader weights range [{weights.min():.3f}, {weights.max():.3f}]")

    train_loader = _make_loader(train_data, train_joint, train_stage1, train_stage2,
                                 train_phys_target_ce, train_phys_gate_ce,
                                 train_sampler is None, sampler=train_sampler)
    val_loader = _make_loader(val_data, val_joint, val_stage1, val_stage2,
                               val_phys_target_ce, val_phys_gate_ce, False)
    test_loader = _make_loader(test_data, test_joint, test_stage1, test_stage2,
                                test_phys_target_ce, test_phys_gate_ce, False)

    result = (train_loader, val_loader, test_loader,
              train_data, train_joint, val_data, val_joint, test_data, test_joint)

    if return_meta:
        meta = {
            "class_names_joint": JOINT_CLASS_NAMES,
            "class_names_stage1": STAGE1_CLASS_NAMES,
            "class_names_stage2": STAGE2_CLASS_NAMES,
            "n_ic_per_class": n_ic_per_class,
        }
        return result + (meta,)
    return result


# ---------------------------------------------------------------------------
# Models -- every backbone emits logits: [B, T, num_classes] (per-timestep, not pooled)
# ---------------------------------------------------------------------------
class LSTMSequenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """x: [batch_size, seq_length, input_size]"""
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        logits = self.classifier(out)  # [B, T, num_classes]
        return logits


class MambaSequenceClassifier(nn.Module):
    """Bi-directional Mamba: one forward scan and one over the time-reversed input, with the two
    per-timestep hidden states concatenated before the classification head.

    Mamba's selective scan is causal, which made this the ONLY backbone in the file that could not
    see a thrust event's trailing edge when labelling a timestep inside it -- LSTMSequenceClassifier
    sets bidirectional=True, TransformerSequenceClassifier applies no causal mask, and
    InceptionModule convolves with symmetric padding=k//2. So every Mamba-vs-other comparison here
    was partly a directionality comparison rather than an architecture one. The reverse scan removes
    that confound. Two independent Mamba stacks (not one with shared weights) -- forward and
    backward dynamics are genuinely different functions, and weight sharing would force one set of
    SSM parameters to model both. Costs ~2x the parameters and ~2x the training time."""
    def __init__(self, config, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mamba = Mamba(config)
        self.mamba_rev = Mamba(config)
        # config.d_inner (== expand_factor * d_model) is what a classifer=True Mamba returns per
        # timestep; build_model picks expand_factor so that equals hidden_size, but read it off the
        # config rather than re-deriving it here.
        self.fc = nn.Linear(config.d_inner * 2, num_classes)

    def forward(self, x):
        """x: [batch_size, seq_length, input_size]"""
        fwd = self.mamba(x)                        # [B, T, d_inner]
        rev = self.mamba_rev(x.flip(1)).flip(1)    # [B, T, d_inner], re-aligned to forward time
        logits = self.fc(torch.cat([fwd, rev], dim=-1))   # [B, T, num_classes]
        return logits


class TransformerSequenceClassifier(nn.Module):
    """Encoder-only Transformer with a learnable positional embedding, adapted from the
    whole-trajectory script's CLS-token-pooled TransformerClassifier. The CLS token is dropped
    entirely here -- per-timestep classification doesn't need a single global summary token, and
    keeping it would just complicate the position bookkeeping. The classification head is
    applied to every timestep of the encoder output instead."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, nhead=8, dim_feedforward=64, dropout=0.1, max_len=4096):
        super().__init__()

        self.d_model = hidden_size

        if self.d_model % nhead != 0:
            for cand in (8, 4, 2, 1):
                if self.d_model % cand == 0:
                    nhead = cand
                    break

        self.embedding = nn.Linear(input_size, self.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        """x: [batch_size, seq_length, input_size]"""
        seq_len = x.shape[1]
        x = self.embedding(x)                     # [B, T, d_model]
        x = x + self.pos_embedding[:, :seq_len, :]
        out = self.encoder(x)                      # [B, T, d_model]
        logits = self.fc(out)                       # [B, T, num_classes]
        return logits


class InceptionModule(nn.Module):
    """One InceptionTime module: a 1x1 bottleneck feeding parallel odd-kernel convs plus a
    max-pool branch, concatenated along channels. GroupNorm (not BatchNorm) so training is
    robust to the size-1 trailing batch that an undivided dataset can produce. All branches are
    same-length-preserving (padding=k//2 / maxpool stride=1,padding=1), so this needs no changes
    to support a per-timestep head downstream."""
    def __init__(self, in_channels, n_filters=32, kernel_sizes=(9, 19, 39), bottleneck_channels=32):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        bt_channels = bottleneck_channels if self.use_bottleneck else in_channels
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)

        self.convs = nn.ModuleList([
            nn.Conv1d(bt_channels, n_filters, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.maxpool_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        out_channels = n_filters * (len(kernel_sizes) + 1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [batch_size, channels, seq_length]
        bt = self.bottleneck(x) if self.use_bottleneck else x
        branches = [conv(bt) for conv in self.convs]
        branches.append(self.maxpool_conv(self.maxpool(x)))
        out = torch.cat(branches, dim=1)
        return self.act(self.norm(out))


class InceptionTimeSequenceClassifier(nn.Module):
    """Per-timestep InceptionTime variant of the whole-trajectory script's
    InceptionTimeClassifier: identical stacked-module/residual backbone (all
    same-length-preserving), but skips the final global-average-pool and applies a 1x1 conv head
    at every timestep instead of a single pooled vector."""
    def __init__(self, input_size, num_classes, n_filters=32, kernel_sizes=(9, 19, 39), depth=6):
        super().__init__()
        out_channels = n_filters * (len(kernel_sizes) + 1)

        self.inception_modules = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        in_ch = input_size
        for d in range(depth):
            self.inception_modules.append(InceptionModule(in_ch, n_filters=n_filters, kernel_sizes=kernel_sizes))
            if d % 3 == 2:
                shortcut_in = input_size if d == 2 else out_channels
                self.shortcuts.append(nn.Sequential(
                    nn.Conv1d(shortcut_in, out_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=out_channels),
                ))
            in_ch = out_channels

        self.head = nn.Conv1d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        """x: [batch_size, seq_length, input_size]"""
        x = x.transpose(1, 2)  # [B, C, T]
        res_input = x
        shortcut_idx = 0
        for d, module in enumerate(self.inception_modules):
            x = module(x)
            if d % 3 == 2:
                shortcut = self.shortcuts[shortcut_idx](res_input)
                shortcut_idx += 1
                x = torch.relu(x + shortcut)
                res_input = x
        logits = self.head(x)           # [B, num_classes, T]
        return logits.transpose(1, 2)   # [B, T, num_classes]


def build_model(backbone, num_classes, input_size, hidden_size, num_layers):
    if backbone == "lstm":
        return LSTMSequenceClassifier(input_size, int(3 * hidden_size // 4), num_layers, num_classes)
    elif backbone == "mamba":
        config = MambaConfig(d_model=input_size, n_layers=num_layers, expand_factor=hidden_size // input_size,
                              d_state=32, d_conv=4, classifer=True)
        return MambaSequenceClassifier(config, input_size, hidden_size, num_layers, num_classes)
    elif backbone == "transformer":
        return TransformerSequenceClassifier(input_size, hidden_size, num_layers, num_classes)
    elif backbone == "cnn":
        return InceptionTimeSequenceClassifier(input_size, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


# ---------------------------------------------------------------------------
# "hybrid" backbone -- not a single per-timestep nn.Module like the ones above. Stage 1
# (detector) is a whole-trajectory MiniRocket classifier: the same num_kernels=10000,
# rocket_transform='minirocket' RocketClassifier already used for whole-trajectory comparisons
# in mambaTimeSeriesClassificationGMATThrusts.py, fit on the *entire* ~30-step trajectory rather
# than a sliding window. MiniRocket's PPV-pooling transform is calibrated to and gets its power
# from the full series it's fit on -- an early windowed-per-timestep version of this backbone
# chopped each trajectory into many short, heavily-overlapping sub-series, starving the kernels
# of signal and working against MiniRocket's actual design. Stage 2 (type classifier) is the
# same CNN (InceptionTime) per-timestep model used elsewhere in this script. Consequently
# 'hybrid' only participates in cascade/stage-solo modes -- there is no single joint 4-class
# hybrid model, and stage 1's binary decision applies to a whole trajectory ("does this
# ~30-minute window contain a thrust event anywhere"), not to individual timesteps.
# ---------------------------------------------------------------------------
def printMiniROCKETSize(model):
    import pickle
    size_bytes = len(pickle.dumps(model))
    num_kernels = model.num_kernels_
    print("\n==========================================================================================")
    print(f"Total parameters: {num_kernels}")
    print(f"Total memory (bytes): {size_bytes}")
    print(f"Total memory (MB): {size_bytes / (1024 ** 2):.4f}")
    print("==========================================================================================")


def trainMiniRocketStage1Detector(train_data, train_stage1, num_kernels=10000):
    """train_data: [N,T,C]; train_stage1: [N,T] per-timestep binary labels, reduced here to one
    whole-trajectory label per row (1 if thrust occurs anywhere in that trajectory)."""
    from sktime.classification.kernel_based import RocketClassifier
    X_train = np.transpose(train_data, (0, 2, 1))         # [N,T,C] -> [N,C,T] sktime panel format
    y_train = train_stage1.any(axis=1).astype(np.int64)
    clf = RocketClassifier(num_kernels=num_kernels, rocket_transform='minirocket', n_jobs=-1)
    clf.fit(X_train, y_train)
    printMiniROCKETSize(clf)
    return clf


def predictMiniRocketStage1Trajectory(clf, data):
    """Returns preds[N] in {0,1} -- one whole-trajectory 'thrust present' decision per row."""
    X = np.transpose(data, (0, 2, 1))
    return np.asarray(clf.predict(X)).astype(np.int64)


# ---------------------------------------------------------------------------
# Training / evaluation infra -- shared across joint (4-class), stage1 (binary), and
# stage2 (3-class, background-masked) modes via the `mode` parameter.
# ---------------------------------------------------------------------------
def _select_labels(y_joint, y_stage1, y_stage2, mode):
    if mode == "joint":
        return y_joint
    elif mode == "stage1":
        return y_stage1
    elif mode == "stage2":
        return y_stage2
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _default_label_sets(mode, num_classes):
    """(event_class_labels, all_class_labels) used for P/R/F1 reporting during training."""
    if mode == "joint":
        return list(range(1, num_classes)), list(range(num_classes))
    elif mode == "stage1":
        return [1], [0, 1]
    elif mode == "stage2":
        return list(range(num_classes)), list(range(num_classes))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _infer_class_weights(loader, num_classes, mode, pad_idx=-100, dtype=torch.float32, device="cpu",
                          scheme="effective", beta=0.999):
    """scheme='effective': class-balanced weights from the effective number of samples
    (Cui et al. 2019, 'Class-Balanced Loss Based on Effective Number of Samples'),
    weight_c ~ (1-beta) / (1 - beta**n_c). Scales more gently than raw inverse frequency as
    beta -> 1, which matters here since the NoThrust:other ratio swings from ~2:1 to ~99:1
    across class files/modes -- plain inverse frequency would give the rare classes wildly
    different weight magnitudes across runs. scheme='inverse': plain N/count_c weighting."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    with torch.no_grad():
        for _, y_joint, y_stage1, y_stage2, _, _ in loader:
            y = _select_labels(y_joint, y_stage1, y_stage2, mode).reshape(-1).long()
            if pad_idx is not None:
                y = y[y != pad_idx]
            counts += torch.bincount(y, minlength=num_classes)
    counts = counts.to(dtype)
    if scheme == "effective":
        effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=dtype), torch.clamp(counts, min=1.0))
        w = (1.0 - beta) / torch.clamp(effective_num, min=1e-12)
    elif scheme == "inverse":
        w = counts.sum() / torch.clamp(counts, min=1.0)
    else:
        raise ValueError(f"Unknown loss scheme: {scheme}")
    w = w / w.mean()
    return w.to(device=device, dtype=dtype)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017) with per-class weight and ignore_index support,
    matching nn.CrossEntropyLoss's interface. Down-weights the loss contribution of
    already-well-classified timesteps -- typically the dominant NoThrust background -- so
    gradient stays concentrated on hard/ambiguous frames (e.g. thrust onset/offset) rather than
    on class rarity alone, complementing the --loss-scheme class weights."""

    def __init__(self, gamma=2.0, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        valid = targets != self.ignore_index
        if not valid.any():
            return torch.tensor(float('nan'), device=logits.device, dtype=logits.dtype)
        ce = nn.functional.cross_entropy(logits[valid], targets[valid], weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def physicsConsistencyLoss(logits, mode, y_phys_target, y_phys_gate, eps=1e-8):
    """logits: [B,T,C] (mode in ('joint','stage2') only); y_phys_target/y_phys_gate: [B,T] float
    (see _load_and_label's usePhysicsLoss branch -- gate is already 0 outside Chemical/Electric-
    labeled frames, and naturally shrinks toward 0 at high altitude, since h2=a_J2/g0 and
    h36=a_J3-6/g0 fall off steeply with r while thruster accel does not).

    Chemical-vs-Electric magnitude-scale consistency. Reuses the model's OWN existing logit margin
    as a binary prediction (no new head, no new parameters):
      mode='joint':  margin = logits[...,Chemical] - logits[...,Electric]
      mode='stage2': margin = logits[...,0] - logits[...,1]  (Chemical=0, Electric=1 --
                      _deriveStage1Stage2)

    A Thrust-vs-NoThrust ('detect') term was removed after measurement refuted its premise -- see
    the note in _load_and_label and scripts/two_body/analyzeThrustSeparability.py. This term's own
    premise did verify: Chemical's measured residual (~3.4e-5 km/s^2) sits within ~3x of a_J2 and
    ~12x above Electric's (~2.9e-6), a real and separable magnitude gap.

    Trains that margin toward the physics pseudo-target via BCEWithLogitsLoss, weighted
    per-timestep by y_phys_gate, averaged only over gated frames in the batch. Returns a scalar
    0.0 (never nan) when a batch has no gated frames, so it can never poison the additive total
    loss via torch.isnan."""
    if mode == "joint":
        margin = logits[..., JOINT_LABELS["Chemical"]] - logits[..., JOINT_LABELS["Electric"]]
    elif mode == "stage2":
        margin = logits[..., 0] - logits[..., 1]   # Chemical=0, Electric=1 -- _deriveStage1Stage2
    else:
        raise ValueError(f"physicsConsistencyLoss is only defined for mode in ('joint','stage2'), got {mode}")

    per_timestep = nn.functional.binary_cross_entropy_with_logits(margin, y_phys_target, reduction='none')
    gate_sum = y_phys_gate.sum()
    if gate_sum <= eps:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return (per_timestep * y_phys_gate).sum() / gate_sum


def train_model(model, train_loader, val_loader, num_epochs, num_classes, mode,
                 pad_idx=-100, class_weights=None, schedulerPatience=3, verbose=True,
                 loss_scheme=None, cb_beta=None, focal_gamma=None, physics_loss_weight=None,
                 restore_best=True, lr=1e-3, restore_metric="loss"):
    """restore_best: on return, load back the weights from the epoch with the lowest validation
    loss instead of leaving the model at its final epoch. Early stopping already tracks that
    epoch; without the restore, training continues for ESpatience epochs past the optimum and the
    caller evaluates whatever state it drifted into. That biases any comparison across capacity or
    learning rate specifically, since larger models and higher learning rates overfit furthest in
    those trailing epochs -- exactly the configurations such a comparison exists to measure.
    Pass False to reproduce the previous last-epoch behaviour.

    restore_metric: which validation signal defines "best", for both the restore and the
    early-stopping counter. 'loss' (default) keeps existing behaviour. 'event_f1' tracks event
    macro-F1 instead, and is what any caller ranking models by F1 should use -- under this
    dataset's imbalance the two do not coincide, and checkpointing on loss measurably selects the
    worse-F1 model (0.4448 last-epoch vs 0.4096 best-loss on one leo/30min LSTM fit). Track the
    metric you are actually selecting on."""
    model = model.to(device)
    param_dtype = torch.float64
    model = model.double()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_scheme = lossScheme if loss_scheme is None else loss_scheme
    cb_beta = cbBeta if cb_beta is None else cb_beta
    focal_gamma = focalGamma if focal_gamma is None else focal_gamma
    physics_loss_weight = physicsLossWeight if physics_loss_weight is None else physics_loss_weight

    if class_weights is None:
        class_weights = _infer_class_weights(train_loader, num_classes, mode, pad_idx, dtype=param_dtype,
                                              device=device, scheme=loss_scheme, beta=cb_beta)
    else:
        class_weights = torch.as_tensor(class_weights, device=device, dtype=param_dtype)

    if focal_gamma and focal_gamma > 0:
        criterion = FocalLoss(gamma=focal_gamma, weight=class_weights, ignore_index=pad_idx)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=pad_idx)
    event_labels, all_labels = _default_label_sets(mode, num_classes)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=schedulerPatience)

    best_loss = float('inf')
    # best_score is compared in the direction restore_metric implies: minimized for 'loss',
    # maximized for 'event_f1'. best_loss stays a pure record of the lowest loss seen, for logging.
    best_score = float('inf') if restore_metric == "loss" else -float('inf')
    ESpatience = schedulerPatience * 2
    counter = 0
    best_state, best_epoch = None, -1

    timeToTrain = timer()

    # The physics term applies to joint/stage2, where the Chemical and Electric logits exist side
    # by side. stage1 has no per-type logits, so it is out of scope -- a Thrust-vs-NoThrust term
    # that once covered it was removed after measurement refuted its premise (see _load_and_label).
    use_physics_loss = bool(physics_loss_weight) and physics_loss_weight > 0 and mode in ("joint", "stage2")

    for epoch in range(num_epochs):
        model.train()
        total_loss, loss_batches = 0.0, 0
        total_phys_loss = 0.0
        skipped_train_batches = 0
        for x, y_joint, y_stage1, y_stage2, y_phys_target, y_phys_gate in train_loader:
            x = x.to(device, non_blocking=True)
            if x.dtype != param_dtype:
                x = x.to(param_dtype)
            labels = _select_labels(y_joint, y_stage1, y_stage2, mode).to(device, non_blocking=True).long()

            logits = model(x)
            B, T, C = logits.shape
            loss = criterion(logits.reshape(B * T, C), labels.reshape(B * T))

            if torch.isnan(loss):
                # every position in this batch was pad_idx (e.g. a stage2 batch with no
                # thrusting frames at all) -- ignore_index reduction has nothing to average
                # over. Skip the update: backward() on a nan loss would poison every
                # parameter with nan permanently.
                skipped_train_batches += 1
                continue

            batch_phys_loss = 0.0
            if use_physics_loss:
                y_pt = y_phys_target.to(device, non_blocking=True).to(param_dtype)
                y_pg = y_phys_gate.to(device, non_blocking=True).to(param_dtype)
                phys_loss = physicsConsistencyLoss(logits, mode, y_pt, y_pg)
                loss = loss + physics_loss_weight * phys_loss
                batch_phys_loss = phys_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_phys_loss += batch_phys_loss
            loss_batches += 1

        avg_loss = total_loss / max(1, loss_batches)
        if verbose:
            msg = f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f}"
            if use_physics_loss:
                msg += f"  (avg physics-loss term: {total_phys_loss / max(1, loss_batches):.4f})"
            if skipped_train_batches:
                msg += f"  ({skipped_train_batches} batch(es) skipped: no unmasked labels)"
            print(msg)

        model.eval()
        all_preds, all_targets = [], []
        val_loss, val_loss_batches = 0.0, 0
        with torch.no_grad():
            for x, y_joint, y_stage1, y_stage2, y_phys_target, y_phys_gate in val_loader:
                x = x.to(device, non_blocking=True)
                if x.dtype != param_dtype:
                    x = x.to(param_dtype)
                labels = _select_labels(y_joint, y_stage1, y_stage2, mode).to(device, non_blocking=True).long()

                logits = model(x)
                B, T, C = logits.shape
                loss = criterion(logits.reshape(B * T, C), labels.reshape(B * T))
                if use_physics_loss:
                    y_pt = y_phys_target.to(device, non_blocking=True).to(param_dtype)
                    y_pg = y_phys_gate.to(device, non_blocking=True).to(param_dtype)
                    loss = loss + physics_loss_weight * physicsConsistencyLoss(logits, mode, y_pt, y_pg)
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_loss_batches += 1

                preds = logits.argmax(dim=-1)
                mask = labels != pad_idx
                all_preds.append(preds[mask].detach().cpu())
                all_targets.append(labels[mask].detach().cpu())

        avg_val_loss = val_loss / max(1, val_loss_batches)
        y_pred = torch.cat(all_preds).numpy() if all_preds else np.array([])
        y_true = torch.cat(all_targets).numpy() if all_targets else np.array([])

        # Event macro-F1 is computed every epoch regardless of verbosity, because restore_metric
        # ='event_f1' checkpoints on it. y_true/y_pred are already materialized, so this is cheap.
        val_event_f1 = 0.0
        if len(y_true) > 0:
            p_ev, r_ev, val_event_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=event_labels, average='macro', zero_division=0)
            if verbose:
                print(f"Val Event P(macro {event_labels}): {p_ev:.4f} | R: {r_ev:.4f} | F1: {val_event_f1:.4f}")
                p_pc, r_pc, f_pc, _ = precision_recall_fscore_support(y_true, y_pred, labels=all_labels, average=None, zero_division=0)
                print(f"Per-class P: {p_pc}  R: {r_pc}  F1: {f_pc}")
        print(f"Val Loss: {avg_val_loss:.4f}")

        # The LR schedule always follows val loss -- it is the smoother signal, and F1 moves in
        # discrete jumps as argmax decisions flip, which makes it a poor plateau detector.
        scheduler.step(avg_val_loss)

        improved = (val_event_f1 > best_score) if restore_metric == "event_f1" \
            else (avg_val_loss < best_score)
        if improved:
            best_score = val_event_f1 if restore_metric == "event_f1" else avg_val_loss
            best_loss = min(best_loss, avg_val_loss)
            counter = 0
            if restore_best:
                # .cpu().clone() so the snapshot survives later in-place parameter updates and
                # does not pin a second copy of the model in GPU memory.
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
        else:
            counter += 1
            if counter >= ESpatience:
                print("Early stopping triggered.")
                break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"Restored best weights from epoch {best_epoch + 1} "
                  f"(best {restore_metric} {best_score:.4f}, lowest val loss {best_loss:.4f}).")

    return timeToTrain.toc()


def _reportFromPredictions(y_true, y_pred, class_names, print_report=True):
    """Single formatting/metrics path every model family (neural, and classic-ML in a later
    phase) reports through, so results are directly comparable across the whole script."""
    num_classes = len(class_names)
    labels = list(range(num_classes))

    n = max(1, len(y_true))
    accuracy = 100.0 * (y_true == y_pred).sum() / n

    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if print_report:
        print(f"Accuracy: {accuracy:.2f}% ({int((y_true == y_pred).sum())}/{len(y_true)})")
        print("Per-Class Accuracy:")
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                print(f"  {class_names[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
            else:
                print(f"  {class_names[i]}: No samples")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0))

        print("\nConfusion Matrix (rows = true, cols = predicted):")
        print(pd.DataFrame(cm, index=[f"T_{c}" for c in class_names], columns=[f"P_{c}" for c in class_names]))

    return {"accuracy": accuracy, "class_correct": class_correct, "class_total": class_total, "confusion_matrix": cm}


def _findSegments(row, c):
    """Maximal contiguous (start,end) inclusive index runs where row == c."""
    is_c = (row == c).astype(np.int8)
    d = np.diff(np.concatenate(([0], is_c, [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts, ends))


def _eventLevelReport(y_true_grid, y_pred_grid, class_names, print_report=True,
                       valid_from=0, granularity_note=None):
    """y_true_grid/y_pred_grid: [N,T] int arrays in JOINT_LABELS space (index 0 = NoThrust/
    background). Complements _reportFromPredictions's per-timestep numbers with segment-level
    precision/recall per nonzero class -- catches cases where per-timestep accuracy looks fine but
    the model is scattering wrong-class predictions across a trajectory ("flicker"), which
    per-timestep metrics average away.

    A true/predicted event is a maximal contiguous run of one class along the time axis for one
    row (_findSegments). Detection is "any overlap" (point-adjust, Xu et al. 2018): a true event
    counts as recalled if ANY timestep within its span is predicted as that class; a predicted
    event counts as a false positive only if it has ZERO overlap with any true event of that class
    in that row.

    valid_from: columns before this index have no real prediction (Hankel-window classic-ML/
    PCA+MLP grids hardcode the first hankel_L-1 timesteps to background -- see
    predictClassicPerTimestep/predictPCAMLPPerTimestep). True events entirely before valid_from
    are excluded from the recall denominator (no real prediction ever had a chance to catch them)
    rather than scored as automatic misses -- matching how evaluateClassicPerTimestep already
    drops those frames from the per-timestep metric. Predicted-segment search always covers the
    full row (the padded prefix is hardcoded to class 0, so it can never contribute a spurious
    nonzero segment).

    granularity_note: optional caveat printed under the table, for predictions whose "segment" is
    coarser than genuine per-timestep localization (e.g. MiniRocket/hybrid-stage1's whole-
    trajectory broadcast, where recall collapses to whole-trajectory detection).
    """
    assert -100 not in np.unique(y_true_grid) and -100 not in np.unique(y_pred_grid), \
        "_eventLevelReport expects JOINT_LABELS-space grids (no pad_idx) -- got a masked/stage2 grid"

    num_classes = len(class_names)
    rows_out = []
    for c in range(1, num_classes):  # skip background, same convention as _default_label_sets' event_labels
        n_true = n_pred = tp_true = tp_pred = 0
        for i in range(y_true_grid.shape[0]):
            true_row, pred_row = y_true_grid[i], y_pred_grid[i]
            for (s, e) in _findSegments(true_row, c):
                if e < valid_from:
                    continue  # wholly inside the no-prediction prefix -- exclude, don't penalize
                n_true += 1
                if np.any(pred_row[max(s, valid_from):e + 1] == c):
                    tp_true += 1
            for (s, e) in _findSegments(pred_row, c):
                n_pred += 1
                if np.any(true_row[s:e + 1] == c):
                    tp_pred += 1
        recall = tp_true / n_true if n_true else float('nan')
        precision = tp_pred / n_pred if n_pred else float('nan')
        rows_out.append((class_names[c], n_true, n_pred, recall, precision))

    if print_report:
        print("\nEvent-level (segment) metrics -- 'any overlap' detection per contiguous true/predicted run:")
        print(f"{'Class':<12}{'TrueEvents':>12}{'PredEvents':>12}{'Recall':>10}{'Precision':>12}")
        for name, nt, npred, r, p in rows_out:
            r_str = f"{r*100:.2f}%" if nt else "n/a"
            p_str = f"{p*100:.2f}%" if npred else "n/a"
            print(f"{name:<12}{nt:>12}{npred:>12}{r_str:>10}{p_str:>12}")
        print("Caveat: Impulsive's forward-filled label (burn instant -> end of window) makes its "
              "'any overlap' recall/precision structurally easier to satisfy than Chemical/"
              "Electric's bounded bursts -- not directly comparable across classes. Also note "
              "'any overlap' recall alone does not penalize flicker (a correctly-recalled true "
              "event can still contain stray wrong-class guesses); it's a spurious class's own "
              "precision that catches that.")
        if granularity_note:
            print(f"Note: {granularity_note}")

    return {name: {"true_events": nt, "pred_events": npred, "recall": r, "precision": p}
            for name, nt, npred, r, p in rows_out}


def smoothPerTimestepGrid(pred_grid, max_gap, no_thrust_label=0):
    """pred_grid: [N,T] int labels in JOINT_LABELS space. Closes short INTERIOR gaps in the binary
    'any thrust' view: a run of no_thrust_label frames of length <= max_gap that has a thrusting
    prediction immediately before AND after it (i.e. touches neither end of the row) gets filled
    in. This ONLY ever closes gaps -- it never erases a predicted-positive run, however short. That
    asymmetry is deliberate: a majority-vote window or a minimum-run-length filter erases short
    positive runs, and a lone true positive is indistinguishable from a lone false positive by
    shape alone, so those filters trade real detections for flicker removal. Gap-closing has no
    such failure mode -- it fills the model's missed frames INSIDE one real event and never touches
    an isolated positive, which is why Chemical's event recall is measurably unchanged by it while
    the longer classes can only gain (see sweepSmoothGap.py).

    The 's == 0 or e == T-1' edge check also transparently handles the classic-ML/PCA+MLP Hankel-
    window padded prefix (see buildHankelWindowRowsPerTimestep/predictClassicPerTimestep): a
    no_thrust_label run touching the row's start is never closed, whether it's a genuine leading
    NoThrust stretch or a hardcoded pad -- no hankel_L-specific handling needed here.

    Filled frames' TYPE (which of Chemical/Electric/Impulsive) is forward-filled from the frame
    immediately before the gap, matching this file's existing Impulsive forward-fill precedent
    (_getThrustingTime/_forwardFillFromFirstEvent) rather than inventing a new imputation rule.

    max_gap <= 0 is a no-op (returns pred_grid unchanged, default/off behavior)."""
    if max_gap <= 0:
        return pred_grid
    out = pred_grid.copy()
    N, T = pred_grid.shape
    for i in range(N):
        row = pred_grid[i]
        for (s, e) in _findSegments(row, no_thrust_label):
            if s == 0 or e == T - 1:
                continue  # touches an edge -- not an interior gap, leave as-is
            if (e - s + 1) <= max_gap:
                out[i, s:e + 1] = row[s - 1]  # forward-fill type from the frame just before the gap
    return out


def _reportEventLevelWithSmoothing(y_true_grid, pred_grid, class_names, smooth_max_gap=0,
                                    plot_name=None, plot_save_path=None, mode_label="Joint",
                                    valid_from=0, granularity_note=None, print_report=True):
    """Shared event-level report+plot hook for every joint/cascade call site in the file (see
    smoothPerTimestepGrid). Runs _eventLevelReport (+ plotSequencePrediction, if plot_name/
    plot_save_path given) on the RAW pred_grid exactly as before this flag existed, then -- only
    if smooth_max_gap > 0 -- a second, clearly labeled pass on the gap-closed grid, saved to a
    '_smoothed'-suffixed plot path. The raw report always runs and is never replaced, so the
    payoff (or lack thereof) of --smooth-max-gap is always visible side by side rather than
    silently swapped in.

    Deliberately does NOT feed into _reportFromPredictions' flat per-timestep accuracy/
    classification-report numbers anywhere -- those stay on raw predictions everywhere in this
    file. Gap-closing is an event-shape correction, not a general accuracy claim, and per-timestep
    accuracy is dominated by the NoThrust majority class regardless, so it wouldn't move much and
    would just blur what this flag is actually for."""
    _eventLevelReport(y_true_grid, pred_grid, class_names, print_report=print_report,
                       valid_from=valid_from, granularity_note=granularity_note)
    if plot_name is not None and plot_save_path is not None:
        plotSequencePrediction(plot_name, y_true_grid, pred_grid, class_names, plot_save_path,
                                mode_label=mode_label)

    if smooth_max_gap <= 0:
        return pred_grid

    smoothed = smoothPerTimestepGrid(pred_grid, smooth_max_gap)
    print(f"\n--- After temporal gap-closing (--smooth-max-gap {smooth_max_gap}) ---")
    _eventLevelReport(y_true_grid, smoothed, class_names, print_report=print_report,
                       valid_from=valid_from, granularity_note=granularity_note)
    if plot_name is not None and plot_save_path is not None:
        stem, ext = os.path.splitext(plot_save_path)
        plotSequencePrediction(plot_name + " (smoothed)", y_true_grid, smoothed, class_names,
                                stem + "_smoothed" + ext, mode_label=mode_label)
    return smoothed


def validateInSequenceClassifier(model, loader, mode, num_classes, device, pad_idx=-100,
                                  class_names=None, print_report=True, return_predictions=False):
    model.eval()
    param_dtype = torch.float64
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y_joint, y_stage1, y_stage2, _, _ in loader:
            x = x.to(device, non_blocking=True)
            if x.dtype != param_dtype:
                x = x.to(param_dtype)
            labels = _select_labels(y_joint, y_stage1, y_stage2, mode).to(device, non_blocking=True).long()

            logits = model(x)
            preds = logits.argmax(dim=-1)

            mask = labels != pad_idx
            all_preds.append(preds[mask].detach().cpu())
            all_targets.append(labels[mask].detach().cpu())

    y_pred = torch.cat(all_preds).numpy() if all_preds else np.array([])
    y_true = torch.cat(all_targets).numpy() if all_targets else np.array([])

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    result = _reportFromPredictions(y_true, y_pred, class_names, print_report=print_report)
    if return_predictions:
        result["y_true"] = y_true
        result["y_pred"] = y_pred
    return result


def _predictPerTimestepNeural(model, loader, device):
    """Runs a per-timestep classifier over a non-shuffled loader, returns raw (unmasked, un-
    flattened) [N,T] arrays in loader order: (y_joint, y_stage1, predictions)."""
    model.eval()
    param_dtype = torch.float64
    all_pred, all_joint, all_stage1 = [], [], []
    with torch.no_grad():
        for x, y_joint, y_stage1, y_stage2, _, _ in loader:
            x = x.to(device, non_blocking=True)
            if x.dtype != param_dtype:
                x = x.to(param_dtype)
            preds = model(x).argmax(dim=-1)  # [B,T]
            all_pred.append(preds.cpu().numpy())
            all_joint.append(y_joint.numpy())
            all_stage1.append(y_stage1.numpy())
    return np.concatenate(all_joint, axis=0), np.concatenate(all_stage1, axis=0), np.concatenate(all_pred, axis=0)


def plotSequencePrediction(model_name, y_true, y_pred, class_names, save_path, mode_label="Joint"):
    """Plots true vs. predicted per-timestep class labels over time, one panel per class present
    in y_true, using a representative example trajectory for each -- so a single figure shows how
    well {model_name} tracks every class the model has to distinguish, not just one example.

    y_true/y_pred: [N, T] per-timestep label grids in the SAME label space as class_names (index i
    -> class_names[i]). That's the only requirement, so this same function plots joint 4-class
    predictions from ANY backbone family: per-timestep neural models (via _predictPerTimestepNeural),
    classic-ML/GBDT Hankel-window models (via predictClassicPerTimestep), PCA+MLP (via
    predictPCAMLPPerTimestep), or a whole-trajectory model's single decision broadcast across every
    timestep (MiniRocket/hybrid stage 1) -- whatever produced y_pred, once it's an [N, T] grid in
    this label space.
    """
    num_classes = len(class_names)
    example_rows = {}
    for c in range(num_classes):
        counts = (y_true == c).sum(axis=1)
        rows = np.where(counts > 0)[0]
        if rows.size:
            # row with the MOST timesteps of class c, not just the first row containing any --
            # background (class 0) appears in nearly every trajectory as padding around the
            # actual event, so "first row containing any" trivially matches whatever trajectory
            # happens to be first in eval order, even if it's dominated by a different class.
            example_rows[c] = int(rows[np.argmax(counts[rows])])

    classes_present = sorted(example_rows.keys())
    if not classes_present:
        print(f"[plotSequencePrediction] no examples found for {model_name} {mode_label}; skipping plot.")
        return None

    fig, axes = plt.subplots(len(classes_present), 1, figsize=(10, 2.6 * len(classes_present)),
                              sharex=True, squeeze=False)
    axes = axes[:, 0]

    T = y_true.shape[1]
    t = np.arange(T)
    for ax, c in zip(axes, classes_present):
        idx = example_rows[c]
        ax.step(t, y_true[idx], where='post', label='True', linewidth=2, color='black')
        ax.step(t, y_pred[idx], where='post', label='Predicted', linewidth=1.5, linestyle='--', color='tab:orange')
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels(class_names)
        ax.set_ylim(-0.5, num_classes - 0.5)
        ax.set_ylabel("Class")
        ax.set_title(f"Example trajectory containing '{class_names[c]}' (row {idx})", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"{model_name} {mode_label} -- Per-Timestep Sequence Prediction")
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved sequence prediction plot -> {save_path}")
    return save_path


def combineCascadePredictions(y_true_joint, y_true_stage1, pred_stage1, pred_stage2, print_report=True,
                               plot_name=None, plot_save_path=None, valid_from=0, granularity_note=None,
                               smooth_max_gap=0):
    """y_true_joint/y_true_stage1/pred_stage1: [N,T]; pred_stage2: [N,T] in {0,1,2}. Combines
    stage1 (thrust yes/no) and stage2 (Chemical/Electric/Impulsive) predictions into a single
    4-class per-timestep prediction and reports both the combined result and a stage1-vs-stage2
    error decomposition -- so a cascade shortfall is diagnosable as bad detection vs. bad typing.
    Backbone-agnostic: works whether stage1/stage2 came from matching neural models or the mixed
    MiniRocket-detector + CNN-type-classifier 'hybrid' backbone.

    If plot_name/plot_save_path are given, also saves a seqpred plot of the combined 4-class
    result against y_true_joint (same per-timestep grid plotSequencePrediction uses for joint
    models), so cascade predictions are visually comparable to the joint models' plots.

    valid_from/granularity_note are forwarded to _eventLevelReport (see there) -- pass
    valid_from=hankel_L-1 for classic-ML/PCA+MLP cascades (their grids pad the first hankel_L-1
    timesteps with a hardcoded background prediction) and a granularity_note for whole-trajectory-
    broadcast stage-1 decisions (MiniRocket/hybrid).

    smooth_max_gap: forwarded to _reportEventLevelWithSmoothing (see smoothPerTimestepGrid) --
    0 (default) is a no-op. Only the event-level report/plot below sees the gap-closed grid; the
    stage1/end-to-end flat metrics after this block always use the RAW (unsmoothed) final_pred."""
    final_pred = np.where(pred_stage1 == 0, 0, pred_stage2 + 1)

    _reportEventLevelWithSmoothing(y_true_joint, final_pred, JOINT_CLASS_NAMES,
                                    smooth_max_gap=smooth_max_gap,
                                    plot_name=plot_name, plot_save_path=plot_save_path,
                                    mode_label="Cascade", valid_from=valid_from,
                                    granularity_note=granularity_note, print_report=print_report)

    y_true_joint = y_true_joint.reshape(-1)
    y_true_stage1 = y_true_stage1.reshape(-1)
    pred_stage1 = pred_stage1.reshape(-1)
    final_pred = final_pred.reshape(-1)

    print("\n--- Cascade Stage 1 (Detector) Standalone Metrics ---")
    _reportFromPredictions(y_true_stage1, pred_stage1, STAGE1_CLASS_NAMES, print_report=print_report)

    print("\n--- Cascade End-to-End (Stage1 -> Stage2 combined) Metrics ---")
    result = _reportFromPredictions(y_true_joint, final_pred, JOINT_CLASS_NAMES, print_report=print_report)

    true_positive_mask = (y_true_stage1 == 1) & (pred_stage1 == 1)
    if true_positive_mask.sum() > 0:
        stage2_conditional_acc = 100.0 * (final_pred[true_positive_mask] == y_true_joint[true_positive_mask]).sum() / true_positive_mask.sum()
        print(f"\nStage-2 type accuracy conditioned on correct stage-1 detection: {stage2_conditional_acc:.2f}% ({int(true_positive_mask.sum())} frames)")
    else:
        print("\nNo frames with correct stage-1 positive detection to condition stage-2 accuracy on.")

    stage1_only_acc = 100.0 * (pred_stage1 == y_true_stage1).sum() / max(1, len(y_true_stage1))
    print(f"Stage-1 (detector-only) accuracy: {stage1_only_acc:.2f}%")

    return result


def runCascadeEvaluation(stage1_model, stage2_model, loader, device, print_report=True,
                          plot_name=None, plot_save_path=None, smooth_max_gap=0):
    """Neural-neural cascade: both stages are per-timestep nn.Modules over the same loader."""
    y_true_joint, y_true_stage1, pred_stage1 = _predictPerTimestepNeural(stage1_model, loader, device)
    _, _, pred_stage2 = _predictPerTimestepNeural(stage2_model, loader, device)
    return combineCascadePredictions(y_true_joint, y_true_stage1, pred_stage1, pred_stage2, print_report=print_report,
                                      plot_name=plot_name, plot_save_path=plot_save_path, smooth_max_gap=smooth_max_gap)


# ---------------------------------------------------------------------------
# Classic ML / GBDT per-timestep baselines (LightGBM/XGBoost/CatBoost/RandomForest/ExtraTrees).
# Unlike MiniRocket, tree-based models have no global-pooling requirement -- a windowed context
# feature vector per timestep is a perfectly natural row for them, so (unlike 'hybrid') these
# operate at genuine per-timestep granularity via a sliding trailing window, mirroring the
# whole-trajectory script's pca_mode="hankel" windowing minus its final mean-pool over time.
# ---------------------------------------------------------------------------
def buildHankelWindowRowsPerTimestep(states, labels, hankel_L=5, pad_idx=-100, drop_masked=True):
    """states: [N,T,C], labels: [N,T] -> X[M, C*hankel_L], y[M], row_index[M,2] (ic,t). One row
    per (IC, timestep) with t >= hankel_L-1; earlier timesteps lack enough left context and are
    dropped (a small, documented data loss -- e.g. ~13% of frames at hankel_L=5,T=30). If
    drop_masked, rows where labels==pad_idx are also dropped (GBDTs/sklearn have no ignore_index
    concept). Rows come out ordered by timestep (outer) then IC (inner) -- relied on by
    predictClassicPerTimestep to reshape flat predictions back to [N,T]."""
    N, T, C = states.shape
    rows_X, rows_y, rows_idx = [], [], []
    for t in range(hankel_L - 1, T):
        window = states[:, t - hankel_L + 1:t + 1, :].reshape(N, -1)  # [N, hankel_L*C]
        rows_X.append(window)
        rows_y.append(labels[:, t])
        rows_idx.append(np.stack([np.arange(N), np.full(N, t)], axis=1))
    X = np.concatenate(rows_X, axis=0).astype(np.float32)
    y = np.concatenate(rows_y, axis=0)
    idx = np.concatenate(rows_idx, axis=0)
    if drop_masked:
        keep = y != pad_idx
        X, y, idx = X[keep], y[keep], idx[keep]
    return X, y, idx


def _predictClassicLabels(model, X):
    """model.predict(X) is not uniformly shaped [M] across sklearn-API families: CatBoost always
    returns [M,1] regardless of class count, and XGBoost's multi:softmax/softprob distinction can
    also affect it. reshape(-1) is safe here since we only ever want hard labels (never called on
    genuine multi-column probability output) and is a no-op for already-1D predictions."""
    return np.asarray(model.predict(X)).reshape(-1).astype(np.int64)


def predictClassicPerTimestep(model, data, hankel_L=5):
    """Returns preds[N,T]. Timesteps t < hankel_L-1 default to 0 (background) -- not enough left
    context to build a window."""
    N, T, _ = data.shape
    preds = np.zeros((N, T), dtype=np.int64)
    if T < hankel_L:
        return preds
    dummy_labels = np.zeros((N, T), dtype=np.int64)
    X, _, _ = buildHankelWindowRowsPerTimestep(data, dummy_labels, hankel_L, drop_masked=False)
    y_pred = _predictClassicLabels(model, X)
    preds[:, hankel_L - 1:] = y_pred.reshape(T - hankel_L + 1, N).T
    return preds


def evaluateClassicPerTimestep(model, eval_data, eval_labels, class_names, hankel_L=5, pad_idx=-100, print_report=True):
    """Masked standalone evaluation (drop_masked=True), matching validateInSequenceClassifier's
    masking behavior for the neural backbones -- a no-op filter for joint/stage1 labels (which
    never contain pad_idx), and the intended positive-frames-only filter for stage2."""
    X_eval, y_eval, _ = buildHankelWindowRowsPerTimestep(eval_data, eval_labels, hankel_L, pad_idx=pad_idx, drop_masked=True)
    y_pred = _predictClassicLabels(model, X_eval)
    return _reportFromPredictions(y_eval, y_pred, class_names, print_report=print_report)


def runClassicFamily(name, classifier_ctor, num_classes, mode_name, class_names,
                      train_data, train_labels, eval_data, eval_labels,
                      hankel_L, pad_idx, size_fn, smooth_max_gap=0):
    """classifier_ctor: callable(num_classes) -> unfit sklearn-API classifier instance."""
    print(f"\nEntering {name} ({mode_name}) Training Loop")
    X_train, y_train, _ = buildHankelWindowRowsPerTimestep(train_data, train_labels, hankel_L, pad_idx=pad_idx, drop_masked=True)
    clf = classifier_ctor(num_classes)
    t = timer()
    clf.fit(X_train, y_train)
    t.toc()
    size_fn(clf)

    print(f"\n{name} ({mode_name}) Validation")
    tInf = timer()
    evaluateClassicPerTimestep(clf, eval_data, eval_labels, class_names, hankel_L=hankel_L, pad_idx=pad_idx, print_report=True)
    tInf.tocStr(f"{name} ({mode_name}) Inference Time")

    if mode_name == "joint":
        pred_grid = predictClassicPerTimestep(clf, eval_data, hankel_L)
        _reportEventLevelWithSmoothing(eval_labels, pred_grid, class_names, smooth_max_gap=smooth_max_gap,
                                        plot_name=name,
                                        plot_save_path=os.path.join(plotLoc, f"seqpred_{name.replace(' ', '_')}_joint_{logStem}.png"),
                                        mode_label="Joint", valid_from=hankel_L - 1)
    return clf


def runClassicCascade(name, classifier_ctor, train_data, train_joint, eval_data, eval_joint,
                       hankel_L, pad_idx, size_fn, smooth_max_gap=0):
    train_stage1, train_stage2 = _deriveStage1Stage2(train_joint, pad_idx)
    eval_stage1, eval_stage2 = _deriveStage1Stage2(eval_joint, pad_idx)

    print(f"\nEntering {name} Stage 1 (Detector) Training Loop")
    X_train1, y_train1, _ = buildHankelWindowRowsPerTimestep(train_data, train_stage1, hankel_L, pad_idx=pad_idx, drop_masked=True)
    clf1 = classifier_ctor(2)
    t = timer()
    clf1.fit(X_train1, y_train1)
    t.toc()
    size_fn(clf1)

    print(f"\nEntering {name} Stage 2 (Type Classifier) Training Loop")
    X_train2, y_train2, _ = buildHankelWindowRowsPerTimestep(train_data, train_stage2, hankel_L, pad_idx=pad_idx, drop_masked=True)
    clf2 = classifier_ctor(3)
    t = timer()
    clf2.fit(X_train2, y_train2)
    t.toc()
    size_fn(clf2)

    # Classic-ML models have no per-epoch training loop, so -- unlike the neural backbones, whose
    # Stage 2 gets a genuine standalone 3-class evaluation for free as part of its own per-epoch
    # validation -- Stage 2 here would otherwise never be scored except gated through Stage 1's
    # predictions in the combined cascade report below. Score it standalone first so cascade
    # shortfalls stay diagnosable as bad detection vs. bad typing, matching the neural comparison.
    print(f"\n{name} Stage 2 (Type Classifier) Standalone Validation")
    evaluateClassicPerTimestep(clf2, eval_data, eval_stage2, STAGE2_CLASS_NAMES,
                                hankel_L=hankel_L, pad_idx=pad_idx, print_report=True)

    print(f"\n{name} Cascade Evaluation")
    tInf = timer()
    pred_stage1 = predictClassicPerTimestep(clf1, eval_data, hankel_L)
    pred_stage2 = predictClassicPerTimestep(clf2, eval_data, hankel_L)
    combineCascadePredictions(eval_joint, eval_stage1, pred_stage1, pred_stage2, print_report=True,
                               plot_name=name,
                               plot_save_path=os.path.join(plotLoc, f"seqpred_{name.replace(' ', '_')}_cascade_{logStem}.png"),
                               valid_from=hankel_L - 1, smooth_max_gap=smooth_max_gap)
    tInf.tocStr(f"{name} Cascade Inference Time")
    return clf1, clf2


def runClassicMLModes(name, classifier_ctor, size_fn,
                       train_data, train_joint, train_stage1, train_stage2,
                       eval_data, eval_joint, eval_stage1, eval_stage2,
                       hankel_L, pad_idx,
                       run_joint, run_cascade, run_stage1_solo, run_stage2_solo, smooth_max_gap=0):
    if run_joint:
        runClassicFamily(name, classifier_ctor, 4, "joint", JOINT_CLASS_NAMES,
                          train_data, train_joint, eval_data, eval_joint, hankel_L, pad_idx, size_fn,
                          smooth_max_gap=smooth_max_gap)
    if run_cascade:
        runClassicCascade(name, classifier_ctor, train_data, train_joint, eval_data, eval_joint,
                           hankel_L, pad_idx, size_fn, smooth_max_gap=smooth_max_gap)
    if run_stage1_solo:
        runClassicFamily(name, classifier_ctor, 2, "stage1", STAGE1_CLASS_NAMES,
                          train_data, train_stage1, eval_data, eval_stage1, hankel_L, pad_idx, size_fn)
    if run_stage2_solo:
        runClassicFamily(name, classifier_ctor, 3, "stage2", STAGE2_CLASS_NAMES,
                          train_data, train_stage2, eval_data, eval_stage2, hankel_L, pad_idx, size_fn)


# ---------------------------------------------------------------------------
# PCA+MLP per-timestep baseline. Same Hankel-window row framing as the classic-ML baselines
# above, but with a StandardScaler+PCA reduction (fit on the train split only) feeding a small
# MLP -- mirroring the whole-trajectory script's --pca/--mlp comparison, adapted from one
# time-averaged row per trajectory to one windowed row per timestep.
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in, n_classes, width=64, depth=1, p_drop=0.1):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU(inplace=True), nn.Dropout(p_drop)]
            d = width
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, d_in)
        return self.net(x)


def buildPCAHankelFeatures(train_states, train_labels, val_states, val_labels, eval_states, eval_labels,
                            hankel_L, pad_idx, pca_n_components, standardize=True):
    """Fits StandardScaler+PCA on TRAIN Hankel-window rows only, transforms val/eval the same
    way. Returns (X_train,y_train,X_val,y_val,X_eval,y_eval,scaler,pca) -- the fitted scaler/pca
    are needed by predictPCAMLPPerTimestep to reconstruct a full (undropped) [N,T] prediction
    grid for cascade combination."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_train, y_train, _ = buildHankelWindowRowsPerTimestep(train_states, train_labels, hankel_L, pad_idx=pad_idx, drop_masked=True)
    X_val, y_val, _ = buildHankelWindowRowsPerTimestep(val_states, val_labels, hankel_L, pad_idx=pad_idx, drop_masked=True)
    X_eval, y_eval, _ = buildHankelWindowRowsPerTimestep(eval_states, eval_labels, hankel_L, pad_idx=pad_idx, drop_masked=True)

    scaler = StandardScaler().fit(X_train) if standardize else None
    if scaler is not None:
        X_train, X_val, X_eval = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_eval)

    pca = PCA(n_components=pca_n_components, random_state=0).fit(X_train)
    X_train, X_val, X_eval = pca.transform(X_train), pca.transform(X_val), pca.transform(X_eval)

    return (X_train.astype(np.float64), y_train, X_val.astype(np.float64), y_val,
            X_eval.astype(np.float64), y_eval, scaler, pca)


def predictPCAMLPPerTimestep(model, scaler, pca, data, hankel_L, device):
    """Returns preds[N,T], reconstructed over the full (undropped) grid via the same fitted
    scaler/pca used at training time -- needed for cascade combination, parallel to
    predictClassicPerTimestep."""
    N, T, _ = data.shape
    preds = np.zeros((N, T), dtype=np.int64)
    if T < hankel_L:
        return preds
    dummy_labels = np.zeros((N, T), dtype=np.int64)
    X, _, _ = buildHankelWindowRowsPerTimestep(data, dummy_labels, hankel_L, drop_masked=False)
    if scaler is not None:
        X = scaler.transform(X)
    X = pca.transform(X)
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X).double().to(device)).argmax(dim=1).cpu().numpy()
    preds[:, hankel_L - 1:] = y_pred.reshape(T - hankel_L + 1, N).T
    return preds


def _rowLoader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X).double(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def trainMLPRowWise(model, train_loader, val_loader, device, num_epochs=100, schedulerPatience=5, verbose=True):
    """Row-wise (B,d_in)->(B,num_classes) training loop for the PCA+Hankel-row MLP baseline,
    parallel to train_model but operating on flat rows instead of [B,T,C] sequences."""
    model = model.to(device).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=schedulerPatience)

    best_loss = float('inf')
    ESpatience = schedulerPatience * 2
    counter = 0
    timeToTrain = timer()

    for epoch in range(num_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f}")

        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / max(1, val_batches)
        if verbose:
            print(f"Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= ESpatience:
                print("Early stopping triggered.")
                break

    return timeToTrain.toc()


def evaluateMLPRowWise(model, loader, class_names, device, print_report=True):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(dim=1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb.numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return _reportFromPredictions(y_true, y_pred, class_names, print_report=print_report)


def runPCAMLPFamily(mode_name, num_classes, class_names,
                     train_data, train_labels, val_data, val_labels, eval_data, eval_labels,
                     hankel_L, pad_idx, pca_n_components, device, num_epochs, smooth_max_gap=0):
    print(f"\nEntering PCA+MLP ({mode_name}) Training Loop")
    X_train, y_train, X_val, y_val, X_eval, y_eval, scaler, pca = buildPCAHankelFeatures(
        train_data, train_labels, val_data, val_labels, eval_data, eval_labels,
        hankel_L, pad_idx, pca_n_components,
    )
    train_loader = _rowLoader(X_train, y_train, 128, True)
    val_loader = _rowLoader(X_val, y_val, 128, False)
    eval_loader = _rowLoader(X_eval, y_eval, 128, False)

    model = MLP(d_in=X_train.shape[1], n_classes=num_classes, width=64, depth=1, p_drop=0.1)
    trainMLPRowWise(model, train_loader, val_loader, device, num_epochs=num_epochs)
    printModelParmSize(model)

    print(f"\nPCA+MLP ({mode_name}) Validation")
    tInf = timer()
    evaluateMLPRowWise(model, eval_loader, class_names, device, print_report=True)
    tInf.tocStr(f"PCA+MLP ({mode_name}) Inference Time")

    if mode_name == "joint":
        pred_grid = predictPCAMLPPerTimestep(model, scaler, pca, eval_data, hankel_L, device)
        _reportEventLevelWithSmoothing(eval_labels, pred_grid, class_names, smooth_max_gap=smooth_max_gap,
                                        valid_from=hankel_L - 1)

    return model


def runPCAMLPCascade(train_data, train_joint, val_data, val_joint, eval_data, eval_joint,
                      hankel_L, pad_idx, pca_n_components, device, num_epochs, smooth_max_gap=0):
    train_stage1, train_stage2 = _deriveStage1Stage2(train_joint, pad_idx)
    val_stage1, val_stage2 = _deriveStage1Stage2(val_joint, pad_idx)
    eval_stage1, eval_stage2 = _deriveStage1Stage2(eval_joint, pad_idx)

    print("\nEntering PCA+MLP Stage 1 (Detector) Training Loop")
    X_train1, y_train1, X_val1, y_val1, _, _, scaler1, pca1 = buildPCAHankelFeatures(
        train_data, train_stage1, val_data, val_stage1, eval_data, eval_stage1, hankel_L, pad_idx, pca_n_components)
    model1 = MLP(d_in=X_train1.shape[1], n_classes=2, width=64, depth=1, p_drop=0.1)
    trainMLPRowWise(model1, _rowLoader(X_train1, y_train1, 128, True), _rowLoader(X_val1, y_val1, 128, False),
                     device, num_epochs=num_epochs)
    printModelParmSize(model1)

    print("\nEntering PCA+MLP Stage 2 (Type Classifier) Training Loop")
    X_train2, y_train2, X_val2, y_val2, _, _, scaler2, pca2 = buildPCAHankelFeatures(
        train_data, train_stage2, val_data, val_stage2, eval_data, eval_stage2, hankel_L, pad_idx, pca_n_components)
    model2 = MLP(d_in=X_train2.shape[1], n_classes=3, width=64, depth=1, p_drop=0.1)
    trainMLPRowWise(model2, _rowLoader(X_train2, y_train2, 128, True), _rowLoader(X_val2, y_val2, 128, False),
                     device, num_epochs=num_epochs)
    printModelParmSize(model2)

    print("\nPCA+MLP Cascade Evaluation")
    tInf = timer()
    pred_stage1 = predictPCAMLPPerTimestep(model1, scaler1, pca1, eval_data, hankel_L, device)
    pred_stage2 = predictPCAMLPPerTimestep(model2, scaler2, pca2, eval_data, hankel_L, device)
    combineCascadePredictions(eval_joint, eval_stage1, pred_stage1, pred_stage2, print_report=True,
                               plot_name="PCA+MLP",
                               plot_save_path=os.path.join(plotLoc, f"seqpred_PCA+MLP_cascade_{logStem}.png"),
                               valid_from=hankel_L - 1, smooth_max_gap=smooth_max_gap)
    tInf.tocStr("PCA+MLP Cascade Inference Time")
    return model1, model2


# ---------------------------------------------------------------------------
# Per-timestep MiniRocket baseline (--minirocket). Genuine end-to-end sequence classification in
# both joint and cascade modes, at the same granularity as every other backbone here.
#
# Shape of the thing: each timestep gets the trailing MINIROCKET_WINDOW frames ending at it, so a
# [N,T,C] batch becomes N*T short series; MiniRocket transforms those once, and three ridge heads
# (joint 4-class, stage-1 binary, stage-2 3-class) are solved on that ONE feature matrix. The
# transform is label-agnostic, so cascade is nearly free once joint exists -- only the K x n_class
# right-hand side differs, the K x K Gram is shared.
#
# Two earlier claims in this file were wrong and are corrected here, both measured on
# leo/30min-1500 with the IC-grouped split:
#
#  * "a windowed per-timestep version starves the kernels of signal, so stage 1 must be whole-
#    trajectory." Window length is not what binds. Sweeping MINIROCKET_WINDOW over 9/15/21/30
#    moves per-timestep macro-F1 by less than the seed-to-seed spread, and at 30 (the full
#    trajectory, causally left-padded) it is no better than at 9. The real limit is that PPV
#    pooling is a *summary statistic over the window* -- it answers "what fraction of this span
#    exceeded a threshold", which is exactly right for typing a whole trajectory and structurally
#    wrong for localizing an event to one timestep. Widening the window adds context and dilutes
#    localization by the same amount, so the two cancel. Hence 9: the shortest window the kernels
#    accept, and nothing is paid for a longer one.
#
#  * "MiniRocket only supports whole-series input, so joint only." It supports whatever series you
#    hand it. What actually broke before was memory: materializing the N*T x K feature matrix and
#    handing it to RidgeClassifierCV, which upcasts to float64 and runs generalized cross-
#    validation over it. _ridgeFit below accumulates normal equations in chunks instead, so peak
#    memory is O(MINIROCKET_CHUNK * K + K^2) and independent of N*T.
#
# What to expect: ~0.72 joint / ~0.80 cascade macro-F1 on OE+Energy at 2436 kernels, against ~0.93
# / ~0.95 for the CNN, and at-chance on raw ECI. It is a deliberately informative baseline rather
# than a competitive one -- the same transform scores 0.99 typing a whole trajectory, so the gap
# between those two numbers is a clean measurement of what pooling costs on a localization task.
# ---------------------------------------------------------------------------
def _minirocketWindows(states, window=MINIROCKET_WINDOW):
    """[N,T,C] -> (X[N*T, C, window] float32, idx[N*T, 2] of (ic, t)), ordered t-outer/ic-inner.

    Left-edge-padded (the first frame repeated) so that EVERY timestep gets a real window and a
    real prediction -- unlike buildHankelWindowRowsPerTimestep, which drops the first hankel_L-1
    timesteps and forces them to background, and so has to pass valid_from=hankel_L-1 downstream.
    Here valid_from stays 0. Padding costs accuracy on the leading frames (their windows are
    largely a repeated constant), but that is a real cost of predicting them, not one hidden by
    declining to."""
    N, T, C = states.shape
    padded = np.concatenate([np.repeat(states[:, :1], window - 1, axis=1), states], axis=1)
    X = np.concatenate([padded[:, t:t + window, :].transpose(0, 2, 1) for t in range(T)])
    idx = np.concatenate([np.stack([np.arange(N), np.full(N, t)], axis=1) for t in range(T)])
    return X.astype(np.float32), idx


def _fitMiniRocketTransform(X, num_kernels, fit_sample=20000, seed=None):
    """MiniRocket's fit only picks dilations and bias quantiles, which a sample estimates as well
    as the full set does -- and fitting on all N*T windows is the slowest step by far. Sample
    RANDOMLY, not by slicing: _minirocketWindows orders rows t-outer, so X[:n] would be every
    window from the first few timesteps only, and bias quantiles would be calibrated to the start
    of the trajectory.

    random_state is threaded through to sktime, NOT just used for the subsample. MiniRocket draws
    its bias quantiles from randomly chosen training examples, so leaving sktime's random_state at
    None makes the whole backbone non-deterministic even under --seed: three runs at --seed 0
    measured joint macro-F1 of 0.673 / 0.664 / 0.618, a spread wide enough to swamp the real
    seed-to-seed spread the sweep's three seeds are meant to estimate."""
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    tf = MiniRocketMultivariate(num_kernels=num_kernels, n_jobs=-1, random_state=seed)
    if len(X) > fit_sample:
        pick = np.random.default_rng(seed).choice(len(X), fit_sample, replace=False)
        X = X[np.sort(pick)]
    tf.fit(X)
    return tf


def _minirocketFeatures(tf, X, n_feat):
    """Chunked transform -> [M, n_feat+1] float32. The trailing all-ones column is the ridge
    intercept, kept as a feature so _ridgeFit needs no separate centering pass."""
    F = np.empty((len(X), n_feat + 1), dtype=np.float32)
    F[:, -1] = 1.0
    for i in range(0, len(X), MINIROCKET_CHUNK):
        F[i:i + MINIROCKET_CHUNK, :n_feat] = tf.transform(X[i:i + MINIROCKET_CHUNK]).to_numpy(dtype=np.float32)
    return F


def _ridgeFit(F, targets, alpha=1.0):
    """Ridge regression onto one-hot targets, accumulated chunk by chunk.

    Equivalent to sklearn's RidgeClassifier on this problem, but never holds more than one chunk in
    float64: G is K x K and each B is K x n_class, all independent of the row count. That is the
    whole reason this backbone can run per-timestep at all -- see the module comment above.

    targets: list of (y, num_classes) sharing F's rows -> list of coefficient matrices. Several
    heads are passed together because G = F^T F depends only on F, so fitting the joint 4-class and
    stage-1 binary heads separately would build the same ~2437 x 2437 matrix twice -- measured at
    ~124s each on a 126k-row 30-minute cell, i.e. the dominant cost of the whole backbone. Only the
    B accumulation is per-head, and that is a K x n_class GEMM. Stage 2 still needs its own call:
    it is supervised on the thrust-frame subset, so its G is genuinely a different matrix."""
    P = F.shape[1]
    G = np.zeros((P, P))
    Bs = [np.zeros((P, n)) for _, n in targets]
    for i in range(0, len(F), MINIROCKET_CHUNK):
        Fb = F[i:i + MINIROCKET_CHUNK].astype(np.float64)
        G += Fb.T @ Fb
        for B, (y, n) in zip(Bs, targets):
            yb = y[i:i + MINIROCKET_CHUNK]
            Y = np.zeros((len(yb), n))
            Y[np.arange(len(yb)), yb] = 1.0
            B += Fb.T @ Y
    G[np.diag_indices(P - 1)] += alpha      # last column is the intercept -- leave it unpenalised
    return [np.linalg.solve(G, B) for B in Bs]


def _ridgePredict(F, W):
    return np.concatenate([(F[i:i + MINIROCKET_CHUNK].astype(np.float64) @ W).argmax(axis=1)
                           for i in range(0, len(F), MINIROCKET_CHUNK)])


def _minirocketGrid(n, t, idx, flat):
    """Scatter flat per-row predictions back to an [N,T] grid."""
    grid = np.zeros((n, t), dtype=np.int64)
    grid[idx[:, 0], idx[:, 1]] = flat
    return grid


def printMiniRocketRidgeSize(tf, head):
    """Matches printMiniROCKETSize's block format so the log parser reads it the same way. Counts
    ridge coefficients only -- MiniRocket's kernels are fixed, not learned, so they are not
    parameters in the sense the other backbones' counts mean."""
    import pickle
    size_bytes = len(pickle.dumps(tf)) + head.nbytes
    print("\n==========================================================================================")
    print(f"Total parameters: {head.size}")
    print(f"Total memory (bytes): {size_bytes}")
    print(f"Total memory (MB): {size_bytes / (1024 ** 2):.4f}")
    print("==========================================================================================")


def runMiniRocketPerTimestep(train_data, train_joint, eval_data, eval_joint, pad_idx, num_kernels,
                              run_joint, run_cascade, run_stage1_solo, run_stage2_solo,
                              smooth_max_gap=0):
    """One shared windowed MiniRocket transform, then a ridge head per requested mode."""
    train_stage1, train_stage2 = _deriveStage1Stage2(train_joint, pad_idx)
    eval_stage1, _ = _deriveStage1Stage2(eval_joint, pad_idx)
    N_eval, T_eval = eval_joint.shape

    print(f"\nBuilding MiniRocket windows (length {MINIROCKET_WINDOW}, stride 1, left-padded)")
    X_train, idx_train = _minirocketWindows(train_data)
    X_eval, idx_eval = _minirocketWindows(eval_data)
    print(f"  train rows {X_train.shape}, eval rows {X_eval.shape}")

    tTf = timer()
    tf = _fitMiniRocketTransform(X_train, num_kernels, seed=runSeed)
    n_feat = tf.transform(X_train[:2]).shape[1]
    print(f"  requested {num_kernels} kernels -> sktime produced {n_feat} features")
    F_train = _minirocketFeatures(tf, X_train, n_feat)
    F_eval = _minirocketFeatures(tf, X_eval, n_feat)
    tTf.tocStr("MiniRocket Transform Time (shared by all modes)")
    del X_train, X_eval

    y_train_joint = train_joint[idx_train[:, 0], idx_train[:, 1]]
    y_train_stage1 = train_stage1[idx_train[:, 0], idx_train[:, 1]]
    y_eval_joint_flat = eval_joint[idx_eval[:, 0], idx_eval[:, 1]]
    thrust = y_train_joint > 0     # stage 2 is only ever supervised on thrusting frames

    # One pass over F_train fits every head supervised on ALL rows (joint, stage 1) -- see _ridgeFit
    # for why they are batched rather than fitted separately.
    #
    # The log parser (displaySeqLogData.RE_ENTER / iter_blocks) slices the log into blocks running
    # from one "Entering <X> Training[ Loop]" line to the next, and attributes everything in a
    # block to that X. So each head's "Entering" line must be IMMEDIATELY followed by that head's
    # own output and nothing else -- printing both up front, as an earlier version did, put the
    # joint metrics inside the stage-1 block and lost the joint row entirely. The shared fit
    # therefore runs first, unannounced, and each block is emitted where its results are.
    want_s1 = run_cascade or run_stage1_solo
    heads = ([(y_train_joint, 4)] if run_joint else []) + ([(y_train_stage1, 2)] if want_s1 else [])
    W_joint = W_s1 = None
    if heads:
        tFit = timer()
        fitted = _ridgeFit(F_train, heads)
        if run_joint:
            W_joint = fitted.pop(0)
        if want_s1:
            W_s1 = fitted[0]

    if run_joint:
        print("\nEntering MiniRocket (joint) Training Loop")
        tFit.tocStr("MiniRocket (joint) Training Time"
                    + (" [shared ridge pass, also solves stage 1]" if want_s1 else ""))
        printMiniRocketRidgeSize(tf, W_joint)

        print("\nMiniRocket (joint) Validation")
        tInf = timer()
        pred_joint = _ridgePredict(F_eval, W_joint)
        _reportFromPredictions(y_eval_joint_flat, pred_joint, JOINT_CLASS_NAMES, print_report=True)
        tInf.tocStr("MiniRocket (joint) Inference Time")
        _reportEventLevelWithSmoothing(eval_joint, _minirocketGrid(N_eval, T_eval, idx_eval, pred_joint),
                                        JOINT_CLASS_NAMES, smooth_max_gap=smooth_max_gap,
                                        plot_name="MiniRocket",
                                        plot_save_path=os.path.join(plotLoc, f"seqpred_MiniRocket_joint_{logStem}.png"),
                                        mode_label="Joint", valid_from=0)

    if want_s1:
        # "(standalone)" in solo mode, "(Detector)" in cascade mode: classify_component keys
        # Stage1_solo vs Stage1 off exactly that parenthetical.
        print("\nEntering MiniRocket Stage 1 "
              + ("(standalone)" if run_stage1_solo else "(Detector)") + " Training Loop")
        if run_joint:
            print("(coefficients came from the joint pass above -- same rows, same Gram)")
        else:
            tFit.tocStr("MiniRocket Stage 1 Training Time")
        printMiniRocketRidgeSize(tf, W_s1)
        if run_stage1_solo:
            print("\nMiniRocket Stage 1 (standalone) Validation")
            _reportFromPredictions((y_eval_joint_flat > 0).astype(np.int64), _ridgePredict(F_eval, W_s1),
                                    STAGE1_CLASS_NAMES, print_report=True)
        # In cascade mode stage 1 carries no metrics of its own -- it is scored inside
        # combineCascadePredictions below, exactly as runClassicCascade's stage-1 block is.

    if run_cascade or run_stage2_solo:
        print("\nEntering MiniRocket Stage 2 "
              + ("(standalone)" if run_stage2_solo else "(Type Classifier)") + " Training Loop")
        t = timer()
        W_s2 = _ridgeFit(F_train[thrust], [(y_train_joint[thrust] - 1, 3)])[0]
        t.toc()
        printMiniRocketRidgeSize(tf, W_s2)

        # Standalone stage-2 score, on true-thrust frames only. Same reason runClassicCascade does
        # it: without this, stage 2 is only ever seen gated through stage 1, and a weak cascade is
        # not diagnosable as bad detection vs. bad typing. The header is word-for-word what
        # RE_STAGE2_STANDALONE_HDR matches -- reword it and the cascade_stage2_standalone row
        # silently disappears from the parsed CSVs.
        print("\nMiniRocket Stage 2 (Type Classifier) Standalone Validation")
        mask = y_eval_joint_flat > 0
        _reportFromPredictions(y_eval_joint_flat[mask] - 1, _ridgePredict(F_eval[mask], W_s2),
                                STAGE2_CLASS_NAMES, print_report=True)

    if run_cascade:
        print("\nMiniRocket Cascade Evaluation")
        tInf = timer()
        g1 = _minirocketGrid(N_eval, T_eval, idx_eval, _ridgePredict(F_eval, W_s1))
        g2 = _minirocketGrid(N_eval, T_eval, idx_eval, _ridgePredict(F_eval, W_s2))
        combineCascadePredictions(eval_joint, eval_stage1, g1, g2, print_report=True,
                                   plot_name="MiniRocket",
                                   plot_save_path=os.path.join(plotLoc, f"seqpred_MiniRocket_cascade_{logStem}.png"),
                                   valid_from=0, smooth_max_gap=smooth_max_gap)
        tInf.tocStr("MiniRocket Cascade Inference Time")


def main():
    import yaml
    with open("data.yaml", 'r') as f:
        dataConfig = yaml.safe_load(f)
    print(f"Processing datasets for {orbitType} with {numMinProp} minutes and {numRandSys} random systems.")

    yaml_config = {
        'useOE': useOE,
        'useNorm': useNorm,
        'useNoise': useNoise,
        'useEnergy': useEnergy,
        'useEnergyRate': useEnergyRate,
        'numSinusoids': numSinusoids,
        'usePhysicsLoss': usePhysicsLoss,
        'useJ2Energy': useJ2Energy,
        'useResidualLadder': useResidualLadder,
        'useResidualLadderFull': useResidualLadderFull,
        'prop_time': numMinProp,
        'orbit': orbitType,
        'systems': numRandSys,
        'test_dataset': testSet,
        'test_systems': testSys,
    }

    if train_ratio == 0.7:
        val_ratio = 0.15
        test_ratio = 0.15
    else:
        val_ratio = train_ratio
        test_ratio = 1.0 - train_ratio - val_ratio

    (train_loader, val_loader, test_loader,
     train_data, train_joint, val_data, val_joint, test_data, test_joint) = prepareInSequenceThrustClassificationDatasets(
        yaml_config, dataConfig,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        pos_noise_std=1e3 * velNoise, vel_noise_std=velNoise,
        batch_size=16, oversample=useOversample, standardize=useStandardize,
        seed=runSeed,
    )

    input_size = train_data.shape[2]
    hidden_factor = 8
    hidden_size = int(input_size * hidden_factor)
    num_layers = 1
    num_epochs = 100
    if useOnePass:
        num_epochs = 1
    schedulerPatience = 5

    # Without --eval-test an in-distribution run falls through to val_loader -- the same split early
    # stopping and best-checkpoint restore select on, so those numbers are optimistically biased and
    # not comparable against a cross-orbit run's held-out numbers.
    use_test_eval = (testSet != orbitType or testSys != numRandSys or evalTest)
    eval_loader = test_loader if use_test_eval else val_loader
    eval_data = test_data if use_test_eval else val_data
    eval_joint = test_joint if use_test_eval else val_joint
    eval_stage1, eval_stage2 = _deriveStage1Stage2(eval_joint)
    train_stage1, train_stage2 = _deriveStage1Stage2(train_joint)
    val_stage1_np, val_stage2_np = _deriveStage1Stage2(val_joint)

    backbones = (["lstm"] if use_lstm else []) + (["mamba"] if use_mamba else [])
    if use_transformer:
        backbones.append("transformer")
    if use_cnn:
        backbones.append("cnn")
    if use_hybrid:
        backbones.append("hybrid")

    run_joint = runMode in ("all", "joint")
    run_cascade = runMode in ("all", "cascade")
    run_stage1_solo = runMode == "stage1"
    run_stage2_solo = runMode == "stage2"

    for backbone in backbones:
        print(f"\n{'='*80}\nBackbone: {backbone.upper()}\n{'='*80}")

        if backbone == "hybrid":
            print("[note] HYBRID = whole-trajectory MiniRocket stage-1 detector ('does this "
                  "~30-minute window contain thrust anywhere') + CNN (InceptionTime) stage-2 "
                  "per-timestep type classifier. No joint 4-class form, so --mode joint is a "
                  "no-op for this backbone. Stage 1's trajectory-level decision is broadcast "
                  "across every timestep of a trajectory for the combined per-timestep report "
                  "below -- a positive trajectory has no further per-timestep background "
                  "suppression, so stage 2 alone determines which minutes look idle vs. "
                  "thrusting within it.")

            stage1_clf = None
            if run_cascade or run_stage1_solo:
                print("\nEntering HYBRID Stage 1 (MiniRocket, whole-trajectory) Training")
                miniRocketTimer = timer()
                stage1_clf = trainMiniRocketStage1Detector(train_data, train_stage1)
                miniRocketTimer.tocStr("HYBRID Stage 1 (MiniRocket) Training Time")

            model_stage2 = None
            if run_cascade or run_stage2_solo:
                print("\nEntering HYBRID Stage 2 (CNN Type Classifier) Training Loop")
                model_stage2 = build_model("cnn", 3, input_size, hidden_size, num_layers)
                train_model(model_stage2, train_loader, val_loader, num_epochs=num_epochs, num_classes=3,
                            mode='stage2', schedulerPatience=schedulerPatience)
                printModelParmSize(model_stage2)

            if run_stage1_solo:
                print("\nHYBRID Stage 1 (MiniRocket, whole-trajectory) Validation")
                pred_stage1_traj = predictMiniRocketStage1Trajectory(stage1_clf, eval_data)
                true_stage1_traj = eval_stage1.any(axis=1).astype(np.int64)
                _reportFromPredictions(true_stage1_traj, pred_stage1_traj, STAGE1_CLASS_NAMES, print_report=True)

            if run_stage2_solo:
                print("\nHYBRID Stage 2 (CNN) Validation")
                validateInSequenceClassifier(model_stage2, eval_loader, mode='stage2', num_classes=3, device=device,
                                              class_names=STAGE2_CLASS_NAMES, print_report=True)

            if run_cascade:
                print("\nHYBRID Cascade Evaluation")
                cascadeInference = timer()

                pred_stage1_traj = predictMiniRocketStage1Trajectory(stage1_clf, eval_data)
                true_stage1_traj = eval_stage1.any(axis=1).astype(np.int64)
                print("\nStage 1 (MiniRocket, whole-trajectory) Standalone Metrics:")
                _reportFromPredictions(true_stage1_traj, pred_stage1_traj, STAGE1_CLASS_NAMES, print_report=True)

                T = eval_data.shape[1]
                pred_stage1_bcast = np.repeat(pred_stage1_traj[:, None], T, axis=1)  # [N] -> [N,T]
                _, _, pred_stage2 = _predictPerTimestepNeural(model_stage2, eval_loader, device)

                print("\nCombined Per-Timestep Report (stage-1 decision broadcast across each trajectory):")
                combineCascadePredictions(eval_joint, eval_stage1, pred_stage1_bcast, pred_stage2, print_report=True,
                                           plot_name="HYBRID",
                                           plot_save_path=os.path.join(plotLoc, f"seqpred_HYBRID_cascade_{logStem}.png"),
                                           granularity_note="stage-1 (MiniRocket) decision is whole-trajectory, "
                                                             "broadcast across every timestep -- recall/precision "
                                                             "on the detection side collapses to whole-trajectory "
                                                             "detection; only stage-2's CNN typing is genuinely "
                                                             "per-timestep.",
                                           smooth_max_gap=smoothMaxGap)
                cascadeInference.tocStr("HYBRID Cascade Inference Time")

            continue

        if run_joint:
            print(f"\nEntering {backbone.upper()} Joint (4-class) Training Loop")
            model_joint = build_model(backbone, 4, input_size, hidden_size, num_layers)
            train_model(model_joint, train_loader, val_loader, num_epochs=num_epochs, num_classes=4,
                        mode='joint', schedulerPatience=schedulerPatience)
            printModelParmSize(model_joint)
            print(f"\n{backbone.upper()} Joint Validation")
            jointInference = timer()
            validateInSequenceClassifier(model_joint, eval_loader, mode='joint', num_classes=4, device=device,
                                          class_names=JOINT_CLASS_NAMES, print_report=True)
            jointInference.tocStr(f"{backbone.upper()} Joint Inference Time")

            y_true_grid, _, pred_grid = _predictPerTimestepNeural(model_joint, eval_loader, device)
            _reportEventLevelWithSmoothing(y_true_grid, pred_grid, JOINT_CLASS_NAMES,
                                            smooth_max_gap=smoothMaxGap,
                                            plot_name=backbone.upper(),
                                            plot_save_path=os.path.join(plotLoc, f"seqpred_{backbone.upper()}_joint_{logStem}.png"),
                                            mode_label="Joint")

        if run_cascade:
            print(f"\nEntering {backbone.upper()} Stage 1 (Detector) Training Loop")
            model_stage1 = build_model(backbone, 2, input_size, hidden_size, num_layers)
            train_model(model_stage1, train_loader, val_loader, num_epochs=num_epochs, num_classes=2,
                        mode='stage1', schedulerPatience=schedulerPatience)
            printModelParmSize(model_stage1)

            print(f"\nEntering {backbone.upper()} Stage 2 (Type Classifier) Training Loop")
            model_stage2 = build_model(backbone, 3, input_size, hidden_size, num_layers)
            train_model(model_stage2, train_loader, val_loader, num_epochs=num_epochs, num_classes=3,
                        mode='stage2', schedulerPatience=schedulerPatience)
            printModelParmSize(model_stage2)

            # Standalone post-training snapshot on the same class_names/eval_loader as the classic-ML
            # cascade path's equivalent print, so Stage 2 is directly comparable across every backbone
            # family rather than only via its own per-epoch training validation.
            print(f"\n{backbone.upper()} Stage 2 (Type Classifier) Standalone Validation")
            validateInSequenceClassifier(model_stage2, eval_loader, mode='stage2', num_classes=3, device=device,
                                          class_names=STAGE2_CLASS_NAMES, print_report=True)

            print(f"\n{backbone.upper()} Cascade Evaluation")
            cascadeInference = timer()
            runCascadeEvaluation(model_stage1, model_stage2, eval_loader, device=device,
                                  plot_name=backbone.upper(),
                                  plot_save_path=os.path.join(plotLoc, f"seqpred_{backbone.upper()}_cascade_{logStem}.png"),
                                  smooth_max_gap=smoothMaxGap)
            cascadeInference.tocStr(f"{backbone.upper()} Cascade Inference Time")

        if run_stage1_solo:
            print(f"\nEntering {backbone.upper()} Stage 1 (Detector, standalone) Training Loop")
            model_stage1 = build_model(backbone, 2, input_size, hidden_size, num_layers)
            train_model(model_stage1, train_loader, val_loader, num_epochs=num_epochs, num_classes=2,
                        mode='stage1', schedulerPatience=schedulerPatience)
            printModelParmSize(model_stage1)
            print(f"\n{backbone.upper()} Stage 1 Validation")
            validateInSequenceClassifier(model_stage1, eval_loader, mode='stage1', num_classes=2, device=device,
                                          class_names=STAGE1_CLASS_NAMES, print_report=True)

        if run_stage2_solo:
            print(f"\nEntering {backbone.upper()} Stage 2 (Type Classifier, standalone) Training Loop")
            model_stage2 = build_model(backbone, 3, input_size, hidden_size, num_layers)
            train_model(model_stage2, train_loader, val_loader, num_epochs=num_epochs, num_classes=3,
                        mode='stage2', schedulerPatience=schedulerPatience)
            printModelParmSize(model_stage2)
            print(f"\n{backbone.upper()} Stage 2 Validation")
            validateInSequenceClassifier(model_stage2, eval_loader, mode='stage2', num_classes=3, device=device,
                                          class_names=STAGE2_CLASS_NAMES, print_report=True)

    # -----------------------------------------------------------------------
    # Classic ML / GBDT + PCA+MLP + standalone MiniRocket baselines (Phase C) -- all operate on
    # Hankel-windowed rows (buildHankelWindowRowsPerTimestep) rather than the [B,T,C] DataLoaders
    # above, so they live outside the backbone loop, matching the whole-trajectory script's
    # structure where these are one-shot blocks rather than part of a per-architecture loop.
    # -----------------------------------------------------------------------
    hankel_L = min(5, numMinProp)

    if use_classic or use_xgboost or use_catboost or use_random_forest or use_extra_trees:
        print(f"\n{'='*80}\nClassic ML / GBDT baselines (Hankel window length={hankel_L})\n{'='*80}")

    if use_classic:
        from lightgbm import LGBMClassifier
        from qutils.ml.classic.classifier import printDTModelSize
        ctor = lambda nc: LGBMClassifier(objective="multiclass", num_classes=nc, n_estimators=30, max_depth=-1,
                                          learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=-1)
        runClassicMLModes("LightGBM", ctor, printDTModelSize,
                           train_data, train_joint, train_stage1, train_stage2,
                           eval_data, eval_joint, eval_stage1, eval_stage2,
                           hankel_L, -100, run_joint, run_cascade, run_stage1_solo, run_stage2_solo,
                           smooth_max_gap=smoothMaxGap)

    if use_xgboost:
        from xgboost import XGBClassifier
        from qutils.ml.classic.classifier import printClassicModelSize
        # multi:softmax (not softprob) -- softprob's sklearn .predict() returns a per-class
        # probability matrix instead of hard labels when num_class=2 (confirmed empirically;
        # the whole-trajectory script's --xgboost never hits this since it's always 4-class).
        # softmax returns 1D hard labels for any class count.
        ctor = lambda nc: XGBClassifier(objective="multi:softmax", num_class=nc, n_estimators=200, max_depth=6,
                                         learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                         eval_metric="mlogloss", n_jobs=-1)
        runClassicMLModes("XGBoost", ctor, printClassicModelSize,
                           train_data, train_joint, train_stage1, train_stage2,
                           eval_data, eval_joint, eval_stage1, eval_stage2,
                           hankel_L, -100, run_joint, run_cascade, run_stage1_solo, run_stage2_solo,
                           smooth_max_gap=smoothMaxGap)

    if use_catboost:
        from catboost import CatBoostClassifier
        from qutils.ml.classic.classifier import printClassicModelSize
        ctor = lambda nc: CatBoostClassifier(loss_function="MultiClass", classes_count=nc, iterations=200, depth=6,
                                              learning_rate=0.05, bootstrap_type="Bernoulli", subsample=0.8,
                                              colsample_bylevel=0.8, verbose=False, allow_writing_files=False)
        runClassicMLModes("CatBoost", ctor, printClassicModelSize,
                           train_data, train_joint, train_stage1, train_stage2,
                           eval_data, eval_joint, eval_stage1, eval_stage2,
                           hankel_L, -100, run_joint, run_cascade, run_stage1_solo, run_stage2_solo,
                           smooth_max_gap=smoothMaxGap)

    if use_random_forest:
        from sklearn.ensemble import RandomForestClassifier
        from qutils.ml.classic.classifier import printClassicModelSize
        ctor = lambda nc: RandomForestClassifier(n_estimators=300, max_depth=30, n_jobs=-1)
        runClassicMLModes("Random Forest", ctor, printClassicModelSize,
                           train_data, train_joint, train_stage1, train_stage2,
                           eval_data, eval_joint, eval_stage1, eval_stage2,
                           hankel_L, -100, run_joint, run_cascade, run_stage1_solo, run_stage2_solo,
                           smooth_max_gap=smoothMaxGap)

    if use_extra_trees:
        from sklearn.ensemble import ExtraTreesClassifier
        from qutils.ml.classic.classifier import printClassicModelSize
        ctor = lambda nc: ExtraTreesClassifier(n_estimators=300, max_depth=30, n_jobs=-1)
        runClassicMLModes("Extra Trees", ctor, printClassicModelSize,
                           train_data, train_joint, train_stage1, train_stage2,
                           eval_data, eval_joint, eval_stage1, eval_stage2,
                           hankel_L, -100, run_joint, run_cascade, run_stage1_solo, run_stage2_solo,
                           smooth_max_gap=smoothMaxGap)

    if use_mlp:
        print(f"\n{'='*80}\nPCA+MLP baseline (Hankel window length={hankel_L})\n{'='*80}")
        if run_joint:
            runPCAMLPFamily("joint", 4, JOINT_CLASS_NAMES,
                             train_data, train_joint, val_data, val_joint, eval_data, eval_joint,
                             hankel_L, -100, pca_n_components, device, num_epochs, smooth_max_gap=smoothMaxGap)
        if run_cascade:
            runPCAMLPCascade(train_data, train_joint, val_data, val_joint, eval_data, eval_joint,
                              hankel_L, -100, pca_n_components, device, num_epochs, smooth_max_gap=smoothMaxGap)
        if run_stage1_solo:
            runPCAMLPFamily("stage1", 2, STAGE1_CLASS_NAMES,
                             train_data, train_stage1, val_data, val_stage1_np, eval_data, eval_stage1,
                             hankel_L, -100, pca_n_components, device, num_epochs)
        if run_stage2_solo:
            runPCAMLPFamily("stage2", 3, STAGE2_CLASS_NAMES,
                             train_data, train_stage2, val_data, val_stage2_np, eval_data, eval_stage2,
                             hankel_L, -100, pca_n_components, device, num_epochs)

    if use_minirocket:
        print(f"\n{'='*80}\nBackbone: MINIROCKET (per-timestep, windowed)\n{'='*80}")
        if train_data.shape[1] < MINIROCKET_WINDOW:
            print(f"[skip] MINIROCKET needs at least {MINIROCKET_WINDOW} timesteps per trajectory "
                  f"(kernel length); this run has {train_data.shape[1]}.")
        else:
            runMiniRocketPerTimestep(train_data, train_joint, eval_data, eval_joint, -100,
                                      minirocketKernels, run_joint, run_cascade,
                                      run_stage1_solo, run_stage2_solo,
                                      smooth_max_gap=smoothMaxGap)


if __name__ == "__main__":
    if save_to_log:
        with open(logFileLoc, 'w', buffering=1, encoding='utf-8') as f, \
                redirect_stdout(f), redirect_stderr(f):
            main()
    else:
        main()
