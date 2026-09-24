# In-sequence (per-timestep) thrust classification: at every timestep, is thrust occurring and if
# so is it Chemical/Electric/Impulsive. Ground truth is each class file's per-timestep
# 'thrustingTime' (Impulsive is forward-filled from the burn to the end of the window). Data
# loading lives in seqData.py, shared with the whole-trajectory script's --seq-data.
#
# Two approaches, compared in one run (--mode all|joint|cascade|stage1|stage2):
#   joint    one 4-class per-timestep model (0=NoThrust,1=Chemical,2=Electric,3=Impulsive)
#   cascade  binary thrust detector (stage 1) -> 3-class type classifier on thrusting frames
#            (stage 2), recombined into the joint label space so the reports compare directly.
#
# Backbones: LSTM and Mamba by default (--no-lstm/--no-mamba); --transformer/--cnn opt in.
# --hybrid = whole-trajectory MiniRocket stage 1 + CNN stage 2 (cascade modes only).
# LightGBM (on by default), --xgboost/--catboost/--rf/--extratrees and --mlp (PCA+MLP) train on
# Hankel-windowed per-timestep rows; --minirocket on short trailing windows with ridge heads.
#
# Notable options:
#   --loss-scheme/--cb-beta/--focal-gamma  class-weighted loss; --oversample resamples whole
#                                          trajectories (neural backbones only)
#   --j2-energy          J2-inclusive energy. The Keplerian form swings with J2 every orbit (~55x
#                        an electric thruster on leo); removing it takes per-frame Electric-vs-
#                        NoThrust AUC from 0.52 to 0.83. Recommended for low-thrust work.
#   --sinusoids K        replace each channel with its K dominant FFT components (C -> C*K)
#   --physics-loss-weight W  auxiliary Chemical-vs-Electric term from the J2 / J3-J6 residual
#                        scales, gated by their ratio to two-body gravity (fades out toward GEO)
#   --smooth-max-gap N   post-hoc: close interior NoThrust gaps <= N between thrust predictions
#                        before event-level reporting. Measured on leo/30min: no effect on LSTM/CNN,
#                        ~0.9pp Impulsive event recall on Transformer at N=3 (sweepSmoothGap.py).
#
# $ python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
# --systems 1500 --propMin 30 --orbit vleo --mode all
import argparse

# MiniRocket per-timestep constants -- up here because argparse help and strAdd read them.
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
                          "run, which does report on a genuinely held-out split. The IC-disjoint 15%% test "
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
    # the split is seeded separately; this covers weight init, dropout and the sampler
    torch.manual_seed(runSeed)
    np.random.seed(runSeed)

device = getDevice()

from seqData import *  # noqa: F401,F403 -- residual-ladder names, label conventions, data loading


strAdd = ""
if useEnergy:
    strAdd = strAdd + "Energy_"
if useEnergyRate:
    strAdd = strAdd + "EnergyRate_"
if useJ2Energy:
    strAdd = strAdd + "J2Energy_"
if useResidualLadder:
    # channel count in the tag so 3- and 5-channel runs don't overwrite each other; old logs tagged
    # bare "ResidLadder_" are 5-channel
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
    """Bi-directional Mamba: independent forward and time-reversed stacks, per-timestep states
    concatenated before the head. Every other backbone here sees both directions, so a causal-only
    Mamba would confound architecture with directionality. ~2x params and training time."""
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
    """Encoder-only Transformer with a learnable positional embedding and a per-timestep head
    (the whole-trajectory version's CLS token is dropped)."""
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
    """One InceptionTime module: 1x1 bottleneck -> parallel odd-kernel convs + max-pool branch,
    concatenated. Length-preserving; GroupNorm so a size-1 trailing batch is fine."""
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
# "hybrid" backbone: stage 1 is a whole-trajectory MiniRocket detector (does this window contain
# thrust anywhere, broadcast to every timestep); stage 2 is the per-timestep CNN. Cascade and
# stage-solo modes only -- there is no joint hybrid model.
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
    """scheme='effective': class-balanced weights (1-beta)/(1-beta**n_c) (Cui et al. 2019), gentler
    than inverse frequency when imbalance swings from ~2:1 to ~99:1 across modes.
    scheme='inverse': N/count_c."""
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
    """Multi-class focal loss (Lin et al. 2017) with class weights and ignore_index, same interface
    as nn.CrossEntropyLoss. Focuses gradient on hard frames (e.g. thrust onset/offset)."""

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
    """Pushes the model's existing Chemical-vs-Electric logit margin toward the physics pseudo-target
    (is the measured residual nearer a_J2 or a_J3-6?) with BCE, weighted per timestep by
    y_phys_gate (0 outside Chemical/Electric frames, fading toward 0 at high altitude).
    logits [B,T,C], mode 'joint' or 'stage2'; targets/gate [B,T]. Returns 0.0, never nan, when no
    frame is gated."""
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
    """restore_best: reload the best-validation epoch's weights on return instead of keeping the last
    epoch (False reproduces the old behaviour).
    restore_metric: what "best" means for both the restore and early stopping -- 'loss' (default)
    or 'event_f1'. Use 'event_f1' when ranking models by F1; under this imbalance the best-loss
    epoch can have noticeably worse F1."""
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
                # all-pad batch (e.g. stage2 with no thrust frames): skip, a nan backward would
                # poison every parameter
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
    """Segment-level precision/recall per thrust class, to catch flicker that per-timestep metrics
    average away. y_true_grid/y_pred_grid: [N,T] in JOINT_LABELS space.
    
    An event is a maximal run of one class in a row. A true event is recalled if any of its frames
    is predicted as that class (point-adjust, Xu et al. 2018); a predicted event is a false positive
    only if it overlaps no true event of that class.
    
    valid_from: leading columns with no real prediction (Hankel-window models pad hankel_L-1 frames);
    true events entirely before it are left out of recall.
    granularity_note: caveat printed under the table, e.g. for whole-trajectory broadcasts."""
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
    """Close interior NoThrust gaps of length <= max_gap that have thrust on both sides, taking the
    type from the frame before the gap. pred_grid: [N,T] in JOINT_LABELS space.
    
    Only closes gaps, never removes short positive runs, so it can't erase a real short burst the
    way majority-vote or minimum-run filters can. Runs touching either end of a row (including the
    Hankel pad prefix) are left alone. max_gap <= 0 is a no-op."""
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
    """Event-level report (+ plot if plot_name/plot_save_path) on the raw grid, then -- if
    smooth_max_gap > 0 -- again on the gap-closed grid, plot suffixed '_smoothed'. Per-timestep
    metrics always stay on raw predictions."""
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
    """True vs. predicted per-timestep labels, one panel per class present in y_true, each on a
    representative trajectory. y_true/y_pred: [N,T] in the same label space as class_names, from any
    backbone family."""
    num_classes = len(class_names)
    example_rows = {}
    for c in range(num_classes):
        counts = (y_true == c).sum(axis=1)
        rows = np.where(counts > 0)[0]
        if rows.size:
            # row with the MOST timesteps of class c (background is in nearly every row)
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
    """Combine stage-1 (thrust yes/no) and stage-2 (type) predictions into one 4-class [N,T] grid and
    report it, plus a detection-vs-typing error split. pred_stage2 is in {0,1,2}.
    plot_name/plot_save_path: also save a seqpred plot of the combined result.
    valid_from/granularity_note: passed to _eventLevelReport.
    smooth_max_gap: event-level report/plot only; flat metrics use the raw prediction."""
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
# Classic ML / GBDT per-timestep baselines (LightGBM/XGBoost/CatBoost/RandomForest/ExtraTrees):
# one row per timestep from a trailing Hankel window.
# ---------------------------------------------------------------------------
def buildHankelWindowRowsPerTimestep(states, labels, hankel_L=5, pad_idx=-100, drop_masked=True):
    """states: [N,T,C], labels: [N,T] -> X[M, C*hankel_L], y[M], row_index[M,2] (ic,t).
    One row per (IC, t) with t >= hankel_L-1 (earlier frames lack context and are dropped, ~13% at
    hankel_L=5, T=30). drop_masked also drops pad_idx rows. Ordered t-outer/IC-inner, which
    predictClassicPerTimestep relies on to reshape back to [N,T]."""
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

    # score stage 2 standalone too, so a weak cascade is diagnosable as bad detection vs. bad typing
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
# PCA+MLP per-timestep baseline: Hankel-window rows -> StandardScaler+PCA (train split) -> MLP.
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
# Per-timestep MiniRocket baseline (--minirocket), joint and cascade. Each timestep gets its
# trailing MINIROCKET_WINDOW frames; one transform feeds three ridge heads (joint, stage 1,
# stage 2) that share one Gram matrix. Normal equations are accumulated in chunks, so memory is
# independent of N*T.
#
# Window 9 is the kernels' minimum; 9/15/21/30 all score within seed noise on leo/30min, because
# PPV pooling summarizes the window rather than localizing within it. Expect ~0.72 joint / ~0.80
# cascade macro-F1 (OE+Energy) vs ~0.93/~0.95 for the CNN; the same transform scores 0.99 on whole
# trajectories, so the gap measures what pooling costs for localization.
# ---------------------------------------------------------------------------
def _minirocketWindows(states, window=MINIROCKET_WINDOW):
    """[N,T,C] -> (X[N*T, C, window] float32, idx[N*T, 2] of (ic, t)), ordered t-outer/ic-inner.
    Left-padded by repeating the first frame, so every timestep gets a prediction (valid_from = 0)."""
    N, T, C = states.shape
    padded = np.concatenate([np.repeat(states[:, :1], window - 1, axis=1), states], axis=1)
    X = np.concatenate([padded[:, t:t + window, :].transpose(0, 2, 1) for t in range(T)])
    idx = np.concatenate([np.stack([np.arange(N), np.full(N, t)], axis=1) for t in range(T)])
    return X.astype(np.float32), idx


def _fitMiniRocketTransform(X, num_kernels, fit_sample=20000, seed=None):
    """Fit on a random sample of windows (the fit only picks dilations and bias quantiles). Random,
    not X[:n]: rows are t-outer, so a prefix would calibrate to the first timesteps only.
    random_state goes to sktime too -- left at None, three --seed 0 runs spread 0.62-0.67 macro-F1."""
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    tf = MiniRocketMultivariate(num_kernels=num_kernels, n_jobs=-1, random_state=seed)
    if len(X) > fit_sample:
        pick = np.random.default_rng(seed).choice(len(X), fit_sample, replace=False)
        X = X[np.sort(pick)]
    tf.fit(X)
    return tf


def _minirocketChunks(tf, X, n_feat, mask=None):
    """Stream the transform: yield (chunk_start, F_chunk [rows, n_feat+1] float32); the trailing ones
    column is the ridge intercept. The full matrix is never built (at T=100 it would be ~5.7 GB
    and OOM a 16 GB host); re-transforming per pass costs ~6s vs ~125s for the solve.
    mask: optional [M] bool; only kept rows are yielded, fully-masked chunks skipped."""
    for i in range(0, len(X), MINIROCKET_CHUNK):
        sl = slice(i, i + MINIROCKET_CHUNK)
        keep = None
        if mask is not None:
            keep = mask[sl]
            if not keep.any():
                continue
        block = tf.transform(X[sl]).to_numpy(dtype=np.float32)
        F = np.empty((len(block), n_feat + 1), dtype=np.float32)
        F[:, :n_feat] = block
        F[:, -1] = 1.0
        yield i, (F[keep] if keep is not None else F)


def _ridgeFit(tf, X, n_feat, targets, mask=None, alpha=1.0):
    """Ridge regression onto one-hot targets, accumulated chunk by chunk (equivalent to
    RidgeClassifier, but only one chunk ever in float64). targets: list of (y, num_classes) -> list
    of coefficient matrices. Heads passed together share the K x K Gram, which is the dominant cost
    (~124s per build at 30 min); stage 2 needs its own call since it is masked to thrust frames."""
    P = n_feat + 1
    G = np.zeros((P, P))
    Bs = [np.zeros((P, n)) for _, n in targets]
    for i, F in _minirocketChunks(tf, X, n_feat, mask):
        Fb = F.astype(np.float64)
        G += Fb.T @ Fb
        sl = slice(i, i + MINIROCKET_CHUNK)
        for B, (y, n) in zip(Bs, targets):
            yb = y[sl][mask[sl]] if mask is not None else y[sl]
            Y = np.zeros((len(yb), n))
            Y[np.arange(len(yb)), yb] = 1.0
            B += Fb.T @ Y
    G[np.diag_indices(P - 1)] += alpha      # last column is the intercept -- leave it unpenalised
    return [np.linalg.solve(G, B) for B in Bs]


def _ridgePredict(tf, X, n_feat, Ws, mask=None):
    """One streamed pass, every head predicted from it -> list of flat label arrays, one per W.
    Batched for the same reason _ridgeFit batches heads: the transform, not the matmul, is what a
    second pass would repeat."""
    outs = [[] for _ in Ws]
    for _, F in _minirocketChunks(tf, X, n_feat, mask):
        Fb = F.astype(np.float64)
        for o, W in zip(outs, Ws):
            o.append((Fb @ W).argmax(axis=1))
    return [np.concatenate(o) if o else np.zeros(0, dtype=np.int64) for o in outs]


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
    """One shared windowed MiniRocket transform, then a ridge head per requested mode. All heads are
    fitted, and all eval predictions made in one streamed pass, before anything is reported --
    features are never stored, so interleaving would re-transform per report."""
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
    tTf.tocStr("MiniRocket Transform Fit Time")

    y_train_joint = train_joint[idx_train[:, 0], idx_train[:, 1]]
    y_train_stage1 = train_stage1[idx_train[:, 0], idx_train[:, 1]]
    y_eval_joint_flat = eval_joint[idx_eval[:, 0], idx_eval[:, 1]]
    thrust = y_train_joint > 0     # stage 2 is only ever supervised on thrusting frames
    # Full-length stage-2 target. Only the `thrust` rows are ever read, but keeping it full-length
    # lets _ridgeFit slice it with the same chunk bounds as the mask, instead of materializing a
    # thrust-only feature subset (1.1 GB at T=100).
    y_train_stage2 = np.maximum(y_train_joint - 1, 0)

    want_s1 = run_cascade or run_stage1_solo
    want_s2 = run_cascade or run_stage2_solo
    W_joint = W_s1 = W_s2 = None

    # Pass 1: every head supervised on ALL rows (joint, stage 1) -- batched because they share
    # G = F^T F; see _ridgeFit.
    heads = ([(y_train_joint, 4)] if run_joint else []) + ([(y_train_stage1, 2)] if want_s1 else [])
    if heads:
        tFit = timer()
        fitted = _ridgeFit(tf, X_train, n_feat, heads)
        fit_s = tFit.tocVal()   # tocVal, not toc: toc() also prints, which would emit a
        # stray timing line before the 'Entering' block the parser attributes it to.
        if run_joint:
            W_joint = fitted.pop(0)
        if want_s1:
            W_s1 = fitted[0]

    # Pass 2: stage 2, masked to thrust frames -- a different row set, hence a different Gram.
    if want_s2:
        tFit2 = timer()
        W_s2 = _ridgeFit(tf, X_train, n_feat, [(y_train_stage2, 3)], mask=thrust)[0]
        fit2_s = tFit2.tocVal()

    # Pass 3: one streamed pass over the eval windows produces every head's predictions; the
    # reports below only index into them.
    tInf = timer()
    order = [W for W in (W_joint, W_s1, W_s2) if W is not None]
    preds = _ridgePredict(tf, X_eval, n_feat, order) if order else []
    pred_joint = preds.pop(0) if W_joint is not None else None
    pred_s1 = preds.pop(0) if W_s1 is not None else None
    pred_s2 = preds.pop(0) if W_s2 is not None else None
    tInf.tocStr("MiniRocket Inference Time (all heads, one transform pass)")
    del X_train, X_eval

    # --- reporting only from here on ---------------------------------------------------------
    # displaySeqLogData.py splits the log on "Entering <X> Training Loop", so each head's line
    # must be followed directly by that head's own output.
    if run_joint:
        print("\nEntering MiniRocket (joint) Training Loop")
        print(f"\tElapsed time is {fit_s:.4f} seconds."
              + ("  [shared ridge pass, also solves stage 1]" if want_s1 else ""))
        printMiniRocketRidgeSize(tf, W_joint)

        print("\nMiniRocket (joint) Validation")
        _reportFromPredictions(y_eval_joint_flat, pred_joint, JOINT_CLASS_NAMES, print_report=True)
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
            print(f"\tElapsed time is {fit_s:.4f} seconds.")
        printMiniRocketRidgeSize(tf, W_s1)
        if run_stage1_solo:
            print("\nMiniRocket Stage 1 (standalone) Validation")
            _reportFromPredictions((y_eval_joint_flat > 0).astype(np.int64), pred_s1,
                                    STAGE1_CLASS_NAMES, print_report=True)
        # In cascade mode stage 1 carries no metrics of its own -- it is scored inside
        # combineCascadePredictions below, exactly as runClassicCascade's stage-1 block is.

    if want_s2:
        print("\nEntering MiniRocket Stage 2 "
              + ("(standalone)" if run_stage2_solo else "(Type Classifier)") + " Training Loop")
        print(f"\tElapsed time is {fit2_s:.4f} seconds.")
        printMiniRocketRidgeSize(tf, W_s2)

        # standalone stage-2 score on true-thrust frames (see runClassicCascade). The header must
        # match displaySeqLogData's RE_STAGE2_STANDALONE_HDR exactly -- don't reword it.
        print("\nMiniRocket Stage 2 (Type Classifier) Standalone Validation")
        mask = y_eval_joint_flat > 0
        _reportFromPredictions(y_eval_joint_flat[mask] - 1, pred_s2[mask],
                                STAGE2_CLASS_NAMES, print_report=True)

    if run_cascade:
        print("\nMiniRocket Cascade Evaluation")
        tCasc = timer()
        g1 = _minirocketGrid(N_eval, T_eval, idx_eval, pred_s1)
        g2 = _minirocketGrid(N_eval, T_eval, idx_eval, pred_s2)
        combineCascadePredictions(eval_joint, eval_stage1, g1, g2, print_report=True,
                                   plot_name="MiniRocket",
                                   plot_save_path=os.path.join(plotLoc, f"seqpred_MiniRocket_cascade_{logStem}.png"),
                                   valid_from=0, smooth_max_gap=smooth_max_gap)
        tCasc.tocStr("MiniRocket Cascade Inference Time")


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
    # Classic ML / GBDT, PCA+MLP and MiniRocket baselines: windowed rows, outside the backbone loop.
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
        # multi:softmax, not softprob: softprob's .predict() returns probabilities when num_class=2
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
