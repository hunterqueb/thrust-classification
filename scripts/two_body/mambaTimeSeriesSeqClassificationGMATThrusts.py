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
# timestep of a trajectory) paired with a Mamba stage-2 per-timestep type classifier -- and only
# participates in cascade/stage-solo modes (no single joint 4-class 'hybrid' model exists).
#
# Classic ML / GBDT baselines (LightGBM on by default via --no-classic to disable; --xgboost/
# --catboost/--rf/--extratrees to opt in) and --mlp (PCA+MLP) operate on Hankel-windowed rows
# (a trailing per-timestep context window, via buildHankelWindowRowsPerTimestep) rather than
# full [B,T,C] sequences -- tree/linear models have no global-pooling requirement, unlike
# MiniRocket, so per-timestep windowed rows are a natural fit for them. --minirocket adds a
# standalone whole-trajectory 4-class MiniRocket comparison (broadcast like --hybrid's stage 1,
# --mode joint/all only) -- MiniRocket itself still only supports whole-series input.
#
# $ python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
# --systems 1500 --propMin 30 --orbit vleo --mode all
import argparse

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
parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data to use for training")
parser.add_argument("--mode", type=str, default="all", choices=["all", "joint", "cascade", "stage1", "stage2"],
                     help="'joint': single 4-class per-timestep model. 'cascade': binary detector + "
                          "3-class type classifier, combined and compared against the joint model. "
                          "'stage1'/'stage2': train/evaluate only that cascade stage standalone. "
                          "'all' (default): joint and cascade both.")
parser.add_argument("--transformer", dest="use_transformer", action="store_true", help="Enable per-timestep Transformer model comparison (disabled by default)")
parser.add_argument("--cnn", dest="use_cnn", action="store_true", help="Enable per-timestep 1D-CNN (InceptionTime) model comparison (disabled by default)")
parser.add_argument("--hybrid", dest="use_hybrid", action="store_true", help="Enable 'hybrid' cascade comparison: whole-trajectory MiniRocket stage-1 detector + Mamba stage-2 type classifier (disabled by default; cascade/stage-solo modes only, no joint form)")
parser.add_argument("--no-classic", dest="use_classic", action="store_false", help="Disable the LightGBM per-timestep classic-ML comparison (enabled by default)")
parser.add_argument("--xgboost", dest="use_xgboost", action="store_true", help="Enable XGBoost per-timestep classic-ML comparison (disabled by default)")
parser.add_argument("--catboost", dest="use_catboost", action="store_true", help="Enable CatBoost per-timestep classic-ML comparison (disabled by default)")
parser.add_argument("--rf", dest="use_random_forest", action="store_true", help="Enable Random Forest per-timestep classic-ML comparison (disabled by default)")
parser.add_argument("--extratrees", dest="use_extra_trees", action="store_true", help="Enable Extra Trees per-timestep classic-ML comparison (disabled by default)")
parser.add_argument("--pca", type=int, default=None, help="If set, PCA-reduce the Hankel-window features to this many components for the --mlp comparison (default: keep 95%% variance)")
parser.add_argument("--mlp", dest="use_mlp", action="store_true", help="Enable PCA+MLP per-timestep comparison on Hankel-windowed rows (disabled by default)")
parser.add_argument("--minirocket", dest="use_minirocket", action="store_true", help="Enable standalone whole-trajectory MiniRocket 4-class comparison, broadcast across timesteps (disabled by default; --mode joint/all only)")

parser.set_defaults(use_lstm=True)
parser.set_defaults(use_mamba=True)
parser.set_defaults(OE=False)
parser.set_defaults(noise=False)
parser.set_defaults(norm=False)
parser.set_defaults(one_pass=False)
parser.set_defaults(save_to_log=False)
parser.set_defaults(use_energy=False)
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
if args.pca is not None and args.pca > 0:
    pca_n_components = args.pca
else:
    pca_n_components = 0.95

import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

from qutils.tictoc import timer
from qutils.ml.utils import getDevice, printModelParmSize
from qutils.ml.classifer import apply_noise
from qutils.ml.mamba import Mamba, MambaConfig

device = getDevice()

strAdd = ""
if useEnergy:
    strAdd = strAdd + "Energy_"
if useOE:
    strAdd = strAdd + "OE_"
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

if strAdd.endswith("_"):
    strAdd = strAdd[:-1]

print(f"Training with {int(4*train_ratio*numRandSys)} systems")

logLoc = "gmat/data/seqClassification/" + str(orbitType) + "/" + str(numMinProp) + "min-" + str(numRandSys) + "/"
logFileLoc = logLoc + str(numMinProp) + "min" + str(numRandSys) + strAdd + '.log'

if save_to_log:
    from contextlib import redirect_stdout, redirect_stderr

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 10000)
    pd.set_option('display.expand_frame_repr', False)

    import warnings
    warnings.filterwarnings("ignore")

    import os
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


def _forwardFillFromFirstEvent(thrustingTime):
    """Marks every timestep from the first '1' onward as thrusting -- an impulsive burn
    permanently alters the orbit, so 'thrust occurred' is treated as true for the remainder of
    the propagation window, not just the instant of the delta-v. Idempotent if the array already
    encodes the persisted window rather than just the instant."""
    return np.maximum.accumulate(thrustingTime, axis=1)


def _applyTransforms(states_by_class, useOE, useNorm, useNoise, useEnergy, pos_noise_std, vel_noise_std):
    if useNoise:
        for c in CLASS_ORDER:
            states_by_class[c] = apply_noise(states_by_class[c], pos_noise_std, vel_noise_std)

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

    if useEnergy:
        from qutils.orbital import orbitalEnergy
        energy_by_class = {}
        norming_energy = None
        for c in CLASS_ORDER:
            s = states_by_class[c] if oe_by_class is None else oe_by_class[c][:, :, 0:6]
            n_ic, T = s.shape[0], s.shape[1]
            energy = np.zeros((n_ic, T, 1))
            for i in range(n_ic):
                energy[i, :, 0] = orbitalEnergy(s[i, :, :])
            if useNorm:
                if norming_energy is None:
                    norming_energy = energy[0, 0, 0]
                energy[:, :, 0] = energy[:, :, 0] / norming_energy
            energy_by_class[c] = energy

        if oe_by_class is not None:
            for c in CLASS_ORDER:
                states_by_class[c] = np.concatenate((oe_by_class[c], energy_by_class[c]), axis=2)
        else:
            for c in CLASS_ORDER:
                states_by_class[c] = energy_by_class[c]

    return states_by_class


def _load_and_label(loc, useOE, useNorm, useNoise, useEnergy, pos_noise_std, vel_noise_std):
    states_by_class = {}
    thrusting_by_class = {}
    for class_name in CLASS_ORDER:
        npz = np.load(f"{loc}/statesArray{class_name}.npz")
        states = npz[f"statesArray{class_name}"]
        N, T = states.shape[0], states.shape[1]
        tt = _getThrustingTime(npz, class_name, N, T, warn_if_missing=(class_name != "NoThrust"))
        if class_name == "ImpBurn":
            tt = _forwardFillFromFirstEvent(tt)
        states_by_class[class_name] = states
        thrusting_by_class[class_name] = tt

    n_ic_per_class = [states_by_class[c].shape[0] for c in CLASS_ORDER]
    T_per_class = [states_by_class[c].shape[1] for c in CLASS_ORDER]
    assert len(set(T_per_class)) == 1, f"Timestep count mismatch across classes: {dict(zip(CLASS_ORDER, T_per_class))}"

    states_by_class = _applyTransforms(states_by_class, useOE, useNorm, useNoise, useEnergy, pos_noise_std, vel_noise_std)

    states_cat = np.concatenate([states_by_class[c] for c in CLASS_ORDER], axis=0)

    joint_labels_list = [thrusting_by_class[c].squeeze(-1).astype(np.int64) * JOINT_LABELS[c] for c in CLASS_ORDER]
    y_joint = np.concatenate(joint_labels_list, axis=0)

    return states_cat, y_joint, n_ic_per_class


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


def prepareInSequenceThrustClassificationDatasets(
    yaml_config, data_config,
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
    pos_noise_std=1e-3, vel_noise_std=1e-3,
    batch_size=16, pad_idx=-100, seed=None,
    supress_print=False, return_meta=False,
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
    """
    useOE = yaml_config['useOE']
    useNorm = yaml_config['useNorm']
    useNoise = yaml_config['useNoise']
    useEnergy = yaml_config['useEnergy']

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

    states, y_joint, n_ic_per_class = _load_and_label(
        dataLoc, useOE, useNorm, useNoise, useEnergy, pos_noise_std, vel_noise_std
    )

    train_mask, val_mask, test_mask = _icGroupSplit(n_ic_per_class, train_ratio, val_ratio, test_ratio, seed=seed)

    train_data, train_joint = states[train_mask], y_joint[train_mask]
    val_data, val_joint = states[val_mask], y_joint[val_mask]

    if test_set != train_set or test_systems != systems:
        states_t, y_joint_t, n_ic_per_class_t = _load_and_label(
            dataLoc_test, useOE, useNorm, useNoise, useEnergy, pos_noise_std, vel_noise_std
        )
        _, _, test_mask_t = _icGroupSplit(n_ic_per_class_t, train_ratio, val_ratio, test_ratio, seed=seed)
        test_data, test_joint = states_t[test_mask_t], y_joint_t[test_mask_t]
    else:
        test_data, test_joint = states[test_mask], y_joint[test_mask]

    train_stage1, train_stage2 = _deriveStage1Stage2(train_joint, pad_idx)
    val_stage1, val_stage2 = _deriveStage1Stage2(val_joint, pad_idx)
    test_stage1, test_stage2 = _deriveStage1Stage2(test_joint, pad_idx)

    if not supress_print:
        print(f"train_data {train_data.shape}  val_data {val_data.shape}  test_data {test_data.shape}")

    def _make_loader(data, yj, y1, y2, shuffle):
        ds = TensorDataset(
            torch.from_numpy(data),
            torch.from_numpy(yj).long(),
            torch.from_numpy(y1).long(),
            torch.from_numpy(y2).long(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    train_loader = _make_loader(train_data, train_joint, train_stage1, train_stage2, True)
    val_loader = _make_loader(val_data, val_joint, val_stage1, val_stage2, False)
    test_loader = _make_loader(test_data, test_joint, test_stage1, test_stage2, False)

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
    def __init__(self, config, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mamba = Mamba(config)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """x: [batch_size, seq_length, input_size]"""
        h_n = self.mamba(x)          # [B, T, hidden_size]
        logits = self.fc(h_n)        # [B, T, num_classes]
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
# same Mamba per-timestep model used elsewhere in this script. Consequently 'hybrid' only
# participates in cascade/stage-solo modes -- there is no single joint 4-class hybrid model, and
# stage 1's binary decision applies to a whole trajectory ("does this ~30-minute window contain
# a thrust event anywhere"), not to individual timesteps.
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


def _infer_class_weights(loader, num_classes, mode, pad_idx=-100, dtype=torch.float32, device="cpu"):
    counts = torch.zeros(num_classes, dtype=torch.long)
    with torch.no_grad():
        for _, y_joint, y_stage1, y_stage2 in loader:
            y = _select_labels(y_joint, y_stage1, y_stage2, mode).reshape(-1).long()
            if pad_idx is not None:
                y = y[y != pad_idx]
            counts += torch.bincount(y, minlength=num_classes)
    w = counts.sum() / torch.clamp(counts.to(dtype), min=1.0)
    w = w / w.mean()
    return w.to(device=device, dtype=dtype)


def train_model(model, train_loader, val_loader, num_epochs, num_classes, mode,
                 pad_idx=-100, class_weights=None, schedulerPatience=3, verbose=True):
    model = model.to(device)
    param_dtype = torch.float64
    model = model.double()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if class_weights is None:
        class_weights = _infer_class_weights(train_loader, num_classes, mode, pad_idx, dtype=param_dtype, device=device)
    else:
        class_weights = torch.as_tensor(class_weights, device=device, dtype=param_dtype)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=pad_idx)
    event_labels, all_labels = _default_label_sets(mode, num_classes)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=schedulerPatience)

    best_loss = float('inf')
    ESpatience = schedulerPatience * 2
    counter = 0

    timeToTrain = timer()

    for epoch in range(num_epochs):
        model.train()
        total_loss, loss_batches = 0.0, 0
        skipped_train_batches = 0
        for x, y_joint, y_stage1, y_stage2 in train_loader:
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loss_batches += 1

        avg_loss = total_loss / max(1, loss_batches)
        if verbose:
            msg = f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f}"
            if skipped_train_batches:
                msg += f"  ({skipped_train_batches} batch(es) skipped: no unmasked labels)"
            print(msg)

        model.eval()
        all_preds, all_targets = [], []
        val_loss, val_loss_batches = 0.0, 0
        with torch.no_grad():
            for x, y_joint, y_stage1, y_stage2 in val_loader:
                x = x.to(device, non_blocking=True)
                if x.dtype != param_dtype:
                    x = x.to(param_dtype)
                labels = _select_labels(y_joint, y_stage1, y_stage2, mode).to(device, non_blocking=True).long()

                logits = model(x)
                B, T, C = logits.shape
                loss = criterion(logits.reshape(B * T, C), labels.reshape(B * T))
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

        if verbose and len(y_true) > 0:
            p_ev, r_ev, f_ev, _ = precision_recall_fscore_support(y_true, y_pred, labels=event_labels, average='macro', zero_division=0)
            print(f"Val Event P(macro {event_labels}): {p_ev:.4f} | R: {r_ev:.4f} | F1: {f_ev:.4f}")
            p_pc, r_pc, f_pc, _ = precision_recall_fscore_support(y_true, y_pred, labels=all_labels, average=None, zero_division=0)
            print(f"Per-class P: {p_pc}  R: {r_pc}  F1: {f_pc}")
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


def validateInSequenceClassifier(model, loader, mode, num_classes, device, pad_idx=-100,
                                  class_names=None, print_report=True, return_predictions=False):
    model.eval()
    param_dtype = torch.float64
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y_joint, y_stage1, y_stage2 in loader:
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
        for x, y_joint, y_stage1, y_stage2 in loader:
            x = x.to(device, non_blocking=True)
            if x.dtype != param_dtype:
                x = x.to(param_dtype)
            preds = model(x).argmax(dim=-1)  # [B,T]
            all_pred.append(preds.cpu().numpy())
            all_joint.append(y_joint.numpy())
            all_stage1.append(y_stage1.numpy())
    return np.concatenate(all_joint, axis=0), np.concatenate(all_stage1, axis=0), np.concatenate(all_pred, axis=0)


def combineCascadePredictions(y_true_joint, y_true_stage1, pred_stage1, pred_stage2, print_report=True):
    """y_true_joint/y_true_stage1/pred_stage1: [N,T]; pred_stage2: [N,T] in {0,1,2}. Combines
    stage1 (thrust yes/no) and stage2 (Chemical/Electric/Impulsive) predictions into a single
    4-class per-timestep prediction and reports both the combined result and a stage1-vs-stage2
    error decomposition -- so a cascade shortfall is diagnosable as bad detection vs. bad typing.
    Backbone-agnostic: works whether stage1/stage2 came from matching neural models or the mixed
    MiniRocket-detector + Mamba-type-classifier 'hybrid' backbone."""
    final_pred = np.where(pred_stage1 == 0, 0, pred_stage2 + 1)

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


def runCascadeEvaluation(stage1_model, stage2_model, loader, device, print_report=True):
    """Neural-neural cascade: both stages are per-timestep nn.Modules over the same loader."""
    y_true_joint, y_true_stage1, pred_stage1 = _predictPerTimestepNeural(stage1_model, loader, device)
    _, _, pred_stage2 = _predictPerTimestepNeural(stage2_model, loader, device)
    return combineCascadePredictions(y_true_joint, y_true_stage1, pred_stage1, pred_stage2, print_report=print_report)


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
                      hankel_L, pad_idx, size_fn):
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
    return clf


def runClassicCascade(name, classifier_ctor, train_data, train_joint, eval_data, eval_joint,
                       hankel_L, pad_idx, size_fn):
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

    print(f"\n{name} Cascade Evaluation")
    tInf = timer()
    pred_stage1 = predictClassicPerTimestep(clf1, eval_data, hankel_L)
    pred_stage2 = predictClassicPerTimestep(clf2, eval_data, hankel_L)
    combineCascadePredictions(eval_joint, eval_stage1, pred_stage1, pred_stage2, print_report=True)
    tInf.tocStr(f"{name} Cascade Inference Time")
    return clf1, clf2


def runClassicMLModes(name, classifier_ctor, size_fn,
                       train_data, train_joint, train_stage1, train_stage2,
                       eval_data, eval_joint, eval_stage1, eval_stage2,
                       hankel_L, pad_idx,
                       run_joint, run_cascade, run_stage1_solo, run_stage2_solo):
    if run_joint:
        runClassicFamily(name, classifier_ctor, 4, "joint", JOINT_CLASS_NAMES,
                          train_data, train_joint, eval_data, eval_joint, hankel_L, pad_idx, size_fn)
    if run_cascade:
        runClassicCascade(name, classifier_ctor, train_data, train_joint, eval_data, eval_joint,
                           hankel_L, pad_idx, size_fn)
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
                     hankel_L, pad_idx, pca_n_components, device, num_epochs):
    print(f"\nEntering PCA+MLP ({mode_name}) Training Loop")
    X_train, y_train, X_val, y_val, X_eval, y_eval, _, _ = buildPCAHankelFeatures(
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
    return model


def runPCAMLPCascade(train_data, train_joint, val_data, val_joint, eval_data, eval_joint,
                      hankel_L, pad_idx, pca_n_components, device, num_epochs):
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
    combineCascadePredictions(eval_joint, eval_stage1, pred_stage1, pred_stage2, print_report=True)
    tInf.tocStr("PCA+MLP Cascade Inference Time")
    return model1, model2


# ---------------------------------------------------------------------------
# Standalone whole-trajectory MiniRocket baseline (--minirocket): a direct per-timestep-
# comparable analog of the whole-trajectory script's 4-class --minirocket comparison. Only
# supports joint mode -- one 4-class label per whole ~30-step trajectory, broadcast across every
# timestep -- for the same reason 'hybrid' stage 1 is whole-trajectory (MiniRocket's PPV-pooling
# transform is calibrated to and gets its power from the full series it's fit on).
# ---------------------------------------------------------------------------
def trainMiniRocketJointDetector(train_data, train_joint, num_kernels=10000):
    from sktime.classification.kernel_based import RocketClassifier
    X_train = np.transpose(train_data, (0, 2, 1))     # [N,T,C] -> [N,C,T] sktime panel format
    y_train = train_joint.max(axis=1).astype(np.int64)  # each trajectory's single class index
    clf = RocketClassifier(num_kernels=num_kernels, rocket_transform='minirocket', n_jobs=-1)
    clf.fit(X_train, y_train)
    printMiniROCKETSize(clf)
    return clf


def predictMiniRocketJointTrajectory(clf, data):
    X = np.transpose(data, (0, 2, 1))
    return np.asarray(clf.predict(X)).astype(np.int64)


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
        batch_size=16,
    )

    input_size = train_data.shape[2]
    hidden_factor = 8
    hidden_size = int(input_size * hidden_factor)
    num_layers = 1
    num_epochs = 100
    if useOnePass:
        num_epochs = 1
    schedulerPatience = 5

    use_test_eval = (testSet != orbitType or testSys != numRandSys)
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
                  "~30-minute window contain thrust anywhere') + Mamba stage-2 per-timestep type "
                  "classifier. No joint 4-class form, so --mode joint is a no-op for this "
                  "backbone. Stage 1's trajectory-level decision is broadcast across every "
                  "timestep of a trajectory for the combined per-timestep report below -- a "
                  "positive trajectory has no further per-timestep background suppression, so "
                  "stage 2 alone determines which minutes look idle vs. thrusting within it.")

            stage1_clf = None
            if run_cascade or run_stage1_solo:
                print("\nEntering HYBRID Stage 1 (MiniRocket, whole-trajectory) Training")
                miniRocketTimer = timer()
                stage1_clf = trainMiniRocketStage1Detector(train_data, train_stage1)
                miniRocketTimer.tocStr("HYBRID Stage 1 (MiniRocket) Training Time")

            model_stage2 = None
            if run_cascade or run_stage2_solo:
                print("\nEntering HYBRID Stage 2 (Mamba Type Classifier) Training Loop")
                model_stage2 = build_model("mamba", 3, input_size, hidden_size, num_layers)
                train_model(model_stage2, train_loader, val_loader, num_epochs=num_epochs, num_classes=3,
                            mode='stage2', schedulerPatience=schedulerPatience)
                printModelParmSize(model_stage2)

            if run_stage1_solo:
                print("\nHYBRID Stage 1 (MiniRocket, whole-trajectory) Validation")
                pred_stage1_traj = predictMiniRocketStage1Trajectory(stage1_clf, eval_data)
                true_stage1_traj = eval_stage1.any(axis=1).astype(np.int64)
                _reportFromPredictions(true_stage1_traj, pred_stage1_traj, STAGE1_CLASS_NAMES, print_report=True)

            if run_stage2_solo:
                print("\nHYBRID Stage 2 (Mamba) Validation")
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
                combineCascadePredictions(eval_joint, eval_stage1, pred_stage1_bcast, pred_stage2, print_report=True)
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

            print(f"\n{backbone.upper()} Cascade Evaluation")
            cascadeInference = timer()
            runCascadeEvaluation(model_stage1, model_stage2, eval_loader, device=device)
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
                           hankel_L, -100, run_joint, run_cascade, run_stage1_solo, run_stage2_solo)

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
                           hankel_L, -100, run_joint, run_cascade, run_stage1_solo, run_stage2_solo)

    if use_catboost:
        from catboost import CatBoostClassifier
        from qutils.ml.classic.classifier import printClassicModelSize
        ctor = lambda nc: CatBoostClassifier(loss_function="MultiClass", classes_count=nc, iterations=200, depth=6,
                                              learning_rate=0.05, bootstrap_type="Bernoulli", subsample=0.8,
                                              colsample_bylevel=0.8, verbose=False, allow_writing_files=False)
        runClassicMLModes("CatBoost", ctor, printClassicModelSize,
                           train_data, train_joint, train_stage1, train_stage2,
                           eval_data, eval_joint, eval_stage1, eval_stage2,
                           hankel_L, -100, run_joint, run_cascade, run_stage1_solo, run_stage2_solo)

    if use_random_forest:
        from sklearn.ensemble import RandomForestClassifier
        from qutils.ml.classic.classifier import printClassicModelSize
        ctor = lambda nc: RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1)
        runClassicMLModes("Random Forest", ctor, printClassicModelSize,
                           train_data, train_joint, train_stage1, train_stage2,
                           eval_data, eval_joint, eval_stage1, eval_stage2,
                           hankel_L, -100, run_joint, run_cascade, run_stage1_solo, run_stage2_solo)

    if use_extra_trees:
        from sklearn.ensemble import ExtraTreesClassifier
        from qutils.ml.classic.classifier import printClassicModelSize
        ctor = lambda nc: ExtraTreesClassifier(n_estimators=300, max_depth=None, n_jobs=-1)
        runClassicMLModes("Extra Trees", ctor, printClassicModelSize,
                           train_data, train_joint, train_stage1, train_stage2,
                           eval_data, eval_joint, eval_stage1, eval_stage2,
                           hankel_L, -100, run_joint, run_cascade, run_stage1_solo, run_stage2_solo)

    if use_mlp:
        print(f"\n{'='*80}\nPCA+MLP baseline (Hankel window length={hankel_L})\n{'='*80}")
        if run_joint:
            runPCAMLPFamily("joint", 4, JOINT_CLASS_NAMES,
                             train_data, train_joint, val_data, val_joint, eval_data, eval_joint,
                             hankel_L, -100, pca_n_components, device, num_epochs)
        if run_cascade:
            runPCAMLPCascade(train_data, train_joint, val_data, val_joint, eval_data, eval_joint,
                              hankel_L, -100, pca_n_components, device, num_epochs)
        if run_stage1_solo:
            runPCAMLPFamily("stage1", 2, STAGE1_CLASS_NAMES,
                             train_data, train_stage1, val_data, val_stage1_np, eval_data, eval_stage1,
                             hankel_L, -100, pca_n_components, device, num_epochs)
        if run_stage2_solo:
            runPCAMLPFamily("stage2", 3, STAGE2_CLASS_NAMES,
                             train_data, train_stage2, val_data, val_stage2_np, eval_data, eval_stage2,
                             hankel_L, -100, pca_n_components, device, num_epochs)

    if use_minirocket:
        print(f"\n{'='*80}\nBackbone: MINIROCKET (whole-trajectory, broadcast)\n{'='*80}")
        if run_joint:
            print("\nEntering MiniRocket (whole-trajectory, Joint 4-class) Training")
            mrTimer = timer()
            mr_clf = trainMiniRocketJointDetector(train_data, train_joint)
            mrTimer.tocStr("MiniRocket Joint Training Time")

            print("\nMiniRocket Joint Validation")
            mrInf = timer()
            pred_traj = predictMiniRocketJointTrajectory(mr_clf, eval_data)
            T = eval_data.shape[1]
            pred_bcast = np.repeat(pred_traj[:, None], T, axis=1)
            _reportFromPredictions(eval_joint.reshape(-1), pred_bcast.reshape(-1), JOINT_CLASS_NAMES, print_report=True)
            mrInf.tocStr("MiniRocket Joint Inference Time")
        else:
            print("[note] MINIROCKET only supports --mode joint/all (whole-trajectory 4-class "
                  "classifier, broadcast across timesteps); no stage1/stage2/cascade form.")


if __name__ == "__main__":
    if save_to_log:
        with open(logFileLoc, 'w', buffering=1, encoding='utf-8') as f, \
                redirect_stdout(f), redirect_stderr(f):
            main()
    else:
        main()
