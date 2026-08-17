# parse at the beginning before long imports
# script usage

# call the script from the main folder directory, adding --save saves the output to a log file in the location of the datasets
# $ python scripts/classification/mambaTimeSeriesClassificationGMATThrusts.py \
# --systems 10000 --propMin 5 --OE --norm --orbit vleo
#
# --frame selects the feature representation:
#   eci    (default) Cartesian ECI states, or OE via --OE -- loaded via qutils.prepareThrustClassificationDatasets
#   aer    Az/El/Range/dAz/dEl/dRange from a radar-realistic ground station (leo/vleo/meo)
#   radec  RA/Dec/dRA/dDec, angles-only, from an optical-realistic ground station (geo/heo)
# aer/radec expect {frame}Array{Chemical,Electric,ImpBurn,NoThrust}.npz in the same per-orbit data
# directory as statesArray*.npz (see generateSpacecraftThrustOptGroundStation.py in GMAT-Thrust-Data).
# --OE/--energy/--noise only apply to --frame eci; --norm still applies but means per-channel
# z-score normalization (fit on the training split) instead of the ECI/OE-specific normalization.
# $ python scripts/classification/mambaTimeSeriesClassificationGMATThrusts.py \
# --systems 800 --propMin 30 --frame aer --orbit vleo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-lstm',dest="use_lstm", action='store_false', help='Use LSTM model')
parser.add_argument("--systems", type=int, default=10000, help="Number of random systems to access")
parser.add_argument("--propMin", type=int, default=30, help="Minimum propagation time in minutes")
parser.add_argument("--orbit", type=str, default="vleo", help="Orbit type: vleo, leo")
parser.add_argument("--test", type=str, default=None, help="Orbit type for test set: vleo, leo, OR the same as --orbit and an integer number of random systems to use for testing")
parser.add_argument("--testSys", type=int, default=10000, help="Number of systems to use for testing if --test is a different string than --orbit")
parser.add_argument("--OE", action='store_true', help="Use OE elements instead of ECI states")
parser.add_argument("--noise", action='store_true', help="Add noise to the data")
parser.add_argument("--velNoise",type=float,default=1e-3,help="std of noise to add to velocity terms")
parser.add_argument("--norm", action='store_true', help="Normalize the semi-major axis by Earth's radius")
parser.add_argument("--one-pass",dest="one_pass",action='store_true', help="Use one pass learning.")
parser.add_argument("--save",dest="save_to_log",action="store_true",help="output console printout to log file in the same location as datasets")
parser.add_argument("--energy",dest="use_energy",action="store_true",help="Use energy as a feature.")
parser.add_argument("--hybrid",dest="use_hybrid",action="store_true",help="Use a hybrid network.")
parser.add_argument("--superweight",dest="find_SW",action="store_true",help="Superweight analysis")
parser.add_argument("--no-classic",dest="use_classic",action="store_false",help="Use classic ML classification for comparison")
parser.add_argument("--nearest",dest="use_nearestNeighbor",action="store_true",help="Use classic ML classification (1-nearest neighbor w/ DTW) for comparison")
parser.add_argument('--saveNets', dest="saveNets",action='store_true', help='Save the trained networks. Saves to the same location as a saved log file.')
parser.add_argument('--classic', dest="old_classic",action='store_true', help='DO NOT USE. DUMMY ARGUMENT TO AVOID BREAKING OLD SCRIPTS.')
parser.add_argument('--shap',dest="run_shap",action='store_true', help='run shap analysis for interpretation of feature importance.')
parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data to use for training")
parser.add_argument("--pca", type=int, default=None, help="If set to an integer, use PCA to reduce the input features to this number of components.")
parser.add_argument('--mlp', dest="useMLP", action='store_true', help='Use a simple MLP on Hankel+PCA pooled data for comparison.')
parser.add_argument("--transformer", dest="use_transformer", action="store_true", help="Enable Transformer model comparison (disabled by default)")
parser.add_argument('--minirocket', dest="use_minirocket", action='store_true', help='Use MiniRocket classifier for comparison')
parser.add_argument('--xgboost', dest="use_xgboost", action='store_true', help='Use XGBoost classifier for comparison (same flattened features as LightGBM)')
parser.add_argument('--catboost', dest="use_catboost", action='store_true', help='Use CatBoost classifier for comparison (same flattened features as LightGBM)')
parser.add_argument('--rf', dest="use_random_forest", action='store_true', help='Use Random Forest classifier for comparison (same flattened features as LightGBM)')
parser.add_argument('--extratrees', dest="use_extra_trees", action='store_true', help='Use Extra Trees classifier for comparison (same flattened features as LightGBM)')
parser.add_argument('--cnn', dest="use_cnn", action='store_true', help='Use a 1D-CNN (InceptionTime-style) classifier for comparison')
parser.add_argument("--frame", type=str, default="eci", choices=["eci", "aer", "radec"],
                     help="Feature representation: 'eci' (default, Cartesian/OE via --OE), "
                          "'aer' (Az/El/Range/rates, radar-realistic ground station), "
                          "'radec' (RA/Dec/rates, angles-only, optical-realistic ground station)")

parser.set_defaults(use_lstm=True)
parser.set_defaults(OE=False)
parser.set_defaults(noise=False)
parser.set_defaults(norm=False)
parser.set_defaults(one_pass=False)
parser.set_defaults(save_to_log=False)
parser.set_defaults(use_energy=False)
parser.set_defaults(use_hybrid=False)
parser.set_defaults(find_SW=False)
parser.set_defaults(use_classic=True)
parser.set_defaults(use_nearestNeighbor=False)
parser.set_defaults(saveNets=False)
parser.set_defaults(run_shap=False)
parser.set_defaults(use_transformer=False)
parser.set_defaults(use_minirocket=False)
parser.set_defaults(use_xgboost=False)
parser.set_defaults(use_catboost=False)
parser.set_defaults(use_random_forest=False)
parser.set_defaults(use_extra_trees=False)
parser.set_defaults(use_cnn=False)

args = parser.parse_args()
use_lstm = args.use_lstm
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
useEnergy=args.use_energy
useHybrid=args.use_hybrid
find_SW=args.find_SW
use_classic = args.use_classic
use_nearestNeighbor = args.use_nearestNeighbor
saveNets = args.saveNets
velNoise = args.velNoise
run_shap = args.run_shap
train_ratio = args.train_ratio
use_transformer = args.use_transformer
use_minirocket = args.use_minirocket
use_xgboost = args.use_xgboost
use_catboost = args.use_catboost
use_random_forest = args.use_random_forest
use_extra_trees = args.use_extra_trees
use_cnn = args.use_cnn
frame = args.frame

if frame != "eci":
    _disabled = []
    if useOE:
        _disabled.append("--OE")
    if useEnergy:
        _disabled.append("--energy")
    if useNoise:
        _disabled.append("--noise")
    if _disabled:
        print(f"[warning] {', '.join(_disabled)} only apply to --frame eci; ignoring for --frame {frame}.")
    useOE = False
    useEnergy = False
    useNoise = False
    if useNorm:
        print(f"[note] --norm for --frame {frame} means per-channel z-score normalization "
              f"fit on the training split, not the ECI/OE-specific normalization.")

if args.pca is not None and args.pca > 0:
    pca_enabled = True
    pca_n_components = args.pca
else:
    pca_enabled = False
    pca_n_components = None
useMLP = args.useMLP

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from qutils.tictoc import timer
from qutils.ml.utils import getDevice, printModelParmSize
from qutils.ml.classifer import trainClassifier, LSTMClassifier, validateMultiClassClassifier
from qutils.ml.mamba import Mamba, MambaConfig, MambaClassifier
from qutils.ml.superweight import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight, findMambaSuperActivation,plotSuperActivation
from qutils.ml.shap import run_shap_analysis

#tranformer classifier for time series data
class TransformerClassifier(nn.Module):
    """Encoder-only Transformer with a learnable CLS token + positional embedding.

    The previous implementation ran a full encoder-decoder nn.Transformer with x fed as both
    src and tgt, which is not a standard classification setup. This is encoder-only with a
    CLS-token pooling head, matching the LSTM/Mamba/CNN classifiers' interface.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, nhead=8, dim_feedforward=64, dropout=0.1, max_len=4096):
        super(TransformerClassifier, self).__init__()

        self.d_model = hidden_size  # Output of transformer & input to fc

        # d_model must be divisible by nhead; fall back to the largest divisor <= 8 if not
        if self.d_model % nhead != 0:
            for cand in (8, 4, 2, 1):
                if self.d_model % cand == 0:
                    nhead = cand
                    break

        self.embedding = nn.Linear(input_size, self.d_model)  # Project input to match d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len + 1, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(self.d_model, num_classes)  # Final classification layer

    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        batch_size, seq_len, _ = x.shape
        x = self.embedding(x)                                    # [batch_size, seq_length, d_model]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)    # [batch_size, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)                     # [batch_size, seq_length+1, d_model]
        x = x + self.pos_embedding[:, :seq_len + 1, :]
        out = self.encoder(x)                                     # [batch_size, seq_length+1, d_model]
        cls_out = out[:, 0, :]                                    # [batch_size, d_model]
        logits = self.fc(cls_out)                                 # [batch_size, num_classes]
        return logits


class InceptionModule(nn.Module):
    """One InceptionTime module: a 1x1 bottleneck feeding parallel odd-kernel convs plus a
    max-pool branch, concatenated along channels. GroupNorm (not BatchNorm) so training is
    robust to the size-1 trailing batch that an undivided dataset can produce."""
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


class InceptionTimeClassifier(nn.Module):
    """1D-CNN, non-SSM/non-Transformer deep baseline built from stacked InceptionTime modules
    with residual connections every 3 modules, global average pooling, and a linear head."""
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

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        x = x.transpose(1, 2)  # [batch_size, channels, seq_length]
        res_input = x
        shortcut_idx = 0
        for d, module in enumerate(self.inception_modules):
            x = module(x)
            if d % 3 == 2:
                shortcut = self.shortcuts[shortcut_idx](res_input)
                shortcut_idx += 1
                x = torch.relu(x + shortcut)
                res_input = x
        x = self.gap(x).squeeze(-1)  # [batch_size, out_channels]
        logits = self.fc(x)
        return logits


class MLP(nn.Module):
    def __init__(self, d_in, n_classes=4, width=256, depth=2, p_drop=0.1):
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

def trainMLP(model, train_loader, val_loader, opt, scheduler, device, class_weights=None,num_epochs=100):
    timeToTrain = timer()

    ESpatience = 10
    model.train()
    ce = nn.CrossEntropyLoss(weight=class_weights)
    total, correct, loss_sum = 0, 0, 0.0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        for xb, yb in train_loader:
            loss_sum = 0.0
            # xb: (B, 1, d) from your dataset → squeeze
            xb = xb.squeeze(1).to(device)  # (B, d)
            yb = yb.view(-1).long().to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item()*xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
        va_loss, va_acc = eval_epoch(model, val_loader, device)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss_sum/total:.4f}")
        
        # Early stopping logic
        if va_loss < best_loss:
            best_loss = va_loss
            counter = 0
            # Optional: save model checkpoint here
        else:
            counter += 1
            if counter >= ESpatience:
                print("Early stopping triggered.")
                break

    return timeToTrain.toc()

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    num_classes = 4  

    class_correct = torch.zeros(num_classes, dtype=torch.int32)
    class_total = torch.zeros(num_classes, dtype=torch.int32)

    # Collect predictions and labels for scikit-learn metrics
    y_true = []
    y_pred = []

    for xb, yb in loader:
        xb = xb.squeeze(1).to(device)
        yb = yb.view(-1).long().to(device)
        logits = model(xb)
        loss = ce(logits, yb)
        loss_sum += loss.item()*xb.size(0)
        predicted = logits.argmax(dim=1)
        correct += (predicted == yb).sum().item()
        total += xb.size(0)

        # Per-class accuracy calculation
        for i in range(yb.size(0)):
            label = yb[i]
            pred = predicted[i]
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

    avg_val_loss = loss / len(loader)
    val_accuracy = 100.0 * correct / total

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")
    classlabels = ['No Thrust','Chemical','Electric','Impulsive']

    print("Per-Class Validation Accuracy:")
    for i in range(4):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i].item() / class_total[i].item()
            if classlabels is not None:
                print(f"  {classlabels[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
            else:
                print(f"  Class {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            if classlabels is not None:
                print(f"  {classlabels[i]}: No samples")
            else:
                print(f"  Class {i}: No samples")

    return loss_sum/total, correct/total


strAdd = ""
if frame != "eci":
    strAdd = strAdd + frame.upper() + "_"
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
if useHybrid:
    strAdd = strAdd + "Hybrid_"
# if use_classic:
#     strAdd = strAdd + "DT_"
# if use_nearestNeighbor:
#     strAdd = strAdd + "1-NN_"
if train_ratio != 0.7:
    strAdd = strAdd + f"Train_{int(4*train_ratio*numRandSys)}_"
if testSet != orbitType:
    strAdd = strAdd + "Test_" + testSet + "_"
if velNoise != 1e-3:
    strAdd = strAdd + f"VelNoise{velNoise}_"
# if pca_enabled:
#     strAdd = strAdd + f"PCA{pca_n_components}_"
# if useMLP:
#     strAdd = strAdd + "MLP_"

# remove trailing _
if strAdd.endswith("_"):
    strAdd = strAdd[:-1]

print(f"Training with {int(4*train_ratio*numRandSys)} systems")

logLoc = "gmat/data/classification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys) + "/"
logFileLoc = logLoc + str(numMinProp) + "min" + str(numRandSys)+ strAdd +'.log'
_frameTag = frame if frame != "eci" else ("OE" if useOE else "cart")
shap_dir_mamba = logLoc+ f"shap/mamba_{orbitType}_eval_{_frameTag}_"+str(strAdd)
shap_dir_lstm = logLoc+ f"shap/lstm_{orbitType}_eval_{_frameTag}_"+str(strAdd)

if save_to_log:
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    import pandas as pd

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 10000)          # big number to avoid wrapping
    pd.set_option('display.expand_frame_repr', False)

    import warnings

    # Nuke everything (blunt):
    warnings.filterwarnings("ignore")

    # if location does not exist, create it
    import os
    if not os.path.exists("gmat/data/classification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys)):
        os.makedirs("gmat/data/classification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys))
    print("saving log output to {}".format(logFileLoc))

# display the data by calling the displayLogData.py script from its contained folder

class HybridClassifier(nn.Module):
    def __init__(self,config, input_size, hidden_size, num_layers, num_classes):
        super(HybridClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # Bidirectional LSTM
        )
        self.mamba = Mamba(config)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        """""
        x: [batch_size, seq_length, input_size]
        """
        # h0, c0 default to zero if not provided
        out, (h_n, c_n) = self.lstm(x)
        h_n = self.mamba(out) # [batch_size, seq_length, hidden_size]

        # h_n is shape [num_layers, batch_size, hidden_size].
        # We typically take the last layer's hidden state: h_n[-1]
        last_hidden = h_n[:,-1,:]  # [batch_size, hidden_size]
        
        # Pass the last hidden state through a linear layer for classification
        logits = self.fc(last_hidden)  # [batch_size, num_classes]

        return logits


def loadGroundStationDataset(
    frame,                 # "aer" or "radec"
    data_config,
    orbit_type, systems, num_min_prop,
    test_set, test_systems,
    train_ratio, val_ratio, test_ratio,
    batch_size=16,
    normalize=False,
):
    """Load {frame}Array{Chemical,Electric,ImpBurn,NoThrust}.npz from the same per-orbit data
    directory prepareThrustClassificationDatasets uses for statesArray*.npz (see
    generateSpacecraftThrustOptGroundStation.py in GMAT-Thrust-Data), and mirror that function's
    labeling / IC-group train-val-test split / DataLoader construction so every classifier below
    -- all of which only assume `train_data.shape[2]` as the feature width -- works unchanged.

    AER channels:   [Az, El, Range, dAz, dEl, dRange]   (6)
    RA/Dec channels: [RA, Dec, dRA, dDec]                (4)
    """
    prefix = "aerArray" if frame == "aer" else "radecArray"
    noThrustLabel, chemicalLabel, electricLabel, impBurnLabel = 0, 1, 2, 3

    dataLoc      = data_config['classification'] + orbit_type + "/" + str(num_min_prop) + "min-" + str(systems)
    dataLoc_test = data_config['classification'] + test_set   + "/" + str(num_min_prop) + "min-" + str(test_systems)
    print(f"Training data location: {dataLoc}")
    print(f"Test data location: {dataLoc_test}")

    def _load(loc):
        chem = np.load(f"{loc}/{prefix}Chemical.npz")[f"{prefix}Chemical"]
        elec = np.load(f"{loc}/{prefix}Electric.npz")[f"{prefix}Electric"]
        imp  = np.load(f"{loc}/{prefix}ImpBurn.npz")[f"{prefix}ImpBurn"]
        none = np.load(f"{loc}/{prefix}NoThrust.npz")[f"{prefix}NoThrust"]
        return chem, elec, imp, none

    def _labels(chem, elec, imp, none):
        return (
            np.full((chem.shape[0], 1), chemicalLabel),
            np.full((elec.shape[0], 1), electricLabel),
            np.full((imp.shape[0], 1),  impBurnLabel),
            np.full((none.shape[0], 1), noThrustLabel),
        )

    def _ic_split(n_ic):
        n_train_ic = int(np.floor(train_ratio * n_ic))
        n_val_ic   = int(np.floor(val_ratio   * n_ic))
        n_test_ic  = n_ic - n_train_ic - n_val_ic
        assert n_test_ic > 0, "Ratios leave no ICs for test; reduce train/val."
        perm_ic = np.random.permutation(n_ic)
        return perm_ic[:n_train_ic], perm_ic[n_train_ic:n_train_ic + n_val_ic], perm_ic[n_train_ic + n_val_ic:]

    chem, elec, imp, none = _load(dataLoc)
    n_ic = chem.shape[0]
    labelsChem, labelsElec, labelsImp, labelsNone = _labels(chem, elec, imp, none)
    dataset = np.concatenate((chem, elec, imp, none), axis=0)
    dataset_label = np.concatenate((labelsChem, labelsElec, labelsImp, labelsNone), axis=0)

    groups = np.tile(np.arange(n_ic, dtype=np.int64), 4)
    train_ic, val_ic, test_ic = _ic_split(n_ic)
    train_mask = np.isin(groups, train_ic)
    val_mask   = np.isin(groups, val_ic)
    test_mask  = np.isin(groups, test_ic)

    train_data, train_label = dataset[train_mask], dataset_label[train_mask]
    val_data,   val_label   = dataset[val_mask],   dataset_label[val_mask]

    if test_set != orbit_type or test_systems != systems:
        chem_t, elec_t, imp_t, none_t = _load(dataLoc_test)
        n_ic_test = chem_t.shape[0]
        labelsChem_t, labelsElec_t, labelsImp_t, labelsNone_t = _labels(chem_t, elec_t, imp_t, none_t)
        dataset_test = np.concatenate((chem_t, elec_t, imp_t, none_t), axis=0)
        dataset_label_test = np.concatenate((labelsChem_t, labelsElec_t, labelsImp_t, labelsNone_t), axis=0)

        groups_t = np.tile(np.arange(n_ic_test, dtype=np.int64), 4)
        _, _, test_ic_t = _ic_split(n_ic_test)
        test_mask_t = np.isin(groups_t, test_ic_t)
        test_data, test_label = dataset_test[test_mask_t], dataset_label_test[test_mask_t]
    else:
        test_data, test_label = dataset[test_mask], dataset_label[test_mask]

    # ----- optional per-channel z-score normalization, fit on the training split only -----
    if normalize:
        mean = train_data.reshape(-1, train_data.shape[-1]).mean(axis=0)
        std  = train_data.reshape(-1, train_data.shape[-1]).std(axis=0)
        std[std == 0] = 1.0
        train_data = (train_data - mean) / std
        val_data   = (val_data   - mean) / std
        test_data  = (test_data  - mean) / std

    from torch.utils.data import TensorDataset, DataLoader
    train_ds = TensorDataset(torch.from_numpy(train_data).double(), torch.from_numpy(train_label).squeeze(1).long())
    val_ds   = TensorDataset(torch.from_numpy(val_data).double(),   torch.from_numpy(val_label).squeeze(1).long())
    test_ds  = TensorDataset(torch.from_numpy(test_data).double(),  torch.from_numpy(test_label).squeeze(1).long())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader, train_data, train_label, val_data, val_label, test_data, test_label


def main():
    import yaml
    with open("data.yaml", 'r') as f:
        dataConfig = yaml.safe_load(f)
    dataLoc = dataConfig['classification'] + orbitType +"/" + str(numMinProp) + "min-" + str(numRandSys)
    print(f"Processing datasets for {orbitType} with {numMinProp} minutes and {numRandSys} random systems.")
    # dataLoc = "c/Users/hu650776/GMAT-Thrust-Data/data/classification/data/classification/"+ orbitType +"/" + str(numMinProp) + "min-" + str(numRandSys)


    device = getDevice()

    batchSize = 16
    problemDim = 6

    # create a dictionary to hold yaml config values
    # TODO: change to pyyaml reading from a file 
    yaml_config = {}

    yaml_config['useOE'] = useOE
    yaml_config['useNorm'] = useNorm
    yaml_config['useNoise'] = useNoise
    yaml_config['useEnergy'] = useEnergy

    yaml_config['prop_time'] = numMinProp

    yaml_config['orbit'] = orbitType
    yaml_config['systems'] = numRandSys

    yaml_config['test_dataset'] = testSet
    yaml_config['test_systems'] = testSys

    from qutils.ml.classifer import prepareThrustClassificationDatasets

    if train_ratio == 0.7:
        val_ratio = 0.15
        test_ratio = 0.15
    else:
        val_ratio = train_ratio  
        test_ratio = (1.0 - train_ratio - val_ratio) # not used in network training, only for splitting the data and final evaluation

    if frame == "eci":
        train_loader, val_loader, test_loader, train_data,train_label,val_data,val_label,test_data,test_label = prepareThrustClassificationDatasets(yaml_config,dataConfig,output_np=True,vel_noise_std=velNoise,pos_noise_std=1e3*velNoise,train_ratio=train_ratio,test_ratio=test_ratio,val_ratio=val_ratio,pca_enabled=pca_enabled,pca_mode="hankel",hankel_pool="mean")
    else:
        if pca_enabled:
            print(f"[warning] --pca is only wired up for --frame eci (via prepareThrustClassificationDatasets); ignoring for --frame {frame}.")
        train_loader, val_loader, test_loader, train_data,train_label,val_data,val_label,test_data,test_label = loadGroundStationDataset(
            frame=frame,
            data_config=dataConfig,
            orbit_type=orbitType, systems=numRandSys, num_min_prop=numMinProp,
            test_set=testSet, test_systems=testSys,
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
            batch_size=16,
            normalize=useNorm,
        )

    # Hyperparameters
    input_size = train_data.shape[2] 
    hidden_factor = 8  # hidden size is a multiple of input size
    hidden_size = int(input_size * hidden_factor) # must be multiple of train dim
    num_layers = 1
    num_classes = 4  # e.g., multiclass classification
    learning_rate = 1e-3
    num_epochs = 100

    if useOnePass:
        num_epochs = 1

    criterion = torch.nn.CrossEntropyLoss()

    config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=4,classifer=True)
    model_mamba = MambaClassifier(config,input_size, hidden_size, num_layers, num_classes).to(device).double()
    optimizer_mamba = torch.optim.Adam(model_mamba.parameters(), lr=learning_rate)

    schedulerPatience = 5

    scheduler_mamba = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_mamba,
        mode='min',             # or 'max' for accuracy
        factor=0.5,             # shrink LR by 50%
        patience=schedulerPatience             # wait for 3 epochs of no improvement
    )

    classlabels = ['No Thrust','Chemical','Electric','Impulsive']

    if useHybrid:
        config_hybrid = MambaConfig(d_model=hidden_size * 2,n_layers = 1,expand_factor=1,d_state=32,d_conv=16,classifer=True)

        model_hybrid = HybridClassifier(config_hybrid,input_size,hidden_size,num_layers,num_classes).to(device).double()
        optimizer_hybrid = torch.optim.Adam(model_hybrid.parameters(), lr=learning_rate)
        scheduler_hybrid = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_hybrid,
            mode='min',             # or 'max' for accuracy
            factor=0.5,             # shrink LR by 50%
            patience=schedulerPatience
        )

        print('\nEntering Hybrid Training Loop')
        trainClassifier(model_hybrid,optimizer_hybrid,scheduler_hybrid,[train_loader,test_loader,val_loader],criterion,num_epochs,device,classLabels=classlabels)
        printModelParmSize(model_hybrid)

        if testSet != orbitType:
            validateMultiClassClassifier(model_hybrid,test_loader,criterion,num_classes,device,classlabels,printReport=True)
        else:
            validateMultiClassClassifier(model_hybrid,val_loader,criterion,num_classes,device,classlabels,printReport=True)

    # shared flattened-feature setup for all GBDT / classic-ML bake-off classifiers
    if use_classic or use_xgboost or use_catboost or use_random_forest or use_extra_trees:
        X_train = train_data.reshape(train_data.shape[0], -1).astype(np.float32)    # (number of systems to train on, network features * length of time series)
        y_train = train_label.reshape(-1).astype(np.int32)             # (number of systems to train on,)
        _eval_loader_classic = test_loader if testSet != orbitType else val_loader

    if use_classic:
        from lightgbm import LGBMClassifier
        from qutils.ml.classic.classifier import printDTModelSize, validate_lightgbm

        print("\nEntering Decision Trees (LightGBM) Training Loop")
        classicModel = LGBMClassifier(objective="multiclass",num_classes=num_classes,n_estimators=30,max_depth=-1,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,verbosity=-1)   # or 'verbose' for older builds)
        DTTimer = timer()
        classicModel.fit(X_train, y_train)
        DTTimer.toc()
        printDTModelSize(classicModel)
        print("\nDecision Trees (LightGBM) Validation")
        DTTimerInference = timer()
        validate_lightgbm(classicModel, _eval_loader_classic, num_classes, classlabels=classlabels, print_report=True)
        DTTimerInference.tocStr("Decision Trees (LightGBM) Inference Time")

    if use_xgboost:
        from xgboost import XGBClassifier
        from qutils.ml.classic.classifier import printClassicModelSize, validate_classic_classifier

        print("\nEntering XGBoost Training Loop")
        xgbModel = XGBClassifier(objective="multi:softprob", num_class=num_classes, n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", n_jobs=-1)
        xgbTimer = timer()
        xgbModel.fit(X_train, y_train)
        xgbTimer.toc()
        printClassicModelSize(xgbModel)
        print("\nXGBoost Validation")
        xgbTimerInference = timer()
        validate_classic_classifier(xgbModel, _eval_loader_classic, num_classes, classlabels=classlabels, print_report=True)
        xgbTimerInference.tocStr("XGBoost Inference Time")

    if use_catboost:
        from catboost import CatBoostClassifier
        from qutils.ml.classic.classifier import printClassicModelSize, validate_classic_classifier

        print("\nEntering CatBoost Training Loop")
        catboostModel = CatBoostClassifier(loss_function="MultiClass", classes_count=num_classes, iterations=200, depth=6, learning_rate=0.05, bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.8, verbose=False, allow_writing_files=False)
        catboostTimer = timer()
        catboostModel.fit(X_train, y_train)
        catboostTimer.toc()
        printClassicModelSize(catboostModel)
        print("\nCatBoost Validation")
        catboostTimerInference = timer()
        validate_classic_classifier(catboostModel, _eval_loader_classic, num_classes, classlabels=classlabels, print_report=True)
        catboostTimerInference.tocStr("CatBoost Inference Time")

    if use_random_forest:
        from sklearn.ensemble import RandomForestClassifier
        from qutils.ml.classic.classifier import printClassicModelSize, validate_classic_classifier

        print("\nEntering Random Forest Training Loop")
        rfModel = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1)
        rfTimer = timer()
        rfModel.fit(X_train, y_train)
        rfTimer.toc()
        printClassicModelSize(rfModel)
        print("\nRandom Forest Validation")
        rfTimerInference = timer()
        validate_classic_classifier(rfModel, _eval_loader_classic, num_classes, classlabels=classlabels, print_report=True)
        rfTimerInference.tocStr("Random Forest Inference Time")

    if use_extra_trees:
        from sklearn.ensemble import ExtraTreesClassifier
        from qutils.ml.classic.classifier import printClassicModelSize, validate_classic_classifier

        print("\nEntering Extra Trees Training Loop")
        etModel = ExtraTreesClassifier(n_estimators=300, max_depth=None, n_jobs=-1)
        etTimer = timer()
        etModel.fit(X_train, y_train)
        etTimer.toc()
        printClassicModelSize(etModel)
        print("\nExtra Trees Validation")
        etTimerInference = timer()
        validate_classic_classifier(etModel, _eval_loader_classic, num_classes, classlabels=classlabels, print_report=True)
        etTimerInference.tocStr("Extra Trees Inference Time")

    if use_nearestNeighbor:
        from qutils.ml.classic.classifier import validate_1NN, print1_NNModelSize
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

        print("\nEntering Nearest Neighbor Training Loop")
        # [N,T,C] -> [N,C,T]
        train_data_NN = np.transpose(train_data, (0, 2, 1))

        # train_data_NN = train_data_z_normalize(train_data_NN)  # Z-normalize along time axis

        clf = KNeighborsTimeSeriesClassifier(
            n_neighbors=1,
            distance="dtw",
            distance_params={"sakoe_chiba_radius": 10}
    )
        dtw = timer()
        clf.fit(train_data_NN, train_label)
        dtw.toc()
        print1_NNModelSize(clf)
        print("\n1-NN Validation")
        dtwInference = timer()
        if testSet != orbitType:
            validate_1NN(clf, test_loader, num_classes, classlabels=classlabels)
        else:
            validate_1NN(clf, val_loader, num_classes, classlabels=classlabels)
        dtwInference.tocStr("1-NN Inference Time")

    if use_minirocket:
        def printMiniROCKETSize(model):\
            # reports number of kernels as number of parameters 
            import pickle
            size_bytes = len(pickle.dumps(model))
            num_kernels = model.num_kernels_
            print("\n==========================================================================================")
            print(f"Total parameters: {num_kernels}")
            print(f"Total memory (bytes): {size_bytes}")
            print(f"Total memory (MB): {size_bytes / (1024 ** 2):.4f}")
            print("==========================================================================================")

        def validate_minirocket(clf, val_loader, num_classes, classlabels=None):
            """Evaluate a MiniRocket (RocketClassifier) on a PyTorch DataLoader."""
            X_val_list, y_val_list = [], []
            for seq, lab in val_loader:
                X_val_list.append(seq.cpu().numpy())
                y_val_list.append(lab.cpu().numpy())

            X_val_np = np.concatenate(X_val_list, axis=0)
            y_true = np.concatenate(y_val_list)

            # [N,T,C] -> [N,C,T] for sktime
            X_val_np = np.transpose(X_val_np, (0, 2, 1))

            y_pred = clf.predict(X_val_np)

            correct = (y_pred == y_true).sum()
            accuracy = 100.0 * correct / len(y_true)
            print(f"Validation Loss: NaN, Validation Accuracy: {accuracy:.2f}%\n")

            class_corr = np.zeros(num_classes, dtype=int)
            class_tot = np.zeros(num_classes, dtype=int)
            for yt, yp in zip(y_true, y_pred):
                class_tot[yt] += 1
                if yt == yp:
                    class_corr[yt] += 1

            print("Per-Class Validation Accuracy:")
            for i in range(num_classes):
                label = classlabels[i] if classlabels else f"Class {i}"
                if class_tot[i]:
                    print(f"  {label}: {100.0 * class_corr[i] / class_tot[i]:.2f}% ({class_corr[i]}/{class_tot[i]})")
                else:
                    print(f"  {label}: No samples")

            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                        target_names=(classlabels if classlabels else None),
                                        digits=4, zero_division=0))
            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            print("\nConfusion Matrix (rows = true, cols = predicted):")
            print(pd.DataFrame(cm,
                                index=[f"T_{cls}" for cls in (classlabels if classlabels else range(num_classes))],
                                columns=[f"P_{cls}" for cls in (classlabels if classlabels else range(num_classes))]))

        from sktime.classification.kernel_based import RocketClassifier

        print("\nEntering MiniRocket Training Loop")
        # [N,T,C] -> [N,C,T]
        train_data_MR = np.transpose(train_data, (0, 2, 1))

        clf_mr = RocketClassifier(num_kernels=10000, rocket_transform='minirocket', n_jobs=-1)
        mrTimer = timer()
        clf_mr.fit(train_data_MR, train_label)
        mrTimer.toc()
        printMiniROCKETSize(clf_mr)

        print("\nMiniRocket Validation")
        mrInference = timer()
        _eval_loader_MR = test_loader if testSet != orbitType else val_loader
        validate_minirocket(clf_mr, _eval_loader_MR, num_classes, classlabels=classlabels)
        mrInference.tocStr("MiniRocket Inference Time")

    if use_lstm:
        model_LSTM = LSTMClassifier(input_size, hidden_size, num_layers, num_classes,SA=True).to(device).double()
        optimizer_LSTM = torch.optim.Adam(model_LSTM.parameters(), lr=learning_rate)
        scheduler_LSTM = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_LSTM,
            mode='min',             # or 'max' for accuracy
            factor=0.5,             # shrink LR by 50%
            patience=schedulerPatience
        )

        print('\nEntering LSTM Training Loop')
        trainClassifier(model_LSTM,optimizer_LSTM,scheduler_LSTM,[train_loader,test_loader,val_loader],criterion,num_epochs,device,classLabels=classlabels)
        printModelParmSize(model_LSTM)
        print("\nLSTM Validation")
        LSTMInference = timer()
        _eval_loader = test_loader if (testSet != orbitType) else val_loader
        validateMultiClassClassifier(model_LSTM,_eval_loader,criterion,num_classes,device,classlabels,printReport=True)
        LSTMInference.tocStr("LSTM Inference Time")
        if frame == "aer":
            feat_names = ['Az','El','Range','dAz','dEl','dRange']
        elif frame == "radec":
            feat_names = ['RA','Dec','dRA','dDec']
        elif useOE:
            feat_names = ['a','ecc','inc','RAAN','argp','nu']
        else:
            feat_names = ['x','y','z','vx','vy','vz']
        if run_shap:
            _ = run_shap_analysis(
                model=model_LSTM,
                train_loader=train_loader,
                eval_loader=_eval_loader,
                device=device,                        # e.g., "cuda" or "cpu"
                classlabels=classlabels,
                feature_names=feat_names,  # or None
                out_dir=shap_dir_lstm,
                method="gradshap",
                baseline_nsamples=32,
                gs_samples=8,
                n_eval=None,
                internal_batch_size=32,
                use_cpu=False,
                group_by="true"     # <<— important
            )
            print(f"[SHAP] CSVs written to: {shap_dir_lstm}")

    print('\nEntering Mamba Training Loop')
    trainClassifier(model_mamba,optimizer_mamba,scheduler_mamba,[train_loader,test_loader,val_loader],criterion,num_epochs,device,classLabels=classlabels)
    printModelParmSize(model_mamba)

    print("\nMamba Validation")
    mambaInference = timer()
    _eval_loader = test_loader if (testSet != orbitType) else val_loader
    validateMultiClassClassifier(model_mamba, _eval_loader, criterion, num_classes, device, classlabels, printReport=True)
    mambaInference.tocStr("Mamba Inference Time")

    if use_transformer:
        print("\nEntering Transformer Training Loop")
        model_transformer = TransformerClassifier(input_size, hidden_size, num_layers, num_classes).to(device).double()
        optimizer_transformer = torch.optim.Adam(model_transformer.parameters(), lr=learning_rate)
        scheduler_transformer = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_transformer,
            mode='min',             # or 'max' for accuracy
            factor=0.5,             # shrink LR by 50%
            patience=schedulerPatience
        )
        trainClassifier(model_transformer,optimizer_transformer,scheduler_transformer,[train_loader,test_loader,val_loader],criterion,num_epochs,device,classLabels=classlabels)
        printModelParmSize(model_transformer)

        print("\nTransformer Validation")
        transformerInference = timer()
        _eval_loader = test_loader if (testSet != orbitType) else val_loader
        validateMultiClassClassifier(model_transformer, _eval_loader, criterion, num_classes, device, classlabels, printReport=True)
        transformerInference.tocStr("Transformer Inference Time")

    if use_cnn:
        print("\nEntering 1D-CNN (InceptionTime) Training Loop")
        model_cnn = InceptionTimeClassifier(input_size, num_classes).to(device).double()
        optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=learning_rate)
        scheduler_cnn = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_cnn,
            mode='min',             # or 'max' for accuracy
            factor=0.5,             # shrink LR by 50%
            patience=schedulerPatience
        )
        trainClassifier(model_cnn,optimizer_cnn,scheduler_cnn,[train_loader,test_loader,val_loader],criterion,num_epochs,device,classLabels=classlabels)
        printModelParmSize(model_cnn)

        print("\n1D-CNN (InceptionTime) Validation")
        cnnInference = timer()
        _eval_loader = test_loader if (testSet != orbitType) else val_loader
        validateMultiClassClassifier(model_cnn, _eval_loader, criterion, num_classes, device, classlabels, printReport=True)
        cnnInference.tocStr("1D-CNN (InceptionTime) Inference Time")

    if frame == "aer":
        feat_names = ['Az','El','Range','dAz','dEl','dRange']
    elif frame == "radec":
        feat_names = ['RA','Dec','dRA','dDec']
    elif useOE:
        feat_names = ['a','ecc','inc','RAAN','argp','nu']
    else:
        feat_names = ['x','y','z','vx','vy','vz']
    if run_shap:
        _ = run_shap_analysis(
            model=model_mamba,
            train_loader=train_loader,
            eval_loader=_eval_loader,
            device=device,                        # e.g., "cuda" or "cpu"
            classlabels=classlabels,
            feature_names=feat_names,  # or None
            out_dir=shap_dir_mamba,
            method="gradshap",
            baseline_nsamples=32,
            gs_samples=8,
            n_eval=None,
            internal_batch_size=32,
            use_cpu=False,
            group_by="true"     # <<— important
    )
        print(f"[SHAP] CSVs written to: {shap_dir_mamba}")

    if saveNets:
        import os
        if not os.path.exists("gmat/data/classification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys)):
            os.makedirs("gmat/data/classification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys))
        print(f"Saving networks to gmat/data/classification/{orbitType}/{numMinProp}min-{numRandSys}/")
        if use_lstm:
            torch.save(model_LSTM.state_dict(), f"{logLoc}lstm_"+ orbitType +"_"+strAdd+".pt")
        if useHybrid:
            torch.save(model_hybrid.state_dict(), f"{logLoc}hybrid_"+ orbitType +"_"+strAdd+".pt")
        torch.save(model_mamba.state_dict(), f"{logLoc}mamba_"+ orbitType +"_"+strAdd+".pt")

    if find_SW:
        magnitude, index = findMambaSuperActivation(model_mamba,torch.tensor(test_data).to(device))
        # super activation returns the entire mamba network parameters, but the classifier does not use the out_proj layer
        # so we drop it
        magnitude = magnitude[:-1]
        index = index[:-1]
        # also drop the x_proj layer, no longer needed as well
        magnitude.pop(2)
        index.pop(2)

        normedMagsMRP = np.zeros((len(magnitude),))
        for i in range(len(magnitude)):
            normedMagsMRP[i] = magnitude[i].norm().detach().cpu()

        printoutMaxLayerWeight(model_mamba)
        getSuperWeight(model_mamba)
        plotSuperWeight(model_mamba)
        plotSuperActivation(magnitude, index,printOutValues=True,mambaLayerAttributes = ["in_proj","conv1d","dt_proj"])
        plt.title("Mamba Classifier Super Activations")


    # needs to be last, PCA is not used for the other models and we want to keep the same data for all non-MLP models if PCA is enabled, so we do it after training and evaluating those models
    if useMLP is True and frame != "eci":
        print(f"[warning] --mlp (PCA+Hankel-pooled features) is currently only implemented for "
              f"--frame eci; skipping for --frame {frame}.")
    if useMLP is True and frame == "eci":
        train_loader, val_loader, test_loader, train_data,train_label,val_data,val_label,test_data,test_label, pca_state = prepareThrustClassificationDatasets(
        yaml_config,
        dataConfig,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        pos_noise_std=1e3*velNoise, vel_noise_std=velNoise,
        batch_size=128,                     
        output_np=True,
        pca_enabled=True,
        pca_mode="hankel",
        pca_n_components=0.95,
        pca_whiten=False,
        pca_standardize=True,
        hankel_L=1,
        hankel_step=1,
        hankel_pool="mean",
        return_pca=True,
        supress_print=True
        )

        # Infer input dim d from one batch
        xb0, yb0 = next(iter(train_loader))
        d_in = xb0.squeeze(1).shape[-1]
        num_classes = 4

        # === Model, optimizer, scheduler ===
        model_mlp = MLP(d_in=d_in, n_classes=num_classes, width=64, depth=1, p_drop=0.1).to(device).double()
        from torch.optim import AdamW
        optimizer = AdamW(model_mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        # === Train ===
        print('\nEntering MLP Training Loop')
        trainMLP(model_mlp, train_loader, val_loader, optimizer, scheduler,device, class_weights=None, num_epochs=num_epochs)
        printModelParmSize(model_mlp)

        # === Validation ===
        print("\nMLP Validation")
        MLPInference = timer()
        _eval_loader = test_loader if (testSet != orbitType) else val_loader
        eval_epoch(model_mlp, _eval_loader, device)
        MLPInference.tocStr("MLP Inference Time")


        model_mlp.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for xb, yb in _eval_loader:
                logits = model_mlp(xb.squeeze(1).to(device))
                pred = logits.argmax(dim=1).cpu().numpy()
                all_p.append(pred)
                all_y.append(yb.view(-1).cpu().numpy())
        all_p = np.concatenate(all_p); all_y = np.concatenate(all_y)


        print("\nClassification Report:")
        print(
            classification_report(
                all_y,
                all_p,
                labels=list(range(num_classes)),
                digits=4,
                zero_division=0,
            )
        )
                # Confusion-matrix -----------------------------------------------------
        cm = confusion_matrix(all_y, all_p, labels=list(range(num_classes)))
        print("\nConfusion Matrix (rows = true, cols = predicted):")
        print(pd.DataFrame(cm,
                            index=[f"T_{cls}" for cls in (classlabels if classlabels else range(num_classes))],
                            columns=[f"P_{cls}" for cls in (classlabels if classlabels else range(num_classes))]))

        


# # example onnx export
# # # generate example inputs for ONNX export
# example_inputs = torch.randn(1, numMinProp, input_size).to(device).double()
# # export the model to ONNX format
# # Note: `dynamo=True` is used to enable PyTorch's dynamo for better performance and compatibility.
# onnx_path = f"{dataLoc}/mambaTimeSeriesClassificationGMATThrusts.onnx"
# onnx_program = torch.onnx.export(model_mamba, example_inputs,onnx_path)
# print(f"ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    if save_to_log:
        log = logFileLoc  # path
        with open(log, 'w', buffering=1, encoding='utf-8') as f, \
            redirect_stdout(f), redirect_stderr(f):
            main()
    else:
        main()
    if run_shap: 
        from qutils.ml.shap import plot_global_feature_importance, plot_global_time_importance, plot_all_per_class_heatmaps,plot_feature_time_importance_heatmap
        plot_global_feature_importance(shap_dir_mamba, topk=20, save=True,as_percent=True)
        plot_global_time_importance(shap_dir_mamba, save=True)
        # One heatmap per class CSV; lock_vmax=True to use the same color scale across classes
        plot_all_per_class_heatmaps(shap_dir_mamba, topk_features=None, lock_vmax=True)
        plot_feature_time_importance_heatmap(shap_dir_mamba, topk=None, save=True)

        plot_global_feature_importance(shap_dir_lstm, topk=20, save=True,as_percent=True)
        plot_global_time_importance(shap_dir_lstm, save=True)
        # One heatmap per class CSV; lock_vmax=True to use the same color scale across classes
        plot_all_per_class_heatmaps(shap_dir_lstm, topk_features=None, lock_vmax=True)
        plot_feature_time_importance_heatmap(shap_dir_lstm, topk=None, save=True)

    if not save_to_log:
        plt.show()