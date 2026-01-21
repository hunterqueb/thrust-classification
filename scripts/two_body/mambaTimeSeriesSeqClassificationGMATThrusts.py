# parse at the beginning before long imports
# script usage

# call the script from the main folder directory, adding --save saves the output to a log file in the location of the datasets
# $ python scripts/classification/mambaTimeSeriesClassificationGMATThrusts.py \
# --systems 10000 --propMin 5 --OE --norm --orbit vleo 

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
velNoise = args.velNoise
train_ratio = args.train_ratio


import torch
import numpy as np
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from qutils.tictoc import timer
from qutils.ml.utils import getDevice, printModelParmSize
from qutils.ml.classifer import trainClassifier, LSTMClassifier, validateMultiClassClassifier
from qutils.ml.mamba import Mamba, MambaConfig, MambaClassifier

dt = 60
t = np.linspace(0,dt*numMinProp)
seq_len = len(t)


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
    strAdd = strAdd + "Test_" + testSet
if velNoise != 1e-3:
    strAdd = strAdd + f"VelNoise{velNoise}_"

# remove trailing _
strAdd = strAdd[:-1]

print(f"Training with {int(4*train_ratio*numRandSys)} systems")

logLoc = "gmat/data/seqClassification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys) + "/"
logFileLoc = logLoc + str(numMinProp) + "min" + str(numRandSys)+ strAdd +'.log'
shap_dir_mamba = logLoc+ f"shap/mamba_{orbitType}_eval_{'OE' if useOE else 'cart'}_"+str(strAdd)
shap_dir_lstm = logLoc+ f"shap/lstm_{orbitType}_eval_{'OE' if useOE else 'cart'}_"+str(strAdd)

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
    if not os.path.exists("gmat/data/seqClassification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys)):
        os.makedirs("gmat/data/seqClassification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys))
    print("saving log output to {}".format(logFileLoc))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def _infer_class_weights(loader, num_classes: int, pad_idx: int = -100, dtype=torch.float32, device="cpu"):
    counts = torch.zeros(num_classes, dtype=torch.long)
    with torch.no_grad():
        for _, y in loader:
            y = y.reshape(-1).long()
            if pad_idx is not None:
                y = y[y != pad_idx]
            counts += torch.bincount(y, minlength=num_classes)
    w = counts.sum() / torch.clamp(counts.to(dtype), min=1.0)   # inverse frequency
    w = w / w.mean()
    return w.to(device=device, dtype=dtype)

def train_model(model, train_loader, val_loader, num_epochs=10, num_classes=3,
                pad_idx: int = -100, class_weights: torch.Tensor | None = None):

    device = next(model.parameters()).device
    param_dtype = next(model.parameters()).dtype  # ensures Float32 vs Float64 consistency

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ----- class weights with correct dtype -----
    if class_weights is None:
        class_weights = _infer_class_weights(
            train_loader, num_classes, pad_idx, dtype=param_dtype, device=device
        )
    else:
        class_weights = torch.as_tensor(class_weights, device=device, dtype=param_dtype)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=pad_idx)

    schedulerPatience = 3
    best_loss = float('inf')
    ESpatience = schedulerPatience * 2
    counter = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=schedulerPatience
    )

    last_targets = None
    last_preds = None

    for epoch in range(num_epochs):
        # -------- TRAIN --------
        model.train()
        total_loss = 0.0

        for sequences, labels in train_loader:
            sequences = sequences.to(device, non_blocking=True)
            if sequences.dtype != param_dtype:
                sequences = sequences.to(param_dtype)  # align to model dtype

            labels = labels.to(device, non_blocking=True).long()  # [B, T]

            logits = model(sequences)  # [B, T, C]
            if logits.dtype != param_dtype:
                logits = logits.to(param_dtype)

            B, T, C = logits.shape
            loss = criterion(logits.view(B*T, C), labels.view(B*T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f}")

        # -------- VALIDATION (metrics) --------
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device, non_blocking=True)
                if sequences.dtype != param_dtype:
                    sequences = sequences.to(param_dtype)
                labels = labels.to(device, non_blocking=True).long()

                logits = model(sequences)          # [B, T, C]
                preds = logits.argmax(dim=-1)      # [B, T]

                if pad_idx is not None:
                    mask = (labels != pad_idx)
                    all_preds.append(preds[mask].detach().cpu())
                    all_targets.append(labels[mask].detach().cpu())
                else:
                    all_preds.append(preds.view(-1).detach().cpu())
                    all_targets.append(labels.view(-1).detach().cpu())

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()

        p_ev, r_ev, f_ev, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1, 2], average='macro', zero_division=0
        )
        print(f"Val Event P(macro 1&2): {p_ev:.4f} | R: {r_ev:.4f} | F1: {f_ev:.4f}")

        p_pc, r_pc, f_pc, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0,1,2], average=None, zero_division=0
        )
        print(f"Per-class P: {p_pc}  R: {r_pc}  F1: {f_pc}")

        # -------- VALIDATION (loss for LR sched + ES) --------
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device, non_blocking=True)
                if sequences.dtype != param_dtype:
                    sequences = sequences.to(param_dtype)
                labels = labels.to(device, non_blocking=True).long()
                logits = model(sequences)
                if logits.dtype != param_dtype:
                    logits = logits.to(param_dtype)
                B, T, C = logits.shape
                loss = criterion(logits.view(B*T, C), labels.view(B*T))
                val_loss += loss.item()
        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)
        last_targets, last_preds = y_true, y_pred

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= ESpatience:
                print("Early stopping")
                break

    cm = confusion_matrix(last_targets, last_preds, labels=[0,1,2])
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

# Evaluation
def evaluate_model(model, loader, sample_batch_idx=0, sample_in_batch_idx=0,modelString="Mamba"):
    """
    Visualizes per-timestep classification for a sample from a DataLoader.
    
    Parameters:
        model: trained model
        loader: DataLoader (e.g., val_loader)
        sample_batch_idx: index of the batch to sample from
        sample_in_batch_idx: index within the selected batch
    """
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            if batch_idx == sample_batch_idx:
                x_sample = batch_x[sample_in_batch_idx].unsqueeze(0).to(device)  # [1, T, D]
                y_sample = batch_y[sample_in_batch_idx].to(device)               # [T]
                global_index = batch_idx * loader.batch_size + sample_in_batch_idx
                break
        else:
            raise ValueError("Sample batch index out of range.")

        logits = model(x_sample).squeeze(0)           # [T]
        probs = torch.sigmoid(logits)                 # [T]
        predictions = (probs > 0.5).long()            # [T]


        x_sample = x_sample.cpu().numpy().squeeze(0)  # [T, D]
        y_sample = y_sample.cpu().numpy()            # [T]
        predictions = predictions.cpu().numpy()       # [T]

def plotConfusionMatrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Validation Confusion Matrix', pad=20)

    # Tick labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Force (0)', 'Force (1)'])
    ax.set_yticklabels(['No Force (0)', 'Force (1)'])
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    # Annotate cells with values
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=12)

    quad_labels = np.array([["TN", "FP"], ["FN", "TP"]])
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.25, quad_labels[i, j], ha='center', va='top',
                    fontsize=10, color='gray')

    plt.tight_layout()  


from qutils.ml.classifer import apply_noise
def prepareThrustClassificationDatasets(yaml_config,data_config,train_ratio=0.7,val_ratio=0.15,test_ratio=0.15,pos_noise_std=1e-3,vel_noise_std=1e-3,batch_size=16,output_np=False):
    '''
    assumes 2 classes: chemical, electric
    assumes equal number of ICs for each class
    '''
    useOE = yaml_config['useOE']
    useNorm = yaml_config['useNorm']
    useNoise = yaml_config['useNoise']
    useEnergy = yaml_config['useEnergy']

    numMinProp = yaml_config['prop_time']

    train_set = yaml_config['orbit']
    systems = yaml_config['systems']

    test_set = yaml_config['test_dataset']
    test_systems = yaml_config['test_systems']

    dataLoc = data_config['classification'] + train_set +"/" + str(numMinProp) + "min-" + str(systems)
    dataLoc_test = data_config['classification'] + test_set +"/" + str(numMinProp) + "min-" + str(test_systems)

    print(f"Training data location: {dataLoc}")
    print(f"Test data location: {dataLoc_test}")

    a = np.load(f"{dataLoc}/statesArrayChemical.npz")
    statesArrayChemical = a['statesArrayChemical']
    labelsChemical = a['thrustingTime'] * 1
    a = np.load(f"{dataLoc}/statesArrayElectric.npz")
    statesArrayElectric = a['statesArrayElectric']
    labelsElectric = a['thrustingTime'] * 2

    n_ic = statesArrayChemical.shape[0]

    if useNoise:
        statesArrayChemical = apply_noise(statesArrayChemical, pos_noise_std, vel_noise_std)
        statesArrayElectric = apply_noise(statesArrayElectric, pos_noise_std, vel_noise_std)
    if useOE:
        # convert to OE 
        from qutils.orbital import ECI2OE
        OEArrayChemical = np.zeros((systems,numMinProp,7))
        OEArrayElectric = np.zeros((systems,numMinProp,7))

        for i in range(systems):
            for j in range(numMinProp):
                OEArrayChemical[i,j,:] = ECI2OE(statesArrayChemical[i,j,0:3],statesArrayChemical[i,j,3:6])
                OEArrayElectric[i,j,:] = ECI2OE(statesArrayElectric[i,j,0:3],statesArrayElectric[i,j,3:6])
        if useNorm:
            R = 6378.1363
            OEArrayChemical[:,:,0] = OEArrayChemical[:,:,0] / R
            OEArrayElectric[:,:,0] = OEArrayElectric[:,:,0] / R

        statesArrayChemical = OEArrayChemical[:,:,0:6]
        statesArrayElectric = OEArrayElectric[:,:,0:6]

    if useNorm and not useOE:
        from qutils.orbital import dim2NonDim6
        for i in range(n_ic):
            statesArrayChemical[i,:,:] = dim2NonDim6(statesArrayChemical[i,:,:])
            statesArrayElectric[i,:,:] = dim2NonDim6(statesArrayElectric[i,:,:])

    n_ic = statesArrayChemical.shape[0]

    # Create labels for each dataset
    # Combine datasets and labels
    dataset = np.concatenate((statesArrayChemical, statesArrayElectric), axis=0)

    if useEnergy:
        from qutils.orbital import orbitalEnergy
        energyChemical = np.zeros((n_ic,statesArrayChemical.shape[1],1))
        energyElectric= np.zeros((n_ic,statesArrayChemical.shape[1],1))
        for i in range(n_ic):
            energyChemical[i,:,0] = orbitalEnergy(statesArrayChemical[i,:,:])
            energyElectric[i,:,0] = orbitalEnergy(statesArrayElectric[i,:,:])
        if useNorm:
            normingEnergy = energyChemical[0,0,0]
            energyChemical[:,:,0] = energyChemical[:,:,0] / normingEnergy
            energyElectric[:,:,0] = energyElectric[:,:,0] / normingEnergy

        dataset = np.concatenate((energyChemical, energyElectric), axis=0)
    if useEnergy and useOE:
        combinedChemical = np.concatenate((OEArrayChemical,energyChemical),axis=2) 
        combinedElectric = np.concatenate((OEArrayElectric,energyElectric),axis=2) 
        dataset = np.concatenate((combinedChemical, combinedElectric), axis=0)

    dataset_label = np.concatenate((labelsChemical, labelsElectric), axis=0)

    # shuffle the dataset completely
    groups = np.tile(np.arange(n_ic, dtype=np.int64), 2)   # len == 40000

    # Ratios (must satisfy train+val <= 1.0; test gets the remainder)
    # example:
    # train_ratio, val_ratio = 0.7, 0.15
    n_train_ic = int(np.floor(train_ratio * n_ic))
    n_val_ic   = int(np.floor(val_ratio   * n_ic))
    n_test_ic  = n_ic - n_train_ic - n_val_ic
    assert n_test_ic > 0, "Ratios leave no ICs for test; reduce train/val."

    # Shuffle ICs and partition
    perm_ic = np.random.permutation(n_ic)
    train_ic = perm_ic[:n_train_ic]
    val_ic   = perm_ic[n_train_ic:n_train_ic + n_val_ic]
    test_ic  = perm_ic[n_train_ic + n_val_ic:]

    # Masks select ALL thrust variants for each IC
    train_mask = np.isin(groups, train_ic)
    val_mask   = np.isin(groups, val_ic)
    test_mask  = np.isin(groups, test_ic)

    # Apply masks
    train_data,  train_label  = dataset[train_mask], dataset_label[train_mask]
    val_data,    val_label    = dataset[val_mask],   dataset_label[val_mask]

    if test_set != train_set or test_systems != systems:

        a = np.load(f"{dataLoc_test}/statesArrayChemical.npz")
        statesArrayChemical = a['statesArrayChemical']
        labelsChemical = a['thrustingTime'] * 1
        a = np.load(f"{dataLoc_test}/statesArrayElectric.npz")
        statesArrayElectric = a['statesArrayElectric']
        labelsElectric = a['thrustingTime'] * 2

        n_ic = statesArrayChemical.shape[0]

        if useNoise:
            statesArrayChemical = apply_noise(statesArrayChemical, pos_noise_std, vel_noise_std)
            statesArrayElectric = apply_noise(statesArrayElectric, pos_noise_std, vel_noise_std)
        if useOE:
            # convert to OE 
            from qutils.orbital import ECI2OE
            OEArrayChemical = np.zeros((test_systems,numMinProp,7))
            OEArrayElectric = np.zeros((test_systems,numMinProp,7))

            for i in range(test_systems):
                for j in range(numMinProp):
                    OEArrayChemical[i,j,:] = ECI2OE(statesArrayChemical[i,j,0:3],statesArrayChemical[i,j,3:6])
                    OEArrayElectric[i,j,:] = ECI2OE(statesArrayElectric[i,j,0:3],statesArrayElectric[i,j,3:6])
            if useNorm:
                R = 6378.1363
                OEArrayChemical[:,:,0] = OEArrayChemical[:,:,0] / R
                OEArrayElectric[:,:,0] = OEArrayElectric[:,:,0] / R

            statesArrayChemical = OEArrayChemical[:,:,0:6]
            statesArrayElectric = OEArrayElectric[:,:,0:6]

        if useNorm and not useOE:
            from qutils.orbital import dim2NonDim6
            for i in range(n_ic):
                statesArrayChemical[i,:,:] = dim2NonDim6(statesArrayChemical[i,:,:])
                statesArrayElectric[i,:,:] = dim2NonDim6(statesArrayElectric[i,:,:])
        del a

        dataset_test = np.concatenate((statesArrayChemical, statesArrayElectric), axis=0)
        if useEnergy:
            from qutils.orbital import orbitalEnergy
            energyChemical = np.zeros((n_ic,statesArrayChemical.shape[1],1))
            energyElectric= np.zeros((n_ic,statesArrayChemical.shape[1],1))
            for i in range(n_ic):
                energyChemical[i,:,0] = orbitalEnergy(statesArrayChemical[i,:,:])
                energyElectric[i,:,0] = orbitalEnergy(statesArrayElectric[i,:,:])
            if useNorm:
                normingEnergy = energyChemical[0,0,0]
                energyChemical[:,:,0] = energyChemical[:,:,0] / normingEnergy
                energyElectric[:,:,0] = energyElectric[:,:,0] / normingEnergy

            dataset_test = np.concatenate((energyChemical, energyElectric), axis=0)

        if useEnergy and useOE:
            combinedChemical = np.concatenate((OEArrayChemical,energyChemical),axis=2) 
            combinedElectric = np.concatenate((OEArrayElectric,energyElectric),axis=2) 
            dataset_test = np.concatenate((combinedChemical, combinedElectric), axis=0)

        dataset_label_test = np.concatenate((labelsChemical, labelsElectric), axis=0)
        
        groups = np.tile(np.arange(n_ic, dtype=np.int64), 2)

        # Ratios (must satisfy train+val <= 1.0; test gets the remainder)
        # example:
        # train_ratio, val_ratio = 0.7, 0.15
        n_train_ic = int(np.floor(train_ratio * n_ic))
        n_val_ic   = int(np.floor(val_ratio   * n_ic))
        n_test_ic  = n_ic - n_train_ic - n_val_ic
        perm_ic = np.random.permutation(n_ic)
        train_ic = perm_ic[:n_train_ic]
        val_ic   = perm_ic[n_train_ic:n_train_ic + n_val_ic]
        test_ic  = perm_ic[n_train_ic + n_val_ic:]

        # Masks select ALL thrust variants for each IC
        train_mask = np.isin(groups, train_ic)
        val_mask   = np.isin(groups, val_ic)
        test_mask  = np.isin(groups, test_ic)

        test_data,   test_label   = dataset_test[test_mask],  dataset_label_test[test_mask]
    else:
        test_data,   test_label   = dataset[test_mask],  dataset_label[test_mask]

    print(test_data.shape)
    print(test_label.shape)
    print(train_data.shape)
    print(train_label.shape)
    print(val_data.shape)
    print(val_label.shape)

    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label).squeeze(1).long())
    val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_label).squeeze(1).long())
    test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label).squeeze(1).long())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)



    if output_np:
        return train_loader, val_loader, test_loader, train_data,train_label,val_data,val_label,test_data,test_label
    else:
        return train_loader, val_loader, test_loader


# Define the model
class LSTMSequenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        # h0, c0 default to zero if not provided
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        # h_n is shape [num_layers, batch_size, hidden_size].
        # We typically take the last layer's hidden state: h_n[-1]
        
        # Pass the last hidden state through a linear layer for classification on a per timestep basis
        logits = self.classifier(out)       # logits: [B, T, 1]
        logits = logits.squeeze(-1)         # logits: [B, T]
        return logits

class MambaSequenceClassifier(nn.Module):
    def __init__(self,config, input_size, hidden_size, num_layers, num_classes):
        super(MambaSequenceClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.mamba = Mamba(config)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        
        h_n = self.mamba(x)
        
        # h_n is shape [batch_size, seq_length, hidden_size].
        # We typically take the last layer's hidden state: h_n[:,-1,:]
        
        # Pass the last hidden state through a linear layer for classification on a per timestep basis
        logits = self.fc(h_n)               # logits: [B, T, 1]
        logits = logits.squeeze(-1)         # logits: [B, T]

        return logits


def main():
    import yaml
    with open("data.yaml", 'r') as f:
        dataConfig = yaml.safe_load(f)
    dataLoc = dataConfig['classification'] + orbitType +"/" + str(numMinProp) + "min-" + str(numRandSys)
    print(f"Processing datasets for {orbitType} with {numMinProp} minutes and {numRandSys} random systems.")
    # dataLoc = "c/Users/hu650776/GMAT-Thrust-Data/data/classification/data/classification/"+ orbitType +"/" + str(numMinProp) + "min-" + str(numRandSys)


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

    # from qutils.ml.classifer import prepareThrustClassificationDatasets

    if train_ratio == 0.7:
        val_ratio = 0.15
        test_ratio = 0.15
    else:
        val_ratio = train_ratio  
        test_ratio = (1.0 - train_ratio - val_ratio) # not used in network training, only for splitting the data and final evaluation

    train_loader, val_loader, test_loader, train_data,train_label,val_data,val_label,test_data,test_label = prepareThrustClassificationDatasets(yaml_config,dataConfig,output_np=True,vel_noise_std=velNoise,pos_noise_std=1e3*velNoise,train_ratio=train_ratio,test_ratio=test_ratio,val_ratio=val_ratio)

    # Hyperparameters
    input_size = train_data.shape[2] 
    hidden_factor = 8  # hidden size is a multiple of input size
    hidden_size = int(input_size * hidden_factor) # must be multiple of train dim
    num_layers = 1
    num_classes = 3  # e.g., multiclass classification
    learning_rate = 1e-3
    num_epochs = 100

    if useOnePass:
        num_epochs = 1

    criterion = torch.nn.CrossEntropyLoss()

    config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=4,classifer=True)
    model_mamba = MambaSequenceClassifier(config,input_size, hidden_size, num_layers, num_classes).to(device).double()
    optimizer_mamba = torch.optim.Adam(model_mamba.parameters(), lr=learning_rate)

    schedulerPatience = 5

    scheduler_mamba = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_mamba,
        mode='min',             # or 'max' for accuracy
        factor=0.5,             # shrink LR by 50%
        patience=schedulerPatience             # wait for 3 epochs of no improvement
    )

    if use_lstm:
        model_LSTM = LSTMSequenceClassifier(input_size, int(3*hidden_size//4), num_layers, num_classes).to(device).double()
        optimizer_LSTM = torch.optim.Adam(model_LSTM.parameters(), lr=learning_rate)
        scheduler_LSTM = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_LSTM,
            mode='min',             # or 'max' for accuracy
            factor=0.5,             # shrink LR by 50%
            patience=schedulerPatience
        )

        print('\nEntering LSTM Training Loop')
        train_model(model_LSTM, train_loader,val_loader, num_epochs=1000)
        printModelParmSize(model_LSTM)
        print("\nLSTM Validation")
        LSTMInference = timer()
        _eval_loader = test_loader if (testSet != orbitType) else val_loader
        evaluate_model(model_LSTM, _eval_loader,modelString="LSTM")
        LSTMInference.tocStr("LSTM Inference Time")

    print('\nEntering Mamba Training Loop')
    train_model(model_mamba, train_loader,val_loader, num_epochs=1000)
    printModelParmSize(model_mamba)

    print("\nMamba Validation")
    mambaInference = timer()
    _eval_loader = test_loader if (testSet != orbitType) else val_loader
    evaluate_model(model_mamba, val_loader,modelString="Mamba")
    mambaInference.tocStr("Mamba Inference Time")


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
    
    plt.show()