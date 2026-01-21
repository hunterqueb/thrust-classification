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
parser.add_argument("--pca", type=int, default=None, help="If set to an integer, use PCA to reduce the input features to this number of components.")
parser.add_argument('--mlp', dest="useMLP", action='store_true', help='Use a simple MLP on Hankel+PCA pooled data for comparison.')

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
useHybrid=args.use_hybrid
find_SW=args.find_SW
use_classic = args.use_classic
use_nearestNeighbor = args.use_nearestNeighbor
saveNets = args.saveNets
velNoise = args.velNoise
run_shap = args.run_shap
train_ratio = args.train_ratio
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
if use_nearestNeighbor:
    strAdd = strAdd + "1-NN_"
if train_ratio != 0.7:
    strAdd = strAdd + f"Train_{int(4*train_ratio*numRandSys)}_"
if testSet != orbitType:
    strAdd = strAdd + "Test_" + testSet + "_"
if velNoise != 1e-3:
    strAdd = strAdd + f"VelNoise{velNoise}_"
if pca_enabled:
    strAdd = strAdd + f"PCA{pca_n_components}_"
if useMLP:
    strAdd = strAdd + "MLP_"

# remove trailing _
if strAdd.endswith("_"):
    strAdd = strAdd[:-1]

print(f"Training with {int(4*train_ratio*numRandSys)} systems")

logLoc = "gmat/data/classification/"+str(orbitType)+"/" + str(numMinProp) + "min-" + str(numRandSys) + "/"
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

    train_loader, val_loader, test_loader, train_data,train_label,val_data,val_label,test_data,test_label = prepareThrustClassificationDatasets(yaml_config,dataConfig,output_np=True,vel_noise_std=velNoise,pos_noise_std=1e3*velNoise,train_ratio=train_ratio,test_ratio=test_ratio,val_ratio=val_ratio,pca_enabled=pca_enabled,pca_mode="hankel",hankel_pool="mean")

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

    if use_classic:
        from lightgbm import LGBMClassifier
        from qutils.ml.classic.classifier import printDTModelSize, validate_lightgbm

        print("\nEntering Decision Trees Training Loop")
        # flatten features
        X_train = train_data.reshape(train_data.shape[0], -1).astype(np.float32)    # (number of systems to train on, network features * length of time series)    
        y_train = train_label.reshape(-1).astype(np.int32)             # (number of systems to train on,)
        classicModel = LGBMClassifier(objective="multiclass",num_classes=num_classes,n_estimators=30,max_depth=-1,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,verbosity=-1)   # or 'verbose' for older builds)       
        DTTimer = timer()
        classicModel.fit(X_train, y_train)
        DTTimer.toc()
        printDTModelSize(classicModel)
        print("\nDecision Trees Validation")
        DTTimerInference = timer()
        if testSet != orbitType:
            validate_lightgbm(classicModel, test_loader, num_classes, classlabels=classlabels, print_report=True)
        else:
            validate_lightgbm(classicModel, val_loader, num_classes, classlabels=classlabels, print_report=True)
        DTTimerInference.tocStr("Decision Trees Inference Time")
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
        if useOE:
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

    if useOE:
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

    if useMLP is True:
        print("\nEntering MLP Training Loop")
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
        return_pca=True
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