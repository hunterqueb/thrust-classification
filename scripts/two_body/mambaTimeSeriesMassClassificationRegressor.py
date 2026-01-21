import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import control as ct
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from qutils.tictoc import timer

from qutils.integrators import ode45
from qutils.ml.utils import printModelParmSize, getDevice
from qutils.ml.mamba import Mamba, MambaConfig

#set webagg backend for matplotlib - i've been liking it 
plt.switch_backend('WebAgg')

import argparse

parser = argparse.ArgumentParser(description='Mamba Time Series Classification for Mass Classification')
parser.add_argument('--num_classes', type=int, default=5, help='Number of classes for mass classification')
parser.add_argument('--numRandSys', type=int, default=10000, help='Number of random systems to generate')
parser.add_argument('--mass', type=int, default=10, help='Maximum mass for classification')
parser.add_argument('--damping', type=float, default=0.1, help='Drag constant for the system')

# LSTM comparison (default is True, disable with --no-lstm)
parser.add_argument("--no-lstm", dest="use_lstm", action="store_false",
                    help="Disable LSTM comparison (enabled by default)")
parser.set_defaults(use_lstm=True)

args = parser.parse_args()
num_classes = args.num_classes
numRandSys = args.numRandSys
mass_max = args.mass
dragConst = args.damping
use_lstm = args.use_lstm

print(f"Maximum mass for classification : {mass_max} kg")
print(f"Number of equispaced classes    : {num_classes}")
print(f"Number of random systems        : {numRandSys}")
print(f"Use LSTM comparison             : {use_lstm}")

import torch.nn as nn
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.backbone = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_shared = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.backbone(x)
        x = x[:, -1, :]
        x = torch.relu(self.fc_shared(x))

        class_logits = self.classifier(x)
        mass_pred = self.regressor(x).squeeze(-1)

        return class_logits, mass_pred
        

class MambaClassifier(nn.Module):
    def __init__(self,config, input_size, hidden_size, num_layers, num_classes):
        super(MambaClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.backbone = Mamba(config)
        self.fc_shared = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = x[:, -1, :]
        x = torch.relu(self.fc_shared(x))

        class_logits = self.classifier(x)
        mass_pred = self.regressor(x).squeeze(-1)

        return class_logits, mass_pred




rng = np.random.default_rng(seed=1) # Seed for reproducibility

device = getDevice()

problemDim = 2
t0 = 0; tf = 10
dt = 0.005
t = np.linspace(t0,tf,int(tf/dt))

# Hyperparameters
batchSize = 32
input_size = problemDim   # 2
hidden_size = 64
num_layers = 1
learning_rate = 1e-2
num_epochs = 100
lambda_regression = 1

bin_edges = np.linspace(0, mass_max, num_classes + 1)  # +1 because edges are one more than bins

config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=16,classifer=True)

numericalResultForced = np.zeros((numRandSys,len(t),problemDim))
numericalResultUnforced = np.zeros((numRandSys,len(t),problemDim))
numericalResultLabel = np.zeros((numRandSys,1))
mass_array = np.zeros((numRandSys,))

timeToGenData = timer()
for i in range(numRandSys):
    k = rng.random() * 10
    m = rng.random() * 10
    wr = np.sqrt(k/m)
    F0_const = 0.1
    F0 = F0_const * rng.random()
    c = dragConst * m

    A = np.array(([0, 1], [-k/m, -c/m]))
    B = np.array([0, 1])
    C = np.array(([1, 0], [0, 0]))
    D = 0

    u = F0 * np.sin(t)

    sys = ct.StateSpace(A, B, C, D)
    x0 = [rng.random(),rng.random()]
    
    resultsForced = ct.forced_response(sys, t, u, x0)

    numericalResultForced[i, :, :] = resultsForced.x.T
    mass_array[i] = m

    label = np.digitize(m, bin_edges) - 1  # binning
    numericalResultLabel[i, 0] = label
        
print("Time to generate data: {:.2f} seconds".format(timeToGenData.tocVal()))

class MassClassificationRegressionDataset(Dataset):
    def __init__(self, sequences, labels_class, labels_mass):
        self.sequences = torch.from_numpy(sequences).float()
        self.labels_class = torch.from_numpy(labels_class).long()
        self.labels_mass = torch.from_numpy(labels_mass).float()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels_class[idx], self.labels_mass[idx]

# Prepare datasets
sequences = numericalResultForced
labels_class = numericalResultLabel.squeeze(1)
labels_mass = mass_array

full_dataset = MassClassificationRegressionDataset(sequences, labels_class, labels_mass)

# Split
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

criterion = nn.CrossEntropyLoss()
model_mamba = MambaClassifier(config,input_size, hidden_size, num_layers, num_classes).to(device)
optimizer_mamba = torch.optim.Adam(model_mamba.parameters(), lr=learning_rate)

schedulerPatience = 3
scheduler_mamba = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_mamba,
    mode='min',             # or 'max' for accuracy
    factor=0.5,             # shrink LR by 50%
    patience=schedulerPatience
)


# Training loop
def trainClassifierRegressor(model,optimizer,scheduler):
    best_val_loss = float('inf')
    counter = 0
    ESpatience = schedulerPatience * 2

    criterion_classification = nn.CrossEntropyLoss()
    criterion_regression = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for sequences, labels_class, labels_mass in train_loader:
            sequences = sequences.to(device, non_blocking=True)
            labels_class = labels_class.to(device, non_blocking=True)
            labels_mass = labels_mass.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                class_logits, mass_pred = model(sequences)
                loss_class = criterion_classification(class_logits, labels_class)
                loss_regress = criterion_regression(mass_pred, labels_mass)
                loss = loss_class + lambda_regression * loss_regress

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_class_correct = 0
        val_class_total = 0
        val_regression_error = 0.0

        with torch.no_grad():
            for sequences, labels_class, labels_mass in val_loader:
                sequences = sequences.to(device, non_blocking=True)
                labels_class = labels_class.to(device, non_blocking=True)
                labels_mass = labels_mass.to(device, non_blocking=True)

                class_logits, mass_pred = model(sequences)
                loss_class = criterion_classification(class_logits, labels_class)
                loss_regress = criterion_regression(mass_pred, labels_mass)
                loss = loss_class + lambda_regression * loss_regress
                val_loss += loss.item()

                _, predicted_classes = torch.max(class_logits, dim=1)
                val_class_total += labels_class.size(0)
                val_class_correct += (predicted_classes == labels_class).sum().item()

                val_regression_error += torch.nn.functional.mse_loss(mass_pred, labels_mass, reduction='sum').item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * val_class_correct / val_class_total
        val_rmse = (val_regression_error / val_class_total) ** 0.5

        print(f"Validation Loss: {avg_val_loss:.4f}, Classification Accuracy: {val_accuracy:.2f}%, Regression RMSE: {val_rmse:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            # torch.save(model.state_dict(), "best_model_checkpoint.pth")
        else:
            counter += 1
            if counter >= ESpatience:
                print("Early stopping triggered.")
                break

if use_lstm:
    model_LSTM = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)

    optimizer_LSTM = torch.optim.Adam(model_LSTM.parameters(), lr=learning_rate)
    scheduler_LSTM = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_LSTM,
        mode='min',             # or 'max' for accuracy
        factor=0.5,             # shrink LR by 50%
        patience=schedulerPatience
    )

    print('\nEntering LSTM Training Loop')
    LSTMTrainTime = timer()
    trainClassifierRegressor(model_LSTM,optimizer_LSTM,scheduler_LSTM)
    LSTMTrainTime.toc()
    printModelParmSize(model_LSTM)

print('\nEntering Mamba Training Loop')
mambaTrainTime = timer()
trainClassifierRegressor(model_mamba,optimizer_mamba,scheduler_mamba)
mambaTrainTime.toc()
printModelParmSize(model_mamba)
