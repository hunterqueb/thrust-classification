import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import control as ct
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from qutils.tictoc import timer

from qutils.integrators import ode45
from qutils.ml.utils import printModelParmSize, getDevice
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml.superweight import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight,findMambaSuperActivation,plotSuperActivation,zeroModelWeight

#set webagg backend for matplotlib - i've been liking it 
plt.switch_backend('WebAgg')

import argparse

parser = argparse.ArgumentParser(description='Mamba Time Series Classification for Mass Classification')
parser.add_argument('--num_classes', type=int, default=5, help='Number of classes for mass classification')
parser.add_argument('--numRandSys', type=int, default=10000, help='Number of random systems to generate')
parser.add_argument('--mass', type=int, default=10, help='Maximum mass for classification')
parser.add_argument('--layers', type=int, default=1, help='Number of Layers for NN')
parser.add_argument('--damping',type=float,default=0.1,help="Damping Ratio of the Mass")
# LSTM comparison (default is True, disable with --no-lstm)
parser.add_argument("--no-lstm", dest="use_lstm", action="store_false",
                    help="Disable LSTM comparison (enabled by default)")
parser.set_defaults(use_lstm=True)

# Transformer comparison (default is False, enable with --transformer)
parser.add_argument("--transformer", dest="use_transformer", action="store_true",
                    help="Enable Transformer model comparison (disabled by default)")
parser.set_defaults(use_transformer=False)

args = parser.parse_args()
num_classes = args.num_classes
numRandSys = args.numRandSys
mass_max = args.mass
use_lstm = args.use_lstm
use_transformer = args.use_transformer
num_layers = args.layers
damping = args.damping

print(f"Maximum mass for classification : {mass_max} kg")
print(f"Number of equispaced classes    : {num_classes}")
print(f"Number of random systems        : {numRandSys}")
print(f"Use LSTM comparison             : {use_lstm}")
print(f"Use Transformer comparison      : {use_transformer}")
print(f"Number of Layers                : {num_layers}")
print(f"Damping of System               : {damping}")

import torch.nn as nn
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        # h0, c0 default to zero if not provided
        out, (h_n, c_n) = self.lstm(x)
        
        # h_n is shape [num_layers, batch_size, hidden_size].
        # We typically take the last layer's hidden state: h_n[-1]
        last_hidden = h_n[-1]  # [batch_size, hidden_size]
        
        # Pass the last hidden state through a linear layer for classification
        logits = self.fc(last_hidden)  # [batch_size, num_classes]
        
        return logits
        

class MambaClassifier(nn.Module):
    def __init__(self,config, input_size, hidden_size, num_layers, num_classes):
        super(MambaClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.mamba = Mamba(config)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        
        h_n = self.mamba(x) # [batch_size, seq_length, hidden_size]
        
        # h_n is shape [batch_size, seq_length, hidden_size].
        # We typically take the last layer's hidden state: h_n[:,-1,:]
        last_hidden = h_n[:,-1,:]  # [batch_size, hidden_size]
        
        # Pass the last hidden state through a linear layer for classification
        logits = self.fc(last_hidden)  # [batch_size, num_classes]
        
        return logits




rng = np.random.default_rng(seed=1) # Seed for reproducibility

device = getDevice()

batchSize = 32
problemDim = 2
t0 = 0; tf = 10
dt = 0.005
t = np.linspace(t0,tf,int(tf/dt))

# Hyperparameters
input_size = problemDim   # 2
hidden_size = 64
learning_rate = 1e-2
num_epochs = 100

bin_edges = np.linspace(0, mass_max, num_classes + 1)  # +1 because edges are one more than bins

config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=16,classifer=True)

numericalResultForced = np.zeros((numRandSys,len(t),problemDim))
numericalResultUnforced = np.zeros((numRandSys,len(t),problemDim))
numericalResultLabel = np.zeros((numRandSys,1))

timeToGenData = timer()
for i in range(numRandSys):
    # linear system for a simple harmonic oscillator
    k = rng.random() * mass_max; m = rng.random() * mass_max
    wr = np.sqrt(k/m)
    F0_const = 0.1 
    F0 = F0_const * rng.random()
    c = damping * m # consider proportional damping 

    A = np.array(([0,1],[-k/m,-c/m]))

    B = np.array([0,1])

    C = np.array(([1,0],[0,0]))

    D = 0

    u = F0 * np.sin(t)

    sys = ct.StateSpace(A,B,C,D)

    x0 = [rng.random(),rng.random()]

    resultsForced = ct.forced_response(sys,t,u,x0)
    resultsUnforced = ct.forced_response(sys,t,u * 0,x0)

    numericalResultForced[i,:,:] = resultsForced.x.T
    numericalResultUnforced[i,:,:] = resultsUnforced.x.T
    # check mass range and assign label in 5 bins mutually exclusive
    label = np.digitize(m, bin_edges) - 1  # -1 to make it 0-indexed
    numericalResultLabel[i, 0] = label
        
print("Time to generate data: {:.2f} seconds".format(timeToGenData.tocVal()))

plt.figure()
plt.plot(t,resultsForced.x.T[:,0])
plt.plot(t,resultsUnforced.x.T[:,0])
plt.grid()
plt.title("Random Sample Forced and Unforced Responses")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.legend(["Forced by [0,{}] N Force".format(F0_const),"Unforced Response"])
# plt.show()

# ForcedLabel = np.ones(numRandSys)
# UnforcedLabel = np.zeros(numRandSys)

# dataset = np.concatenate((numericalResultForced,numericalResultUnforced),axis=0)
# dataset_label = np.concatenate((ForcedLabel,UnforcedLabel),axis=0)

# construct a dataset labeling mass ranges from 0 to 10kg, in 5 bins
dataset = numericalResultUnforced
dataset_label = numericalResultLabel

dataset = np.concatenate((numericalResultForced,numericalResultUnforced),axis=0)
dataset_label = np.concatenate((numericalResultLabel,numericalResultLabel),axis=0)


indices = np.random.permutation(dataset.shape[0])

dataset = dataset[indices]
dataset_label = dataset_label[indices]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15


total_samples = len(dataset)
train_end = int(train_ratio * total_samples)
val_end = int((train_ratio + val_ratio) * total_samples)

# Split the data
train_data = dataset[:train_end]
train_label = dataset_label[:train_end]

val_data = dataset[train_end:val_end]
val_label = dataset_label[train_end:val_end]

test_data = dataset[val_end:]
test_label = dataset_label[val_end:]

train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label).squeeze(1).long())
val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_label).squeeze(1).long())
test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label).squeeze(1).long())

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False,pin_memory=True)

criterion = nn.CrossEntropyLoss()
model_mamba = MambaClassifier(config,input_size, hidden_size, num_layers, num_classes).to(device).double()
optimizer_mamba = torch.optim.Adam(model_mamba.parameters(), lr=learning_rate)

schedulerPatience = 3
scheduler_mamba = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_mamba,
    mode='min',             # or 'max' for accuracy
    factor=0.5,             # shrink LR by 50%
    patience=schedulerPatience             # wait for 3 epochs of no improvement
)


# Training loop
def trainClassifier(model,optimizer,scheduler):
    best_loss = float('inf')
    ESpatience = schedulerPatience * 2  # early stopping patience
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for sequences, labels in train_loader:
            sequences = sequences.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)

            # Forward
            logits = model(sequences)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device,non_blocking=True)
                labels = labels.to(device,non_blocking=True)

                outputs = model(sequences)  # [batch_size, num_classes]
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * correct / total
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Early stopping logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            # Optional: save model checkpoint here
        else:
            counter += 1
            if counter >= ESpatience:
                print("Early stopping triggered.")
                break

if use_lstm:
    model_LSTM = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device).double()

    optimizer_LSTM = torch.optim.Adam(model_LSTM.parameters(), lr=learning_rate)
    scheduler_LSTM = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_LSTM,
        mode='min',             # or 'max' for accuracy
        factor=0.5,             # shrink LR by 50%
        patience=schedulerPatience
    )

    print('\nEntering LSTM Training Loop')
    LSTMTrainTime = timer()
    trainClassifier(model_LSTM,optimizer_LSTM,scheduler_LSTM)
    LSTMTrainTime.toc()
    printModelParmSize(model_LSTM)

print('\nEntering Mamba Training Loop')
mambaTrainTime = timer()
trainClassifier(model_mamba,optimizer_mamba,scheduler_mamba)
mambaTrainTime.toc()
printModelParmSize(model_mamba)

if use_transformer:
    model_transformer = MambaClassifier(config,input_size, hidden_size, num_layers, num_classes).to(device).double()
    optimizer_transformer = torch.optim.Adam(model_transformer.parameters(), lr=learning_rate)
    scheduler_transformer = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_transformer,
        mode='min',             # or 'max' for accuracy
        factor=0.5,             # shrink LR by 50%
        patience=schedulerPatience             # wait for 3 epochs of no improvement
    )

    print('\nEntering Transformer Training Loop')
    transformerTrainTime = timer()
    trainClassifier(model_transformer,optimizer_transformer,scheduler_transformer)
    transformerTrainTime.toc()
    printModelParmSize(model_transformer)