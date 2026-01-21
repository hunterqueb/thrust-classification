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
        
        h_n = self.mamba(x)
        
        # h_n is shape [batch_size, seq_length, hidden_size].
        # We typically take the last layer's hidden state: h_n[:,-1,:]
        last_hidden = h_n[:,-1,:]  # [batch_size, hidden_size]
        
        # Pass the last hidden state through a linear layer for classification
        logits = self.fc(last_hidden)  # [batch_size, num_classes]
        
        return logits




rng = np.random.default_rng(seed=1) # Seed for reproducibility

device = getDevice()

batchSize = 32
numRandSys = 10000
problemDim = 2
t0 = 0; tf = 10
dt = 0.005
t = np.linspace(t0,tf,int(tf/dt))

# Hyperparameters
input_size = problemDim   # 2
hidden_size = 64
num_layers = 1
num_classes = 2  # e.g., binary classification
learning_rate = 1e-3
num_epochs = 10

config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=16,classifer=True)



numericalResultForced = np.zeros((numRandSys,len(t),problemDim))
numericalResultUnforced = np.zeros((numRandSys,len(t),problemDim))

timeToGenData = timer()
for i in range(numRandSys):
    # linear system for a simple harmonic oscillator
    k = rng.random(); m = rng.random()
    wr = np.sqrt(k/m)
    F0_const = 0.1 
    F0 = F0_const * rng.random()
    c = 0.1 # consider damping for now

    A = np.array(([0,1],[-k/m,-c/m]))

    B = np.array([0,1])

    C = np.array(([1,0],[0,0]))

    D = 0

    u = F0 * np.sin(t)

    sys = ct.StateSpace(A,B,C,D)

    resultsForced = ct.forced_response(sys,t,u,[1,0])
    resultsUnforced = ct.forced_response(sys,t,u * 0,[1,0])

    numericalResultForced[i,:,:] = resultsForced.x.T
    numericalResultUnforced[i,:,:] = resultsUnforced.x.T

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

ForcedLabel = np.ones(numRandSys)
UnforcedLabel = np.zeros(numRandSys)

dataset = np.concatenate((numericalResultForced,numericalResultUnforced),axis=0)
dataset_label = np.concatenate((ForcedLabel,UnforcedLabel),axis=0)

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

train_dataset = TensorDataset(torch.from_numpy(train_data),torch.from_numpy(train_label).long())
val_dataset = TensorDataset(torch.from_numpy(val_data),torch.from_numpy(val_label).long())
test_dataset = TensorDataset(torch.from_numpy(test_data),torch.from_numpy(test_label).long())

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

model_mamba = MambaClassifier(config,input_size, hidden_size, num_layers, num_classes).to(device).double()
model_LSTM = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device).double()
criterion = nn.CrossEntropyLoss()
optimizer_mamba = torch.optim.Adam(model_mamba.parameters(), lr=learning_rate)
optimizer_LSTM = torch.optim.Adam(model_LSTM.parameters(), lr=learning_rate)

# Training loop
def trainClassifier(model,optimizer):
    # early stopping by user control ctrl+c to break the training loop
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for sequences, labels in train_loader:
                # Move data to GPU
                sequences = sequences.to(device)  # [batch_size, seq_length, input_size]
                labels = labels.to(device)        # [batch_size]
                
                # Forward pass
                logits = model(sequences)        # [batch_size, num_classes]
                loss = criterion(logits, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

            # Validation (optional quick check)
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    outputs = model(sequences)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100.0 * correct / total
            print(f"Validation Accuracy: {accuracy:.2f}%")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return

print('\nEntering Mamba Training Loop')
mambaTrainTime = timer()
trainClassifier(model_mamba,optimizer_mamba)
mambaTrainTime.toc()

print('\nEntering LSTM Training Loop')
LSTMTrainTime = timer()
trainClassifier(model_LSTM,optimizer_LSTM)
LSTMTrainTime.toc()

printModelParmSize(model_mamba)
printModelParmSize(model_LSTM)
