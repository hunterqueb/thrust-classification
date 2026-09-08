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
from qutils.orbital import OE2ECI, dim2NonDim6
#set webagg backend for matplotlib - i've been liking it 
plt.switch_backend('WebAgg')

import sys


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
        
#tranformer classifier for time series data
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        
        self.d_model = hidden_size  # Output of transformer & input to fc
        self.embedding = nn.Linear(input_size, self.d_model)  # Project input to match d_model

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=8,  # Make sure d_model % nhead == 0
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=64,  # Internal feedforward layer size inside Transformer
            batch_first=True
        )
        
        self.fc = nn.Linear(self.d_model, num_classes)  # Final classification layer

    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        x = self.embedding(x)         # [batch_size, seq_length, d_model]
        out = self.transformer(x, x)  # [batch_size, seq_length, d_model]
        last_output = out[:, -1, :]   # [batch_size, d_model]
        logits = self.fc(last_output) # [batch_size, num_classes]
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
numRandSys = 1000
problemDim = 4
num_layers = 1
plotOn = True

# Hyperparameters
input_size = problemDim  
hidden_size = 64
# num_layers = 1
num_classes = 1  # e.g., binary classification
learning_rate = 1e-2
num_epochs = 100

config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=16,classifer=True)

DU = 6378.1*1e3 # radius of earth
TU = 806.80415

muR = 3.96800e14
rLEO = np.array([6778e3,0])
vLEO = np.array([0, 8.435e3])
a = 8578e3
TLEO = 2*np.pi * np.sqrt(np.linalg.norm(a)**3 / muR)

e = 0.2098

mu = 1
a = a/DU
r = rLEO/DU
v = vLEO * TU / DU
T = TLEO / TU
pam = [mu,1.08263e-3]
h = np.cross(r,v)

m_sat = 1
c_d = 2.1 #shperical model
A_sat = 1.0013 / (DU ** 2)
h_scale = 50 * 1000 / DU
rho_0 = 1.29 * 1000 ** 2 / (DU**2)

def system(t, y, p=pam):
    r = y[0:2]
    R = np.linalg.norm(r)
    v = y[2:4]
    v_norm = np.linalg.norm(v)

    mu = p[0]; J2 = p[1]
    dydt1 = y[2]
    dydt2 = y[3]

    factor = 1.5 * J2 * (1 / R)**2 / R**3
    j2_accel_x = factor * (1) * r[0]
    j2_accel_y = factor * (3) * r[1]

    rho = rho_0 * np.exp(-R / h_scale)  # Atmospheric density model
    drag_factor = -0.5 * (rho / m_sat) * c_d * A_sat * v_norm
    a_drag_x = drag_factor * y[2]
    a_drag_y = drag_factor *  y[3]

    init_acc = 4e-3 / 1000 / DU * rng.random() # km/s^2, thrust acceleration
    thrust_acc = init_acc * (v / v_norm)

    dydt3 = -mu / R**3 * y[0] + j2_accel_x + a_drag_x + thrust_acc[0]
    dydt4 = -mu / R**3 * y[1] + j2_accel_y + a_drag_y + thrust_acc[1]

    return np.array([dydt1, dydt2,dydt3,dydt4])


def system_without_thrust(t, y, p=pam):
    r = y[0:2]
    R = np.linalg.norm(r)
    v = y[2:4]
    v_norm = np.linalg.norm(v)

    mu = p[0]; J2 = p[1]
    dydt1 = y[2]
    dydt2 = y[3]

    factor = 1.5 * J2 * (1 / R)**2 / R**3
    j2_accel_x = factor * (1) * r[0]
    j2_accel_y = factor * (3) * r[1]

    rho = rho_0 * np.exp(-R / h_scale)  # Atmospheric density model
    drag_factor = -0.5 * (rho / m_sat) * c_d * A_sat * v_norm
    a_drag_x = drag_factor * y[2]
    a_drag_y = drag_factor *  y[3]


    dydt3 = -mu / R**3 * y[0] + j2_accel_x + a_drag_x
    dydt4 = -mu / R**3 * y[1] + j2_accel_y + a_drag_y

    return np.array([dydt1, dydt2,dydt3,dydt4])


IC_const = np.concatenate((r,v))

IC = np.zeros((numRandSys,problemDim))

timeToGenData = timer()
for i in range(numRandSys):
    # Generate random system parameters for a orbit problem with and without forcing
        
        IC[i] = IC_const + 0.1 * rng.random((problemDim,)) # randomize initial conditions

        t0 = 0; tf = 0.25*T
        dt = 0.01
        t = np.linspace(t0,tf,int(tf/dt))   

        t_forced,resultsForced = ode45(system, [t0,tf], IC[i],t_eval=t)
        t_unforced,resultsUnforced  = ode45(system_without_thrust, [t0,tf], IC[i],t_eval=t)

        if i == 0:
            numericalResultForced = np.zeros((numRandSys,len(t),problemDim))
            numericalResultUnforced = np.zeros((numRandSys,len(t),problemDim))

        numericalResultForced[i,:,:] = resultsForced
        numericalResultUnforced[i,:,:] = resultsUnforced
        # display progress to user
        if i % 100 == 0:
            print("Completed {} of {} random systems".format(i,numRandSys))
# generate box plots of IC in each dimension in seperate figures
dimensionLabels = ['x','y','vx','vy']
unitLabels = ['DU','DU','DU/TU','DU/TU']
for i in range(problemDim):
    plt.figure()
    plt.boxplot(IC[:,i])
    plt.title("Random Initial Condition in Dimension {}".format(dimensionLabels[i]))
    plt.ylabel("Value in {}".format(unitLabels[i]))
    plt.grid()


print("Time to generate data: {:.2f} seconds".format(timeToGenData.tocVal()))

# plt.figure()
# plt.plot(t,resultsForced.x.T[:,0])
# plt.plot(t,resultsUnforced.x.T[:,0])
# plt.grid()
# plt.title("Random Sample Forced and Unforced Responses")
# plt.xlabel("Time (s)")
# plt.ylabel("Displacement (m)")
# plt.legend(["Forced by [0,{}] N Force".format(F0_const),"Unforced Response"])

# make data labels for a two class problem of shape numRandSys x 2, 1, 0 for forced and unforced respectively
ForcedLabel = np.ones(numRandSys)
UnforcedLabel = np.zeros(numRandSys)

dataset = np.concatenate((numericalResultForced,numericalResultUnforced),axis=0)
dataset_label = np.concatenate((ForcedLabel,UnforcedLabel),axis=0).reshape(2*numRandSys,1)

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

train_dataset = TensorDataset(torch.from_numpy(train_data).float(),torch.from_numpy(train_label).long())
val_dataset = TensorDataset(torch.from_numpy(val_data).float(),torch.from_numpy(val_label).long())
test_dataset = TensorDataset(torch.from_numpy(test_data).float(),torch.from_numpy(test_label).long())

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

model_mamba = MambaClassifier(config,input_size, hidden_size, num_layers, num_classes).to(device).float()
model_LSTM = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device).float()
model_transformer = TransformerClassifier(input_size, hidden_size, num_layers, num_classes).to(device).float()

criterion = nn.BCEWithLogitsLoss() # for binary classification
optimizer_mamba = torch.optim.Adam(model_mamba.parameters(), lr=learning_rate)
optimizer_LSTM = torch.optim.Adam(model_LSTM.parameters(), lr=learning_rate)
optimizer_transformer = torch.optim.Adam(model_transformer.parameters(), lr=learning_rate)

schedulerPatience = 3
scheduler_mamba = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_mamba,
    mode='min',             # or 'max' for accuracy
    factor=0.5,             # shrink LR by 50%
    patience=schedulerPatience
)

scheduler_LSTM = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_LSTM,
    mode='min',             # or 'max' for accuracy
    factor=0.5,             # shrink LR by 50%
    patience=schedulerPatience
)


scheduler_transformer = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_transformer,
    mode='min',             # or 'max' for accuracy
    factor=0.5,             # shrink LR by 50%
    patience=schedulerPatience
)


# Training loop
def trainClassifier(model,optimizer,scheduler):
    best_loss = float('inf')
    ESpatience = schedulerPatience * 2  # patience for early stopping
    counter = 0
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
                loss = criterion(logits, labels.float())  # [batch_size]
                
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
                    probs = torch.sigmoid(outputs)                  # Convert logits to probabilities
                    predicted = (probs >= 0.5).float()              # Threshold at 0.5
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100.0 * correct / total
            print(f"Validation Accuracy: {accuracy:.2f}%")

            val_loss = 0.0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device).float()
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            scheduler.step(avg_val_loss)  # <-- update LR based on validation loss
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                counter = 0  # reset
                # save best model if desired
            else:
                counter += 1
                if counter >= ESpatience:
                    print("Early stopping")
                    break

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return

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

print('\nEntering Transformer Training Loop')
transformerTrainTime = timer()
trainClassifier(model_transformer,optimizer_transformer,scheduler_transformer)
transformerTrainTime.toc()
printModelParmSize(model_transformer)



if plotOn:
    plt.show()
 