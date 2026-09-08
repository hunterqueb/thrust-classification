import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# use webagg for plotting
plt.switch_backend('WebAgg')

from qutils.ml.utils import printModelParmSize, getDevice
from qutils.integrators import ode45
from qutils.orbital import OE2ECI, dim2NonDim6, ECI2OE
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.tictoc import timer

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
        """
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

def twoBodyJ2Drag(t, y, mu,m_sat):
    # two body problem with J2 perturbation in 6 dimensions taken from astroforge library
    # https://github.com/mit-ll/AstroForge/blob/main/src/astroforge/force_models/_models.py
    # https://github.com/mit-ll/AstroForge/blob/main/src/astroforge/force_models/_forces.py

    # x, v = np.split(y, 2) # orginal line in Astroforge
    # faster than above
    x = y[:3]
    v = y[3:]

    J2 = 4.84165368e-4 * np.sqrt(5)

    M2 = J2 * np.diag(np.array([0.5, 0.5, -1.0]))
    r = np.sqrt(x @ x) # faster than np.linalg.norm(x) (original line in Astroforge)
    v_norm = np.sqrt(v @ v) # faster than np.linalg.norm(v)

    # compute monopole force
    F0 = -mu * x / r**3

    # compute the quadropole force in ITRS
    acc = (mu * R**2 / r**5) * (-5 * x * (x @ M2 @ x) / r**2 + 2 * M2 @ x) + F0

    # ydot = np.hstack((v, acc)) # orginal line in Astroforge
    # faster than above
    ydot = np.empty(6)
    ydot[:3] = v
    ydot[3:] = acc

    rho = rho_0 * np.exp(-(r-R) / h_scale)  # Atmospheric density model
    drag_factor = -0.5 * (rho / m_sat) * c_d * A_sat * v_norm

    a_drag = v * drag_factor
    ydot[3:] += a_drag

    # print(f"rho: {rho}, satellite mass: {m_sat}, a_drag: {a_drag}, force: {np.linalg.norm(ydot[3:]*m_sat)}")

    return ydot

device = getDevice()

import argparse

parser = argparse.ArgumentParser(description='Mamba Time Series Classification for Mass Classification')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for mass classification')
parser.add_argument('--numRandSys', type=int, default=10000, help='Number of random systems to generate')
parser.add_argument('--mass', type=int, default=100, help='Maximum mass for classification')
parser.add_argument('--layers', type=int, default=1, help='Number of Layers for NN')
parser.add_argument('--orbits',type=int,default=1,help="Number of Orbits for Propagation")
parser.add_argument('--timeSeriesLength',type=int,default=1000,help="Number of time steps in the time series")
parser.add_argument('--e',type=float,default=0,help="Max eccentricity of the Orbit Regimes")
parser.add_argument('--i',type=float,default=0,help="Max inclination of the orbit regimes in degrees")

# LSTM comparison (default is True, disable with --no-lstm)
parser.add_argument("--no-lstm", dest="use_lstm", action="store_false",
                    help="Disable LSTM comparison (enabled by default)")
parser.set_defaults(use_lstm=True)
# Transformer comparison (default is False, enable with --transformer)
parser.add_argument("--transformer", dest="use_transformer", action="store_true",
                    help="Enable Transformer model comparison (disabled by default)")
parser.set_defaults(use_transformer=False)
parser.add_argument("--constant-a",dest="constant-a", action="store_false",
                    help="Randomize the semi-major axis (default is false)")
parser.set_defaults(constant_a=True)

args = parser.parse_args()
num_classes = args.num_classes
numRandSys = args.numRandSys
mass_max = args.mass
use_lstm = args.use_lstm
use_transformer = args.use_transformer
num_layers = args.layers
numOrbits = args.orbits
timeSeriesLength = args.timeSeriesLength
e_max = args.e
inc_max = args.i
constant_a = args.constant_a

print(f"Maximum mass for classification : {mass_max * 2} kg")
print(f"Number of equispaced classes    : {num_classes}")
print(f"Number of random systems        : {numRandSys}")
print(f"Number of Layers                : {num_layers}")
print(f"Number of Orbits to Propagate   : {numOrbits}")
print(f"Max Eccentricity of Orbits      : {e_max}")
print(f"Max Inclination of Orbits in deg: {inc_max}")
print(f"Length of Time Series           : {timeSeriesLength}")

print(f"Constant Semimajor Axis         : {constant_a}")
print(f"Use LSTM comparison             : {use_lstm}")
print(f"Use Transformer comparison      : {use_transformer}")


rng = np.random.default_rng() # Seed for reproducibility

# Hyperparameters
problemDim = 6
input_size = problemDim   # 2
hidden_size = 48 # needs to be divisible by input_size
learning_rate = 1e-3
num_epochs = 100
batchSize = 32

# Orbital Parameters
G = 6.67430e-11 # m^3/kg/s^2, gravitational constant
M_earth = 5.97219e24 # kg, mass of Earth
mu = 3.986004418e14  # Earth’s mu in m^3/s^2
R = 6371e3 # radius of earth in m

DU = R
TU = np.sqrt(R**3/mu) # time unit in seconds
A_sat = 10 # m^2, cross section area of satellite

# Atmospheric model parameters
rho_0 = 1.29 # kg/m^3
c_d = 2.1 #shperical model
h_scale = 5000


timeToGenData = timer()

bin_edges = np.linspace(mass_max, mass_max * 2, num_classes + 1)  # +1 because edges are one more than bins
numericalResult = np.zeros((numRandSys,timeSeriesLength,problemDim))
numericalResultLabel = np.zeros((numRandSys,1))

# define the semimajor axis as a function of the random number generator if constant_a is False
# otherwise, use a constant value
if constant_a:
    semimajorAxis = lambda: rng.uniform(R + 100e3,R + 200e3) # random semimajor axis in m
else:
    semimajorAxis = lambda: (R + 150e3) # constant semimajor axis in m

for i in range(numRandSys):
    # Random Conditions for dataset generation
    m_sat = mass_max * rng.random() + mass_max # mass of satellite in kg
    e = e_max * rng.random() # eccentricity
    inc = np.deg2rad(inc_max * rng.random()) # inclination
    a = semimajorAxis()
    nu = np.deg2rad(5*rng.random()) # true anomaly

    # calc mu from mass of satellite and earth to get better accuracy(??)
    mu = G * (M_earth + m_sat) # gravitational parameter in m^3/s^2

    h = np.sqrt(mu*a*(1-e)) # specific angular momentum

    OE = [a,e,inc,0,0,nu]
    y0 = OE2ECI(OE,mu=mu)
    # print(y0)

    tf = 2*np.pi*a**2*np.sqrt(1-e**2)/h * numOrbits # time of flight

    teval = np.linspace(0, tf, timeSeriesLength) # time to evaluate the solution

    t,y = ode45(fun=lambda t, y: twoBodyJ2Drag(t, y, mu,m_sat),tspan=(0, tf),y0=y0, t_eval=teval, rtol=1e-8, atol=1e-10)

    for j in range(timeSeriesLength):
        eci = ECI2OE(y[j,0:3], y[j,3:6])
        numericalResult[i,j,:] = eci[0:6]
    # numericalResult[i,:,:] = dim2NonDim6(y)
    # check mass range and assign label in 5 bins mutually exclusive
    label = np.digitize(m_sat, bin_edges) - 1  # -1 to make it 0-indexed
    numericalResultLabel[i, 0] = label
print("Time to generate data: {:.2f} seconds".format(timeToGenData.tocVal()))

dataset = numericalResult
dataset_label = numericalResultLabel

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
config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=16,classifer=True)
model_mamba = MambaClassifier(config,input_size, hidden_size, num_layers, num_classes).to(device).double()
optimizer_mamba = torch.optim.Adam(model_mamba.parameters(), lr=learning_rate)

schedulerPatience = 5
scheduler_mamba = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_mamba,
    mode='min',             # or 'max' for accuracy
    factor=0.5,             # shrink LR by 50%
    patience=schedulerPatience             # wait for 3 epochs of no improvement
)


# Training loop
def trainClassifier(model,optimizer,scheduler):
    best_loss = float('inf')
    ESpatience = schedulerPatience * 2
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
    model_transformer = TransformerClassifier(input_size, hidden_size, num_layers, num_classes).to(device).double()
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

config_hybrid = MambaConfig(d_model=hidden_size,n_layers = 1,expand_factor=2,d_state=32,d_conv=16,classifer=True)

# model_hybrid = HybridClassifier(config_hybrid,input_size, hidden_size, 1, num_classes).to(device).double()
# optimizer_hybrid = torch.optim.Adam(model_hybrid.parameters(), lr=learning_rate)
# scheduler_hybrid = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer_hybrid,
#     mode='min',             # or 'max' for accuracy
#     factor=0.5,             # shrink LR by 50%
#     patience=schedulerPatience             # wait for 3 epochs of no improvement
# )

# print("\nEntering Hybrid Training Loop")
# hybridTrainTime = timer()
# trainClassifier(model_hybrid,optimizer_hybrid,scheduler_hybrid)
# hybridTrainTime.toc()
# printModelParmSize(model_hybrid)