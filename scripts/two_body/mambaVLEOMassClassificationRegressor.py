import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

# use webagg for plotting
plt.switch_backend('WebAgg')

from qutils.ml.utils import printModelParmSize, getDevice
from qutils.integrators import ode45
from qutils.orbital import OE2ECI
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.tictoc import timer


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
    
def calcDragForce(y,m_sat):
    x = y[:3]
    v = y[3:]

    r = np.sqrt(x @ x) # faster than np.linalg.norm(x) (original line in Astroforge)
    v_norm = np.sqrt(v @ v) # faster than np.linalg.norm(v)

    # Atmospheric drag force
    rho = rho_0 * np.exp(-(r-R) / h_scale)  # Atmospheric density model
    drag_factor = -0.5 * (rho / m_sat) * c_d * A_sat * v_norm

    a_drag = v * drag_factor

    return a_drag

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
parser.add_argument('--rLambda',type=float,default=1,help="Lambda for regression loss")

# LSTM comparison (default is True, disable with --no-lstm)
parser.add_argument("--no-lstm", dest="use_lstm", action="store_false",
                    help="Disable LSTM comparison (enabled by default)")
parser.set_defaults(use_lstm=True)
# Transformer comparison (default is False, enable with --transformer)
parser.add_argument("--transformer", dest="use_transformer", action="store_true",
                    help="Enable Transformer model comparison (disabled by default)")
parser.set_defaults(use_transformer=False)
parser.add_argument("--constant-a",dest="constant-a", action="store_true",
                    help="Apply constant semi-major axis (default is false)")
parser.set_defaults(constant_a=False)

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
lambda_regression = args.rLambda

print(f"Maximum mass for classification : {mass_max * 2} kg")
print(f"Number of equispaced classes    : {num_classes}")
print(f"Number of random systems        : {numRandSys}")
print(f"Number of Layers                : {num_layers}")
print(f"Number of Orbits to Propagate   : {numOrbits}")
print(f"Max Eccentricity of Orbits      : {e_max}")
print(f"Max Inclination of Orbits in deg: {inc_max}")
print(f"Length of Time Series           : {timeSeriesLength}")
print(f"Lambda for Regression Loss      : {lambda_regression}")

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
mass_array = np.zeros((numRandSys,))

# define the semimajor axis as a function of the random number generator if constant_a is False
# otherwise, use a constant value
if not constant_a:
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

    numericalResult[i,:,:] = y
    # check mass range and assign label in 5 bins mutually exclusive
    label = np.digitize(m_sat, bin_edges) - 1  # -1 to make it 0-indexed
    numericalResultLabel[i, 0] = label
    mass_array[i] = m_sat

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

sequences = numericalResult
labels_class = numericalResultLabel.squeeze(1)
labels_mass = mass_array

full_dataset = MassClassificationRegressionDataset(sequences, labels_class, labels_mass)

# Split
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Loaders
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, pin_memory=True)

config = MambaConfig(d_model=input_size,n_layers = num_layers,expand_factor=hidden_size//input_size,d_state=32,d_conv=16,classifer=True)
model_mamba = MambaClassifier(config,input_size, hidden_size, num_layers, num_classes).to(device)
optimizer_mamba = torch.optim.Adam(model_mamba.parameters(), lr=learning_rate)

schedulerPatience = 5
scheduler_mamba = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_mamba,
    mode='min',             # or 'max' for accuracy
    factor=0.5,             # shrink LR by 50%
    patience=schedulerPatience             # wait for 3 epochs of no improvement
)


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


        


# if use_transformer:
#     class TransformerClassifier(nn.Module):
#         def __init__(self, input_size, hidden_size, num_layers, num_classes):
#             super(TransformerClassifier, self).__init__()
            
#             self.d_model = hidden_size  # Output of transformer & input to fc
#             self.embedding = nn.Linear(input_size, self.d_model)  # Project input to match d_model

#             self.transformer = nn.Transformer(
#                 d_model=self.d_model,
#                 nhead=8,  # Make sure d_model % nhead == 0
#                 num_encoder_layers=num_layers,
#                 num_decoder_layers=num_layers,
#                 dim_feedforward=64,  # Internal feedforward layer size inside Transformer
#                 batch_first=True
#             )
            
#             self.fc = nn.Linear(self.d_model, num_classes)  # Final classification layer

#         def forward(self, x):
#             """
#             x: [batch_size, seq_length, input_size]
#             """
#             x = self.embedding(x)         # [batch_size, seq_length, d_model]
#             out = self.transformer(x, x)  # [batch_size, seq_length, d_model]
#             last_output = out[:, -1, :]   # [batch_size, d_model]
#             logits = self.fc(last_output) # [batch_size, num_classes]
#             return logits

#     model_transformer = TransformerClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
#     optimizer_transformer = torch.optim.Adam(model_transformer.parameters(), lr=learning_rate)
#     scheduler_transformer = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer_transformer,
#         mode='min',             # or 'max' for accuracy
#         factor=0.5,             # shrink LR by 50%
#         patience=schedulerPatience             # wait for 3 epochs of no improvement
#     )

#     print('\nEntering Transformer Training Loop')
#     transformerTrainTime = timer()
#     trainClassifierRegressor(model_transformer,optimizer_transformer,scheduler_transformer)
#     transformerTrainTime.toc()
#     printModelParmSize(model_transformer)


# class HybridClassifier(nn.Module):
#     def __init__(self,config, input_size, hidden_size, num_layers, num_classes):
#         super(HybridClassifier, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=True  # Bidirectional LSTM
#         )
#         self.mamba = Mamba(config)
#         self.fc = nn.Linear(hidden_size * 2, num_classes)
        
#     def forward(self, x):
#         """
#         x: [batch_size, seq_length, input_size]
#         """
#         # h0, c0 default to zero if not provided
#         out, (h_n, c_n) = self.lstm(x)
#         h_n = self.mamba(out) # [batch_size, seq_length, hidden_size]

#         # h_n is shape [num_layers, batch_size, hidden_size].
#         # We typically take the last layer's hidden state: h_n[-1]
#         last_hidden = h_n[:,-1,:]  # [batch_size, hidden_size]
        
#         # Pass the last hidden state through a linear layer for classification
#         logits = self.fc(last_hidden)  # [batch_size, num_classes]
        
#         return logits

# config_hybrid = MambaConfig(d_model=hidden_size,n_layers = 1,expand_factor=2,d_state=32,d_conv=16,classifer=True)

# model_hybrid = HybridClassifier(config_hybrid,input_size, hidden_size, 1, num_classes).to(device)
# optimizer_hybrid = torch.optim.Adam(model_hybrid.parameters(), lr=learning_rate)
# scheduler_hybrid = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer_hybrid,
#     mode='min',             # or 'max' for accuracy
#     factor=0.5,             # shrink LR by 50%
#     patience=schedulerPatience             # wait for 3 epochs of no improvement
# )
