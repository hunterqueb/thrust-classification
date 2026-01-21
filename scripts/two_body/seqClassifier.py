import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import control as ct
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml.utils import printModelParmSize, getDevice
from qutils.tictoc import timer

plt.switch_backend('WebAgg')

import argparse

# Command line arguments
def get_args():
    parser = argparse.ArgumentParser(
        description="Configure and launch time series sequence classification training run."
    )

    parser.add_argument(
        "--forceConst", type=float, default=1.0,
        help="Force constant to apply to random systems (default: 1.0)"
    )

    parser.add_argument(
        "--numRandSys", type=int, default=10000,
        help="Number of randomized systems to generate (default: 10000)"
    )

    parser.add_argument("--no-plot", dest="plot", action="store_false",
                        help="Disable plotting (enabled by default)")
    parser.set_defaults(plot=True)

    return parser.parse_args()

args = get_args()

# Use the parsed values directly
forceConst = args.forceConst
numRandSys = args.numRandSys
plotOn = args.plot

print(f"Force Constant        : {forceConst}")
print(f"Number of Rand Systems: {numRandSys}")
print(f"Plotting Enabled?     : {plotOn}")

# forceConst = 1
# plotOn = False
# numRandSys = 1000


# Parameters
num_features = 2     # toy example, 2 features
num_classes = 1    # binary classification
hiddenSize = 32

sampling_rate = 100  # Hz
duration = 10        # seconds
seq_len = sampling_rate * duration

t = np.linspace(0, duration, seq_len)

device = getDevice()
rng = np.random.default_rng(seed=1) # Seed for reproducibility

# Generate synthetic data
forces_all = np.zeros((numRandSys, seq_len), dtype=np.float32)

def generate_batch_forced_oscillators(num_systems=numRandSys):
    features_all = np.zeros((num_systems, seq_len, 2), dtype=np.float32)
    labels_all = np.zeros((num_systems, seq_len), dtype=np.float32)

    for n in range(num_systems):
        k = rng.random(); m = rng.random()
        
        F0 = forceConst * rng.random()
        c = 0.1 # consider damping for now

        x0 = [rng.random(),rng.random()]

        A = np.array(([0,1],[-k/m,-c/m]))

        B = np.array([0,1])

        C = np.array(([1,0],[0,0]))
        
        D = 0
        # Random force duration in (0, 1] seconds
        force_duration = rng.uniform(low=0.1, high=1.0)

        # Random start time such that force doesn't go past simulation end
        max_start_time = duration - force_duration
        force_start_time = rng.uniform(low=0.0, high=max_start_time)
        force_end_time = force_start_time + force_duration

        # External force signal
        f = np.zeros(seq_len)
        f[(t >= force_start_time) & (t < force_end_time)] = F0

        # Integrate using ct
        sys = ct.StateSpace(A,B,C,D)

        resultsForced = ct.forced_response(sys,t,f,x0)
        
        # Store features: [x, v]
        features_all[n, :, :] = resultsForced.x.T

        # Store label = 1 during force application
        labels_all[n, (t >= force_start_time) & (t < force_end_time)] = 1.0

        # Store forces for visualization
        forces_all[n, :] = f

    # plt.figure()
    # plt.plot(t,resultsForced.x.T, label="Forced Response")
    # plt.plot(t,f, label="Forcing Function")
    # plt.grid()
    # plt.title("Random Sample of Forced Response")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.legend(["Position",'Velocity',"Forced by [0,{}] N Linear Force".format(forceConst)])
    # plt.show()
    return (
        torch.tensor(features_all, dtype=torch.float32),  # shape [N, T, 2]
        torch.tensor(labels_all, dtype=torch.float32)     # shape [N, T]
    )

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

# Training loop
def train_model(model, train_loader, val_loader, num_epochs=10):
    # uses 3 new metrics
    # 1. Precision
    # true positives / (true positives + false positives)
    # >= 0.8 -- few false alarms
    # 2. Recall
    # true positives / (true positives + false negatives)
    # >= 0.8 -- rarely misses true events
    # 3. F1 Score
    # 2 * (precision * recall) / (precision + recall)
    # >= 0.8 -- well balanced detection

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    schedulerPatience = 3
    best_loss = float('inf')
    ESpatience = schedulerPatience * 2
    counter = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=schedulerPatience
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(sequences)               # [B, T]
            loss = criterion(logits, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")

        # -------------------
        # Validation Metrics
        # -------------------
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                logits = model(sequences)               # [B, T]
                probs = torch.sigmoid(logits)           # [B, T]
                preds = (probs >= 0.5).int()             # [B, T]

                all_preds.append(preds.cpu().flatten())
                all_targets.append(labels.cpu().flatten().int())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)

        print(f"Validation Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

        # -------------------
        # Validation Loss
        # -------------------
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device).float()
                logits = model(sequences)
                loss = criterion(logits, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            # Optionally save the best model
        else:
            counter += 1
            if counter >= ESpatience:
                print("Early stopping")
                break
    cm = confusion_matrix(all_targets, all_preds)
    plotConfusionMatrix(cm)

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
                f_sample = val_force[global_index]
                break
        else:
            raise ValueError("Sample batch index out of range.")

        logits = model(x_sample).squeeze(0)           # [T]
        probs = torch.sigmoid(logits)                 # [T]
        predictions = (probs > 0.5).long()            # [T]


        x_sample = x_sample.cpu().numpy().squeeze(0)  # [T, D]
        y_sample = y_sample.cpu().numpy()            # [T]
        predictions = predictions.cpu().numpy()       # [T]

        plt.figure()
        plt.plot(t,x_sample, label="Forced Response")
        plt.plot(t,f_sample, label="Forcing Function")
        # Highlight regions where prediction == 1
        in_region = False
        start_idx = 0
        for i in range(seq_len):
            if predictions[i] == 1 and not in_region:
                start_idx = i
                in_region = True
            elif predictions[i] == 0 and in_region:
                plt.axvspan(t[start_idx], t[i], color='orange', alpha=0.3)
                in_region = False
        if in_region:
            plt.axvspan(t[start_idx], t[seq_len-1], color='orange', alpha=0.3)

        plt.grid()
        plt.title("Random Sample of Forced Response with Prediction")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend(["Position",'Velocity',"Forced by [0,{}] N Linear Force".format(forceConst),'Predicted Force Application Region'],loc='upper right')


        plt.figure(figsize=(12, 4))
        plt.plot(t,y_sample, label='True Label', linewidth=2)
        plt.plot(t,predictions, linestyle='--', label='Predicted Label', linewidth=1)
        plt.legend()
        plt.title(f"Per-Timestep Classification (Batch {sample_batch_idx}, Sample {sample_in_batch_idx})")
        plt.xlabel("Time Step")
        plt.ylabel("Label")
        plt.grid(True)
        plt.tight_layout()

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


# Run the pipeline
timeToGenData = timer()
x, y = generate_batch_forced_oscillators()
print("Time to generate data: {:.2f} seconds".format(timeToGenData.tocVal()))

batchSize = 16

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15


total_samples = len(x)
train_end = int(train_ratio * total_samples)
val_end = int((train_ratio + val_ratio) * total_samples)

# Split the data
train_data = x[:train_end]
train_label = y[:train_end]

val_data = x[train_end:val_end]
val_label = y[train_end:val_end]
val_force = forces_all[train_end:val_end]

test_data = x[val_end:]
test_label = y[val_end:]

train_dataset = TensorDataset(train_data,train_label)
val_dataset = TensorDataset(val_data,val_label)
test_dataset = TensorDataset(test_data,test_label)

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

model = LSTMSequenceClassifier(input_dim=num_features, hidden_dim=hiddenSize, num_layers=1, num_classes=num_classes).to(device)
config = MambaConfig(d_model=num_features,n_layers = 2,expand_factor=hiddenSize//num_features,d_state=hiddenSize,d_conv=16,classifer=True)
mambaModel = MambaSequenceClassifier(config, input_size=num_features, hidden_size=hiddenSize, num_layers=1, num_classes=num_classes).to(device)

print("LSTM Model:")
train_model(model, train_loader,val_loader, num_epochs=1000)
plt.gcf()
plt.title("LSTM Confusion Matrix")
evaluate_model(model, val_loader,modelString="LSTM")
plt.gcf()
plt.title("LSTM Model Evaluation on Random Sample")
printModelParmSize(model)

print("\n")

print("Mamba Model:")
train_model(mambaModel, train_loader,val_loader, num_epochs=1000)
plt.gcf()
plt.title("Mamba Confusion Matrix")
evaluate_model(mambaModel, val_loader,modelString="Mamba")
plt.gcf()
plt.title("Mamba Model Evaluation on Random Sample")
printModelParmSize(mambaModel)

if plotOn:
    plt.show()
