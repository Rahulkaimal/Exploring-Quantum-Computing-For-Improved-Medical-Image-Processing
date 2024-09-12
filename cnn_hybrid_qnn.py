import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Load the PneumoniaMNIST dataset
from medmnist import PneumoniaMNIST

data_train = PneumoniaMNIST(split='train', download=True)
data_test = PneumoniaMNIST(split='test', download=True)

X_train = data_train.imgs / 255.0
y_train = data_train.labels.flatten()
X_test = data_test.imgs / 255.0
y_test = data_test.labels.flatten()

# Expand dimensions to match the expected input shape of the neural network
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Reshape the data to fit the PyTorch format
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

# Convert to torch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)

# Apply Random Undersampling to balance the classes
undersampler = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = undersampler.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
X_train_res = X_train_res.reshape(-1, 1, 28, 28)

# Convert the resampled dataset back to torch tensors
X_train_torch_res = torch.tensor(X_train_res, dtype=torch.float32)
y_train_torch_res = torch.tensor(y_train_res, dtype=torch.float32)

# Split the training data into training and validation sets
X_train_torch, X_val_torch, y_train_torch, y_val_torch = train_test_split(X_train_torch_res, y_train_torch_res, test_size=0.2, random_state=42)

# Save label distributions before and after undersampling
def save_label_distribution(y_train, y_train_res, path):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.countplot(y_train)
    plt.title('Before Undersampling')
    
    plt.subplot(1, 2, 2)
    sns.countplot(y_train_res)
    plt.title('After Undersampling')
    
    plt.savefig(path)
    plt.close()

save_label_distribution(y_train, y_train_res, "results/label_distribution.png")

# Define the Classical CNN Model
class ClassicalCNN(nn.Module):
    def __init__(self):
        super(ClassicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

# Define the Hybrid Quantum-Classical Model
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        # Classical CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 4)  # Adjust to output 4 features
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
        # Quantum layer
        n_qubits = 4
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface='torch')
        def quantum_circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RX(inputs[i], wires=i)
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (2, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Final output layer
        self.fc2 = nn.Linear(n_qubits, 1)  # Adjusted to match the quantum layer output

    def forward(self, x):
        # Classical CNN forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Quantum layer forward pass (processing each item in the batch independently)
        batch_size = x.shape[0]
        qlayer_outputs = []
        for i in range(batch_size):
            qlayer_output = self.qlayer(x[i])
            qlayer_outputs.append(qlayer_output)
        x = torch.stack(qlayer_outputs)
        
        # Output layer
        x = self.fc2(x)
        return torch.sigmoid(x)

# Initialize the models
classical_model = ClassicalCNN()
hybrid_model = HybridModel()

# Prepare DataLoaders
train_dataset = TensorDataset(X_train_torch, y_train_torch)
val_dataset = TensorDataset(X_val_torch, y_val_torch)
test_dataset = TensorDataset(X_test_torch, y_test_torch)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Function to train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=15, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Function to evaluate the model and save results
def evaluate_model(model, X_test, y_test, results_dir):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test).squeeze()
        predictions = (outputs >= 0.5).float()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "predictions": predictions
    }

    np.save(os.path.join(results_dir, "model_performance.npy"), results)

    # Save confusion matrix plot
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    return results

# Train the models
train_model(classical_model, X_train_torch, y_train_torch, X_val_torch, y_val_torch)
train_model(hybrid_model, X_train_torch, y_train_torch, X_val_torch, y_val_torch)

# Evaluate the models
classical_results_dir = "results/classical_model"
os.makedirs(classical_results_dir, exist_ok=True)
classical_results = evaluate_model(classical_model, X_test_torch, y_test_torch, classical_results_dir)

hybrid_results_dir = "results/hybrid_model"
os.makedirs(hybrid_results_dir, exist_ok=True)
hybrid_results = evaluate_model(hybrid_model, X_test_torch, y_test_torch, hybrid_results_dir)

# Save models
torch.save(classical_model.state_dict(), os.path.join(classical_results_dir, "classical_model.pth"))
torch.save(hybrid_model.state_dict(), os.path.join(hybrid_results_dir, "hybrid_model.pth"))
