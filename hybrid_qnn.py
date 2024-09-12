import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import transforms
from medmnist import PneumoniaMNIST
import pennylane as qml
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Data transformation and normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the PneumoniaMNIST dataset
train_data = PneumoniaMNIST(split='train', transform=transform, download=True)
test_data = PneumoniaMNIST(split='test', transform=transform, download=True)

# Convert to torch tensors
X_train = train_data.imgs / 255.0
y_train = train_data.labels.flatten()
X_test = test_data.imgs / 255.0
y_test = test_data.labels.flatten()

# Expand dimensions to match the expected input shape of the neural network
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Convert to torch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)

# Permute the dimensions to match the input requirements of PyTorch Conv2D (batch, channels, height, width)
X_train_torch = X_train_torch.permute(0, 3, 1, 2)
X_test_torch = X_test_torch.permute(0, 3, 1, 2)

# Split the training data into training and validation sets
X_train_torch, X_val_torch, y_train_torch, y_val_torch = train_test_split(X_train_torch, y_train_torch, test_size=0.2, random_state=42)

# Compute class weights for imbalanced classes
unique_classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

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
        x = x.reshape(-1, 64 * 7 * 7)
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
        x = x.reshape(-1, 64 * 7 * 7)
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

# Prepare DataLoaders (with weighted sampling for handling class imbalance)
train_dataset = TensorDataset(X_train_torch, y_train_torch)
val_dataset = TensorDataset(X_val_torch, y_val_torch)
test_dataset = TensorDataset(X_test_torch, y_test_torch)

# WeightedRandomSampler for handling class imbalance
sample_weights = [class_weights_tensor[int(label)] for label in y_train_torch]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training function
def train_model(model, train_loader, val_loader, num_epochs=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
    
    return model, train_losses, val_losses

# Train the hybrid model
hybrid_model, hybrid_train_losses, hybrid_val_losses = train_model(hybrid_model, train_loader, val_loader)

# Save the trained hybrid model
torch.save(hybrid_model.state_dict(), 'hybrid_model.pth')

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test).squeeze()
        predictions = (outputs >= 0.5).float()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
    # Save confusion matrix plot
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('results/hybrid_confusion_matrix.png')
    plt.close()

    return accuracy, precision, recall, f1

# Evaluate the hybrid model on the test set
hybrid_accuracy, hybrid_precision, hybrid_recall, hybrid_f1 = evaluate_model(hybrid_model, X_test_torch, y_test_torch)

# Save evaluation results
results = {
    "accuracy": hybrid_accuracy,
    "precision": hybrid_precision,
    "recall": hybrid_recall,
    "f1_score": hybrid_f1
}
np.save('results/hybrid_model_performance.npy', results)

# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(hybrid_train_losses, label='Train Loss')
plt.plot(hybrid_val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Losses')
plt.savefig('results/hybrid_loss_curves.png')
plt.close()
