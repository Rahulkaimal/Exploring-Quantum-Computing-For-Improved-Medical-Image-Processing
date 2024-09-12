import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pennylane as qml
import streamlit as st
from PIL import Image
from torchvision import transforms
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the saved model
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 4)
        self.dropout = nn.Dropout(0.5)

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
        self.fc2 = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.reshape(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        batch_size = x.shape[0]
        qlayer_outputs = []
        for i in range(batch_size):
            qlayer_output = self.qlayer(x[i])
            qlayer_outputs.append(qlayer_output)
        x = torch.stack(qlayer_outputs)
        x = self.fc2(x)
        return torch.sigmoid(x)

model = HybridModel()
model.load_state_dict(torch.load('hybrid_model.pth', map_location=torch.device('cpu')))
model.eval()

# Image transformation for preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Streamlit app
st.set_page_config(page_title="Pneumonia Detection Dashboard", layout="wide")

st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Choose the action", ["Upload X-ray", "About"])

if app_mode == "Upload X-ray":
    st.title("Pneumonia Detection")

    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type="png")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Chest X-ray', use_column_width=True)

        # Process the image and make prediction
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        output = model(image_tensor).item()

        # Display the prediction
        if output > 0.5:
            prediction = 'Pneumonia Detected'
        else:
            prediction = 'No Pneumonia'
        
        st.write(f"Prediction: **{prediction}** (Confidence: {output:.2f})")

        # Create a simple 3D plot as an example
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.random.rand(100)
        y = np.random.rand(100)
        z = np.random.rand(100)
        ax.scatter(x, y, z, c='r', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        st.pyplot(fig)

elif app_mode == "About":
    st.title("About the Pneumonia Detection Dashboard")
    st.write("""
    This application is designed to assist with the detection of pneumonia in chest X-ray images.
    
    **Features:**
    - Upload an X-ray image and receive a prediction (Pneumonia detected or not).
    - View predictions in a clear and intuitive 3D plot.

    **How it works:**
    - The model used in this app is a hybrid quantum-classical neural network.
    - The model was trained on the PneumoniaMNIST dataset.
    """)

if __name__ == '__main__':
    pass

