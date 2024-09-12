# Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from medmnist import PneumoniaMNIST

# Load PneumoniaMNIST dataset
data = PneumoniaMNIST(split='train', download=True)
test_data = PneumoniaMNIST(split='test', download=True)

# Convert to numpy arrays
X_train = data.imgs
y_train = data.labels.flatten()
X_test = test_data.imgs
y_test = test_data.labels.flatten()

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Expand dimensions to match the expected input shape of the neural network
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Fit the data generator to the training data
datagen.fit(X_train)

# Define the generators for training and validation sets
train_generator = datagen.flow(X_train, y_train, batch_size=32)
val_generator = ImageDataGenerator().flow(X_val, y_val, batch_size=32)

# EDA: basic information about the dataset
print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Test data shape:", X_test.shape)
print("Training labels distribution:", np.bincount(y_train))
print("Validation labels distribution:", np.bincount(y_val))
print("Test labels distribution:", np.bincount(y_test))

# Check for balance/imbalance
def check_balance(label_counts):
    total = sum(label_counts)
    balance = [count / total for count in label_counts]
    return balance

train_balance = check_balance(np.bincount(y_train))
val_balance = check_balance(np.bincount(y_val))
test_balance = check_balance(np.bincount(y_test))

print("Training set balance:", train_balance)
print("Validation set balance:", val_balance)
print("Test set balance:", test_balance)

# EDA: Plot some sample images 
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i in np.arange(0, 10):
    axes[i].imshow(X_train[i].reshape(28, 28), cmap='gray')
    axes[i].set_title(f'Label: {y_train[i]}')
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.5)
plt.savefig('sample_images.png')

# EDA: Plot
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

train_counts = np.bincount(y_train)
val_counts = np.bincount(y_val)
test_counts = np.bincount(y_test)

ax[0].bar(np.arange(len(train_counts)), train_counts, color='blue')
ax[0].set_title('Training Labels Distribution')
ax[0].set_xlabel('Label')
ax[0].set_ylabel('Frequency')
ax[0].set_xticks([0, 1])
for i, v in enumerate(train_counts):
    ax[0].text(i, v + 20, str(v), ha='center')
ax[0].legend(['Training'], loc='upper left', bbox_to_anchor=(1, 1))

ax[1].bar(np.arange(len(val_counts)), val_counts, color='green')
ax[1].set_title('Validation Labels Distribution')
ax[1].set_xlabel('Label')
ax[1].set_ylabel('Frequency')
ax[1].set_xticks([0, 1])
for i, v in enumerate(val_counts):
    ax[1].text(i, v + 10, str(v), ha='center')
ax[1].legend(['Validation'], loc='upper left', bbox_to_anchor=(1, 1))

ax[2].bar(np.arange(len(test_counts)), test_counts, color='red')
ax[2].set_title('Test Labels Distribution')
ax[2].set_xlabel('Label')
ax[2].set_ylabel('Frequency')
ax[2].set_xticks([0, 1])
for i, v in enumerate(test_counts):
    ax[2].text(i, v + 5, str(v), ha='center')
ax[2].legend(['Test'], loc='upper left', bbox_to_anchor=(1, 1))

plt.savefig('label_distribution.png')

mean_image = np.mean(X_train, axis=0)
std_image = np.std(X_train, axis=0)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

im1 = ax[0].imshow(mean_image.squeeze(), cmap='gray')
ax[0].set_title('Mean Image')
fig.colorbar(im1, ax=ax[0], orientation='vertical')
ax[0].legend(['Mean'], loc='upper left', bbox_to_anchor=(1, 1))

im2 = ax[1].imshow(std_image.squeeze(), cmap='gray')
ax[1].set_title('Standard Deviation Image')
fig.colorbar(im2, ax=ax[1], orientation='vertical')
ax[1].legend(['Std Dev'], loc='upper left', bbox_to_anchor=(1, 1))

plt.savefig('mean_std_images.png')

print("EDA and data preprocessing completed.")
