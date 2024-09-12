import os
from PIL import Image
import numpy as np
from medmnist import PneumoniaMNIST

# Create a directory to save the images
os.makedirs('images', exist_ok=True)

# Load the PneumoniaMNIST dataset
data = PneumoniaMNIST(split='train', download=True)

# Separate images by label
label_0_images = []
label_1_images = []

for img, label in zip(data.imgs, data.labels):
    if label == 0 and len(label_0_images) < 10:
        label_0_images.append(img)
    elif label == 1 and len(label_1_images) < 10:
        label_1_images.append(img)
    if len(label_0_images) == 10 and len(label_1_images) == 10:
        break

# Save images with label 0
for i, img in enumerate(label_0_images):
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(f'images/label_0_image_{i+1}.png')

# Save images with label 1
for i, img in enumerate(label_1_images):
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(f'images/label_1_image_{i+1}.png')

print("Images saved in the 'images' directory.")
