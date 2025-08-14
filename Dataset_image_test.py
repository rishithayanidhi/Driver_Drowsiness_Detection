import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import DrowsinessDetectionCNN

# ---------------------------
# Config
# ---------------------------
model_path = "models/drowsiness_model.pth"  # Path to your saved model
test_data_path = "dataset_new/test"             # Path to your test set

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Dataset & Loader
dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load Model
model = DrowsinessDetectionCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Class names
class_names = dataset.classes
print("Class labels:", class_names)

# ---------------------------
# Visualization Loop
# ---------------------------
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

with torch.no_grad():
    for i, (image, label) in enumerate(loader):
        image, label = image.to(device), label.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

        print(f"Predicted: {class_names[predicted.item()]}, Actual: {class_names[label.item()]}")
        imshow(image.cpu().squeeze())

        if i == 9:  # Show first 10 samples
            break
