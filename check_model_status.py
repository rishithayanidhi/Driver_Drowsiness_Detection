import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DrowsinessDetectionCNN

# ---------------------------
# Device config
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Image transform (force RGB)
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # force RGB
    transforms.ToTensor()
])

# ---------------------------
# Load test dataset
# ---------------------------
test_dataset = datasets.ImageFolder(root="dataset_new/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------------------
# Load model
# ---------------------------
model = DrowsinessDetectionCNN().to(device)
model.load_state_dict(torch.load("models/drowsiness_model.pth", map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

# ---------------------------
# Evaluation
# ---------------------------
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_loss = test_loss / len(test_loader)
accuracy = 100 * correct / total

print(f"\n📊 Test Loss: {avg_loss:.4f}")
print(f"✅ Test Accuracy: {accuracy:.2f}%")
