import torch
import torch.nn as nn

# ---------------------------
# CNN Model Definition
# ---------------------------
class DrowsinessDetectionCNN(nn.Module):
    def __init__(self):
        super(DrowsinessDetectionCNN, self).__init__()
        
        # Convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # Input: RGB image
        self.pool = nn.MaxPool2d(2, 2)                            # Halves spatial dimensions
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        # NOTE: Input must be 128x128x3 for this to match 128*16*16 = 32768 features
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 classes (adjust if your dataset changes)
        
        # Activation & regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 128x128 -> 64x64
        x = self.pool(self.relu(self.conv2(x)))  # 64x64 -> 32x32
        x = self.pool(self.relu(self.conv3(x)))  # 32x32 -> 16x16
        x = x.view(x.size(0), -1)                # Flatten to (batch, 32768)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------
# Training code (only runs if executed directly)
# ---------------------------
if __name__ == "__main__":
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import copy
    import json

    train_accuracies = []
    val_accuracies = []
    train_losses = []

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------------------------
    # Data transforms (ensure all images are 128x128 RGB)
    # ---------------------------
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # ---------------------------
    # Load datasets
    # ---------------------------
    train_dataset = datasets.ImageFolder('dataset_new/train', transform=train_transform)
    valid_dataset = datasets.ImageFolder('dataset_new/valid', transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # ---------------------------
    # Model, Loss, Optimizer
    # ---------------------------
    model = DrowsinessDetectionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---------------------------
    # Training loop with Early Stopping
    # ---------------------------
    epochs = 50
    patience = 5
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

        # Early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            print("✅ New best model found. Saving...")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                break

        # Save metrics
        results = {
            "train_acc": train_accuracies,
            "val_acc": val_accuracies,
            "train_loss": train_losses
        }
        with open("training_results.json", "w") as f:
            json.dump(results, f)

    # Save best model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'models/drowsiness_model.pth')
    print(f"Best model saved with Val Acc: {best_val_acc:.2f}%")
    print("Training metrics saved to training_results.json")
