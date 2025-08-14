import json
import matplotlib.pyplot as plt
import subprocess

# Step 1: Run model.py (and let it save training_results.json)
subprocess.run(["python", "model.py"], text=True)

# Step 2: Read the saved results
with open("training_results.json", "r") as f:
    results = json.load(f)

train_acc = results["train_acc"]
val_acc = results["val_acc"]

epochs = list(range(1, len(train_acc) + 1))

# Step 3: Plot accuracy
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
plt.plot(epochs, val_acc, label="Validation Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training & Validation Accuracy Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_accuracy_graph.png")
plt.show()

# Step 4: Plot loss
if "train_loss" in results:
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, results["train_loss"], label="Train Loss", marker='o', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_graph.png")
    plt.show()
