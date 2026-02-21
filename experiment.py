import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

EPOCHS = 2
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
DEVICE = torch.device("cpu")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

print("Loading dataset from local folder...")
# We assume data is located in `data/train` and `data/test`
train_data_full = datasets.ImageFolder(root="data/train", transform=train_transform)
test_ds = datasets.ImageFolder(root="data/test", transform=eval_transform)
# Note: eval_transform should be used for val, but we simplify here by extracting 
# only 12 samples for speed
train_indices = list(range(min(12, len(train_data_full))))
val_indices = list(range(min(12, 24 if len(train_data_full) > 24 else len(train_data_full)))) # Take next 12 if available
test_indices = list(range(min(12, len(test_ds))))

train_ds = Subset(train_data_full, train_indices)
val_ds = Subset(datasets.ImageFolder(root="data/train", transform=eval_transform), val_indices)
test_ds = Subset(test_ds, test_indices)

def train_and_eval(lr, batch_size):
    print(f"\n--- Testing LR={lr}, Batch Size={batch_size} ---")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / max(1, len(train_loader))
        train_losses.append(train_loss)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        v_loss = val_loss / max(1, len(val_loader))
        val_losses.append(v_loss)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {v_loss:.4f}, Val Acc: {100*correct/total:.2f}%")

    val_acc = 100 * correct / total
    return model, val_acc, train_losses, val_losses

settings = [
    {"lr": 1e-3, "bs": 16},
    {"lr": 1e-4, "bs": 32},
    {"lr": 5e-4, "bs": 16}
]

best_acc = -1
best_setting = None
best_model = None

plt.figure(figsize=(10, 6))

results = []
for s in settings:
    model, acc, t_loss, v_loss = train_and_eval(s["lr"], s["bs"])
    plt.plot(range(1, EPOCHS+1), t_loss, marker='o', label=f"LR={s['lr']}, BS={s['bs']} (Train)")
    results.append((s, acc))
    if acc >= best_acc:
        best_acc = acc
        best_setting = s
        best_model = model

plt.title("Training Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("hyperparam_curves.png")
print("Saved hyperparam_curves.png")

print(f"\nBest Settings: LR={best_setting['lr']}, Batch Size={best_setting['bs']} with Val Acc={best_acc:.2f}%")

# Save analysis to file
with open("setup_analysis.md", "a") as f:
    f.write("\n\n## Hyperparameter Experiment Results\n")
    for s, acc in results:
         f.write(f"- LR={s['lr']}, Batch Size={s['bs']} -> Validation Accuracy: {acc:.2f}%\n")
    f.write(f"\n**Best Setting:** LR={best_setting['lr']}, Batch Size={best_setting['bs']}\n")

print("\nRunning Evaluation on Test Set with Best Model...")
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
best_model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = best_model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

overall_acc = 100 * (all_preds == all_labels).sum() / max(1, len(all_labels))
class7_idx = 7
mask7 = (all_labels == class7_idx)
c7_acc = 100 * (all_preds[mask7] == all_labels[mask7]).sum() / mask7.sum() if mask7.sum() > 0 else 0.0

print(f"Overall Test Accuracy: {overall_acc:.2f}%")
print(f"Class 7 Test Accuracy: {c7_acc:.2f}%")

with open("setup_analysis.md", "a") as f:
    f.write("\n## Best Setting Evaluation\n")
    f.write(f"- Overall Accuracy: {overall_acc:.2f}%\n")
    f.write(f"- Class 7 (Horse) Accuracy: {c7_acc:.2f}%\n")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"Confusion Matrix (Best Setup: LR={best_setting['lr']}, BS={best_setting['bs']})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_best_new.png")
print("Saved confusion_matrix_best_new.png")
