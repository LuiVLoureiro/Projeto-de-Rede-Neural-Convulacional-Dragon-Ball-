import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATASET_DIR = "data/processed"
MODEL_PATH = "models/goku_e_vegeta.pth"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=train_transforms)

val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])

val_ds.dataset.transform = val_transforms

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_dataset.classes)
print("Classes encontradas:", train_dataset.classes)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# --- Fase 1: Feature Extraction ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float()
        optimizer.zero_grad()
        outputs = model(images).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.sigmoid(outputs) >= 0.5
        correct += (preds == labels.byte()).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100.0 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float()
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) >= 0.5
            val_correct += (preds == labels.byte()).sum().item()
            val_total += labels.size(0)

    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_epoch_acc = 100.0 * val_correct / val_total
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% "
          f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Evolução do Loss")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.title("Evolução da Acurácia")
plt.xlabel("Épocas")
plt.ylabel("Acurácia (%)")
plt.legend()

plt.tight_layout()
plt.show()

# --- Fase 2: Fine-Tuning ---
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params_to_update, lr=1e-5)

FINE_TUNE_EPOCHS = 5
for epoch in range(FINE_TUNE_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float()
        optimizer.zero_grad()
        outputs = model(images).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.sigmoid(outputs) >= 0.5
        correct += (preds == labels.byte()).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100.0 * correct / total

    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float()
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) >= 0.5
            val_correct += (preds == labels.byte()).sum().item()
            val_total += labels.size(0)

    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_epoch_acc = 100.0 * val_correct / val_total

    print(f"[Fine-Tuning] Epoch [{epoch+1}/{FINE_TUNE_EPOCHS}] "
          f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% "
          f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%")

# --- Avaliação Final ---
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images).view(-1)
        preds = torch.sigmoid(outputs) >= 0.5
        all_preds.extend(preds.cpu().numpy().astype(int))
        all_labels.extend(labels.numpy().astype(int))

cm = confusion_matrix(all_labels, all_preds)
print("Matriz de Confusão:\n", cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - ResNet50")
plt.show()

print("Relatório de Classificação:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# --- Salvar Modelo ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "class_to_idx": train_dataset.class_to_idx,
}
torch.save(checkpoint, MODEL_PATH)
print(f"Modelo salvo em {MODEL_PATH}")

# --- Visualização de Classificações ---
def classify_image(image_path, model, transform, class_names):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_transformed = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_transformed).view(-1)
        pred = torch.sigmoid(output).item()
    pred_label = class_names[1] if pred >= 0.5 else class_names[0]
    return img, pred_label


test_images = []
for class_name in train_dataset.classes:
    class_dir = os.path.join(DATASET_DIR, class_name)
    images_in_class = [f for f in os.listdir(class_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if len(images_in_class) >= 2:
        selected = random.sample(images_in_class, 2)
        test_images.extend([os.path.join(class_dir, img) for img in selected])

plt.figure(figsize=(10, 10))
for i, image_path in enumerate(test_images):
    img, label = classify_image(image_path, model, val_transforms, train_dataset.classes)
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Classificação: {label}")
plt.tight_layout()
plt.show()
