#!/usr/bin/env python3
"""
Pipeline completa: clustering + extrapolacion + CNN.
Total: < 3 minutos.
"""

import os, sys, json, random, time
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# -- Config --
DATA_DIR = Path("/workspace/ssd/solucion_practica/data/anime_faces/images")
OUTPUT_DIR = Path("/workspace/ssd/solucion_practica/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for d in ["cluster", "cnn"]:
    (OUTPUT_DIR / d).mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_SIZE = 200
N_CLUSTERS = 4
RANDOM_SEED = 42
BATCH_SIZE_EXTRACT = 512
BATCH_SIZE_CNN = 128
EPOCHS = 20
PATIENCE = 5
LR = 1e-3
IMG_SIZE = 64

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT_DIR}")
print()


# ============================================================
# STEP 1: Feature extraction
# ============================================================

def get_resnet18():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.to(DEVICE)
    model.eval()
    return model

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def extract_features(model, paths, desc="Extracting features"):
    feats, fnames = [], []
    it = range(0, len(paths), BATCH_SIZE_EXTRACT)
    for i in tqdm(it, desc=desc):
        batch_paths = paths[i:i + BATCH_SIZE_EXTRACT]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))
        batch = torch.stack(tensors).to(DEVICE)
        feats.append(model(batch).cpu().numpy())
        fnames.extend([p.name for p in batch_paths])
    return np.concatenate(feats), fnames


# ============================================================
# STEP 2: Clustering
# ============================================================

print("=" * 50)
print("STEP 1: Feature extraction & clustering")
print("=" * 50)

all_paths = sorted(DATA_DIR.glob("*.jpg"))
print(f"Total images: {len(all_paths)}")

model_resnet = get_resnet18()

sample_paths = random.sample(all_paths, SAMPLE_SIZE)
sample_feats, sample_fnames = extract_features(model_resnet, sample_paths, "Sample features")
print(f"  Sample features shape: {sample_feats.shape}")

km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
sample_labels = km.fit_predict(sample_feats)
sil = silhouette_score(sample_feats, sample_labels)
print(f"  K-means done | Silhouette: {sil:.4f}")

label_map = {i: f"Cluster_{i}" for i in range(N_CLUSTERS)}
cluster_counts = Counter(sample_labels)
for c in sorted(cluster_counts):
    print(f"  {label_map[c]}: {cluster_counts[c]} images")

# -- Cluster grid --
fig, axes = plt.subplots(N_CLUSTERS, 8, figsize=(16, 2 * N_CLUSTERS))
for row in range(N_CLUSTERS):
    idxs_in_cluster = [i for i, l in enumerate(sample_labels) if l == row]
    selected = random.sample(idxs_in_cluster, min(8, len(idxs_in_cluster)))
    for col, idx in enumerate(selected):
        img = Image.open(sample_paths[idx])
        axes[row, col].imshow(img)
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(f"{label_map[row]}", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cluster" / "grid.png", dpi=150)
plt.close()
print(f"  Saved grid.png")


# ============================================================
# STEP 3: SVM extrapolation
# ============================================================

print("\n" + "=" * 50)
print("STEP 2: SVM extrapolation to full dataset")
print("=" * 50)

t_start = time.time()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(sample_feats)
svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_SEED)
svm.fit(X_scaled, sample_labels)
train_pred = svm.predict(X_scaled)
train_acc = (train_pred == sample_labels).mean()
print(f"  SVM train accuracy (on sample): {train_acc:.4f}")
print(f"  SVM trained in {time.time() - t_start:.1f}s")

# Full dataset features
all_feats, all_fnames = extract_features(model_resnet, all_paths, "Full dataset features")
print(f"  Full features shape: {all_feats.shape}")

all_scaled = scaler.transform(all_feats)
all_preds = svm.predict(all_scaled)
print("  All images classified")

final_counts = Counter(all_preds)
for c in sorted(final_counts):
    print(f"  {label_map[c]}: {final_counts[c]} ({final_counts[c]/len(all_preds)*100:.1f}%)")

# Save labels
label_data = [{"filename": f, "label": label_map[int(l)]}
              for f, l in zip(all_fnames, all_preds)]
with open(OUTPUT_DIR / "predicciones.json", "w") as f:
    json.dump(label_data, f, indent=2)

import csv
with open(OUTPUT_DIR / "predicciones.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "label"])
    for fname, label in zip(all_fnames, all_preds):
        w.writerow([fname, label_map[int(label)]])

# Distribution plot
fig, ax = plt.subplots(figsize=(8, 4))
colors = plt.cm.Set2(np.linspace(0, 1, N_CLUSTERS))
bars = ax.bar([label_map[c] for c in range(N_CLUSTERS)],
              [final_counts.get(c, 0) for c in range(N_CLUSTERS)],
              color=colors)
ax.set_title("Distribución de categorías (63k imágenes)")
ax.set_ylabel("Número de imágenes")
for bar, val in zip(bars, [final_counts.get(c, 0) for c in range(N_CLUSTERS)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            str(val), ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cluster" / "distribucion.png", dpi=150)
plt.close()
print(f"  Saved distribucion.png")


# ============================================================
# STEP 4: CNN training from scratch
# ============================================================

print("\n" + "=" * 50)
print("STEP 3: CNN training from scratch")
print("=" * 50)

class AnimeDataset(Dataset):
    def __init__(self, fnames, labels, label_to_idx, transform):
        self.fnames = fnames
        self.labels = [label_to_idx[l] for l in labels]
        self.transform = transform
    def __len__(self):
        return len(self.fnames)
    def __getitem__(self, idx):
        img = Image.open(DATA_DIR / self.fnames[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(256 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Split data
all_labels_str = [label_map[int(l)] for l in all_preds]
label_names = sorted(set(all_labels_str))
label_to_idx = {l: i for i, l in enumerate(label_names)}
idx_to_label = {i: l for l, i in label_to_idx.items()}

indices = np.random.permutation(len(all_fnames))
n_train = int(0.7 * len(indices))
n_val = int(0.15 * len(indices))
train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.1, 0.1, 0.1, 0.05),
    T.RandomRotation(5),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

train_ds = AnimeDataset([all_fnames[i] for i in train_idx],
                        [all_labels_str[i] for i in train_idx], label_to_idx, train_transform)
val_ds = AnimeDataset([all_fnames[i] for i in val_idx],
                      [all_labels_str[i] for i in val_idx], label_to_idx, eval_transform)
test_ds = AnimeDataset([all_fnames[i] for i in test_idx],
                       [all_labels_str[i] for i in test_idx], label_to_idx, eval_transform)

train_loader = DataLoader(train_ds, BATCH_SIZE_CNN, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, BATCH_SIZE_CNN, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, BATCH_SIZE_CNN, shuffle=False, num_workers=4, pin_memory=True)

model = CNN(num_classes=len(label_names)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

def train_epoch():
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0
patience_counter = 0

pbar = tqdm(range(1, EPOCHS + 1), desc="Training CNN")
for epoch in pbar:
    t_loss, t_acc = train_epoch()
    v_loss, v_acc = eval_epoch(val_loader)
    scheduler.step(v_loss)

    train_losses.append(t_loss)
    val_losses.append(v_loss)
    train_accs.append(t_acc)
    val_accs.append(v_acc)

    pbar.set_postfix({
        "loss": f"{t_loss:.4f}/{v_loss:.4f}",
        "acc": f"{t_acc:.4f}/{v_acc:.4f}",
        "best": f"{best_val_acc:.4f}"
    })

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), OUTPUT_DIR / "cnn" / "best_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

# Test
model.load_state_dict(torch.load(OUTPUT_DIR / "cnn" / "best_model.pth"))
test_loss, test_acc = eval_epoch(test_loader)
print(f"\n  Test accuracy: {test_acc:.4f}")

# Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label="Train")
ax1.plot(val_losses, label="Val")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.set_title("Loss")
ax2.plot(train_accs, label="Train")
ax2.plot(val_accs, label="Val")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.set_title("Accuracy")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cnn" / "training_curves.png", dpi=150)
plt.close()
print(f"  Saved training_curves.png")

# Confusion matrix on test
model.eval()
all_preds_test, all_targets_test = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        out = model(x)
        all_preds_test.extend(out.argmax(1).cpu().numpy())
        all_targets_test.extend(y.numpy())

cm = confusion_matrix(all_targets_test, all_preds_test)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(label_names)))
ax.set_yticks(range(len(label_names)))
ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(label_names, fontsize=8)
ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
for i in range(len(label_names)):
    for j in range(len(label_names)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cnn" / "confusion_matrix.png", dpi=150)
plt.close()
print(f"  Saved confusion_matrix.png")

# Save results
results = {
    "silhouette_score": float(sil),
    "cluster_distribution": {label_map[c]: int(cluster_counts[c]) for c in range(N_CLUSTERS)},
    "final_distribution": {label_map[c]: int(final_counts.get(c, 0)) for c in range(N_CLUSTERS)},
    "svm_train_accuracy": float(train_acc),
    "cnn_test_accuracy": float(test_acc),
    "cnn_best_val_accuracy": float(best_val_acc),
    "cnn_epochs_trained": len(train_losses),
}
with open(OUTPUT_DIR / "resultados.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 50)
print("PIPELINE COMPLETE")
print("=" * 50)
print(json.dumps(results, indent=2))
