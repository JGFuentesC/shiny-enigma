#!/usr/bin/env python3
"""
Train a CNN from scratch (VGG-style) on the labeled anime face dataset.
Uses the labels from step 4 (extrapolation with SVM).
"""

import os, sys, json, csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("/workspace/ssd/solucion_practica/data/anime_faces/images")
LABELS_PATH = Path("/workspace/ssd/solucion_practica/cluster/04_predicciones_completas.csv")
OUTPUT_DIR = Path("/workspace/ssd/solucion_practica/cnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 7
IMAGE_SIZE = 64


class AnimeFaceDataset(Dataset):
    def __init__(self, filenames, labels, label_to_idx, transform=None):
        self.filenames = filenames
        self.labels = [label_to_idx[l] for l in labels]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = DATA_DIR / self.filenames[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


class VGGStyleCNN(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_labels():
    filenames, labels = [], []
    with open(LABELS_PATH) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            filenames.append(row[0])
            labels.append(row[1])
    label_names = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    print(f"Loaded {len(filenames)} labeled images, {len(label_names)} classes: {label_names}")
    return filenames, labels, label_names, label_to_idx, idx_to_label


def split_data(filenames, labels, train_ratio=0.7, val_ratio=0.15, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(filenames))
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_files = [filenames[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_files = [filenames[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_files = [filenames[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    return train_files, train_labels, val_files, val_labels, test_files, test_labels


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_targets


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Val")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss")

    ax2.plot(train_accs, label="Train")
    ax2.plot(val_accs, label="Val")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Accuracy")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_accuracy.png", dpi=150)
    plt.close()
    print("Saved loss_accuracy.png")


def plot_confusion_matrix(all_targets, all_preds, label_names):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")


def main():
    filenames, labels, label_names, label_to_idx, idx_to_label = load_labels()

    train_files, train_labels, val_files, val_labels, test_files, test_labels = split_data(
        filenames, labels
    )

    train_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        T.RandomRotation(5),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    eval_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_ds = AnimeFaceDataset(train_files, train_labels, label_to_idx, train_transform)
    val_ds = AnimeFaceDataset(val_files, val_labels, label_to_idx, eval_transform)
    test_ds = AnimeFaceDataset(test_files, test_labels, label_to_idx, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = VGGStyleCNN(num_classes=len(label_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting training for {EPOCHS} epochs...")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
            patience_counter = 0
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model for test evaluation
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth"))
    test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nTest results: Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")

    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(test_targets, test_preds, label_names)

    # Save results
    with open(OUTPUT_DIR / "resultados.json", "w") as f:
        json.dump({
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "best_val_accuracy": float(best_val_acc),
            "num_classes": len(label_names),
            "classes": label_names,
            "epochs_trained": len(train_losses),
        }, f, indent=2)

    # Also save per-class accuracy
    from sklearn.metrics import classification_report
    report = classification_report(test_targets, test_preds,
                                   target_names=label_names, digits=4, output_dict=True)
    with open(OUTPUT_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nTraining complete!")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
