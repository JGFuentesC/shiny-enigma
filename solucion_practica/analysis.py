#!/usr/bin/env python3
"""
Post-hoc analysis: gemma4 cluster labeling, UMAP, inference grids, Grad-CAM.
Generates all figures for the professional report.
"""

import os, sys, json, random, base64, io, asyncio, aiohttp
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image
from sklearn.preprocessing import StandardScaler
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch

from tqdm import tqdm

# ---------------------------------------------------------------------------
OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
MODEL = "gemma4:e4b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

DATA_DIR = Path("/workspace/ssd/solucion_practica/data/anime_faces/images")
MODEL_PATH = Path("/workspace/ssd/solucion_practica/output/cnn/best_model.pth")
PRED_PATH = Path("/workspace/ssd/solucion_practica/output/predicciones.csv")
CLUSTER_GRID = Path("/workspace/ssd/solucion_practica/output/cluster/grid.png")
ANALYSIS_DIR = Path("/workspace/ssd/solucion_practica/output/analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device: {DEVICE}")
print(f"Output: {ANALYSIS_DIR}")

# ---------------------------------------------------------------------------
# CNN model definition (must match training)
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=4):
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
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features(x)
        if x.requires_grad:
            x.register_hook(self.activations_hook)
            self.activations = x
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def get_cam(self, target_class):
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        cam = torch.mean(self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("\n=== Loading model and data ===")

# Map cluster IDs to meaningful names initially
old_names = {0: "Cluster_0", 1: "Cluster_1", 2: "Cluster_2", 3: "Cluster_3"}

# Load predictions
import csv
all_fnames, all_labels = [], []
with open(PRED_PATH) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        all_fnames.append(row[0])
        all_labels.append(row[1])

label_names = sorted(set(all_labels))
label_to_idx = {l: i for i, l in enumerate(label_names)}
print(f"Loaded {len(all_fnames)} predictions, classes: {label_names}")

# Load model
model = CNN(num_classes=len(label_names)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded")

# Split same as training
indices = np.random.permutation(len(all_fnames))
n_train = int(0.7 * len(indices))
n_val = int(0.15 * len(indices))
test_idx = indices[n_train + n_val:]
test_files = [all_fnames[i] for i in test_idx]
test_labels = [label_to_idx[all_labels[i]] for i in test_idx]

eval_transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ---------------------------------------------------------------------------
# 2. GAMMA4 CLUSTER LABELING
# ---------------------------------------------------------------------------
print("\n=== Gemma4 cluster labeling ===")

async def label_with_gemma4():
    # Read the grid image and send to gemma4
    with open(CLUSTER_GRID, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = """
You see a 4-row grid of anime face images. Each row is a different cluster.
Analyze the visual characteristics that distinguish each cluster from the others.

For each cluster (row 1 to 4), provide a concise 1-2 word semantic label in English
that captures the dominant visual feature of that cluster.

Respond ONLY with JSON:
{
  "clusters": [
    {"row": 1, "label": "Label1", "explanation": "one line"},
    {"row": 2, "label": "Label2", "explanation": "one line"},
    {"row": 3, "label": "Label3", "explanation": "one line"},
    {"row": 4, "label": "Label4", "explanation": "one line"}
  ],
  "overall": "brief description of classification scheme"
}
"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.2, "max_tokens": 512}
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            result = await resp.json()
            text = result.get("response", "")
            j = text.find('{')
            k = text.rfind('}') + 1
            if j >= 0 and k > j:
                return json.loads(text[j:k])
            return {"error": "no JSON", "raw": text[:300]}

retry = 0
gemma4_result = {"error": "not run"}
while retry < 3:
    try:
        gemma4_result = asyncio.run(label_with_gemma4())
        if "clusters" in gemma4_result:
            break
    except Exception as e:
        print(f"  Retry {retry+1}: {e}")
    retry += 1

if "clusters" in gemma4_result:
    semantic_names = {}
    for c in gemma4_result["clusters"]:
        row = c["row"] - 1
        semantic_names[row] = c["label"].replace(" ", "_")
    semantic_names_inv = {v: k for k, v in semantic_names.items()}
    print(f"  Semantic labels: {semantic_names}")
else:
    semantic_names = old_names
    print(f"  gemma4 failed, using old names. Result: {gemma4_result}")

# Save labels
with open(ANALYSIS_DIR / "gemma4_labels.json", "w") as f:
    json.dump(gemma4_result, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 3. UMAP / T-SNE visualization with image overlays
# ---------------------------------------------------------------------------
print("\n=== UMAP projection ===")

# Re-extract features from sample
from torchvision.models import resnet18, ResNet18_Weights
resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()
resnet.to(DEVICE)
resnet.eval()

rn_transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Take N samples for UMAP (too many overwhelms)
N_UMAP = 1000
all_paths = sorted(DATA_DIR.glob("*.jpg"))
umap_paths = random.sample(all_paths, N_UMAP)

umap_feats = []
umap_labels_idx = []
umap_fnames = []
for p in tqdm(umap_paths, desc="UMAP features"):
    img = Image.open(p).convert("RGB")
    t = rn_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = resnet(t).cpu().numpy().squeeze()
    umap_feats.append(f)
    idx = all_fnames.index(p.name)
    umap_labels_idx.append(label_to_idx[all_labels[idx]])
    umap_fnames.append(p.name)

umap_feats = np.array(umap_feats)
scaler = StandardScaler()
umap_scaled = scaler.fit_transform(umap_feats)

# UMAP projection
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1, verbose=False)
    proj = reducer.fit_transform(umap_scaled)
    print("  UMAP done")
except:
    print("  UMAP not available, using PCA...")
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2, random_state=SEED)
    proj = reducer.fit_transform(umap_scaled)
    print("  PCA done")

# Plot UMAP with sample images
fig, ax = plt.subplots(figsize=(14, 10))
colors = plt.cm.Set2(np.linspace(0, 1, len(label_names)))

# Scatter background
for i, name in enumerate(label_names):
    mask = np.array(umap_labels_idx) == i
    ax.scatter(proj[mask, 0], proj[mask, 1], c=[colors[i]], alpha=0.3, s=5,
               label=f"{semantic_names.get(i, old_names[i])}")

# Overlay sample images
N_OVERLAY = 40
overlay_idx = random.sample(range(len(umap_paths)), N_OVERLAY)
for idx in overlay_idx:
    x, y = proj[idx, 0], proj[idx, 1]
    img = Image.open(umap_paths[idx]).resize((28, 28))
    img_arr = np.array(img)
    imagebox = OffsetImage(img_arr, zoom=0.7, cmap=None)
    ab = AnnotationBbox(imagebox, (x, y), frameon=True,
                        bboxprops=dict(edgecolor=colors[umap_labels_idx[idx]], linewidth=1.5))
    ax.add_artist(ab)

ax.set_title("Proyección UMAP del espacio latente (ResNet18)", fontsize=14, fontweight="bold")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend(markerscale=3, loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "umap_projection.png", dpi=200)
plt.close()
print("  Saved umap_projection.png")


# ---------------------------------------------------------------------------
# 4. INFERENCE GRIDS: Correct vs Incorrect
# ---------------------------------------------------------------------------
print("\n=== Inference grids ===")

test_transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

model.eval()
correct_info, incorrect_info = [], []
with torch.no_grad():
    for i, fname in enumerate(tqdm(test_files, desc="Inference")):
        img = Image.open(DATA_DIR / fname).convert("RGB")
        tensor = test_transform(img).unsqueeze(0).to(DEVICE)
        out = model(tensor)
        pred = out.argmax(1).item()
        prob = torch.softmax(out, dim=1).max().item()
        true = test_labels[i]
        info = (fname, true, pred, prob)
        if pred == true:
            correct_info.append(info)
        else:
            incorrect_info.append(info)

random.shuffle(correct_info)
random.shuffle(incorrect_info)

# Correct grid - 4 rows (one per true class) x 6 columns
fig, axes = plt.subplots(4, 6, figsize=(14, 9))
fig.suptitle("Predicciones Correctas", fontsize=14, fontweight="bold", y=0.98)
for true_cls in range(4):
    cls_correct = [(f, t, p, prob) for f, t, p, prob in correct_info if t == true_cls]
    for col in range(6):
        ax = axes[true_cls, col]
        if col < len(cls_correct):
            fname, true, pred, prob = cls_correct[col]
            img = Image.open(DATA_DIR / fname)
            ax.imshow(img)
            ax.set_title(f"→ {semantic_names.get(pred, old_names[pred])}\n{prob:.2f}", fontsize=7)
        ax.axis("off")
    axes[true_cls, 0].set_ylabel(f"Verdadero:\n{semantic_names.get(true_cls, old_names[true_cls])}",
                                  fontsize=8, fontweight="bold", rotation=0, labelpad=40, ha="right")
plt.tight_layout(rect=[0.08, 0, 1, 0.96])
plt.savefig(ANALYSIS_DIR / "inference_correct.png", dpi=150)
plt.close()

# Incorrect grid
fig, axes = plt.subplots(4, 6, figsize=(14, 9))
fig.suptitle("Predicciones Incorrectas", fontsize=14, fontweight="bold", y=0.98)
for true_cls in range(4):
    cls_incorrect = [(f, t, p, prob) for f, t, p, prob in incorrect_info if t == true_cls]
    for col in range(6):
        ax = axes[true_cls, col]
        if col < len(cls_incorrect):
            fname, true, pred, prob = cls_incorrect[col]
            img = Image.open(DATA_DIR / fname)
            ax.imshow(img)
            ax.set_title(f"Real: {semantic_names.get(true, old_names[true])}\n"
                         f"Pred: {semantic_names.get(pred, old_names[pred])}\n{prob:.2f}",
                         fontsize=6, color="red")
        ax.axis("off")
    axes[true_cls, 0].set_ylabel(f"Verdadero:\n{semantic_names.get(true_cls, old_names[true_cls])}",
                                  fontsize=8, fontweight="bold", rotation=0, labelpad=40, ha="right")
plt.tight_layout(rect=[0.08, 0, 1, 0.96])
plt.savefig(ANALYSIS_DIR / "inference_incorrect.png", dpi=150)
plt.close()
print(f"  Saved inference grids: {len(correct_info)} correct, {len(incorrect_info)} incorrect")


# ---------------------------------------------------------------------------
# 5. GRAD-CAM
# ---------------------------------------------------------------------------
print("\n=== Grad-CAM ===")

cam_samples = []
for true_cls in range(len(label_names)):
    cls_samples = [(f, t) for f, t in zip(test_files, test_labels) if t == true_cls][:4]
    cam_samples.extend(cls_samples)

num_cam = len(cam_samples)
fig, axes = plt.subplots(2, num_cam // 2, figsize=(14, 5))
fig.suptitle("Mapas de Activación Grad-CAM", fontsize=14, fontweight="bold")

for i, (fname, true_cls) in enumerate(cam_samples):
    row, col = i // (num_cam // 2), i % (num_cam // 2)
    img = Image.open(DATA_DIR / fname).convert("RGB")
    tensor = test_transform(img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad = True

    out = model(tensor)
    pred = out.argmax(1).item()
    model.zero_grad()
    out[0, pred].backward()

    cam = model.get_cam(pred)
    # Resize cam to image size
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((64, 64)))
    cam_resized = cam_resized / 255.0

    axes[row, col].imshow(img)
    axes[row, col].imshow(cam_resized, cmap="jet", alpha=0.4)
    axes[row, col].set_title(f"True: {semantic_names.get(true_cls, old_names[true_cls])}\n"
                              f"Pred: {semantic_names.get(pred, old_names[pred])}",
                              fontsize=7)
    axes[row, col].axis("off")

plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "gradcam.png", dpi=150)
plt.close()
print("  Saved gradcam.png")


# ---------------------------------------------------------------------------
# 6. ANNOTATED CLUSTER GRID (with semantic names)
# ---------------------------------------------------------------------------
print("\n=== Annotated cluster grid ===")

# Reload cluster grid and annotate with gemma4 labels
fig, axes = plt.subplots(4, 8, figsize=(18, 10))
fig.suptitle("Clusters Semánticos de Rostros Anime", fontsize=16, fontweight="bold", y=0.99)

# Re-cluster to get exactly the same results
from sklearn.cluster import KMeans
all_sample_paths = sorted(DATA_DIR.glob("*.jpg"))
sample_paths_list = random.Random(SEED).sample(all_sample_paths, 200)

sample_feats_v2 = []
for p in tqdm(sample_paths_list, desc="Re-extract for cluster grid"):
    img = Image.open(p).convert("RGB")
    t = rn_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = resnet(t).cpu().numpy().squeeze()
    sample_feats_v2.append(f)
sample_feats_v2 = np.array(sample_feats_v2)

km = KMeans(n_clusters=4, random_state=SEED, n_init=10)
sample_labels_v2 = km.fit_predict(sample_feats_v2)

for row in range(4):
    idxs_in = [i for i, l in enumerate(sample_labels_v2) if l == row]
    selected = random.sample(idxs_in, min(8, len(idxs_in)))
    for col, idx in enumerate(selected):
        img = Image.open(sample_paths_list[idx])
        axes[row, col].imshow(img)
        axes[row, col].axis("off")
    name = semantic_names.get(row, old_names[row])
    axes[row, 0].set_ylabel(f"{name}\n(n={len(idxs_in)})",
                             fontsize=11, fontweight="bold", rotation=0,
                             labelpad=60, ha="right", va="center")
    # Add a category description
    if "clusters" in gemma4_result:
        expl = gemma4_result["clusters"][row].get("explanation", "")
        axes[row, 0].set_title(expl, fontsize=7, ha="left", loc="left", pad=5)

plt.tight_layout(rect=[0.1, 0, 1, 0.97])
plt.savefig(ANALYSIS_DIR / "clusters_annotated.png", dpi=150)
plt.close()
print("  Saved clusters_annotated.png")


# ---------------------------------------------------------------------------
# 7. PER-CLASS METRICS
# ---------------------------------------------------------------------------
print("\n=== Per-class metrics ===")

from sklearn.metrics import classification_report
model.eval()
all_preds, all_trues = [], []
with torch.no_grad():
    for fname, true_lbl in tqdm(zip(test_files, test_labels), desc="Eval", total=len(test_files)):
        img = Image.open(DATA_DIR / fname).convert("RGB")
        tensor = test_transform(img).unsqueeze(0).to(DEVICE)
        out = model(tensor)
        pred = out.argmax(1).item()
        all_preds.append(pred)
        all_trues.append(true_lbl)

# Use semantic names
display_names = [semantic_names.get(i, old_names[i]) for i in range(len(label_names))]
report = classification_report(all_trues, all_preds, target_names=display_names,
                               digits=4, output_dict=True)

with open(ANALYSIS_DIR / "classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

# Tabulate
for name in display_names:
    d = report[name]
    print(f"  {name}: precision={d['precision']:.4f} recall={d['recall']:.4f} f1={d['f1-score']:.4f}")

accuracy = report["accuracy"]
print(f"\n  Overall accuracy: {accuracy:.4f}")
print(f"  Random baseline: {1.0/len(label_names):.4f}")

# Save all results summary
summary = {
    "labels_semantic": [semantic_names.get(i, old_names[i]) for i in range(len(label_names))],
    "labels_raw": list(label_names),
    "gemma4_response": gemma4_result,
    "test_accuracy": float(accuracy),
    "per_class_metrics": {n: report[n] for n in display_names},
    "total_test": len(test_files),
    "correct_predictions": sum(1 for p, t in zip(all_preds, all_trues) if p == t),
    "incorrect_predictions": sum(1 for p, t in zip(all_preds, all_trues) if p != t),
}
with open(ANALYSIS_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Analysis complete ===")
print(f"All files saved to {ANALYSIS_DIR}")
