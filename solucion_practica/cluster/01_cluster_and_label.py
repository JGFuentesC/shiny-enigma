#!/usr/bin/env python3
"""
Step 1: Cluster anime faces using ResNet18 features + K-means.
Then ask gemma4 ONCE to label the clusters semantically.
Total: < 2 minutes.
"""

import os, sys, json, io, base64, random, asyncio, aiohttp
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
MODEL = "gemma4:e4b"
SAMPLE_SIZE = 200
RANDOM_SEED = 42
N_CLUSTERS = 4

DATA_DIR = Path("/workspace/ssd/solucion_practica/data/anime_faces/images")
OUTPUT_DIR = Path("/workspace/ssd/solucion_practica/cluster")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def load_image_paths():
    all_paths = sorted(DATA_DIR.glob("*.jpg"))
    print(f"Found {len(all_paths)} images")
    return all_paths


def extract_features(paths, batch_size=512):
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.to(DEVICE)
    model.eval()

    features = []
    filenames = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch_tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch_tensors.append(transform(img))
        batch = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad():
            feats = model(batch).cpu().numpy()
        features.append(feats)
        filenames.extend([p.name for p in batch_paths])
        if (i + 1) % 2000 == 0:
            print(f"  Features: {i + len(batch_paths)}/{len(paths)}")
    return np.concatenate(features), filenames


def select_sample(features, filenames, n=SAMPLE_SIZE):
    random.seed(RANDOM_SEED)
    idxs = random.sample(range(len(filenames)), min(n, len(filenames)))
    return features[idxs], [filenames[i] for i in idxs], idxs


def find_best_k(features, k_range=range(4, 9)):
    best_k, best_score = N_CLUSTERS, -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(features)
        score = silhouette_score(features, labels)
        print(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    print(f"Best k={best_k} (silhouette={best_score:.4f})")
    return best_k


def plot_cluster_grid(sample_paths, cluster_labels, filenames, label_map):
    """Plot a grid of 16 sample images per cluster."""
    clusters = sorted(set(cluster_labels))
    fig, axes = plt.subplots(len(clusters), 8, figsize=(16, 3 * len(clusters)))
    if len(clusters) == 1:
        axes = axes[np.newaxis, :]

    for row, cluster in enumerate(clusters):
        cluster_idxs = [i for i, l in enumerate(cluster_labels) if l == cluster]
        selected = random.sample(cluster_idxs, min(8, len(cluster_idxs)))
        for col, idx in enumerate(selected):
            img_path = DATA_DIR / filenames[idx]
            img = Image.open(img_path)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                lbl = label_map.get(cluster, f"Cluster {cluster}")
                axes[row, col].set_ylabel(lbl, fontsize=9)

    plt.tight_layout()
    path = OUTPUT_DIR / "01_clusters_grid.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


async def label_clusters_with_gemma4(cluster_dirs):
    """Send one composite image per cluster to gemma4 for semantic labeling."""
    descriptions = []
    for cluster_name, image_paths in cluster_dirs.items():
        # Create a 4x4 montage of images for this cluster
        n = min(16, len(image_paths))
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(16):
            ax = axes[i // 4][i % 4]
            if i < n:
                img = Image.open(random.choice(image_paths))
                ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=80)
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        descriptions.append((cluster_name, img_b64))

    prompt = """
I will show you 4 groups of anime face images, one group at a time.
For each group, describe the single dominant visual characteristic that
makes this group distinct from the others.

Then, assign a short 1-2 word category name to each group.

Finally, suggest a single overall classification scheme that uses exactly
4 categories to separate these anime faces.

Respond with JSON:
{
  "groups": [
    {"group": "A", "dominant_feature": "...", "category_name": "..."},
    {"group": "B", "dominant_feature": "...", "category_name": "..."},
    {"group": "C", "dominant_feature": "...", "category_name": "..."},
    {"group": "D", "dominant_feature": "...", "category_name": "..."}
  ],
  "classification_scheme": "short description"
}
"""

    all_images_b64 = [b64 for _, b64 in descriptions]
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": all_images_b64,
        "stream": False,
        "options": {"temperature": 0.3, "max_tokens": 512}
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            result = await resp.json()
            text = result.get("response", "")
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(text[json_start:json_end])
            return {"raw": text, "error": "no JSON"}


def save_markdown(sample_fnames, cluster_labels, label_map, gemma4_result, filenames_all, labels_all):
    lines = []
    lines.append("# Clustering de Rostros Anime\n")
    lines.append(f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Método**: ResNet18 features + K-Means\n")
    lines.append(f"**Muestra**: {len(sample_fnames)} imágenes\n")
    lines.append(f"**Clusters**: {N_CLUSTERS}\n\n")

    lines.append("## Distribución de Clusters (muestra)\n\n")
    lines.append("| Cluster | Nombre | Conteo |\n")
    lines.append("|---------|--------|--------|\n")
    counter = Counter(cluster_labels)
    for c in sorted(counter.keys()):
        name = label_map.get(c, f"Cluster {c}")
        lines.append(f"| {c} | {name} | {counter[c]} |\n")
    lines.append("\n")

    if "error" not in gemma4_result:
        lines.append("## Etiquetado Semántico (gemma4)\n\n")
        for g in gemma4_result.get("groups", []):
            lines.append(f"- **{g['group']}**: {g['category_name']} — {g['dominant_feature']}\n")
        lines.append(f"\n**Esquema**: {gemma4_result.get('classification_scheme', '')}\n\n")

    lines.append("## Figuras\n\n")
    lines.append("- `01_clusters_grid.png`: Grid de muestras por cluster\n\n")

    with open(OUTPUT_DIR / "01_clustering.md", "w") as f:
        f.writelines(lines)
    print(f"Saved {OUTPUT_DIR / '01_clustering.md'}")


def main():
    t0 = datetime.now()

    # 1. Load image paths
    all_paths = load_image_paths()

    # 2. Extract features for ~200 sample + ALL (do sample first)
    print("\nExtracting sample features...")
    sample_paths = random.Random(RANDOM_SEED).sample(all_paths, SAMPLE_SIZE)
    sample_feats, sample_fnames = extract_features(sample_paths)

    # 3. K-means clustering
    print(f"\nK-means clustering (k={N_CLUSTERS})...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    sample_labels = km.fit_predict(sample_feats)
    print(f"Silhouette: {silhouette_score(sample_feats, sample_labels):.4f}")

    t1 = datetime.now()
    print(f"Clustering done in {(t1 - t0).total_seconds():.1f}s")

    # 4. Ask gemma4 to label clusters (ONE call)
    print("\nSending cluster grids to gemma4 for labeling...")
    cluster_paths = {i: [] for i in range(N_CLUSTERS)}
    for fname, label in zip(sample_fnames, sample_labels):
        cluster_paths[label].append(DATA_DIR / fname)

    gemma4_result = asyncio.run(label_clusters_with_gemma4(
        {f"Cluster {k}": v for k, v in cluster_paths.items()}
    ))
    print(f"gemma4 result: {json.dumps(gemma4_result, indent=2, ensure_ascii=False)[:500]}")

    # Build label map
    label_map = {}
    if "error" not in gemma4_result:
        for i, g in enumerate(gemma4_result.get("groups", [])):
            label_map[i] = g.get("category_name", f"Cluster {i}")
    else:
        label_map = {i: f"Cluster {i}" for i in range(N_CLUSTERS)}

    t2 = datetime.now()
    print(f"Labeling done in {(t2 - t1).total_seconds():.1f}s")

    # 5. Plot cluster grid
    plot_cluster_grid(sample_paths, sample_labels, sample_fnames, label_map)

    # 6. Save labeled sample data
    labeled_data = [{"filename": f, "cluster": int(l), "label": label_map[int(l)]}
                    for f, l in zip(sample_fnames, sample_labels)]
    with open(OUTPUT_DIR / "01_muestra_etiquetada.json", "w") as f:
        json.dump(labeled_data, f, indent=2)
    with open(OUTPUT_DIR / "01_label_map.json", "w") as f:
        json.dump(label_map, f)

    # 7. Save markdown
    save_markdown(sample_fnames, sample_labels, label_map, gemma4_result, [], [])

    t3 = datetime.now()
    print(f"\nTotal: {(t3 - t0).total_seconds():.1f}s")


if __name__ == "__main__":
    main()
