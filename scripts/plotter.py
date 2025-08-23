from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# DIRECTORIES AND FILENAMES MIGHT CHANGE--BASED ON WHERE YOU GET THE CODE FROM--SO
# MAKE SURE THEY MATCH

# Try importing umap, (colab/local env compatibility issue, keep for now)
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
    # Plot tsne instead
    from sklearn.manifold import TSNE

# ///////////////// CONSTANTS ////////////////////////////////////////////////////////////////////////
EPOCHS = [1, 8, 15] # Checkpoints we want to plot (hardcoded)
CHECKPOINTS_DIR = Path('/content/FGED_baseline_1/checkpoints')
LABELS = Path('/content/FGED_baseline_1/data/labels.json')
OUTPUT_DIR = Path('/content/FGED_baseline_1/figures/clusters')
RANDOM_STATE = 42

# UMAP parameters (simple)
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.15


# ///////////////// HELPERS //////////////////////////////////////////////////////////////////////////
def load_epoch(epoch: int):
    # Keep them on CPU, as UMAP/TSNE works on CPU
    E = torch.load(CHECKPOINTS_DIR / f'train_embeddings_epoch{epoch}.pt', map_location='cpu')
    L = torch.load(CHECKPOINTS_DIR / f'train_labels_epoch{epoch}.pt', map_location='cpu')
    P = torch.load(CHECKPOINTS_DIR / f'prototypes_epoch{epoch}.pt', map_location='cpu')
    # Tensors → Numpys
    if isinstance(E, torch.Tensor):
        E = E.detach().cpu().numpy()
    if isinstance(L, torch.Tensor):
        L = L.detach().cpu().numpy()
    if isinstance(P, torch.Tensor):
        P = P.detach().cpu().numpy()
    return E, L.astype(bool), P


def get_label_names(num_classes: int):
    if LABELS.exists():
        with open(LABELS, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        if isinstance(labels, dict) and 'labels' in labels and len(labels['labels']) == num_classes:
            return list(map(str, labels['labels']))
    return [f'label_{i}' for i in range(num_classes)]


def colour_list(num_classes: int, neutral_index: int):
    try:
        # Cool colours (install if you don't have it)
        import colorcet as cc
        base = list(cc.glasbey) # Base colours
    except Exception:
        # If you don't install it we would have to create a pallett
        # By combining three palletts
        base = list(plt.get_cmap('tab20').colors) + \
               list(plt.get_cmap('tab20b').colors) + \
               list(plt.get_cmap('tab20c').colors)
    if len(base) < num_classes:
        # If not enough, add more colours
        import colorsys
        base += [colorsys.hsv_to_rgb(h, 0.65, 0.95) for h in np.linspace(0, 1, num_classes - len(base), endpoint=False)]
    colours = base[:num_classes]
    if 0 <= neutral_index < num_classes: # If we have neutral index and its embeddings/prototype
        colours[neutral_index] = '#000000' # In the baseline version we have neutral
    return colours


def projector():
    # If UMAP exists, initialise it, otherwise do TSNE
    if HAS_UMAP:
        return umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=2, metric='cosine', random_state=RANDOM_STATE)
    return TSNE(n_components=2, perplexity=35, learning_rate='auto', init='pca', random_state=RANDOM_STATE)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialise the plot then iterate and plot each epoch separately on it (faster this way)
    EFIRST, LFIRST, PFIRST = load_epoch(EPOCHS[0])
    C = LFIRST.shape[1] # Classes
    names = get_label_names(C)
    neutral_index = C - 1 # Always last
    colours = colour_list(C, neutral_index)

    for epoch in EPOCHS:
        E, L, P = load_epoch(epoch)

        # Fit projection on embeddings + prototypes together --per epoch
        projection = projector()
        X = np.concatenate([E, P], axis=0)
        fitted = projection.fit_transform(X)
        FE = fitted[:len(E)]
        FP = fitted[len(E):]

        plt.figure(figsize=(10, 8), dpi=160)
        ax = plt.gca()

        # Draw each label layer (multi‑label → points may be drawn multiple times)
        for c in range(C):
            mask = L[:, c]
            # Skip if label data doesn't exist
            if not np.any(mask):
                continue
            # Plot ...
            ax.scatter(FE[mask, 0], FE[mask, 1], s=6.0, alpha=0.40, c=[colours[c]], linewidths=0, marker='o', rasterized=True, label=names[c])

        # Prototypes: gold stars + label text above
        ax.scatter(FP[:, 0], FP[:, 1], s=190, c=['#FFC400'], marker='*', edgecolor='k', linewidths=0.9, alpha=0.98)
        
        y_span = (np.nanmax(FE[:, 1]) - np.nanmin(FE[:, 1])) if FE.size else 1.0
        y_off = 0.015 * (y_span + 1e-9)
        
        for i in range(C):
            txt = ax.text(FP[i, 0], FP[i, 1] + y_off, names[i], color='black', fontsize=9,
                          ha='center', va='bottom', weight='bold')
            txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

        ax.set_title(f'Epoch {epoch}')
        ax.grid(True, linewidth=0.3, alpha=0.3)
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, ncol=1, frameon=False)
        if legend:
            legend.set_draggable(True)

        png = OUTPUT_DIR / f'epoch{epoch}.png'
        pdf = OUTPUT_DIR / f'epoch{epoch}.pdf'
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig(png)
        plt.savefig(pdf)
        plt.close()
        print(f'[saved] {png}')
        print(f'[saved] {pdf}')

    print('Done :)')


if __name__ == '__main__':
    main()