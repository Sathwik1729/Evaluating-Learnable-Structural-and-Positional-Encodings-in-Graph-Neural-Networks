"""
Optional t-SNE visualizations for Cora node-level representations.

The training pipeline saves histories, not model checkpoints or hidden states, so
this script visualizes the two embeddings that are always reproducible from the
dataset loader: raw node features and Laplacian positional encodings.

Run from the repository root:
  python visualizations/tsne.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('visualizations/.matplotlib')))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from datasets import load_dataset


FIGURES_DIR = Path('visualizations/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _plot_embedding(points, labels, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap='tab10',
        s=10,
        alpha=0.85,
        linewidths=0,
    )
    ax.set_title(title, fontsize=13, weight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    legend = ax.legend(
        *scatter.legend_elements(),
        title='Class',
        frameon=False,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
    )
    ax.add_artist(legend)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def run_tsne(matrix, seed=42):
    matrix = StandardScaler(with_mean=False).fit_transform(matrix)
    return TSNE(
        n_components=2,
        perplexity=30,
        init='pca',
        learning_rate='auto',
        random_state=seed,
    ).fit_transform(matrix)


def main():
    data, _, _ = load_dataset(name='cora', pe_dim=8)
    labels = data.y.cpu().numpy()

    feature_points = run_tsne(data.x.cpu().numpy())
    _plot_embedding(
        feature_points,
        labels,
        't-SNE of Cora Raw Node Features',
        FIGURES_DIR / 'tsne_cora_features.png',
    )

    pe_points = run_tsne(data.pe.cpu().numpy())
    _plot_embedding(
        pe_points,
        labels,
        't-SNE of Cora Laplacian Positional Encodings',
        FIGURES_DIR / 'tsne_cora_laplacian_pe.png',
    )


if __name__ == '__main__':
    main()
