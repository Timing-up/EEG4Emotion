"""
Visualization utilities for DG-MAGNet
======================================

- t-SNE of learned feature embeddings (colored by emotion / subject)
- Brain topography heatmaps (channel importance from adjacency)
- Ablation bar charts with error bars
- Confusion matrices
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE


# Standard 62-channel 10-20 positions (approximate 2D projection for SEED)
# Subset of key positions; full mapping should be loaded from dataset metadata
SEED_62_POSITIONS_2D = None  # Populated by load_channel_positions()


def _ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "t-SNE",
    label_names: list[str] | None = None,
    perplexity: float = 30.0,
    figsize: tuple = (8, 6),
):
    """Plot t-SNE of feature embeddings colored by class label.

    Args:
        features: (N, D) array of feature vectors
        labels: (N,) integer class labels
        save_path: output image path
        label_names: optional class name mapping
        perplexity: t-SNE perplexity
    """
    _ensure_dir(os.path.dirname(save_path))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    emb = tsne.fit_transform(features)

    unique_labels = sorted(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    fig, ax = plt.subplots(figsize=figsize)
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = label_names[lbl] if label_names else str(lbl)
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[cmap(i)], label=name,
                   s=15, alpha=0.7, edgecolors="none")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved t-SNE: {save_path}")


def plot_tsne_by_subject(
    features: np.ndarray,
    subject_ids: np.ndarray,
    save_path: str,
    title: str = "t-SNE (by subject)",
    perplexity: float = 30.0,
    figsize: tuple = (8, 6),
):
    """t-SNE colored by subject ID — useful to check domain invariance."""
    _ensure_dir(os.path.dirname(save_path))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    emb = tsne.fit_transform(features)

    unique_subs = sorted(np.unique(subject_ids))
    cmap = plt.cm.get_cmap("tab20", len(unique_subs))

    fig, ax = plt.subplots(figsize=figsize)
    for i, sid in enumerate(unique_subs):
        mask = subject_ids == sid
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[cmap(i)], label=f"S{sid}",
                   s=10, alpha=0.6, edgecolors="none")
    if len(unique_subs) <= 16:
        ax.legend(fontsize=7, markerscale=2, ncol=2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved t-SNE (subject): {save_path}")


# ---------------------------------------------------------------------------
# Brain topography
# ---------------------------------------------------------------------------

def load_channel_positions(filepath: str | None = None) -> np.ndarray:
    """Load 2D channel positions for topographic plotting.

    If filepath is provided, expects a (C, 2) CSV/npy of x, y coords.
    Otherwise uses a default circle layout for 62 channels.

    Returns:
        (C, 2) array of 2D positions
    """
    if filepath and os.path.exists(filepath):
        pos = np.load(filepath) if filepath.endswith(".npy") else np.loadtxt(filepath, delimiter=",")
        return pos

    # Default: uniform circle layout
    n_channels = 62
    angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def plot_brain_topo(
    channel_values: np.ndarray,
    save_path: str,
    positions: np.ndarray | None = None,
    title: str = "Channel Importance",
    cmap: str = "RdYlBu_r",
    figsize: tuple = (6, 5),
):
    """Plot brain topography heatmap.

    Args:
        channel_values: (C,) values per channel (e.g., adjacency row-sum)
        save_path: output image path
        positions: (C, 2) 2D electrode positions
        title: plot title
    """
    _ensure_dir(os.path.dirname(save_path))

    if positions is None:
        positions = load_channel_positions()

    C = len(channel_values)
    positions = positions[:C]

    fig, ax = plt.subplots(figsize=figsize)

    # Draw head outline
    theta = np.linspace(0, 2 * np.pi, 100)
    r = 1.1
    ax.plot(r * np.cos(theta), r * np.sin(theta), "k-", linewidth=1.5)
    # Nose
    ax.plot([0, -0.08, 0, 0.08, 0], [r, r + 0.12, r + 0.18, r + 0.12, r],
            "k-", linewidth=1.5)

    # Scatter with colormap
    sc = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=channel_values, cmap=cmap, s=80,
        edgecolors="k", linewidths=0.5, zorder=5,
    )
    fig.colorbar(sc, ax=ax, shrink=0.7, label="Value")
    ax.set_title(title)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved brain topo: {save_path}")


def plot_adjacency_topo(
    A: np.ndarray,
    save_path: str,
    positions: np.ndarray | None = None,
    title: str = "Learned Adjacency (row-sum)",
):
    """Plot channel importance derived from adjacency matrix row-sums."""
    channel_importance = A.sum(axis=-1)  # (C,) or handle (K, C, C)
    if channel_importance.ndim > 1:
        channel_importance = channel_importance.mean(axis=0)
    if channel_importance.ndim > 1:
        channel_importance = channel_importance.mean(axis=0)
    plot_brain_topo(channel_importance, save_path, positions, title)


# ---------------------------------------------------------------------------
# Ablation bar chart
# ---------------------------------------------------------------------------

def plot_ablation_bars(
    results: dict[str, dict],
    save_path: str,
    metric: str = "accuracy",
    title: str = "Ablation Study",
    figsize: tuple = (10, 5),
):
    """Bar chart comparing ablation variants.

    Args:
        results: {variant_name: {"mean_<metric>": float, "std_<metric>": float}}
        save_path: output image path
        metric: which metric to plot
    """
    _ensure_dir(os.path.dirname(save_path))

    names = list(results.keys())
    means = [results[n][f"mean_{metric}"] for n in names]
    stds = [results[n][f"std_{metric}"] for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#4C72B0",
                  edgecolor="white", linewidth=0.8, alpha=0.9)

    # Value labels
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                f"{m:.1%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, max(means) + max(stds) + 0.08)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved ablation chart: {save_path}")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: tuple = (6, 5),
):
    """Plot confusion matrix heatmap.

    Args:
        cm: (K, K) confusion matrix (counts)
        save_path: output path
        class_names: list of class labels
        normalize: if True, show percentages per true class
    """
    _ensure_dir(os.path.dirname(save_path))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_plot = cm.astype(float) / row_sums
        fmt = ".1%"
    else:
        cm_plot = cm
        fmt = "d"

    K = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(K)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, cmap="Blues", aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Annotations
    for i in range(K):
        for j in range(K):
            val = cm_plot[i, j]
            text = f"{val:{fmt}}" if normalize else f"{val}"
            color = "white" if val > cm_plot.max() / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix: {save_path}")
