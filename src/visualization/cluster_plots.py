"""
Module: cluster_plots.py
========================

Visualizations for clustering results and analysis.

Provides elbow curves, cluster projections, silhouette plots,
and confusion matrices for cluster evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_DOUBLE = (14, 6)
FIGSIZE_LARGE = (14, 10)
DPI = 150


def plot_cluster_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    elbow_data: Dict[int, float] = None,
    silhouette_data: Dict[int, float] = None,
    centroids: np.ndarray = None,
    true_labels: np.ndarray = None,
    output_dir: str = "outputs/figures"
) -> None:
    """
    Create comprehensive clustering analysis visualizations.

    Args:
        features: Feature matrix used for clustering
        labels: Predicted cluster labels
        elbow_data: Dictionary of K -> inertia values
        silhouette_data: Dictionary of K -> silhouette scores
        centroids: Cluster centers
        true_labels: Ground truth labels (optional)
        output_dir: Directory to save figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating clustering visualizations...")

    # Elbow and silhouette curves
    if elbow_data and silhouette_data:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

        k_values = sorted(elbow_data.keys())
        inertias = [elbow_data[k] for k in k_values]
        silhouettes = [silhouette_data[k] for k in k_values]

        # Elbow curve
        axes[0].plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_title('Elbow Method for Optimal K', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Number of Clusters (K)')
        axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
        axes[0].grid(True, alpha=0.3)

        optimal_k = k_values[np.argmax(silhouettes)]
        axes[0].axvline(x=optimal_k, color='red', linestyle='--', label=f'Selected K={optimal_k}')
        axes[0].legend()

        # Silhouette curve
        axes[1].plot(k_values, silhouettes, 'go-', linewidth=2, markersize=8)
        axes[1].set_title('Silhouette Score vs K', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Number of Clusters (K)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(x=optimal_k, color='red', linestyle='--', label=f'Selected K={optimal_k}')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_path / "fig13_cluster_elbow.png", dpi=DPI, bbox_inches='tight')
        plt.close()

    # 2D cluster projection
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # PCA projection
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    scatter1 = axes[0].scatter(
        features_pca[:, 0],
        features_pca[:, 1],
        c=labels,
        cmap='viridis',
        alpha=0.5,
        s=20
    )
    axes[0].set_title('Clusters in PCA Space', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    # Plot centroids if available
    if centroids is not None:
        centroids_pca = pca.transform(centroids)
        axes[0].scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            c='red',
            marker='X',
            s=200,
            edgecolors='black',
            linewidths=2
        )

    # t-SNE projection (sample if too large)
    n_samples = min(5000, len(features))
    sample_idx = np.random.choice(len(features), n_samples, replace=False)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_tsne = tsne.fit_transform(features[sample_idx])

    scatter2 = axes[1].scatter(
        features_tsne[:, 0],
        features_tsne[:, 1],
        c=labels[sample_idx],
        cmap='viridis',
        alpha=0.5,
        s=20
    )
    axes[1].set_title('Clusters in t-SNE Space (sampled)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')

    plt.tight_layout()
    plt.savefig(output_path / "fig14_cluster_projection.png", dpi=DPI, bbox_inches='tight')
    plt.close()

    # Cluster sizes
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    unique_labels, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(unique_labels)))

    bars = ax.bar(unique_labels, counts, color=colors, edgecolor='black')

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f'{count:,}',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Events')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "fig15_cluster_sizes.png", dpi=DPI, bbox_inches='tight')
    plt.close()

    # Cluster vs ground truth
    if true_labels is not None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        true_encoded = le.fit_transform(true_labels)

        n_pred_clusters = len(np.unique(labels))
        n_true_clusters = len(np.unique(true_encoded))

        confusion = np.zeros((n_pred_clusters, n_true_clusters))
        for pred, true in zip(labels, true_encoded):
            confusion[pred, true] += 1

        confusion_normalized = confusion / confusion.sum(axis=1, keepdims=True)

        im = ax.imshow(confusion_normalized, cmap='Blues', aspect='auto')

        ax.set_xticks(range(n_true_clusters))
        ax.set_yticks(range(n_pred_clusters))
        ax.set_xticklabels(le.classes_, rotation=45, ha='right')
        ax.set_yticklabels(range(n_pred_clusters))

        ax.set_xlabel('True Cluster')
        ax.set_ylabel('Predicted Cluster')
        ax.set_title('Cluster Assignment vs Ground Truth', fontsize=14, fontweight='bold')

        for i in range(n_pred_clusters):
            for j in range(n_true_clusters):
                val = confusion_normalized[i, j]
                if val > 0.01:
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color='white' if val > 0.5 else 'black', fontsize=9)

        plt.colorbar(im, ax=ax, label='Proportion')
        plt.tight_layout()
        plt.savefig(output_path / "fig16_cluster_confusion.png", dpi=DPI, bbox_inches='tight')
        plt.close()

    print(f"Clustering figures saved to: {output_path}")


def plot_silhouette_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Optional[str] = None
) -> None:
    """
    Create detailed silhouette analysis plot.

    Shows the silhouette coefficient for each sample, grouped by cluster.

    Args:
        features: Feature matrix
        labels: Cluster labels
        output_path: Path to save figure
    """
    from sklearn.metrics import silhouette_samples, silhouette_score

    fig, ax = plt.subplots(figsize=(10, 8))

    silhouette_avg = silhouette_score(features, labels)
    sample_silhouette_values = silhouette_samples(features, labels)

    n_clusters = len(np.unique(labels))
    y_lower = 10

    colors = plt.cm.viridis(np.linspace(0, 0.8, n_clusters))

    for i in range(n_clusters):
        cluster_silhouettes = sample_silhouette_values[labels == i]
        cluster_silhouettes.sort()

        cluster_size = len(cluster_silhouettes)
        y_upper = y_lower + cluster_size

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouettes,
            facecolor=colors[i],
            edgecolor=colors[i],
            alpha=0.7
        )

        ax.text(-0.05, y_lower + 0.5 * cluster_size, str(i), fontsize=10, fontweight='bold')

        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2,
               label=f'Average: {silhouette_avg:.3f}')

    ax.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()
