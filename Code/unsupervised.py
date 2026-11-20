"""Unsupervised learning companion script.

Clustering (k-means, hierarchical, DBSCAN) and dimensionality reduction (PCA,
t-SNE) on the handwritten-digits dataset.  Run from the repository root:

    python Code/unsupervised.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def cluster_sweep(
    X: np.ndarray, ks: np.ndarray, sample_idx: np.ndarray, seed: int = 42
) -> tuple[list[float], list[float], KMeans]:
    """Fit k-means for each ``k`` and return inertia and silhouette scores."""
    inertias, silhouettes = [], []
    models = {}
    for k in ks:
        model = KMeans(n_clusters=int(k), n_init=10, random_state=seed).fit(X)
        models[int(k)] = model
        inertias.append(float(model.inertia_))
        silhouettes.append(float(
            silhouette_score(X[sample_idx], model.labels_[sample_idx])
        ))
    best_k = int(ks[int(np.argmax(silhouettes))])
    return inertias, silhouettes, models[best_k]


def dbscan_grid(
    X: np.ndarray, eps_values: np.ndarray, min_samples_values: np.ndarray
) -> dict[tuple[float, int], dict[str, float]]:
    """Score DBSCAN settings by silhouette (ignoring noise-labelled points)."""
    results = {}
    for eps in eps_values:
        for min_samples in min_samples_values:
            labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(X)
            core = labels != -1
            n_clusters = len(set(labels[core]))
            score = (float(silhouette_score(X[core], labels[core]))
                     if 1 < n_clusters < len(X) else float("-inf"))
            results[(float(eps), int(min_samples))] = {
                "silhouette": score,
                "n_clusters": float(n_clusters),
                "noise_fraction": float(np.mean(~core)),
            }
    return results


def plot_cluster_metrics(
    ax: Axes, ks: np.ndarray, inertias: list[float], silhouettes: list[float],
    best_k: int,
) -> None:
    """Plot elbow and silhouette curves on twin axes."""
    ax.plot(ks, inertias, marker="o", label="inertia")
    second = ax.twinx()
    second.plot(ks, silhouettes, marker="s", color="darkorange", label="silhouette")
    ax.axvline(best_k, color="red", linestyle="--",
               label=f"best k = {best_k}")
    ax.set(xlabel="Number of clusters", ylabel="Inertia", title="Elbow and silhouette")
    second.set_ylabel("Silhouette")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-tsne", action="store_true",
                        help="skip the relatively slow t-SNE projection")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    digits = load_digits()
    X, y = digits.data, digits.target
    X_scaled = StandardScaler().fit_transform(X)
    sample_idx = rng.choice(len(X_scaled), size=800, replace=False)

    # ------------------------------------------------------------------
    # 1. K-means: elbow and silhouette
    # ------------------------------------------------------------------
    ks = np.arange(2, 11)
    inertias, silhouettes, best_model = cluster_sweep(
        X_scaled, ks, sample_idx
    )
    best_k = best_model.n_clusters
    print(f"K-means: best k by silhouette = {best_k}")
    print(f"  best-k ARI vs true labels = {adjusted_rand_score(y, best_model.labels_):.3f}")
    kmeans10 = KMeans(n_clusters=10, n_init=10, random_state=42).fit(X_scaled)
    print(f"  k=10 ARI vs true labels = {adjusted_rand_score(y, kmeans10.labels_):.3f}")

    # ------------------------------------------------------------------
    # 2. Hierarchical clustering on a subset (dendrogram)
    # ------------------------------------------------------------------
    subset = rng.choice(len(X_scaled), size=80, replace=False)
    Z = linkage(X_scaled[subset], method="ward")
    hierarchical_labels = AgglomerativeClustering(n_clusters=10).fit_predict(
        X_scaled[subset]
    )
    print(f"  hierarchical ARI (10 clusters) = "
          f"{adjusted_rand_score(y[subset], hierarchical_labels):.3f}")

    # ------------------------------------------------------------------
    # 3. DBSCAN with a small parameter grid
    # ------------------------------------------------------------------
    grid = dbscan_grid(
        X_scaled[sample_idx], eps_values=np.arange(2.0, 5.1, 0.5),
        min_samples_values=[4, 8, 12],
    )
    best_params, best = max(grid.items(), key=lambda item: item[1]["silhouette"])
    print(f"  DBSCAN best eps={best_params[0]}, min_samples={best_params[1]}: "
          f"clusters={int(best['n_clusters'])}, "
          f"noise={best['noise_fraction']:.1%}, silhouette={best['silhouette']:.3f}")

    # ------------------------------------------------------------------
    # 4. PCA and (optionally) t-SNE
    # ------------------------------------------------------------------
    pca90 = PCA(n_components=0.90, random_state=42).fit(X_scaled)
    print(f"  PCA components for 90% variance = {pca90.n_components_}")
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    X_tsne = None
    if not args.no_tsne:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca",
                    learning_rate="auto")
        X_tsne = tsne.fit_transform(X_scaled)

    if args.no_plots:
        return

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    plot_cluster_metrics(axes[0, 0], ks, inertias, silhouettes, best_k)
    dendrogram(Z, ax=axes[0, 1], no_labels=True, color_threshold=0)
    axes[0, 1].set(title="Ward dendrogram (80 samples)", xlabel="Sample")
    scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", s=8)
    axes[1, 0].set(title="PCA projection", xlabel="PC 1", ylabel="PC 2")
    if X_tsne is not None:
        axes[1, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", s=8)
    axes[1, 1].set(title="t-SNE projection" if X_tsne is not None else "t-SNE skipped",
                   xlabel="dimension 1", ylabel="dimension 2")
    fig.colorbar(scatter, ax=list(axes[1, :]), label="true digit", shrink=0.8)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
