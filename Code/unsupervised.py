"""Unsupervised learning companion script.

Clustering (k-means, hierarchical, DBSCAN) and dimensionality reduction (PCA,
t-SNE) on the handwritten-digits dataset.  Run from the repository root:

    python Code/unsupervised.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    rng = np.random.default_rng(42)
    digits = load_digits()
    X, y = digits.data, digits.target
    X_scaled = StandardScaler().fit_transform(X)

    # ------------------------------------------------------------------
    # 1. K-means: elbow and silhouette
    # ------------------------------------------------------------------
    ks = np.arange(2, 11)
    inertias, silhouettes = [], []
    sample_idx = rng.choice(len(X_scaled), size=800, replace=False)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled[sample_idx],
                                            km.labels_[sample_idx]))
    best_k = ks[int(np.argmax(silhouettes))]
    print(f"K-means: best k by silhouette = {best_k}")
    kmeans = KMeans(n_clusters=10, n_init=10, random_state=42).fit(X_scaled)
    print(f"  k=10 ARI vs true labels = {adjusted_rand_score(y, kmeans.labels_):.3f}")

    # ------------------------------------------------------------------
    # 2. Hierarchical clustering on a subset (dendrogram)
    # ------------------------------------------------------------------
    subset = rng.choice(len(X_scaled), size=80, replace=False)
    Z = linkage(X_scaled[subset], method="ward")
    print(f"  hierarchical ARI (10 clusters) = "
          f"{adjusted_rand_score(y[subset], AgglomerativeClustering(n_clusters=10).fit_predict(X_scaled[subset])):.3f}")

    # ------------------------------------------------------------------
    # 3. DBSCAN
    # ------------------------------------------------------------------
    db = DBSCAN(eps=4.0, min_samples=8).fit(X_scaled)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"  DBSCAN clusters = {n_clusters}, noise points = {np.sum(db.labels_ == -1)}")

    # ------------------------------------------------------------------
    # 4. PCA and t-SNE
    # ------------------------------------------------------------------
    pca = PCA(n_components=0.90, random_state=42).fit(X_scaled)
    print(f"  PCA components for 90% variance = {pca.n_components_}")
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca",
                learning_rate="auto")
    X_tsne = tsne.fit_transform(X_scaled)

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    axes[0, 0].plot(ks, inertias, marker="o", label="inertia")
    ax2 = axes[0, 0].twinx()
    ax2.plot(ks, silhouettes, marker="s", color="darkorange", label="silhouette")
    axes[0, 0].set(xlabel="Number of clusters", ylabel="Inertia",
                   title="Elbow and silhouette")
    ax2.set_ylabel("Silhouette")
    axes[0, 0].axvline(10, color="gray", linestyle=":")

    dendrogram(Z, ax=axes[0, 1], no_labels=True, color_threshold=0)
    axes[0, 1].set(title="Ward dendrogram (80 samples)", xlabel="Sample")

    scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", s=8)
    axes[1, 0].set(title="PCA projection", xlabel="PC 1", ylabel="PC 2")

    axes[1, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", s=8)
    axes[1, 1].set(title="t-SNE projection", xlabel="dimension 1", ylabel="dimension 2")

    fig.colorbar(scatter, ax=list(axes[1, :]), label="true digit", shrink=0.8)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
