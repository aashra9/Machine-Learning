"""k-Nearest Neighbours companion script.

Bridges the k-NN notebooks with the other command-line companions: it implements
the classifier from scratch, checks it against scikit-learn, demonstrates why
feature scaling matters, tunes ``k`` by cross-validation, and draws the resulting
decision regions.  Run from the repository root:

    python Code/knn.py --no-plots
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from knn_utils import best_row, build_knn_pipeline, plot_k_curve, tune_k
from matplotlib.axes import Axes
from plotting_utils import decision_grid
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split


def pairwise_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Euclidean distance from every row of ``X`` to every row of ``Y``."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    return np.sqrt(
        np.maximum(
            np.sum(X ** 2, axis=1)[:, None]
            + np.sum(Y ** 2, axis=1)[None, :]
            - 2 * X @ Y.T,
            0.0,
        )
    )


class KNNClassifier:
    """Minimal k-NN classifier implemented with NumPy only."""

    def __init__(self, k: int = 5, weights: str = "uniform") -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k = k
        self.weights = weights
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNClassifier:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self.X_train, self.y_train = X, y
        self.classes_ = np.unique(y)
        if self.k > len(X):
            raise ValueError("k cannot exceed the number of training samples")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None or self.classes_ is None:
            raise RuntimeError("KNNClassifier must be fitted before prediction")
        distances = pairwise_distances(X, self.X_train)
        nearest = np.argpartition(distances, self.k - 1, axis=1)[:, :self.k]
        weights = (np.ones_like(distances[:, :self.k])
                   if self.weights == "uniform"
                   else 1.0 / np.maximum(distances[
                       np.arange(len(X))[:, None], nearest], 1e-12))
        proba = np.zeros((len(X), len(self.classes_)))
        for class_idx, cls in enumerate(self.classes_):
            mask = self.y_train[nearest] == cls
            proba[:, class_idx] = np.sum(weights * mask, axis=1) / np.sum(
                weights, axis=1
            )
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def plot_regions(
    ax: Axes,
    model: KNNClassifier,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    resolution: int = 200,
) -> None:
    """Draw the 1-NN-style decision regions on the first two features."""
    xx, yy, grid = decision_grid(X, resolution=resolution)
    labels = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, labels, alpha=0.3, cmap="tab10")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", edgecolors="k", s=20)
    ax.set(title=title, xlabel="feature 1", ylabel="feature 2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dataset", choices=["iris", "synthetic"], default="iris")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    if args.dataset == "iris":
        data = load_iris()
        X, y = data.data[:, :2], data.target
        names = data.target_names
    else:
        X, y = make_classification(
            n_samples=400, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, class_sep=1.2, random_state=42,
        )
        names = np.array(["class 0", "class 1"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"Dataset: {args.dataset} — train={X_train.shape}, test={X_test.shape}")

    # ------------------------------------------------------------------
    # 2. From scratch versus scikit-learn
    # ------------------------------------------------------------------
    scratch = KNNClassifier(k=args.k).fit(X_train, y_train)
    library = build_knn_pipeline(k=args.k).fit(X_train, y_train)
    print("\nScratch implementation")
    print(f"  from-scratch accuracy = {accuracy_score(y_test, scratch.predict(X_test)):.4f}")
    print(f"  sklearn accuracy      = {accuracy_score(y_test, library.predict(X_test)):.4f}")

    # ------------------------------------------------------------------
    # 3. Effect of scaling
    # ------------------------------------------------------------------
    raw_model = build_knn_pipeline(k=args.k, scale=False).fit(X_train, y_train)
    scaled_model = build_knn_pipeline(k=args.k, scale=True).fit(X_train, y_train)
    print("\nEffect of standardisation")
    print(f"  raw features    = {accuracy_score(y_test, raw_model.predict(X_test)):.4f}")
    print(f"  scaled features = {accuracy_score(y_test, scaled_model.predict(X_test)):.4f}")

    # ------------------------------------------------------------------
    # 4. Tune k by cross-validation
    # ------------------------------------------------------------------
    results = tune_k(X_train, y_train, ks=range(1, 21), cv=5)
    best = best_row(results)
    print("\nCross-validated tuning")
    print(f"  best k = {int(best['k'])} (score = {best['mean_score']:.4f})")

    if args.no_plots:
        return

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    plot_regions(axes[0], scratch, X_train, y_train, f"Scratch k-NN (k={args.k})")
    plot_regions(
        axes[1],
        KNNClassifier(k=int(best["k"])).fit(X_train, y_train),
        X_train, y_train, f"Best k = {int(best['k'])}",
    )
    plot_k_curve(results, ax=axes[2])
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, scratch.predict(X_test),
        display_labels=names, cmap="Blues", ax=ax,
    )
    ax.set_title("k-NN confusion matrix")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
