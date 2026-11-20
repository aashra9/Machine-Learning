"""Shared helpers for the k-nearest-neighbours notebooks and companion script.

The k-NN worksheets historically duplicated these functions in every notebook.
Importing them from one module keeps the notebooks short and makes the helpers
importable (and therefore testable) outside Jupyter.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.datasets import fetch_openml, load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_knn_pipeline(
    k: int = 5,
    scale: bool = True,
    weights: str = "uniform",
    metric: str = "minkowski",
    task: str = "classification",
    **model_kwargs: Any,
) -> Pipeline:
    """Return a scikit-learn pipeline with optional scaling and a k-NN model."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if task == "classification":
        estimator: Any = KNeighborsClassifier(
            n_neighbors=k, weights=weights, metric=metric, **model_kwargs
        )
    elif task == "regression":
        estimator = KNeighborsRegressor(
            n_neighbors=k, weights=weights, metric=metric, **model_kwargs
        )
    else:
        raise ValueError("task must be 'classification' or 'regression'")
    steps = [("scale", StandardScaler())] if scale else []
    steps.append(("knn", estimator))
    return Pipeline(steps)


def tune_k(
    X: np.ndarray,
    y: np.ndarray,
    ks: range | list[int] | np.ndarray = range(1, 31),
    cv: int = 5,
    scale: bool = True,
    metric: str = "minkowski",
    weights: str = "uniform",
    scoring: str = "accuracy",
    n_jobs: int | None = None,
) -> dict[str, np.ndarray]:
    """Cross-validate k-NN for several values of ``k``.

    Returns a dictionary with ``k``, ``mean_score`` and ``std_score`` arrays.
    """
    rows = {"k": [], "mean_score": [], "std_score": []}
    for k in ks:
        model = build_knn_pipeline(
            k=int(k), scale=scale, metric=metric, weights=weights
        )
        scores = cross_val_score(
            model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs
        )
        rows["k"].append(int(k))
        rows["mean_score"].append(float(scores.mean()))
        rows["std_score"].append(float(scores.std()))
    return {key: np.asarray(values) for key, values in rows.items()}


def best_row(results: dict[str, np.ndarray]) -> dict[str, float]:
    """Return the k with the highest mean cross-validated score."""
    idx = int(np.argmax(results["mean_score"]))
    return {
        "k": float(results["k"][idx]),
        "mean_score": float(results["mean_score"][idx]),
        "std_score": float(results["std_score"][idx]),
    }


def plot_k_curve(
    results: dict[str, np.ndarray],
    ax: Axes | None = None,
    title: str = "Cross-validated score versus k",
) -> Axes:
    """Plot mean cross-validated score with a +/-1 standard-deviation band."""
    if ax is None:
        _, ax = plt.subplots()
    mean = results["mean_score"]
    std = results["std_score"]
    ax.plot(results["k"], mean, marker="o", markersize=3, label="mean score")
    ax.fill_between(results["k"], mean - std, mean + std, alpha=0.2)
    best = best_row(results)
    ax.axvline(best["k"], color="red", linestyle="--",
               label=f"best k = {int(best['k'])}")
    ax.set(xlabel="k", ylabel="score", title=title)
    ax.legend()
    return ax


def fit_and_time(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Fit ``model``, measure prediction speed, and return accuracy/timings."""
    model.fit(X_train, y_train)
    model.predict(X_test[:5])
    start = time.perf_counter()
    predictions = model.predict(X_test)
    elapsed = time.perf_counter() - start
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "fit_predict_seconds": float(elapsed),
        "ms_per_query": 1000.0 * elapsed / max(len(X_test), 1),
    }


def load_mnist(
    force_digits: bool | None = None,
    download: bool = True,
    data_home: str | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load MNIST from OpenML, falling back to the small ``load_digits`` set.

    ``force_digits`` defaults to the ``KNN_FORCE_DIGITS`` environment variable.
    Returns ``(X, y, used_mnist)``.
    """
    if force_digits is None:
        force_digits = os.environ.get("KNN_FORCE_DIGITS", "0").lower() not in {
            "0", "", "false"
        }
    if force_digits or not download:
        digits = load_digits()
        return digits.data, digits.target, False
    try:
        mnist = fetch_openml(
            "mnist_784", version=1, as_frame=False,
            parser="auto", data_home=data_home,
        )
        return np.asarray(mnist.data), mnist.target.astype(np.uint8), True
    except Exception:
        digits = load_digits()
        return digits.data, digits.target, False


def plot_image_gallery(
    images: np.ndarray,
    labels: np.ndarray,
    image_shape: tuple[int, int] | None = None,
    n_per_class: int = 5,
    classes: np.ndarray | None = None,
    ax: Axes | np.ndarray | None = None,
    title: str = "Sample images",
) -> Axes | np.ndarray:
    """Show up to ``n_per_class`` examples of each class in a grid.

    ``ax`` may be a 2-D array of axes; a single ``Axes`` is accepted only when
    the gallery contains exactly one image.
    """
    if classes is None:
        classes = np.unique(labels)
    if image_shape is None:
        side = int(np.sqrt(images.shape[1]))
        image_shape = (side, side)
    n_rows, n_cols = len(classes), n_per_class
    if ax is None:
        _, ax = plt.subplots(n_rows, n_cols,
                             figsize=(1.6 * n_cols, 1.6 * n_rows))
    if isinstance(ax, Axes):
        if n_rows * n_cols != 1:
            raise ValueError("pass a 2-D axes array for a multi-image gallery")
        axes = np.array([[ax]])
    else:
        axes = np.atleast_2d(ax)
    for row, cls in enumerate(classes):
        idx = np.where(labels == cls)[0][:n_per_class]
        for col in range(n_cols):
            current = axes[row, col]
            current.axis("off")
            if col < len(idx):
                current.imshow(images[idx[col]].reshape(image_shape),
                               cmap="gray", vmin=0, vmax=1)
                current.set_title(f"{cls}", fontsize=9)
    axes.flat[0].figure.suptitle(title, fontsize=10, x=0.01, ha="left")
    return ax


def explained_variance_curve(
    X: np.ndarray,
    max_components: int = 50,
    ax: Axes | None = None,
) -> Axes:
    """Plot cumulative explained variance for a PCA projection."""
    pca = PCA(n_components=min(max_components, X.shape[1]), random_state=42)
    pca.fit(X)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
    ax.axhline(0.9, color="gray", linestyle="--", label="90% variance")
    ax.set(xlabel="Number of components", ylabel="Cumulative explained variance",
           title="PCA explained variance")
    ax.legend()
    return ax


def find_class_dirs(root: str | Path, max_classes: int | None = None) -> dict[str, Path]:
    """Return ``{class_name: folder}`` for immediate subdirectories of ``root``."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    dirs = {p.name: p for p in sorted(root.iterdir()) if p.is_dir()}
    if not dirs:
        raise ValueError(f"No class folders found in {root}")
    if max_classes is not None:
        dirs = dict(list(dirs.items())[:max_classes])
    return dirs


def load_image_folder_dataset(
    root: str | Path,
    image_size: tuple[int, int] = (64, 64),
    max_per_class: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load images from ``class_name/`` subfolders using Pillow.

    Each image is resized to ``image_size`` and flattened to ``[0, 1]`` RGB.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for image-folder loading: pip install pillow"
        ) from exc
    class_dirs = find_class_dirs(root)
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for class_idx, (name, folder) in enumerate(class_dirs.items()):
        paths = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if max_per_class is not None and len(paths) > max_per_class:
            paths = list(rng.choice(paths, size=max_per_class, replace=False))
        for path in paths:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("RGB").resize(image_size),
                                 dtype=np.float32) / 255.0
            images.append(arr.ravel())
            labels.append(class_idx)
    if not images:
        raise ValueError(f"No images found under {root}")
    return np.asarray(images), np.asarray(labels), list(class_dirs)


def make_synthetic_images(
    n_per_class: int = 40,
    image_size: tuple[int, int] = (32, 32),
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create a tiny deterministic image dataset when no real folder exists."""
    rng = np.random.default_rng(seed)
    height, width = image_size
    yy, xx = np.mgrid[0:height, 0:width]
    images, labels = [], []
    for cls in range(n_classes):
        center_y = height * (0.25 + 0.25 * cls)
        center_x = width * (0.25 + 0.25 * (cls % 2))
        for _ in range(n_per_class):
            noise = rng.normal(0.0, 0.08, size=(height, width))
            blob = np.exp(-((yy - center_y) ** 2 + (xx - center_x) ** 2)
                          / (2 * (0.12 * height) ** 2))
            images.append(np.clip(blob + noise, 0.0, 1.0).ravel())
            labels.append(cls)
    return np.asarray(images), np.asarray(labels), [f"synthetic_{i}" for i in range(n_classes)]


def distance_concentration(
    X: np.ndarray,
    n_query: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Measure how concentrated nearest-neighbour distances are in ``X``."""
    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(X), size=min(n_query, len(X)), replace=False)
    queries = X[query_idx]
    all_distances = np.linalg.norm(X[:, None, :] - queries[None, :, :], axis=2)
    all_distances[query_idx, np.arange(len(query_idx))] = np.inf
    nearest = all_distances.min(axis=0)
    # Self-matches are excluded from both statistics; including the infinities
    # here would make the farthest distance infinite whenever every point is a query.
    farthest = np.where(np.isinf(all_distances), -np.inf, all_distances).max(axis=0)
    # A single-point dataset has no non-self neighbour, so report zero distances.
    no_neighbour = ~np.isfinite(farthest)
    nearest = np.where(no_neighbour, 0.0, nearest)
    farthest = np.where(no_neighbour, 0.0, farthest)
    return {
        "mean_nearest": float(nearest.mean()),
        "mean_farthest": float(farthest.mean()),
        "ratio": float(nearest.mean() / max(farthest.mean(), 1e-12)),
    }


__all__ = [
    "build_knn_pipeline",
    "tune_k",
    "best_row",
    "plot_k_curve",
    "fit_and_time",
    "load_mnist",
    "plot_image_gallery",
    "explained_variance_curve",
    "find_class_dirs",
    "load_image_folder_dataset",
    "make_synthetic_images",
    "distance_concentration",
]
