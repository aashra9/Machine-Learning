"""Reusable matplotlib helpers for the ML companion scripts.

Keeping the mesh-grid and contour code here removes the near-identical
implementations that used to live in ``svm.py`` and ``neural_networks.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def decision_grid(
    X: np.ndarray,
    resolution: int = 300,
    padding: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a regular mesh covering the first two columns of ``X``.

    Returns ``(xx, yy, grid)`` where ``grid`` has shape ``(resolution**2, 2)``
    and can be passed straight to an estimator's ``predict`` method.
    """
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("decision_grid expects a 2-D array with at least two columns")
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid


def score_grid(
    model: object,
    grid: np.ndarray,
) -> np.ndarray:
    """Return a one-dimensional score for every point in ``grid``.

    Preference order: ``decision_function`` (raw margins for binary models),
    ``predict_proba`` (positive-class probability), then hard ``predict`` labels.
    Multiclass outputs are reduced to predicted class indices so that all
    callers receive a 1-D array.
    """
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(grid))
        if scores.ndim == 1:
            return scores
        if scores.shape[1] == 2:
            return scores[:, 1] - scores[:, 0]
        return np.argmax(scores, axis=1).astype(float)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(grid))
        if proba.ndim == 1:
            return proba
        if proba.shape[1] == 2:
            return proba[:, 1]
        return np.argmax(proba, axis=1).astype(float)
    return np.asarray(model.predict(grid), dtype=float)  # type: ignore[attr-defined]


def plot_score_boundary(
    ax: Axes,
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    title: str = "",
    resolution: int = 300,
    padding: float = 0.5,
    levels: int = 20,
    cmap: str = "coolwarm",
    alpha: float = 0.3,
) -> None:
    """Draw a continuous score surface, its 0.5 contour, and the samples.

    ``predict`` must accept an ``(n, 2)`` array and return one score per row.
    """
    xx, yy, grid = decision_grid(X, resolution, padding)
    zz = np.asarray(predict(grid), dtype=float).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, alpha=alpha)
    if zz.min() <= 0.5 <= zz.max():
        ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors="k", s=25)
    ax.set(title=title, xlabel="feature 1", ylabel="feature 2")


def plot_margin_boundary(
    ax: Axes,
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "",
    resolution: int = 300,
    padding: float = 0.5,
    highlight_support_vectors: bool = True,
) -> None:
    """Draw an SVM-style margin boundary and optionally mark support vectors."""
    xx, yy, grid = decision_grid(X, resolution, padding)
    scores = score_grid(model, grid).reshape(xx.shape)
    ax.contourf(
        xx, yy, scores, levels=[-1e9, 0, 1e9], alpha=0.15,
        colors=["tab:red", "tab:blue"],
    )
    ax.contour(
        xx, yy, scores, levels=[-1, 0, 1], colors="k",
        linestyles=["--", "-", "--"], linewidths=[1, 2, 1],
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=30)
    if highlight_support_vectors and hasattr(model, "support_vectors_"):
        sv = np.asarray(model.support_vectors_)
        ax.scatter(
            sv[:, 0], sv[:, 1], s=120, facecolors="none", edgecolors="green",
            linewidths=1.5, label="support vectors",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set(title=title, xlabel="feature 1", ylabel="feature 2")


def new_figure(n_cols: int, width_per_axes: float = 5.0, height: float = 4.8):
    """Small convenience wrapper around ``plt.subplots`` for 1-row layouts."""
    return plt.subplots(1, n_cols, figsize=(width_per_axes * n_cols, height))


__all__ = [
    "decision_grid",
    "score_grid",
    "plot_score_boundary",
    "plot_margin_boundary",
    "new_figure",
]
