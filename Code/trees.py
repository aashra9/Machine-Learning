"""Decision trees, random forests, and gradient boosting companion script.

Run from the repository root:

    python Code/trees.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree


def make_models(seed: int = 42) -> dict[str, BaseEstimator]:
    """Return the tree-based models compared in the demo."""
    return {
        "Decision tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=seed),
        "Decision tree (full)": DecisionTreeClassifier(random_state=seed),
        "Random forest": RandomForestClassifier(
            n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=seed
        ),
        "Gradient boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=2, random_state=seed
        ),
    }


def compare_models(
    models: dict[str, BaseEstimator], X: np.ndarray, y: np.ndarray, cv: StratifiedKFold
) -> dict[str, np.ndarray]:
    """Cross-validate each model and return its fold scores."""
    return {
        name: cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        for name, model in models.items()
    }


def depth_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    depths: np.ndarray,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Measure train and validation accuracy for a grid of tree depths.

    The test set is deliberately not passed in: using it to choose ``max_depth``
    would leak information and invalidate the final test accuracy.
    """
    train_acc, valid_acc = [], []
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=int(depth), random_state=seed)
        tree.fit(X_train, y_train)
        train_acc.append(float(tree.score(X_train, y_train)))
        valid_acc.append(float(tree.score(X_valid, y_valid)))
    return train_acc, valid_acc


def plot_importance_bars(
    ax: Axes, values: np.ndarray, names: np.ndarray, title: str,
    errors: np.ndarray | None = None, top: int = 10,
) -> None:
    """Draw a horizontal bar chart of the ``top`` most important features."""
    order = np.argsort(values)[-top:]
    ax.barh(np.arange(len(order)), values[order],
            xerr=None if errors is None else errors[order],
            color="steelblue")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set(xlabel=title.split("(")[0].strip(), title=title)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    X_inner, X_valid, y_inner, y_valid = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=7
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    # ------------------------------------------------------------------
    # 1. Compare tree-based models
    # ------------------------------------------------------------------
    models = make_models()
    print("Cross-validated accuracy")
    cv_scores = compare_models(models, X_inner, y_inner, cv)
    for name, scores in cv_scores.items():
        print(f"  {name:24s} {scores.mean():.4f} +/- {scores.std():.4f}")

    # ------------------------------------------------------------------
    # 2. Tree complexity: validation curve over depth
    # ------------------------------------------------------------------
    depths = np.arange(1, 13)
    train_acc, valid_acc = depth_sweep(X_inner, y_inner, X_valid, y_valid, depths)

    # ------------------------------------------------------------------
    # 3. Feature importance (fit on the full training set)
    # ------------------------------------------------------------------
    forest = make_models()["Random forest"].fit(X_train, y_train)
    impurity_importance = forest.feature_importances_
    perm = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    print(f"\nRandom forest test accuracy = {forest.score(X_test, y_test):.4f}")
    print("Top features by impurity importance:")
    for idx in np.argsort(impurity_importance)[-10:][::-1]:
        print(f"  {data.feature_names[idx]:22s} {impurity_importance[idx]:.4f}")

    if args.no_plots:
        return

    # ------------------------------------------------------------------
    # 4. Plots (one blocking call)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_tree(
        models["Decision tree (depth 3)"].fit(X_train, y_train),
        feature_names=data.feature_names,
        class_names=data.target_names,
        filled=True, rounded=True, fontsize=6, ax=axes[0, 0],
    )
    axes[0, 0].set_title("Decision tree (depth 3)")

    axes[0, 1].plot(depths, train_acc, marker="o", label="Training")
    axes[0, 1].plot(depths, valid_acc, marker="o", label="Validation")
    axes[0, 1].set(xlabel="Maximum depth", ylabel="Accuracy",
                   title="Tree complexity (held-out validation)")
    axes[0, 1].legend()

    plot_importance_bars(axes[1, 0], impurity_importance, data.feature_names,
                         "Impurity importance")
    plot_importance_bars(axes[1, 1], perm.importances_mean, data.feature_names,
                         "Permutation importance (test set)",
                         errors=perm.importances_std)
    fig.tight_layout()

    ConfusionMatrixDisplay.from_estimator(
        forest, X_test, y_test, display_labels=data.target_names, cmap="Blues"
    )
    plt.title("Random forest confusion matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
