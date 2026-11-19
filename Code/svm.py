"""Support vector machine companion script.

Visualises maximum-margin decision boundaries on a two-dimensional problem and
tunes an RBF SVM on a real dataset.  Run from the repository root:

    python Code/svm.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from plotting_utils import plot_margin_boundary


def plot_boundary(
    ax: Axes, model: SVC, X, y, title: str, resolution: int = 300
) -> None:
    """Plot an SVM decision boundary and highlight its support vectors."""
    plot_margin_boundary(ax, model, X, y, title, resolution=resolution)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Two-dimensional illustration of the margin
    # ------------------------------------------------------------------
    X, y = make_moons(n_samples=200, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    linear_svm = SVC(kernel="linear", C=args.c).fit(X_train, y_train)
    rbf_svm = SVC(kernel="rbf", C=args.c, gamma=args.gamma).fit(X_train, y_train)
    print("Two-moons SVM")
    print(f"  linear accuracy = {accuracy_score(y_test, linear_svm.predict(X_test)):.3f}")
    print(f"  RBF    accuracy = {accuracy_score(y_test, rbf_svm.predict(X_test)):.3f}")
    print(f"  linear support vectors = {len(linear_svm.support_)} / {len(X_train)}")
    print(f"  RBF    support vectors = {len(rbf_svm.support_)} / {len(X_train)}")

    # ------------------------------------------------------------------
    # 2. Hyperparameter search with cross-validation
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target, random_state=42
    )
    pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42))
    param_grid = {
        "svc__C": [0.1, 1.0, 10.0, 100.0],
        "svc__gamma": [0.001, 0.01, 0.1, 1.0],
    }
    search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1,
        return_train_score=True,
    )
    search.fit(Xc_train, yc_train)
    print("\nRBF SVM grid search (breast cancer)")
    print(f"  best params = {search.best_params_}")
    print(f"  CV accuracy = {search.best_score_:.4f}")
    print(f"  test accuracy = {search.score(Xc_test, yc_test):.4f}")

    if args.no_plots:
        return

    # ------------------------------------------------------------------
    # 3. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    plot_boundary(axes[0], linear_svm, X_train, y_train, "Linear SVM",
                  resolution=args.resolution)
    plot_boundary(axes[1], rbf_svm, X_train, y_train, "RBF SVM",
                  resolution=args.resolution)

    ConfusionMatrixDisplay.from_estimator(
        search.best_estimator_, Xc_test, yc_test,
        display_labels=data.target_names, cmap="Blues", colorbar=False, ax=axes[2],
    )
    axes[2].set_title("Tuned RBF SVM on breast cancer")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
