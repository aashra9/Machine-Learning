"""Support vector machine companion script.

Visualises maximum-margin decision boundaries on a two-dimensional problem and
tunes an RBF SVM on a real dataset.  Run from the repository root:

    python Code/svm.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def plot_boundary(ax, model, X, y, title: str) -> None:
    """Plot a decision boundary and highlight support vectors."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = model.decision_function(grid).reshape(xx.shape)

    ax.contourf(xx, yy, scores, levels=[-1e9, 0, 1e9], alpha=0.15,
                colors=["tab:red", "tab:blue"])
    ax.contour(xx, yy, scores, levels=[-1, 0, 1], colors="k",
               linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=30)
    if hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors="none",
                   edgecolors="green", linewidths=1.5, label="support vectors")
        ax.legend(loc="upper right", fontsize=8)
    ax.set(title=title, xlabel="feature 1", ylabel="feature 2")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Two-dimensional illustration of the margin
    # ------------------------------------------------------------------
    X, y = make_moons(n_samples=200, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    linear_svm = SVC(kernel="linear", C=1.0).fit(X_train, y_train)
    rbf_svm = SVC(kernel="rbf", C=1.0, gamma=1.0).fit(X_train, y_train)
    print("Two-moons SVM")
    print(f"  linear accuracy = {accuracy_score(y_test, linear_svm.predict(X_test)):.3f}")
    print(f"  RBF    accuracy = {accuracy_score(y_test, rbf_svm.predict(X_test)):.3f}")
    print(f"  linear support vectors = {len(linear_svm.support_)} / {len(X_train)}")

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
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(Xc_train, yc_train)
    print("\nRBF SVM grid search (breast cancer)")
    print(f"  best params = {search.best_params_}")
    print(f"  CV accuracy = {search.best_score_:.4f}")
    print(f"  test accuracy = {search.score(Xc_test, yc_test):.4f}")

    # ------------------------------------------------------------------
    # 3. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    plot_boundary(axes[0], linear_svm, X_train, y_train, "Linear SVM")
    plot_boundary(axes[1], rbf_svm, X_train, y_train, "RBF SVM")

    ConfusionMatrixDisplay.from_estimator(
        search.best_estimator_, Xc_test, yc_test,
        display_labels=data.target_names, cmap="Blues", colorbar=False, ax=axes[2],
    )
    axes[2].set_title("Tuned RBF SVM on breast cancer")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
