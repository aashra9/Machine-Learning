"""MNIST supervised-learning example.

Demonstrates supervised learning on handwritten digit classification.  Uses
scikit-learn's ``fetch_openml`` for MNIST and falls back to ``load_digits``
offline.  Run from the repository root:

    python Code/mnist_supervised.py
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def load_digit_data(force_digits: bool = False) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load MNIST (or the small digits fallback) and return ``(X, y, used_mnist)``."""
    if force_digits:
        digits = load_digits()
        return digits.data, digits.target.astype(np.uint8), False
    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        return (np.asarray(mnist.data), mnist.target.astype(np.uint8), True)
    except Exception as exc:
        print(f"Fell back to load_digits: {exc}")
        digits = load_digits()
        return digits.data, digits.target.astype(np.uint8), False


def plot_gallery(
    X: np.ndarray, y: np.ndarray, image_shape: tuple[int, int],
    n_show: int = 10, title: str = "Sample digits",
    predictions: np.ndarray | None = None,
) -> None:
    """Show one example per digit (or misclassified samples)."""
    n_cols = 5
    n_rows = max(1, int(np.ceil(n_show / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat:
        ax.axis("off")
    for ax, idx in zip(axes_flat[:n_show], range(n_show)):
        ax.imshow(X[idx].reshape(image_shape), cmap="gray")
        label = f"True: {y[idx]}" if predictions is None else (
            f"True: {y[idx]}  Pred: {predictions[idx]}"
        )
        color = "red" if predictions is not None else "black"
        ax.set_title(label, color=color, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def make_models(seed: int = 42) -> dict:
    """Return the classifiers compared in the demo."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed
        ),
    }


def evaluate_models(models: dict, X_train: np.ndarray, X_test: np.ndarray,
                    y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Fit every model and return its test accuracy."""
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = float(acc)
        print(f"{name:18s}  test accuracy = {acc:.4f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force-digits", action="store_true",
                        help="use scikit-learn's 8x8 digits dataset instead of MNIST")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of training samples (useful for a quick run)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    X, y, used_mnist = load_digit_data(args.force_digits)
    print(f"Loaded {'MNIST' if used_mnist else 'load_digits'}")
    print(f"X shape: {X.shape}  y shape: {y.shape}")
    print(f"Pixel range: [{X.min()}, {X.max()}]")
    print(f"Classes: {np.unique(y)}")
    print(f"Class counts: {np.bincount(y)}")

    side = int(np.sqrt(X.shape[1]))
    image_shape = (side, side)
    max_value = float(X.max())
    if max_value <= 0:
        raise ValueError("all pixel values are zero")
    X = X / max_value

    if not args.no_plots:
        samples = []
        for digit in np.unique(y):
            idx = np.where(y == digit)[0]
            if len(idx):
                samples.append(idx[0])
        if samples:
            sample_X = X[np.array(samples)]
            sample_y = y[np.array(samples)]
            plot_gallery(sample_X, sample_y, image_shape,
                         n_show=len(samples), title="One example per digit")

    test_size = 10000 if used_mnist and len(y) >= 20000 else 0.2
    train_size = 60000 if used_mnist and len(y) >= 70000 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, train_size=train_size,
        random_state=42, stratify=y,
    )
    if args.limit is not None and args.limit < len(X_train):
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test:  {X_test.shape},  {y_test.shape}")

    models = make_models()
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    best_name = max(results, key=results.get)
    best_model = models[best_name]
    y_pred = best_model.predict(X_test)
    print(f"\nBest model: {best_name} (accuracy = {results[best_name]:.4f})\n")
    print(classification_report(y_test, y_pred, digits=4))

    if args.no_plots:
        return

    names, accs = list(results), list(results.values())
    plt.bar(names, accs, color="steelblue")
    plt.ylim(0, 1)
    plt.ylabel("Test accuracy")
    plt.title("MNIST Classifier Comparison")
    plt.xticks(rotation=15)
    for i, value in enumerate(accs):
        plt.text(i, value + 0.01, f"{value:.4f}", ha="center")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    plt.show()

    wrong = np.where(y_pred != y_test)[0]
    print(f"Misclassified: {len(wrong)} / {len(y_test)} test samples")
    n_show = min(10, len(wrong))
    if n_show:
        plot_gallery(
            X_test[wrong[:n_show]], y_test[wrong[:n_show]], image_shape,
            n_show=n_show, title="Misclassified digits",
            predictions=y_pred[wrong[:n_show]],
        )
    print("\nDone!")


if __name__ == "__main__":
    main()
