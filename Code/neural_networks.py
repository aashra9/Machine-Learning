"""Neural-network companion script.

Implements a perceptron and a one-hidden-layer MLP from scratch with NumPy, then
visualises the learned decision boundary on the two-moons problem.  Run from the
repository root:

    python Code/neural_networks.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from plotting_utils import plot_score_boundary
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def _check_fitted(model: object, attributes: tuple[str, ...]) -> None:
    """Raise a helpful error if ``model`` has not been fitted yet."""
    if any(getattr(model, name, None) is None for name in attributes):
        raise RuntimeError(f"{type(model).__name__} must be fitted before prediction")


class Perceptron:
    """Classic perceptron with a step activation and binary labels.

    Accepts labels in either {0, 1} or {-1, 1} and returns predictions in the
    same encoding used during fitting.
    """

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Perceptron:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if not set(np.unique(y)).issubset({0, 1, -1}):
            raise ValueError("Perceptron expects binary labels in {0, 1} or {-1, 1}")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("Perceptron needs exactly two classes")
        self.classes_ = np.array([classes.min(), classes.max()])
        y_signed = np.where(y > 0, 1.0, -1.0)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        for _ in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y_signed):
                if yi * (xi @ self.weights + self.bias) <= 0:
                    self.weights += self.learning_rate * yi * xi
                    self.bias += self.learning_rate * yi
                    errors += 1
            if errors == 0:
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        _check_fitted(self, ("weights", "classes_"))
        signed = np.asarray(X) @ self.weights + self.bias > 0
        return np.where(signed, self.classes_[1], self.classes_[0])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(accuracy_score(y, self.predict(X)))


class MLP:
    """One-hidden-layer MLP with tanh hidden units and a sigmoid output."""

    EPS = 1e-12

    def __init__(self, n_inputs: int, n_hidden: int = 16, learning_rate: float = 0.1,
                 n_epochs: int = 2000, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 1.0 / np.sqrt(n_inputs), size=(n_hidden, n_inputs))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0.0, 1.0 / np.sqrt(n_hidden), size=n_hidden)
        self.b2 = 0.0
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.fitted_ = False
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z1 = X @ self.W1.T + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        return self._sigmoid(z2), a1

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLP:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        n = len(X)
        for _ in range(self.n_epochs):
            y_hat, a1 = self._forward(X)
            loss = -np.mean(y * np.log(y_hat + self.EPS)
                            + (1 - y) * np.log(1 - y_hat + self.EPS))
            self.loss_history.append(float(loss))

            dz2 = (y_hat - y) / n
            dW2 = a1.T @ dz2
            db2 = dz2.sum()
            da1 = np.outer(dz2, self.W2)
            dz1 = da1 * (1.0 - a1 ** 2)
            dW1 = dz1.T @ X
            db1 = dz1.sum(axis=0)

            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
        self.fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("MLP must be fitted before prediction")
        return self._forward(np.asarray(X, dtype=float))[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(accuracy_score(y, self.predict(X)))


def plot_boundary(
    ax: Axes, predict, X: np.ndarray, y: np.ndarray, title: str,
    resolution: int = 300,
) -> None:
    """Plot a probability surface and the samples (shared implementation)."""
    plot_score_boundary(ax, predict, X, y, title, resolution=resolution)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--resolution", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Perceptron on XOR: a linear model cannot solve it
    # ------------------------------------------------------------------
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([0, 1, 1, 0])
    perceptron = Perceptron(n_epochs=100).fit(X_xor, y_xor)
    print("Perceptron on XOR")
    print(f"  predictions = {perceptron.predict(X_xor)}")
    print(f"  accuracy    = {perceptron.score(X_xor, y_xor):.2f}")

    # ------------------------------------------------------------------
    # 2. MLP on two moons
    # ------------------------------------------------------------------
    X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    mlp = MLP(n_inputs=2, n_hidden=16, learning_rate=0.5,
              n_epochs=args.epochs, seed=0).fit(X_train_s, y_train.astype(float))

    sklearn_mlp = MLPClassifier(
        hidden_layer_sizes=(16,), activation="tanh", solver="adam",
        max_iter=3000, random_state=42,
    ).fit(X_train_s, y_train)

    print("\nMLP on two moons")
    print(f"  NumPy MLP accuracy    = {mlp.score(X_test_s, y_test):.4f}")
    print(f"  sklearn MLP accuracy  = {sklearn_mlp.score(X_test_s, y_test):.4f}")

    if args.no_plots:
        return

    # ------------------------------------------------------------------
    # 3. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1],
                    c="tab:red", s=100, label="class 0")
    axes[0].scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
                    c="tab:blue", s=100, label="class 1")
    axes[0].set(title="XOR is not linearly separable", xlabel="x1", ylabel="x2")
    axes[0].legend()

    plot_boundary(axes[1], mlp.predict_proba, X_train_s, y_train,
                  "NumPy MLP decision boundary", resolution=args.resolution)
    axes[1].scatter(X_test_s[:, 0], X_test_s[:, 1], c=y_test, cmap="coolwarm",
                    edgecolors="k", marker="^", s=25)

    axes[2].plot(mlp.loss_history)
    axes[2].set(xlabel="Epoch", ylabel="Binary cross-entropy", title="Training loss")
    axes[2].set_yscale("log")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
