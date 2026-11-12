"""Neural-network companion script.

Implements a perceptron and a one-hidden-layer MLP from scratch with NumPy, then
visualises the learned decision boundary on the two-moons problem.  Run from the
repository root:

    python Code/neural_networks.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class Perceptron:
    """Classic perceptron with a step activation."""

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights: np.ndarray | None = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
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
        return (X @ self.weights + self.bias > 0).astype(int)


class MLP:
    """One-hidden-layer MLP with tanh hidden units and a sigmoid output."""

    def __init__(self, n_inputs: int, n_hidden: int = 16, learning_rate: float = 0.1,
                 n_epochs: int = 2000, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 1.0 / np.sqrt(n_inputs), size=(n_hidden, n_inputs))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0.0, 1.0 / np.sqrt(n_hidden), size=n_hidden)
        self.b2 = 0.0
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z1 = X @ self.W1.T + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        return self._sigmoid(z2), a1

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        n = len(X)
        for _ in range(self.n_epochs):
            y_hat, a1 = self._forward(X)
            eps = 1e-12
            loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
            self.loss_history.append(loss)

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
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


def plot_boundary(ax, predict, X, y, title: str) -> None:
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.asarray(predict(grid)).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap="coolwarm", alpha=0.25)
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=25)
    ax.set(title=title, xlabel="feature 1", ylabel="feature 2")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Perceptron on XOR: a linear model cannot solve it
    # ------------------------------------------------------------------
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([0, 1, 1, 0])
    perceptron = Perceptron(n_epochs=100).fit(X_xor, y_xor)
    print("Perceptron on XOR")
    print(f"  predictions = {perceptron.predict(X_xor)}")
    print(f"  accuracy    = {accuracy_score(y_xor, perceptron.predict(X_xor)):.2f}")

    # ------------------------------------------------------------------
    # 2. MLP on two moons
    # ------------------------------------------------------------------
    X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    mlp = MLP(n_inputs=2, n_hidden=16, learning_rate=0.5, n_epochs=3000, seed=0)
    mlp.fit(X_train_s, y_train.astype(float))
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test_s))

    sklearn_mlp = MLPClassifier(
        hidden_layer_sizes=(16,), activation="tanh", solver="adam",
        max_iter=3000, random_state=42,
    ).fit(X_train_s, y_train)
    sklearn_acc = sklearn_mlp.score(X_test_s, y_test)

    print("\nMLP on two moons")
    print(f"  NumPy MLP accuracy    = {mlp_acc:.4f}")
    print(f"  sklearn MLP accuracy  = {sklearn_acc:.4f}")

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
                  "NumPy MLP decision boundary")
    axes[1].scatter(X_test_s[:, 0], X_test_s[:, 1], c=y_test, cmap="coolwarm",
                    edgecolors="k", marker="^", s=25)

    axes[2].plot(mlp.loss_history)
    axes[2].set(xlabel="Epoch", ylabel="Binary cross-entropy", title="Training loss")
    axes[2].set_yscale("log")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
