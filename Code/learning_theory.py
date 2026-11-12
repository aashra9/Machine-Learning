"""Statistical learning theory companion script.

Empirically illustrates the bias--variance decomposition, learning curves, and
double descent.  Run from the repository root:

    python Code/learning_theory.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def true_function(x: np.ndarray) -> np.ndarray:
    return np.sin(1.5 * np.pi * x)


def bias_variance_experiment(degrees: list[int], n_train: int = 30,
                             n_sims: int = 200, noise: float = 0.25,
                             seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_test = np.linspace(0.0, 1.0, 200)
    f_test = true_function(x_test)
    results = {"bias2": [], "variance": [], "mse": []}

    for degree in degrees:
        predictions = []
        for _ in range(n_sims):
            x_train = rng.uniform(0.0, 1.0, size=n_train)
            y_train = true_function(x_train) + rng.normal(0.0, noise, size=n_train)
            model = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=False),
                LinearRegression(),
            ).fit(x_train[:, None], y_train)
            predictions.append(model.predict(x_test[:, None]))
        predictions = np.asarray(predictions)
        mean_pred = predictions.mean(axis=0)
        bias2 = np.mean((mean_pred - f_test) ** 2)
        variance = np.mean(predictions.var(axis=0))
        mse = np.mean((predictions - f_test) ** 2)
        results["bias2"].append(bias2)
        results["variance"].append(variance)
        results["mse"].append(mse)

    return {key: np.asarray(value) for key, value in results.items()}


def double_descent_experiment(n_train: int = 25, n_test: int = 500,
                              max_degree: int = 40,
                              noise: float = 0.2,
                              seed: int = 1) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = np.sort(rng.uniform(0.0, 1.0, size=n_train))
    y_train = true_function(x_train) + rng.normal(0.0, noise, size=n_train)
    x_test = np.linspace(0.0, 1.0, n_test)
    y_test = true_function(x_test)

    degrees = np.arange(1, max_degree + 1)
    train_err, test_err = [], []
    for degree in degrees:
        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression(),
        ).fit(x_train[:, None], y_train)
        train_err.append(np.mean((model.predict(x_train[:, None]) - y_train) ** 2))
        test_err.append(np.mean((model.predict(x_test[:, None]) - y_test) ** 2))
    return degrees, np.asarray(train_err), np.asarray(test_err)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Bias-variance decomposition
    # ------------------------------------------------------------------
    degrees = [1, 2, 3, 5, 7, 10, 14]
    bv = bias_variance_experiment(degrees)
    print("Bias-variance decomposition")
    for i, degree in enumerate(degrees):
        print(f"  degree {degree:2d}: bias^2={bv['bias2'][i]:.4f}  "
              f"variance={bv['variance'][i]:.4f}  mse={bv['mse'][i]:.4f}")

    # ------------------------------------------------------------------
    # 2. Learning curve on a classification problem
    # ------------------------------------------------------------------
    X, y = make_moons(n_samples=600, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    sizes, train_scores, valid_scores = learning_curve(
        model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="accuracy", n_jobs=-1,
    )

    # ------------------------------------------------------------------
    # 3. Double descent in polynomial regression
    # ------------------------------------------------------------------
    dd_degrees, dd_train, dd_test = double_descent_experiment()

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].plot(degrees, bv["bias2"], marker="o", label="bias$^2$")
    axes[0].plot(degrees, bv["variance"], marker="o", label="variance")
    axes[0].plot(degrees, bv["mse"], marker="o", label="expected MSE")
    axes[0].set(xlabel="Polynomial degree", ylabel="Error",
                title="Bias-variance decomposition")
    axes[0].legend()

    axes[1].plot(sizes, train_scores.mean(axis=1), marker="o", label="Training")
    axes[1].plot(sizes, valid_scores.mean(axis=1), marker="o", label="Validation")
    axes[1].fill_between(sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                         train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
    axes[1].fill_between(sizes, valid_scores.mean(axis=1) - valid_scores.std(axis=1),
                         valid_scores.mean(axis=1) + valid_scores.std(axis=1), alpha=0.15)
    axes[1].set(xlabel="Training examples", ylabel="Accuracy", title="Learning curve")
    axes[1].legend()

    axes[2].plot(dd_degrees, dd_train, marker="o", markersize=3, label="Train")
    axes[2].plot(dd_degrees, dd_test, marker="o", markersize=3, label="Test")
    axes[2].axvline(24, color="gray", linestyle=":", label="interpolation threshold")
    axes[2].set_yscale("log")
    axes[2].set(xlabel="Polynomial degree", ylabel="MSE", title="Double descent")
    axes[2].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
