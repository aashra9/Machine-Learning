"""Statistical learning theory companion script.

Empirically illustrates the bias--variance decomposition, learning curves, and
double descent.  Run from the repository root:

    python Code/learning_theory.py
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def true_function(x: np.ndarray) -> np.ndarray:
    """The smooth target function used by the regression experiments."""
    return np.sin(1.5 * np.pi * x)


def bias_variance_experiment(
    degrees: list[int], n_train: int = 30, n_sims: int = 200,
    noise: float = 0.25, seed: int = 0,
) -> dict[str, np.ndarray]:
    """Estimate bias^2, variance, and MSE over repeated polynomial fits.

    ``noise_floor`` equals ``noise**2``; for squared error the identity
    ``MSE = bias^2 + variance + noise_floor`` holds in expectation when MSE is
    measured against fresh noisy test targets (as done here).
    """
    rng = np.random.default_rng(seed)
    x_test = np.linspace(0.0, 1.0, 200)
    f_test = true_function(x_test)
    results = {"bias2": [], "variance": [], "mse": [], "noise_floor": []}

    for degree in degrees:
        predictions = []
        test_targets = []
        for _ in range(n_sims):
            x_train = rng.uniform(0.0, 1.0, size=n_train)
            y_train = true_function(x_train) + rng.normal(0.0, noise, size=n_train)
            model = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=False),
                LinearRegression(),
            ).fit(x_train[:, None], y_train)
            predictions.append(model.predict(x_test[:, None]))
            test_targets.append(f_test + rng.normal(0.0, noise, size=f_test.shape))
        predictions = np.asarray(predictions)
        test_targets = np.asarray(test_targets)
        mean_pred = predictions.mean(axis=0)
        results["bias2"].append(np.mean((mean_pred - f_test) ** 2))
        results["variance"].append(np.mean(predictions.var(axis=0)))
        results["mse"].append(np.mean((predictions - test_targets) ** 2))
        results["noise_floor"].append(noise ** 2)

    return {key: np.asarray(value) for key, value in results.items()}


def double_descent_experiment(
    n_train: int = 25, n_test: int = 500, max_degree: int = 40,
    noise: float = 0.2, seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return degrees, train MSE, test MSE, and the interpolation threshold."""
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
    threshold = int(n_train - 1)
    return degrees, np.asarray(train_err), np.asarray(test_err), threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degrees", type=int, nargs="+",
                        default=[1, 2, 3, 5, 7, 10, 14])
    parser.add_argument("--n-sims", type=int, default=200)
    parser.add_argument("--max-degree", type=int, default=40)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Bias-variance decomposition
    # ------------------------------------------------------------------
    bv = bias_variance_experiment(args.degrees, n_sims=args.n_sims)
    print("Bias-variance decomposition")
    for i, degree in enumerate(args.degrees):
        print(f"  degree {degree:2d}: bias^2={bv['bias2'][i]:.4f}  "
              f"variance={bv['variance'][i]:.4f}  mse={bv['mse'][i]:.4f}  "
              f"floor={bv['noise_floor'][i]:.4f}")

    # ------------------------------------------------------------------
    # 2. Learning curve on a classification problem
    # ------------------------------------------------------------------
    X, y = make_moons(n_samples=600, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    sizes, train_scores, valid_scores = learning_curve(
        model, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy", n_jobs=-1,
    )

    # ------------------------------------------------------------------
    # 3. Double descent in polynomial regression
    # ------------------------------------------------------------------
    dd_degrees, dd_train, dd_test, threshold = double_descent_experiment(
        max_degree=args.max_degree
    )

    if args.no_plots:
        return

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(args.degrees, bv["bias2"], marker="o", label="bias$^2$")
    axes[0].plot(args.degrees, bv["variance"], marker="o", label="variance")
    axes[0].plot(args.degrees, bv["mse"], marker="o", label="expected MSE")
    axes[0].axhline(bv["noise_floor"][0], color="gray", linestyle=":",
                    label="noise floor")
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
    axes[2].axvline(threshold, color="gray", linestyle=":",
                    label=f"threshold={threshold}")
    axes[2].set_yscale("log")
    axes[2].set(xlabel="Polynomial degree", ylabel="MSE", title="Double descent")
    axes[2].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
