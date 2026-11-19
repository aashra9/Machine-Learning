"""Regularisation and sparsity companion script.

Compares ridge (L2), LASSO (L1), and elastic net on a high-dimensional synthetic
problem with correlated features.  Run from the repository root:

    python Code/regularization.py
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import (
    ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_correlated_data(
    n_samples: int = 250,
    n_features: int = 40,
    n_informative: int = 8,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate correlated columns and a sparse coefficient vector.

    Returns ``(X, y, coef)``; exactly ``n_informative`` entries of ``coef`` are
    non-zero.
    """
    if n_samples < 5 or n_features < 1:
        raise ValueError("n_samples >= 5 and n_features >= 1 are required")
    if not 1 <= n_informative <= n_features:
        raise ValueError("n_informative must be between 1 and n_features")
    rng = np.random.default_rng(random_state)
    base = rng.normal(size=(n_samples, 4))
    loadings = rng.normal(size=(4, n_features))
    X = base @ loadings + 0.35 * rng.normal(size=(n_samples, n_features))

    coef = np.zeros(n_features)
    coef[:n_informative] = rng.choice([-1, 1], size=n_informative) * rng.uniform(
        1.0, 3.0, size=n_informative
    )
    y = X @ coef + rng.normal(0.0, 1.5, size=n_samples)
    return X, y, coef


def coefficient_paths(
    X: np.ndarray, y: np.ndarray, alphas: np.ndarray, max_iter: int = 50_000
) -> tuple[np.ndarray, np.ndarray]:
    """Refit ridge and LASSO on a grid of regularisation strengths."""
    ridge_path = [Ridge(alpha=a).fit(X, y).coef_ for a in alphas]
    lasso_path = [
        Lasso(alpha=a, max_iter=max_iter).fit(X, y).coef_ for a in alphas
    ]
    return np.asarray(ridge_path), np.asarray(lasso_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-features", type=int, default=40)
    parser.add_argument("--n-informative", type=int, default=8)
    parser.add_argument("--n-alphas", type=int, default=60)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    X, y, true_coef = make_correlated_data(
        n_features=args.n_features, n_informative=args.n_informative
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    alphas = np.logspace(-2, 3, args.n_alphas)
    nonzero_tol = 1e-8

    # ------------------------------------------------------------------
    # 1. Tune each penalty by cross-validation
    # ------------------------------------------------------------------
    ridge = RidgeCV(alphas=alphas, cv=5).fit(X_train_s, y_train)
    lasso = LassoCV(alphas=alphas, cv=5, max_iter=50_000, random_state=42).fit(
        X_train_s, y_train
    )
    enet = ElasticNetCV(
        alphas=alphas, l1_ratio=[0.2, 0.5, 0.8, 1.0], cv=5, max_iter=50_000,
        random_state=42,
    ).fit(X_train_s, y_train)
    ols = LinearRegression().fit(X_train_s, y_train)

    models = {
        "OLS": ols, "Ridge": ridge, "LASSO": lasso, "Elastic net": enet,
    }
    print("Model comparison on held-out data")
    for name, model in models.items():
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_s)))
        nonzero = int(np.sum(np.abs(model.coef_) > nonzero_tol))
        print(f"  {name:12s} test RMSE={rmse:6.3f}  nonzero coefs={nonzero:2d}")
    print(f"  true nonzero coefs={int(np.sum(np.abs(true_coef) > nonzero_tol))}")
    print(f"  selected alpha: ridge={ridge.alpha_:.4g}  lasso={lasso.alpha_:.4g}  "
          f"elastic net={enet.alpha_:.4g} (l1_ratio={enet.l1_ratio_:.2f})")

    if args.no_plots:
        return

    # ------------------------------------------------------------------
    # 2. Coefficient paths and diagnostics
    # ------------------------------------------------------------------
    ridge_path, lasso_path = coefficient_paths(X_train_s, y_train, alphas)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for j in range(ridge_path.shape[1]):
        axes[0, 0].plot(alphas, ridge_path[:, j], linewidth=1, alpha=0.6)
    axes[0, 0].set_xscale("log")
    axes[0, 0].axvline(ridge.alpha_, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set(xlabel=r"$\lambda$", ylabel="Coefficient",
                   title="Ridge coefficient paths")

    for j in range(lasso_path.shape[1]):
        axes[0, 1].plot(alphas, lasso_path[:, j], linewidth=1, alpha=0.6)
    axes[0, 1].set_xscale("log")
    axes[0, 1].axvline(lasso.alpha_, color="red", linestyle="--", linewidth=1)
    axes[0, 1].set(xlabel=r"$\lambda$", ylabel="Coefficient",
                   title="LASSO coefficient paths")

    mse_path = np.atleast_2d(lasso.mse_path_)
    axes[1, 0].errorbar(
        alphas, mse_path.mean(axis=1),
        yerr=mse_path.std(axis=1) / np.sqrt(mse_path.shape[1]),
        fmt="-", linewidth=1,
    )
    axes[1, 0].axvline(lasso.alpha_, color="red", linestyle="--", linewidth=1)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set(xlabel=r"$\lambda$", ylabel="Cross-validated MSE",
                   title="LASSO CV curve")

    width = 0.2
    idx = np.arange(len(true_coef))
    axes[1, 1].bar(idx - 1.5 * width, true_coef, width, label="true")
    axes[1, 1].bar(idx - 0.5 * width, ridge.coef_, width, label="ridge")
    axes[1, 1].bar(idx + 0.5 * width, lasso.coef_, width, label="LASSO")
    axes[1, 1].bar(idx + 1.5 * width, enet.coef_, width, label="elastic net")
    axes[1, 1].set(xlabel="Feature index", ylabel="Coefficient",
                   title="Recovered coefficients")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
