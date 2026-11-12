"""Regularisation and sparsity companion script.

Compares ridge (L2), LASSO (L1), and elastic net on a high-dimensional synthetic
problem with correlated features.  Run from the repository root:

    python Code/regularization.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_correlated_data(
    n_samples: int = 250,
    n_features: int = 40,
    n_informative: int = 8,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data with correlated columns and a sparse coefficient vector."""
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


def main() -> None:
    X, y, true_coef = make_correlated_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    alphas = np.logspace(-2, 3, 60)

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
        "OLS": ols,
        "Ridge": ridge,
        "LASSO": lasso,
        "Elastic net": enet,
    }
    print("Model comparison on held-out data")
    for name, model in models.items():
        pred = model.predict(X_test_s)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        nonzero = np.sum(np.abs(model.coef_) > 1e-8)
        print(f"  {name:12s} test RMSE={rmse:6.3f}  nonzero coefs={nonzero:2d}")
    print(f"  true nonzero coefs={np.sum(np.abs(true_coef) > 1e-8)}")
    print(f"  selected alpha: ridge={ridge.alpha_:.4g}  lasso={lasso.alpha_:.4g}  "
          f"elastic net={enet.alpha_:.4g} (l1_ratio={enet.l1_ratio_:.2f})")

    # ------------------------------------------------------------------
    # 2. Coefficient paths
    # ------------------------------------------------------------------
    ridge_path, lasso_path = [], []
    for alpha in alphas:
        ridge_path.append(Ridge(alpha=alpha).fit(X_train_s, y_train).coef_)
        lasso_path.append(
            Lasso(alpha=alpha, max_iter=50_000).fit(X_train_s, y_train).coef_
        )
    ridge_path = np.asarray(ridge_path)
    lasso_path = np.asarray(lasso_path)

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

    axes[1, 0].errorbar(
        alphas, lasso.mse_path_.mean(axis=1),
        yerr=lasso.mse_path_.std(axis=1) / np.sqrt(lasso.mse_path_.shape[1]),
        fmt="-", linewidth=1,
    )
    axes[1, 0].axvline(lasso.alpha_, color="red", linestyle="--", linewidth=1)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set(xlabel=r"$\lambda$", ylabel="Cross-validated MSE",
                   title="LASSO CV curve")

    width = 0.2
    idx = np.arange(ridge_path.shape[1])
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
