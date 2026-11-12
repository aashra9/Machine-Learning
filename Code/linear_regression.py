"""Linear regression companion script.

Covers simple and multiple linear regression, the normal equation, residual
diagnostics, and common goodness-of-fit measures.  Run from the repository root:

    python Code/linear_regression.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def simple_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return the least-squares intercept and slope for one predictor."""
    x_mean, y_mean = x.mean(), y.mean()
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    return intercept, slope


def multiple_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the normal equations for a design matrix that includes an intercept."""
    X_bias = np.column_stack([np.ones(len(X)), X])
    return np.linalg.solve(X_bias.T @ X_bias, X_bias.T @ y)


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R^2 for p predictors and n observations."""
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def main() -> None:
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # 1. Simple linear regression
    # ------------------------------------------------------------------
    n = 100
    x = np.linspace(0.0, 10.0, n)
    true_intercept, true_slope = 1.0, 2.5
    y = true_intercept + true_slope * x + rng.normal(0.0, 3.0, size=n)

    beta0, beta1 = simple_ols(x, y)
    print("Simple linear regression")
    print(f"  true      intercept={true_intercept:.3f}  slope={true_slope:.3f}")
    print(f"  hand OLS  intercept={beta0:.3f}  slope={beta1:.3f}")

    model = LinearRegression().fit(x.reshape(-1, 1), y)
    print(f"  sklearn   intercept={model.intercept_:.3f}  slope={model.coef_[0]:.3f}")

    y_hat = model.predict(x.reshape(-1, 1))
    rss = np.sum((y - y_hat) ** 2)
    rse = np.sqrt(rss / (n - 2))
    print(f"  RSE={rse:.3f}  R^2={r2_score(y, y_hat):.4f}")

    # ------------------------------------------------------------------
    # 2. Multiple linear regression and the normal equation
    # ------------------------------------------------------------------
    n_multi, p_multi = 200, 3
    X_multi = rng.normal(size=(n_multi, p_multi))
    true_coefs = np.array([1.5, -2.0, 3.0])
    y_multi = 0.5 + X_multi @ true_coefs + rng.normal(0.0, 1.0, size=n_multi)

    beta = multiple_ols(X_multi, y_multi)
    print("\nMultiple linear regression")
    print(f"  normal equation intercept={beta[0]:.3f}  coefs={np.round(beta[1:], 3)}")
    print(f"  true            intercept=0.500  coefs={true_coefs}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_multi, y_multi, test_size=0.25, random_state=42
    )
    multi = LinearRegression().fit(X_train, y_train)
    train_pred = multi.predict(X_train)
    test_pred = multi.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    print(f"  train RMSE={np.sqrt(mean_squared_error(y_train, train_pred)):.3f}"
          f"  R^2={train_r2:.3f}  adjusted R^2={adjusted_r2(train_r2, len(y_train), p_multi):.3f}")
    print(f"  test  RMSE={np.sqrt(mean_squared_error(y_test, test_pred)):.3f}"
          f"  R^2={test_r2:.3f}  adjusted R^2={adjusted_r2(test_r2, len(y_test), p_multi):.3f}")

    # ------------------------------------------------------------------
    # 3. Polynomial basis expansion (still linear in the parameters)
    # ------------------------------------------------------------------
    x_grid = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
    degrees = [1, 3, 9]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(x, y, alpha=0.6, label="data")
    for degree in degrees:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(x.reshape(-1, 1))
        poly_model = LinearRegression().fit(X_poly, y)
        axes[0].plot(x_grid, poly_model.predict(poly.fit_transform(x_grid)),
                     label=f"degree {degree}")
    axes[0].set(xlabel="x", ylabel="y", title="Polynomial fits")
    axes[0].legend()

    # Residuals versus fitted values for the simple linear model
    residuals = y - y_hat
    axes[1].scatter(y_hat, residuals, alpha=0.7)
    axes[1].axhline(0.0, color="red", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Fitted values", ylabel="Residuals",
                title="Residuals vs fitted (simple model)")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
