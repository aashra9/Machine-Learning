"""Linear regression companion script.

Covers simple and multiple linear regression, the normal equation, bootstrap
confidence intervals, and residual diagnostics.  Run from the repository root:

    python Code/linear_regression.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def _check_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return arrays with matching numbers of rows."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be one- or two-dimensional")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if X.shape[0] < 2:
        raise ValueError("at least two observations are required")
    return X, y


def simple_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return the least-squares intercept and slope for one predictor.

    Raises:
        ValueError: if ``x`` is constant (the slope is undefined).
    """
    x_arr, y_arr = _check_xy(x, y)
    if x_arr.shape[1] != 1:
        raise ValueError("simple_ols expects exactly one predictor")
    column = x_arr[:, 0]
    if np.ptp(column) == 0.0:
        raise ValueError("x is constant; the least-squares slope is undefined")
    centered = column - column.mean()
    denominator = np.sum(centered ** 2)
    slope = np.sum(centered * (y_arr - y_arr.mean())) / denominator
    intercept = y_arr.mean() - slope * x_arr[:, 0].mean()
    return float(intercept), float(slope)


def multiple_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares fit for an intercept-augmented design matrix.

    Uses ``np.linalg.lstsq`` directly: it handles rank-deficient designs and
    avoids squaring the condition number as the normal equations do.
    """
    X_arr, y_arr = _check_xy(X, y)
    X_bias = np.column_stack([np.ones(len(X_arr)), X_arr])
    return np.linalg.lstsq(X_bias, y_arr, rcond=None)[0]


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R-squared for ``p`` predictors and ``n`` observations."""
    if n - p - 1 <= 0:
        raise ValueError("need n > p + 1 observations to compute adjusted R^2")
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


def bootstrap_ci_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """Return an ``(n_boot, p + 1)`` array of bootstrapped OLS coefficients."""
    X_arr, y_arr = _check_xy(X, y)
    rng = np.random.default_rng(seed)
    n = len(X_arr)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        estimates.append(multiple_ols(X_arr[idx], y_arr[idx]))
    return np.asarray(estimates)


def plot_residual_diagnostics(
    fitted: np.ndarray,
    residuals: np.ndarray,
    axes: np.ndarray,
) -> None:
    """Draw residuals-vs-fitted, histogram, Q-Q, and scale-location plots."""
    standardised = residuals / max(residuals.std(), 1e-12)
    axes[0].scatter(fitted, residuals, alpha=0.7)
    axes[0].axhline(0.0, color="red", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")

    axes[1].hist(residuals, bins=20, density=True, alpha=0.75, color="steelblue")
    x_curve = np.linspace(residuals.min(), residuals.max(), 200)
    axes[1].plot(x_curve, stats.norm.pdf(x_curve, residuals.mean(), residuals.std()),
                 color="red", linewidth=1.5)
    axes[1].set(xlabel="Residual", ylabel="Density", title="Residual distribution")

    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Normal Q-Q plot")

    axes[3].scatter(fitted, np.sqrt(np.abs(standardised)), alpha=0.7)
    axes[3].set(xlabel="Fitted values", ylabel=r"$\sqrt{|standardised\ residual|}$",
                title="Scale-location")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 3, 9])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

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
    true_intercept_multi = 0.5
    true_coefs = np.array([1.5, -2.0, 3.0])
    y_multi = (true_intercept_multi + X_multi @ true_coefs
               + rng.normal(0.0, 1.0, size=n_multi))

    beta = multiple_ols(X_multi, y_multi)
    print("\nMultiple linear regression")
    print(f"  normal equation intercept={beta[0]:.3f}  coefs={np.round(beta[1:], 3)}")
    print(f"  true            intercept={true_intercept_multi:.3f}  coefs={true_coefs}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_multi, y_multi, test_size=0.25, random_state=args.seed
    )
    multi = LinearRegression().fit(X_train, y_train)
    train_pred, test_pred = multi.predict(X_train), multi.predict(X_test)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    print(f"  train RMSE={np.sqrt(mean_squared_error(y_train, train_pred)):.3f}"
          f"  R^2={train_r2:.3f}  adjusted R^2={adjusted_r2(train_r2, len(y_train), p_multi):.3f}")
    print(f"  test  RMSE={np.sqrt(mean_squared_error(y_test, test_pred)):.3f}"
          f"  R^2={test_r2:.3f}  adjusted R^2={adjusted_r2(test_r2, len(y_test), p_multi):.3f}")

    boot = bootstrap_ci_coefficients(X_multi, y_multi, n_boot=args.n_boot,
                                     seed=args.seed)
    lower, upper = np.percentile(boot, [2.5, 97.5], axis=0)
    print("\nBootstrap 95% confidence intervals (intercept, coefs)")
    for j in range(len(lower)):
        label = "intercept" if j == 0 else f"coef {j}"
        print(f"  {label:10s} [{lower[j]:7.3f}, {upper[j]:7.3f}]")

    if args.no_plots:
        return

    # ------------------------------------------------------------------
    # 3. Polynomial basis expansion and residual diagnostics
    # ------------------------------------------------------------------
    x_grid = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    flat = np.asarray(axes).ravel()

    flat[0].scatter(x, y, alpha=0.6, label="data")
    for degree in args.degrees:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(x.reshape(-1, 1))
        poly_model = LinearRegression().fit(X_poly, y)
        flat[0].plot(x_grid, poly_model.predict(poly.transform(x_grid)),
                     label=f"degree {degree}")
    flat[0].set(xlabel="x", ylabel="y", title="Polynomial fits")
    flat[0].legend()

    plot_residual_diagnostics(y_hat, y - y_hat, flat[1:5])

    labels = ["intercept"] + [f"coef {i}" for i in range(1, len(lower))]
    centers = np.arange(len(labels))
    flat[5].errorbar(centers, boot.mean(axis=0),
                     yerr=[boot.mean(axis=0) - lower, upper - boot.mean(axis=0)],
                     fmt="o", capsize=4)
    flat[5].set_xticks(centers, labels, rotation=20)
    flat[5].set(title="Bootstrap coefficient estimates", ylabel="Estimate")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
