import numpy as np
import pytest
from linear_regression import (
    adjusted_r2,
    bootstrap_ci_coefficients,
    multiple_ols,
    simple_ols,
)


def test_simple_ols_recovers_line() -> None:
    x = np.linspace(0, 10, 50)
    y = 1.0 + 2.5 * x
    intercept, slope = simple_ols(x, y)
    assert intercept == pytest.approx(1.0)
    assert slope == pytest.approx(2.5)


def test_simple_ols_rejects_constant_x() -> None:
    with pytest.raises(ValueError, match="constant"):
        simple_ols(np.ones(10), np.arange(10))


def test_multiple_ols_matches_lstsq() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = 0.5 + X @ np.array([1.0, -2.0, 3.0])
    expected = np.linalg.lstsq(np.column_stack([np.ones(60), X]), y, rcond=None)[0]
    np.testing.assert_allclose(multiple_ols(X, y), expected, atol=1e-8)


def test_multiple_ols_rank_deficient() -> None:
    X = np.ones((20, 2))
    y = np.arange(20, dtype=float)
    assert np.isfinite(multiple_ols(X, y)).all()


def test_adjusted_r2() -> None:
    assert adjusted_r2(0.9, 100, 3) == pytest.approx(
        1 - (1 - 0.9) * 99 / 96
    )
    with pytest.raises(ValueError):
        adjusted_r2(0.9, 4, 4)


def test_bootstrap_ci_shape() -> None:
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(50, 2)), rng.normal(size=50)
    samples = bootstrap_ci_coefficients(X, y, n_boot=20, seed=0)
    assert samples.shape == (20, 3)
