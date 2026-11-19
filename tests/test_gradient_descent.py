import numpy as np
import pytest
from scipy.optimize import check_grad

from gradient_descent import (
    batch_gradient,
    full_loss,
    grad,
    loss,
    run_gd,
    train_sgd,
)


def test_gradient_matches_finite_differences() -> None:
    theta = np.array([0.7, -0.3])
    kappa = 5.0
    error = check_grad(
        lambda t: loss(t, kappa), lambda t: grad(t, kappa), theta
    )
    assert error < 1e-6


def test_run_gd_converges() -> None:
    path = run_gd(kappa=10.0, lr=0.05, steps=200)
    assert loss(path[-1], 10.0) < 1e-6


def test_run_gd_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        run_gd(1.0, -0.1, 10)
    with pytest.raises(ValueError):
        run_gd(1.0, 0.1, 10, momentum=0.9, use_adam=True)
    with pytest.raises(ValueError):
        run_gd(1.0, 0.1, 10, use_adam=True, beta1=1.0)
    with pytest.raises(ValueError):
        run_gd(1.0, 0.1, 10, use_adam=True, eps=0.0)


def test_batch_gradient_shape_and_direction() -> None:
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(30), rng.normal(size=(30, 2))])
    y = X @ np.array([1.0, 2.0, -1.0])
    beta = np.zeros(3)
    g = batch_gradient(X, y, beta, np.arange(30))
    assert g.shape == (3,)
    assert full_loss(X, y, beta - 0.01 * g) < full_loss(X, y, beta)


def test_train_sgd_reduces_loss() -> None:
    rng = np.random.default_rng(2)
    X = np.column_stack([np.ones(100), rng.normal(size=(100, 2))])
    y = X @ np.array([0.5, 1.5, -0.5])
    beta, history = train_sgd(X, y, epochs=40)
    assert history[-1] < history[0]
    assert np.linalg.norm(beta - np.array([0.5, 1.5, -0.5])) < 0.2
