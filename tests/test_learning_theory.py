import numpy as np

from learning_theory import (
    bias_variance_experiment, double_descent_experiment, true_function,
)


def test_true_function() -> None:
    np.testing.assert_allclose(true_function(np.array([0.0])), [0.0], atol=1e-12)


def test_bias_variance_identity() -> None:
    results = bias_variance_experiment([1, 3], n_sims=30, seed=0)
    for key in ("bias2", "variance", "mse", "noise_floor"):
        assert results[key].shape == (2,)
    decomposition = results["bias2"] + results["variance"] + results["noise_floor"]
    np.testing.assert_allclose(results["mse"], decomposition, atol=0.05)


def test_double_descent_threshold() -> None:
    degrees, train, test, threshold = double_descent_experiment(
        n_train=12, max_degree=15
    )
    assert degrees.tolist() == list(range(1, 16))
    assert len(train) == len(test) == 15
    assert threshold == 11
