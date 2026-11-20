import numpy as np
import pytest
from logistic_regression import coefficient_paths, fit_binary, threshold_metrics


def test_fit_binary_validates_c_and_fits() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="positive"):
        fit_binary(X, y, C=0.0)
    model = fit_binary(X, y)
    assert model[-1].coef_.shape == (1, 1)
    assert model.predict(X).shape == (4,)


def test_coefficient_paths_shape() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=60) > 0).astype(int)
    Cs = np.logspace(-2, 1, 3)
    strengths, coefficients = coefficient_paths(X, y, Cs)
    np.testing.assert_allclose(strengths, Cs)
    assert coefficients.shape == (len(Cs), X.shape[1])


def test_threshold_metrics_known_counts() -> None:
    y_true = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.4, 0.6, 0.1])
    metrics = threshold_metrics(y_true, proba, 0.5)
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 1, 1)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="threshold"):
        threshold_metrics(y_true, proba, 1.5)
