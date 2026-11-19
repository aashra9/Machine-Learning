import numpy as np
import pytest

from regularization import coefficient_paths, make_correlated_data


def test_make_correlated_data() -> None:
    X, y, coef = make_correlated_data(n_samples=60, n_features=12, n_informative=4)
    assert X.shape == (60, 12)
    assert y.shape == (60,)
    assert int(np.count_nonzero(coef)) == 4
    assert np.corrcoef(X @ coef, y)[0, 1] > 0.9


def test_make_correlated_data_validation() -> None:
    with pytest.raises(ValueError):
        make_correlated_data(n_features=3, n_informative=4)
    with pytest.raises(ValueError):
        make_correlated_data(n_samples=2)


def test_coefficient_paths_shape() -> None:
    X, y, _ = make_correlated_data(n_samples=40, n_features=6)
    alphas = np.logspace(-2, 1, 4)
    ridge, lasso = coefficient_paths(X, y, alphas, max_iter=500)
    assert ridge.shape == lasso.shape == (4, 6)
