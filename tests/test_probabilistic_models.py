import numpy as np
import pytest
from scipy.stats import multivariate_normal

from probabilistic_models import GaussianMixtureEM, log_gaussian, viterbi


def test_log_gaussian_matches_scipy() -> None:
    mean = np.array([0.0, 1.0])
    cov = np.array([[1.0, 0.4], [0.4, 2.0]])
    X = np.array([[0.0, 1.0], [1.0, -1.0], [-0.5, 0.5]])
    expected = multivariate_normal(mean, cov).logpdf(X)
    np.testing.assert_allclose(log_gaussian(X, mean, cov), expected, atol=1e-10)


def test_gmm_recovers_clusters() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=-5.0, scale=0.4, size=(120, 2)),
        rng.normal(loc=5.0, scale=0.4, size=(120, 2)),
    ])
    model = GaussianMixtureEM(n_components=2, seed=0).fit(X)
    labels = model.predict(X)
    assert len(np.unique(labels)) == 2
    assert np.all(np.diff(model.log_likelihood_)[-5:] >= -1e-6)
    assert np.isfinite(model.bic(X))


def test_gmm_validation() -> None:
    with pytest.raises(ValueError):
        GaussianMixtureEM(n_components=2).fit(np.zeros((1, 2)))
    with pytest.raises(RuntimeError):
        GaussianMixtureEM().predict(np.zeros((3, 2)))


def test_viterbi_simple_chain() -> None:
    start = np.array([0.6, 0.4])
    transition = np.array([[0.7, 0.3], [0.4, 0.6]])
    emission = np.array([[0.9, 0.1], [0.2, 0.8]])
    observations = np.array([0, 1, 1])
    path = viterbi(start, transition, emission, observations)
    assert path.shape == (3,)
    assert set(path).issubset({0, 1})


def test_viterbi_validation() -> None:
    with pytest.raises(ValueError):
        viterbi(np.array([0.5, 0.5]), np.ones((2, 2)),
                np.ones((2, 2)) / 2, np.array([0]))
    with pytest.raises(ValueError):
        viterbi(np.array([0.5, 0.5]), np.ones((2, 2)) / 2,
                np.ones((2, 2)) / 2, np.array([]))
