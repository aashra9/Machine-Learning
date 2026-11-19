import numpy as np
import pytest

from information_theory import (
    bayes_posterior,
    categorical_cross_entropy,
    cross_entropy,
    entropy,
    gaussian_kl,
    gaussian_posterior,
    hellinger,
    jensen_shannon_divergence,
    kl_divergence,
    naive_bayes_log_odds,
    softmax,
    total_variation,
    wasserstein_1d,
)


def test_entropy_known_values() -> None:
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy([1.0, 0.0]) == pytest.approx(0.0)
    assert entropy([0.25] * 4) == pytest.approx(2.0)


def test_cross_entropy_minus_entropy_is_kl() -> None:
    p, q = [0.7, 0.3], [0.4, 0.6]
    assert kl_divergence(p, q) == pytest.approx(cross_entropy(p, q) - entropy(p))


def test_kl_zero_identical_and_infinite_for_missing_support() -> None:
    assert kl_divergence([0.3, 0.7], [0.3, 0.7]) == pytest.approx(0.0)
    assert kl_divergence([0.3, 0.7], [0.0, 1.0]) == np.inf


def test_jensen_shannon_is_symmetric_and_bounded() -> None:
    p, q = [0.9, 0.1], [0.1, 0.9]
    js = jensen_shannon_divergence(p, q)
    assert js == pytest.approx(jensen_shannon_divergence(q, p))
    assert jensen_shannon_divergence(p, p) == pytest.approx(0.0)
    assert js == pytest.approx(1.0 - entropy(p))
    assert 0.0 <= js <= 1.0


def test_probability_distance_metrics() -> None:
    p, q = [0.6, 0.4], [0.5, 0.5]
    assert hellinger(p, p) == pytest.approx(0.0)
    assert total_variation(p, p) == pytest.approx(0.0)
    assert hellinger([1, 0], [0, 1]) == pytest.approx(1.0)
    assert total_variation([1, 0], [0, 1]) == pytest.approx(1.0)
    assert total_variation(p, q) == pytest.approx(0.1)
    assert 0.0 < hellinger(p, q) < 1.0


def test_wasserstein_1d_examples() -> None:
    assert wasserstein_1d([0.0], [3.0]) == pytest.approx(3.0)
    assert wasserstein_1d([0.0], [1e-6]) == pytest.approx(1e-6)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 5000)
    y = rng.uniform(0, 2, 5000)
    assert wasserstein_1d(x, y) == pytest.approx(0.5, abs=0.03)


def test_gaussian_kl_known_value() -> None:
    assert gaussian_kl(0.0, 1.0, 1.0, 1.0) == pytest.approx(0.5)


def test_bayes_posterior_normalises() -> None:
    posterior, evidence = bayes_posterior([0.2, 0.5, 0.3], [0.1, 0.8, 0.4])
    assert evidence == pytest.approx(0.45)
    assert posterior.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(posterior, [0.2 * 0.1 / 0.45, 0.5 * 0.8 / 0.45,
                                           0.3 * 0.4 / 0.45])


def test_naive_bayes_log_odds_matches_direct_posterior() -> None:
    log_odds = naive_bayes_log_odds(0.1, [10.0, 20.0])
    posterior = 1 / (1 + np.exp(-log_odds))
    assert posterior == pytest.approx(0.1 * 10 * 20 / (0.1 * 10 * 20 + 0.9))


def test_gaussian_posterior_precision_weighting() -> None:
    mean, var = gaussian_posterior(0.0, 2.0, 2.0, 1.0)
    assert var == pytest.approx(2 / 3)
    assert mean == pytest.approx(4 / 3)


def test_softmax_shift_invariance_and_cross_entropy() -> None:
    logits = np.array([[2.0, 1.0, 0.1], [0.1, 3.0, 0.2]])
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    np.testing.assert_allclose(probs, softmax(logits + 100))
    losses = categorical_cross_entropy(logits, [0, 1])
    assert losses.shape == (2,)
    assert np.all(losses > 0)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        entropy([0.2, 0.3])
    with pytest.raises(ValueError):
        cross_entropy([0.5, 0.5], [0.4, 0.4])
    with pytest.raises(ValueError):
        gaussian_kl(0, -1, 0, 1)
    with pytest.raises(ValueError):
        naive_bayes_log_odds(1.0, [2.0])
