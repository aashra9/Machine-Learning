"""Information-theory companion script.

Computes entropy, cross-entropy, KL divergence and Bayesian updating, then
prints worked examples and optional plots. Run from the repository root:

    python Code/information_theory.py --no-plots
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


def _as_proba(p: object, name: str = "p") -> np.ndarray:
    """Coerce a probability vector, accepting a Bernoulli scalar."""
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 0:
        arr = np.array([arr, 1.0 - arr])
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a probability vector or a scalar")
    if np.any(arr < 0.0) or not np.isclose(arr.sum(), 1.0):
        raise ValueError(f"{name} must be non-negative and sum to one")
    return arr


def entropy(p: object, base: float = 2.0) -> float:
    """Shannon entropy in units determined by ``base`` (2 = bits)."""
    probs = _as_proba(p)
    if base <= 0.0 or base == 1.0:
        raise ValueError("base must be positive and different from 1")
    positive = probs[probs > 0.0]
    return float(-np.sum(positive * np.log(positive)) / np.log(base))


def cross_entropy(p: object, q: object, base: float = 2.0) -> float:
    """Cross-entropy H(p,q); returns inf when q assigns zero mass to p."""
    probs = _as_proba(p, "p")
    model = _as_proba(q, "q")
    if probs.shape != model.shape:
        raise ValueError("p and q must have the same shape")
    if base <= 0.0 or base == 1.0:
        raise ValueError("base must be positive and different from 1")
    support = probs > 0.0
    with np.errstate(divide="ignore"):
        return float(-np.sum(probs[support] * np.log(model[support])) / np.log(base))


def kl_divergence(p: object, q: object, base: float = 2.0) -> float:
    """KL(p || q) = H(p,q) - H(p)."""
    return cross_entropy(p, q, base) - entropy(p, base)


def jensen_shannon_divergence(p: object, q: object, base: float = 2.0) -> float:
    """Symmetric Jensen-Shannon divergence with the midpoint mixture."""
    probs = _as_proba(p, "p")
    model = _as_proba(q, "q")
    if probs.shape != model.shape:
        raise ValueError("p and q must have the same shape")
    mixture = 0.5 * (probs + model)
    return 0.5 * kl_divergence(probs, mixture, base) + 0.5 * kl_divergence(
        model, mixture, base
    )


def hellinger(p: object, q: object) -> float:
    """Hellinger distance between two probability vectors."""
    probs = _as_proba(p, "p")
    model = _as_proba(q, "q")
    if probs.shape != model.shape:
        raise ValueError("p and q must have the same shape")
    return float(np.sqrt(0.5 * np.sum((np.sqrt(probs) - np.sqrt(model)) ** 2)))


def total_variation(p: object, q: object) -> float:
    """Total variation distance between two probability vectors."""
    probs = _as_proba(p, "p")
    model = _as_proba(q, "q")
    if probs.shape != model.shape:
        raise ValueError("p and q must have the same shape")
    return float(0.5 * np.sum(np.abs(probs - model)))


def wasserstein_1d(samples_p: object, samples_q: object) -> float:
    """Empirical 1-Wasserstein distance between one-dimensional samples."""
    x = np.asarray(samples_p, dtype=float).ravel()
    y = np.asarray(samples_q, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("samples must not be empty")
    return float(wasserstein_distance(x, y))


def gaussian_kl(
    mean_p: float, var_p: float, mean_q: float, var_q: float,
) -> float:
    """KL(N(mean_p, var_p) || N(mean_q, var_q)) in nats."""
    if var_p <= 0.0 or var_q <= 0.0:
        raise ValueError("variances must be positive")
    return float(
        0.5 * (
            np.log(var_q / var_p)
            + (var_p + (mean_p - mean_q) ** 2) / var_q
            - 1.0
        )
    )


def bayes_posterior(
    prior: object, likelihood: object,
) -> tuple[np.ndarray, float]:
    """Return normalised posterior probabilities and the evidence."""
    prior_arr = _as_proba(prior, "prior")
    like_arr = np.asarray(likelihood, dtype=float)
    if like_arr.ndim != 1 or like_arr.shape != prior_arr.shape:
        raise ValueError("prior and likelihood must be 1-D arrays of equal length")
    if np.any(like_arr < 0.0):
        raise ValueError("likelihood must be non-negative")
    unnormalised = prior_arr * like_arr
    evidence = float(unnormalised.sum())
    if evidence <= 0.0:
        raise ValueError("evidence is zero; posterior is undefined")
    return unnormalised / evidence, evidence


def naive_bayes_log_odds(prior: float, likelihood_ratios: object) -> float:
    """Log posterior odds from prior odds and independent likelihood ratios."""
    if not 0.0 < prior < 1.0:
        raise ValueError("prior must lie strictly between 0 and 1")
    ratios = np.asarray(likelihood_ratios, dtype=float)
    if ratios.ndim != 1 or np.any(ratios <= 0.0):
        raise ValueError("likelihood ratios must be positive")
    return float(np.log(prior / (1.0 - prior)) + np.sum(np.log(ratios)))


def gaussian_posterior(
    prior_mean: float, prior_var: float, observation: float, obs_var: float,
) -> tuple[float, float]:
    """Conjugate update for a Gaussian mean with known observation variance."""
    if prior_var <= 0.0 or obs_var <= 0.0:
        raise ValueError("variances must be positive")
    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    post_mean = post_var * (prior_mean / prior_var + observation / obs_var)
    return float(post_mean), float(post_var)


def softmax(logits: object) -> np.ndarray:
    """Numerically stable softmax for a 1-D or 2-D array of logits."""
    z = np.asarray(logits, dtype=float)
    if z.ndim not in (1, 2):
        raise ValueError("logits must be 1-D or 2-D")
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def categorical_cross_entropy(logits: object, labels: object) -> np.ndarray:
    """Cross-entropy for one-hot targets encoded as class indices."""
    probs = softmax(logits)
    y = np.asarray(labels, dtype=int).ravel()
    if probs.ndim == 1:
        if y.size != 1:
            raise ValueError("one set of logits needs one label")
        return np.array([-np.log(probs[y[0]] + 1e-12)])
    if len(y) != len(probs):
        raise ValueError("logits and labels must have the same length")
    rows = np.arange(len(y))
    return -np.log(probs[rows, y] + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-plots", action="store_true", help="skip matplotlib figures")
    parser.add_argument("--base", type=float, default=2.0, help="log base (2 or 10)")
    args = parser.parse_args()

    print("Binary entropy")
    for p in (0.1, 0.25, 0.5, 0.9):
        print(f"  p={p:.2f}: H={entropy(p, args.base):.4f}")

    p, q = 0.7, 0.4
    print("\nCross-entropy and KL")
    print(f"  H({p},{q}) = {cross_entropy(p, q, args.base):.4f}")
    print(f"  H({p})     = {entropy(p, args.base):.4f}")
    print(f"  KL({p}||{q}) = {kl_divergence(p, q, args.base):.4f}")
    print(f"  JS({p},{q}) = {jensen_shannon_divergence(p, q, args.base):.4f}")
    print(f"  Hellinger({p},{q}) = {hellinger(p, q):.4f}")
    print(f"  TV({p},{q}) = {total_variation(p, q):.4f}")
    print(f"  W1(delta_0, delta_3) = {wasserstein_1d([0.0], [3.0]):.4f}")
    print(f"  KL(N(0,1)||N(1,1)) = {gaussian_kl(0, 1, 1, 1):.4f} nats")

    posterior, evidence = bayes_posterior([0.2, 0.5, 0.3], [0.1, 0.8, 0.4])
    print("\nBayes' rule")
    print(f"  evidence  = {evidence:.4f}")
    print(f"  posterior = {np.round(posterior, 4)}")
    log_odds = naive_bayes_log_odds(0.1, [0.5 / 0.05, 0.4 / 0.02])
    print(f"  spam log odds = {log_odds:.4f}, posterior = {1 / (1 + np.exp(-log_odds)):.4f}")

    mean, var = gaussian_posterior(0.0, 2.0, 2.0, 1.0)
    print(f"  Gaussian posterior N({mean:.4f}, {var:.4f})")

    if args.no_plots:
        return

    grid = np.linspace(0.001, 0.999, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(grid, [entropy(float(x), args.base) for x in grid])
    axes[0].set(xlabel="p", ylabel="entropy", title="Binary entropy")
    axes[1].plot(grid, [cross_entropy(p, float(x), args.base) for x in grid])
    axes[1].set(xlabel="q", ylabel="cross-entropy", title="H(p,q) is minimised at q=p")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
