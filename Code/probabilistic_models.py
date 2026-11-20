"""Probabilistic models companion script.

Implements a Gaussian mixture model with EM from scratch and compares it with
scikit-learn.  Also runs Viterbi decoding on a tiny hidden Markov model.
Run from the repository root:

    python Code/probabilistic_models.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture


def log_gaussian(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log density of a multivariate Gaussian for every row of ``X``.

    Uses a Cholesky solve instead of an explicit inverse for stability.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    d = X.shape[1]
    diff = X - mean
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.full(len(X), -np.inf)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    solved = np.linalg.solve(L, diff.T).T
    quad = np.einsum("ij,ij->i", solved, solved)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


class GaussianMixtureEM:
    """Gaussian mixture fitted by the EM algorithm (fully observed data)."""

    EPS = 1e-12

    def __init__(self, n_components: int = 3, max_iter: int = 200, tol: float = 1e-6,
                 seed: int = 0, reg: float = 1e-6):
        if n_components < 1 or max_iter < 1 or tol <= 0 or reg < 0:
            raise ValueError("invalid GMM hyperparameters")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.reg = reg
        self.means: np.ndarray | None = None
        self.covs: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.log_likelihood_: list[float] = []

    def _check_fitted(self) -> None:
        if self.means is None or self.covs is None or self.weights is None:
            raise RuntimeError("GaussianMixtureEM must be fitted before use")

    def _log_prob(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.column_stack([
            np.log(self.weights[k] + self.EPS)
            + log_gaussian(X, self.means[k], self.covs[k])
            for k in range(self.n_components)
        ])

    def fit(self, X: np.ndarray) -> GaussianMixtureEM:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        n, d = X.shape
        if self.n_components > n:
            raise ValueError("n_components cannot exceed the number of samples")
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(n, size=self.n_components, replace=False)
        self.means = X[idx].copy()
        self.covs = np.array([
            np.cov(X, rowvar=False) + self.reg * np.eye(d)
            for _ in range(self.n_components)
        ])
        self.weights = np.full(self.n_components, 1.0 / self.n_components)
        self.log_likelihood_ = []

        previous = -np.inf
        for _ in range(self.max_iter):
            log_prob = self._log_prob(X)
            log_norm = logsumexp(log_prob, axis=1, keepdims=True)
            resp = np.exp(log_prob - log_norm)
            current = float(log_norm.mean())
            self.log_likelihood_.append(current)
            if abs(current - previous) < self.tol:
                break
            previous = current

            # M-step
            nk = resp.sum(axis=0) + self.EPS
            self.weights = nk / n
            self.means = (resp.T @ X) / nk[:, None]
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covs[k] = (resp[:, k, None] * diff).T @ diff / nk[k] \
                    + self.reg * np.eye(d)
        return self

    def score(self, X: np.ndarray) -> float:
        """Mean log-likelihood of ``X`` under the fitted model."""
        return float(logsumexp(self._log_prob(X), axis=1).mean())

    def bic(self, X: np.ndarray) -> float:
        """Bayesian information criterion (lower is better)."""
        n, d = X.shape
        n_params = self.n_components * (d + d * (d + 1) / 2 + 1) - 1
        return float(n_params * np.log(n) - 2 * n * self.score(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_prob = self._log_prob(X)
        return np.exp(log_prob - logsumexp(log_prob, axis=1, keepdims=True))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def viterbi(
    start: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Most likely state sequence for a small HMM (all inputs are probabilities)."""
    start = np.asarray(start, dtype=float)
    transition = np.asarray(transition, dtype=float)
    emission = np.asarray(emission, dtype=float)
    observations = np.asarray(observations, dtype=int)
    if observations.size == 0:
        raise ValueError("observations must not be empty")
    if not np.allclose(start.sum(), 1.0) or not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("start and transition distributions must sum to one")
    if not np.allclose(emission.sum(axis=1), 1.0):
        raise ValueError("each emission distribution must sum to one")
    if observations.min() < 0 or observations.max() >= emission.shape[1]:
        raise ValueError("observation index outside the emission alphabet")

    with np.errstate(divide="ignore"):
        log_start = np.log(start)
        log_trans = np.log(transition)
        log_emit = np.log(emission)
    n_states = len(start)
    T = len(observations)

    delta = np.full((T, n_states), -np.inf)
    psi = np.zeros((T, n_states), dtype=int)
    delta[0] = log_start + log_emit[:, observations[0]]
    for t in range(1, T):
        for j in range(n_states):
            candidates = delta[t - 1] + log_trans[:, j]
            psi[t, j] = int(np.argmax(candidates))
            delta[t, j] = candidates[psi[t, j]] + log_emit[j, observations[t]]

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # 1. GMM with EM on synthetic data
    # ------------------------------------------------------------------
    true_means = np.array([[-3.0, -1.0], [0.0, 3.0], [3.0, -1.5]])
    true_covs = [
        np.array([[1.0, 0.6], [0.6, 1.0]]),
        np.array([[1.4, -0.4], [-0.4, 0.8]]),
        np.array([[0.7, 0.0], [0.0, 1.2]]),
    ]
    X = np.vstack([
        rng.multivariate_normal(true_means[k], true_covs[k], size=200)
        for k in range(3)
    ])

    gmm = GaussianMixtureEM(n_components=3, seed=1).fit(X)
    labels = gmm.predict(X)
    print("Gaussian mixture model (EM)")
    print(f"  estimated weights = {np.round(gmm.weights, 3)}")
    print(f"  estimated means   =\n{np.round(gmm.means, 3)}")
    print(f"  BIC               = {gmm.bic(X):.1f}")

    sklearn_gmm = GaussianMixture(n_components=3, covariance_type="full",
                                  random_state=42).fit(X)
    print(f"  sklearn means     =\n{np.round(sklearn_gmm.means_, 3)}")

    # ------------------------------------------------------------------
    # 2. Viterbi decoding for a tiny HMM
    # ------------------------------------------------------------------
    states = ["Rainy", "Sunny"]
    obs_ids = np.array([0, 1, 2, 0, 0, 1, 2])
    obs_names = ["walk", "shop", "clean"]
    start = np.array([0.6, 0.4])
    transition = np.array([[0.7, 0.3], [0.4, 0.6]])
    emission = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
    path = viterbi(start, transition, emission, obs_ids)
    print("\nHMM Viterbi decoding")
    print(f"  observations = {[obs_names[i] for i in obs_ids]}")
    print(f"  states       = {[states[i] for i in path]}")

    if args.no_plots:
        return

    # ------------------------------------------------------------------
    # 3. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=12, alpha=0.7)
    axes[0].scatter(gmm.means[:, 0], gmm.means[:, 1], c="black", marker="x",
                    s=160, linewidths=3, label="estimated means")
    axes[0].set(title="GMM clusters (EM)", xlabel="x1", ylabel="x2")
    axes[0].legend()
    axes[1].plot(gmm.log_likelihood_, marker="o", markersize=3)
    axes[1].set(xlabel="Iteration", ylabel="Mean log-likelihood",
                title="EM convergence")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
