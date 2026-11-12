"""Probabilistic models companion script.

Implements a Gaussian mixture model with EM from scratch and compares it with
scikit-learn.  Also runs Viterbi decoding on a tiny hidden Markov model.
Run from the repository root:

    python Code/probabilistic_models.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture


def log_gaussian(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log density of a multivariate Gaussian for every row of X."""
    d = X.shape[1]
    diff = X - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return np.full(len(X), -np.inf)
    quad = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


class GaussianMixtureEM:
    def __init__(self, n_components: int = 3, max_iter: int = 200, tol: float = 1e-6,
                 seed: int = 0, reg: float = 1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.reg = reg
        self.means: np.ndarray | None = None
        self.covs: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.log_likelihood_: list[float] = []

    def fit(self, X: np.ndarray) -> "GaussianMixtureEM":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        idx = rng.choice(n, size=self.n_components, replace=False)
        self.means = X[idx].copy()
        self.covs = np.array([np.cov(X, rowvar=False) + self.reg * np.eye(d)
                              for _ in range(self.n_components)])
        self.weights = np.full(self.n_components, 1.0 / self.n_components)

        previous = -np.inf
        for _ in range(self.max_iter):
            # E-step
            log_prob = np.column_stack([
                np.log(self.weights[k] + 1e-12) + log_gaussian(X, self.means[k], self.covs[k])
                for k in range(self.n_components)
            ])
            log_norm = logsumexp(log_prob, axis=1, keepdims=True)
            log_resp = log_prob - log_norm
            resp = np.exp(log_resp)
            current = float(log_norm.mean())
            self.log_likelihood_.append(current)
            if current - previous < self.tol:
                break
            previous = current

            # M-step
            nk = resp.sum(axis=0) + 1e-12
            self.weights = nk / n
            self.means = (resp.T @ X) / nk[:, None]
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covs[k] = (resp[:, k, None] * diff).T @ diff / nk[k] \
                    + self.reg * np.eye(d)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_prob = np.column_stack([
            np.log(self.weights[k] + 1e-12) + log_gaussian(X, self.means[k], self.covs[k])
            for k in range(self.n_components)
        ])
        return np.exp(log_prob - logsumexp(log_prob, axis=1, keepdims=True))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def viterbi(start: np.ndarray, transition: np.ndarray, emission: np.ndarray,
            observations: np.ndarray) -> np.ndarray:
    """Most likely state sequence for a small HMM (all inputs are probabilities)."""
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

    sklearn_gmm = GaussianMixture(n_components=3, covariance_type="full",
                                  random_state=42).fit(X)
    print(f"  sklearn means     =\n{np.round(sklearn_gmm.means_, 3)}")

    # ------------------------------------------------------------------
    # 2. Viterbi decoding for a tiny HMM
    # ------------------------------------------------------------------
    states = ["Rainy", "Sunny"]
    observations = np.array([0, 1, 2, 0, 0, 1, 2])
    obs_names = ["walk", "shop", "clean"]
    start = np.array([0.6, 0.4])
    transition = np.array([[0.7, 0.3], [0.4, 0.6]])
    emission = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
    path = viterbi(start, transition, emission, observations)
    print("\nHMM Viterbi decoding")
    print(f"  observations = {[obs_names[i] for i in observations]}")
    print(f"  states       = {[states[i] for i in path]}")

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
