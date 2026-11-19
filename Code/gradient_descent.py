"""Gradient descent and optimisation companion script.

Visualises the effect of the learning rate, momentum, and Adam on an
ill-conditioned quadratic, then trains a linear model with mini-batch SGD.
Run from the repository root:

    python Code/gradient_descent.py
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt


def loss(theta: np.ndarray, kappa: float) -> float:
    """Value of the ill-conditioned quadratic at ``theta``."""
    return float(0.5 * (theta[0] ** 2 + kappa * theta[1] ** 2))


def grad(theta: np.ndarray, kappa: float) -> np.ndarray:
    """Analytic gradient of :func:`loss`."""
    return np.array([theta[0], kappa * theta[1]])


def run_gd(
    kappa: float,
    lr: float,
    steps: int,
    momentum: float = 0.0,
    use_adam: bool = False,
    theta0: np.ndarray | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> np.ndarray:
    """Run gradient descent and return the parameter trajectory.

    Raises:
        ValueError: for non-positive learning rates or conflicting settings.
    """
    if lr <= 0:
        raise ValueError("learning rate must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if use_adam and momentum:
        raise ValueError("momentum and Adam cannot be used together")
    if use_adam:
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("Adam betas must lie in [0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
    theta = (np.array([2.5, 1.8], dtype=float) if theta0 is None
             else np.asarray(theta0, dtype=float).copy())
    path = [theta.copy()]
    velocity = np.zeros_like(theta)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    for t in range(1, steps + 1):
        g = grad(theta, kappa)
        if use_adam:
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        elif momentum > 0:
            velocity = momentum * velocity - lr * g
            theta = theta + velocity
        else:
            theta = theta - lr * g
        path.append(theta.copy())
    return np.asarray(path)


def full_loss(X_bias: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """Mean squared error (with a 1/2 factor) for the linear regression demo."""
    residual = y - X_bias @ beta
    return float(0.5 * np.mean(residual ** 2))


def batch_gradient(
    X_bias: np.ndarray, y: np.ndarray, beta: np.ndarray, idx: np.ndarray
) -> np.ndarray:
    """Gradient of :func:`full_loss` on the mini-batch indexed by ``idx``."""
    residual = y[idx] - X_bias[idx] @ beta
    return -(X_bias[idx].T @ residual) / len(idx)


def train_sgd(
    X_bias: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 60,
    batch_size: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Fit linear regression by mini-batch SGD and record the loss history."""
    n, p = X_bias.shape
    if batch_size < 1 or batch_size > n:
        raise ValueError("batch_size must be between 1 and the number of samples")
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    history = []
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            beta = beta - lr * batch_gradient(X_bias, y, beta, idx)
        history.append(full_loss(X_bias, y, beta))
    return beta, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    kappa = 25.0

    # ------------------------------------------------------------------
    # 1. Learning-rate comparison on an ill-conditioned quadratic
    # ------------------------------------------------------------------
    learning_rates = [0.005, 0.05, 0.08]
    paths = {lr: run_gd(kappa, lr, args.steps) for lr in learning_rates}

    # ------------------------------------------------------------------
    # 2. Momentum and Adam
    # ------------------------------------------------------------------
    accelerated = {
        "GD": run_gd(kappa, 0.05, args.steps),
        "Momentum (0.9)": run_gd(kappa, 0.02, args.steps, momentum=0.9),
        "Adam": run_gd(kappa, 0.15, args.steps, use_adam=True),
    }

    # ------------------------------------------------------------------
    # 3. Mini-batch SGD for linear regression
    # ------------------------------------------------------------------
    n, p = 500, 3
    X = rng.normal(size=(n, p))
    true_beta = np.array([1.5, -2.0, 0.75])
    y = 0.4 + X @ true_beta + rng.normal(0.0, 0.5, size=n)
    X_bias = np.column_stack([np.ones(n), X])

    beta, sgd_path = train_sgd(X_bias, y, epochs=args.epochs)
    closed_form = np.linalg.lstsq(X_bias, y, rcond=None)[0]
    print("Mini-batch SGD for linear regression")
    print(f"  closed-form coefficients = {np.round(closed_form, 3)}")
    print(f"  SGD coefficients          = {np.round(beta, 3)}")
    print(f"  final loss                = {full_loss(X_bias, y, beta):.5f}")

    if args.no_plots:
        return

    # ------------------------------------------------------------------
    # 4. Plots (including the previously unused SGD convergence curve)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))

    theta1 = np.linspace(-2.8, 2.8, 250)
    theta2 = np.linspace(-2.2, 2.2, 250)
    T1, T2 = np.meshgrid(theta1, theta2)
    Z = 0.5 * (T1 ** 2 + kappa * T2 ** 2)
    axes[0].contour(T1, T2, Z, levels=np.logspace(-1, 2.2, 18),
                    cmap="Greys", linewidths=0.6)
    for lr_value, path in paths.items():
        axes[0].plot(path[:, 0], path[:, 1], marker=".", markersize=2,
                     label=f"lr={lr_value}")
    axes[0].plot(0, 0, "r*", markersize=12)
    axes[0].set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$",
                title="Learning-rate trajectories")
    axes[0].legend(fontsize=8)

    for lr_value, path in paths.items():
        losses = [loss(theta, kappa) for theta in path]
        axes[1].plot(losses, label=f"lr={lr_value}")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Iteration", ylabel="Loss", title="Effect of learning rate")
    axes[1].legend(fontsize=8)

    for name, path in accelerated.items():
        losses = [loss(theta, kappa) for theta in path]
        axes[2].plot(losses, label=name)
    axes[2].set_yscale("log")
    axes[2].set(xlabel="Iteration", ylabel="Loss", title="Momentum and Adam")
    axes[2].legend(fontsize=8)

    axes[3].plot(sgd_path)
    axes[3].set_yscale("log")
    axes[3].set(xlabel="Epoch", ylabel="Loss", title="Mini-batch SGD convergence")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
