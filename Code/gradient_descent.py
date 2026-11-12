"""Gradient descent and optimisation companion script.

Visualises the effect of the learning rate, momentum, and Adam on an
ill-conditioned quadratic, then trains a linear model with mini-batch SGD.
Run from the repository root:

    python Code/gradient_descent.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def loss(theta: np.ndarray, kappa: float) -> float:
    return 0.5 * (theta[0] ** 2 + kappa * theta[1] ** 2)


def grad(theta: np.ndarray, kappa: float) -> np.ndarray:
    return np.array([theta[0], kappa * theta[1]])


def run_gd(kappa: float, lr: float, steps: int, momentum: float = 0.0,
           use_adam: bool = False) -> np.ndarray:
    """Run gradient descent and return the parameter trajectory."""
    theta = np.array([2.5, 1.8])
    path = [theta.copy()]
    velocity = np.zeros_like(theta)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

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


def main() -> None:
    rng = np.random.default_rng(0)
    kappa = 25.0
    steps = 80

    # ------------------------------------------------------------------
    # 1. Learning-rate comparison on an ill-conditioned quadratic
    # ------------------------------------------------------------------
    learning_rates = [0.005, 0.05, 0.08]
    paths = {lr: run_gd(kappa, lr, steps) for lr in learning_rates}

    # ------------------------------------------------------------------
    # 2. Momentum and Adam
    # ------------------------------------------------------------------
    accelerated = {
        "GD": run_gd(kappa, 0.05, steps),
        "Momentum (0.9)": run_gd(kappa, 0.02, steps, momentum=0.9),
        "Adam": run_gd(kappa, 0.15, steps, use_adam=True),
    }

    # ------------------------------------------------------------------
    # 3. Mini-batch SGD for linear regression
    # ------------------------------------------------------------------
    n, p = 500, 3
    X = rng.normal(size=(n, p))
    true_beta = np.array([1.5, -2.0, 0.75])
    y = 0.4 + X @ true_beta + rng.normal(0.0, 0.5, size=n)
    X_bias = np.column_stack([np.ones(n), X])

    def full_loss(beta: np.ndarray) -> float:
        residual = y - X_bias @ beta
        return 0.5 * np.mean(residual ** 2)

    def batch_gradient(beta: np.ndarray, idx: np.ndarray) -> np.ndarray:
        residual = y[idx] - X_bias[idx] @ beta
        return -(X_bias[idx].T @ residual) / len(idx)

    beta = np.zeros(p + 1)
    lr = 0.05
    epochs = 60
    batch_size = 32
    sgd_path = []
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            beta = beta - lr * batch_gradient(beta, idx)
        sgd_path.append(full_loss(beta))
    closed_form = np.linalg.solve(X_bias.T @ X_bias, X_bias.T @ y)
    print("Mini-batch SGD for linear regression")
    print(f"  closed-form coefficients = {np.round(closed_form, 3)}")
    print(f"  SGD coefficients          = {np.round(beta, 3)}")
    print(f"  final loss                = {full_loss(beta):.5f}")

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    theta1 = np.linspace(-2.8, 2.8, 250)
    theta2 = np.linspace(-2.2, 2.2, 250)
    T1, T2 = np.meshgrid(theta1, theta2)
    Z = 0.5 * (T1 ** 2 + kappa * T2 ** 2)
    axes[0].contour(T1, T2, Z, levels=np.logspace(-1, 2.2, 18), cmap="Greys", linewidths=0.6)
    for lr, path in paths.items():
        axes[0].plot(path[:, 0], path[:, 1], marker=".", markersize=2,
                     label=f"lr={lr}")
    axes[0].plot(0, 0, "r*", markersize=12)
    axes[0].set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$",
                title="Learning-rate trajectories")
    axes[0].legend(fontsize=8)

    for lr, path in paths.items():
        losses = [loss(theta, kappa) for theta in path]
        axes[1].plot(losses, label=f"lr={lr}")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Iteration", ylabel="Loss", title="Effect of learning rate")
    axes[1].legend(fontsize=8)

    for name, path in accelerated.items():
        losses = [loss(theta, kappa) for theta in path]
        axes[2].plot(losses, label=name)
    axes[2].set_yscale("log")
    axes[2].set(xlabel="Iteration", ylabel="Loss", title="Momentum and Adam")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
