"""Reinforcement-learning companion script.

Solves a small grid world with value iteration, Q-learning, and SARSA.
Run from the repository root:

    python Code/reinforcement_learning.py
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
ACTION_NAMES = ["up", "down", "left", "right"]
ARROWS = {0: (0, 0.35), 1: (0, -0.35), 2: (-0.35, 0), 3: (0.35, 0)}


class GridWorld:
    """A deterministic grid world with a goal, a pit, and blocked cells."""

    def __init__(
        self,
        size: int = 5,
        gamma: float = 0.95,
        obstacles: set[tuple[int, int]] | None = None,
        pit: tuple[int, int] | None = None,
        step_reward: float = -0.04,
    ):
        if size < 4:
            raise ValueError("size must be at least 4 for the default layout")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        self.size = size
        self.gamma = gamma
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = ({(1, 1), (2, 2), (3, 1)} if obstacles is None
                          else set(obstacles))
        self.pit = (1, size - 2) if pit is None else pit
        self.step_reward = step_reward
        self.n_states = size * size
        self.n_actions = len(ACTIONS)
        special = self.obstacles | {self.goal, self.pit}
        if self.start in special:
            raise ValueError("start may not coincide with an obstacle, goal, or pit")
        for state in special:
            if not (0 <= state[0] < size and 0 <= state[1] < size):
                raise ValueError(f"special state {state} lies outside the grid")

    def state_id(self, state: tuple[int, int]) -> int:
        """Map an ``(row, col)`` pair to a row-major index."""
        return state[0] * self.size + state[1]

    def id_state(self, idx: int) -> tuple[int, int]:
        """Inverse of :meth:`state_id`."""
        return divmod(idx, self.size)

    def is_terminal(self, state: tuple[int, int]) -> bool:
        """Whether ``state`` ends the episode."""
        return state == self.goal or state == self.pit

    def step(self, state: tuple[int, int], action: int) -> tuple[tuple[int, int], float, bool]:
        """Apply ``action``; blocked or out-of-grid moves leave the agent in place."""
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions - 1}]")
        if self.is_terminal(state):
            return state, 0.0, True
        dr, dc = ACTIONS[action]
        nxt = (state[0] + dr, state[1] + dc)
        if not (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size):
            nxt = state
        if nxt in self.obstacles:
            nxt = state
        if nxt == self.goal:
            return nxt, 1.0, True
        if nxt == self.pit:
            return nxt, -1.0, True
        return nxt, self.step_reward, False

    def value_iteration(
        self, tol: float = 1e-9, max_iter: int = 10_000
    ) -> np.ndarray:
        """Compute the optimal state-value function with in-place updates."""
        if tol <= 0:
            raise ValueError("tol must be positive")
        V = np.zeros(self.n_states)
        for _ in range(max_iter):
            delta = 0.0
            for s_idx in range(self.n_states):
                state = self.id_state(s_idx)
                if self.is_terminal(state):
                    continue
                values = []
                for a in range(self.n_actions):
                    nxt, reward, done = self.step(state, a)
                    values.append(
                        reward + (0.0 if done
                                  else self.gamma * V[self.state_id(nxt)])
                    )
                best = max(values)
                delta = max(delta, abs(best - V[s_idx]))
                V[s_idx] = best
            if delta < tol:
                break
        return V

    def q_from_values(self, V: np.ndarray) -> np.ndarray:
        """Derive the greedy action-value function from a value function."""
        Q = np.zeros((self.n_states, self.n_actions))
        for s_idx in range(self.n_states):
            state = self.id_state(s_idx)
            if self.is_terminal(state):
                continue
            for a in range(self.n_actions):
                nxt, reward, done = self.step(state, a)
                Q[s_idx, a] = reward + (0.0 if done else
                                        self.gamma * V[self.state_id(nxt)])
        return Q

    def render(self, policy: np.ndarray | None = None) -> str:
        """Return an ASCII rendering of the grid ('S', 'G', 'X', '!', arrows)."""
        lines = []
        for row in range(self.size):
            cells = []
            for col in range(self.size):
                state = (row, col)
                if state == self.start:
                    cells.append("S")
                elif state == self.goal:
                    cells.append("G")
                elif state == self.pit:
                    cells.append("!")
                elif state in self.obstacles:
                    cells.append("X")
                elif policy is not None:
                    cells.append(ACTION_NAMES[int(policy[self.state_id(state)])][0].upper())
                else:
                    cells.append(".")
            lines.append(" ".join(cells))
        return "\n".join(lines)


def epsilon_greedy(
    Q: np.ndarray, state_idx: int, epsilon: float, rng: np.random.Generator
) -> int:
    """Pick an action with random ties among the current best actions."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    best = np.flatnonzero(Q[state_idx] == Q[state_idx].max())
    return int(rng.choice(best))


def td_learning(
    env: GridWorld,
    episodes: int = 800,
    alpha: float = 0.1,
    epsilon: float = 0.2,
    seed: int = 0,
    on_policy: bool = False,
    max_steps: int = 100,
) -> tuple[np.ndarray, list[float]]:
    """Run Q-learning (``on_policy=False``) or SARSA (``on_policy=True``)."""
    if episodes < 1 or alpha <= 0 or not 0.0 <= epsilon <= 1.0:
        raise ValueError("invalid TD learning parameters")
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(episodes):
        state = env.start
        action = epsilon_greedy(Q, env.state_id(state), epsilon, rng)
        total = 0.0
        for _ in range(max_steps):
            s_idx = env.state_id(state)
            nxt, reward, done = env.step(state, action)
            nxt_idx = env.state_id(nxt)
            next_action = epsilon_greedy(Q, nxt_idx, epsilon, rng)
            target = reward + (0.0 if done else env.gamma * (
                Q[nxt_idx, next_action] if on_policy else Q[nxt_idx].max()
            ))
            Q[s_idx, action] += alpha * (target - Q[s_idx, action])
            total += reward
            state, action = nxt, next_action
            if done:
                break
        returns.append(total)
    return Q, returns


def q_learning(env: GridWorld, **kwargs) -> tuple[np.ndarray, list[float]]:
    """Off-policy TD control (Q-learning)."""
    return td_learning(env, on_policy=False, **kwargs)


def sarsa(env: GridWorld, **kwargs) -> tuple[np.ndarray, list[float]]:
    """On-policy TD control (SARSA)."""
    return td_learning(env, on_policy=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    env = GridWorld(size=5)
    V = env.value_iteration().reshape(env.size, env.size)
    Q_opt = env.q_from_values(V.ravel())
    Q_ql, returns_ql = q_learning(env, episodes=args.episodes)
    Q_sarsa, returns_sarsa = sarsa(env, episodes=args.episodes)

    print("Grid world")
    print(env.render(np.argmax(Q_opt, axis=1)))
    print(f"  value at start = {V[env.start]:.3f}")
    print(f"  optimal action at start = "
          f"{ACTION_NAMES[int(np.argmax(Q_opt[env.state_id(env.start)]))]}")
    print(f"  Q-learning final average return (last 100) = "
          f"{np.mean(returns_ql[-100:]):.3f}")
    print(f"  SARSA final average return (last 100)      = "
          f"{np.mean(returns_sarsa[-100:]):.3f}")

    if args.no_plots:
        return

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    image = axes[0].imshow(V, cmap="viridis")
    fig.colorbar(image, ax=axes[0], shrink=0.8)
    axes[0].set_title("Optimal value function")
    for state in env.obstacles:
        axes[0].add_patch(plt.Rectangle((state[1] - 0.5, state[0] - 0.5), 1, 1,
                                        color="black"))
    axes[0].scatter(env.start[1], env.start[0], marker="s", s=120,
                    facecolors="none", edgecolors="white", linewidths=2)
    axes[0].scatter(env.goal[1], env.goal[0], marker="*", s=250, color="gold")
    axes[0].scatter(env.pit[1], env.pit[0], marker="x", s=150, color="red")
    axes[0].set_xticks(range(env.size))
    axes[0].set_yticks(range(env.size))

    for s_idx in range(env.n_states):
        state = env.id_state(s_idx)
        if env.is_terminal(state) or state in env.obstacles:
            continue
        dx, dy = ARROWS[int(np.argmax(Q_opt[s_idx]))]
        axes[0].arrow(state[1], state[0], dx, dy, color="white", head_width=0.08,
                      length_includes_head=True)

    window = 25
    kernel = np.ones(window) / window
    episodes_axis = np.arange(window - 1, len(returns_ql))
    axes[1].plot(returns_ql, alpha=0.35, label="Q-learning")
    axes[1].plot(returns_sarsa, alpha=0.35, label="SARSA")
    axes[1].plot(episodes_axis, np.convolve(returns_ql, kernel, mode="valid"),
                 color="tab:blue", linewidth=2)
    axes[1].plot(episodes_axis, np.convolve(returns_sarsa, kernel, mode="valid"),
                 color="tab:orange", linewidth=2)
    axes[1].set(xlabel="Episode", ylabel="Return", title="Learning curves")
    axes[1].legend()

    greedy_actions = np.argmax(Q_ql, axis=1).reshape(env.size, env.size)
    axes[2].imshow(greedy_actions, cmap="tab10", vmin=0, vmax=3)
    axes[2].set_title("Q-learning greedy policy")
    axes[2].set_xticks(range(env.size))
    axes[2].set_yticks(range(env.size))
    for state in env.obstacles:
        axes[2].add_patch(plt.Rectangle((state[1] - 0.5, state[0] - 0.5), 1, 1,
                                        color="black"))

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
