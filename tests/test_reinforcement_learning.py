import numpy as np
import pytest
from reinforcement_learning import (
    GridWorld,
    epsilon_greedy,
    q_learning,
    sarsa,
    td_learning,
)


def test_state_id_roundtrip() -> None:
    env = GridWorld(size=5)
    for idx in range(env.n_states):
        assert env.state_id(env.id_state(idx)) == idx


def test_step_terminal_and_blocked() -> None:
    env = GridWorld(size=5)
    next_state, reward, done = env.step(env.goal, 0)
    assert next_state == env.goal and done and reward == 0.0
    next_state, reward, done = env.step((0, 0), 0)
    assert next_state == (0, 0) and not done and reward == env.step_reward
    with pytest.raises(ValueError):
        env.step((0, 0), 99)


def test_value_iteration_converges() -> None:
    env = GridWorld(size=5)
    V = env.value_iteration()
    assert V.shape == (env.n_states,)
    assert V[env.state_id(env.start)] > 0
    assert V[env.state_id(env.pit)] == 0


def test_render() -> None:
    env = GridWorld(size=5)
    rendered = env.render()
    assert "S" in rendered and "G" in rendered and "X" in rendered


def test_td_learning_returns_and_validation() -> None:
    env = GridWorld(size=4)
    Q, returns = td_learning(env, episodes=30, seed=0, on_policy=False)
    assert Q.shape == (env.n_states, env.n_actions)
    assert len(returns) == 30
    with pytest.raises(ValueError):
        td_learning(env, episodes=0)


def test_td_target_includes_reward_on_nonterminal_step() -> None:
    class ChainEnv:
        n_states = 2
        n_actions = 1
        gamma = 0.5
        start = 0

        def state_id(self, state: int) -> int:
            return int(state)

        def step(self, state: int, action: int) -> tuple[int, float, bool]:
            if state == 0:
                return 1, 1.0, False
            return 1, 0.0, True

    Q, _ = td_learning(ChainEnv(), episodes=1, alpha=0.2, epsilon=0.0)
    assert Q[0, 0] == pytest.approx(0.2)


def test_q_learning_and_sarsa_wrappers() -> None:
    env = GridWorld(size=4)
    assert len(q_learning(env, episodes=20)[1]) == 20
    assert len(sarsa(env, episodes=20)[1]) == 20


def test_epsilon_greedy_random_tie_break() -> None:
    Q = np.zeros((2, 4))
    rng = np.random.default_rng(0)
    actions = {epsilon_greedy(Q, 0, 0.0, rng) for _ in range(50)}
    assert len(actions) > 1
