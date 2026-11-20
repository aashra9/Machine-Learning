import numpy as np
from trees import depth_sweep, make_models


def test_make_models_covers_the_demo_models() -> None:
    models = make_models(seed=0)
    assert set(models) == {
        "Decision tree (depth 3)",
        "Decision tree (full)",
        "Random forest",
        "Gradient boosting",
    }


def test_depth_sweep_scores_train_and_validation() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0], [-1.5], [1.5]])
    y = np.array([0, 0, 1, 1, 0, 1])
    train_acc, valid_acc = depth_sweep(
        X[:4], y[:4], X[4:], y[4:], np.array([1, 2]), seed=0,
    )
    assert len(train_acc) == len(valid_acc) == 2
    assert all(0.0 <= value <= 1.0 for value in train_acc + valid_acc)
    assert valid_acc[0] == 1.0
