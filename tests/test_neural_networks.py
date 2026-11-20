import numpy as np
import pytest
from neural_networks import MLP, Perceptron


def test_perceptron_preserves_signed_labels() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([-1, 1])
    model = Perceptron(learning_rate=0.1, n_epochs=10).fit(X, y)
    assert set(np.unique(model.predict(X))).issubset({-1, 1})
    assert model.score(X, y) == 1.0


def test_perceptron_preserves_zero_one_labels() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    model = Perceptron(learning_rate=0.1, n_epochs=10).fit(X, y)
    assert set(np.unique(model.predict(X))).issubset({0, 1})
    assert model.score(X, y) == 1.0


def test_mlp_requires_fit() -> None:
    model = MLP(2, n_hidden=4)
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(np.zeros((1, 2)))


def test_mlp_validates_input_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        MLP(2, n_epochs=1).fit(np.zeros((3, 2)), np.array([0, 1]))


def test_mlp_fits_separable_data() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0],
                  [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]])
    y = np.array([0, 0, 1, 1, 1, 1])
    model = MLP(2, n_hidden=8, learning_rate=0.5, n_epochs=400, seed=0).fit(X, y)
    assert model.fitted_
    assert len(model.loss_history) == 400
    assert model.loss_history[-1] < model.loss_history[0]
    assert model.score(X, y) == 1.0
