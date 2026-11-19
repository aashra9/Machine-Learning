import numpy as np

from neural_networks import Perceptron


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
