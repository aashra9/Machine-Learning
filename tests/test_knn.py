import numpy as np
import pytest
from knn import KNNClassifier, pairwise_distances


def test_pairwise_distances() -> None:
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    Y = np.array([[0.0, 0.0], [1.0, 1.0]])
    expected = np.array([[0.0, np.sqrt(2)], [5.0, np.sqrt(13)]])
    np.testing.assert_allclose(pairwise_distances(X, Y), expected, atol=1e-10)


def test_knn_predicts_and_proba() -> None:
    X = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 2.0], [2.1, 2.0]])
    y = np.array([0, 0, 1, 1])
    model = KNNClassifier(k=2).fit(X, y)
    predictions = model.predict(np.array([[0.0, 0.1], [2.0, 1.9]]))
    np.testing.assert_array_equal(predictions, [0, 1])
    proba = model.predict_proba(np.array([[0.0, 0.1]]))
    np.testing.assert_allclose(proba.sum(axis=1), [1.0])


def test_knn_validation() -> None:
    with pytest.raises(ValueError):
        KNNClassifier(k=0)
    with pytest.raises(ValueError):
        KNNClassifier(weights="bad")
    with pytest.raises(RuntimeError):
        KNNClassifier().predict(np.zeros((2, 2)))
