import numpy as np
from mnist_supervised import evaluate_models, load_digit_data, make_models
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def test_load_digit_data_force_digits() -> None:
    X, y, used_mnist = load_digit_data(force_digits=True)
    assert X.shape == (1797, 64)
    assert y.shape == (1797,)
    assert set(np.unique(y)) <= set(range(10))
    assert used_mnist is False


def test_make_models_covers_the_demo_classifiers() -> None:
    assert set(make_models(seed=0)) == {
        "Logistic Regression", "k-NN (k=5)", "SVM (RBF)", "Random Forest",
    }


def test_evaluate_models_returns_test_accuracy() -> None:
    X, y, _ = load_digit_data(force_digits=True)
    mask = np.isin(y, [0, 1])
    X, y = X[mask][:80], y[mask][:80]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0,
    )
    results = evaluate_models(
        {"k-NN": KNeighborsClassifier(n_neighbors=3)},
        X_train, X_test, y_train, y_test,
    )
    assert set(results) == {"k-NN"}
    assert 0.0 <= results["k-NN"] <= 1.0
