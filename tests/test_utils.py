import matplotlib.pyplot as plt
import numpy as np
import pytest
from knn_utils import (
    best_row,
    build_knn_pipeline,
    distance_concentration,
    fit_and_time,
    make_synthetic_images,
    plot_image_gallery,
    plot_k_curve,
    tune_k,
)
from plotting_utils import decision_grid, plot_score_boundary, score_grid


def test_build_knn_pipeline() -> None:
    pipeline = build_knn_pipeline(k=3, scale=True, weights="distance")
    assert [name for name, _ in pipeline.steps] == ["scale", "knn"]
    assert pipeline.named_steps["knn"].n_neighbors == 3
    with pytest.raises(ValueError):
        build_knn_pipeline(k=0)
    with pytest.raises(ValueError):
        build_knn_pipeline(task="unknown")


def test_tune_k_and_best_row() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    y = (X[:, 0] > 0).astype(int)
    results = tune_k(X, y, ks=[1, 3, 5], cv=3)
    assert results["k"].tolist() == [1, 3, 5]
    assert results["mean_score"].shape == (3,)
    best = best_row(results)
    assert best["k"] in {1.0, 3.0, 5.0}


def test_fit_and_time() -> None:
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 2)), rng.integers(0, 2, 40)
    metrics = fit_and_time(build_knn_pipeline(k=2), X[:30], y[:30], X[30:], y[30:])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["ms_per_query"] >= 0.0


def test_plot_k_curve() -> None:
    results = {"k": np.array([1, 2]), "mean_score": np.array([0.7, 0.8]),
               "std_score": np.array([0.02, 0.03])}
    fig, ax = plt.subplots()
    plot_k_curve(results, ax=ax)
    assert ax.get_xlabel() == "k"
    plt.close(fig)


def test_decision_grid_and_score_grid() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    xx, yy, grid = decision_grid(X, resolution=10, padding=0.1)
    assert xx.shape == (10, 10)
    assert grid.shape == (100, 2)

    class ConstantModel:
        def predict_proba(self, values):
            return np.column_stack([np.ones(len(values)) * 0.7,
                                    np.ones(len(values)) * 0.3])

    scores = score_grid(ConstantModel(), grid)
    assert scores.shape == (100,)
    np.testing.assert_allclose(scores, 0.3)
    fig, ax = plt.subplots()
    plot_score_boundary(ax, lambda g: score_grid(ConstantModel(), g), X, np.array([0, 1]))
    plt.close(fig)


def test_synthetic_images_and_distance_concentration() -> None:
    X, y, names = make_synthetic_images(n_per_class=3, image_size=(8, 8), n_classes=2)
    assert X.shape == (6, 64)
    assert set(y.tolist()) == {0, 1}
    assert len(names) == 2
    stats = distance_concentration(X, n_query=3)
    assert 0.0 <= stats["ratio"] <= 1.0


def test_distance_concentration_excludes_self_matches() -> None:
    X = np.eye(4) * 10.0
    stats = distance_concentration(X, n_query=4)
    assert stats["mean_nearest"] == pytest.approx(10.0 * np.sqrt(2))
    assert stats["mean_farthest"] == pytest.approx(10.0 * np.sqrt(2))
    assert stats["ratio"] == pytest.approx(1.0)


def test_plot_image_gallery_accepts_single_axes_for_one_image() -> None:
    X, y, _ = make_synthetic_images(n_per_class=1, image_size=(4, 4), n_classes=1)
    fig, ax = plt.subplots()
    result = plot_image_gallery(X, y, image_shape=(4, 4), n_per_class=1, ax=ax)
    assert result is ax
    assert fig._suptitle is not None
    plt.close(fig)
