import numpy as np
from sklearn.datasets import make_blobs
from unsupervised import cluster_sweep, dbscan_grid


def _blobs() -> tuple[np.ndarray, np.ndarray]:
    return make_blobs(
        n_samples=150,
        centers=np.array([[-5.0, 0.0], [0.0, 5.0], [5.0, 0.0]]),
        cluster_std=0.4,
        random_state=0,
    )


def test_cluster_sweep_selects_the_best_k() -> None:
    X, _ = _blobs()
    inertias, silhouettes, best = cluster_sweep(
        X, np.array([2, 3, 4]), np.arange(len(X)),
    )
    assert len(inertias) == len(silhouettes) == 3
    assert inertias[0] > inertias[-1]
    assert best.n_clusters == 3
    assert silhouettes[1] == max(silhouettes)


def test_dbscan_grid_reports_clusters_and_noise() -> None:
    X, _ = _blobs()
    results = dbscan_grid(X, np.array([0.5]), np.array([3, 5]))
    assert set(results) == {(0.5, 3), (0.5, 5)}
    for metrics in results.values():
        assert set(metrics) == {"silhouette", "n_clusters", "noise_fraction"}
        assert 0.0 <= metrics["noise_fraction"] <= 1.0
