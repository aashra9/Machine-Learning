import numpy as np
import pytest
from cross_validation import inject_missing, make_model
from sklearn.pipeline import Pipeline


def test_make_model_returns_scaled_pipeline() -> None:
    model = make_model(C=0.5)
    assert isinstance(model, Pipeline)
    assert len(model.steps) == 2
    assert model[-1].C == 0.5


def test_inject_missing_validates_arguments() -> None:
    X = np.arange(20.0).reshape(5, 4)
    with pytest.raises(ValueError, match="rate"):
        inject_missing(X, -0.1, np.random.default_rng(0))
    with pytest.raises(ValueError, match="rate"):
        inject_missing(X, 1.0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="min_observations"):
        inject_missing(X, 0.5, np.random.default_rng(0), min_observations_per_column=6)


def test_inject_missing_preserves_observations() -> None:
    X = np.arange(20.0).reshape(5, 4)
    result = inject_missing(X, 0.9, np.random.default_rng(0),
                            min_observations_per_column=2)
    assert result.shape == X.shape
    assert np.all(np.isfinite(result).sum(axis=0) >= 2)
    observed = ~np.isnan(result)
    np.testing.assert_allclose(result[observed], X[observed])
    assert result[~observed].size > 0
