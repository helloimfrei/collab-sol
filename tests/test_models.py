from __future__ import annotations

import numpy as np
import optuna
import pytest
from sklearn.datasets import make_classification

from collab_sol.models import XGBoostModel


def test_xgboost_exposes_search_space_through_shared_contract() -> None:
    expected = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 2.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.01,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    trial = optuna.trial.FixedTrial(expected)

    params = XGBoostModel.suggest_params(trial)

    assert params == expected


@pytest.mark.parametrize("n_classes", [2, 4])
def test_xgboost_fits_binary_and_multiclass_targets(n_classes: int) -> None:
    generated_X, generated_y = make_classification(
        n_samples=240,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=[0.85, 0.15] if n_classes == 2 else None,
        random_state=42,
    )
    X = np.asarray(generated_X)
    y = np.asarray(generated_y)
    split = 180
    model = XGBoostModel(
        params={"n_estimators": 25, "max_depth": 3},
        early_stopping_rounds=5,
    )

    model.fit(X[:split], y[:split], X[split:], y[split:])
    probabilities = model.predict_proba(X[split:])

    assert probabilities.shape == (len(X) - split, n_classes)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_xgboost_requires_a_complete_validation_pair() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    model = XGBoostModel(params={"n_estimators": 2})

    with pytest.raises(ValueError, match="both X_valid and y_valid"):
        model.fit(X, y, X_valid=X)


def test_xgboost_can_fit_without_validation_data() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = XGBoostModel(params={"n_estimators": 2})

    model.fit(X, y)

    assert model.predict_proba(X).shape == (4, 2)


def test_xgboost_rejects_noncontiguous_labels() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1, 2, 1, 2])
    model = XGBoostModel(params={"n_estimators": 2})

    with pytest.raises(ValueError, match="contiguous integers starting at 0"):
        model.fit(X, y)
