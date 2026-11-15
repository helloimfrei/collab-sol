from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier

from collab_sol.models.anomaly import IsolationForestRiskModel
from collab_sol.models.classical import (
    build_logistic_regression,
    build_multiclass_logistic_regression,
    build_ridge_regression,
)


def test_logistic_regression_uses_scalable_log_loss_solver() -> None:
    model = build_logistic_regression(
        {"alpha": 1e-4, "penalty": "l2", "class_weight": None},
        random_state=42,
    )

    assert isinstance(model, SGDClassifier)
    assert model.loss == "log_loss"
    assert model.early_stopping is True


def test_ridge_regression_is_the_linear_severity_baseline() -> None:
    model = build_ridge_regression({"alpha": 1.0}, random_state=42)

    assert isinstance(model, Ridge)
    assert model.alpha == 1.0


def test_multiclass_logistic_regression_is_the_linear_classifier() -> None:
    model = build_multiclass_logistic_regression(
        {"C": 1.0, "class_weight": None},
        random_state=42,
    )

    assert isinstance(model, LogisticRegression)
    assert model.C == 1.0
    assert model.solver == "lbfgs"


def test_isolation_forest_returns_anomaly_percentiles() -> None:
    rng = np.random.default_rng(42)
    training = rng.normal(size=(128, 4)).astype(np.float32)
    evaluation = np.vstack([training[:4], np.full((1, 4), 10.0)]).astype(
        np.float32
    )
    model = IsolationForestRiskModel(
        n_estimators=20,
        max_samples=64,
        score_reference_size=32,
        random_state=42,
    )

    assert model.fit(csr_matrix(training)) is model
    probabilities = model.predict_proba(csr_matrix(evaluation))

    assert probabilities.shape == (5, 2)
    assert model.training_anomaly_scores_ is not None
    assert len(model.training_anomaly_scores_) == 32
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[-1, 1] > probabilities[:-1, 1].mean()
