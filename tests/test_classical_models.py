from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

from collab_sol.models.anomaly import IsolationForestRiskModel
from collab_sol.models.classical import build_logistic_regression


def test_logistic_regression_uses_scalable_log_loss_solver() -> None:
    model = build_logistic_regression(
        {"alpha": 1e-4, "penalty": "l2", "class_weight": None},
        random_state=42,
    )

    assert isinstance(model, SGDClassifier)
    assert model.loss == "log_loss"
    assert model.early_stopping is True


def test_isolation_forest_returns_anomaly_percentiles() -> None:
    rng = np.random.default_rng(42)
    training = rng.normal(size=(128, 4)).astype(np.float32)
    evaluation = np.vstack([training[:4], np.full((1, 4), 10.0)]).astype(
        np.float32
    )
    model = IsolationForestRiskModel(
        n_estimators=20,
        max_samples=64,
        random_state=42,
    )

    assert model.fit(training) is model
    probabilities = model.predict_proba(evaluation)

    assert probabilities.shape == (5, 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[-1, 1] > probabilities[:-1, 1].mean()
