"""Classical estimators that do not require neural-model dependencies."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest


class IsolationForestRiskModel:
    """Expose Isolation Forest anomaly percentiles as ranking scores."""

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_samples: int | float | str = 256,
        max_features: float = 1.0,
        score_reference_size: int | None = 50_000,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.score_reference_size = score_reference_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model_: IsolationForest | None = None
        self.training_anomaly_scores_: NDArray[np.float64] | None = None

    def fit(self, X: Any, y: Any = None) -> Self:
        """Fit without labels and retain the training score distribution."""

        model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,  # pyright: ignore[reportArgumentType]
            max_features=self.max_features,
            contamination="auto",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        model.fit(X)
        reference = X
        n_rows = int(X.shape[0])
        if (
            self.score_reference_size is not None
            and n_rows > self.score_reference_size
        ):
            rng = np.random.default_rng(self.random_state)
            reference_rows = rng.choice(
                n_rows,
                size=self.score_reference_size,
                replace=False,
            )
            reference = (
                X.iloc[reference_rows]
                if hasattr(X, "iloc")
                else X[reference_rows]
            )
        anomaly_scores = -model.score_samples(reference)
        self.model_ = model
        self.training_anomaly_scores_ = np.sort(
            np.asarray(anomaly_scores, dtype=np.float64)
        )
        return self

    def predict_proba(self, X: Any) -> NDArray[np.float32]:
        """Return anomaly percentiles in the positive-class score column."""

        if self.model_ is None or self.training_anomaly_scores_ is None:
            raise RuntimeError("Call fit before predict_proba.")
        anomaly_scores = -self.model_.score_samples(X)
        reference_percentiles = np.linspace(
            1.0 / len(self.training_anomaly_scores_),
            1.0,
            len(self.training_anomaly_scores_),
        )
        positive = np.interp(
            anomaly_scores,
            self.training_anomaly_scores_,
            reference_percentiles,
            left=0.0,
            right=1.0,
        )
        return np.column_stack([1.0 - positive, positive]).astype(
            np.float32, copy=False
        )
