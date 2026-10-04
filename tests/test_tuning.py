from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from collab_sol.models.base import (
    BaseModel,
    FeatureMatrix,
    ModelParams,
    ParameterTrial,
    TargetVector,
)
from collab_sol.splitting import make_expanding_temporal_split
from collab_sol.tuning import tune_model


class IdentityPreprocessor:
    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> FeatureMatrix:
        del y
        return X.to_numpy()

    def transform(self, X: pd.DataFrame) -> FeatureMatrix:
        return X.to_numpy()


class RecordingModel(BaseModel):
    name = "recording"
    seen_rows: list[int] = []

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        random_state: int = 42,
    ) -> None:
        super().__init__(params, random_state=random_state)

    @classmethod
    def suggest_params(cls, trial: ParameterTrial) -> ModelParams:
        return {"probability": trial.suggest_float("probability", 0.1, 0.9)}

    def fit(
        self,
        X_train: FeatureMatrix,
        y_train: TargetVector,
        X_valid: FeatureMatrix | None = None,
        y_valid: TargetVector | None = None,
    ) -> Self:
        del y_train, y_valid
        self.seen_rows.extend(np.asarray(X_train)[:, 0].astype(int).tolist())
        if X_valid is not None:
            self.seen_rows.extend(np.asarray(X_valid)[:, 0].astype(int).tolist())
        return self

    def predict_proba(self, X: FeatureMatrix) -> NDArray[np.floating[Any]]:
        probability = float(self.params["probability"])
        return np.tile([1.0 - probability, probability], (len(X), 1))


def test_tuner_never_passes_holdout_rows_to_model() -> None:
    frame = pd.DataFrame(
        {
            "row_id": np.arange(12),
            "feature_year": [year for year in range(2018, 2024) for _ in range(2)],
            "target_year": [year for year in range(2019, 2025) for _ in range(2)],
            "risk_target": [0, 1] * 6,
        }
    )
    split = make_expanding_temporal_split(frame, target_year_column="target_year")
    RecordingModel.seen_rows = []

    study = tune_model(
        model_class=RecordingModel,
        X=cast(pd.DataFrame, frame[["row_id"]]),
        y=cast(pd.Series, frame["risk_target"]),
        split=split,
        preprocessor_factory=IdentityPreprocessor,
        n_trials=2,
    )

    holdout_rows = set(frame.iloc[split.holdout_indices]["row_id"])
    assert len(study.trials) == 2
    assert holdout_rows.isdisjoint(RecordingModel.seen_rows)
