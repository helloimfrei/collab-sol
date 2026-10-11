from __future__ import annotations

import numpy as np
import pandas as pd

from collab_sol.models.base import BaseModel
from collab_sol.splitting import make_expanding_temporal_split
from collab_sol.tuning import tune_model


class IdentityPreprocessor:
    def fit_transform(self, X, y=None):
        del y
        return X.to_numpy()

    def transform(self, X):
        return X.to_numpy()


class RecordingModel(BaseModel):
    name = "recording"
    seen_rows: list[int] = []

    def __init__(self, params=None, *, random_state=42):
        super().__init__(params, random_state=random_state)

    @classmethod
    def suggest_params(cls, trial):
        return {"probability": trial.suggest_float("probability", 0.1, 0.9)}

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        del y_train, y_valid
        self.seen_rows.extend(np.asarray(X_train)[:, 0].astype(int).tolist())
        if X_valid is not None:
            self.seen_rows.extend(np.asarray(X_valid)[:, 0].astype(int).tolist())
        return self

    def predict_proba(self, X):
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
        X=frame[["row_id"]],
        y=frame["risk_target"],
        split=split,
        preprocessor_factory=IdentityPreprocessor,
        n_trials=2,
    )

    holdout_rows = set(frame.iloc[split.holdout_indices]["row_id"])
    assert len(study.trials) == 2
    assert holdout_rows.isdisjoint(RecordingModel.seen_rows)
