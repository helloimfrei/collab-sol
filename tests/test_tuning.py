from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from collab_sol.splitting import make_expanding_temporal_split
from collab_sol.tuning import tune_model


class IdentityPreprocessor:
    def fit_transform(self, X, y=None):
        del y
        return X.to_numpy()

    def transform(self, X):
        return X.to_numpy()


class RecordingModel:
    seen_rows: list[int] = []

    def __init__(self, probability):
        self.probability = probability

    def fit(self, X_train, y_train):
        del y_train
        self.seen_rows.extend(np.asarray(X_train)[:, 0].astype(int).tolist())
        return self

    def predict_proba(self, X):
        return np.tile([1.0 - self.probability, self.probability], (len(X), 1))


def suggest_probability(trial):
    return {"probability": trial.suggest_float("probability", 0.1, 0.9)}


def build_recording_model(params, random_state):
    del random_state
    return RecordingModel(**params)


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
        model_factory=build_recording_model,
        suggest_params=suggest_probability,
        X=frame[["row_id"]],
        y=frame["risk_target"],
        split=split,
        preprocessor_factory=IdentityPreprocessor,
        n_trials=2,
    )

    holdout_rows = set(frame.iloc[split.holdout_indices]["row_id"])
    assert len(study.trials) == 2
    assert holdout_rows.isdisjoint(RecordingModel.seen_rows)


def test_tuner_accepts_an_ordinary_sklearn_model() -> None:
    frame = pd.DataFrame(
        {
            "feature": np.arange(12),
            "year": [year for year in range(2018, 2024) for _ in range(2)],
            "target": [0, 1] * 6,
        }
    )
    split = make_expanding_temporal_split(frame, year_column="year")

    def build_model(params, random_state):
        return LogisticRegression(**params, random_state=random_state)

    study = tune_model(
        model_factory=build_model,
        suggest_params=lambda trial: {
            "C": trial.suggest_float("C", 0.1, 1.0)
        },
        X=frame[["feature"]],
        y=frame["target"],
        split=split,
        n_trials=1,
    )

    assert len(study.trials) == 1


def test_custom_fit_function_receives_validation_data() -> None:
    frame = pd.DataFrame(
        {
            "row_id": np.arange(12),
            "year": [year for year in range(2018, 2024) for _ in range(2)],
            "target": [0, 1] * 6,
        }
    )
    split = make_expanding_temporal_split(frame, year_column="year")
    validation_sizes = []

    def fit_model(model, X_train, y_train, X_valid, y_valid):
        validation_sizes.append(len(X_valid))
        model.fit(X_train, y_train)

    tune_model(
        model_factory=build_recording_model,
        suggest_params=suggest_probability,
        X=frame[["row_id"]],
        y=frame["target"],
        split=split,
        fit_model=fit_model,
        n_trials=1,
    )

    assert validation_sizes == [2, 2, 2, 2]
