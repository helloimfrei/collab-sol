"""Model-agnostic Bayesian tuning over expanding temporal folds."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, cast

import numpy as np
import optuna
import pandas as pd
from numpy.typing import NDArray

from collab_sol.evaluate import binary_average_precision
from collab_sol.models.base import BaseModel, FeatureMatrix
from collab_sol.splitting import TemporalSplit


class FoldPreprocessor(Protocol):
    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> FeatureMatrix: ...

    def transform(self, X: pd.DataFrame) -> FeatureMatrix: ...


ProbabilityScorer = Callable[
    [NDArray[Any], NDArray[np.floating[Any]]],
    float,
]


def tune_model(
    *,
    model_class: type[BaseModel],
    X: pd.DataFrame,
    y: pd.Series,
    split: TemporalSplit,
    preprocessor_factory: Callable[[], FoldPreprocessor],
    scorer: ProbabilityScorer = binary_average_precision,
    n_trials: int = 20,
    random_state: int = 42,
    study_name: str | None = None,
    storage: str | None = None,
) -> optuna.Study:
    """Tune one model without exposing the final temporal holdout."""

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    def objective(trial: optuna.Trial) -> float:
        params = model_class.suggest_params(trial)
        fold_scores: list[float] = []

        for fold_number, fold in enumerate(split.folds):
            X_train = X.iloc[fold.train_indices]
            y_train = y.iloc[fold.train_indices]
            X_valid = X.iloc[fold.validation_indices]
            y_valid = y.iloc[fold.validation_indices]

            preprocessor = preprocessor_factory()
            transformed_train = preprocessor.fit_transform(X_train, y_train)
            transformed_valid = preprocessor.transform(X_valid)
            model = model_class(
                params=params,
                random_state=random_state + fold_number,
            )
            model.fit(
                transformed_train,
                y_train.to_numpy(),
                transformed_valid,
                y_valid.to_numpy(),
            )
            probabilities = model.predict_proba(transformed_valid)
            fold_score = scorer(y_valid.to_numpy(), probabilities)
            fold_scores.append(fold_score)

            trial.report(float(np.mean(fold_scores)), step=fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr(
            "validation_years",
            [fold.validation_year for fold in split.folds],
        )
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return cast(optuna.Study, study)
