"""Bayesian tuning over expanding temporal folds."""

from __future__ import annotations

import numpy as np
import optuna

from collab_sol.evaluate import binary_average_precision
from collab_sol.splitting import TemporalSplit


def tune_model(
    *,
    model_factory,
    suggest_params,
    X,
    y,
    split: TemporalSplit,
    preprocessor_factory=None,
    fit_model=None,
    scorer=binary_average_precision,
    n_trials: int = 20,
    random_state: int = 42,
    study_name: str | None = None,
    storage: str | None = None,
) -> optuna.Study:
    """Tune any probabilistic classifier without exposing the holdout.

    ``model_factory(params, random_state)`` must return an object with
    ``fit(X, y)`` and ``predict_proba(X)`` methods. Supply ``fit_model`` only
    when the model needs validation data during fitting, such as for early
    stopping or a custom neural-network training loop.
    """

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        fold_scores: list[float] = []

        for fold_number, fold in enumerate(split.folds):
            X_train = _take_rows(X, fold.train_indices)
            y_train = _take_rows(y, fold.train_indices)
            X_valid = _take_rows(X, fold.validation_indices)
            y_valid = _take_rows(y, fold.validation_indices)

            if preprocessor_factory is None:
                transformed_train = X_train
                transformed_valid = X_valid
            else:
                preprocessor = preprocessor_factory()
                transformed_train = preprocessor.fit_transform(X_train, y_train)
                transformed_valid = preprocessor.transform(X_valid)

            model = model_factory(params, random_state + fold_number)
            train_labels = np.asarray(y_train)
            validation_labels = np.asarray(y_valid)
            if fit_model is None:
                model.fit(transformed_train, train_labels)
            else:
                fit_model(
                    model,
                    transformed_train,
                    train_labels,
                    transformed_valid,
                    validation_labels,
                )
            probabilities = model.predict_proba(transformed_valid)
            fold_score = scorer(validation_labels, probabilities)
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
    return study


def _take_rows(values, indices):
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    return values[indices]
