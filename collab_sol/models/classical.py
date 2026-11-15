"""Factories for supervised classical likelihood models."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier


def build_logistic_regression(
    params: dict[str, Any],
    *,
    random_state: int,
) -> SGDClassifier:
    """Build scalable regularized logistic regression."""

    return SGDClassifier(
        **params,
        loss="log_loss",
        max_iter=100,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        average=True,
        random_state=random_state,
    )


def build_xgboost_classifier(
    params: dict[str, Any],
    *,
    random_state: int,
    device: str = "cpu",
    n_jobs: int = -1,
    early_stopping_rounds: int | None = 50,
):
    """Build the histogram-based XGBoost likelihood model."""

    try:
        classifier = import_module("xgboost").XGBClassifier
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "Install model dependencies with `uv sync --extra models`."
        ) from error
    return classifier(
        **params,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device=device,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )


def build_multiclass_logistic_regression(
    params: dict[str, Any],
    *,
    random_state: int,
) -> LogisticRegression:
    """Build the linear three-class conditional-severity baseline."""

    return LogisticRegression(
        **params,
        solver="lbfgs",
        max_iter=1_000,
        random_state=random_state,
    )


def build_xgboost_multiclass_classifier(
    params: dict[str, Any],
    *,
    random_state: int,
    num_class: int,
    device: str = "cpu",
    n_jobs: int = -1,
    early_stopping_rounds: int | None = 50,
):
    """Build the histogram-based three-class severity classifier."""

    try:
        classifier = import_module("xgboost").XGBClassifier
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "Install model dependencies with `uv sync --extra models`."
        ) from error
    return classifier(
        **params,
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        tree_method="hist",
        device=device,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )


def build_ridge_regression(
    params: dict[str, Any],
    *,
    random_state: int,
) -> Ridge:
    """Build the linear baseline for direct severity regression."""

    return Ridge(**params, random_state=random_state)


def build_xgboost_regressor(
    params: dict[str, Any],
    *,
    random_state: int,
    device: str = "cpu",
    n_jobs: int = -1,
    early_stopping_rounds: int | None = 50,
):
    """Build the histogram-based XGBoost severity regressor."""

    try:
        regressor = import_module("xgboost").XGBRegressor
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "Install model dependencies with `uv sync --extra models`."
        ) from error
    return regressor(
        **params,
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        device=device,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )
