"""Factories for supervised classical likelihood models."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sklearn.linear_model import LogisticRegression


def build_logistic_regression(
    params: dict[str, Any],
    *,
    random_state: int,
) -> LogisticRegression:
    """Build the regularized logistic-regression likelihood model."""

    return LogisticRegression(
        **params,
        solver="saga",
        max_iter=1_000,
        tol=1e-3,
        random_state=random_state,
    )


def build_xgboost_classifier(
    params: dict[str, Any],
    *,
    random_state: int,
    device: str = "cpu",
    n_jobs: int = -1,
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
        random_state=random_state,
    )
