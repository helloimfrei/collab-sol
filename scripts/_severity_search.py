"""Shared Optuna search spaces for severity model comparisons."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


SEARCH_SPACES: dict[tuple[str, str], dict[str, dict[str, Any]]] = {
    ("classification", "logistic_regression"): {
        "C": {"kind": "float", "low": 1e-3, "high": 1e3, "log": True},
        "class_weight": {"kind": "categorical", "choices": [None, "balanced"]},
    },
    ("regression", "ridge"): {
        "alpha": {"kind": "float", "low": 1e-4, "high": 100.0, "log": True},
    },
}

TREE_SPACE = {
    "n_estimators": {"kind": "int", "low": 100, "high": 800, "step": 100},
    "max_depth": {"kind": "int", "low": 2, "high": 8},
    "learning_rate": {"kind": "float", "low": 0.01, "high": 0.2, "log": True},
    "min_child_weight": {"kind": "float", "low": 1.0, "high": 30.0, "log": True},
    "subsample": {"kind": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"kind": "float", "low": 0.5, "high": 1.0},
    "reg_lambda": {"kind": "float", "low": 1e-3, "high": 100.0, "log": True},
    "reg_alpha": {"kind": "float", "low": 1e-5, "high": 10.0, "log": True},
}
MLP_SPACE = {
    "hidden_width": {"kind": "categorical", "choices": [64, 128, 256, 512]},
    "n_layers": {"kind": "int", "low": 1, "high": 4},
    "dropout": {"kind": "float", "low": 0.0, "high": 0.5},
    "learning_rate": {"kind": "float", "low": 5e-5, "high": 5e-3, "log": True},
    "weight_decay": {"kind": "float", "low": 1e-8, "high": 1e-2, "log": True},
}
TRANSFORMER_SPACE = {
    "n_blocks": {"kind": "int", "low": 1, "high": 3},
    "d_block": {"kind": "categorical", "choices": [64, 128, 192]},
    "attention_dropout": {"kind": "float", "low": 0.0, "high": 0.5},
    "ffn_dropout": {"kind": "float", "low": 0.0, "high": 0.5},
    "residual_dropout": {"kind": "float", "low": 0.0, "high": 0.2},
    "learning_rate": {"kind": "float", "low": 1e-5, "high": 1e-3, "log": True},
    "weight_decay": {"kind": "float", "low": 1e-6, "high": 1e-3, "log": True},
}

for formulation in ("classification", "regression"):
    SEARCH_SPACES[(formulation, "xgboost")] = TREE_SPACE
    SEARCH_SPACES[(formulation, "mlp")] = MLP_SPACE
    SEARCH_SPACES[(formulation, "ft_transformer")] = TRANSFORMER_SPACE

SMOKE_PARAMS = {
    "logistic_regression": {"C": 1.0, "class_weight": None},
    "ridge": {"alpha": 1.0},
    "xgboost": {
        "n_estimators": 30,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    "mlp": {
        "hidden_width": 64,
        "n_layers": 1,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
    },
    "ft_transformer": {
        "n_blocks": 1,
        "d_block": 64,
        "attention_dropout": 0.1,
        "ffn_dropout": 0.1,
        "residual_dropout": 0.0,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
    },
}


def suggest_params(trial, formulation: str, model: str, *, smoke: bool = False):
    """Suggest one parameter set from the documented search space."""

    if smoke:
        return {
            name: trial.suggest_categorical(name, [value])
            for name, value in SMOKE_PARAMS[model].items()
        }

    parameters = {}
    for name, specification in SEARCH_SPACES[(formulation, model)].items():
        if specification["kind"] == "categorical":
            parameters[name] = trial.suggest_categorical(
                name, specification["choices"]
            )
        elif specification["kind"] == "int":
            parameters[name] = trial.suggest_int(
                name,
                specification["low"],
                specification["high"],
                step=specification.get("step", 1),
                log=specification.get("log", False),
            )
        else:
            parameters[name] = trial.suggest_float(
                name,
                specification["low"],
                specification["high"],
                step=specification.get("step"),
                log=specification.get("log", False),
            )
    return parameters


def write_search_spaces(path: str | Path) -> None:
    """Write every tuned parameter and range to one reviewable CSV."""

    rows = []
    for formulation, model in sorted(SEARCH_SPACES):
        parameters = SEARCH_SPACES[(formulation, model)]
        for parameter, specification in parameters.items():
            rows.append(
                {
                    "formulation": formulation,
                    "model": model,
                    "parameter": parameter,
                    "distribution": specification["kind"],
                    "low": specification.get("low", ""),
                    "high": specification.get("high", ""),
                    "choices": " | ".join(
                        "None" if value is None else str(value)
                        for value in specification.get("choices", [])
                    ),
                    "step": specification.get("step", ""),
                    "log_scale": specification.get("log", False),
                }
            )
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
