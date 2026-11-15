"""Tune and evaluate a conditional-severity regression model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any, cast

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CODE_ROOT = PACKAGE_ROOT.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model",
    choices=["ridge", "xgboost", "mlp", "ft_transformer"],
    required=True,
)
parser.add_argument(
    "--dataset",
    type=Path,
    default=PROJECT_CODE_ROOT / "data" / "final" / "final_dataset.parquet",
)
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--trials", type=int, default=50)
parser.add_argument("--max-epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--n-jobs", type=int, default=-1)
parser.add_argument(
    "--xgboost-device", choices=["cpu", "cuda"], default="cpu"
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--require-cuda", action="store_true")
parser.add_argument(
    "--smoke",
    action="store_true",
    help="use a small yearly sample, one trial, and two neural epochs",
)
args = parser.parse_args()

if args.smoke:
    args.trials = 1
    args.max_epochs = 2
    args.patience = 2

dataset_path = args.dataset.resolve()
if args.output_dir is None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "_smoke" if args.smoke else ""
    args.output_dir = (
        PACKAGE_ROOT
        / "runs"
        / f"severity_regression_{args.model}_{timestamp}{suffix}"
    )
run_dir = args.output_dir.resolve()

if not dataset_path.is_file():
    raise FileNotFoundError(f"Final dataset not found: {dataset_path}")
if run_dir.exists() and any(run_dir.iterdir()):
    raise FileExistsError(f"Output directory is not empty: {run_dir}")
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
run_dir.mkdir(parents=True, exist_ok=True)

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from collab_sol import (
    SEVERITY_VALUES,
    generate_severity_regression_report,
    make_expanding_temporal_split,
    severity_regression_mae,
    tune_model,
)
from collab_sol.features import (
    EXCLUDED_RISK_FEATURES,
    split_risk_feature_columns,
)
from collab_sol.models import (
    FTTransformerRegressor,
    TorchMLPRegressor,
    build_ridge_regression,
    build_xgboost_regressor,
)
from collab_sol.preprocessing import (
    make_ft_transformer_preprocessor,
    make_one_hot_preprocessor,
)
from _severity_search import suggest_params as suggest_search_params

numeric_features, categorical_features = split_risk_feature_columns(
    pq.read_schema(dataset_path).names
)
feature_columns = numeric_features + categorical_features
frame = cast(
    pd.DataFrame,
    pd.read_parquet(
        dataset_path,
        columns=[
            "crossingid",
            "asof_year",
            "incident",
            "severity_target",
            *feature_columns,
        ],
        filters=[("incident", "=", 1)],
    ),
)
invalid_targets = ~frame["severity_target"].isin(SEVERITY_VALUES)
if bool(invalid_targets.any()):
    invalid = sorted(frame.loc[invalid_targets, "severity_target"].unique())
    raise ValueError(f"Unexpected severity targets: {invalid}")

if args.smoke:
    frame = cast(
        pd.DataFrame,
        frame.groupby("asof_year", group_keys=False)
        .sample(n=min(240, int(frame.groupby("asof_year").size().min())), random_state=args.seed)
        .reset_index(drop=True),
    )

frame[numeric_features] = frame[numeric_features].astype(np.float32)
for column in categorical_features:
    frame[column] = frame[column].astype("category")

y = cast(
    pd.Series,
    frame["severity_target"]
    .astype(str)
    .map(lambda value: SEVERITY_VALUES[value])
    .astype(np.float32),
)
yearly_summary = frame.assign(severity_value=y).groupby("asof_year").agg(
    crash_rows=("crossingid", "size"),
    mean_severity=("severity_value", "mean"),
)
yearly_summary.to_csv(run_dir / "severity_by_year.csv")
print(yearly_summary)
print("Dataset:", dataset_path)
print(
    f"Using {len(feature_columns)} predictors: "
    f"{len(numeric_features)} numeric + {len(categorical_features)} categorical"
)

X = cast(pd.DataFrame, frame[feature_columns])
split = make_expanding_temporal_split(frame, year_column="asof_year")
crossing_ids = np.asarray(frame["crossingid"])
del frame


def new_preprocessor():
    if args.model == "ft_transformer":
        return make_ft_transformer_preprocessor(
            numeric_features,
            categorical_features,
            random_state=args.seed,
        )
    return make_one_hot_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=args.model in {"ridge", "mlp"},
    )


def suggest_params(trial):
    return suggest_search_params(
        trial,
        "regression",
        args.model,
        smoke=args.smoke,
    )


def build_model(
    params: dict[str, Any],
    random_state: int,
    *,
    fixed_epochs: int | None = None,
    final_xgboost: bool = False,
) -> Any:
    if args.model == "ridge":
        return build_ridge_regression(params, random_state=random_state)
    if args.model == "xgboost":
        model = build_xgboost_regressor(
            params,
            random_state=random_state,
            device=args.xgboost_device,
            n_jobs=args.n_jobs,
            early_stopping_rounds=None if final_xgboost else (10 if args.smoke else 50),
        )
        setattr(model, "tuned_params_", dict(params))
        return model
    if args.model == "mlp":
        return TorchMLPRegressor(
            **params,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            random_state=random_state,
        )
    if args.model == "ft_transformer":
        model = FTTransformerRegressor(
            **params,
            n_cat_features=len(categorical_features),
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            fixed_epochs=fixed_epochs,
            random_state=random_state,
        )
        model.tuned_params_ = dict(params)
        return model
    raise ValueError(f"Unsupported model: {args.model}")


xgboost_best_rounds: dict[str, list[int]] = {}
transformer_best_epochs: dict[str, list[int]] = {}


def fit_with_validation(model, X_train, y_train, X_valid, y_valid):
    key = json.dumps(getattr(model, "tuned_params_", {}), sort_keys=True)
    if args.model == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        xgboost_best_rounds.setdefault(key, []).append(int(model.best_iteration) + 1)
    elif args.model == "ft_transformer":
        model.fit_with_validation(X_train, y_train, X_valid, y_valid)
        transformer_best_epochs.setdefault(key, []).append(model.best_epoch_)
    else:
        model.fit(X_train, y_train)


if args.require_cuda and args.model in {"mlp", "ft_transformer"}:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is False.")

study = tune_model(
    model_factory=build_model,
    suggest_params=suggest_params,
    X=X,
    y=y,
    split=split,
    preprocessor_factory=new_preprocessor,
    fit_model=(
        fit_with_validation
        if args.model in {"xgboost", "ft_transformer"}
        else None
    ),
    predict_model=lambda model, values: model.predict(values),
    scorer=severity_regression_mae,
    direction="minimize",
    metric_name="MAE",
    n_trials=args.trials,
    random_state=args.seed,
    study_name=f"conditional_severity_regression_{args.model}",
    cache_preprocessing=True,
    verbose=True,
)
study.trials_dataframe().to_csv(run_dir / "optuna_trials.csv", index=False)
print("Mean validation MAE:", study.best_value)
print("Best parameters:", study.best_params)

development_rows = split.development_indices
test_rows = split.holdout_indices
preprocessor = new_preprocessor()
X_development = preprocessor.fit_transform(
    X.iloc[development_rows], y.iloc[development_rows]
)
X_test = preprocessor.transform(X.iloc[test_rows])

final_params = dict(study.best_params)
fixed_epochs = None
best_key = json.dumps(study.best_params, sort_keys=True)
if args.model == "xgboost":
    best_rounds = xgboost_best_rounds[best_key]
    final_params["n_estimators"] = max(1, int(round(float(np.median(best_rounds)))))
if args.model == "ft_transformer":
    best_epochs = transformer_best_epochs[best_key]
    fixed_epochs = max(1, int(round(float(np.median(best_epochs)))))

model = build_model(
    final_params,
    args.seed,
    fixed_epochs=fixed_epochs,
    final_xgboost=True,
)
model.fit(X_development, y.iloc[development_rows].to_numpy())
metrics = generate_severity_regression_report(
    model,
    X_test,
    y.iloc[test_rows].to_numpy(),
    output_dir=run_dir,
    row_ids=crossing_ids[test_rows],
)

joblib.dump(preprocessor, run_dir / "preprocessor.joblib")
if args.model in {"mlp", "ft_transformer"}:
    import torch

    if model.network_ is None:
        raise RuntimeError("The final neural regressor was not fitted.")
    torch.save(model.network_.state_dict(), run_dir / "model_state.pt")
else:
    joblib.dump(model, run_dir / "model.joblib")

metadata = {
    "task": "conditional_severity_regression",
    "model": args.model,
    "dataset_path": str(dataset_path),
    "holdout_year": split.holdout_year,
    "feature_count": len(feature_columns),
    "excluded_features": EXCLUDED_RISK_FEATURES,
    "target_definition": SEVERITY_VALUES,
    "score_range": [1.0, 3.0],
    "mean_validation_mae": study.best_value,
    "validation_fold_scores": study.best_trial.user_attrs.get("fold_scores"),
    "test_metrics": metrics.to_dict(),
    "best_params": study.best_params,
    "final_params": final_params,
    "fixed_epochs": fixed_epochs,
    "n_trials": args.trials,
    "smoke_run": args.smoke,
    "seed": args.seed,
    "python_version": platform.python_version(),
    "command": " ".join(sys.argv),
}
(run_dir / "run.json").write_text(json.dumps(metadata, indent=2))

print(pd.DataFrame([metrics.to_dict()]))
print("Saved results to:", run_dir)
