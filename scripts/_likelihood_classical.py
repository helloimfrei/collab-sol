"""Shared runner for classical crash-likelihood models."""

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


def main(model_name: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Tune and evaluate the {model_name} crash-likelihood model."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_CODE_ROOT / "data" / "final" / "final_dataset.parquet",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--xgboost-device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use 2,000 crossings, one trial, and small model settings",
    )
    args = parser.parse_args()

    if args.smoke:
        args.trials = 1

    dataset_path = args.dataset.resolve()
    if args.output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = "_smoke" if args.smoke else ""
        args.output_dir = (
            PACKAGE_ROOT / "runs" / f"likelihood_{model_name}_{timestamp}{suffix}"
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
        generate_risk_report,
        make_expanding_temporal_split,
        tune_model,
    )
    from collab_sol.features import (
        EXCLUDED_RISK_FEATURES,
        split_risk_feature_columns,
    )
    from collab_sol.models import (
        IsolationForestRiskModel,
        build_logistic_regression,
        build_xgboost_classifier,
    )
    from collab_sol.preprocessing import make_one_hot_preprocessor

    numeric_features, categorical_features = split_risk_feature_columns(
        pq.read_schema(dataset_path).names
    )
    feature_columns = numeric_features + categorical_features
    frame = cast(
        pd.DataFrame,
        pd.read_parquet(
            dataset_path,
            columns=["crossingid", "asof_year", "incident", *feature_columns],
        ),
    )
    if args.smoke:
        crossing_ids = frame["crossingid"].drop_duplicates()
        sampled_ids = crossing_ids.sample(
            n=min(2_000, len(crossing_ids)), random_state=args.seed
        )
        frame = cast(
            pd.DataFrame,
            frame[frame["crossingid"].isin(sampled_ids)].reset_index(drop=True),
        )

    frame[numeric_features] = frame[numeric_features].astype(np.float32)
    for column in categorical_features:
        frame[column] = frame[column].astype("category")

    yearly_summary = frame.groupby("asof_year")["incident"].agg(
        rows="size",
        incidents="sum",
        prevalence="mean",
    )
    yearly_summary.to_csv(run_dir / "class_balance_by_year.csv")
    print(yearly_summary)
    print("Dataset:", dataset_path)
    print(
        f"Using {len(feature_columns)} predictors: "
        f"{len(numeric_features)} numeric + "
        f"{len(categorical_features)} categorical"
    )

    X = cast(pd.DataFrame, frame[feature_columns])
    y = cast(pd.Series, frame["incident"].astype(int))
    split = make_expanding_temporal_split(frame, year_column="asof_year")
    crossing_ids = np.asarray(frame["crossingid"])
    del frame

    scale_numeric = model_name == "logistic_regression"

    def new_preprocessor():
        return make_one_hot_preprocessor(
            numeric_features,
            categorical_features,
            scale_numeric=scale_numeric,
        )

    def suggest_params(trial):
        if model_name == "logistic_regression":
            return {
                "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
                "l1_ratio": trial.suggest_categorical(
                    "l1_ratio", [0.0, 1.0]
                ),
            }
        if model_name == "xgboost":
            if args.smoke:
                return {
                    "n_estimators": trial.suggest_categorical(
                        "n_estimators", [30]
                    ),
                    "max_depth": trial.suggest_categorical("max_depth", [4]),
                    "learning_rate": trial.suggest_categorical(
                        "learning_rate", [0.1]
                    ),
                    "min_child_weight": trial.suggest_categorical(
                        "min_child_weight", [1.0]
                    ),
                    "subsample": trial.suggest_categorical("subsample", [0.8]),
                    "colsample_bytree": trial.suggest_categorical(
                        "colsample_bytree", [0.8]
                    ),
                    "reg_lambda": trial.suggest_categorical("reg_lambda", [1.0]),
                    "scale_pos_weight": trial.suggest_categorical(
                        "scale_pos_weight", [1.0]
                    ),
                }
            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 200, 1_000, step=100
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1.0, 20.0, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-3, 100.0, log=True
                ),
                "scale_pos_weight": trial.suggest_float(
                    "scale_pos_weight", 1.0, 100.0, log=True
                ),
            }
        if model_name == "isolation_forest":
            estimators = [30] if args.smoke else [100, 200, 400]
            samples = [256] if args.smoke else [256, 512, 1_024, 2_048]
            return {
                "n_estimators": trial.suggest_categorical(
                    "n_estimators", estimators
                ),
                "max_samples": trial.suggest_categorical("max_samples", samples),
                "max_features": trial.suggest_float(
                    "max_features", 0.5, 1.0
                ),
            }
        raise ValueError(f"Unsupported model: {model_name}")

    def build_model(params: dict[str, Any], random_state: int):
        if model_name == "logistic_regression":
            return build_logistic_regression(
                params,
                random_state=random_state,
            )
        if model_name == "xgboost":
            return build_xgboost_classifier(
                params,
                random_state=random_state,
                device=args.xgboost_device,
                n_jobs=args.n_jobs,
            )
        if model_name == "isolation_forest":
            return IsolationForestRiskModel(
                **params,
                n_jobs=args.n_jobs,
                random_state=random_state,
            )
        raise ValueError(f"Unsupported model: {model_name}")

    study = tune_model(
        model_factory=build_model,
        suggest_params=suggest_params,
        X=X,
        y=y,
        split=split,
        preprocessor_factory=new_preprocessor,
        n_trials=args.trials,
        random_state=args.seed,
        study_name=f"likelihood_{model_name}",
    )
    study.trials_dataframe().to_csv(run_dir / "optuna_trials.csv", index=False)
    print("Mean validation AP:", study.best_value)
    print("Best parameters:", study.best_params)

    development_rows = split.development_indices
    test_rows = split.holdout_indices
    preprocessor = new_preprocessor()
    X_development = preprocessor.fit_transform(
        X.iloc[development_rows], y.iloc[development_rows]
    )
    X_test = preprocessor.transform(X.iloc[test_rows])

    model = build_model(study.best_params, random_state=args.seed)
    model.fit(X_development, y.iloc[development_rows].to_numpy())
    metrics = generate_risk_report(
        model,
        X_test,
        y.iloc[test_rows].to_numpy(),
        output_dir=run_dir,
        row_ids=crossing_ids[test_rows],
    )

    joblib.dump(model, run_dir / "model.joblib")
    joblib.dump(preprocessor, run_dir / "preprocessor.joblib")
    score_semantics = {
        "logistic_regression": "uncalibrated estimated incident probability",
        "xgboost": "uncalibrated weighted-classifier ranking score",
        "isolation_forest": "training-distribution anomaly percentile",
    }[model_name]
    metadata = {
        "task": "incident_likelihood",
        "model": model_name,
        "dataset_path": str(dataset_path),
        "smoke_run": args.smoke,
        "holdout_year": split.holdout_year,
        "feature_count": len(feature_columns),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "excluded_features": EXCLUDED_RISK_FEATURES,
        "best_params": study.best_params,
        "mean_validation_average_precision": study.best_value,
        "validation_fold_scores": study.best_trial.user_attrs.get("fold_scores"),
        "test_metrics": metrics.to_dict(),
        "score_semantics": score_semantics,
        "n_trials": args.trials,
        "seed": args.seed,
        "python_version": platform.python_version(),
        "command": " ".join(sys.argv),
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2))

    print(pd.DataFrame([metrics.to_dict()]))
    print("Saved results to:", run_dir)
