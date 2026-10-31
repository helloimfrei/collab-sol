"""Tune and evaluate Lucas Campa's FT-Transformer crash-likelihood model.

The script uses the same expanding temporal folds, average-precision objective,
2024 holdout, feature set, and report format as the other risk models.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import cast

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CODE_ROOT = PACKAGE_ROOT.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dataset",
    type=Path,
    default=PROJECT_CODE_ROOT / "data" / "final" / "final_dataset.parquet",
)
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--max-epochs", type=int, default=12)
parser.add_argument("--patience", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--screening-fraction", type=float, default=0.25)
parser.add_argument("--finalists", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--require-cuda", action="store_true")
parser.add_argument(
    "--smoke",
    action="store_true",
    help="use 2,000 crossings, one trial, two epochs, and a small model",
)
args = parser.parse_args()

if args.smoke:
    args.trials = 1
    args.max_epochs = 2
    args.patience = 2
    args.screening_fraction = 1.0
    args.finalists = 1

if not 0.0 < args.screening_fraction <= 1.0:
    raise ValueError("--screening-fraction must be greater than 0 and at most 1.")
if not 1 <= args.finalists <= args.trials:
    raise ValueError("--finalists must be between 1 and --trials.")

DATASET_PATH = args.dataset.resolve()
if args.output_dir is None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "_smoke" if args.smoke else ""
    args.output_dir = (
        PACKAGE_ROOT / "runs" / f"likelihood_ft_transformer_{timestamp}{suffix}"
    )
RUN_DIR = args.output_dir.resolve()

if not DATASET_PATH.is_file():
    raise FileNotFoundError(f"Final dataset not found: {DATASET_PATH}")
if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
    raise FileExistsError(f"Output directory is not empty: {RUN_DIR}")

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
RUN_DIR.mkdir(parents=True, exist_ok=True)

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from optuna.pruners import NopPruner
from optuna.trial import TrialState

from collab_sol import (
    generate_risk_report,
    make_expanding_temporal_split,
    stratified_crossing_sample,
    tune_model,
)
from collab_sol.features import (
    EXCLUDED_RISK_FEATURES,
    split_risk_feature_columns,
)
from collab_sol.models import FTTransformerClassifier
from collab_sol.preprocessing import make_ft_transformer_preprocessor

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
elif args.require_cuda:
    raise RuntimeError("CUDA is required but torch.cuda.is_available() is False.")
else:
    print("CUDA is unavailable; running on CPU.")

NUMERIC_FEATURES, CATEGORICAL_FEATURES = split_risk_feature_columns(
    pq.read_schema(DATASET_PATH).names
)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

frame = cast(
    pd.DataFrame,
    pd.read_parquet(
        DATASET_PATH,
        columns=["crossingid", "asof_year", "incident", *FEATURE_COLUMNS],
    ),
)
if args.smoke:
    crossing_ids = frame["crossingid"].drop_duplicates()
    sampled_ids = crossing_ids.sample(
        n=min(2_000, len(crossing_ids)),
        random_state=args.seed,
    )
    frame = cast(
        pd.DataFrame,
        frame[frame["crossingid"].isin(sampled_ids)].reset_index(drop=True),
    )

frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(np.float32)
for column in CATEGORICAL_FEATURES:
    frame[column] = frame[column].astype("category")

yearly_summary = frame.groupby("asof_year")["incident"].agg(
    rows="size",
    incidents="sum",
    prevalence="mean",
)
yearly_summary.to_csv(RUN_DIR / "class_balance_by_year.csv")
print(yearly_summary)
print("Dataset:", DATASET_PATH)
print(
    f"Using {len(FEATURE_COLUMNS)} predictors: "
    f"{len(NUMERIC_FEATURES)} numeric + "
    f"{len(CATEGORICAL_FEATURES)} categorical"
)

X = cast(pd.DataFrame, frame[FEATURE_COLUMNS])
y = cast(pd.Series, frame["incident"].astype(int))
split = make_expanding_temporal_split(frame, year_column="asof_year")
crossing_ids = np.asarray(frame["crossingid"])

development_frame = cast(
    pd.DataFrame,
    frame.iloc[split.development_indices],
)
sampled_crossings = stratified_crossing_sample(
    development_frame,
    sample_fraction=args.screening_fraction,
    random_state=args.seed,
)
screening_rows = np.flatnonzero(
    frame["crossingid"].isin(sampled_crossings.tolist())
)
screening_frame = cast(
    pd.DataFrame,
    frame.iloc[screening_rows].reset_index(drop=True),
)
screening_X = cast(pd.DataFrame, screening_frame[FEATURE_COLUMNS])
screening_y = cast(pd.Series, screening_frame["incident"].astype(int))
screening_split = make_expanding_temporal_split(
    screening_frame,
    year_column="asof_year",
)
full_development_summary = development_frame.groupby("asof_year")[
    "incident"
].agg(full_rows="size", full_incidents="sum", full_prevalence="mean")
sampled_development_summary = screening_frame.iloc[
    screening_split.development_indices
].groupby("asof_year")["incident"].agg(
    sampled_rows="size",
    sampled_incidents="sum",
    sampled_prevalence="mean",
)
screening_prevalence = full_development_summary.join(
    sampled_development_summary,
)
screening_prevalence["prevalence_difference"] = (
    screening_prevalence["sampled_prevalence"]
    - screening_prevalence["full_prevalence"]
)
screening_prevalence.to_csv(RUN_DIR / "screening_prevalence_by_year.csv")
print(
    "Screening sample:",
    f"{len(sampled_crossings):,} crossings / {len(screening_frame):,} rows",
    f"({args.screening_fraction:.0%} of development crossings)",
)
print(screening_prevalence)
del frame


def new_preprocessor():
    return make_ft_transformer_preprocessor(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        random_state=args.seed,
    )


def suggest_params(trial):
    if args.smoke:
        n_blocks = trial.suggest_categorical("n_blocks", [1])
        d_block = trial.suggest_categorical("d_block", [64])
    else:
        n_blocks = trial.suggest_int("n_blocks", 1, 2)
        d_block = trial.suggest_categorical("d_block", [64, 128])
    return {
        "n_blocks": n_blocks,
        "d_block": d_block,
        "attention_dropout": trial.suggest_float(
            "attention_dropout", 0.0, 0.5
        ),
        "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.5),
        "residual_dropout": trial.suggest_float(
            "residual_dropout", 0.0, 0.2
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True
        ),
        "pos_weight": trial.suggest_float(
            "pos_weight", 1.0, 50.0, log=True
        ),
    }


def build_model(params, random_state):
    model = FTTransformerClassifier(
        n_cat_features=len(CATEGORICAL_FEATURES),
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        random_state=random_state,
        **params.copy(),
    )
    model.tuned_params_ = dict(params)
    return model


def make_fit_model(stage_name, parameter_set_count, fold_count):
    best_epochs_by_params: dict[str, list[int]] = {}
    parameter_set_numbers: dict[str, int] = {}

    def fit_model(model, X_train, y_train, X_valid, y_valid):
        key = json.dumps(model.tuned_params_, sort_keys=True)
        if key not in parameter_set_numbers:
            parameter_set_numbers[key] = len(parameter_set_numbers) + 1
        fold_number = len(best_epochs_by_params.get(key, [])) + 1
        model.log_prefix = (
            f"{stage_name} {parameter_set_numbers[key]}/{parameter_set_count} | "
            f"fold {fold_number}/{fold_count}"
        )
        model.fit_with_validation(X_train, y_train, X_valid, y_valid)
        best_epochs_by_params.setdefault(key, []).append(model.best_epoch_)

    return fit_model, best_epochs_by_params


screening_fit_model, _ = make_fit_model(
    "screening trial",
    args.trials,
    len(screening_split.folds),
)
screening_study = tune_model(
    model_factory=build_model,
    suggest_params=suggest_params,
    X=screening_X,
    y=screening_y,
    split=screening_split,
    preprocessor_factory=new_preprocessor,
    fit_model=screening_fit_model,
    n_trials=args.trials,
    random_state=args.seed,
    study_name="incident_likelihood_ft_transformer_screening",
)
screening_study.trials_dataframe().to_csv(
    RUN_DIR / "optuna_screening_trials.csv",
    index=False,
)

completed_screening_trials = sorted(
    (
        trial
        for trial in screening_study.trials
        if trial.state == TrialState.COMPLETE and trial.value is not None
    ),
    key=lambda trial: (
        trial.value if trial.value is not None else float("-inf")
    ),
    reverse=True,
)
if len(completed_screening_trials) < args.finalists:
    raise RuntimeError(
        "Fewer screening trials completed than requested finalists."
    )
finalist_params = [
    trial.params for trial in completed_screening_trials[: args.finalists]
]
print(f"Re-evaluating {len(finalist_params)} finalists on full temporal CV.")

full_fit_model, full_best_epochs = make_fit_model(
    "full-data finalist",
    len(finalist_params),
    len(split.folds),
)
study = tune_model(
    model_factory=build_model,
    suggest_params=suggest_params,
    X=X,
    y=y,
    split=split,
    preprocessor_factory=new_preprocessor,
    fit_model=full_fit_model,
    n_trials=len(finalist_params),
    random_state=args.seed,
    study_name="incident_likelihood_ft_transformer_finalists",
    enqueued_trials=finalist_params,
    pruner=NopPruner(),
)
study.trials_dataframe().to_csv(RUN_DIR / "optuna_trials.csv", index=False)
print("Full-data mean validation AP:", study.best_value)
print("Selected parameters:", study.best_params)

best_key = json.dumps(study.best_params, sort_keys=True)
best_epochs = full_best_epochs[best_key]
final_epochs = max(1, int(round(float(np.median(best_epochs)))))

development_rows = split.development_indices
test_rows = split.holdout_indices
preprocessor = new_preprocessor()
X_development = preprocessor.fit_transform(
    X.iloc[development_rows],
    y.iloc[development_rows],
)
X_test = preprocessor.transform(X.iloc[test_rows])

model = build_model(study.best_params, random_state=args.seed)
model.fixed_epochs = final_epochs
model.log_prefix = "final refit"
model.fit(X_development, y.iloc[development_rows].to_numpy())

metrics = generate_risk_report(
    model,
    X_test,
    y.iloc[test_rows].to_numpy(),
    output_dir=RUN_DIR,
    row_ids=crossing_ids[test_rows],
)

pd.DataFrame(model.history_).to_csv(
    RUN_DIR / "final_training_history.csv", index=False
)
joblib.dump(preprocessor, RUN_DIR / "preprocessor.joblib")
if model.network_ is None:
    raise RuntimeError("The final FT-Transformer was not fitted.")
torch.save(model.network_.state_dict(), RUN_DIR / "model.pt")

run_metadata = {
    "task": "incident_likelihood",
    "model": type(model).__name__,
    "adapted_from": (
        "https://github.com/lucascampa/crossings/commit/"
        "ae8df19dd924800efb4dc209983325302a53c436"
    ),
    "dataset_path": str(DATASET_PATH),
    "smoke_run": args.smoke,
    "holdout_year": split.holdout_year,
    "feature_count": len(FEATURE_COLUMNS),
    "numeric_feature_count": len(NUMERIC_FEATURES),
    "categorical_feature_count": len(CATEGORICAL_FEATURES),
    "excluded_features": EXCLUDED_RISK_FEATURES,
    "best_params": study.best_params,
    "mean_validation_average_precision": study.best_value,
    "validation_fold_scores": study.best_trial.user_attrs.get("fold_scores"),
    "best_epoch_per_fold": best_epochs,
    "final_refit_epochs": final_epochs,
    "test_metrics": metrics.to_dict(),
    "device": str(model.device),
    "n_trials": args.trials,
    "screening_fraction": args.screening_fraction,
    "screening_sampling": "crossing_presence_incident_pattern_stratified",
    "screening_rows": len(screening_X),
    "screening_crossings": len(sampled_crossings),
    "screening_best_average_precision": screening_study.best_value,
    "full_data_finalists": len(finalist_params),
    "max_epochs": args.max_epochs,
    "patience": args.patience,
    "batch_size": args.batch_size,
    "seed": args.seed,
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "command": " ".join(sys.argv),
}
(RUN_DIR / "run.json").write_text(json.dumps(run_metadata, indent=2))

print(pd.DataFrame([metrics.to_dict()]))
print("Saved results to:", RUN_DIR)
