"""Tune and evaluate the three-class conditional-severity MLP."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import cast

PROJECT_CODE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_CODE_ROOT / "collab-sol"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dataset",
    type=Path,
    default=PROJECT_CODE_ROOT / "data" / "final" / "final_dataset.parquet",
)
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--max-epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--require-cuda", action="store_true")
parser.add_argument(
    "--smoke",
    action="store_true",
    help="run one trial with two epochs over the complete severity cohort",
)
args = parser.parse_args()

if args.smoke:
    args.trials = 1
    args.max_epochs = 2
    args.patience = 2

DATASET_PATH = args.dataset.resolve()
if args.output_dir is None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "_smoke" if args.smoke else ""
    args.output_dir = PACKAGE_ROOT / "runs" / f"severity_mlp_{timestamp}{suffix}"
RUN_DIR = args.output_dir.resolve()

if not PACKAGE_ROOT.is_dir():
    raise FileNotFoundError(f"collab-sol package not found: {PACKAGE_ROOT}")
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

from collab_sol import (
    SEVERITY_CLASSES,
    SEVERITY_CODES,
    SEVERITY_WEIGHTS,
    generate_severity_report,
    make_expanding_temporal_split,
    multiclass_log_loss,
    tune_model,
)
from collab_sol.features import EXCLUDED_RISK_FEATURES, split_risk_feature_columns
from collab_sol.preprocessing import make_one_hot_preprocessor

ALL_COLUMNS = pq.read_schema(DATASET_PATH).names
NUMERIC_FEATURES, CATEGORICAL_FEATURES = split_risk_feature_columns(ALL_COLUMNS)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

assert len(FEATURE_COLUMNS) == 224
assert len(CATEGORICAL_FEATURES) == 37

frame = cast(
    pd.DataFrame,
    pd.read_parquet(
        DATASET_PATH,
        columns=[
            "crossingid",
            "asof_year",
            "incident",
            "severity_target",
            *FEATURE_COLUMNS,
        ],
        filters=[("incident", "=", 1)],
    ),
)
invalid_targets = ~frame["severity_target"].isin(SEVERITY_CODES)
if bool(invalid_targets.any()):
    invalid = sorted(frame.loc[invalid_targets, "severity_target"].unique())
    raise ValueError(f"Unexpected severity targets: {invalid}")

frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(np.float32)
for column in CATEGORICAL_FEATURES:
    frame[column] = frame[column].astype("category")
severity_codes = (
    frame["severity_target"]
    .astype(str)
    .map(lambda class_name: SEVERITY_CODES[class_name])
    .astype(int)
)

yearly_summary = cast(
    pd.DataFrame,
    frame.groupby(["asof_year", "severity_target"], observed=True)
    .size()
    .unstack(fill_value=0),
)
yearly_summary = yearly_summary.reindex(
    columns=list(SEVERITY_CLASSES),
    fill_value=0,
)
yearly_summary.to_csv(RUN_DIR / "class_balance_by_year.csv")
print(yearly_summary)
print("Package:", PACKAGE_ROOT)
print("Dataset:", DATASET_PATH)
print(
    f"Using {len(FEATURE_COLUMNS)} predictors: "
    f"{len(NUMERIC_FEATURES)} numeric + "
    f"{len(CATEGORICAL_FEATURES)} categorical"
)

X = cast(pd.DataFrame, frame[FEATURE_COLUMNS])
y = cast(pd.Series, severity_codes)
split = make_expanding_temporal_split(frame, year_column="asof_year")
crossing_ids = np.asarray(frame["crossingid"])
del frame


def make_preprocessor():
    return make_one_hot_preprocessor(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        scale_numeric=True,
    )


print("Validation years:", [fold.validation_year for fold in split.folds])
print("Untouched test year:", split.holdout_year)

import torch

from collab_sol.models import TorchMLPClassifier

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
else:
    if args.require_cuda:
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is False.")
    print("CUDA is unavailable; the script will use CPU.")


def suggest_params(trial):
    return {
        "hidden_width": trial.suggest_categorical(
            "hidden_width", [64, 128, 256, 512]
        ),
        "n_layers": trial.suggest_int("n_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float(
            "learning_rate", 5e-5, 5e-3, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-8, 1e-2, log=True
        ),
    }


def build_model(params, random_state):
    return TorchMLPClassifier(
        n_classes=len(SEVERITY_CLASSES),
        random_state=random_state,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        **params.copy(),
    )


study = tune_model(
    model_factory=build_model,
    suggest_params=suggest_params,
    X=X,
    y=y,
    split=split,
    preprocessor_factory=make_preprocessor,
    scorer=multiclass_log_loss,
    direction="minimize",
    metric_name="log loss",
    n_trials=args.trials,
    random_state=args.seed,
    study_name="conditional_severity_mlp",
    cache_preprocessing=True,
    verbose=True,
)

study.trials_dataframe().to_csv(RUN_DIR / "optuna_trials.csv", index=False)
print("Mean validation log loss:", study.best_value)
print("Best parameters:", study.best_params)

development_rows = split.development_indices
test_rows = split.holdout_indices
preprocessor = make_preprocessor()
X_development = preprocessor.fit_transform(
    X.iloc[development_rows],
    y.iloc[development_rows],
)
X_test = preprocessor.transform(X.iloc[test_rows])

model = build_model(study.best_params, random_state=args.seed)
model.fit(X_development, y.iloc[development_rows].to_numpy())
metrics = generate_severity_report(
    model,
    X_test,
    y.iloc[test_rows].to_numpy(),
    output_dir=RUN_DIR,
    row_ids=crossing_ids[test_rows],
    class_names=SEVERITY_CLASSES,
    class_weights=SEVERITY_WEIGHTS,
)

joblib.dump(preprocessor, RUN_DIR / "preprocessor.joblib")
if model.network_ is None:
    raise RuntimeError("The final severity MLP was not fitted.")
torch.save(model.network_.state_dict(), RUN_DIR / "model_state.pt")

run_metadata = {
    "task": "conditional_severity",
    "model": type(model).__name__,
    "dataset_path": str(DATASET_PATH),
    "holdout_year": split.holdout_year,
    "feature_count": len(FEATURE_COLUMNS),
    "excluded_features": EXCLUDED_RISK_FEATURES,
    "severity_classes": SEVERITY_CLASSES,
    "severity_weights": SEVERITY_WEIGHTS,
    "severity_score_definition": (
        "P(property_damage_only) + 2 * P(injury) + 3 * P(fatal)"
    ),
    "smoke_run": args.smoke,
    "best_params": study.best_params,
    "mean_validation_multiclass_log_loss": study.best_value,
    "validation_fold_scores": study.best_trial.user_attrs.get("fold_scores"),
    "device": str(model.device),
    "epochs_completed": model.n_iter_,
    "training_loss": model.loss_,
    "n_trials": args.trials,
    "max_epochs": args.max_epochs,
    "patience": args.patience,
    "batch_size": model.batch_size,
    "test_metrics": metrics.to_dict(),
    "seed": args.seed,
    "command": " ".join(sys.argv),
}
(RUN_DIR / "run.json").write_text(json.dumps(run_metadata, indent=2))

print(pd.DataFrame([metrics.to_dict()]))
print("Saved results to:", RUN_DIR)
