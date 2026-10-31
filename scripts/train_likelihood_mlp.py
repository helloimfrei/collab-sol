"""Tune and evaluate the CUDA-capable MLP crash-likelihood model."""

import argparse
from datetime import datetime, timezone
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
parser.add_argument(
    "--output-dir",
    type=Path,
)
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--max-epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=16_384)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--require-cuda", action="store_true")
parser.add_argument(
    "--smoke",
    action="store_true",
    help="use 2,000 crossings, one trial, and two epochs",
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
    args.output_dir = PACKAGE_ROOT / "runs" / f"likelihood_mlp_{timestamp}{suffix}"
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

# %%
# 2. Load the already-filtered modeling dataset.
import json

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from collab_sol import generate_risk_report, make_expanding_temporal_split, tune_model
from collab_sol.features import (
    EXCLUDED_RISK_FEATURES,
    split_risk_feature_columns,
)
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
        columns=["crossingid", "asof_year", "incident", *FEATURE_COLUMNS],
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
frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(np.float32)
for column in CATEGORICAL_FEATURES:
    frame[column] = frame[column].astype("category")

yearly_summary = frame.groupby("asof_year")["incident"].agg(
    rows="size", crashes="sum", prevalence="mean"
)
yearly_summary.to_csv(RUN_DIR / "class_balance_by_year.csv")
print(yearly_summary)
print("Package:", PACKAGE_ROOT)
print("Dataset:", DATASET_PATH)
print(f"Using {len(FEATURE_COLUMNS)} predictors: "
      f"{len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical")

# %%
# 3. Create the temporal folds and fold-safe preprocessing.
X = cast(pd.DataFrame, frame[FEATURE_COLUMNS])
y = cast(pd.Series, frame["incident"].astype(int))
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

# %%
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
        "hidden_width": trial.suggest_categorical("hidden_width", [128, 256, 512]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-4, 3e-3, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-7, 1e-3, log=True
        ),
    }

def build_model(params, random_state):
    return TorchMLPClassifier(
        random_state=random_state,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        **params.copy(),
    )

# 4. Tune on 2020–2023 validation folds. The tuner never sees 2024.
study = tune_model(
    model_factory=build_model,
    suggest_params=suggest_params,
    X=X,
    y=y,
    split=split,
    preprocessor_factory=make_preprocessor,
    n_trials=args.trials,
    random_state=args.seed,
    study_name="incident_likelihood_mlp",
)

study.trials_dataframe().to_csv(RUN_DIR / "optuna_trials.csv", index=False)
print("Mean validation AP:", study.best_value)
print("Best parameters:", study.best_params)

# %%
# 5. Refit the winning model on 2019–2023 and evaluate 2024 once.
development_rows = split.development_indices
test_rows = split.holdout_indices

preprocessor = make_preprocessor()
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
    output_dir=RUN_DIR,
    row_ids=crossing_ids[test_rows],
)

joblib.dump(preprocessor, RUN_DIR / "preprocessor.joblib")
if model.network_ is None:
    raise RuntimeError("The final MLP was not fitted.")
torch.save(model.network_.state_dict(), RUN_DIR / "model_state.pt")

run_metadata = {
    "task": "incident_likelihood",
    "model": type(model).__name__,
    "dataset_path": str(DATASET_PATH),
    "holdout_year": split.holdout_year,
    "feature_count": len(FEATURE_COLUMNS),
    "excluded_features": EXCLUDED_RISK_FEATURES,
    "smoke_run": args.smoke,
    "best_params": study.best_params,
    "mean_validation_average_precision": study.best_value,
    "device": str(model.device),
    "epochs_completed": model.n_iter_,
    "training_loss": model.loss_,
    "n_trials": args.trials,
    "max_epochs": args.max_epochs,
    "patience": args.patience,
    "batch_size": model.batch_size,
    "stopping_tolerance": model.tolerance,
    "test_metrics": metrics.to_dict(),
    "seed": args.seed,
    "command": " ".join(sys.argv),
}
(RUN_DIR / "run.json").write_text(json.dumps(run_metadata, indent=2))

print(pd.DataFrame([metrics.to_dict()]))
print("Saved results to:", RUN_DIR)
