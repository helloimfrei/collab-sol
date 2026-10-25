"""Tune and evaluate the CUDA-capable grade-crossing risk MLP."""

import argparse
from pathlib import Path
import sys

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
    default=PACKAGE_ROOT / "runs" / "pytorch_mlp",
)
parser.add_argument("--trials", type=int, default=2)
parser.add_argument("--max-epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--require-cuda", action="store_true")
args = parser.parse_args()

DATASET_PATH = args.dataset.resolve()
RUN_DIR = args.output_dir.resolve()

if not PACKAGE_ROOT.is_dir():
    raise FileNotFoundError(f"collab-sol package not found: {PACKAGE_ROOT}")
if not DATASET_PATH.is_file():
    raise FileNotFoundError(f"Final dataset not found: {DATASET_PATH}")

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
RUN_DIR.mkdir(parents=True, exist_ok=True)

# %%
# 2. Load the already-filtered modeling dataset.
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from collab_sol import generate_risk_report, make_expanding_temporal_split, tune_model

ID_COLUMNS = ["crossingid", "asof_year"]
TARGET_COLUMNS = ["incident", "severity_target"]
ALL_COLUMNS = pq.read_schema(DATASET_PATH).names
FEATURE_COLUMNS = [
    column for column in ALL_COLUMNS
    if column not in ID_COLUMNS + TARGET_COLUMNS
]

# The final dataset stores its 38 categorical predictors after the numeric predictors.
categorical_start = FEATURE_COLUMNS.index("reportingagencytypecode")
NUMERIC_FEATURES = FEATURE_COLUMNS[:categorical_start]
CATEGORICAL_FEATURES = FEATURE_COLUMNS[categorical_start:]

assert len(FEATURE_COLUMNS) == 229
assert len(CATEGORICAL_FEATURES) == 38

frame = pd.read_parquet(
    DATASET_PATH,
    columns=["crossingid", "asof_year", "incident", *FEATURE_COLUMNS],
)
frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(np.float32)
for column in CATEGORICAL_FEATURES:
    frame[column] = frame[column].astype("category")

yearly_summary = frame.groupby("asof_year")["incident"].agg(
    rows="size", crashes="sum", prevalence="mean"
)
print(yearly_summary)
print("Package:", PACKAGE_ROOT)
print("Dataset:", DATASET_PATH)
print(f"Using {len(FEATURE_COLUMNS)} predictors: "
      f"{len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical")

# %%
# 3. Create the temporal folds and fold-safe preprocessing.
X = frame[FEATURE_COLUMNS]
y = frame["incident"].astype(int)
split = make_expanding_temporal_split(frame, year_column="asof_year")
crossing_ids = frame["crossingid"].to_numpy()
del frame

def make_preprocessor():
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(
                        handle_unknown="ignore",
                        dtype=np.float32,  # pyright: ignore[reportArgumentType]
                    ),
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=1.0,
    )

print("Validation years:", [fold.validation_year for fold in split.folds])
print("Untouched test year:", split.holdout_year)

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class TorchMLPClassifier:
    """Small sklearn-shaped PyTorch classifier for the shared tuner."""

    def __init__(
        self,
        *,
        hidden_width,
        n_layers,
        dropout,
        learning_rate,
        weight_decay,
        batch_size,
        max_epochs=50,
        patience=5,
        tolerance=5e-4,
        random_state=42,
    ):
        self.hidden_width = hidden_width
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.tolerance = tolerance
        self.random_state = random_state
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.network_ = None

    @staticmethod
    def _dense_float32(values):
        if hasattr(values, "toarray"):
            values = values.toarray()
        return np.asarray(values, dtype=np.float32, order="C")

    def _build_network(self, input_width):
        layers = []
        layer_width = input_width
        for _ in range(self.n_layers):
            layers.extend(
                [
                    nn.Linear(layer_width, self.hidden_width),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ]
            )
            layer_width = self.hidden_width
        layers.append(nn.Linear(layer_width, 1))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        features = torch.from_numpy(self._dense_float32(X))
        targets = torch.from_numpy(
            np.asarray(y, dtype=np.float32).reshape(-1, 1)
        )
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(features, targets),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=self.device.type == "cuda",
        )

        network = self._build_network(features.shape[1])
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_function = nn.BCEWithLogitsLoss()
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_loss = float("inf")
        best_state = None
        stale_epochs = 0

        for epoch in range(self.max_epochs):
            network.train()
            epoch_loss = 0.0
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(
                    self.device, non_blocking=use_amp
                )
                batch_targets = batch_targets.to(
                    self.device, non_blocking=use_amp
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    logits = network(batch_features)
                    loss = loss_function(logits, batch_targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.detach().item() * len(batch_features)

            epoch_loss /= len(features)
            self.n_iter_ = epoch + 1
            if best_loss - epoch_loss > self.tolerance:
                best_loss = epoch_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is None:
            raise RuntimeError("The MLP did not complete a training epoch.")
        network.load_state_dict(best_state)
        self.network_ = network
        self.loss_ = best_loss
        print(
            f"PyTorch MLP | device={self.device} | "
            f"epochs={self.n_iter_} | loss={self.loss_:.6f}"
        )
        return self

    def predict_proba(self, X):
        if self.network_ is None:
            raise RuntimeError("Call fit before predict_proba.")

        features = torch.from_numpy(self._dense_float32(X))
        loader = DataLoader(
            TensorDataset(features),
            batch_size=self.batch_size * 2,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )
        probabilities = []
        self.network_.eval()
        with torch.no_grad():
            for (batch_features,) in loader:
                batch_features = batch_features.to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    logits = self.network_(batch_features)
                probabilities.append(
                    torch.sigmoid(logits.float()).cpu().numpy().reshape(-1)
                )

        positive = np.concatenate(probabilities)
        return np.column_stack([1.0 - positive, positive])

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
        batch_size=16384,
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
    study_name="pytorch_mlp_risk",
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

model = build_model(study.best_params, random_state=42)
model.fit(X_development, y.iloc[development_rows].to_numpy())

metrics = generate_risk_report(
    model,
    X_test,
    y.iloc[test_rows].to_numpy(),
    output_dir=RUN_DIR,
    row_ids=crossing_ids[test_rows],
)

run_metadata = {
    "model": type(model).__name__,
    "dataset_path": str(DATASET_PATH),
    "holdout_year": split.holdout_year,
    "feature_count": len(FEATURE_COLUMNS),
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
}
(RUN_DIR / "run.json").write_text(json.dumps(run_metadata, indent=2))

print(pd.DataFrame([metrics.to_dict()]))
print("Saved results to:", RUN_DIR)
