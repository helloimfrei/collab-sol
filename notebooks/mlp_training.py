import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GPU MLP training with `collab-sol`

    This notebook tunes a PyTorch MLP on the 2019–2023 temporal folds and evaluates it once on the untouched 2024 test year. CUDA and mixed precision are used when available; otherwise the same code falls back to CPU.

    To try a different risk model, change **only the cell labeled CHANGE THE MODEL HERE**. The replacement model must provide `fit(X, y)` and `predict_proba(X)`.

    Run locally from `project_code/collab-sol`:

    ```bash
    uv sync --extra dev --extra gpu
    uv run --extra dev --extra gpu marimo edit notebooks/mlp_training.py
    ```

    On the Linux GPU machine, confirm that `torch.cuda.is_available()` prints `True`. If it does not, install the CUDA build selected for that machine at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
    """)
    return


@app.cell
def _():
    # 1. Resolve the local package, dataset, and output paths.
    from pathlib import Path
    import sys

    PROJECT_CODE_ROOT = Path(__file__).resolve().parents[2]
    PACKAGE_ROOT = PROJECT_CODE_ROOT / "collab-sol"
    DATASET_PATH = PROJECT_CODE_ROOT / "data" / "final" / "final_dataset.parquet"
    RUN_DIR = PACKAGE_ROOT / "runs" / "pytorch_mlp"

    if not PACKAGE_ROOT.is_dir():
        raise FileNotFoundError(f"collab-sol package not found: {PACKAGE_ROOT}")
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Final dataset not found: {DATASET_PATH}")

    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    return DATASET_PATH, PACKAGE_ROOT, RUN_DIR


@app.cell
def _(DATASET_PATH, PACKAGE_ROOT):
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
    from collab_sol.features import (
        EXCLUDED_RISK_FEATURES,
        split_risk_feature_columns,
    )

    ALL_COLUMNS = pq.read_schema(DATASET_PATH).names
    NUMERIC_FEATURES, CATEGORICAL_FEATURES = split_risk_feature_columns(
        ALL_COLUMNS
    )
    FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    assert len(FEATURE_COLUMNS) == 224
    assert len(CATEGORICAL_FEATURES) == 37

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
    return (
        CATEGORICAL_FEATURES,
        ColumnTransformer,
        EXCLUDED_RISK_FEATURES,
        FEATURE_COLUMNS,
        NUMERIC_FEATURES,
        OneHotEncoder,
        SimpleImputer,
        StandardScaler,
        frame,
        generate_risk_report,
        json,
        make_expanding_temporal_split,
        make_pipeline,
        np,
        pd,
        tune_model,
    )


@app.cell
def _(
    CATEGORICAL_FEATURES,
    ColumnTransformer,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
    frame,
    make_expanding_temporal_split,
    make_pipeline,
    np,
):
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
                        OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ],
            sparse_threshold=1.0,
        )

    print("Validation years:", [fold.validation_year for fold in split.folds])
    print("Untouched test year:", split.holdout_year)
    return X, crossing_ids, make_preprocessor, split, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CHANGE THE MODEL HERE

    `suggest_params` defines what Optuna searches. `build_model` turns those parameters into a model. Teammates can replace this cell without changing the splitting, tuning, or evaluation code below it.
    """)
    return


@app.cell
def _():
    import torch

    from collab_sol.models import TorchMLPClassifier

    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is unavailable; the notebook will use CPU.")

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
            max_epochs=50,
            patience=5,
            **params.copy(),
        )
    return build_model, suggest_params


@app.cell
def _(
    RUN_DIR,
    X,
    build_model,
    make_preprocessor,
    split,
    suggest_params,
    tune_model,
    y,
):
    # 4. Tune on 2020–2023 validation folds. The tuner never sees 2024.
    N_TRIALS = 2  # Keep this small for the demo; increase it for a real search.

    study = tune_model(
        model_factory=build_model,
        suggest_params=suggest_params,
        X=X,
        y=y,
        split=split,
        preprocessor_factory=make_preprocessor,
        n_trials=N_TRIALS,
        study_name="pytorch_mlp_risk",
    )

    study.trials_dataframe().to_csv(RUN_DIR / "optuna_trials.csv", index=False)
    print("Mean validation AP:", study.best_value)
    print("Best parameters:", study.best_params)
    return (study,)


@app.cell
def _(
    DATASET_PATH,
    EXCLUDED_RISK_FEATURES,
    FEATURE_COLUMNS,
    RUN_DIR,
    X,
    build_model,
    crossing_ids,
    generate_risk_report,
    json,
    make_preprocessor,
    pd,
    split,
    study,
    y,
):
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
        "excluded_features": EXCLUDED_RISK_FEATURES,
        "best_params": study.best_params,
        "mean_validation_average_precision": study.best_value,
        "device": str(model.device),
        "epochs_completed": model.n_iter_,
        "training_loss": model.loss_,
        "batch_size": model.batch_size,
        "stopping_tolerance": model.tolerance,
    }
    (RUN_DIR / "run.json").write_text(json.dumps(run_metadata, indent=2))

    print(pd.DataFrame([metrics.to_dict()]))
    print("Saved results to:", RUN_DIR)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
