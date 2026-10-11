# collab-sol

Small shared utilities for comparing grade-crossing models. The package does
not own data preparation or model implementations. Teammates can write models
in notebooks and use the same temporal split, Optuna search, and risk report.

## Install in Colab

```python
%pip install -q "collab-sol @ git+https://github.com/helloimfrei/collab-sol.git@main"
```

## Model contract

No base class or package adapter is required. Provide two short functions:

```python
from sklearn.linear_model import LogisticRegression

def suggest_params(trial):
    return {"C": trial.suggest_float("C", 1e-3, 100, log=True)}

def build_model(params, random_state):
    return LogisticRegression(
        **params,
        class_weight="balanced",
        max_iter=300,
        random_state=random_state,
    )
```

`build_model` may return any object with ordinary `fit(X, y)` and
`predict_proba(X)` methods. The latter must return the usual two columns for a
binary classifier. If a model needs each fold's validation data for early
stopping or a custom neural training loop, pass an optional `fit_model`
function to the tuner.

## Typical use

```python
from collab_sol import make_expanding_temporal_split, tune_model

split = make_expanding_temporal_split(frame, year_column="asof_year")

study = tune_model(
    model_factory=build_model,
    suggest_params=suggest_params,
    X=frame[feature_columns],
    y=frame["incident"],
    split=split,
    preprocessor_factory=make_preprocessor,
    n_trials=20,
)
```

For the 2019–2024 panel, this tunes on expanding validation years 2020–2023
and never exposes the 2024 holdout to Optuna. Preprocessing is recreated and
fit inside each fold. After tuning, refit the selected model on 2019–2023 and
evaluate 2024 once.

`tune_model` returns a normal Optuna study. It is in memory unless `storage=`
is supplied; saving `study.trials_dataframe()` to CSV is the simplest Colab
provenance. The risk report writes average precision, prevalence, per-row
continuous risk scores, and a precision-recall curve.

## Files

- `splitting.py`: expanding temporal fold and untouched holdout indices.
- `tuning.py`: one shared Optuna loop for any compatible model.
- `evaluate.py`: continuous binary-risk metrics, reports, and optional SHAP.
- `notebooks/utils_demo.ipynb`: minimal Colab example using logistic
  regression; replace only its model factory and search-space functions.

The final dataset must already contain one row per crossing-year, features,
`asof_year`, and `incident`. Never include outcome columns such as
`incident_count`, `total_killed`, `total_injured`, or derived severity targets
as risk features.
