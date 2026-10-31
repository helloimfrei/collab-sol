# collab-sol

Shared utilities for making our grade-crossing model comparisons consistent.
Team models can use the utilities directly; the package provides:

- expanding temporal cross-validation with a final untouched year;
- Optuna hyperparameter tuning; and
- continuous risk-score evaluation and reporting.

It also contains the shared CUDA MLP and FT-Transformer used by the local
training scripts.

Model code lives under `collab_sol/models/`:

- `classical.py` — logistic-regression and XGBoost factories;
- `anomaly.py` — the Isolation Forest ranking adapter; and
- `neural.py` — the PyTorch MLP and FT-Transformer implementations.

The model package keeps neural imports lazy, so a base utilities installation
does not require PyTorch.

## Install in Colab

```python
%pip install -q "collab-sol @ git+https://github.com/helloimfrei/collab-sol.git@main"
```

## Use it with any model

No inheritance or package-specific model class is required. Define one
function containing the parameters Optuna should search and another that
builds the model:

```python
from sklearn.linear_model import LogisticRegression

def suggest_params(trial):
    return {
        "C": trial.suggest_float("C", 1e-3, 100, log=True),
    }

def build_model(params, random_state):
    return LogisticRegression(
        **params,
        class_weight="balanced",  # Fixed, not tuned.
        max_iter=300,
        random_state=random_state,
    )
```

The returned model only needs normal `fit(X, y)` and `predict_proba(X)`
methods. For binary risk, `predict_proba` must return the usual two columns:
the probabilities of classes 0 and 1.

## Split and tune

```python
from sklearn.impute import SimpleImputer
from collab_sol import make_expanding_temporal_split, tune_model

def make_preprocessor():
    return SimpleImputer(strategy="median", add_indicator=True)

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

print(study.best_value)
print(study.best_params)
```

By default, every trial is scored using average precision. The tuner fits the
same proposed parameter set on every temporal fold and maximizes its mean
validation score. A different higher-is-better function can be passed through
`scorer=`.

For our 2019–2024 dataset, the validation years are 2020–2023. The tuner never
receives 2024. After selecting the parameters, refit on all 2019–2023 rows and
evaluate 2024 once. Preprocessing is independently fit inside every fold.

Models that require validation data while fitting, such as XGBoost early
stopping or a custom neural-network loop, can supply the optional `fit_model`
callback. See the tuner docstring for its arguments.

## Outputs and provenance

`tune_model` returns an ordinary Optuna study. It is in memory unless
`storage=` is provided. In Colab, the simplest record of the search is:

```python
study.trials_dataframe().to_csv("optuna_trials.csv", index=False)
```

`generate_risk_report` saves average precision, prevalence, a
precision-recall curve, one continuous risk score per crossing, and an
`inspection_budget.csv` table for the top 10, 50, 100, 500, 1,000, 5,000,
and 10,000 ranked crossings. The budget table reports incident crossings
captured, precision, recall, lift over prevalence, and the score cutoff. The
report does not convert scores into yes/no predictions.

## Data contract

The input must already contain one row per crossing-year, `asof_year`, the
binary `incident` target, and the chosen features. Never use outcomes such as
`incident_count`, `total_killed`, `total_injured`, or derived severity targets
as risk features.

All likelihood-model comparisons should use
`split_risk_feature_columns(...)`. It keeps 224 predictors and consistently
excludes coordinates (`latitude`, `longitude`), coordinate-provenance metadata
(`latlongsourcecode`), and two ACS population totals (`B01001_E001`,
`B03002_E001`) that duplicate the retained `B01003_E001`.

See [`notebooks/utils_demo.ipynb`](notebooks/utils_demo.ipynb) for the complete
Colab workflow using logistic regression. To test another model, replace only
`suggest_params` and `build_model`.

## Train the crash-likelihood models locally

These scripts are specifically for the binary `incident` likelihood task. The
future severity classifiers will use separate `train_severity_*` entry points.
Every likelihood script uses the same 224 predictors, expanding temporal folds,
mean validation AP objective, and untouched 2024 evaluation.

Individual entry points:

```text
scripts/train_likelihood_logistic_regression.py
scripts/train_likelihood_xgboost.py
scripts/train_likelihood_isolation_forest.py
scripts/train_likelihood_mlp.py
scripts/train_likelihood_ft_transformer.py
```

Install every model dependency and smoke-test the complete panel:

```bash
uv sync --extra models
uv run python scripts/train_likelihood_models.py --smoke
```

On the Linux GPU machine, run the complete comparison with:

```bash
uv run python scripts/train_likelihood_models.py \
  --require-cuda \
  --xgboost-device cuda \
  --trials 20
```

The master script runs models sequentially so each process releases its memory
before the next begins. Results are grouped under one timestamped
`runs/likelihood_panel_*` directory, with a `panel_run.json` manifest and one folder
per model. `model_comparison.csv` collects validation AP, test AP, and test
prevalence for the entire panel. Use `--models xgboost mlp` to run only selected
models.

### Logistic-regression details

The logistic baseline uses `SGDClassifier(loss="log_loss")` to fit the same
linear probabilistic model without SAGA's prohibitive runtime on nearly one
million rows. Optuna tunes regularization strength, L1/L2 penalty, and optional
class balancing. Fold-fitted one-hot preprocessing is cached across trials, and
the runner prints fold AP and elapsed time while it works.

XGBoost uses fold validation for early stopping and refits the selected
configuration for the median best tree count. Isolation Forest builds its
anomaly-percentile reference from at most 50,000 deterministic training rows;
this preserves score ordering while avoiding an unnecessary pass over every
training row. Both models cache fold preprocessing across trials and report
fold AP and elapsed time.

### MLP details

[`notebooks/mlp_training.py`](notebooks/mlp_training.py) is a Marimo notebook
for tuning the PyTorch MLP over the temporal folds and evaluating 2024 once.
It expects the final Parquet at `../data/final/final_dataset.parquet`.
The shared estimator is available as
`from collab_sol.models import TorchMLPClassifier`.

```bash
uv sync --extra dev --extra models
uv run marimo edit notebooks/mlp_training.py
```

The notebook uses CUDA mixed precision when available and otherwise falls back
to CPU. Confirm `torch.cuda.is_available()` is `True` before starting a GPU run.

For a non-interactive run without Marimo:

```bash
uv sync --extra models
uv run python scripts/train_likelihood_mlp.py --require-cuda
```

The script accepts `--trials`, `--max-epochs`, `--patience`, `--dataset`, and
`--output-dir`. It defaults to 20 trials, 50 maximum epochs, and 5 patience
epochs.

### FT-Transformer details

The FT-Transformer was adapted from Lucas Campa's
[`crossings`](https://github.com/lucascampa/crossings) implementation at commit
`ae8df19`. It uses category embeddings, fold-fitted quantile preprocessing for
numeric values, validation-AP early stopping, and the same temporal tuner and
final report as the MLP.

```bash
uv sync --extra models

# Quick end-to-end check
uv run python scripts/train_likelihood_ft_transformer.py \
  --smoke --require-cuda

# Full search on the default final dataset
uv run python scripts/train_likelihood_ft_transformer.py \
  --require-cuda --trials 20
```

The transformer first screens 20 parameter sets on a fixed 25% sample of
crossing IDs stratified by their 2019--2023 presence and incident patterns,
then re-evaluates the best three with full-data temporal CV. A saved audit
compares full and sampled prevalence by development year. The 2024 holdout
remains untouched until the selected model's final evaluation.
Defaults are 1--2 transformer blocks, widths of 64 or 128, 12 maximum epochs,
3 patience epochs, and a batch size of 128 for an 8 GB GPU. Reduce the batch
size to 64 if needed. Fold preprocessing is cached across trials, and any
sampled configuration that exceeds CUDA memory is pruned without terminating
the study. Each run receives its own timestamped folder under `runs/` and
records both tuning stages and the exact command.
