# collab-sol

Modeling code for grade-crossing risk and conditional severity. The package
accepts a completed crossing-year dataset as its input.

The planned model panel is logistic regression, XGBoost, LightGBM, an MLP, and
an FT-Transformer. Each model implements the `BaseModel` interface so the same
temporal cross-validation and Bayesian tuning pipeline can train and score it.

## Input contract

The modeling dataset should contain one row per crossing-year and already
include all model features and targets. At minimum, risk modeling requires:

- a stable crossing identifier such as `crossingid`;
- a cohort year such as `asof_year`;
- the binary target `incident`; and
- the final unencoded feature columns.

Conditional-severity modeling additionally requires a severity target on rows
where `incident == 1`. Encoding, scaling, and imputation must still be fit inside
each temporal training fold to avoid leakage.

`incident_count`, `total_killed`, `total_injured`, and derived severity targets
are outcomes, not risk features. In particular, `incident` is calculated from
`incident_count > 0`, so including `incident_count` would directly leak the
binary target.

Example setup:

```python
import pandas as pd

from collab_sol.splitting import make_expanding_temporal_split

frame = pd.read_parquet("data/processed/final_dataset.parquet")
split = make_expanding_temporal_split(frame, year_column="asof_year")

X = frame[feature_columns]
y = frame["incident"]
```

The split reserves the latest year as the untouched holdout and uses earlier
years for expanding temporal validation folds. Separate fold files are not
needed.

## Package map

- `splitting.py` creates expanding temporal fold indices and the final holdout.
- `tuning.py` runs one Optuna parameter trial across every temporal fold.
- `evaluate.py` writes continuous holdout risk scores, ranking and probability
  metrics, and a precision-recall curve. SHAP helpers support fitted tree and
  PyTorch neural models as an optional extra.
- `models/base.py` defines the interface required by the tuner.
- `models/xgboost.py` implements the current XGBoost model adapter.
- `preprocessing.py` owns fold-fitted model-input transformations.
- `train.py` will own final refitting and holdout evaluation.

See `notebooks/modeling_demo.ipynb` for a small end-to-end Colab example using
the final dataset, temporal tuning, final refitting, and holdout reporting.
