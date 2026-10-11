# collab-sol

Modeling code for grade-crossing risk and conditional severity. The package
starts from a completed crossing-year dataset; raw-source acquisition, cleaning,
geospatial joins, weather aggregation, and target construction happen outside
the package.

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

Example setup:

```python
import pandas as pd

from collab_sol.splitting import make_expanding_temporal_split

frame = pd.read_parquet("data/processed/crossing_year_master.parquet")
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
- `evaluate.py` contains shared evaluation metrics.
- `models/base.py` defines the interface required by the tuner.
- `models/xgboost.py` implements the current XGBoost model adapter.
- `preprocessing.py` owns fold-fitted model-input transformations.
- `train.py` will own final refitting and holdout evaluation.

Dataset-construction work remains available on the local `final-dataset` and
`dataset-pipeline` branches, but it is intentionally outside this package.
