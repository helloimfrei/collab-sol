# collab-sol

Modeling code for next-year grade-crossing risk and conditional severity.

The `main` branch owns temporal splitting, shared model interfaces, training,
Bayesian hyperparameter tuning, and evaluation. Reproducible source-data
construction is retained separately on the `dataset-pipeline` branch.

The planned model panel is logistic regression, XGBoost, LightGBM, an MLP, and
an FT-Transformer. Each model implements the `BaseModel` interface so the same
temporal cross-validation and Bayesian tuning pipeline can train and score it.

## Final dataset

Build the public, open, at-grade, lower-48 crossing-year master dataset from the
six teammate-produced annual files:

```bash
uv run python -m collab_sol.data --data-root ../data
```

The builder keeps the latest Form 71 revision per crossing, derives risk and
severity directly from Form 57, attaches five-year GridMET features, and writes
`data/processed/crossing_year_master.parquet`. Values remain unencoded so each
model's preprocessing can be fit inside its temporal training folds.

The assembly order is:

1. Normalize the six annual schemas where source types drift.
2. Keep the latest pre-year Form 71 revision for each crossing.
3. Keep public, open, at-grade crossings in the lower 48 states.
4. Aggregate Form 57 incidents by crossing and calendar year.
5. Label incidents as property-damage-only, injury, or fatal.
6. Add the prior five years of GridMET weather and its mapping-quality flags.
7. Stream every annual cohort into one compressed Parquet file.

## Package map

- `data.py` assembles the unencoded master dataset and its targets.
- `weather.py` builds and joins five-year GridMET summaries.
- `splitting.py` creates expanding temporal fold indices and the final holdout.
- `tuning.py` runs one Optuna parameter trial across every temporal fold.
- `evaluate.py` contains shared evaluation metrics.
- `models/base.py` defines the interface required by the tuner.
- `models/xgboost.py` implements the current XGBoost model adapter.
- `preprocessing.py` and `train.py` are placeholders for the model-input and
  final-refit pipelines; they do not yet contain implementations.

## GridMET weather processing

Build one five-year weather lookup for all crossing snapshots, then merge it
into the master crossing-year dataset:

```python
from pathlib import Path

import pandas as pd

from collab_sol.weather import add_weather, build_weather_lookup

data_root = Path("../data")
raw_gridmet = data_root / "raw" / "gridmet"
source_files = sorted((data_root / "full_dataset").glob("full_dataset_*.parquet"))
coordinates = pd.concat(
    [
        pd.read_parquet(path, columns=["latitude", "longitude"])
        for path in source_files
    ],
    ignore_index=True,
)

lookup = build_weather_lookup(
    coordinates,
    raw_gridmet,
    asof_years=range(2019, 2025),
    output=data_root / "interim" / "weather" / "gridmet_5y_by_cell_asof.parquet",
)
master = pd.read_parquet(data_root / "processed" / "crossing_year_master.parquet")
master = add_weather(master, raw_gridmet, lookup)
```

The nearest GridMET cell must contain data and be within 10 km. The merged data
includes `gridmet_distance_km` and `gridmet_mapped` so rejected matches are
visible. Each five-year window ends at `asof_year - 1`, preventing incident-year
weather from leaking into model features.

Keep all years in one logical master dataset. It can be one Parquet file or a
Parquet dataset partitioned by year. Temporal cross-validation should create
train and validation row indices in memory rather than writing separate fold
files; load only the chosen model features, target, and `asof_year` when memory
is limited.
