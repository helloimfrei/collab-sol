# collab-sol

Modeling code for next-year grade-crossing risk and conditional severity.

The `main` branch owns temporal splitting, shared model interfaces, training,
Bayesian hyperparameter tuning, and evaluation. Reproducible source-data
construction is retained separately on the `dataset-pipeline` branch.

The planned model panel is logistic regression, XGBoost, LightGBM, an MLP, and
an FT-Transformer. Each model implements the `BaseModel` interface so the same
temporal cross-validation and Bayesian tuning pipeline can train and score it.

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
