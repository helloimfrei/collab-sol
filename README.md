# collab-sol

Modeling code for next-year grade-crossing risk and conditional severity.

The `main` branch owns temporal splitting, shared model interfaces, training,
Bayesian hyperparameter tuning, and evaluation. Reproducible source-data
construction is retained separately on the `dataset-pipeline` branch.

The planned model panel is logistic regression, XGBoost, LightGBM, an MLP, and
an FT-Transformer. Each model implements the `BaseModel` interface so the same
temporal cross-validation and Bayesian tuning pipeline can train and score it.

## GridMET weather processing

Build a leakage-safe lookup from the downloaded daily GridMET files:

```python
from pathlib import Path

from collab_sol.weather import collect_grid_cells, build_weather_lookup

data_root = Path("../data")
datasets = sorted((data_root / "full_dataset").glob("full_dataset_*.parquet"))
raw_gridmet = data_root / "raw" / "gridmet"

cells = collect_grid_cells(datasets, raw_gridmet / "pr_2014.nc")
lookup = build_weather_lookup(
    raw_gridmet,
    cells,
    asof_years=range(2019, 2025),
    output=data_root / "interim" / "weather" / "gridmet_5y_by_cell_asof.parquet",
)
```

Each lookup row is keyed by `asof_year`, `grid_lat_index`, and
`grid_lon_index`. Its five-year window ends at `asof_year - 1`, so incident-year
weather cannot leak into model features. Use `attach_weather` for a validated,
row-preserving merge into a yearly dataset.
