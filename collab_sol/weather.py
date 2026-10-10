"""Convert daily GridMET files into leakage-safe crossing weather features."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

GRIDMET_VARIABLES = {
    "pr": "precipitation_amount",
    "tmmn": "air_temperature",
    "tmmx": "air_temperature",
    "vs": "wind_speed",
}

ANNUAL_METRICS = [
    "pr_total_mm",
    "wet_days",
    "freeze_thaw_days",
    "freezing_days",
    "hot_days",
    "tmin_mean_c",
    "tmax_mean_c",
    "wind_mean_ms",
    "high_wind_days",
]

GRID_COLUMNS = [
    "gridmet_covered",
    "grid_lat_index",
    "grid_lon_index",
    "grid_latitude",
    "grid_longitude",
]


def _require_columns(frame, columns):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _nearest_indices(grid, targets):
    """Return nearest indices for an ascending or descending one-dimensional grid."""

    grid = np.asarray(grid)
    if grid[0] > grid[-1]:
        return len(grid) - 1 - _nearest_indices(grid[::-1], targets)

    positions = np.searchsorted(grid, targets)
    positions = np.clip(positions, 1, len(grid) - 1)
    left = grid[positions - 1]
    right = grid[positions]
    return positions - (np.abs(targets - left) <= np.abs(targets - right))


def assign_grid_cells(frame, sample_file):
    """Add nearest GridMET cell identifiers without changing row count or order."""

    _require_columns(frame, ["latitude", "longitude"])
    existing = sorted(set(GRID_COLUMNS) & set(frame.columns))
    if existing:
        raise ValueError(f"GridMET columns already exist: {existing}")

    result = frame.copy()
    latitude = np.asarray(pd.to_numeric(result["latitude"], errors="coerce"), dtype=float)
    longitude = np.asarray(
        pd.to_numeric(result["longitude"], errors="coerce"), dtype=float
    )

    with xr.open_dataset(sample_file) as dataset:
        grid_latitudes = np.asarray(dataset["lat"].values)
        grid_longitudes = np.asarray(dataset["lon"].values)

    covered = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (latitude >= grid_latitudes.min())
        & (latitude <= grid_latitudes.max())
        & (longitude >= grid_longitudes.min())
        & (longitude <= grid_longitudes.max())
    )
    covered_positions = np.flatnonzero(covered)

    lat_indices = np.full(len(result), -1, dtype=np.int32)
    lon_indices = np.full(len(result), -1, dtype=np.int32)
    lat_indices[covered_positions] = _nearest_indices(
        grid_latitudes, latitude[covered_positions]
    )
    lon_indices[covered_positions] = _nearest_indices(
        grid_longitudes, longitude[covered_positions]
    )

    grid_latitude = np.full(len(result), np.nan)
    grid_longitude = np.full(len(result), np.nan)
    grid_latitude[covered_positions] = np.take(
        grid_latitudes, lat_indices[covered_positions]
    )
    grid_longitude[covered_positions] = np.take(
        grid_longitudes, lon_indices[covered_positions]
    )

    result["gridmet_covered"] = covered
    result["grid_lat_index"] = pd.array(
        [value if value >= 0 else None for value in lat_indices], dtype="Int32"
    )
    result["grid_lon_index"] = pd.array(
        [value if value >= 0 else None for value in lon_indices], dtype="Int32"
    )
    result["grid_latitude"] = grid_latitude
    result["grid_longitude"] = grid_longitude
    return result


def collect_grid_cells(dataset_paths, sample_file):
    """Collect unique coordinate pairs from Parquet datasets and map them to GridMET."""

    coordinates = pd.concat(
        [
            pd.read_parquet(path, columns=["latitude", "longitude"])
            for path in dataset_paths
        ],
        ignore_index=True,
    ).drop_duplicates(ignore_index=True)
    return assign_grid_cells(coordinates, sample_file)


def _point_chunks(path, variable, lat_indices, lon_indices, chunk_days):
    """Yield daily values at occupied cells without loading a full year into memory."""

    with xr.open_dataset(path) as dataset:
        values = dataset[variable].transpose("day", "lat", "lon")
        for start in range(0, values.sizes["day"], chunk_days):
            block = np.asarray(
                values.isel(day=slice(start, start + chunk_days)).values
            )
            points = block[:, lat_indices, lon_indices]
            del block
            yield points


def _safe_mean(totals, counts):
    return np.divide(
        totals,
        counts,
        out=np.full(totals.shape, np.nan, dtype=float),
        where=counts > 0,
    )


def _aggregate_gridmet_year(raw_dir, cells, year, chunk_days):
    lat_indices = cells["grid_lat_index"].to_numpy(dtype=int)
    lon_indices = cells["grid_lon_index"].to_numpy(dtype=int)
    size = len(cells)

    pr_total = np.zeros(size)
    pr_valid_days = np.zeros(size, dtype=int)
    wet_days = np.zeros(size, dtype=int)
    pr_path = raw_dir / f"pr_{year}.nc"
    for values in _point_chunks(
        pr_path, GRIDMET_VARIABLES["pr"], lat_indices, lon_indices, chunk_days
    ):
        valid = np.isfinite(values)
        pr_total += np.where(valid, values, 0).sum(axis=0)
        pr_valid_days += valid.sum(axis=0)
        wet_days += ((values >= 1.0) & valid).sum(axis=0)
    pr_total[pr_valid_days == 0] = np.nan

    tmin_total = np.zeros(size)
    tmax_total = np.zeros(size)
    tmin_valid_days = np.zeros(size, dtype=int)
    tmax_valid_days = np.zeros(size, dtype=int)
    paired_temperature_days = np.zeros(size, dtype=int)
    freeze_thaw_days = np.zeros(size, dtype=int)
    freezing_days = np.zeros(size, dtype=int)
    hot_days = np.zeros(size, dtype=int)
    tmin_path = raw_dir / f"tmmn_{year}.nc"
    tmax_path = raw_dir / f"tmmx_{year}.nc"
    tmin_chunks = _point_chunks(
        tmin_path, GRIDMET_VARIABLES["tmmn"], lat_indices, lon_indices, chunk_days
    )
    tmax_chunks = _point_chunks(
        tmax_path, GRIDMET_VARIABLES["tmmx"], lat_indices, lon_indices, chunk_days
    )
    for tmin, tmax in zip(tmin_chunks, tmax_chunks, strict=True):
        valid_min = np.isfinite(tmin)
        valid_max = np.isfinite(tmax)
        valid_pair = valid_min & valid_max
        tmin_total += np.where(valid_min, tmin, 0).sum(axis=0)
        tmax_total += np.where(valid_max, tmax, 0).sum(axis=0)
        tmin_valid_days += valid_min.sum(axis=0)
        tmax_valid_days += valid_max.sum(axis=0)
        paired_temperature_days += valid_pair.sum(axis=0)
        freeze_thaw_days += (
            valid_pair & (tmin < 273.15) & (tmax > 273.15)
        ).sum(axis=0)
        freezing_days += (valid_max & (tmax < 273.15)).sum(axis=0)
        hot_days += (valid_max & (tmax >= 305.15)).sum(axis=0)

    wind_total = np.zeros(size)
    wind_valid_days = np.zeros(size, dtype=int)
    high_wind_days = np.zeros(size, dtype=int)
    wind_path = raw_dir / f"vs_{year}.nc"
    for values in _point_chunks(
        wind_path, GRIDMET_VARIABLES["vs"], lat_indices, lon_indices, chunk_days
    ):
        valid = np.isfinite(values)
        wind_total += np.where(valid, values, 0).sum(axis=0)
        wind_valid_days += valid.sum(axis=0)
        high_wind_days += (valid & (values >= 10.0)).sum(axis=0)

    result = cells[["grid_lat_index", "grid_lon_index"]].copy()
    result["weather_year"] = year
    result["pr_total_mm"] = pr_total
    result["wet_days"] = np.where(pr_valid_days > 0, wet_days, np.nan)
    result["freeze_thaw_days"] = np.where(
        paired_temperature_days > 0, freeze_thaw_days, np.nan
    )
    result["freezing_days"] = np.where(
        tmax_valid_days > 0, freezing_days, np.nan
    )
    result["hot_days"] = np.where(tmax_valid_days > 0, hot_days, np.nan)
    result["tmin_mean_c"] = _safe_mean(tmin_total, tmin_valid_days) - 273.15
    result["tmax_mean_c"] = _safe_mean(tmax_total, tmax_valid_days) - 273.15
    result["wind_mean_ms"] = _safe_mean(wind_total, wind_valid_days)
    result["high_wind_days"] = np.where(
        wind_valid_days > 0, high_wind_days, np.nan
    )
    return result


def build_weather_lookup(
    raw_dir,
    cells,
    asof_years,
    output=None,
    *,
    history_years=5,
    chunk_days=61,
):
    """Build one leakage-safe weather row per GridMET cell and as-of year."""

    if history_years < 1:
        raise ValueError("history_years must be at least 1.")
    asof_years = sorted(set(int(year) for year in asof_years))
    if not asof_years:
        raise ValueError("At least one as-of year is required.")

    _require_columns(
        cells, ["gridmet_covered", "grid_lat_index", "grid_lon_index"]
    )
    occupied = (
        cells.loc[
            cells["gridmet_covered"], ["grid_lat_index", "grid_lon_index"]
        ]
        .drop_duplicates()
        .astype({"grid_lat_index": "int32", "grid_lon_index": "int32"})
        .reset_index(drop=True)
    )
    if occupied.empty:
        raise ValueError("No coordinates fall within the GridMET domain.")

    raw_dir = Path(raw_dir)
    weather_years = range(
        min(asof_years) - history_years,
        max(asof_years),
    )
    required_files = [
        raw_dir / f"{variable}_{year}.nc"
        for year in weather_years
        for variable in GRIDMET_VARIABLES
    ]
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing GridMET files: {missing_files}")

    annual = []
    for year in weather_years:
        annual.append(_aggregate_gridmet_year(raw_dir, occupied, year, chunk_days))
        print(f"aggregated GridMET {year}")
    annual = pd.concat(annual, ignore_index=True)

    summaries = []
    keys = ["grid_lat_index", "grid_lon_index"]
    for asof_year in asof_years:
        start_year = asof_year - history_years
        end_year = asof_year - 1
        window = annual[annual["weather_year"].between(start_year, end_year)]
        summary = pd.DataFrame(
            window.groupby(keys)[ANNUAL_METRICS].mean().reset_index()
        )
        renamed = {
            metric: f"{metric}_{history_years}y_mean"
            for metric in ANNUAL_METRICS
        }
        summary.columns = [renamed.get(column, column) for column in summary.columns]
        summary["asof_year"] = asof_year
        summary["weather_year_start"] = start_year
        summary["weather_year_end"] = end_year
        metric_columns = [
            f"{metric}_{history_years}y_mean" for metric in ANNUAL_METRICS
        ]
        summary["gridmet_data_available"] = summary[metric_columns].notna().all(axis=1)
        summaries.append(summary)

    lookup = pd.concat(summaries, ignore_index=True)
    lookup = lookup.sort_values(["asof_year", *keys]).reset_index(drop=True)
    if lookup.duplicated(["asof_year", *keys]).any():
        raise ValueError("Weather lookup keys are not unique.")

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        lookup.to_parquet(output, index=False)
    return lookup


def attach_weather(frame, sample_file, weather_lookup):
    """Attach a weather lookup without changing input row count or order."""

    _require_columns(frame, ["asof_year", "latitude", "longitude"])
    lookup = (
        pd.read_parquet(weather_lookup)
        if isinstance(weather_lookup, (str, Path))
        else weather_lookup.copy()
    )
    keys = ["asof_year", "grid_lat_index", "grid_lon_index"]
    _require_columns(lookup, [*keys, "gridmet_data_available"])
    if lookup.duplicated(keys).any():
        raise ValueError("Weather lookup keys are not unique.")

    assigned = assign_grid_cells(frame, sample_file)
    assigned["_weather_row_order"] = np.arange(len(assigned))
    result = assigned.merge(lookup, on=keys, how="left", sort=False, validate="many_to_one")
    result = result.sort_values("_weather_row_order").drop(columns="_weather_row_order")
    result["gridmet_data_available"] = result["gridmet_data_available"].eq(True)
    if len(result) != len(frame):
        raise RuntimeError("Weather merge changed the number of rows.")
    return result
