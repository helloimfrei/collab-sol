"""Build five-year GridMET features for grade crossings."""

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import BallTree

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
    "gridmet_mapped",
    "gridmet_distance_km",
    "grid_lat_index",
    "grid_lon_index",
    "grid_latitude",
    "grid_longitude",
]

MAX_GRIDMET_DISTANCE_KM = 10.0
EARTH_RADIUS_KM = 6371.0088
HISTORY_YEARS = 5
_CHUNK_DAYS = 61


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _sample_file(raw_dir: Path) -> Path:
    files = sorted(raw_dir.glob("pr_*.nc"))
    if not files:
        raise FileNotFoundError(f"No pr_*.nc files found in {raw_dir}")
    return files[0]


def map_gridmet_cells(
    crossings: pd.DataFrame,
    sample_file: str | Path,
    max_distance_km: float = MAX_GRIDMET_DISTANCE_KM,
) -> pd.DataFrame:
    """Map each crossing to the nearest GridMET data cell within a distance limit."""

    _require_columns(crossings, ["latitude", "longitude"])
    existing = sorted(set(GRID_COLUMNS) & set(crossings.columns))
    if existing:
        raise ValueError(f"GridMET columns already exist: {existing}")

    with xr.open_dataset(sample_file) as dataset:
        latitudes = np.asarray(dataset["lat"].values)
        longitudes = np.asarray(dataset["lon"].values)
        precipitation = np.asarray(
            dataset[GRIDMET_VARIABLES["pr"]].isel(day=0).values
        )

    valid_lat_index, valid_lon_index = np.nonzero(np.isfinite(precipitation))
    valid_coordinates = np.column_stack(
        [latitudes[valid_lat_index], longitudes[valid_lon_index]]
    )
    tree = BallTree(np.radians(valid_coordinates), metric="haversine")

    result = crossings.copy()
    crossing_latitude = np.asarray(
        pd.to_numeric(result["latitude"], errors="coerce"), dtype=float
    )
    crossing_longitude = np.asarray(
        pd.to_numeric(result["longitude"], errors="coerce"), dtype=float
    )
    valid_crossing = (
        np.isfinite(crossing_latitude)
        & np.isfinite(crossing_longitude)
        & (crossing_latitude >= -90)
        & (crossing_latitude <= 90)
        & (crossing_longitude >= -180)
        & (crossing_longitude <= 180)
    )

    distance_km = np.full(len(result), np.nan)
    matched_lat_index = np.full(len(result), -1, dtype=np.int32)
    matched_lon_index = np.full(len(result), -1, dtype=np.int32)
    positions = np.flatnonzero(valid_crossing)
    if len(positions):
        distances, matches = tree.query(
            np.radians(
                np.column_stack(
                    [crossing_latitude[positions], crossing_longitude[positions]]
                )
            ),
            k=1,
        )
        distance_km[positions] = distances[:, 0] * EARTH_RADIUS_KM
        matched_lat_index[positions] = valid_lat_index[matches[:, 0]]
        matched_lon_index[positions] = valid_lon_index[matches[:, 0]]

    mapped = valid_crossing & (distance_km <= max_distance_km)
    matched_lat_index[~mapped] = -1
    matched_lon_index[~mapped] = -1
    mapped_positions = np.flatnonzero(mapped)

    grid_latitude = np.full(len(result), np.nan)
    grid_longitude = np.full(len(result), np.nan)
    grid_latitude[mapped_positions] = latitudes[matched_lat_index[mapped_positions]]
    grid_longitude[mapped_positions] = longitudes[matched_lon_index[mapped_positions]]

    result["gridmet_mapped"] = mapped
    result["gridmet_distance_km"] = distance_km
    result["grid_lat_index"] = pd.array(
        [value if value >= 0 else None for value in matched_lat_index], dtype="Int32"
    )
    result["grid_lon_index"] = pd.array(
        [value if value >= 0 else None for value in matched_lon_index], dtype="Int32"
    )
    result["grid_latitude"] = grid_latitude
    result["grid_longitude"] = grid_longitude
    return result


def _point_chunks(
    path: Path,
    variable: str,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
) -> Iterator[np.ndarray]:
    with xr.open_dataset(path) as dataset:
        values = dataset[variable].transpose("day", "lat", "lon")
        for start in range(0, values.sizes["day"], _CHUNK_DAYS):
            block = np.asarray(values.isel(day=slice(start, start + _CHUNK_DAYS)).values)
            yield block[:, lat_indices, lon_indices]


def _safe_mean(totals: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.divide(
        totals,
        counts,
        out=np.full(totals.shape, np.nan, dtype=float),
        where=counts > 0,
    )


def _aggregate_year(raw_dir: Path, cells: pd.DataFrame, year: int) -> pd.DataFrame:
    lat_indices = cells["grid_lat_index"].to_numpy(dtype=int)
    lon_indices = cells["grid_lon_index"].to_numpy(dtype=int)
    size = len(cells)

    pr_total = np.zeros(size)
    pr_valid_days = np.zeros(size, dtype=int)
    wet_days = np.zeros(size, dtype=int)
    for values in _point_chunks(
        raw_dir / f"pr_{year}.nc",
        GRIDMET_VARIABLES["pr"],
        lat_indices,
        lon_indices,
    ):
        valid = np.isfinite(values)
        pr_total += np.where(valid, values, 0).sum(axis=0)
        pr_valid_days += valid.sum(axis=0)
        wet_days += (valid & (values >= 1.0)).sum(axis=0)
    pr_total[pr_valid_days == 0] = np.nan

    tmin_total = np.zeros(size)
    tmax_total = np.zeros(size)
    tmin_valid_days = np.zeros(size, dtype=int)
    tmax_valid_days = np.zeros(size, dtype=int)
    paired_temperature_days = np.zeros(size, dtype=int)
    freeze_thaw_days = np.zeros(size, dtype=int)
    freezing_days = np.zeros(size, dtype=int)
    hot_days = np.zeros(size, dtype=int)
    tmin_chunks = _point_chunks(
        raw_dir / f"tmmn_{year}.nc",
        GRIDMET_VARIABLES["tmmn"],
        lat_indices,
        lon_indices,
    )
    tmax_chunks = _point_chunks(
        raw_dir / f"tmmx_{year}.nc",
        GRIDMET_VARIABLES["tmmx"],
        lat_indices,
        lon_indices,
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
        freeze_thaw_days += (valid_pair & (tmin < 273.15) & (tmax > 273.15)).sum(
            axis=0
        )
        freezing_days += (valid_max & (tmax < 273.15)).sum(axis=0)
        hot_days += (valid_max & (tmax >= 305.15)).sum(axis=0)

    wind_total = np.zeros(size)
    wind_valid_days = np.zeros(size, dtype=int)
    high_wind_days = np.zeros(size, dtype=int)
    for values in _point_chunks(
        raw_dir / f"vs_{year}.nc",
        GRIDMET_VARIABLES["vs"],
        lat_indices,
        lon_indices,
    ):
        valid = np.isfinite(values)
        wind_total += np.where(valid, values, 0).sum(axis=0)
        wind_valid_days += valid.sum(axis=0)
        high_wind_days += (valid & (values >= 10.0)).sum(axis=0)

    result = pd.DataFrame(cells[["grid_lat_index", "grid_lon_index"]].copy())
    result["weather_year"] = year
    result["pr_total_mm"] = pr_total
    result["wet_days"] = np.where(pr_valid_days > 0, wet_days, np.nan)
    result["freeze_thaw_days"] = np.where(
        paired_temperature_days > 0, freeze_thaw_days, np.nan
    )
    result["freezing_days"] = np.where(tmax_valid_days > 0, freezing_days, np.nan)
    result["hot_days"] = np.where(tmax_valid_days > 0, hot_days, np.nan)
    result["tmin_mean_c"] = _safe_mean(tmin_total, tmin_valid_days) - 273.15
    result["tmax_mean_c"] = _safe_mean(tmax_total, tmax_valid_days) - 273.15
    result["wind_mean_ms"] = _safe_mean(wind_total, wind_valid_days)
    result["high_wind_days"] = np.where(
        wind_valid_days > 0, high_wind_days, np.nan
    )
    return result


def build_weather_lookup(
    crossings: pd.DataFrame,
    raw_dir: str | Path,
    asof_years: Iterable[int],
    output: str | Path | None = None,
) -> pd.DataFrame:
    """Build five-year weather summaries for the cells used by the crossings."""

    raw_dir = Path(raw_dir)
    years = sorted(set(int(year) for year in asof_years))
    if not years:
        raise ValueError("At least one as-of year is required.")

    coordinates = pd.DataFrame(
        crossings[["latitude", "longitude"]].drop_duplicates()
    )
    mapped = map_gridmet_cells(coordinates, _sample_file(raw_dir))
    cells = (
        mapped.loc[mapped["gridmet_mapped"], ["grid_lat_index", "grid_lon_index"]]
        .drop_duplicates()
        .astype({"grid_lat_index": "int32", "grid_lon_index": "int32"})
        .reset_index(drop=True)
    )
    if cells.empty:
        raise ValueError("No crossing is within 10 km of a GridMET data cell.")

    weather_years = range(min(years) - HISTORY_YEARS, max(years))
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
        annual.append(_aggregate_year(raw_dir, cells, year))
        print(f"aggregated GridMET {year}")
    annual_frame = pd.concat(annual, ignore_index=True)

    keys = ["grid_lat_index", "grid_lon_index"]
    summaries = []
    for asof_year in years:
        start_year = asof_year - HISTORY_YEARS
        window = annual_frame[
            annual_frame["weather_year"].between(start_year, asof_year - 1)
        ]
        summary = window.groupby(keys)[ANNUAL_METRICS].mean().reset_index()
        summary = summary.rename(
            columns={metric: f"{metric}_5y_mean" for metric in ANNUAL_METRICS}
        )
        summary["asof_year"] = asof_year
        summary["weather_year_start"] = start_year
        summary["weather_year_end"] = asof_year - 1
        weather_columns = [f"{metric}_5y_mean" for metric in ANNUAL_METRICS]
        summary["gridmet_data_available"] = summary[weather_columns].notna().all(axis=1)
        summaries.append(summary)

    lookup = pd.concat(summaries, ignore_index=True)
    lookup = lookup.sort_values(["asof_year", *keys]).reset_index(drop=True)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        lookup.to_parquet(output, index=False)
    return lookup


def add_weather(
    crossings: pd.DataFrame,
    raw_dir: str | Path,
    weather_lookup: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """Map crossings to GridMET and add their five-year weather features."""

    _require_columns(crossings, ["asof_year", "latitude", "longitude"])
    lookup = (
        pd.read_parquet(weather_lookup)
        if isinstance(weather_lookup, (str, Path))
        else weather_lookup.copy()
    )
    keys = ["asof_year", "grid_lat_index", "grid_lon_index"]
    if lookup.duplicated(keys).any():
        raise ValueError("Weather lookup keys are not unique.")

    mapped = map_gridmet_cells(crossings, _sample_file(Path(raw_dir)))
    mapped["_weather_row_order"] = np.arange(len(mapped))
    result = mapped.merge(lookup, on=keys, how="left", sort=False, validate="many_to_one")
    result = result.sort_values("_weather_row_order").drop(columns="_weather_row_order")
    result["gridmet_data_available"] = result["gridmet_data_available"].eq(True)
    return result
