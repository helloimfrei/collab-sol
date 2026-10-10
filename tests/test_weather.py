from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from collab_sol.weather import add_weather, build_weather_lookup, map_gridmet_cells


def _write_gridmet_files(root: Path, year: int) -> None:
    coordinates = {
        "day": pd.date_range(f"{year}-01-01", periods=2),
        "lat": np.array([40.0, 39.95]),
        "lon": np.array([-100.0, -99.95]),
    }
    shape = (2, 2, 2)
    value = year - 2013
    variables = {
        "pr": ("precipitation_amount", np.full(shape, value, dtype=float)),
        "tmmn": ("air_temperature", np.full(shape, 272.0)),
        "tmmx": ("air_temperature", np.full(shape, 274.0)),
        "vs": ("wind_speed", np.full(shape, value, dtype=float)),
    }
    variables["pr"][1][:, 0, 0] = np.nan
    for short_name, (variable, values) in variables.items():
        dataset = xr.Dataset(
            {variable: (("day", "lat", "lon"), values)},
            coords=coordinates,
        )
        dataset.to_netcdf(root / f"{short_name}_{year}.nc")


def _crossings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crossingid": ["coastal", "normal", "far", "missing"],
            "asof_year": [2019, 2019, 2019, 2019],
            "latitude": [40.0, 39.95, 41.0, np.nan],
            "longitude": [-100.0, -99.95, -99.95, np.nan],
        }
    )


def test_cell_mapping_uses_nearest_data_cell_and_flags_distance(tmp_path: Path):
    _write_gridmet_files(tmp_path, 2014)

    mapped = map_gridmet_cells(_crossings(), tmp_path / "pr_2014.nc")

    assert mapped["gridmet_mapped"].tolist() == [True, True, False, False]
    assert mapped.loc[0, "gridmet_distance_km"] < 10
    assert mapped.loc[0, "grid_lon_index"] == 1
    assert mapped.loc[2, "gridmet_distance_km"] > 10
    assert pd.isna(mapped.loc[2, "grid_lat_index"])


def test_lookup_uses_five_years_before_asof_year(tmp_path: Path):
    for year in range(2014, 2019):
        _write_gridmet_files(tmp_path, year)

    lookup = build_weather_lookup(_crossings(), tmp_path, asof_years=[2019])

    assert len(lookup) == 2
    assert set(lookup["weather_year_start"]) == {2014}
    assert set(lookup["weather_year_end"]) == {2018}
    np.testing.assert_allclose(lookup["pr_total_mm_5y_mean"], 6.0)
    np.testing.assert_allclose(lookup["wind_mean_ms_5y_mean"], 3.0)
    np.testing.assert_allclose(lookup["freeze_thaw_days_5y_mean"], 2.0)


def test_add_weather_preserves_rows_and_marks_unmapped_crossings(tmp_path: Path):
    for year in range(2014, 2019):
        _write_gridmet_files(tmp_path, year)
    crossings = _crossings()
    lookup = build_weather_lookup(crossings, tmp_path, asof_years=[2019])

    result = add_weather(crossings, tmp_path, lookup)

    assert result["crossingid"].tolist() == crossings["crossingid"].tolist()
    assert len(result) == len(crossings)
    assert result["gridmet_data_available"].tolist() == [True, True, False, False]
    assert pd.isna(result.loc[2, "pr_total_mm_5y_mean"])
