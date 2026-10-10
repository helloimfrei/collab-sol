from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from collab_sol.weather import (
    assign_grid_cells,
    attach_weather,
    build_weather_lookup,
)


def _write_gridmet_files(root: Path, year: int):
    days = pd.date_range(f"{year}-01-01", periods=2)
    coordinates = {
        "day": days,
        "lat": np.array([40.0, 39.0]),
        "lon": np.array([-100.0, -99.0]),
    }
    shape = (len(days), 2, 2)
    value = year - 2013
    variables = {
        "pr": ("precipitation_amount", np.full(shape, value, dtype=float)),
        "tmmn": ("air_temperature", np.full(shape, 272.0)),
        "tmmx": ("air_temperature", np.full(shape, 274.0)),
        "vs": ("wind_speed", np.full(shape, value, dtype=float)),
    }
    for short_name, (variable, values) in variables.items():
        dataset = xr.Dataset(
            {variable: (("day", "lat", "lon"), values)},
            coords=coordinates,
        )
        dataset.to_netcdf(root / f"{short_name}_{year}.nc")


def test_weather_lookup_uses_only_years_before_asof_year(tmp_path: Path):
    for year in range(2014, 2019):
        _write_gridmet_files(tmp_path, year)

    coordinates = pd.DataFrame(
        {
            "latitude": [39.9, 39.1, 50.0, np.nan],
            "longitude": [-99.9, -99.1, -99.5, -99.5],
        }
    )
    cells = assign_grid_cells(coordinates, tmp_path / "pr_2014.nc")
    lookup = build_weather_lookup(
        tmp_path,
        cells,
        asof_years=[2019],
        chunk_days=1,
    )

    assert cells["gridmet_covered"].tolist() == [True, True, False, False]
    assert len(lookup) == 2
    assert set(lookup["weather_year_start"]) == {2014}
    assert set(lookup["weather_year_end"]) == {2018}
    np.testing.assert_allclose(lookup["pr_total_mm_5y_mean"], 6.0)
    np.testing.assert_allclose(lookup["wind_mean_ms_5y_mean"], 3.0)
    np.testing.assert_allclose(lookup["freeze_thaw_days_5y_mean"], 2.0)


def test_weather_merge_preserves_rows_and_marks_uncovered_locations(tmp_path: Path):
    for year in range(2014, 2019):
        _write_gridmet_files(tmp_path, year)

    frame = pd.DataFrame(
        {
            "crossingid": ["A", "B", "C"],
            "asof_year": [2019, 2019, 2019],
            "latitude": [39.9, 39.9, np.nan],
            "longitude": [-99.9, -99.9, np.nan],
        }
    )
    cells = assign_grid_cells(frame, tmp_path / "pr_2014.nc")
    lookup = build_weather_lookup(tmp_path, cells, asof_years=[2019])

    result = attach_weather(frame, tmp_path / "pr_2014.nc", lookup)

    assert result["crossingid"].tolist() == ["A", "B", "C"]
    assert len(result) == len(frame)
    assert result["gridmet_data_available"].tolist() == [True, True, False]
    assert pd.isna(result.loc[2, "pr_total_mm_5y_mean"])
