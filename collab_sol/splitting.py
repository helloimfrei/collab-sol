"""Leakage-safe expanding temporal splits for crossing-year cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class TemporalFold:
    train_years: tuple[int, ...]
    validation_year: int
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]


@dataclass(frozen=True)
class TemporalSplit:
    folds: tuple[TemporalFold, ...]
    development_indices: NDArray[np.int64]
    holdout_indices: NDArray[np.int64]
    holdout_year: int


def make_expanding_temporal_split(
    frame: pd.DataFrame,
    *,
    year_column: str = "feature_year",
    target_year_column: str | None = None,
    holdout_year: int | None = None,
) -> TemporalSplit:
    """Create expanding folds while reserving the latest cohort as holdout."""

    if year_column not in frame:
        raise ValueError(f"Missing year column: {year_column}")
    feature_years = cast(pd.Series, frame[year_column])
    if bool(feature_years.isna().any()):
        raise ValueError(f"{year_column} cannot contain missing values.")

    years = tuple(sorted(int(year) for year in feature_years.unique()))
    if len(years) < 3:
        raise ValueError("At least three feature years are required.")
    if any(later != earlier + 1 for earlier, later in zip(years, years[1:])):
        raise ValueError("Feature years must be consecutive.")

    resolved_holdout_year = holdout_year if holdout_year is not None else years[-1]
    if resolved_holdout_year != years[-1]:
        raise ValueError("The holdout must be the latest feature year.")

    if target_year_column is not None:
        if target_year_column not in frame:
            raise ValueError(f"Missing target year column: {target_year_column}")
        expected = feature_years.astype(int) + 1
        actual = cast(pd.Series, frame[target_year_column])
        if bool(actual.isna().any()) or not actual.astype(int).equals(expected):
            raise ValueError("Each target year must be exactly one year after its feature year.")

    year_values = feature_years.to_numpy()
    development_years = years[:-1]
    folds = tuple(
        TemporalFold(
            train_years=development_years[:validation_position],
            validation_year=validation_year,
            train_indices=np.flatnonzero(
                np.isin(year_values, development_years[:validation_position])
            ),
            validation_indices=np.flatnonzero(year_values == validation_year),
        )
        for validation_position, validation_year in enumerate(
            development_years[1:], start=1
        )
    )

    return TemporalSplit(
        folds=folds,
        development_indices=np.flatnonzero(year_values != resolved_holdout_year),
        holdout_indices=np.flatnonzero(year_values == resolved_holdout_year),
        holdout_year=resolved_holdout_year,
    )
