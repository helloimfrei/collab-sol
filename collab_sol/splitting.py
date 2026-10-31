"""Leakage-safe expanding temporal splits for crossing-year cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


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


def stratified_crossing_sample(
    frame: pd.DataFrame,
    *,
    sample_fraction: float,
    random_state: int,
    crossing_column: str = "crossingid",
    year_column: str = "asof_year",
    target_column: str = "incident",
) -> pd.Index:
    """Sample crossing IDs while preserving yearly incident patterns."""

    required = {crossing_column, year_column, target_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing sampling columns: {sorted(missing)}")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be greater than 0 and at most 1.")
    target = cast(pd.Series, frame[target_column])
    if bool(target.isna().any()):
        raise ValueError(f"{target_column} cannot contain missing values.")

    years = sorted(int(year) for year in frame[year_column].unique())
    presence = frame.assign(_present=1).pivot_table(
        index=crossing_column,
        columns=year_column,
        values="_present",
        aggfunc="max",
        fill_value=0,
    )
    presence = presence.reindex(columns=years, fill_value=0)
    incidents = frame.pivot_table(
        index=crossing_column,
        columns=year_column,
        values=target_column,
        aggfunc="max",
        fill_value=0,
    )
    incidents = incidents.reindex(
        index=presence.index,
        columns=years,
        fill_value=0,
    )

    if sample_fraction == 1.0:
        return presence.index

    weights = np.left_shift(1, np.arange(len(years), dtype=np.int64))
    presence_codes = presence.to_numpy(dtype=np.int64) @ weights
    incident_codes = (incidents.to_numpy(dtype=np.int64) > 0) @ weights
    strata = pd.Series(
        [
            f"{presence_code}:{incident_code}"
            for presence_code, incident_code in zip(
                presence_codes,
                incident_codes,
            )
        ],
        index=presence.index,
    )

    minimum_size = max(2, int(np.ceil(1.0 / sample_fraction)))
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= minimum_size, "rare")
    stratum_counts = strata.value_counts()
    if int(stratum_counts.min()) < 2:
        largest_stratum = str(stratum_counts.index[0])
        singleton_strata = [
            label for label, count in stratum_counts.items() if int(count) < 2
        ]
        strata = strata.where(~strata.isin(singleton_strata), largest_stratum)

    _, sampled_ids = train_test_split(
        presence.index.to_numpy(),
        test_size=sample_fraction,
        random_state=random_state,
        stratify=strata.to_numpy(),
    )
    return pd.Index(sampled_ids)


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
