from __future__ import annotations

import pandas as pd
import pytest

from collab_sol.splitting import make_expanding_temporal_split


def test_expanding_split_keeps_latest_year_as_holdout() -> None:
    frame = pd.DataFrame(
        {
            "feature_year": [year for year in range(2018, 2024) for _ in range(2)],
            "target_year": [year for year in range(2019, 2025) for _ in range(2)],
        }
    )

    split = make_expanding_temporal_split(frame, target_year_column="target_year")

    assert split.holdout_year == 2023
    assert len(split.folds) == 4
    assert [fold.validation_year for fold in split.folds] == [2019, 2020, 2021, 2022]
    assert split.folds[0].train_years == (2018,)
    assert split.folds[-1].train_years == (2018, 2019, 2020, 2021)
    assert set(frame.iloc[split.holdout_indices]["feature_year"]) == {2023}
    assert set(frame.iloc[split.development_indices]["feature_year"]) == {
        2018,
        2019,
        2020,
        2021,
        2022,
    }


def test_expanding_split_rejects_misaligned_target_years() -> None:
    frame = pd.DataFrame(
        {
            "feature_year": [2018, 2019, 2020],
            "target_year": [2019, 2021, 2021],
        }
    )

    with pytest.raises(ValueError, match="exactly one year after"):
        make_expanding_temporal_split(frame, target_year_column="target_year")
