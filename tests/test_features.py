from __future__ import annotations

import numpy as np
import pandas as pd

from collab_sol.features import split_risk_feature_columns
from collab_sol.preprocessing import (
    make_ft_transformer_preprocessor,
    make_one_hot_preprocessor,
)


def test_shared_risk_features_drop_ids_targets_and_exclusions() -> None:
    numeric, categorical = split_risk_feature_columns(
        [
            "crossingid",
            "asof_year",
            "latitude",
            "B01003_E001",
            "B01001_E001",
            "reportingagencytypecode",
            "latlongsourcecode",
            "crossingclosed",
            "incident",
            "severity_target",
        ]
    )

    assert numeric == ["B01003_E001"]
    assert categorical == ["reportingagencytypecode", "crossingclosed"]


def test_ft_preprocessor_puts_category_codes_last() -> None:
    train = pd.DataFrame(
        {
            "numeric": [1.0, np.nan, 3.0],
            "category": ["a", "b", None],
        }
    )
    validation = pd.DataFrame(
        {"numeric": [2.0], "category": ["unseen"]}
    )
    preprocessor = make_ft_transformer_preprocessor(
        ["numeric"], ["category"], n_quantiles=3
    )

    transformed_train = preprocessor.fit_transform(train)
    transformed_validation = preprocessor.transform(validation)

    assert transformed_train.shape == (3, 3)
    assert transformed_validation.shape == (1, 3)
    assert transformed_validation[0, -1] == -1


def test_one_hot_preprocessor_handles_missing_and_unseen_values() -> None:
    train = pd.DataFrame(
        {
            "numeric": [1.0, np.nan, 3.0],
            "category": ["a", "b", None],
        }
    )
    validation = pd.DataFrame(
        {"numeric": [np.nan], "category": ["unseen"]}
    )
    preprocessor = make_one_hot_preprocessor(
        ["numeric"], ["category"], scale_numeric=True
    )

    transformed_train = preprocessor.fit_transform(train)
    transformed_validation = preprocessor.transform(validation)

    assert transformed_train.shape[0] == 3  # pyright: ignore[reportOptionalSubscript]
    assert transformed_validation.shape == (  # pyright: ignore[reportOptionalSubscript]
        1,
        transformed_train.shape[1],  # pyright: ignore[reportOptionalSubscript]
    )
