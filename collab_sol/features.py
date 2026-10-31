"""Shared predictor selection for grade-crossing risk models."""

from __future__ import annotations

from collections.abc import Sequence

ID_COLUMNS = ("crossingid", "asof_year")
TARGET_COLUMNS = ("incident", "severity_target")
FIRST_CATEGORICAL_COLUMN = "reportingagencytypecode"

EXCLUDED_RISK_FEATURES = {
    "latitude": "Near-unique crossing coordinate that can act like an ID.",
    "longitude": "Near-unique crossing coordinate that can act like an ID.",
    "latlongsourcecode": "Coordinate-provenance metadata and record-quality proxy.",
    "B01001_E001": "ACS total population duplicated by B01003_E001.",
    "B03002_E001": "ACS total population duplicated by B01003_E001.",
}


def split_risk_feature_columns(
    all_columns: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return the shared numeric and categorical risk predictors."""

    features = [
        column
        for column in all_columns
        if column not in ID_COLUMNS + TARGET_COLUMNS
        and column not in EXCLUDED_RISK_FEATURES
    ]
    categorical_start = features.index(FIRST_CATEGORICAL_COLUMN)
    return features[:categorical_start], features[categorical_start:]
