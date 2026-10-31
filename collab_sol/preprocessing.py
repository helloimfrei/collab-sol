"""Model-specific, fold-safe preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    QuantileTransformer,
)

MISSING_LEVEL = "__missing__"


def _categorical_labels(frame: pd.DataFrame) -> np.ndarray:
    values = pd.DataFrame(frame).astype(object)
    return values.where(values.notna(), MISSING_LEVEL).astype(str).to_numpy()


def make_ft_transformer_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    n_quantiles: int = 1_000,
    random_state: int = 42,
) -> ColumnTransformer:
    """Encode continuous values and categories for an FT-Transformer."""

    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    QuantileTransformer(
                        n_quantiles=n_quantiles,
                        output_distribution="normal",
                        subsample=1_000_000,
                        random_state=random_state,
                    ),
                ),
                numeric_features,
            ),
            (
                "missing_indicators",
                MissingIndicator(features="missing-only", error_on_new=False),
                numeric_features,
            ),
            (
                "categorical",
                make_pipeline(
                    FunctionTransformer(_categorical_labels),
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        dtype=np.float32,  # pyright: ignore[reportArgumentType]
                    ),
                ),
                categorical_features,
            ),
        ],
        sparse_threshold=0.0,
    )
