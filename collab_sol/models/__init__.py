"""Model implementations used by the shared training entry points."""

from typing import TYPE_CHECKING

from collab_sol.models.anomaly import IsolationForestRiskModel
from collab_sol.models.classical import (
    build_logistic_regression,
    build_multiclass_logistic_regression,
    build_ridge_regression,
    build_xgboost_classifier,
    build_xgboost_multiclass_classifier,
    build_xgboost_regressor,
)

if TYPE_CHECKING:
    from collab_sol.models.neural import (
        FTTransformerClassifier,
        FTTransformerMulticlassClassifier,
        FTTransformerRegressor,
        TorchMLPClassifier,
        TorchMLPRegressor,
    )

__all__ = [
    "FTTransformerClassifier",
    "FTTransformerMulticlassClassifier",
    "FTTransformerRegressor",
    "IsolationForestRiskModel",
    "TorchMLPClassifier",
    "TorchMLPRegressor",
    "build_logistic_regression",
    "build_multiclass_logistic_regression",
    "build_ridge_regression",
    "build_xgboost_classifier",
    "build_xgboost_multiclass_classifier",
    "build_xgboost_regressor",
]


def __getattr__(name: str):
    if name in {
        "FTTransformerClassifier",
        "FTTransformerMulticlassClassifier",
        "FTTransformerRegressor",
        "TorchMLPClassifier",
        "TorchMLPRegressor",
    }:
        from collab_sol.models.neural import (
            FTTransformerClassifier,
            FTTransformerMulticlassClassifier,
            FTTransformerRegressor,
            TorchMLPClassifier,
            TorchMLPRegressor,
        )

        return {
            "FTTransformerClassifier": FTTransformerClassifier,
            "FTTransformerMulticlassClassifier": FTTransformerMulticlassClassifier,
            "FTTransformerRegressor": FTTransformerRegressor,
            "TorchMLPClassifier": TorchMLPClassifier,
            "TorchMLPRegressor": TorchMLPRegressor,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
