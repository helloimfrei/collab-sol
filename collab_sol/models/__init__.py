"""Model implementations used by the shared training entry points."""

from typing import TYPE_CHECKING

from collab_sol.models.anomaly import IsolationForestRiskModel
from collab_sol.models.classical import (
    build_logistic_regression,
    build_xgboost_classifier,
)

if TYPE_CHECKING:
    from collab_sol.models.neural import (
        FTTransformerClassifier,
        TorchMLPClassifier,
    )

__all__ = [
    "FTTransformerClassifier",
    "IsolationForestRiskModel",
    "TorchMLPClassifier",
    "build_logistic_regression",
    "build_xgboost_classifier",
]


def __getattr__(name: str):
    if name in {"FTTransformerClassifier", "TorchMLPClassifier"}:
        from collab_sol.models.neural import (
            FTTransformerClassifier,
            TorchMLPClassifier,
        )

        return {
            "FTTransformerClassifier": FTTransformerClassifier,
            "TorchMLPClassifier": TorchMLPClassifier,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
