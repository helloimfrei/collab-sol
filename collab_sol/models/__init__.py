"""Models that conform to the shared tuning interface."""

from collab_sol.models.base import BaseModel, ParameterTrial
from collab_sol.models.xgboost import XGBoostModel

__all__ = ["BaseModel", "ParameterTrial", "XGBoostModel"]
