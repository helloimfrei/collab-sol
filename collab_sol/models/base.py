"""Shared interface for classifiers used by the tuning pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol, Self, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FeatureMatrix: TypeAlias = NDArray[Any] | pd.DataFrame
TargetVector: TypeAlias = NDArray[Any] | pd.Series
ModelParams: TypeAlias = dict[str, Any]


class ParameterTrial(Protocol):
    """The subset of an Optuna trial that model search spaces may use."""

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int: ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float: ...

    def suggest_categorical(
        self,
        name: str,
        choices: Sequence[Any],
    ) -> Any: ...


class BaseModel(ABC):
    """Common contract used by temporal CV and Bayesian tuning."""

    name: ClassVar[str]

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        random_state: int = 42,
    ) -> None:
        self.params = dict(params or {})
        self.random_state = random_state

    @classmethod
    @abstractmethod
    def suggest_params(cls, trial: ParameterTrial) -> ModelParams:
        """Return one hyperparameter set suggested by ``trial``."""

    @abstractmethod
    def fit(
        self,
        X_train: FeatureMatrix,
        y_train: TargetVector,
        X_valid: FeatureMatrix | None = None,
        y_valid: TargetVector | None = None,
    ) -> Self:
        """Fit a fresh model, optionally using validation data for early stopping."""

    @abstractmethod
    def predict_proba(self, X: FeatureMatrix) -> NDArray[np.floating[Any]]:
        """Return one probability column per integer-encoded class."""
