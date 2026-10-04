"""Shared interface for classifiers used by the tuning pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Common contract used by temporal CV and Bayesian tuning."""

    name: str

    def __init__(
        self,
        params: dict | None = None,
        *,
        random_state: int = 42,
    ) -> None:
        self.params = dict(params or {})
        self.random_state = random_state

    @classmethod
    @abstractmethod
    def suggest_params(cls, trial):
        """Return one hyperparameter set suggested by ``trial``."""

    @abstractmethod
    def fit(
        self,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
    ):
        """Fit a fresh model, optionally using validation data for early stopping."""

    @abstractmethod
    def predict_proba(self, X):
        """Return one probability column per integer-encoded class."""
