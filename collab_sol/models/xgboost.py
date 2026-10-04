"""XGBoost adapter for risk and conditional-severity classification."""

from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from collab_sol.models.base import BaseModel


class XGBoostModel(BaseModel):
    """Train XGBoost on features encoded by the preprocessing pipeline."""

    name = "xgboost"

    def __init__(
        self,
        params: dict | None = None,
        *,
        random_state: int = 42,
        balance_classes: bool = True,
        early_stopping_rounds: int = 30,
    ) -> None:
        super().__init__(params, random_state=random_state)
        self.balance_classes = balance_classes
        self.early_stopping_rounds = early_stopping_rounds
        self._model: XGBClassifier | None = None

    @classmethod
    def suggest_params(cls, trial):
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 300, 2_000, step=100
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 20.0, log=True
            ),
        }

    def fit(
        self,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
    ):
        if (X_valid is None) != (y_valid is None):
            raise ValueError("Provide both X_valid and y_valid, or neither.")

        labels = np.asarray(y_train)
        classes = np.unique(labels)
        if len(classes) < 2:
            raise ValueError("Training data must contain at least two classes.")
        if not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError("Labels must be contiguous integers starting at 0.")

        n_classes = len(classes)
        model_params = {
            "n_estimators": 2_000,
            "max_depth": 12,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "n_jobs": -1,
            **self.params,
            "random_state": self.random_state,
            "objective": (
                "binary:logistic" if n_classes == 2 else "multi:softprob"
            ),
            "eval_metric": "aucpr" if n_classes == 2 else "mlogloss",
        }
        if n_classes > 2:
            model_params["num_class"] = n_classes
        if X_valid is not None:
            model_params["early_stopping_rounds"] = self.early_stopping_rounds

        model = XGBClassifier(**model_params)
        sample_weight = (
            compute_sample_weight(class_weight="balanced", y=labels)
            if self.balance_classes
            else None
        )
        eval_set = (
            [(X_valid, y_valid)]
            if X_valid is not None and y_valid is not None
            else None
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=False,
        )
        self._model = model
        return self

    def predict_proba(self, X):
        if self._model is None:
            raise RuntimeError("Call fit before predict_proba.")
        return self._model.predict_proba(X)

    @property
    def native_model(self) -> XGBClassifier:
        """Expose the fitted estimator for diagnostics and persistence."""

        if self._model is None:
            raise RuntimeError("Call fit before accessing native_model.")
        return self._model
