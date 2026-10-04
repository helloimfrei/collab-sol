"""Metrics shared by model tuning and final evaluation."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score


def binary_average_precision(
    y_true: NDArray[Any], probabilities: NDArray[np.floating[Any]]
) -> float:
    """Score rare-event predictions using the positive-class probability."""

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Binary probabilities must have shape (rows, 2).")
    return float(average_precision_score(y_true, probabilities[:, 1]))
