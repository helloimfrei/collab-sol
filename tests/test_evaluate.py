from __future__ import annotations

import csv

import numpy as np
import pytest

from collab_sol.evaluate import (
    evaluate_binary_risk,
    generate_risk_report,
    select_f1_threshold,
)


class FixedProbabilityModel:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = probabilities

    def predict_proba(self, X) -> np.ndarray:
        assert len(X) == len(self.probabilities)
        return self.probabilities


def test_binary_risk_metrics_include_threshold_free_and_threshold_metrics() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.1, 0.9]]
    )

    metrics = evaluate_binary_risk(labels, probabilities, threshold=0.5)

    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.true_positives == 2
    assert metrics.true_negatives == 2


def test_threshold_is_selected_from_validation_probabilities() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [[0.95, 0.05], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]]
    )

    assert select_f1_threshold(labels, probabilities) == pytest.approx(0.3)


def test_report_writes_metrics_predictions_and_curve(tmp_path) -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.1, 0.9]]
    )
    model = FixedProbabilityModel(probabilities)

    metrics = generate_risk_report(
        model,
        np.zeros((4, 1)),
        labels,
        threshold=0.5,
        output_dir=tmp_path,
        row_ids=np.array(["A", "B", "C", "D"]),
    )

    assert metrics.average_precision == pytest.approx(1.0)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "precision_recall_curve.png").exists()
    with (tmp_path / "predictions.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["row_id"] == "A"
    assert rows[-1]["predicted"] == "1"


def test_report_rejects_mismatched_row_ids(tmp_path) -> None:
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9]])

    with pytest.raises(ValueError, match="row_ids"):
        generate_risk_report(
            FixedProbabilityModel(probabilities),
            np.zeros((2, 1)),
            np.array([0, 1]),
            threshold=0.5,
            output_dir=tmp_path,
            row_ids=["only-one"],
        )
