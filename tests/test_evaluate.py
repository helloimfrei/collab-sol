from __future__ import annotations

import csv

import numpy as np
import pytest

from collab_sol.evaluate import (
    _inspection_budget_rows,
    evaluate_binary_risk,
    generate_risk_report,
)


class FixedProbabilityModel:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = probabilities

    def predict_proba(self, X) -> np.ndarray:
        assert len(X) == len(self.probabilities)
        return self.probabilities


def test_binary_risk_metrics_evaluate_continuous_scores() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.1, 0.9]]
    )

    metrics = evaluate_binary_risk(labels, probabilities)

    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.prevalence == pytest.approx(0.5)


def test_report_writes_metrics_scores_budgets_and_curve(tmp_path) -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.1, 0.9]]
    )
    model = FixedProbabilityModel(probabilities)

    metrics = generate_risk_report(
        model,
        np.zeros((4, 1)),
        labels,
        output_dir=tmp_path,
        row_ids=np.array(["A", "B", "C", "D"]),
    )

    assert metrics.average_precision == pytest.approx(1.0)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "risk_scores.csv").exists()
    assert (tmp_path / "inspection_budget.csv").exists()
    assert (tmp_path / "precision_recall_curve.png").exists()
    with (tmp_path / "risk_scores.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["row_id"] == "A"
    assert float(rows[-1]["risk_score"]) == pytest.approx(0.9)

    with (tmp_path / "inspection_budget.csv").open() as file:
        budget_rows = list(csv.DictReader(file))
    assert [int(row["inspection_budget_crossings"]) for row in budget_rows] == [
        10,
        100,
        500,
        1_000,
    ]
    assert int(budget_rows[0]["incident_crossings_captured"]) == 2
    assert float(budget_rows[0]["precision"]) == pytest.approx(0.5)
    assert float(budget_rows[0]["recall"]) == pytest.approx(1.0)
    assert float(budget_rows[0]["lift_over_random"]) == pytest.approx(1.0)
    assert float(budget_rows[0]["risk_score_cutoff"]) == pytest.approx(0.1)


def test_report_rejects_mismatched_row_ids(tmp_path) -> None:
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9]])

    with pytest.raises(ValueError, match="row_ids"):
        generate_risk_report(
            FixedProbabilityModel(probabilities),
            np.zeros((2, 1)),
            np.array([0, 1]),
            output_dir=tmp_path,
            row_ids=["only-one"],
        )


def test_inspection_budgets_use_absolute_top_counts() -> None:
    labels = np.zeros(1_000, dtype=int)
    labels[[0, 99, 499, 999]] = 1
    scores = np.linspace(1.0, 0.001, 1_000)

    rows = _inspection_budget_rows(labels, scores)

    assert all("crossings_reviewed" not in row for row in rows)
    assert [row["incident_crossings_captured"] for row in rows] == [1, 2, 3, 4]
