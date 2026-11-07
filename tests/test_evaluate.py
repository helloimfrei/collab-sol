from __future__ import annotations

import csv

import numpy as np
import pytest

from collab_sol.evaluate import (
    _inspection_budget_rows,
    _severity_budget_rows,
    evaluate_binary_risk,
    evaluate_severity,
    expected_severity,
    generate_risk_report,
    generate_severity_report,
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
    assert [int(row["inspection_budget_crossings"]) for row in budget_rows] == [4]
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
    assert [row["incident_crossings_captured"] for row in rows] == [1, 1, 2, 3, 4]


def test_severity_metrics_use_three_class_probabilities() -> None:
    labels = np.array([0, 1, 2])
    probabilities = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.2, 0.8],
        ]
    )

    metrics = evaluate_severity(labels, probabilities)
    scores = expected_severity(probabilities)

    assert metrics.multiclass_log_loss > 0
    assert metrics.macro_f1 == pytest.approx(1.0)
    assert metrics.ordinal_mae == pytest.approx(0.0)
    assert scores == pytest.approx([1.2, 2.0, 2.8])


def test_severity_report_writes_probabilities_metrics_and_budgets(tmp_path) -> None:
    labels = np.array([0, 1, 2, 2])
    probabilities = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.1, 0.9],
            [0.1, 0.2, 0.7],
        ]
    )

    metrics = generate_severity_report(
        FixedProbabilityModel(probabilities),
        np.zeros((4, 1)),
        labels,
        output_dir=tmp_path,
        row_ids=np.array(["A", "B", "C", "D"]),
    )

    assert metrics.macro_f1 == pytest.approx(1.0)
    for filename in (
        "metrics.csv",
        "severity_scores.csv",
        "per_class_metrics.csv",
        "confusion_matrix.csv",
        "severity_budget.csv",
    ):
        assert (tmp_path / filename).exists()

    with (tmp_path / "severity_budget.csv").open() as file:
        budget = list(csv.DictReader(file))[0]
    assert int(budget["severity_review_budget_incidents"]) == 4
    assert int(budget["fatal_incidents_captured"]) == 2
    assert int(budget["injury_or_fatal_incidents_captured"]) == 3


def test_severity_budget_prioritizes_expected_severity() -> None:
    labels = np.zeros(100, dtype=int)
    labels[:5] = [2, 2, 1, 1, 0]
    scores = np.linspace(2.0, 0.0, 100)

    rows = _severity_budget_rows(labels, scores)

    assert rows[0]["severity_review_budget_incidents"] == 10
    assert rows[0]["fatal_incidents_captured"] == 2
    assert rows[0]["injury_or_fatal_incidents_captured"] == 4
