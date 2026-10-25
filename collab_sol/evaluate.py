"""Evaluation helpers shared by binary crash-risk models."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)

DEFAULT_INSPECTION_BUDGETS = (10, 50, 100, 500, 1_000, 5_000, 10_000)


@dataclass(frozen=True)
class RiskMetrics:
    """Metrics for continuous binary crash-risk scores."""

    prevalence: float
    average_precision: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _binary_targets(y_true) -> np.ndarray:
    labels = np.asarray(y_true)
    if labels.ndim != 1:
        raise ValueError("Binary targets must be one-dimensional.")
    if labels.size == 0:
        raise ValueError("Binary targets cannot be empty.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Binary targets must contain only 0 and 1.")
    return labels.astype(int)


def positive_class_probabilities(probabilities) -> np.ndarray:
    """Return positive-class probabilities from a binary model output."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Binary probabilities must have shape (rows, 2).")
    positive = values[:, 1]
    if not np.isfinite(positive).all() or ((positive < 0) | (positive > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    return positive


def binary_average_precision(y_true, probabilities: np.ndarray) -> float:
    """Score rare-event ranking using the positive-class risk score."""

    labels = _binary_targets(y_true)
    positive = positive_class_probabilities(probabilities)
    if len(labels) != len(positive):
        raise ValueError("Targets and probabilities must have the same length.")
    return float(average_precision_score(labels, positive))


def evaluate_binary_risk(
    y_true,
    probabilities: np.ndarray,
) -> RiskMetrics:
    """Evaluate continuous positive-class risk scores."""

    labels = _binary_targets(y_true)
    positive = positive_class_probabilities(probabilities)
    if len(labels) != len(positive):
        raise ValueError("Targets and probabilities must have the same length.")

    return RiskMetrics(
        prevalence=float(labels.mean()),
        average_precision=float(average_precision_score(labels, positive)),
    )


def generate_risk_report(
    model,
    X,
    y_true,
    *,
    output_dir: str | Path,
    row_ids=None,
) -> RiskMetrics:
    """Save metrics, risk scores, budget performance, and a PR curve."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    labels = _binary_targets(y_true)
    probabilities = np.asarray(model.predict_proba(X))
    positive = positive_class_probabilities(probabilities)
    metrics = evaluate_binary_risk(labels, probabilities)

    if row_ids is None:
        identifiers = np.arange(len(labels))
    else:
        identifiers = np.asarray(row_ids)
        if identifiers.ndim != 1 or len(identifiers) != len(labels):
            raise ValueError("row_ids must be one-dimensional and match y_true.")

    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.to_dict().items())

    with (output / "risk_scores.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "actual", "risk_score"])
        writer.writerows(zip(identifiers, labels, positive))

    budget_rows = _inspection_budget_rows(labels, positive)
    with (output / "inspection_budget.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(budget_rows[0]))
        writer.writeheader()
        writer.writerows(budget_rows)

    _save_precision_recall_curve(labels, positive, output / "precision_recall_curve.png")
    return metrics


def _inspection_budget_rows(
    labels: np.ndarray,
    positive: np.ndarray,
) -> list[dict[str, float | int]]:
    """Calculate performance for fixed counts of top-ranked crossings."""

    order = np.argsort(-positive, kind="stable")
    ranked_labels = labels[order]
    ranked_scores = positive[order]
    cumulative_incidents = np.cumsum(ranked_labels)
    total_incidents = int(cumulative_incidents[-1])
    prevalence = float(labels.mean())

    rows = []
    for budget in DEFAULT_INSPECTION_BUDGETS:
        reviewed = min(budget, len(labels))
        captured = int(cumulative_incidents[reviewed - 1])
        precision = captured / reviewed
        recall = captured / total_incidents if total_incidents else 0.0
        lift = precision / prevalence if prevalence else 0.0
        rows.append(
            {
                "inspection_budget_crossings": budget,
                "incident_crossings_captured": captured,
                "precision": precision,
                "recall": recall,
                "lift_over_random": lift,
                "risk_score_cutoff": float(ranked_scores[reviewed - 1]),
            }
        )
    return rows


def _save_precision_recall_curve(
    labels: np.ndarray,
    positive: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(labels, positive)
    figure, axis = plt.subplots()
    axis.plot(recall, precision)
    axis.axhline(labels.mean(), color="gray", linestyle="--", label="prevalence")
    axis.set(xlabel="Recall", ylabel="Precision", title="Crash-risk precision-recall curve")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_tree_shap_summary(
    native_model,
    X,
    output_path: str | Path,
    *,
    feature_names: list[str] | None = None,
    max_rows: int = 5_000,
    random_state: int = 42,
) -> None:
    """Save an optional SHAP summary for a fitted XGBoost or LightGBM model."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Install SHAP with: pip install 'collab-sol[explain]'") from error
    try:
        shap = import_module("shap")
    except ImportError as error:
        raise ImportError("Install SHAP with: pip install 'collab-sol[explain]'") from error

    values = _sample_explanation_rows(X, max_rows=max_rows, random_state=random_state)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(native_model)
    shap_values = explainer.shap_values(values)
    shap.summary_plot(
        shap_values,
        values,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close()


def save_neural_shap_summary(
    native_model,
    background,
    X,
    output_path: str | Path,
    *,
    feature_names: list[str] | None = None,
    max_rows: int = 1_000,
    random_state: int = 42,
) -> None:
    """Save a SHAP summary for a PyTorch model with one tabular tensor input."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Install SHAP with: pip install 'collab-sol[explain]'") from error
    try:
        shap = import_module("shap")
    except ImportError as error:
        raise ImportError("Install SHAP with: pip install 'collab-sol[explain]'") from error
    try:
        torch = import_module("torch")
    except ImportError as error:
        raise ImportError("Install PyTorch to explain a neural model.") from error

    background_tensor = torch.as_tensor(background, dtype=torch.float32)
    values = _sample_explanation_rows(X, max_rows=max_rows, random_state=random_state)
    values_tensor = torch.as_tensor(values, dtype=torch.float32)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    explainer = shap.GradientExplainer(native_model, background_tensor)
    shap_values = explainer.shap_values(values_tensor)
    shap.summary_plot(
        shap_values,
        values,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close()


def _sample_explanation_rows(X, *, max_rows: int, random_state: int) -> np.ndarray:
    values = np.asarray(X)
    if values.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1.")
    if len(values) > max_rows:
        rng = np.random.default_rng(random_state)
        values = values[rng.choice(len(values), size=max_rows, replace=False)]
    return values
