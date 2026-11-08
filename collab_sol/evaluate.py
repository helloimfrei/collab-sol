"""Evaluation helpers shared by binary crash-risk models."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
from collab_sol.features import SEVERITY_CLASSES, SEVERITY_WEIGHTS
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_recall_curve,
)

DEFAULT_INSPECTION_BUDGETS = (10, 50, 100, 500, 1_000, 5_000, 10_000)
DEFAULT_SEVERITY_CLASSES = SEVERITY_CLASSES
DEFAULT_SEVERITY_WEIGHTS = SEVERITY_WEIGHTS


@dataclass(frozen=True)
class RiskMetrics:
    """Metrics for continuous binary crash-risk scores."""

    prevalence: float
    average_precision: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class SeverityMetrics:
    """Metrics for three-class conditional-severity probabilities."""

    multiclass_log_loss: float
    macro_f1: float
    ordinal_mae: float

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


def multiclass_log_loss(y_true, probabilities: np.ndarray) -> float:
    """Score multiclass probabilities for minimization during tuning."""

    labels, values = _validated_severity_inputs(y_true, probabilities)
    return float(log_loss(labels, values, labels=np.arange(values.shape[1])))


def expected_severity(
    probabilities: np.ndarray,
    *,
    class_weights: tuple[float, ...] = DEFAULT_SEVERITY_WEIGHTS,
) -> np.ndarray:
    """Return the probability-weighted ordinal severity score."""

    values = np.asarray(probabilities, dtype=float)
    weights = np.asarray(class_weights, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(weights):
        raise ValueError("Probability columns must match the severity weights.")
    if not np.isfinite(weights).all():
        raise ValueError("Severity weights must be finite.")
    values = _validated_probability_matrix(values)
    return values @ weights


def evaluate_severity(
    y_true,
    probabilities: np.ndarray,
) -> SeverityMetrics:
    """Evaluate conditional-severity probabilities and ordinal predictions."""

    labels, values = _validated_severity_inputs(y_true, probabilities)
    predicted = values.argmax(axis=1)
    return SeverityMetrics(
        multiclass_log_loss=float(
            log_loss(labels, values, labels=np.arange(values.shape[1]))
        ),
        macro_f1=float(f1_score(labels, predicted, average="macro")),
        ordinal_mae=float(np.mean(np.abs(labels - predicted))),
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


def generate_severity_report(
    model,
    X,
    y_true,
    *,
    output_dir: str | Path,
    row_ids=None,
    class_names: tuple[str, ...] = DEFAULT_SEVERITY_CLASSES,
    class_weights: tuple[float, ...] = DEFAULT_SEVERITY_WEIGHTS,
) -> SeverityMetrics:
    """Save multiclass metrics, probabilities, and severity-budget results."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    labels, probabilities = _validated_severity_inputs(y_true, probabilities)
    if len(class_names) != probabilities.shape[1]:
        raise ValueError("Class names must match the probability columns.")
    metrics = evaluate_severity(labels, probabilities)
    predicted = probabilities.argmax(axis=1)
    severity_scores = expected_severity(
        probabilities,
        class_weights=class_weights,
    )
    identifiers = _validated_row_ids(row_ids, len(labels))

    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.to_dict().items())

    probability_columns = [f"probability_{name}" for name in class_names]
    with (output / "severity_scores.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "row_id",
                "actual_class",
                "predicted_class",
                "severity_score",
                *probability_columns,
            ]
        )
        for row_id, actual, prediction, score, row_probabilities in zip(
            identifiers,
            labels,
            predicted,
            severity_scores,
            probabilities,
        ):
            writer.writerow(
                [
                    row_id,
                    class_names[actual],
                    class_names[prediction],
                    score,
                    *row_probabilities,
                ]
            )

    per_class = precision_recall_fscore_support(
        labels,
        predicted,
        labels=np.arange(len(class_names)),
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )
    precision = np.asarray(per_class[0], dtype=float)
    recall = np.asarray(per_class[1], dtype=float)
    f1 = np.asarray(per_class[2], dtype=float)
    support = np.asarray(per_class[3], dtype=int)
    with (output / "per_class_metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        writer.writerows(
            [class_name, class_precision, class_recall, class_f1, class_support]
            for class_name, class_precision, class_recall, class_f1, class_support
            in zip(class_names, precision, recall, f1, support)
        )

    matrix = confusion_matrix(
        labels,
        predicted,
        labels=np.arange(len(class_names)),
    )
    with (output / "confusion_matrix.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual_class", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])
    _save_severity_confusion_matrix(
        matrix,
        class_names,
        output / "confusion_matrix.png",
    )

    budget_rows = _severity_budget_rows(
        labels,
        severity_scores,
        class_weights=class_weights,
    )
    with (output / "severity_budget.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(budget_rows[0]))
        writer.writeheader()
        writer.writerows(budget_rows)
    return metrics


def _validated_severity_inputs(
    y_true,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    values = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("Severity targets must be a non-empty one-dimensional array.")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.equal(labels, labels.astype(int)).all():
            raise ValueError("Severity targets must be integer class codes.")
    labels = labels.astype(int)
    values = _validated_probability_matrix(values)
    if len(labels) != len(values):
        raise ValueError("Targets and probabilities must have the same length.")
    if labels.min() < 0 or labels.max() >= values.shape[1]:
        raise ValueError("Severity targets are outside the probability columns.")
    return labels, values


def _validated_probability_matrix(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Multiclass probabilities must have at least two columns.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    row_sums = values.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to 1.")
    return values / row_sums


def _validated_row_ids(row_ids, size: int) -> np.ndarray:
    if row_ids is None:
        return np.arange(size)
    identifiers = np.asarray(row_ids)
    if identifiers.ndim != 1 or len(identifiers) != size:
        raise ValueError("row_ids must be one-dimensional and match y_true.")
    return identifiers


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

    available_budgets = [
        budget for budget in DEFAULT_INSPECTION_BUDGETS if budget <= len(labels)
    ]
    if not available_budgets:
        available_budgets = [len(labels)]

    rows = []
    for budget in available_budgets:
        reviewed = budget
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


def _severity_budget_rows(
    labels: np.ndarray,
    severity_scores: np.ndarray,
    *,
    class_weights: tuple[float, ...] = DEFAULT_SEVERITY_WEIGHTS,
) -> list[dict[str, float | int]]:
    """Evaluate the highest expected-severity crash cases at fixed budgets."""

    order = np.argsort(-severity_scores, kind="stable")
    ranked_labels = labels[order]
    ranked_scores = severity_scores[order]
    weights = np.asarray(class_weights, dtype=float)
    observed_severity = weights[ranked_labels]
    fatal_class = len(weights) - 1
    fatal = ranked_labels == fatal_class
    serious = ranked_labels > 0
    total_fatal = int(fatal.sum())
    total_serious = int(serious.sum())
    overall_mean = float(observed_severity.mean())

    budgets = [
        budget for budget in DEFAULT_INSPECTION_BUDGETS if budget <= len(labels)
    ]
    if not budgets:
        budgets = [len(labels)]

    rows = []
    for budget in budgets:
        fatal_captured = int(fatal[:budget].sum())
        serious_captured = int(serious[:budget].sum())
        mean_observed = float(observed_severity[:budget].mean())
        rows.append(
            {
                "severity_review_budget_incidents": budget,
                "fatal_incidents_captured": fatal_captured,
                "fatal_precision": fatal_captured / budget,
                "fatal_recall": (
                    fatal_captured / total_fatal if total_fatal else 0.0
                ),
                "injury_or_fatal_incidents_captured": serious_captured,
                "injury_or_fatal_precision": serious_captured / budget,
                "injury_or_fatal_recall": (
                    serious_captured / total_serious if total_serious else 0.0
                ),
                "mean_observed_severity": mean_observed,
                "severity_lift_over_all_incidents": (
                    mean_observed / overall_mean if overall_mean else 0.0
                ),
                "severity_score_cutoff": float(ranked_scores[budget - 1]),
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


def _save_severity_confusion_matrix(
    matrix: np.ndarray,
    class_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """Save counts and actual-class percentages in one confusion-matrix plot."""

    import matplotlib.pyplot as plt

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    display_names = [name.replace("_", " ").title() for name in class_names]

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Share of actual class")
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=display_names,
        yticklabels=display_names,
        xlabel="Predicted severity",
        ylabel="Actual severity",
        title="Conditional-severity confusion matrix",
    )
    axis.tick_params(axis="x", rotation=25)

    for actual in range(len(class_names)):
        for predicted in range(len(class_names)):
            percentage = normalized[actual, predicted]
            axis.text(
                predicted,
                actual,
                f"{matrix[actual, predicted]:,}\n{percentage:.1%}",
                ha="center",
                va="center",
                color="white" if percentage > 0.5 else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
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
