"""Small utilities for reproducible grade-crossing model comparisons."""

from collab_sol.models import IsolationForestRiskModel
from collab_sol.evaluate import (
    RiskMetrics,
    SeverityMetrics,
    SeverityRegressionMetrics,
    evaluate_binary_risk,
    evaluate_severity,
    evaluate_severity_regression,
    expected_severity,
    generate_risk_report,
    generate_severity_report,
    generate_severity_regression_report,
    multiclass_log_loss,
    severity_regression_mae,
)
from collab_sol.features import (
    EXCLUDED_RISK_FEATURES,
    SEVERITY_CLASSES,
    SEVERITY_CODES,
    SEVERITY_VALUES,
    SEVERITY_WEIGHTS,
    split_risk_feature_columns,
)
from collab_sol.splitting import (
    TemporalFold,
    TemporalSplit,
    make_expanding_temporal_split,
    stratified_crossing_sample,
)
from collab_sol.tuning import tune_model

__all__ = [
    "RiskMetrics",
    "SeverityMetrics",
    "SeverityRegressionMetrics",
    "TemporalFold",
    "TemporalSplit",
    "EXCLUDED_RISK_FEATURES",
    "SEVERITY_CLASSES",
    "SEVERITY_CODES",
    "SEVERITY_VALUES",
    "SEVERITY_WEIGHTS",
    "IsolationForestRiskModel",
    "evaluate_binary_risk",
    "evaluate_severity",
    "evaluate_severity_regression",
    "expected_severity",
    "generate_risk_report",
    "generate_severity_report",
    "generate_severity_regression_report",
    "make_expanding_temporal_split",
    "multiclass_log_loss",
    "severity_regression_mae",
    "stratified_crossing_sample",
    "split_risk_feature_columns",
    "tune_model",
]
