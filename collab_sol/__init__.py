"""Small utilities for reproducible grade-crossing model comparisons."""

from collab_sol.models import IsolationForestRiskModel
from collab_sol.evaluate import (
    RiskMetrics,
    SeverityMetrics,
    evaluate_binary_risk,
    evaluate_severity,
    generate_risk_report,
    generate_severity_report,
    multiclass_log_loss,
)
from collab_sol.features import (
    EXCLUDED_RISK_FEATURES,
    SEVERITY_CLASSES,
    SEVERITY_CODES,
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
    "TemporalFold",
    "TemporalSplit",
    "EXCLUDED_RISK_FEATURES",
    "SEVERITY_CLASSES",
    "SEVERITY_CODES",
    "SEVERITY_WEIGHTS",
    "IsolationForestRiskModel",
    "evaluate_binary_risk",
    "evaluate_severity",
    "generate_risk_report",
    "generate_severity_report",
    "make_expanding_temporal_split",
    "multiclass_log_loss",
    "stratified_crossing_sample",
    "split_risk_feature_columns",
    "tune_model",
]
