# pyright: reportMissingImports=false

"""Small, explicitly non-benchmark TabFM risk experiment.

TabFM's pretrained weights are non-commercial. Keep TabFM out of the project's
normal dependency set: its checkpoint is large and its reference environment
can conflict with the training stack.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch

TARGET = "risk_target"
YEAR = "feature_year"
CATEGORICAL_FEATURES = [
    "statecode",
    "trafficlane",
    "highwaypaved",
    "tracksignaled",
    "roadatcrossing",
    "crossingsurfaceid1",
    "highwaysystem",
]
NUMERIC_FEATURES = [
    YEAR,
    "latitude",
    "longitude",
    "numberofmaintracks",
    "numberofsidingtracks",
    "numberofyardtracks",
    "maximumtimetablespeed",
    "highwayspeedlimit",
    "annualaveragedailytrafficcount",
    "trainsperweek",
    "numberpassengertrainperday",
    "totaldaylightthrutrains",
    "totalnighttimethrutrains",
    "totalswitchingtrains",
    "totaltransittrains",
    "numbercrossbuckassemblies",
    "numberstopsigns",
    "numberyieldsigns",
    "countroadwaygatearms",
    "countmastmountedflashinglight",
    "countflashinglightpair",
    "numberofbells",
    "B01003_E001",
    "B19013_E001",
    "pr_total_mm_5y_mean",
    "wet_days_5y_mean",
    "freeze_thaw_days_5y_mean",
    "hot_days_5y_mean",
    "wind_mean_ms_5y_mean",
    "high_wind_days_5y_mean",
]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-year", type=int, default=2022)
    parser.add_argument("--test-year", type=int, default=2023)
    parser.add_argument("--context-per-class", type=int, default=64)
    parser.add_argument("--test-positives", type=int, default=128)
    parser.add_argument("--test-negatives", type=int, default=384)
    parser.add_argument("--n-estimators", type=int, default=2)
    parser.add_argument("--prediction-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=137)
    return parser.parse_args()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = cast(pd.DataFrame, frame[FEATURES].copy())
    for column in NUMERIC_FEATURES:
        source = cast(pd.Series, prepared[column])
        numeric = cast(pd.Series, pd.to_numeric(source, errors="coerce"))
        prepared[column] = numeric.astype("float64")
    for column in CATEGORICAL_FEATURES:
        source = cast(pd.Series, prepared[column])
        prepared[column] = source.astype("string").fillna("<missing>")
    return prepared


def stratified_sample(
    frame: pd.DataFrame, positive_rows: int, negative_rows: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    positive_pool = frame.loc[frame[TARGET] == 1]
    negative_pool = frame.loc[frame[TARGET] == 0]
    n_positive = min(positive_rows, len(positive_pool))
    n_negative = min(negative_rows, len(negative_pool))
    positives = positive_pool.sample(n=n_positive, random_state=seed)
    negatives = negative_pool.sample(n=n_negative, random_state=seed + 1)
    sampled = pd.concat([positives, negatives]).sample(frac=1, random_state=seed + 2)
    weights = np.where(
        sampled[TARGET].to_numpy() == 1,
        len(positive_pool) / n_positive,
        len(negative_pool) / n_negative,
    )
    return prepare(sampled), sampled[TARGET].to_numpy(dtype=np.int8), weights


def adjust_prior(probability: np.ndarray, context_rate: float, population_rate: float) -> np.ndarray:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    odds = clipped / (1 - clipped)
    multiplier = (population_rate / (1 - population_rate)) / (
        context_rate / (1 - context_rate)
    )
    adjusted_odds = odds * multiplier
    return adjusted_odds / (1 + adjusted_odds)


def priority_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    fraction: float,
) -> dict[str, float]:
    order = np.argsort(-probability)
    ordered_weight = weights[order]
    cutoff = fraction * weights.sum()
    selected_count = int(np.searchsorted(np.cumsum(ordered_weight), cutoff, side="left") + 1)
    selected = order[:selected_count]
    selected_positive_weight = float(np.sum(weights[selected] * y_true[selected]))
    return {
        "precision": selected_positive_weight / float(np.sum(weights[selected])),
        "recall": selected_positive_weight / float(np.sum(weights * y_true)),
    }


def git_state() -> tuple[str | None, bool | None]:
    repository = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repository, text=True
            ).strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def main() -> None:
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")

    columns = [*FEATURES, TARGET]
    context_pool = pd.read_parquet(
        args.data, columns=columns, filters=[(YEAR, "=", args.context_year)]
    )
    test_pool = pd.read_parquet(
        args.data, columns=columns, filters=[(YEAR, "=", args.test_year)]
    )
    X_context, y_context, _ = stratified_sample(
        context_pool,
        args.context_per_class,
        args.context_per_class,
        args.seed,
    )
    X_test, y_test, test_weights = stratified_sample(
        test_pool,
        args.test_positives,
        args.test_negatives,
        args.seed + 10,
    )

    started = time.perf_counter()
    model = tabfm_v1_0_0_pytorch.load(device=args.device)
    loaded = time.perf_counter()
    classifier = TabFMClassifier(
        model=model,
        n_estimators=args.n_estimators,
        max_num_rows=None,
        random_state=args.seed,
        batch_size=1,
        cache_context=True,
        maybe_quantize_kv_cache=False,
        keep_cache_on_device=True,
    )
    classifier.fit(X_context, y_context)
    fitted = time.perf_counter()

    probability_chunks = []
    for start in range(0, len(X_test), args.prediction_batch_size):
        stop = start + args.prediction_batch_size
        probability_chunks.append(classifier.predict_proba(X_test.iloc[start:stop])[:, 1])
    raw_probability = np.concatenate(probability_chunks)
    predicted = time.perf_counter()

    context_rate = float(np.mean(y_context))
    context_population_rate = float(context_pool[TARGET].mean())
    adjusted_probability = adjust_prior(
        raw_probability, context_rate, context_population_rate
    )
    metrics: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_test, adjusted_probability, sample_weight=test_weights)),
        "average_precision": float(
            average_precision_score(y_test, adjusted_probability, sample_weight=test_weights)
        ),
        "brier_prior_adjusted": float(
            brier_score_loss(y_test, adjusted_probability, sample_weight=test_weights)
        ),
        "log_loss_prior_adjusted": float(
            log_loss(y_test, adjusted_probability, sample_weight=test_weights)
        ),
        "weighted_test_prevalence": float(np.average(y_test, weights=test_weights)),
        "weighted_mean_prediction": float(
            np.average(adjusted_probability, weights=test_weights)
        ),
        "top_1_percent": priority_metrics(
            y_test, adjusted_probability, test_weights, 0.01
        ),
        "top_5_percent": priority_metrics(
            y_test, adjusted_probability, test_weights, 0.05
        ),
    }
    revision, dirty = git_state()
    result = {
        "experiment": "TabFM binary next-year crossing risk smoke test",
        "interpretation": (
            "Exploratory only: small balanced in-context sample, case-control test "
            "sample with population weights, and no tuned feature pipeline."
        ),
        "license": "TabFM pretrained weights: non-commercial, non-production use only",
        "model_repository": "google/tabfm-1.0.0-pytorch",
        "data": str(args.data.resolve()),
        "git_revision": revision,
        "git_dirty": dirty,
        "parameters": vars(args) | {"data": str(args.data), "output": str(args.output)},
        "features": FEATURES,
        "rows": {
            "context_pool": len(context_pool),
            "context_sample": len(X_context),
            "context_sample_positives": int(y_context.sum()),
            "test_pool": len(test_pool),
            "test_sample": len(X_test),
            "test_sample_positives": int(y_test.sum()),
        },
        "versions": {
            "tabfm": importlib.metadata.version("tabfm"),
            "torch": torch.__version__,
            "pandas": pd.__version__,
        },
        "device": args.device,
        "timing_seconds": {
            "model_load": loaded - started,
            "fit_and_context_cache": fitted - loaded,
            "prediction": predicted - fitted,
            "total_model": predicted - started,
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
