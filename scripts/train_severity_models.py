"""Run the matched classification-versus-regression severity panel."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from _severity_search import write_search_spaces

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CODE_ROOT = PACKAGE_ROOT.parent

RUNS = [
    ("classification", "logistic_regression"),
    ("classification", "xgboost"),
    ("classification", "mlp"),
    ("classification", "ft_transformer"),
    ("regression", "ridge"),
    ("regression", "xgboost"),
    ("regression", "mlp"),
    ("regression", "ft_transformer"),
]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dataset",
    type=Path,
    default=PROJECT_CODE_ROOT / "data" / "final" / "final_dataset.parquet",
)
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--trials", type=int, default=50)
parser.add_argument("--max-epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--mlp-batch-size", type=int, default=256)
parser.add_argument("--transformer-batch-size", type=int, default=64)
parser.add_argument("--n-jobs", type=int, default=-1)
parser.add_argument(
    "--xgboost-device", choices=["cpu", "cuda"], default="cpu"
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--require-cuda", action="store_true")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

dataset_path = args.dataset.resolve()
if not dataset_path.is_file():
    raise FileNotFoundError(f"Final dataset not found: {dataset_path}")
if args.output_dir is None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "_smoke" if args.smoke else ""
    args.output_dir = (
        PACKAGE_ROOT / "runs" / f"severity_comparison_{timestamp}{suffix}"
    )
run_root = args.output_dir.resolve()
if run_root.exists() and any(run_root.iterdir()):
    raise FileExistsError(f"Output directory is not empty: {run_root}")
run_root.mkdir(parents=True, exist_ok=True)
write_search_spaces(run_root / "hyperparameter_search_spaces.csv")

manifest = {
    "task": "conditional_severity_classification_vs_regression",
    "dataset_path": str(dataset_path),
    "trials_per_run": 1 if args.smoke else args.trials,
    "runs": {},
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
}
manifest_path = run_root / "panel_run.json"

for formulation, model in RUNS:
    run_name = f"{formulation}_{model}"
    script = f"train_severity_{formulation}.py"
    command = [
        sys.executable,
        str(PACKAGE_ROOT / "scripts" / script),
        "--model",
        model,
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(run_root / run_name),
        "--trials",
        str(args.trials),
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--n-jobs",
        str(args.n_jobs),
        "--xgboost-device",
        args.xgboost_device,
        "--seed",
        str(args.seed),
    ]
    if model == "mlp":
        command.extend(["--batch-size", str(args.mlp_batch_size)])
    if model == "ft_transformer":
        command.extend(["--batch-size", str(args.transformer_batch_size)])
    if args.require_cuda and model in {"mlp", "ft_transformer"}:
        command.append("--require-cuda")
    if args.smoke:
        command.append("--smoke")

    manifest["runs"][run_name] = {
        "status": "running",
        "command": command,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n=== {run_name} ===", flush=True)
    environment = os.environ.copy()
    if args.n_jobs == 1:
        environment["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(command, check=False, env=environment)
    manifest["runs"][run_name]["return_code"] = result.returncode
    manifest["runs"][run_name]["status"] = (
        "complete" if result.returncode == 0 else "failed"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if result.returncode != 0:
        raise RuntimeError(f"{run_name} failed with code {result.returncode}.")

manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
manifest_path.write_text(json.dumps(manifest, indent=2))


def budget_values(path: Path, budget: int) -> dict[str, str] | None:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return next(
        (
            row
            for row in rows
            if int(row["severity_review_budget_incidents"]) == budget
        ),
        None,
    )


comparison_rows = []
for formulation, model in RUNS:
    run_name = f"{formulation}_{model}"
    run_dir = run_root / run_name
    metadata = json.loads((run_dir / "run.json").read_text())
    if formulation == "classification":
        validation_metric = "multiclass_log_loss"
        validation_score = metadata["mean_validation_log_loss"]
        test_score = metadata["test_metrics"]["multiclass_log_loss"]
        continuous_mae = metadata["test_continuous_severity_mae"]
    else:
        validation_metric = "mean_absolute_error"
        validation_score = metadata["mean_validation_mae"]
        test_score = metadata["test_metrics"]["mean_absolute_error"]
        continuous_mae = test_score

    row = {
        "formulation": formulation,
        "model": model,
        "validation_metric": validation_metric,
        "mean_validation_score": validation_score,
        "native_test_score": test_score,
        "test_continuous_severity_mae": continuous_mae,
    }
    for budget in (50, 100, 500, 1_000):
        values = budget_values(run_dir / "severity_budget.csv", budget)
        if values is None:
            row[f"fatal_captured_at_{budget}"] = ""
            row[f"serious_captured_at_{budget}"] = ""
            row[f"severity_lift_at_{budget}"] = ""
            continue
        row[f"fatal_captured_at_{budget}"] = values[
            "fatal_incidents_captured"
        ]
        row[f"serious_captured_at_{budget}"] = values[
            "injury_or_fatal_incidents_captured"
        ]
        row[f"severity_lift_at_{budget}"] = values[
            "severity_lift_over_all_incidents"
        ]
    comparison_rows.append(row)

with (run_root / "model_comparison.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0]))
    writer.writeheader()
    writer.writerows(comparison_rows)

print("\nAll severity comparison runs completed:", run_root)
