"""Run the complete crash-likelihood model panel sequentially."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CODE_ROOT = PACKAGE_ROOT.parent

MODEL_SCRIPTS = {
    "logistic_regression": "train_likelihood_logistic_regression.py",
    "xgboost": "train_likelihood_xgboost.py",
    "isolation_forest": "train_likelihood_isolation_forest.py",
    "mlp": "train_likelihood_mlp.py",
    "ft_transformer": "train_likelihood_ft_transformer.py",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dataset",
    type=Path,
    default=PROJECT_CODE_ROOT / "data" / "final" / "final_dataset.parquet",
)
parser.add_argument("--output-dir", type=Path)
parser.add_argument(
    "--models",
    nargs="+",
    choices=list(MODEL_SCRIPTS),
    default=list(MODEL_SCRIPTS),
)
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--max-epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--mlp-batch-size", type=int, default=16_384)
parser.add_argument("--transformer-batch-size", type=int, default=64)
parser.add_argument("--n-jobs", type=int, default=-1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--xgboost-device", choices=["cpu", "cuda"], default="cpu"
)
parser.add_argument("--require-cuda", action="store_true")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

dataset_path = args.dataset.resolve()
if not dataset_path.is_file():
    raise FileNotFoundError(f"Final dataset not found: {dataset_path}")

if args.output_dir is None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "_smoke" if args.smoke else ""
    args.output_dir = PACKAGE_ROOT / "runs" / f"likelihood_panel_{timestamp}{suffix}"
run_root = args.output_dir.resolve()
if run_root.exists() and any(run_root.iterdir()):
    raise FileExistsError(f"Output directory is not empty: {run_root}")
run_root.mkdir(parents=True, exist_ok=True)

manifest = {
    "task": "incident_likelihood",
    "dataset_path": str(dataset_path),
    "models": args.models,
    "trials_per_model": 1 if args.smoke else args.trials,
    "smoke_run": args.smoke,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "runs": {},
}
manifest_path = run_root / "panel_run.json"
effective_trials = 1 if args.smoke else args.trials
effective_epochs = 2 if args.smoke else args.max_epochs
effective_patience = 2 if args.smoke else args.patience

for model_name in args.models:
    output_dir = run_root / model_name
    command = [
        sys.executable,
        str(PACKAGE_ROOT / "scripts" / MODEL_SCRIPTS[model_name]),
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--trials",
        str(effective_trials),
        "--seed",
        str(args.seed),
    ]
    if model_name in {"logistic_regression", "xgboost", "isolation_forest"}:
        command.extend(["--n-jobs", str(args.n_jobs)])
    if model_name == "xgboost":
        command.extend(["--xgboost-device", args.xgboost_device])
    if model_name == "mlp":
        command.extend(
            [
                "--max-epochs",
                str(effective_epochs),
                "--patience",
                str(effective_patience),
                "--batch-size",
                str(args.mlp_batch_size),
            ]
        )
    if model_name == "ft_transformer":
        command.extend(
            [
                "--max-epochs",
                str(effective_epochs),
                "--patience",
                str(effective_patience),
                "--batch-size",
                str(args.transformer_batch_size),
            ]
        )
    if args.require_cuda and model_name in {"mlp", "ft_transformer"}:
        command.append("--require-cuda")
    if args.smoke:
        command.append("--smoke")

    manifest["runs"][model_name] = {
        "status": "running",
        "output_dir": str(output_dir),
        "command": command,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n=== Training {model_name} ===", flush=True)
    result = subprocess.run(command, check=False)
    manifest["runs"][model_name]["return_code"] = result.returncode
    manifest["runs"][model_name]["status"] = (
        "complete" if result.returncode == 0 else "failed"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if result.returncode != 0:
        raise RuntimeError(
            f"{model_name} failed with exit code {result.returncode}. "
            f"See the console output and {manifest_path}."
        )

manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
manifest_path.write_text(json.dumps(manifest, indent=2))

comparison_rows = []
for model_name in args.models:
    metadata = json.loads((run_root / model_name / "run.json").read_text())
    comparison_rows.append(
        {
            "model": model_name,
            "mean_validation_average_precision": metadata[
                "mean_validation_average_precision"
            ],
            "test_prevalence": metadata["test_metrics"]["prevalence"],
            "test_average_precision": metadata["test_metrics"][
                "average_precision"
            ],
        }
    )
with (run_root / "model_comparison.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0]))
    writer.writeheader()
    writer.writerows(comparison_rows)

print("\nAll requested incident-likelihood models completed:", run_root)
