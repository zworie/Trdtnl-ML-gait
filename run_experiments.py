"""
Batch Experiment Runner for the Gait Analysis ML Pipeline
==========================================================
Runs multiple experiments with different settings and saves each result
in a separate timestamped folder under ``experiments/``.

Usage
-----
  # Run all experiments defined in EXPERIMENTS list below
  python run_experiments.py

  # Dry run — print commands without executing
  python run_experiments.py --dry-run

  # Run only experiments whose name contains a substring
  python run_experiments.py --filter smote

  # Custom base output directory
  python run_experiments.py --base-dir results

Edit the EXPERIMENTS list below to define your own experiment configurations.
Each experiment is a dict with any subset of the pipeline's CLI options.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── Define your experiments here ────────────────────────────────────────────
#
# Each dict can contain any of these keys (all optional — omitted = pipeline default):
#
#   name             str   Short label used in the folder name (required)
#   smote            bool  Use SMOTE oversampling (default False)
#   hpo              str   'optuna' or 'grid' (default 'optuna')
#   n_outer          int   Outer CV folds (default 5)
#   n_inner          int   Inner CV folds (default 5)
#   n_trials         int   Optuna trials per model per fold (default 50)
#   features         str   Path to feature CSV (default gait_features_rich.csv)
#   selected_features list  Exact feature names — skips Boruta when provided
#

THESIS_FEATURES = [
    "LHip_min", "RKnee_skew", "RAnkle_skew", "LAnkle_kurt",
    "LDP_cv", "TrunkY_max", "CoM_Y_min", "StepLength",
]

EXPERIMENTS = [
    # ── Baseline experiments ─────────────────────────────────────────────
    {
        "name": "baseline_optuna",
        "hpo": "optuna",
    },
    {
        "name": "baseline_grid",
        "hpo": "grid",
    },

    # ── SMOTE experiments ────────────────────────────────────────────────
    {
        "name": "smote_optuna",
        "smote": True,
        "hpo": "optuna",
    },
    {
        "name": "smote_grid",
        "smote": True,
        "hpo": "grid",
    },

    # ── Thesis features (skip Boruta) ────────────────────────────────────
    {
        "name": "thesis_features_optuna",
        "selected_features": THESIS_FEATURES,
        "hpo": "optuna",
    },
    {
        "name": "thesis_features_grid",
        "selected_features": THESIS_FEATURES,
        "hpo": "grid",
    },
    {
        "name": "thesis_features_smote_optuna",
        "selected_features": THESIS_FEATURES,
        "smote": True,
        "hpo": "optuna",
    },

    # ── Fold count variations ────────────────────────────────────────────
    {
        "name": "cv10x5_optuna",
        "n_outer": 10,
        "n_inner": 5,
        "hpo": "optuna",
    },
    {
        "name": "cv3x3_quick",
        "n_outer": 3,
        "n_inner": 3,
        "n_trials": 10,
        "hpo": "optuna",
    },

    # ── Trial count variations ───────────────────────────────────────────
    {
        "name": "optuna_100trials",
        "n_trials": 100,
        "hpo": "optuna",
    },
]

# ─── End of experiment definitions ───────────────────────────────────────────


def build_folder_name(experiment: dict, timestamp: str) -> str:
    """Build a descriptive folder name from experiment settings and timestamp.

    Format: YYYY-MM-DD_HHMMSS_<name>
    """
    return f"{timestamp}_{experiment['name']}"


def build_command(experiment: dict, out_dir: str, log_path: str) -> list[str]:
    """Build the CLI command list for a single experiment."""
    cmd = [sys.executable, "pipeline/ml_pipeline.py", "--out", out_dir]

    if experiment.get("smote"):
        cmd.append("--smote")

    if "hpo" in experiment:
        cmd.extend(["--hpo", experiment["hpo"]])

    if "n_outer" in experiment:
        cmd.extend(["--n-outer", str(experiment["n_outer"])])

    if "n_inner" in experiment:
        cmd.extend(["--n-inner", str(experiment["n_inner"])])

    if "n_trials" in experiment:
        cmd.extend(["--n-trials", str(experiment["n_trials"])])

    if "features" in experiment:
        cmd.extend(["--features", experiment["features"]])

    if "selected_features" in experiment:
        cmd.extend(["--selected-features"] + experiment["selected_features"])

    cmd.extend(["--log", log_path])

    return cmd


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def write_experiment_info(out_dir: str, experiment: dict, cmd: list[str]):
    """Write a metadata file describing the experiment settings."""
    info_path = Path(out_dir) / "experiment_info.txt"
    lines = [
        f"Experiment: {experiment['name']}",
        f"Timestamp:  {datetime.now().isoformat()}",
        f"Command:    {' '.join(cmd)}",
        "",
        "Settings:",
    ]
    for key, val in sorted(experiment.items()):
        if key == "name":
            continue
        lines.append(f"  {key}: {val}")

    # Show defaults for omitted settings
    defaults = {
        "smote": False, "hpo": "optuna", "n_outer": 5,
        "n_inner": 5, "n_trials": 50,
    }
    lines.append("")
    lines.append("Effective settings (including defaults):")
    for key, default in defaults.items():
        val = experiment.get(key, default)
        source = "explicit" if key in experiment else "default"
        lines.append(f"  {key}: {val}  ({source})")

    if "selected_features" in experiment:
        lines.append(f"  selected_features: {experiment['selected_features']}  (explicit)")
    else:
        lines.append(f"  selected_features: Boruta  (default)")

    info_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for gait ML pipeline experiments."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only run experiments whose name contains this substring.",
    )
    parser.add_argument(
        "--base-dir", type=str, default="experiments",
        help="Base directory for experiment outputs (default: experiments/).",
    )
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.filter:
        experiments = [e for e in experiments if args.filter in e["name"]]
        if not experiments:
            print(f"No experiments match filter '{args.filter}'.")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    total = len(experiments)

    print(f"{'=' * 60}")
    print(f"Batch Experiment Runner")
    print(f"{'=' * 60}")
    print(f"Date:        {timestamp}")
    print(f"Experiments: {total}")
    print(f"Base dir:    {args.base_dir}/")
    if args.dry_run:
        print(f"Mode:        DRY RUN (no commands will be executed)")
    print(f"{'=' * 60}")
    print()

    results = []
    batch_start = time.time()

    for i, experiment in enumerate(experiments, 1):
        folder = build_folder_name(experiment, timestamp)
        out_dir = str(Path(args.base_dir) / folder)
        log_path = str(Path(out_dir) / "run.log")
        cmd = build_command(experiment, out_dir, log_path)

        print(f"[{i}/{total}] {experiment['name']}")
        print(f"  Output: {out_dir}")
        print(f"  Command: {' '.join(cmd)}")

        if args.dry_run:
            print(f"  Status: SKIPPED (dry run)")
            print()
            continue

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        write_experiment_info(out_dir, experiment, cmd)

        start = time.time()
        try:
            result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
            elapsed = time.time() - start
            status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        except Exception as exc:
            elapsed = time.time() - start
            status = f"ERROR: {exc}"

        results.append((experiment["name"], status, elapsed))
        print(f"  Status: {status}  ({format_duration(elapsed)})")
        print()

    if args.dry_run:
        return

    # Print summary
    batch_elapsed = time.time() - batch_start
    print(f"{'=' * 60}")
    print(f"BATCH COMPLETE — {format_duration(batch_elapsed)} total")
    print(f"{'=' * 60}")
    print(f"{'Experiment':<35} {'Status':<20} {'Time':>10}")
    print(f"{'-' * 35} {'-' * 20} {'-' * 10}")
    for name, status, elapsed in results:
        print(f"{name:<35} {status:<20} {format_duration(elapsed):>10}")
    print()

    # Write summary to base dir
    summary_path = Path(args.base_dir) / f"{timestamp}_summary.txt"
    lines = [
        f"Batch run: {timestamp}",
        f"Total time: {format_duration(batch_elapsed)}",
        "",
        f"{'Experiment':<35} {'Status':<20} {'Time':>10}",
        f"{'-' * 35} {'-' * 20} {'-' * 10}",
    ]
    for name, status, elapsed in results:
        lines.append(f"{name:<35} {status:<20} {format_duration(elapsed):>10}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
