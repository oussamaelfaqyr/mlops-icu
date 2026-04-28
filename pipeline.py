import subprocess
import sys
import os


def run_step(command, description):
    print(f"\n[PIPELINE] Running Step: {description}...")
    try:
        # We use a context that handles the directory changes correctly
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Logic failed in {description}:")
        print(e.stderr)
        sys.exit(1)


import argparse


def main():
    parser = argparse.ArgumentParser(
        description="--- ICU MLOps Pipeline Orchestrator ---"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["data", "features", "train", "monitor", "test", "all"],
        default="all",
        help="Pipeline step to run",
    )

    args = parser.parse_args()

    # Define steps mapping
    steps = {
        "data": (
            "python src/data_processing.py",
            "Data Processing (Loading & Cleaning)",
        ),
        "features": (
            "python src/features.py",
            "Feature Engineering (Sliding Window & Labeling)",
        ),
        "validate": (
            "python src/data_validation.py",
            "Data Consistency & Range Validation",
        ),
        "train": ("python src/train.py", "Model Training & MLflow Tracking"),
        "monitor": ("python src/monitoring.py", "Final Drift Check"),
        "test": ("pytest tests/", "Automated Tests (QA Layer)"),
    }

    print("--- ICU MLOps Pipeline Orchestrator ---")

    if args.step == "all":
        for step_key in ["data", "features", "validate", "train", "monitor", "test"]:
            cmd, desc = steps[step_key]
            run_step(cmd, desc)
    else:
        cmd, desc = steps[args.step]
        run_step(cmd, desc)

    print(f"\n[SUCCESS] Pipeline step '{args.step}' completed successfully!")


if __name__ == "__main__":
    main()
