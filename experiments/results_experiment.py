"""Report-oriented closed-loop mission experiment.

This script keeps the original minimal experiment intact and creates a richer
artifact for website/report analysis. It saves the canonical data as compressed
NPZ, plus JSON summary/metadata files and a CSV table for quick inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mpc_spacecraft.analysis import build_mission_analysis_artifact, save_mission_artifact
from mpc_spacecraft.simulation import ClosedLoopTestConfig, run_closed_loop_test


OUTPUT_DIR = Path("./experiments/results")
SAVE_CSV = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-name",
        default="mission_results",
        help="Base filename stem for result artifacts, for example 'mission_results_baseline'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_name = Path(args.result_name).name
    output_dir = OUTPUT_DIR / result_name
    config = ClosedLoopTestConfig.defaults(show_progress=True)
    result = run_closed_loop_test(config)
    logs = result.to_dataframe()

    artifact = build_mission_analysis_artifact(
        logs=logs,
        targets=config.targets.targets,
        config=config,
    )
    paths = save_mission_artifact(artifact, output_dir, stem=result_name, save_csv=SAVE_CSV)

    summary = artifact["summary"]
    print(summary["outcome"])
    print(f"Saved compressed analysis artifact to {paths.npz}")
    print(f"Saved summary metrics to {paths.summary_json}")
    if paths.metadata_json is not None:
        print(f"Saved run metadata to {paths.metadata_json}")
    if paths.timeseries_csv is not None:
        print(f"Saved quick-inspection time series to {paths.timeseries_csv}")


if __name__ == "__main__":
    main()
