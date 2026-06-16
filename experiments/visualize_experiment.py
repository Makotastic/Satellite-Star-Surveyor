"""Load a stored closed-loop experiment pickle and launch Ursina visualization.

Run this file after generating experiment data with ``notebooks/experiment.py``:

    python notebooks/visualize_experiment.py experiments/closed_loop_experiment.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

from mpc_spacecraft.visualization import UrsinaSpacecraftVisualizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for experiment visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize closed-loop experiment data stored by notebooks/experiment.py."
    )
    parser.add_argument(
        "data_file",
        type=Path,
        help="Path to the stored experiment pickle file.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Visualization playback frames per second.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Use every Nth log row for visualization.",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Load the visualization paused instead of playing immediately.",
    )
    return parser.parse_args()


def load_experiment_payload(data_file: Path) -> dict[str, Any]:
    """Load and validate an experiment pickle payload."""
    if not data_file.exists():
        raise FileNotFoundError(f"Experiment data file does not exist: {data_file}")

    with data_file.open("rb") as data_handle:
        payload = pickle.load(data_handle)

    if not isinstance(payload, dict):
        raise TypeError("Experiment data file must contain a dictionary payload.")
    if "logs" not in payload:
        raise KeyError("Experiment data payload is missing the required 'logs' entry.")

    return payload


def main() -> None:
    """Load stored experiment logs and launch the spacecraft visualizer."""
    args = parse_args()
    payload = load_experiment_payload(args.data_file)

    logs = payload["logs"]
    targets = payload.get("targets")

    viz = UrsinaSpacecraftVisualizer()
    viz.visualize_closed_loop_dataframe(
        logs,
        fps=args.fps,
        every_n=args.every_n,
        play=not args.paused,
        targets=targets,
    )


if __name__ == "__main__":
    main()
