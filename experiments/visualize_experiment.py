"""Load stored closed-loop experiment outputs and launch Ursina visualization.

Run this file after generating experiment data with ``experiments/experiment.py``
or ``experiments/results_experiment.py``:

    python experiments/visualize_experiment.py experiments/closed_loop_experiment.pkl
    python experiments/visualize_experiment.py experiments/results/mission_results/mission_results_timeseries.csv
    python experiments/visualize_experiment.py experiments/results/mission_results/mission_results.npz
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mpc_spacecraft.visualization import UrsinaSpacecraftVisualizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for experiment visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize closed-loop experiment data from experiments/experiment.py or results_experiment.py."
    )
    parser.add_argument(
        "data_file",
        type=Path,
        help=(
            "Path to an experiment output: pickle payload, results_experiment CSV/NPZ, "
            "or a results directory containing one of those files."
        ),
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
    parser.add_argument(
        "--antialias-samples",
        type=int,
        default=4,
        help="Number of multisample anti-aliasing samples for the Ursina window; use 0 to disable.",
    )
    return parser.parse_args()


def load_experiment_payload(data_file: Path) -> dict[str, Any]:
    """Load and validate an experiment payload from pickle, CSV, NPZ, or directory."""
    if not data_file.exists():
        raise FileNotFoundError(f"Experiment data file does not exist: {data_file}")

    if data_file.is_dir():
        data_file = _select_result_file(data_file)

    suffix = data_file.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return _load_pickle_payload(data_file)
    if suffix == ".csv":
        return _load_csv_payload(data_file)
    if suffix == ".npz":
        return _load_npz_payload(data_file)
    if suffix == ".json":
        return _load_json_sidecar_payload(data_file)

    raise ValueError(
        f"Unsupported experiment data file type '{data_file.suffix}'. "
        "Expected .pkl, .pickle, .csv, .npz, .json, or a results directory."
    )


def _load_pickle_payload(data_file: Path) -> dict[str, Any]:
    """Load the legacy pickle payload produced by experiments/experiment.py."""
    with data_file.open("rb") as data_handle:
        payload = pickle.load(data_handle)

    if not isinstance(payload, dict):
        raise TypeError("Experiment data file must contain a dictionary payload.")
    if "logs" not in payload:
        raise KeyError("Experiment data payload is missing the required 'logs' entry.")

    return payload


def _load_csv_payload(data_file: Path) -> dict[str, Any]:
    """Load a results_experiment quick-inspection time-series CSV."""
    logs = pd.read_csv(data_file)
    return {"logs": logs, "targets": _try_load_companion_targets(data_file, row_count=len(logs))}


def _load_npz_payload(data_file: Path) -> dict[str, Any]:
    """Load the compressed analysis artifact produced by results_experiment.py."""
    with np.load(data_file, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}

    log_columns = _log_columns_from_npz(arrays)
    if not log_columns:
        raise KeyError(
            "NPZ artifact does not contain visualizable closed-loop log columns "
            "such as true_position_x and true_quat_w."
        )

    logs = pd.DataFrame(log_columns)
    targets = _targets_from_npz(arrays, row_count=len(logs))
    return {"logs": logs, "targets": targets}


def _load_json_sidecar_payload(data_file: Path) -> dict[str, Any]:
    """Resolve a results_experiment JSON sidecar to its sibling CSV or NPZ artifact."""
    sibling = _select_result_file(data_file.parent, stem_hint=_result_stem_from_sidecar(data_file))
    if sibling == data_file:
        raise ValueError(
            f"JSON sidecar {data_file} only contains summary/metadata and cannot be visualized directly. "
            "Pass the sibling *_timeseries.csv, .npz, or the containing results directory."
        )
    return load_experiment_payload(sibling)


def _select_result_file(directory: Path, stem_hint: str | None = None) -> Path:
    """Pick the best visualizable artifact from a results_experiment output directory."""
    candidates: list[Path] = []
    if stem_hint is not None:
        candidates.extend(
            [
                directory / f"{stem_hint}_timeseries.csv",
                directory / f"{stem_hint}.npz",
                directory / f"{stem_hint}.pkl",
                directory / f"{stem_hint}.pickle",
            ]
        )
    candidates.extend(sorted(directory.glob("*_timeseries.csv")))
    candidates.extend(sorted(directory.glob("*.npz")))
    candidates.extend(sorted(directory.glob("*.pkl")))
    candidates.extend(sorted(directory.glob("*.pickle")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No visualizable experiment artifact found in {directory}. "
        "Expected *_timeseries.csv, *.npz, *.pkl, or *.pickle."
    )


def _try_load_companion_targets(data_file: Path, row_count: int) -> pd.DataFrame | None:
    """Load target rows from the sibling NPZ when a CSV path is provided."""
    stem = data_file.stem.removesuffix("_timeseries")
    npz_path = data_file.with_name(f"{stem}.npz")
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=False) as archive:
        return _targets_from_npz({key: archive[key] for key in archive.files}, row_count=row_count)


def _log_columns_from_npz(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return 1D numeric arrays that correspond to closed-loop log columns."""
    ignored_prefixes = ("derived_", "target_")
    ignored_keys = {"summary_json", "metadata_json", "schema_version"}
    lengths = [value.shape[0] for value in arrays.values() if value.ndim >= 1 and value.shape != ()]
    if not lengths:
        return {}
    row_count = max(set(lengths), key=lengths.count)
    return {
        key: value
        for key, value in arrays.items()
        if key not in ignored_keys
        and not key.startswith(ignored_prefixes)
        and value.ndim == 1
        and value.shape[0] == row_count
    }


def _targets_from_npz(arrays: dict[str, np.ndarray], row_count: int | None = None) -> pd.DataFrame | None:
    """Reconstruct the numeric target table saved into a results_experiment NPZ."""
    candidate_columns = {
        key.removeprefix("target_"): np.asarray(value)
        for key, value in arrays.items()
        if key.startswith("target_")
        and value.ndim == 1
        and not _is_per_timestep_target_log(key, value, row_count)
    }
    if not candidate_columns:
        return None
    target_length = _most_common_length(candidate_columns.values())
    target_columns = {
        key: value for key, value in candidate_columns.items() if value.shape[0] == target_length
    }
    return pd.DataFrame(target_columns)


def _is_per_timestep_target_log(key: str, value: np.ndarray, row_count: int | None) -> bool:
    """Return True for log columns like target_0_obs_time saved alongside target metadata."""
    if row_count is None or value.shape[0] != row_count:
        return False
    return key.removeprefix("target_").split("_", maxsplit=1)[0].isdigit()


def _most_common_length(values: Any) -> int:
    """Return the most common first-axis length from an iterable of arrays."""
    lengths = [value.shape[0] for value in values]
    return max(set(lengths), key=lengths.count)


def _result_stem_from_sidecar(data_file: Path) -> str:
    """Map a results_experiment JSON sidecar name back to the artifact stem."""
    stem = data_file.stem
    for suffix in ("_summary", "_metadata"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def main() -> None:
    """Load stored experiment logs and launch the spacecraft visualizer."""
    args = parse_args()
    payload = load_experiment_payload(args.data_file)

    logs = payload["logs"]
    targets = payload.get("targets")

    viz = UrsinaSpacecraftVisualizer(antialias_samples=args.antialias_samples)
    viz.visualize_closed_loop_dataframe(
        logs,
        fps=args.fps,
        every_n=args.every_n,
        play=not args.paused,
        targets=targets,
    )


if __name__ == "__main__":
    main()
