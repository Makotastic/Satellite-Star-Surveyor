from mpc_spacecraft.utilities.utils import (
    FloatArray,
    RigidBodyState,
    Vec3,
    unit,
    R_EARTH_M,
    BODY_FORWARD_VEC3,
)

from typing import TypeAlias
from enum import IntEnum
import pandas as pd
import numpy as np
import quaternion as qu
from typing import Any, cast

AngleLike: TypeAlias = float | FloatArray
BoolLike: TypeAlias = bool | FloatArray

# Modes
class PlanModes(IntEnum):
    NO_TARGET = 1
    OBSERVING = 2
    SLEWING = 3
    SETTLING = 4


class StarPlanner:
    def __init__(
        self,
        targets: pd.DataFrame,
        sun_keepout_deg: float,
        earth_margin_deg: float,
        stabilizing_threshold_deg: float = 0.5,
        observing_threshold_deg: float = 0.01,
        observing_max_angular_rate: float = 0.01,
    ):
        validate_targets_dataframe(targets)

        self._targets: pd.DataFrame = convert_ECI_pos_to_unit_vec(targets)
        self._targets["obs_time"] = 0.0

        self._sun_keepout = np.deg2rad(sun_keepout_deg)
        self._earth_margin = np.deg2rad(earth_margin_deg)
        self._stabilize_threshold = np.deg2rad(stabilizing_threshold_deg)
        self._observe_threshold = np.deg2rad(observing_threshold_deg)
        self._observe_rate_threshold = np.deg2rad(observing_max_angular_rate)
        self._current_target_idx = None
        self.mode = PlanModes.NO_TARGET

    def _update_target(
        self,
        body_state: RigidBodyState,
        earth_dir_I: Vec3,
        sun_dir_I: Vec3,
        body_dir_I: Vec3,
    ) -> tuple[Vec3 | None, pd.DataFrame]:

        star_vecs = self._targets[["x", "y", "z"]].to_numpy()

        earth_angle, sun_angle, dist_angle = compute_earth_sun_body_angles(
            star_vecs, earth_dir_I, sun_dir_I, body_dir_I
        )

        feasible, earth_ok, sun_ok = self._is_target_feasible(
            earth_angle, sun_angle, body_state
        )

        costs = self._targets.copy()
        costs["earth_angle"] = earth_angle
        costs["sun_angle"] = sun_angle
        costs["dist_angle"] = dist_angle
        costs["earth_ok"] = earth_ok
        costs["sun_ok"] = sun_ok
        costs["feasible"] = feasible
        costs["unfinished"] = costs["req_obs_time"] > costs["obs_time"]
        costs["feasible_unfinished"] = feasible & costs["unfinished"]

        feasible_unfinished = costs[costs["feasible_unfinished"]].copy()
        if feasible_unfinished.empty:
            return None, costs

        feasible_unfinished["score"] = feasible_unfinished["dist_angle"]

        best_idx = feasible_unfinished["score"].idxmin()
        self._current_target_idx = best_idx

        target_row = feasible_unfinished.loc[best_idx]
        target_dir_I = target_row[["x", "y", "z"]].to_numpy()

        return target_dir_I, costs

    def _is_target_feasible(
        self,
        earth_angle: AngleLike,
        sun_angle: AngleLike,
        body_state: RigidBodyState,
    ) -> tuple[BoolLike, BoolLike, BoolLike]:

        pos_I = body_state.position
        r = np.linalg.norm(pos_I)
        earth_occlusion_half_angle = np.arcsin(np.clip(R_EARTH_M / r, -1.0, 1.0))

        earth_ok = earth_angle > (earth_occlusion_half_angle + self._earth_margin)
        sun_ok = sun_angle > self._sun_keepout
        feasible = earth_ok & sun_ok

        return feasible, earth_ok, sun_ok

    def tick(
        self, body_state: RigidBodyState, delta_time: float, sun_dir_I: Vec3
    ) -> tuple[Vec3 | None, PlanModes, bool]:

        if self.mode == PlanModes.OBSERVING:
            if self._current_target_idx is None:
                raise RuntimeError("Planner is observing without an active target")

            obs_time = float(cast(Any, self._targets.at[self._current_target_idx, "obs_time"]))
            self._targets.at[self._current_target_idx, "obs_time"] = obs_time + delta_time

        targets_complete = cast(
            bool, np.all(self._targets["req_obs_time"] <= self._targets["obs_time"])
        )

        earth_dir_I, body_dir_I = compute_earth_body_directions(body_state)

        if self.mode == PlanModes.NO_TARGET:
            star_vec, _ = self._update_target(
                body_state, earth_dir_I, sun_dir_I, body_dir_I
            )
            if star_vec is None:
                self.mode = PlanModes.NO_TARGET
            else:
                self.mode = PlanModes.SLEWING
            return star_vec, self.mode, targets_complete

        target = self._targets.loc[self._current_target_idx]
        star_vec = target[["x", "y", "z"]].to_numpy()

        earth_angle, sun_angle, dist_angle = compute_earth_sun_body_angles(
            star_vec, earth_dir_I, sun_dir_I, body_dir_I
        )

        feasible, _, _ = self._is_target_feasible(earth_angle, sun_angle, body_state)

        unfinished = target["req_obs_time"] > target["obs_time"]
        feasible_unfinished = feasible & unfinished
        switch_target = not feasible_unfinished

        if switch_target:
            star_vec, _ = self._update_target(
                body_state, earth_dir_I, sun_dir_I, body_dir_I
            )
            if star_vec is None:
                self.mode = PlanModes.NO_TARGET
            else:
                self.mode = PlanModes.SLEWING
            return star_vec, self.mode, targets_complete

        mag_omega = np.linalg.norm(body_state.omega)

        if dist_angle > self._stabilize_threshold:
            self.mode = PlanModes.SLEWING

        if dist_angle <= self._stabilize_threshold:
            self.mode = PlanModes.SETTLING

        if (
            dist_angle <= self._observe_threshold
            and mag_omega < self._observe_rate_threshold
        ):
            self.mode = PlanModes.OBSERVING

        return star_vec, self.mode, targets_complete


def compute_earth_sun_body_angles(
    star_vecs: FloatArray,
    earth_dir_I: Vec3,
    sun_dir_I: Vec3,
    body_dir_I: Vec3,
) -> tuple[AngleLike, AngleLike, AngleLike]:

    earth_angle = np.arccos(np.clip(star_vecs @ earth_dir_I, -1.0, 1.0))
    sun_angle = np.arccos(np.clip(star_vecs @ sun_dir_I, -1.0, 1.0))
    dist_angle = np.arccos(np.clip(star_vecs @ body_dir_I, -1.0, 1.0))
    return earth_angle, sun_angle, dist_angle


def compute_earth_body_directions(body_state: RigidBodyState) -> tuple[Vec3, Vec3]:

    pos_I = body_state.position
    earth_dir_I = unit(-pos_I)

    rot_quat = qu.quaternion(*body_state.quat)
    body_dir_I = qu.rotate_vectors(rot_quat, BODY_FORWARD_VEC3)

    return earth_dir_I, body_dir_I


def convert_ECI_pos_to_unit_vec(targets: pd.DataFrame) -> pd.DataFrame:

    unit_targets = targets.assign(
        x=np.cos(targets.Dec) * np.cos(targets.RA),
        y=np.cos(targets.Dec) * np.sin(targets.RA),
        z=np.sin(targets.Dec),
    ).drop(columns=["Dec", "RA"])

    return unit_targets


def validate_targets_dataframe(targets: pd.DataFrame) -> None:
    if not isinstance(targets, pd.DataFrame):
        raise ValueError("targets must be a pandas DataFrame")

    required_columns = {"req_obs_time", "Dec", "RA"}
    missing_columns = required_columns - set(targets.columns)
    if missing_columns:
        missing_sorted = sorted(missing_columns)
        raise ValueError(f"targets is missing required columns: {missing_sorted}")

    for col in sorted(required_columns):
        if not pd.api.types.is_numeric_dtype(targets[col]):
            raise ValueError(f"targets['{col}'] must be numeric")

        if not np.isfinite(targets[col].to_numpy()).all():
            raise ValueError(f"targets['{col}'] contains NaN or infinite values")
