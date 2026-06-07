"""Ursina-based 3D visualization for closed-loop spacecraft results.

The visualizer consumes the outputs produced by
``ClosedLoopTestResult.to_dataframe()`` or ``ClosedLoopTestResult.to_arrays()``
from :mod:`mpc_spacecraft.simulation.closed_loop_testing` and displays an
Earth-centered scene containing Earth, the Sun direction, the satellite body,
its orbit trail, and a satellite-relative orbit camera.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from mpc_spacecraft.guidance import AstropySunDirectionModel
from mpc_spacecraft.utilities.utils import BODY_FORWARD_VEC3, R_EARTH_M


DEFAULT_POSITION_SCALE = 1.0e-6
SPACECRAFT_VIS_SIZE = 0.45
SUN_VIS_DISTANCE = 45.0
SUN_VIS_RADIUS = 2.0
CAMERA_DEFAULT_DISTANCE = 12.0
CAMERA_MIN_DISTANCE = 2.0
CAMERA_MAX_DISTANCE = 80.0


@dataclass(frozen=True)
class ClosedLoopVizData:
    """Normalized closed-loop arrays used by the Ursina visualizer."""

    positions_m: np.ndarray
    quaternions_wxyz: np.ndarray
    times_s: np.ndarray | None = None
    epochs_utc: np.ndarray | None = None
    goal_quaternions_wxyz: np.ndarray | None = None


class UrsinaSpacecraftVisualizer:
    """Interactive 3D visualization for spacecraft closed-loop simulations.

    Controls:
        Space: Play/pause.
        Left/Right: Step one frame backward/forward.
        PageUp/PageDown: Skip ahead/back by ``skip_frames``.
        Home/End: Jump to first/last frame.
        Mouse drag: Orbit camera around the satellite.
        Mouse wheel: Zoom in/out while staying satellite-relative.
        +/-: Increase/decrease playback speed.
        R: Reset camera orbit.
    """

    def __init__(
        self,
        position_scale: float = DEFAULT_POSITION_SCALE,
        playback_speed: float = 1.0,
        skip_frames: int = 25,
        window_title: str = "MPC Spacecraft Ursina Visualizer",
    ) -> None:
        self.position_scale = float(position_scale)
        self.playback_speed = float(playback_speed)
        self.skip_frames = max(1, int(skip_frames))
        self.window_title = window_title

        self._data: ClosedLoopVizData | None = None
        self._frame_index = 0
        self._playing = True
        self._frame_accumulator = 0.0
        self._fps = 30.0
        self._sun_model = SkyfieldSunDirectionModel()

        self._app: Any = None
        self._entities: dict[str, Any] = {}
        self._camera_target = np.zeros(3, dtype=float)
        self._camera_yaw_deg = 35.0
        self._camera_pitch_deg = 22.0
        self._camera_distance = CAMERA_DEFAULT_DISTANCE

    def visualize_closed_loop_dataframe(
        self,
        logs: Any,
        fps: float = 30.0,
        every_n: int = 1,
        play: bool = True,
        run: bool = True,
    ) -> "UrsinaSpacecraftVisualizer":
        """Load closed-loop dataframe logs and optionally start the Ursina app."""
        self.load_closed_loop_dataframe(logs, fps=fps, every_n=every_n, play=play)
        if run:
            self.run()
        return self

    def visualize_closed_loop_arrays(
        self,
        arrays: Mapping[str, Any],
        fps: float = 30.0,
        every_n: int = 1,
        play: bool = True,
        run: bool = True,
    ) -> "UrsinaSpacecraftVisualizer":
        """Load closed-loop arrays and optionally start the Ursina app."""
        self.load_closed_loop_arrays(arrays, fps=fps, every_n=every_n, play=play)
        if run:
            self.run()
        return self

    def load_closed_loop_dataframe(
        self,
        logs: Any,
        fps: float = 30.0,
        every_n: int = 1,
        play: bool = True,
    ) -> None:
        """Normalize dataframe logs produced by ``ClosedLoopTestResult.to_dataframe()``."""
        self._set_data(_downsample_data(_closed_loop_dataframe_to_data(logs), every_n), fps, play)

    def load_closed_loop_arrays(
        self,
        arrays: Mapping[str, Any],
        fps: float = 30.0,
        every_n: int = 1,
        play: bool = True,
    ) -> None:
        """Normalize dictionary logs produced by ``ClosedLoopTestResult.to_arrays()``."""
        self._set_data(_downsample_data(_closed_loop_arrays_to_data(arrays), every_n), fps, play)

    def run(self) -> None:
        """Create the Ursina scene and start the interactive render loop."""
        if self._data is None:
            raise RuntimeError("Load closed-loop data before calling run().")

        from panda3d.core import loadPrcFileData

        loadPrcFileData("", "audio-library-name null")
        from ursina import Ursina, application, camera, window

        try:
            self._app = Ursina(title=self.window_title)
        except Exception as exc:
            raise RuntimeError(
                "Ursina/Panda3D could not open a graphics window. In Docker, "
                "the container needs OpenGL/X11 libraries plus access to the host "
                "display, for example DISPLAY and /tmp/.X11-unix. Rebuild the "
                "devcontainer after Dockerfile changes and run `xhost +local:root` "
                "on the host if X11 rejects the connection."
            ) from exc
        window.color = (0.01, 0.01, 0.03, 1.0)
        application.base.disableMouse()
        self._setup_scene()
        self._apply_frame(0)

        def update() -> None:
            self._update()

        def input(key: str) -> None:
            self._input(key)

        # Ursina discovers module globals named update/input. Attach them to the
        # application object as well so notebooks and repeated imports retain the
        # bound methods for this visualizer instance.
        self._app.update = update
        self._app.input = input
        globals()["update"] = update
        globals()["input"] = input
        camera.clip_plane_far = 100000
        self._app.run()

    def _set_data(self, data: ClosedLoopVizData, fps: float, play: bool) -> None:
        self._data = data
        self._fps = max(1.0, float(fps))
        self._playing = bool(play)
        self._frame_index = 0
        self._frame_accumulator = 0.0

    def _setup_scene(self) -> None:
        from ursina import AmbientLight, Color, DirectionalLight, Entity, Mesh, Text, Vec3, color

        assert self._data is not None
        earth_radius = R_EARTH_M * self.position_scale
        positions = self._scaled_positions()

        self._entities["earth"] = Entity(
            model="sphere",
            scale=earth_radius * 2.0,
            color=color.rgb(32, 87, 190),
        )
        self._entities["sun"] = Entity(
            model="sphere",
            scale=SUN_VIS_RADIUS,
            color=color.rgb(255, 190, 40),
        )
        self._entities["sun_ray"] = Entity(model=Mesh(vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0)], mode="line"), color=color.yellow)

        self._entities["spacecraft"] = Entity()
        self._build_spacecraft_entity(self._entities["spacecraft"])
        self._entities["goal_frame"] = Entity(enabled=self._data.goal_quaternions_wxyz is not None)
        self._build_frame(self._entities["goal_frame"], scale=0.9, alpha=0.45)

        trail_vertices = [Vec3(float(p[0]), float(p[1]), float(p[2])) for p in positions]
        self._entities["orbit_trail"] = Entity(
            model=Mesh(vertices=trail_vertices, mode="line"),
            color=color.rgba(255, 80, 80, 180),
        )
        self._entities["hud"] = Text(
            text="",
            origin=(-0.5, 0.5),
            position=(-0.86, 0.46),
            scale=0.8,
            color=Color(0.9, 0.95, 1.0, 1.0),
        )

        AmbientLight(color=Color(0.35, 0.35, 0.4, 1.0))
        sun_light = DirectionalLight()
        sun_light.look_at(Vec3(-1, -1, -1))
        self._entities["sun_light"] = sun_light

    def _build_spacecraft_entity(self, parent: Any) -> None:
        from ursina import Entity, color

        Entity(parent=parent, model="cube", scale=(SPACECRAFT_VIS_SIZE, SPACECRAFT_VIS_SIZE * 0.55, SPACECRAFT_VIS_SIZE * 0.35), color=color.gray)
        Entity(parent=parent, model="cube", x=SPACECRAFT_VIS_SIZE * 0.95, scale=(SPACECRAFT_VIS_SIZE * 0.45, SPACECRAFT_VIS_SIZE * 0.18, SPACECRAFT_VIS_SIZE * 0.18), color=color.red)
        Entity(parent=parent, model="cube", y=SPACECRAFT_VIS_SIZE * 0.9, scale=(SPACECRAFT_VIS_SIZE * 0.12, SPACECRAFT_VIS_SIZE * 1.3, SPACECRAFT_VIS_SIZE * 0.45), color=color.azure)
        Entity(parent=parent, model="cube", y=-SPACECRAFT_VIS_SIZE * 0.9, scale=(SPACECRAFT_VIS_SIZE * 0.12, SPACECRAFT_VIS_SIZE * 1.3, SPACECRAFT_VIS_SIZE * 0.45), color=color.azure)
        Entity(parent=parent, model="cube", x=SPACECRAFT_VIS_SIZE * 1.65, scale=(SPACECRAFT_VIS_SIZE * 1.8, SPACECRAFT_VIS_SIZE * 0.05, SPACECRAFT_VIS_SIZE * 0.05), color=color.red)
        self._build_frame(parent, scale=0.75, alpha=1.0)

    def _build_frame(self, parent: Any, scale: float, alpha: float) -> None:
        from ursina import Entity, color

        Entity(parent=parent, model="cube", x=scale / 2, scale=(scale, 0.025, 0.025), color=color.rgba(255, 0, 0, int(255 * alpha)))
        Entity(parent=parent, model="cube", y=scale / 2, scale=(0.025, scale, 0.025), color=color.rgba(0, 255, 0, int(255 * alpha)))
        Entity(parent=parent, model="cube", z=scale / 2, scale=(0.025, 0.025, scale), color=color.rgba(0, 90, 255, int(255 * alpha)))

    def _update(self) -> None:
        from ursina import held_keys, mouse, time

        if self._playing and self._data is not None:
            self._frame_accumulator += time.dt * self._fps * self.playback_speed
            if self._frame_accumulator >= 1.0:
                step = int(self._frame_accumulator)
                self._frame_accumulator -= step
                self._apply_frame(min(self._frame_index + step, self.frame_count - 1))
                if self._frame_index >= self.frame_count - 1:
                    self._playing = False

        if held_keys["left mouse"] or held_keys["right mouse"]:
            self._camera_yaw_deg += mouse.velocity[0] * 160.0
            self._camera_pitch_deg = float(np.clip(self._camera_pitch_deg - mouse.velocity[1] * 160.0, -85.0, 85.0))
        self._update_camera()

    def _input(self, key: str) -> None:
        if key == "space":
            self._playing = not self._playing
        elif key == "right arrow":
            self._playing = False
            self._apply_frame(self._frame_index + 1)
        elif key == "left arrow":
            self._playing = False
            self._apply_frame(self._frame_index - 1)
        elif key == "page up":
            self._apply_frame(self._frame_index + self.skip_frames)
        elif key == "page down":
            self._apply_frame(self._frame_index - self.skip_frames)
        elif key == "home":
            self._apply_frame(0)
        elif key == "end":
            self._apply_frame(self.frame_count - 1)
        elif key == "scroll up":
            self._camera_distance = max(CAMERA_MIN_DISTANCE, self._camera_distance * 0.9)
        elif key == "scroll down":
            self._camera_distance = min(CAMERA_MAX_DISTANCE, self._camera_distance * 1.1)
        elif key in {"+", "=", "numpad +"}:
            self.playback_speed *= 1.25
        elif key in {"-", "numpad -"}:
            self.playback_speed = max(0.1, self.playback_speed / 1.25)
        elif key == "r":
            self._camera_yaw_deg = 35.0
            self._camera_pitch_deg = 22.0
            self._camera_distance = CAMERA_DEFAULT_DISTANCE

    def _apply_frame(self, frame_index: int) -> None:
        from ursina import Vec3

        assert self._data is not None
        self._frame_index = int(np.clip(frame_index, 0, self.frame_count - 1))
        pos = self._scaled_positions()[self._frame_index]
        quat = self._data.quaternions_wxyz[self._frame_index]
        self._camera_target = pos

        spacecraft = self._entities["spacecraft"]
        spacecraft.position = Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
        spacecraft.rotation = _ursina_euler_from_quat_wxyz(quat)

        if self._data.goal_quaternions_wxyz is not None:
            goal = self._entities["goal_frame"]
            goal.position = spacecraft.position
            goal.rotation = _ursina_euler_from_quat_wxyz(self._data.goal_quaternions_wxyz[self._frame_index])

        sun_dir = self._sun_direction_for_frame(self._frame_index)
        self._set_sun(sun_dir)
        self._update_hud()

    def _set_sun(self, sun_dir: np.ndarray) -> None:
        from ursina import Mesh, Vec3

        sun_pos = _unit(sun_dir) * SUN_VIS_DISTANCE
        self._entities["sun"].position = Vec3(float(sun_pos[0]), float(sun_pos[1]), float(sun_pos[2]))
        self._entities["sun_ray"].model = Mesh(vertices=[Vec3(0, 0, 0), Vec3(float(sun_pos[0]), float(sun_pos[1]), float(sun_pos[2]))], mode="line")

    def _update_camera(self) -> None:
        from ursina import Vec3, camera

        yaw = np.deg2rad(self._camera_yaw_deg)
        pitch = np.deg2rad(self._camera_pitch_deg)
        direction = np.array(
            [
                np.cos(pitch) * np.cos(yaw),
                np.sin(pitch),
                np.cos(pitch) * np.sin(yaw),
            ],
            dtype=float,
        )
        cam_pos = self._camera_target + direction * self._camera_distance
        camera.position = Vec3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        camera.look_at(Vec3(float(self._camera_target[0]), float(self._camera_target[1]), float(self._camera_target[2])))

    def _update_hud(self) -> None:
        assert self._data is not None
        time_text = ""
        if self._data.times_s is not None:
            time_text = f"t={self._data.times_s[self._frame_index]:.1f}s  "
        state = "PLAY" if self._playing else "PAUSE"
        self._entities["hud"].text = (
            f"{state}  frame {self._frame_index + 1}/{self.frame_count}  "
            f"{time_text}speed={self.playback_speed:.2f}x\n"
            "Space play/pause | ←/→ step | PgUp/PgDn skip | mouse drag orbit | wheel zoom"
        )

    def _scaled_positions(self) -> np.ndarray:
        assert self._data is not None
        return self._data.positions_m * self.position_scale

    def _sun_direction_for_frame(self, frame_index: int) -> np.ndarray:
        assert self._data is not None
        if self._data.epochs_utc is not None:
            return self._sun_model.sun_dir_eci(self._data.epochs_utc[frame_index])
        return np.array([1.0, 0.0, 0.0], dtype=float)

    @property
    def frame_count(self) -> int:
        if self._data is None:
            return 0
        return int(self._data.positions_m.shape[0])


class SkyfieldSunDirectionModel:
    """Compute an Earth-to-Sun unit vector with Skyfield, falling back safely."""

    def __init__(self) -> None:
        self._ts: Any = None
        self._earth: Any = None
        self._sun: Any = None
        self._fallback = AstropySunDirectionModel()
        try:
            from skyfield.api import load

            self._ts = load.timescale()
            eph = load("de421.bsp")
            self._earth = eph["earth"]
            self._sun = eph["sun"]
        except Exception:
            self._ts = None
            self._earth = None
            self._sun = None

    def sun_dir_eci(self, epoch_utc: object) -> np.ndarray:
        epoch = _coerce_datetime(epoch_utc)
        if self._ts is not None and self._earth is not None and self._sun is not None:
            t = self._ts.utc(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second + epoch.microsecond / 1.0e6)
            vec_au = self._earth.at(t).observe(self._sun).position.au
            return _unit(np.asarray(vec_au, dtype=float))
        return self._fallback.sun_dir_eci(epoch)


def _closed_loop_dataframe_to_data(logs: Any) -> ClosedLoopVizData:
    required = [
        "true_position_x",
        "true_position_y",
        "true_position_z",
        "true_quat_w",
        "true_quat_x",
        "true_quat_y",
        "true_quat_z",
    ]
    missing = [column for column in required if column not in logs]
    if missing:
        raise ValueError(f"Closed-loop logs are missing required columns: {missing}")

    positions = logs[["true_position_x", "true_position_y", "true_position_z"]].to_numpy(float)
    quats = logs[["true_quat_w", "true_quat_x", "true_quat_y", "true_quat_z"]].to_numpy(float)
    times = logs["time"].to_numpy(float) if "time" in logs else None
    epochs = np.asarray(logs["epoch_utc"].to_numpy(), dtype=object) if "epoch_utc" in logs else None
    goal_cols = ["goal_quat_w", "goal_quat_x", "goal_quat_y", "goal_quat_z"]
    goal = logs[goal_cols].to_numpy(float) if all(column in logs for column in goal_cols) else None
    return _validate_data(ClosedLoopVizData(positions, quats, times, epochs, goal))


def _closed_loop_arrays_to_data(arrays: Mapping[str, Any]) -> ClosedLoopVizData:
    if "positions" in arrays and "quaternions" in arrays:
        positions = np.asarray(arrays["positions"], dtype=float)
        quats = np.asarray(arrays["quaternions"], dtype=float)
    elif "true_state" in arrays:
        true_state = np.asarray(arrays["true_state"], dtype=float)
        positions = true_state[:, 0:3]
        quats = true_state[:, 6:10]
    else:
        raise ValueError("arrays must include positions/quaternions or true_state")

    goal = None
    if "goal_quaternions" in arrays:
        goal = np.asarray(arrays["goal_quaternions"], dtype=float)
    elif "goal_rotation_state" in arrays:
        goal = np.asarray(arrays["goal_rotation_state"], dtype=float)[:, 0:4]

    times = np.asarray(arrays["time"], dtype=float) if "time" in arrays else None
    epochs = np.asarray(arrays["epochs"], dtype=object) if "epochs" in arrays else None
    return _validate_data(ClosedLoopVizData(positions, quats, times, epochs, goal))


def _validate_data(data: ClosedLoopVizData) -> ClosedLoopVizData:
    if data.positions_m.ndim != 2 or data.positions_m.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if data.quaternions_wxyz.ndim != 2 or data.quaternions_wxyz.shape[1] != 4:
        raise ValueError("quaternions must have shape (N, 4)")
    if len(data.positions_m) != len(data.quaternions_wxyz):
        raise ValueError("positions and quaternions must have the same length")
    if len(data.positions_m) == 0:
        raise ValueError("visualization data must contain at least one frame")
    if data.goal_quaternions_wxyz is not None and data.goal_quaternions_wxyz.shape != data.quaternions_wxyz.shape:
        raise ValueError("goal quaternions must have shape (N, 4)")
    return data


def _downsample_data(data: ClosedLoopVizData, every_n: int) -> ClosedLoopVizData:
    step = max(1, int(every_n))
    if step == 1:
        return data
    idx = np.arange(0, len(data.positions_m), step, dtype=int)
    if idx[-1] != len(data.positions_m) - 1:
        idx = np.append(idx, len(data.positions_m) - 1)
    return ClosedLoopVizData(
        positions_m=data.positions_m[idx],
        quaternions_wxyz=data.quaternions_wxyz[idx],
        times_s=data.times_s[idx] if data.times_s is not None else None,
        epochs_utc=data.epochs_utc[idx] if data.epochs_utc is not None else None,
        goal_quaternions_wxyz=data.goal_quaternions_wxyz[idx] if data.goal_quaternions_wxyz is not None else None,
    )


def _ursina_euler_from_quat_wxyz(quaternion_wxyz: np.ndarray) -> tuple[float, float, float]:
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}.")
    norm = np.linalg.norm(q)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Quaternion must be finite and non-zero.")
    w, x, y, z = q / norm
    return tuple(float(v) for v in Rotation.from_quat([x, y, z, w]).as_euler("xyz", degrees=True))


def _coerce_datetime(epoch: object) -> datetime:
    if hasattr(epoch, "to_pydatetime"):
        epoch = epoch.to_pydatetime()
    if isinstance(epoch, np.datetime64):
        epoch = datetime.fromisoformat(str(epoch).replace("Z", "+00:00"))
    if isinstance(epoch, str):
        epoch = datetime.fromisoformat(epoch.replace("Z", "+00:00"))
    if not isinstance(epoch, datetime):
        raise TypeError("epoch must be datetime-like or ISO datetime string")
    if epoch.tzinfo is not None:
        return epoch.astimezone(timezone.utc).replace(tzinfo=None)
    return epoch


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Expected a finite, non-zero vector.")
    return vector / norm


if not np.allclose(BODY_FORWARD_VEC3, np.array([1.0, 0.0, 0.0])):
    raise RuntimeError("Ursina spacecraft geometry assumes BODY_FORWARD_VEC3 is body +X.")
