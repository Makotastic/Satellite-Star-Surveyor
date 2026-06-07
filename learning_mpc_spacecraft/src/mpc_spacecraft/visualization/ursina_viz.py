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
TARGET_STAR_DISTANCE = 1000.0
TARGET_STAR_RADIUS = 3
BODY_FORWARD_RAY_LENGTH = 100000.0
CAMERA_DEFAULT_DISTANCE = 12.0
CAMERA_MIN_DISTANCE = 2.0
CAMERA_MAX_DISTANCE = 80.0
BODY_FORWARD_CAMERA_BACK_OFFSET = 3.0
BODY_FORWARD_CAMERA_UP_OFFSET = 0.9
BODY_FORWARD_CAMERA_LOOK_AHEAD = 12.0
ATTITUDE_BODY_TO_WORLD = "body_to_world"
EARTH_TEXTURE_SIZE = 500


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
        Left mouse drag: Orbit camera around the satellite.
        Mouse wheel: Zoom in/out while staying satellite-relative.
        +/-: Increase/decrease playback speed.
        R: Reset camera orbit and zoom.
        F: Toggle body-forward camera fixed to the spacecraft.
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
        self._camera_dragging = False
        self._body_forward_camera = False
        self._target_dirs: np.ndarray | None = None

    def visualize_closed_loop_dataframe(
        self,
        logs: Any,
        fps: float = 30.0,
        every_n: int = 1,
        play: bool = True,
        run: bool = True,
        targets: Any | None = None,
    ) -> "UrsinaSpacecraftVisualizer":
        """Load closed-loop dataframe logs and optionally start the Ursina app."""
        self.load_closed_loop_dataframe(logs, fps=fps, every_n=every_n, play=play, targets=targets)
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
        targets: Any | None = None,
    ) -> "UrsinaSpacecraftVisualizer":
        """Load closed-loop arrays and optionally start the Ursina app."""
        self.load_closed_loop_arrays(arrays, fps=fps, every_n=every_n, play=play, targets=targets)
        if run:
            self.run()
        return self

    def load_closed_loop_dataframe(
        self,
        logs: Any,
        fps: float = 30.0,
        every_n: int = 1,
        play: bool = True,
        targets: Any | None = None,
    ) -> None:
        """Normalize dataframe logs produced by ``ClosedLoopTestResult.to_dataframe()``."""
        self._set_data(_downsample_data(_closed_loop_dataframe_to_data(logs), every_n), fps, play)
        self._target_dirs = _targets_to_unit_vectors(targets) if targets is not None else None

    def load_closed_loop_arrays(
        self,
        arrays: Mapping[str, Any],
        fps: float = 30.0,
        every_n: int = 1,
        play: bool = True,
        targets: Any | None = None,
    ) -> None:
        """Normalize dictionary logs produced by ``ClosedLoopTestResult.to_arrays()``."""
        self._set_data(_downsample_data(_closed_loop_arrays_to_data(arrays), every_n), fps, play)
        self._target_dirs = _targets_to_unit_vectors(targets) if targets is not None else None

    def run(self) -> None:
        """Create the Ursina scene and start the interactive render loop."""
        if self._data is None:
            raise RuntimeError("Load closed-loop data before calling run().")

        from panda3d.core import loadPrcFileData  # type: ignore[import-not-found]

        loadPrcFileData("", "audio-library-name null")
        import __main__

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
        setattr(__main__, "update", update)
        setattr(__main__, "input", input)
        camera.clip_plane_far = 100000
        self._app.run()

    def _set_data(self, data: ClosedLoopVizData, fps: float, play: bool) -> None:
        self._data = data
        self._fps = max(1.0, float(fps))
        self._playing = bool(play)
        self._frame_index = 0
        self._frame_accumulator = 0.0

    def _setup_scene(self) -> None:
        from ursina import AmbientLight, Color, DirectionalLight, Entity, Mesh, PointLight, Text, Vec3, color, invoke

        assert self._data is not None
        earth_radius = R_EARTH_M * self.position_scale
        positions = self._scaled_positions()

        self._entities["earth"] = Entity(
            model="sphere",
            scale=earth_radius * 2.0,
            texture=_create_earth_texture(),
            color=color.white,
        )
        invoke(setattr, self._entities["earth"], "texture", _create_earth_texture(), delay=0.0)
        self._entities["earth"].unlit = False
        self._entities["earth"].shininess = 0.08
        self._entities["sun"] = Entity(
            model="sphere",
            scale=SUN_VIS_RADIUS,
            color=color.rgb(255, 190, 40),
            unlit=True,
        )
        self._entities["sun_glow_outer"] = Entity(
            model="sphere",
            scale=SUN_VIS_RADIUS * 2.6,
            color=color.rgba(255, 170, 35, 55),
            unlit=True,
            add_to_scene_entities=False,
        )
        self._entities["sun_glow_inner"] = Entity(
            model="sphere",
            scale=SUN_VIS_RADIUS * 1.55,
            color=color.rgba(255, 230, 90, 95),
            unlit=True,
            add_to_scene_entities=False,
        )
        self._build_target_stars()

        self._entities["spacecraft"] = Entity()
        self._build_spacecraft_entity(self._entities["spacecraft"])
        # Do not force a color on the empty spacecraft parent. A parent-level
        # color render attribute is inherited by children in Panda3D and can
        # wash out individually colored satellite parts. Each child part gets
        # its own forced flat color in _build_spacecraft_entity().
        self._entities["goal_frame"] = Entity(enabled=self._data.goal_quaternions_wxyz is not None)
        self._build_frame(self._entities["goal_frame"], scale=0.9, alpha=0.45)

        trail_vertices = [Vec3(float(p[0]), float(p[1]), float(p[2])) for p in positions]
        self._entities["orbit_trail"] = Entity(
            model=Mesh(vertices=trail_vertices, mode="line"),
            color=color.rgba(255, 80, 80, 180),
        )
        self._entities["hud"] = Text(
            text="",
            origin=(-0.5, -0.5),
            position=(-0.86, -0.47),
            scale=0.8,
            color=Color(0.9, 0.95, 1.0, 1.0),
        )

        AmbientLight(color=Color(0.10, 0.10, 0.12, 1.0))
        sun_light = DirectionalLight(color=Color(0.55, 0.52, 0.48, 1.0))
        sun_light.position = self._entities["sun"].position
        sun_light.look_at(Vec3(0, 0, 0))
        self._entities["sun_light"] = sun_light
        point_light = PointLight(color=Color(0.35, 0.28, 0.18, 1.0))
        point_light.position = self._entities["sun"].position
        self._entities["sun_point_light"] = point_light

    def _build_target_stars(self) -> None:
        from ursina import Entity, Text, Vec3, color

        if self._target_dirs is None:
            return

        stars = []
        labels = []
        for index, direction in enumerate(self._target_dirs):
            position = _unit(direction) * TARGET_STAR_DISTANCE
            pos_vec = Vec3(float(position[0]), float(position[1]), float(position[2]))
            star = Entity(
                model="circle",
                position=pos_vec,
                scale=TARGET_STAR_RADIUS,
                color=color.rgb(255, 35, 210),
                billboard=True,
                unlit=True,
            )
            label_offset = _target_label_offset(direction) * TARGET_STAR_RADIUS * 1.7
            label_pos = position + label_offset
            label = Text(
                text=str(index),
                position=Vec3(float(label_pos[0]), float(label_pos[1]), float(label_pos[2])),
                origin=(0, 0),
                scale=1.0,
                color=color.rgb(255, 185, 245),
                billboard=True,
            )
            labels.append(label)
            stars.append(star)
        self._entities["target_stars"] = stars
        self._entities["target_star_labels"] = labels

    def _build_spacecraft_entity(self, parent: Any) -> None:
        """Build a high-contrast spacecraft model with forced flat colors.

        This version avoids Ursina's built-in ``cube`` primitive for the
        spacecraft parts because, on some Linux/OpenGL/Ursina combinations, the
        default primitive material can render as mostly white even when a color
        is supplied.  Each spacecraft component is instead a small custom mesh
        whose Panda3D node color, color-scale, texture state, material state,
        and lighting state are explicitly forced.
        """
        from ursina import Entity, Mesh, Vec3

        def flat_box(
            center: tuple[float, float, float],
            size: tuple[float, float, float],
            rgba: tuple[int, int, int, int],
        ) -> Any:
            cx, cy, cz = center
            sx, sy, sz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
            vertices = [
                Vec3(cx - sx, cy - sy, cz - sz),
                Vec3(cx + sx, cy - sy, cz - sz),
                Vec3(cx + sx, cy + sy, cz - sz),
                Vec3(cx - sx, cy + sy, cz - sz),
                Vec3(cx - sx, cy - sy, cz + sz),
                Vec3(cx + sx, cy - sy, cz + sz),
                Vec3(cx + sx, cy + sy, cz + sz),
                Vec3(cx - sx, cy + sy, cz + sz),
            ]
            triangles = [
                (0, 1, 2), (0, 2, 3),  # -Z
                (4, 6, 5), (4, 7, 6),  # +Z
                (0, 4, 5), (0, 5, 1),  # -Y
                (3, 2, 6), (3, 6, 7),  # +Y
                (1, 5, 6), (1, 6, 2),  # +X
                (0, 3, 7), (0, 7, 4),  # -X
            ]
            entity = Entity(parent=parent, model=Mesh(vertices=vertices, triangles=triangles), texture=None)
            _force_flat_entity_color(entity, rgba)
            return entity

        def flat_disc(
            center: tuple[float, float, float],
            radius: float,
            rgba: tuple[int, int, int, int],
            segments: int = 24,
        ) -> Any:
            # Disc lies in the local XY plane at constant Z.
            cx, cy, cz = center
            vertices = [Vec3(cx, cy, cz)]
            for i in range(segments):
                angle = 2.0 * np.pi * i / segments
                vertices.append(Vec3(cx + radius * np.cos(angle), cy + radius * np.sin(angle), cz))
            triangles = [(0, i, 1 + (i % segments)) for i in range(1, segments + 1)]
            entity = Entity(parent=parent, model=Mesh(vertices=vertices, triangles=triangles), texture=None)
            _force_flat_entity_color(entity, rgba)
            return entity

        def flat_line(points: list[tuple[float, float, float]], rgba: tuple[int, int, int, int]) -> Any:
            entity = Entity(
                parent=parent,
                model=Mesh(vertices=[Vec3(*point) for point in points], mode="line"),
                texture=None,
            )
            _force_flat_entity_color(entity, rgba)
            return entity

        s = SPACECRAFT_VIS_SIZE

        # Main matte graphite spacecraft bus: dark enough for contrast, but
        # not pure black, so body attitude and shape remain readable.
        flat_box((0.0, 0.0, 0.0), (s * 1.08, s * 0.72, s * 0.54), (16, 20, 27, 255))

        # Six thin matte skin plates sit slightly outside the bus.  They create
        # a readable silhouette and give each face a slightly different shade.
        skin_t = s * 0.018
        flat_box((s * 0.552, 0.0, 0.0), (skin_t, s * 0.75, s * 0.57), (32, 38, 48, 255))      # +X face
        flat_box((-s * 0.552, 0.0, 0.0), (skin_t, s * 0.75, s * 0.57), (8, 10, 14, 255))       # -X face
        flat_box((0.0, s * 0.372, 0.0), (s * 1.10, skin_t, s * 0.57), (18, 27, 38, 255))      # +Y face
        flat_box((0.0, -s * 0.372, 0.0), (s * 1.10, skin_t, s * 0.57), (18, 27, 38, 255))     # -Y face
        flat_box((0.0, 0.0, s * 0.282), (s * 1.10, s * 0.75, skin_t), (24, 29, 37, 255))      # +Z face
        flat_box((0.0, 0.0, -s * 0.282), (s * 1.10, s * 0.75, skin_t), (5, 6, 9, 255))        # -Z face

        # Muted bronze/gold MLI-style front equipment blanket on body +X.
        # Kept warm but not pale, so it does not blow out to white.
        flat_box((s * 0.64, 0.0, 0.0), (s * 0.24, s * 0.76, s * 0.58), (150, 82, 20, 255))

        # Dark radiator/back face on body -X.
        flat_box((-s * 0.64, 0.0, 0.0), (s * 0.14, s * 0.62, s * 0.46), (0, 0, 0, 255))

        # Forward sensor/boresight housing and black lens.
        flat_box((s * 0.86, 0.0, 0.0), (s * 0.42, s * 0.24, s * 0.24), (92, 34, 22, 255))
        flat_box((s * 1.10, 0.0, 0.0), (s * 0.08, s * 0.18, s * 0.18), (0, 0, 0, 255))

        # Solar array booms.
        flat_box((0.0, s * 0.78, 0.0), (s * 0.18, s * 0.62, s * 0.07), (72, 78, 82, 255))
        flat_box((0.0, -s * 0.78, 0.0), (s * 0.18, s * 0.62, s * 0.07), (72, 78, 82, 255))

        # Solar arrays: deep blue surfaces with pale grid lines.
        for panel_y in (s * 1.35, -s * 1.35):
            flat_box((0.0, panel_y, 0.0), (s * 2.25, s * 0.82, s * 0.035), (3, 18, 76, 255))
            for offset in (-0.75, -0.25, 0.25, 0.75):
                flat_box((s * offset, panel_y, s * 0.03), (s * 0.018, s * 0.84, s * 0.018), (86, 155, 220, 255))
            for offset in (-0.25, 0.0, 0.25):
                flat_box((0.0, panel_y + s * offset, s * 0.032), (s * 2.25, s * 0.014, s * 0.018), (86, 155, 220, 255))

        # A small high-gain antenna on the +Z deck.  Made from a disc mesh, not
        # a built-in circle, so its color is also forced.
        flat_box((0.0, 0.0, s * 0.35), (s * 0.08, s * 0.08, s * 0.24), (78, 82, 76, 255))
        flat_disc((0.0, 0.0, s * 0.50), s * 0.23, (78, 82, 76, 255))

        # Add body axes cues on the bus itself: red +X, green +Y, blue +Z.
        flat_box((s * 0.42, 0.0, s * 0.31), (s * 0.42, s * 0.035, s * 0.035), (235, 48, 42, 255))
        flat_box((0.0, s * 0.32, s * 0.31), (s * 0.035, s * 0.32, s * 0.035), (50, 190, 82, 255))
        flat_box((0.0, 0.0, s * 0.43), (s * 0.035, s * 0.035, s * 0.28), (72, 120, 235, 255))

        # Body +X forward/boresight ray.  Bright red and forced flat.
        flat_line([(0.0, 0.0, 0.0), (BODY_FORWARD_RAY_LENGTH, 0.0, 0.0)], (255, 0, 0, 255))

    def _build_frame(self, parent: Any, scale: float, alpha: float) -> None:
        from ursina import Entity, color

        Entity(parent=parent, model="cube", x=scale / 2, scale=(scale, 0.025, 0.025), color=color.rgba(255, 0, 0, int(255 * alpha)))

    def _update(self) -> None:
        from ursina import mouse, time

        if self._playing and self._data is not None:
            self._frame_accumulator += time.dt * self._fps * self.playback_speed
            if self._frame_accumulator >= 1.0:
                step = int(self._frame_accumulator)
                self._frame_accumulator -= step
                self._apply_frame((self._frame_index + step) % self.frame_count)

        # Do not derive drag state from held_keys here. In Ursina, mouse
        # button state is synthesized from down/up events; if the "up" event is
        # missed after a click or focus change, held_keys["left mouse"] can stay
        # truthy and the camera appears to "stick" to the cursor.  Drag state is
        # instead started/stopped in _input(), with this physical-button check as
        # a safety net.
        if self._body_forward_camera:
            self._camera_dragging = False
        elif self._camera_dragging and not self._is_left_mouse_down():
            self._camera_dragging = False

        if self._camera_dragging:
            self._camera_yaw_deg += mouse.velocity[0] * 160.0
            self._camera_pitch_deg = float(np.clip(self._camera_pitch_deg - mouse.velocity[1] * 160.0, -85.0, 85.0))
        self._update_camera()

    def _is_left_mouse_down(self) -> bool:
        """Return the current physical left-button state when available.

        Ursina exposes ``mouse.left`` and also runs on Panda3D, whose
        MouseWatcher can query the current hardware button state.  Checking the
        current state prevents a missed "left mouse up" event from leaving the
        camera in drag mode indefinitely.
        """
        try:
            from panda3d.core import MouseButton  # type: ignore[import-not-found]
            from ursina import application  # type: ignore[import-not-found]

            base = getattr(application, "base", None)
            watcher = getattr(base, "mouseWatcherNode", None)
            if watcher is not None:
                return bool(watcher.is_button_down(MouseButton.one()))
        except Exception:
            pass

        try:
            from ursina import mouse  # type: ignore[import-not-found]

            return bool(getattr(mouse, "left", False))
        except Exception:
            return False

    def _input(self, key: str) -> None:
        if key == "left mouse down":
            self._camera_dragging = True
        elif key == "left mouse up":
            self._camera_dragging = False
        elif key in {"escape", "window focus lost"}:
            # Defensive reset for platforms/IDEs where a release event can be
            # swallowed when the cursor leaves the window or focus changes.
            self._camera_dragging = False
        elif key == "space":
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
        elif key == "f":
            self._body_forward_camera = not self._body_forward_camera
            self._camera_dragging = False
        elif key == "r":
            self._body_forward_camera = False
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
        _apply_attitude_to_entity(spacecraft, quat)

        if self._data.goal_quaternions_wxyz is not None:
            goal = self._entities["goal_frame"]
            goal.position = spacecraft.position
            _apply_attitude_to_entity(goal, self._data.goal_quaternions_wxyz[self._frame_index])

        sun_dir = self._sun_direction_for_frame(self._frame_index)
        self._set_sun(sun_dir)
        self._update_hud()

    def _set_sun(self, sun_dir: np.ndarray) -> None:
        from ursina import Vec3

        sun_pos = _unit(sun_dir) * SUN_VIS_DISTANCE
        sun_vec = Vec3(float(sun_pos[0]), float(sun_pos[1]), float(sun_pos[2]))
        self._entities["sun"].position = sun_vec
        self._entities["sun_glow_outer"].position = sun_vec
        self._entities["sun_glow_inner"].position = sun_vec
        self._entities["sun_point_light"].position = sun_vec
        self._entities["sun_light"].position = sun_vec
        self._entities["sun_light"].look_at(Vec3(0, 0, 0))

    def _update_camera(self) -> None:
        from ursina import Vec3, camera

        if "spacecraft" in self._entities:
            spacecraft_pos = self._entities["spacecraft"].position
            self._camera_target = np.array([float(spacecraft_pos.x), float(spacecraft_pos.y), float(spacecraft_pos.z)], dtype=float)

        if self._body_forward_camera:
            cam_pos, look_target = self._body_forward_camera_pose()
        else:
            direction = self._camera_orbit_direction()
            cam_pos = self._camera_target + direction * self._camera_distance
            look_target = self._camera_target

        camera.position = Vec3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        camera.look_at(Vec3(float(look_target[0]), float(look_target[1]), float(look_target[2])))

    def _body_forward_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return camera position and look target for body-forward view.

        The camera is rigidly offset from the spacecraft in body coordinates,
        sitting just behind and above the body +X axis while looking down that
        +X/body-forward direction.  This makes the red body-forward ray behave
        like a boresight/aiming reference.
        """
        assert self._data is not None
        quat = self._data.quaternions_wxyz[self._frame_index]
        rotation = _attitude_rotation_from_quat_wxyz(quat)
        forward = _unit(rotation.apply(BODY_FORWARD_VEC3))
        body_up = _unit(rotation.apply(np.array([0.0, 0.0, 1.0], dtype=float)))

        cam_pos = (
            self._camera_target
            - forward * BODY_FORWARD_CAMERA_BACK_OFFSET
            + body_up * BODY_FORWARD_CAMERA_UP_OFFSET
        )
        look_target = self._camera_target + forward * BODY_FORWARD_CAMERA_LOOK_AHEAD
        return cam_pos, look_target

    def _camera_orbit_direction(self) -> np.ndarray:
        yaw = np.deg2rad(self._camera_yaw_deg)
        pitch = np.deg2rad(self._camera_pitch_deg)
        return np.array(
            [
                np.cos(pitch) * np.cos(yaw),
                np.sin(pitch),
                np.cos(pitch) * np.sin(yaw),
            ],
            dtype=float,
        )

    def _update_hud(self) -> None:
        assert self._data is not None
        time_text = ""
        if self._data.times_s is not None:
            time_text = f"t={self._data.times_s[self._frame_index]:.1f}s  "
        state = "PLAY" if self._playing else "PAUSE"
        camera_mode = "BODY-FWD" if self._body_forward_camera else "ORBIT"
        self._entities["hud"].text = (
            f"{state}  frame {self._frame_index + 1}/{self.frame_count}  "
            f"{time_text}speed={self.playback_speed:.2f}x  camera={camera_mode}  "
            f"quat={ATTITUDE_BODY_TO_WORLD}\n"
            "Space play/pause | Left/Right step | PgUp/PgDn skip | left mouse drag orbit | "
            "wheel zoom | F body-forward | R reset"
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


def _targets_to_unit_vectors(targets: Any) -> np.ndarray:
    """Convert target table rows with RA/Dec or x/y/z columns to ECI unit vectors."""
    if all(column in targets for column in ("x", "y", "z")):
        vectors = targets[["x", "y", "z"]].to_numpy(float)
    elif all(column in targets for column in ("RA", "Dec")):
        ra = np.asarray(targets["RA"], dtype=float)
        dec = np.asarray(targets["Dec"], dtype=float)
        vectors = np.column_stack(
            [
                np.cos(dec) * np.cos(ra),
                np.cos(dec) * np.sin(ra),
                np.sin(dec),
            ]
        )
    else:
        array = np.asarray(targets, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("targets must include RA/Dec columns, x/y/z columns, or have shape (N, 3)")
        vectors = array

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0.0) or not np.isfinite(norms).all():
        raise ValueError("target vectors must be finite and non-zero")
    return vectors / norms


def _target_label_offset(direction: np.ndarray) -> np.ndarray:
    """Return a stable tangent-space offset for a label beside a target marker."""
    unit_dir = _unit(np.asarray(direction, dtype=float))
    reference = np.array([0.0, 1.0, 0.0], dtype=float)
    tangent = np.cross(unit_dir, reference)
    if np.linalg.norm(tangent) < 1.0e-6:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
        tangent = np.cross(unit_dir, reference)
    return _unit(tangent)


def _create_earth_texture() -> Any:
    """Create a small procedural blue/green Earth texture for Ursina."""
    from PIL import Image
    from ursina import Texture

    width = EARTH_TEXTURE_SIZE * 2
    height = EARTH_TEXTURE_SIZE
    yy, xx = np.mgrid[0:height, 0:width]
    lon = (xx / width) * 2.0 * np.pi - np.pi
    lat = np.pi / 2.0 - (yy / height) * np.pi

    land_signal = (
        0.58 * np.sin(2.5 * lon + 1.3 * np.sin(3.0 * lat))
        + 0.32 * np.sin(5.5 * lon - 2.0 * lat)
        + 0.22 * np.cos(9.0 * lon + 4.0 * lat)
        + 0.16 * np.sin(13.0 * lon - 1.5 * np.cos(5.0 * lat))
    )
    polar_ice = np.abs(lat) > np.deg2rad(70.0)
    land = land_signal > 0.24
    coast = np.abs(land_signal - 0.24) < 0.045

    ocean = np.stack(
        [
            12.0 + 20.0 * np.cos(lat) ** 2,
            70.0 + 45.0 * np.cos(lat) ** 2,
            150.0 + 65.0 * np.cos(lat) ** 2,
        ],
        axis=-1,
    )
    terrain = np.stack(
        [
            30.0 + 42.0 * np.cos(lat) ** 2,
            112.0 + 75.0 * np.cos(lat) ** 2,
            45.0 + 32.0 * np.cos(lat) ** 2,
        ],
        axis=-1,
    )
    texture = np.where(land[..., None], terrain, ocean)
    texture = np.where(coast[..., None], np.array([190.0, 205.0, 105.0]), texture)
    texture = np.where(polar_ice[..., None], np.array([225.0, 240.0, 255.0]), texture)

    cloud_signal = np.sin(10.0 * lon + 6.0 * np.sin(lat)) + np.cos(7.5 * lon - 3.0 * lat)
    clouds = (cloud_signal > 1.2) & (~polar_ice)
    texture = np.where(clouds[..., None], texture * 0.72 + np.array([255.0, 255.0, 255.0]) * 0.28, texture)

    specular_highlight = 0.12 * (1.0 - land.astype(float)) * np.cos(lat) ** 2
    texture = texture + specular_highlight[..., None] * np.array([90.0, 125.0, 180.0])
    image = Image.fromarray(np.clip(texture, 0, 255).astype(np.uint8), mode="RGB")
    return Texture(image)



def _force_flat_entity_color(entity: Any, rgba: tuple[int, int, int, int]) -> None:
    """Force a flat, non-reflective color on an Ursina/Panda3D node.

    This uses high render-state priority so inherited lights, textures,
    materials, shaders, or color scales cannot override spacecraft colors.
    """
    from ursina import color

    r, g, b, a = rgba
    rf, gf, bf, af = r / 255.0, g / 255.0, b / 255.0, a / 255.0
    entity.color = color.rgba(r, g, b, a)
    entity.unlit = True
    entity.texture = None
    entity.shininess = 0.0

    priority = 1000
    # Turn off anything that can make a matte color blow out to white.
    for method_name in ("setTextureOff", "setMaterialOff", "setLightOff", "setShaderOff"):
        try:
            getattr(entity, method_name)(priority)
        except Exception:
            pass

    try:
        entity.clearColorScale()
    except Exception:
        pass
    try:
        entity.setColor(rf, gf, bf, af, priority)
    except Exception:
        try:
            entity.setColor(rf, gf, bf, af)
        except Exception:
            pass
    # ColorScale multiplies the base color; leave it neutral rather than
    # tinting again.  This avoids inherited or accidental overbright scales.
    try:
        entity.setColorScale(1.0, 1.0, 1.0, 1.0, priority)
    except Exception:
        try:
            entity.setColorScale(1.0, 1.0, 1.0, 1.0)
        except Exception:
            pass

    # Also force every child NodePath in case Ursina wraps the mesh under the
    # Entity node on this platform/version.
    try:
        for child in entity.getChildren():
            for method_name in ("setTextureOff", "setMaterialOff", "setLightOff", "setShaderOff"):
                try:
                    getattr(child, method_name)(priority)
                except Exception:
                    pass
            try:
                child.clearColorScale()
            except Exception:
                pass
            try:
                child.setColor(rf, gf, bf, af, priority)
            except Exception:
                pass
            try:
                child.setColorScale(1.0, 1.0, 1.0, 1.0, priority)
            except Exception:
                pass
    except Exception:
        pass


def _rotation_from_quat_wxyz(quaternion_wxyz: np.ndarray) -> Rotation:
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}.")
    norm = np.linalg.norm(q)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Quaternion must be finite and non-zero.")
    w, x, y, z = q / norm
    return Rotation.from_quat([x, y, z, w])


def _attitude_rotation_from_quat_wxyz(quaternion_wxyz: np.ndarray) -> Rotation:
    """Return the body-to-world attitude rotation used by both model and camera."""
    return _rotation_from_quat_wxyz(quaternion_wxyz)


def _apply_attitude_to_entity(entity: Any, quaternion_wxyz: np.ndarray) -> None:
    """Apply spacecraft attitude directly as a Panda3D quaternion.

    This intentionally avoids converting through ``Entity.rotation`` Euler
    angles. Euler conversion can introduce order/sign ambiguity and can make the
    rendered spacecraft disagree with the body-forward camera.  The same
    body-to-world ``Rotation`` produced here is also used in
    ``_body_forward_camera_pose()``, so the red +X body-forward ray, model, and
    camera all share one attitude source of truth.
    """
    from panda3d.core import Quat  # type: ignore[import-not-found]

    rotation = _attitude_rotation_from_quat_wxyz(quaternion_wxyz)
    x, y, z, w = rotation.as_quat()
    entity.setQuat(Quat(float(w), float(x), float(y), float(z)))


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
