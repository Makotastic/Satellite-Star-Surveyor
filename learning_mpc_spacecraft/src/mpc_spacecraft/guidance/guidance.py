from mpc_spacecraft.planner.star_planner import StarPlanner, PlanModes
from mpc_spacecraft.guidance.sun_tracker import (
    AstropySunDirectionModel,
    TimeLike,
)
from mpc_spacecraft.utilities.utils import (
    FloatArray,
    Vec3,
    unit,
    BODY_FORWARD_VEC3,
    Quat
    )

import numpy as np
import quaternion as qu

IDX_TRANS_STATE = slice(0, 6)
IDX_ROT_STATE = slice(6, 13)

class Guideance:

    def __init__(
        self,
        planner: StarPlanner,
        sun_model: AstropySunDirectionModel,
    ):
        self.planner = planner
        self.sun_model = sun_model

    def tick(
        self,
        dt: float,
        body_state: FloatArray,
        epoch_utc: TimeLike,
    ) -> tuple[FloatArray, PlanModes, bool]:

        sun_dir_I = self.sun_model.sun_dir_eci(epoch_utc)

        goal_vec, mode, is_complete = self.planner.tick(body_state, dt, sun_dir_I)

        curr_rot_state = body_state[IDX_ROT_STATE]

        if goal_vec is None or is_complete:
            goal_rot_state = curr_rot_state
        else:
            goal_quat = compute_inertial_rot_quat(goal_vec, BODY_FORWARD_VEC3).as_float_array()
            goal_omega = np.zeros(3)
            goal_rot_state = np.concatenate((goal_quat, goal_omega))

        return goal_rot_state, mode, is_complete


def compute_inertial_rot_quat(goal_vec: Vec3, body_forward_vec: Vec3) -> Quat:
    goal_u = unit(np.asarray(goal_vec, dtype=float))
    body_fwd_u = unit(np.asarray(body_forward_vec, dtype=float))

    dot = float(np.clip(np.dot(body_fwd_u, goal_u), -1.0, 1.0))

    # Aligned: identity rotation.
    if dot > 1.0 - 1e-8:
        return qu.quaternion(1.0, 0.0, 0.0, 0.0)

    # Anti-parallel: deterministic 180deg about a fixed orthogonal axis.
    if dot < -1.0 + 1e-8:
        basis = np.array([1.0, 0.0, 0.0]) if abs(body_fwd_u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = unit(np.cross(body_fwd_u, basis))
        return qu.from_rotation_vector(axis * np.pi)

    axis = unit(np.cross(body_fwd_u, goal_u))
    theta = np.arccos(dot)
    return qu.from_rotation_vector(axis * theta)
    
