from mpc_spacecraft.planner.star_planner import StarPlanner, PlanModes
from mpc_spacecraft.guidance.sun_tracker import AstropySunDirectionModel
from mpc_spacecraft.guidance.sun_tracker import TimeLike
from mpc_spacecraft.utilities.utils import (
    RigidBodyState,
    RotationState,
    BODY_FORWARD_VEC3,
    quat_rotate_vector_a_to_b,
)

import numpy as np
import quaternion as qu

class Guidance:
    def __init__(
        self,
        planner: StarPlanner,
        sun_model: AstropySunDirectionModel,
    ):
        self.planner = planner
        self.sun_model = sun_model

    def tick(
        self, current_epoch_utc: TimeLike, delta_time: float, body_state: RigidBodyState
    ) -> tuple[RotationState, PlanModes, bool]:

        sun_dir_I = self.sun_model.sun_dir_eci(current_epoch_utc)

        goal_vec, mode, is_complete = self.planner.tick(body_state, delta_time, sun_dir_I)

        if goal_vec is None or is_complete:
            goal_rot_state = body_state.rotation.copy()
            goal_rot_state.omega[:] = np.zeros((3))
        else:
            goal_quat = qu.as_float_array(quat_rotate_vector_a_to_b(
                BODY_FORWARD_VEC3, goal_vec
            ))
            goal_rot_state = RotationState.zeros()
            goal_rot_state.quat[:] = goal_quat
            goal_rot_state.omega[:] = np.zeros(3)

        return goal_rot_state, mode, is_complete
