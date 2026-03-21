from mpc_spacecraft.planner.star_planner import StarPlanner, PlanModes
from mpc_spacecraft.guidance.sun_tracker import AstropySunDirectionModel
from mpc_spacecraft.utilities.utils import (
    FloatArray,
    BODY_FORWARD_VEC3,
    FULL_STATE_SLICES,
    quat_rotate_vector_a_to_b,
)

import numpy as np

IDX_TRANS_STATE = FULL_STATE_SLICES.translation
IDX_ROT_STATE = FULL_STATE_SLICES.rotation


class Guidance:
    def __init__(
        self,
        planner: StarPlanner,
        sun_model: AstropySunDirectionModel,
    ):
        self.planner = planner
        self.sun_model = sun_model

    def tick(
        self, dt: float, body_state: FloatArray
    ) -> tuple[FloatArray, PlanModes, bool]:

        sun_dir_I = self.sun_model.sun_dir_eci()

        goal_vec, mode, is_complete = self.planner.tick(body_state, dt, sun_dir_I)

        if goal_vec is None or is_complete:
            curr_rot_state = body_state[IDX_ROT_STATE]
            goal_rot_state = curr_rot_state
        else:
            goal_quat = quat_rotate_vector_a_to_b(
                BODY_FORWARD_VEC3, goal_vec
            ).as_float_array()
            goal_omega = np.zeros(3)
            goal_rot_state = np.concatenate((goal_quat, goal_omega))

        return goal_rot_state, mode, is_complete
