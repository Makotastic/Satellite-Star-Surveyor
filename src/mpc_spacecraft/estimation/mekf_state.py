from dataclasses import dataclass
import numpy as np
import quaternion as qu
from mpc_spacecraft.utilities.utils import Vec3, Quat


@dataclass
class KinematicEstimatedState:
    r_I: Vec3
    v_I: Vec3
    q_BI: Quat
    b_g: Vec3
    b_a: Vec3
