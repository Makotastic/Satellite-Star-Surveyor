import numpy as np
import quaternion as qu
from scipy.linalg import expm, solve

from mpc_spacecraft.utilities.utils import Vec3, skew, FloatArray, Quat
from ..dynamics.disturbances import gravity, jacobian_gravity as J_gravity
from .sensor_models import GNSSConfig, GyroConfig, IMUConfig, StarTrackerConfig
from .mekf_state import KinematicEstimatedState

IDX_R = slice(0, 3)
IDX_V = slice(3, 6)
IDX_THETA = slice(6, 9)
IDX_BG = slice(9, 12)
IDX_BA = slice(12, 15)

I3 = np.eye(3)
z3 = np.zeros((3, 3))


class MEKF:
    def __init__(
        self,
        state: KinematicEstimatedState,
        imu_config: IMUConfig,
        gyro_config: GyroConfig,
        gnss_config: GNSSConfig,
        star_config: StarTrackerConfig,
    ):
        self.state = state

        self.R_gnss = gnss_config.R
        self.R_st = star_config.R

        self.Q_c = np.block(
            [
                [gyro_config.sigma_g2, z3, z3, z3],
                [z3, imu_config.sigma_a2, z3, z3],
                [z3, z3, gyro_config.sigma_bg2, z3],
                [z3, z3, z3, imu_config.sigma_ba2],
            ]
        )

        # Initial covariance P (CubeSat-class, modest accuracy)
        sigma_pos = 5.0  # m
        sigma_vel = 0.1  # m/s
        sigma_theta = 0.01  # rad (~0.57 deg) attitude error
        sigma_bg = 0.5 * np.pi / 180.0 / 3600.0  # rad/s (0.5 deg/hr)
        sigma_ba = 200e-6 * 9.80665  # m/s^2 (200 micro-g)

        self.P = np.diag(
            [
                sigma_pos**2,
                sigma_pos**2,
                sigma_pos**2,
                sigma_vel**2,
                sigma_vel**2,
                sigma_vel**2,
                sigma_theta**2,
                sigma_theta**2,
                sigma_theta**2,
                sigma_bg**2,
                sigma_bg**2,
                sigma_bg**2,
                sigma_ba**2,
                sigma_ba**2,
                sigma_ba**2,
            ]
        )

    def update(
        self,
        dt,
        imu: Vec3,
        gyro: Vec3,
        gnss_measure: Vec3 | None = None,
        st_measure: Quat | None = None,
    ):
        self._predict(imu, gyro, dt)

        if st_measure is not None:
            self._update_startracker(st_measure)

        if gnss_measure is not None:
            self._update_gnss(gnss_measure)

        return self.state

    def _predict(self, imu: Vec3, gyro: Vec3, dt: float):
        omega = gyro - self.state.b_g
        accel_B = imu - self.state.b_a
        R = qu.as_rotation_matrix(self.state.q_BI)
        g_I = gravity(self.state.r_I)

        vel = self.state.v_I + (g_I + R @ accel_B) * dt
        pos = self.state.r_I + (vel * dt)
        q = self.state.q_BI * qu.from_rotation_vector(omega * dt)

        self.state.r_I = pos
        self.state.v_I = vel
        self.state.q_BI = q / abs(q)

        F, G = self._form_F_G(accel_B, omega)
        Phi, Q_d = self._form_phi_Q(F, G, self.Q_c, dt)

        self.P = Phi @ self.P @ Phi.T + Q_d

    def _update_gnss(self, gnss_measure: FloatArray):
        H = np.block([[I3, z3, z3, z3, z3], [z3, I3, z3, z3, z3]])

        y_residual = gnss_measure - np.concatenate([self.state.r_I, self.state.v_I])

        dx, P = self._kalman(H, self.R_gnss, y_residual)

        self._inject_reset(dx, P)

    def _update_startracker(self, st_measure: Quat):
        H = np.block([[z3, z3, I3, z3, z3]])

        q_err = self.state.q_BI.conjugate() * st_measure

        if q_err.w < 0:
            q_err = -q_err

        y_residual = qu.as_rotation_vector(q_err)

        dx, P = self._kalman(H, self.R_st, y_residual)

        self._inject_reset(dx, P)

    def _inject_reset(self, dx: FloatArray, P: FloatArray):
        # Inject
        self.state.r_I += dx[IDX_R]
        self.state.v_I += dx[IDX_V]
        self.state.b_g += dx[IDX_BG]
        self.state.b_a += dx[IDX_BA]

        dq = qu.from_rotation_vector(dx[IDX_THETA])
        q_new = self.state.q_BI * dq
        self.state.q_BI = q_new / abs(q_new)

        # Reset
        G = np.eye(15)
        dtheta = dx[IDX_THETA]

        #   Right-invariant MEKF reset Jacobian block:
        G_theta = I3 - 0.5 * skew(dtheta)
        G[IDX_THETA, IDX_THETA] = G_theta

        P = G @ P @ G.T
        P = 0.5 * (P + P.T)
        self.P = P

    def _kalman(self, H: FloatArray, R: FloatArray, y_residual: FloatArray):
        S = H @ self.P @ H.T + R

        K = self.P @ H.T @ solve(S, np.eye(S.shape[0]))

        dx_correction = K @ y_residual

        I = np.eye(self.P.shape[0])

        # Joseph stabilized covariance update
        IKH = I - K @ H
        P_post = IKH @ self.P @ IKH.T + K @ R @ K.T

        # Symmetrize to reduce numerical asymmetry
        P = 0.5 * (P_post + P_post.T)

        return dx_correction, P

    def _form_F_G(self, accel_B: Vec3, omega_B: Vec3):
        R = qu.as_rotation_matrix(self.state.q_BI)
        A_g = J_gravity(self.state.r_I)
        skew_a = skew(accel_B)
        skew_w = skew(omega_B)

        # dx = [dr, dv, dtheta, dbg, dba]
        # form dx_dot = F@dx + G@noise
        F = np.block(
            [
                [z3, I3, z3, z3, z3],
                [A_g, z3, R @ skew_a, z3, -R],
                [z3, z3, -skew_w, -I3, z3],
                [z3, z3, z3, z3, z3],
                [z3, z3, z3, z3, z3],
            ]
        )
        # noise = [n_g, n_a, n_bg, n_ba].T
        G = np.block(
            [
                [z3, z3, z3, z3],
                [z3, -R, z3, z3],
                [-I3, z3, z3, z3],
                [z3, z3, I3, z3],
                [z3, z3, z3, I3],
            ]
        )
        return F, G

    def _form_phi_Q(self, F: FloatArray, G: FloatArray, Qc: FloatArray, dt: float):
        """
        Continuous: xdot = F x + G w,  E[w(t) w(tau)^T] = Qc * delta(t-tau)
        Returns:
        Phi: discrete state transition
        Q_d:  discrete process noise covariance
        """
        n = F.shape[0]
        A = np.zeros((2 * n, 2 * n))
        A[:n, :n] = F
        A[:n, n:] = G @ Qc @ G.T
        A[n:, n:] = -F.T

        M = expm(A * dt)
        Phi = M[:n, :n]
        Q_d = Phi @ M[:n, n:]

        # Symmetrize to reduce numerical asymmetry
        Q_d = 0.5 * (Q_d + Q_d.T)
        return Phi, Q_d
