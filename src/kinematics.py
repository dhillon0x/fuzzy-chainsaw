import numpy as np
from scipy.optimize import least_squares

class ArmKinematics:
    def __init__(self):
        self.d1 = 150.0   # Base to shoulder offset (mm)
        self.a2 = 200.0   # Upper arm length (mm)
        self.a3 = 180.0   # Forearm length (mm)
        self.d4 = 50.0    # Wrist offset (mm)
        self.d6 = 70.0    # Tool mount offset (mm)

        self.joint_limits = [
            (-170, 170),  # J1: Base Yaw
            (-60, 120),   # J2: Shoulder Pitch
            (-110, 110),  # J3: Elbow Pitch
            (-180, 180),  # J4: Wrist Roll
            (-90, 90),    # J5: Wrist Pitch
            (-360, 360)   # J6: Tool Roll
        ]

    def dh_transform(self, alpha, a, d, theta):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,      sa,       ca,      d],
            [0.0,     0.0,      0.0,    1.0]
        ], dtype=np.float64)

    def forward_kinematics(self, joint_angles_deg):
        q = np.radians(joint_angles_deg)
        dh_params = [
            (np.pi / 2,      0.0, self.d1, q[0]),
            (0.0,        self.a2,     0.0, q[1]),
            (np.pi / 2,      0.0,     0.0, q[2]),
            (-np.pi / 2,     0.0, self.d4, q[3]),
            (np.pi / 2,      0.0,     0.0, q[4]),
            (0.0,            0.0, self.d6, q[5]),
        ]
        t_matrix = np.eye(4, dtype=np.float64)
        for alpha, a, d, theta in dh_params:
            t_matrix = t_matrix @ self.dh_transform(alpha, a, d, theta)
        return t_matrix

    def inverse_kinematics(self, target_xyz, initial_guess_deg=None):
        target = np.array(target_xyz, dtype=np.float64)
        x0 = np.zeros(6) if initial_guess_deg is None else np.radians(initial_guess_deg)
        lower_bounds = [np.radians(limit[0]) for limit in self.joint_limits]
        upper_bounds = [np.radians(limit[1]) for limit in self.joint_limits]

        def error_function(q):
            current_t = np.eye(4)
            dh_params = [
                (np.pi / 2,      0.0, self.d1, q[0]),
                (0.0,        self.a2,     0.0, q[1]),
                (np.pi / 2,      0.0,     0.0, q[2]),
                (-np.pi / 2,     0.0, self.d4, q[3]),
                (np.pi / 2,      0.0,     0.0, q[4]),
                (0.0,            0.0, self.d6, q[5]),
            ]
            for alpha, a, d, theta in dh_params:
                current_t = current_t @ self.dh_transform(alpha, a, d, theta)
            return current_t[:3, 3] - target

        res = least_squares(error_function, x0, bounds=(lower_bounds, upper_bounds), method='trf', ftol=1e-4, xtol=1e-4)
        return np.degrees(res.x) if res.success else None
