"""Numerical IK and sampled collision checks. Not a substitute for MoveIt continuous checks."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import mujoco


class Kinematics:
    def __init__(self, m):
        self.m = m
        self.d = mujoco.MjData(m)
        self.joints = [m.joint(f"joint_{i}").id for i in range(1, 7)]
        self.qa = m.jnt_qposadr[self.joints]
        self.va = m.jnt_dofadr[self.joints]
        self.limits = m.jnt_range[self.joints]
        self.flange = m.body("flange").id

    def ik(self, position, rotation=None, start=None, seed=42, attempts=8):
        rotation = np.diag([1, -1, -1]) if rotation is None else rotation

        def residual(q):
            self.d.qpos[self.qa] = q
            mujoco.mj_forward(self.m, self.d)
            return np.r_[
                self.d.xpos[self.flange] - position,
                Rotation.from_matrix(
                    rotation.T @ self.d.xmat[self.flange].reshape(3, 3)
                ).as_rotvec()
                * 0.3,
            ]

        rng = np.random.default_rng(seed)
        best = None
        for n in range(attempts):
            q0 = (
                np.clip(start, self.limits[:, 0] + 1e-8, self.limits[:, 1] - 1e-8)
                if n == 0 and start is not None
                else rng.uniform(self.limits[:, 0], self.limits[:, 1])
            )
            fit = least_squares(
                residual,
                q0,
                bounds=(self.limits[:, 0], self.limits[:, 1]),
                max_nfev=150,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            err = np.linalg.norm(residual(fit.x))
            if best is None or err < best[0]:
                best = (err, fit.x)
            if err < 1e-5:
                return fit.x
        return None

    def collisions(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_forward(self.m, self.d)
        return [
            (self.m.geom(c.geom1).name, self.m.geom(c.geom2).name, float(c.dist))
            for c in self.d.contact
            if c.dist < -0.001
        ]

    def path(self, start, goal, steps=100):
        qs = np.linspace(start, goal, steps)
        for q in qs:
            if self.collisions(q):
                return None
        return qs


def reachability(m, points):
    k = Kinematics(m)
    result = []
    for p in points:
        # Flange to TCP length is 225 mm, along flange +Z.
        q = k.ik(np.array(p) + [0, 0, 0.225])
        result.append(
            dict(
                tcp=list(p),
                ik=q.tolist() if q is not None else None,
                collision_free=bool(q is not None and not k.collisions(q)),
            )
        )
    return result
