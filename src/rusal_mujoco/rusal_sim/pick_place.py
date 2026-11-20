"""Contact-driven fixture pick-place. Does not claim a validated KUKA trajectory."""

import json
import csv
from pathlib import Path
import numpy as np
import mujoco
from .experiment import Experiment, smoothstep
from .scene import ingot, xml
from .vacuum import Vacuum
from .pallet import targets


class PickPlace(Experiment):
    def __init__(self, cfg, neighbors=True):
        super().__init__(cfg, "hybrid", "smooth")
        if neighbors:
            for i, (x, y) in enumerate([(0, 0.215), (0, -0.215), (0.415, 0), (-0.415, 0)]):
                ingot(self.tree, cfg["ingot"], f"neighbor_{i}", pos=(x, y, 0.021))
        self.m = mujoco.MjModel.from_xml_string(xml(self.tree))
        self.d = mujoco.MjData(self.m)
        self.vacuum = Vacuum(self.m, cfg)
        self.state = "APPROACH"
        self.start = 0.0
        self.origin = np.zeros(4)
        self.command = np.array([0.0, 0.0, 0.12, 0.0])
        self.origin = self.command.copy()
        self.released = False
        self.target = targets(cfg)[6]
        self.history = []
        self.abort = None
        self.neighbor_initial = None

    def transition(self, name):
        self.history.append(dict(time=self.d.time, state=name))
        self.state = name
        self.start = self.d.time
        self.origin = self.command.copy()

    def move(self, target, duration):
        f = smoothstep((self.d.time - self.start) / duration)
        self.command = self.origin + (np.array(target) - self.origin) * f
        return self.d.time - self.start >= duration

    def step(self):
        m, d = self.m, self.d
        c = self.cfg
        t = d.time
        elapsed = t - self.start
        b = m.body("ingot").id
        if self.neighbor_initial is None:
            mujoco.mj_forward(m, d)
            self.neighbor_initial = {
                i: d.xpos[i].copy() for i in range(m.nbody) if m.body(i).name.startswith("neighbor")
            }
        x, y, z = self.target["position"]
        yaw = np.pi / 2
        if self.state == "APPROACH":
            if self.move([0, 0, 0.064, 0], 0.8):
                self.transition("VACUUM_CONTACT")
        elif self.state == "VACUUM_CONTACT":
            if np.mean(self.vacuum.pressures) > 40000 and np.min(self.vacuum.quality) > 0.5:
                self.transition("INITIAL_LIFT")
            elif elapsed > 1:
                self.abort = "seal failure"
                self.transition("ABORT")
        elif self.state == "INITIAL_LIFT":
            if self.move([0, 0, 0.15, 0], 1):
                if d.xpos[b, 2] > 0.08:
                    self.transition("DEPLOY_MECHANICAL_JAWS")
                else:
                    self.abort = "lift failure"
                    self.transition("ABORT")
        elif self.state == "DEPLOY_MECHANICAL_JAWS":
            if elapsed > 0.8:
                self.transition("CHECK_GRASP")
        elif self.state == "CHECK_GRASP":
            jawcontact = [False, False]
            for con in d.contact:
                if b not in m.geom_bodyid[[con.geom1, con.geom2]]:
                    continue
                names = [m.geom(con.geom1).name, m.geom(con.geom2).name]
                for i, side in enumerate(["left", "right"]):
                    jawcontact[i] |= f"jaw_pad_{side}" in names
            if all(jawcontact):
                self.transition("RAISE")
            elif elapsed > 1:
                self.abort = "jaw contact missing"
                self.transition("ABORT")
        elif self.state == "RAISE":
            if self.move([0, 0, 0.5, 0], 1.5):
                self.transition("TRANSPORT")
        elif self.state == "TRANSPORT":
            if self.move([x, y, 0.5, yaw], 4):
                self.transition("PLACE")
        elif self.state == "PLACE":
            if self.move([x, y, z + 0.044, yaw], 2):
                self.transition("WAIT_SUPPORT")
        elif self.state == "WAIT_SUPPORT":
            supported = any(
                m.geom("pallet").id in [cc.geom1, cc.geom2]
                and b in m.geom_bodyid[[cc.geom1, cc.geom2]]
                for cc in d.contact
            )
            if supported and elapsed > 0.4:
                self.transition("RELEASE")
                self.released = True
            elif elapsed > 1.5:
                self.abort = "no pallet support"
                self.transition("ABORT")
        elif self.state == "RELEASE":
            if elapsed > 0.8:
                self.transition("RETRACT")
        elif self.state == "RETRACT":
            if self.move([x, y, 0.5, yaw], 1.5):
                self.transition("SETTLE")
        elif self.state == "SETTLE" and elapsed > 1:
            self.transition("DONE")
        if self.state == "ABORT":
            self.released = True
        for axis, value in zip(
            ["x", "y", "z", "rz"],
            [self.command[0], self.command[1], self.command[2] - 0.12, self.command[3]],
        ):
            d.ctrl[m.actuator(f"fixture_motor_{axis}").id] = value
        self.vacuum.update(
            m, d, on=self.state not in ["APPROACH", "ABORT", "DONE"] and not self.released
        )
        deployed = self.state in [
            "DEPLOY_MECHANICAL_JAWS",
            "CHECK_GRASP",
            "RAISE",
            "TRANSPORT",
            "PLACE",
            "WAIT_SUPPORT",
        ]
        for side in ["left", "right"]:
            d.ctrl[m.actuator(f"deploy_motor_{side}").id] = (
                c["jaws"]["deploy_travel"] if deployed else 0
            )
            j = m.joint(f"jaw_{side}")
            target = c["jaws"]["closed_travel"] if deployed else 0
            d.ctrl[m.actuator(f"jaw_motor_{side}").id] = np.clip(
                c["jaws"]["kp"] * (target - d.qpos[j.qposadr[0]])
                - c["jaws"]["damping"] * d.qvel[j.dofadr[0]],
                -c["jaws"]["force_per_jaw"],
                c["jaws"]["force_per_jaw"],
            )
        mujoco.mj_step(m, d)

    def run_cycle(self, out, video=None):
        import imageio.v2 as imageio

        renderer = mujoco.Renderer(self.m, 720, 960) if video else None
        writer = imageio.get_writer(str(video), fps=30) if video else None
        next_sample = next_frame = 0
        try:
            while self.d.time < 25 and self.state not in ["DONE", "ABORT"]:
                self.step()
                if self.d.time >= next_sample:
                    self.sample()
                    next_sample += 0.02
                if renderer and self.d.time >= next_frame:
                    from .visualization import status_frame

                    renderer.update_scene(self.d, camera="overview")
                    writer.append_data(
                        status_frame(
                            renderer.render(),
                            self.m,
                            self.d,
                            self.vacuum,
                            self.state,
                            "HYBRID FIXTURE",
                        )
                    )
                    next_frame += 1 / 30
        finally:
            if writer:
                writer.close()
            if renderer:
                renderer.close()
        p = self.d.xpos[self.m.body("ingot").id]
        error = float(np.linalg.norm(p - np.array(self.target["position"])))
        neighbor_motion = float(
            max(
                (np.linalg.norm(self.d.xpos[i] - v) for i, v in self.neighbor_initial.items()),
                default=0,
            )
        )
        from scipy.spatial.transform import Rotation

        goal = Rotation.from_euler("z", np.pi / 2).as_matrix() @ np.diag([1, -1, -1])
        orientation_error = float(
            np.rad2deg(
                Rotation.from_matrix(
                    goal.T @ self.d.xmat[self.m.body("ingot").id].reshape(3, 3)
                ).magnitude()
            )
        )
        stable = float(np.linalg.norm(self.d.cvel[self.m.body("ingot").id])) < 0.01
        report = dict(
            orientation_error_deg=orientation_error,
            stable=stable,
            state=self.state,
            abort=self.abort,
            position_error_m=error,
            neighbor_displacement_m=float(neighbor_motion),
            history=self.history,
            success=self.state == "DONE"
            and error < self.cfg["experiment"]["positioning_tolerance"]
            and orientation_error < self.cfg["experiment"]["orientation_tolerance_deg"]
            and stable
            and neighbor_motion < 0.005,
            scope="local fixture, five adjacent free ingots; no KUKA transfer",
        )
        path = Path(out)
        path.with_suffix(".json").write_text(json.dumps(report, indent=2))
        path.with_suffix(".xml").write_text(xml(self.tree))
        with path.with_suffix(".csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=self.rows[0])
            w.writeheader()
            w.writerows(self.rows)
        return report
