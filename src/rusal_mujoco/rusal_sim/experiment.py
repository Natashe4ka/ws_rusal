"""Actual mj_step experiments; never writes load pose after initialization."""

import csv
import json
from pathlib import Path
import numpy as np
import mujoco
from .scene import scene, xml
from .vacuum import Vacuum


def smoothstep(t):
    return np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))


class Experiment:
    def __init__(self, cfg, mode="vacuum", surface="smooth", disabled=(), leak=1):
        if mode not in ("vacuum", "mechanical", "hybrid", "off"):
            raise ValueError(mode)
        self.cfg = cfg
        self.mode = mode
        self.surface = surface
        self.disabled = disabled
        self.leak = leak
        self.tree = scene(cfg, surface)
        self.m = mujoco.MjModel.from_xml_string(xml(self.tree))
        self.d = mujoco.MjData(self.m)
        self.vacuum = Vacuum(self.m, cfg)
        self.rows = []
        self.state = "APPROACH"
        self.reference = None
        self.mujoco_warnings = 0

    def step(self):
        m, d = self.m, self.d
        c = self.cfg
        e = c["experiment"]
        t = d.time
        # Smooth fixture trajectory. The fixture has ideal drive authority, not KUKA dynamics.
        initial = c["fixture"]["initial_height"]
        contact = (
            c["ingot"]["thickness"]
            - c["vacuum"]["pad_center_z"]
            + c["vacuum"]["pad_half_thickness"]
            - c["fixture"]["preload"]
        )
        z = initial + (contact - initial) * smoothstep(t / e["approach_end"])
        z += e["lift_height"] * smoothstep(
            (t - e["lift_start"]) / (e["lift_end"] - e["lift_start"])
        )
        d.ctrl[m.actuator("fixture_motor_z").id] = z - initial
        elapsed = max(0, t - e["lift_end"])
        d.ctrl[m.actuator("fixture_motor_x").id] = (
            0.5 * e.get("transport_acceleration", 0) * elapsed**2
        )
        d.ctrl[m.actuator("fixture_motor_ry").id] = np.deg2rad(
            e.get("transport_tilt_deg", 0)
        ) * smoothstep(elapsed / 0.5)
        active = t < e["release_time"]
        self.vacuum.update(
            m,
            d,
            on=self.mode in ("vacuum", "hybrid") and active and t > 0.45,
            disabled=self.disabled,
            leak=self.leak,
        )
        deploy = (self.mode in ("mechanical", "hybrid")) and t > 0.7 and active
        for side in ("left", "right"):
            d.ctrl[m.actuator(f"deploy_motor_{side}").id] = (
                c["jaws"]["deploy_travel"] if deploy else 0
            )
            jid = m.joint(f"jaw_{side}").id
            q = d.qpos[m.jnt_qposadr[jid]]
            qd = d.qvel[m.jnt_dofadr[jid]]
            target = (
                min(c["jaws"]["closed_travel"], max(0, t - 0.85) * c["jaws"]["max_speed"])
                if deploy
                else 0
            )
            force = c["jaws"]["kp"] * (target - q) - c["jaws"]["damping"] * qd
            d.ctrl[m.actuator(f"jaw_motor_{side}").id] = np.clip(
                force, -c["jaws"]["force_per_jaw"], c["jaws"]["force_per_jaw"]
            )
        if t < e["approach_end"]:
            self.state = "APPROACH"
        elif t < e["lift_start"]:
            self.state = "CHECK_GRASP"
        elif not active:
            self.state = "RELEASE"
        elif t < e["lift_end"]:
            self.state = "LIFT"
        else:
            self.state = "HOLD"
        mujoco.mj_step(m, d)

    def sample(self):
        m, d = self.m, self.d
        b = m.body("ingot").id
        pos = d.xpos[b].copy()
        tool = d.xpos[m.body("tool").id].copy()
        rel = d.xmat[m.body("tool").id].reshape(3, 3).T @ (pos - tool)
        if self.reference is None and d.time >= self.cfg["experiment"]["lift_start"]:
            self.reference = rel.copy()
        slip = np.linalg.norm(rel - self.reference) if self.reference is not None else 0
        contacts = 0.0
        jaws = [0.0, 0.0]
        force = np.zeros(6)
        for i, con in enumerate(d.contact):
            mujoco.mj_contactForce(m, d, i, force)
            if b in m.geom_bodyid[[con.geom1, con.geom2]]:
                contacts += np.linalg.norm(force[:3])
            names = [m.geom(int(g)).name or "" for g in (con.geom1, con.geom2)]
            for j, side in enumerate(("left", "right")):
                if any(n.startswith(f"jaw_pad_{side}") for n in names):
                    jaws[j] += abs(force[0])
        row = dict(
            time=d.time,
            state=self.state,
            x=pos[0],
            y=pos[1],
            z=pos[2],
            tool_z=tool[2],
            pressure_pa=float(np.mean(self.vacuum.pressures)),
            seal_quality=float(np.mean(self.vacuum.quality)),
            vacuum_capacity_n=float(sum(self.vacuum.capacity)),
            contact_force_n=contacts,
            jaw_left_n=jaws[0],
            jaw_right_n=jaws[1],
            relative_displacement=slip,
        )
        for name, arr in [
            ("qpos", d.qpos),
            ("qvel", d.qvel),
            ("qacc", d.qacc),
            ("actuator_force", d.actuator_force),
            ("load_velocity", d.cvel[b]),
        ]:
            row.update({f"{name}_{i}": float(v) for i, v in enumerate(arr)})
        self.rows.append(row)

    def run(self, out=None, video=None):
        m, d = self.m, self.d
        e = self.cfg["experiment"]
        next_sample = 0
        next_frame = 0
        renderer = writer = None
        if video:
            import imageio.v2 as imageio

            Path(video).parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(m, height=720, width=960)
            writer = imageio.get_writer(str(video), fps=30)
        try:
            while d.time < e["duration"]:
                self.step()
                if d.time >= next_sample:
                    self.sample()
                    next_sample += e["sample_period"]
                if renderer and d.time >= next_frame:
                    from .visualization import status_frame

                    renderer.update_scene(d, camera="gripper")
                    writer.append_data(
                        status_frame(renderer.render(), m, d, self.vacuum, self.state, self.mode)
                    )
                    next_frame += 1 / 30
        finally:
            if writer:
                writer.close()
            if renderer:
                renderer.close()
        hold = [r for r in self.rows if e["lift_end"] + 0.1 < r["time"] < e["release_time"] - 0.1]
        success = bool(
            hold
            and min(r["z"] for r in hold) > 0.18
            and max(r["relative_displacement"] for r in hold) < e["max_relative_displacement"]
        )
        result = dict(
            mode=self.mode,
            surface=self.surface,
            mass=self.cfg["ingot"]["mass"],
            pressure_pa=self.cfg["vacuum"]["pressure_pa"],
            friction=self.cfg["contacts"]["aluminum_steel"],
            transport_acceleration=e.get("transport_acceleration", 0),
            transport_tilt_deg=e.get("transport_tilt_deg", 0),
            force_per_jaw=self.cfg["jaws"]["force_per_jaw"],
            disabled_cups=list(self.disabled),
            leak=self.leak,
            timestep=m.opt.timestep,
            success=success,
            max_z=max(r["z"] for r in self.rows),
            final_z=self.rows[-1]["z"],
            max_hold_slip=max((r["relative_displacement"] for r in hold), default=None),
            warnings={str(i): int(w.number) for i, w in enumerate(d.warning) if w.number},
            scope="local driven-fixture contact dynamics; not robot reachability",
        )
        if out:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with Path(str(out) + ".csv").open("w") as f:
                w = csv.DictWriter(f, fieldnames=self.rows[0])
                w.writeheader()
                w.writerows(self.rows)
            Path(str(out) + ".json").write_text(json.dumps(result, indent=2))
            Path(str(out) + ".xml").write_text(xml(self.tree))
        return result
