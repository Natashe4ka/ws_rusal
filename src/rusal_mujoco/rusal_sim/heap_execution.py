"""Ground-truth candidate selection and physical extraction from a settled detailed heap.
Uses a dynamic fixture, not a KUKA or ideal object attachment. All neighbors remain free.
"""

import json, csv
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from . import ROOT, configuration
from .scene import tool, xml, s
from .heap import restore, save
from .vacuum import Vacuum
from .experiment import smoothstep
from .pallet import targets


class HeapExecution:
    def __init__(self, source="results/logs/heap10_detailed", cfg=None):
        source = Path(source)
        self.cfg = configuration() if cfg is None else __import__("copy").deepcopy(cfg)
        c = self.cfg
        self.params = c["heap_execution"]
        stored = json.loads(str(np.load(source.with_suffix(".npz"))["config"]))
        c["ingot"] = stored["ingot"]
        c["seed"] = stored.get("seed", 42)
        if "ingots" in stored:
            c["ingots"] = stored["ingots"]
        else:
            c.pop("ingots", None)
        report_path = source.with_suffix(".json")
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report.get("count", 0) > 25 and not report.get("validated_equilibrium", False):
                raise ValueError("Large heap needs independent equilibrium validation")
            if (
                not report.get("settled", False)
                or report.get("validated_equilibrium", True) is False
            ):
                raise ValueError("Heap has not passed stability checks")
        original = mujoco.MjModel.from_xml_path(str(source.with_suffix(".xml")))
        old = restore(original, source.with_suffix(".npz"))
        tree = E.parse(source.with_suffix(".xml")).getroot()
        pg = tree.find("worldbody/geom[@name='pallet']")
        pg.set("pos", s([*c["pallet"]["center"], c["pallet"]["height"] / 2]))
        tool(tree, c)
        self.tree = tree
        self.m = mujoco.MjModel.from_xml_string(xml(tree))
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        d.qpos[: original.nq] = old.qpos
        d.qvel[: original.nv] = old.qvel
        self.bodies = [i for i in range(m.nbody) if m.body(i).name.startswith("ingot_")]
        # Initial cell layout transform, applied once before dynamics starts.
        for b in self.bodies:
            j = m.body_jntadr[b]
            adr = m.jnt_qposadr[j]
            d.qpos[adr : adr + 2] += self.params["heap_shift"]
        self.axes = ["x", "y", "z", "rx", "ry", "rz"]
        self.qadr = [m.joint("fixture_" + a).qposadr[0] for a in self.axes]
        self.command_pos = np.array([0.0, 0.0, self.params["home_height"]])
        self.command_R = np.eye(3)
        self.euler = np.zeros(3)
        d.qpos[self.qadr[:3]] = [0, 0, self.params["home_height"] - c["fixture"]["initial_height"]]
        self.vacuum = None
        self.selected = None
        self.placed = set()
        self.removed = set()
        self.slot_cursor = 0
        self.rows = []
        self.phase = "IDLE"
        self.attempts = []
        self.renderer = self.writer = None
        self.next_frame = 0
        self.next_sample = 0
        self.neighbor_reference = {}
        self.neighbor_max = 0.0
        mujoco.mj_forward(m, d)

    def candidates(self):
        m, d = self.m, self.d
        result = []
        h = self.cfg["ingot"]["thickness"]
        gid = np.zeros(1, dtype=np.int32)
        radius = self.cfg["vacuum"]["diameter"] / 2
        footprint = np.array(
            [
                [x + dx, dy, 0]
                for x in self.cfg["vacuum"]["cup_x"]
                for dx, dy in [(0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius)]
            ]
        )
        for b in self.bodies:
            if b in self.removed:
                continue
            R = d.xmat[b].reshape(3, 3)
            n = -R[:, 2]
            if n[2] < self.params["minimum_face_cosine"]:
                continue
            headR = R @ np.diag([1, -1, -1])
            face = d.xpos[b] + n * h / 2
            clear = True
            for off in footprint:
                point = face + headR @ off
                distance = mujoco.mj_ray(
                    m,
                    d,
                    point + n * self.params["ray_clearance"],
                    -n,
                    np.ones(6, dtype=np.uint8),
                    1,
                    -1,
                    gid,
                )
                if distance < 0 or m.geom_bodyid[gid[0]] != b:
                    clear = False
                    break
            if clear:
                result.append((float(face[2]), b, face, headR))
        return sorted(result, reverse=True, key=lambda v: v[0])

    def advance(self, pos, R, seconds, vac=False, jaws=False):
        m, d, c = self.m, self.d, self.cfg
        p0 = self.command_pos.copy()
        R0 = self.command_R.copy()
        interp = Slerp([0, 1], Rotation.from_matrix([R0, R]))
        for i in range(round(seconds / m.opt.timestep)):
            u = smoothstep((i + 1) / (seconds / m.opt.timestep))
            p = p0 + (pos - p0) * u
            rr = interp(float(u)).as_matrix()
            angles = Rotation.from_matrix(rr).as_euler("XYZ")
            angles = self.euler + (angles - self.euler + np.pi) % (2 * np.pi) - np.pi
            self.euler = angles
            values = [p[0], p[1], p[2] - c["fixture"]["initial_height"], *angles]
            if hasattr(self, "_drive_pose"):
                self._drive_pose(p, rr)
            else:
                for a, value in zip(self.axes, values):
                    d.ctrl[m.actuator("fixture_motor_" + a).id] = value
            if self.vacuum:
                self.vacuum.update(m, d, vac)
            if hasattr(self, "_jaw_setpoint"):
                jaw_target = self._jaw_setpoint(jaws)
            else:
                jaw_target = c["jaws"]["closed_travel"] if jaws else 0
            for side in ["left", "right"] if c["jaws"].get("enabled", True) else []:
                if not c["jaws"].get("fixed", False):
                    d.ctrl[m.actuator("deploy_motor_" + side).id] = (
                        self._deploy_setpoint(jaws, side)
                        if hasattr(self, "_deploy_setpoint")
                        else c["jaws"]["deploy_travel"]
                        if jaws
                        else 0
                    )
                j = m.joint("jaw_" + side)
                target = jaw_target
                force = (
                    c["jaws"]["kp"] * (target - d.qpos[j.qposadr[0]])
                    - c["jaws"]["damping"] * d.qvel[j.dofadr[0]]
                )
                confirmed = bool(
                    self.attempts
                    and (
                        self.attempts[-1].get("clamped")
                        or self.attempts[-1].get("clamped_after_lift")
                    )
                )
                if (
                    jaws
                    and not getattr(self, "deploy_only", False)
                    and confirmed
                    and c["jaws"].get("hold_force_per_jaw")
                ):
                    force = (
                        c["jaws"]["hold_force_per_jaw"] - c["jaws"]["damping"] * d.qvel[j.dofadr[0]]
                    )
                d.ctrl[m.actuator("jaw_motor_" + side).id] = np.clip(
                    force, -c["jaws"]["force_per_jaw"], c["jaws"]["force_per_jaw"]
                )
            mujoco.mj_step(m, d)
            if hasattr(self, "_after_step"):
                self._after_step()
            if d.time >= self.next_sample:
                row = dict(
                    time=float(d.time),
                    phase=self.phase,
                    selected=m.body(self.selected).name if self.selected is not None else "",
                    load_x=float(d.xpos[self.selected, 0]) if self.selected else 0,
                    load_y=float(d.xpos[self.selected, 1]) if self.selected else 0,
                    load_z=float(d.xpos[self.selected, 2]) if self.selected else 0,
                    pressure_pa=self.vacuum.mean_pressure if self.vacuum else 0,
                    seal=self.vacuum.mean_quality if self.vacuum else 0,
                )
                if self.selected is not None:
                    for key, values in [
                        ("qpos", d.qpos),
                        ("qvel", d.qvel),
                        ("qacc", d.qacc),
                        ("actuator_force", d.actuator_force),
                        ("load_velocity", d.cvel[self.selected]),
                    ]:
                        row.update({f"{key}_{j}": float(value) for j, value in enumerate(values)})
                    row["vacuum_capacity_n"] = (
                        float(np.sum(self.vacuum.capacity)) if self.vacuum else 0.0
                    )
                    contact_force = np.zeros(6)
                    row["contact_force_n"] = 0.0
                    for ci, cc in enumerate(d.contact):
                        if self.selected in m.geom_bodyid[[cc.geom1, cc.geom2]]:
                            mujoco.mj_contactForce(m, d, ci, contact_force)
                            row["contact_force_n"] += float(np.linalg.norm(contact_force[:3]))
                self.neighbor_max = max(
                    self.neighbor_max,
                    max(
                        (
                            float(np.linalg.norm(d.xpos[b] - p))
                            for b, p in self.neighbor_reference.items()
                        ),
                        default=0,
                    ),
                )
                self.rows.append(row)
                self.next_sample += self.params["sample_period"]
            if self.writer and d.time >= self.next_frame:
                from .visualization import status_frame

                self.renderer.update_scene(d, camera=self.camera)
                self.writer.append_data(
                    status_frame(
                        self.renderer.render(),
                        m,
                        d,
                        self.vacuum,
                        self.phase,
                        getattr(self, "display_mode", "HEAP FIXTURE"),
                        m.body(self.selected).name,
                        c.get("gripper", {}).get("video_note", ""),
                    )
                )
                self.next_frame += 1 / 30
        self.command_pos = np.array(pos)
        self.command_R = np.array(R)

    def execute(self, max_attempts=3, out="results/logs/heap_execution", video=None):
        if video:
            import imageio.v2 as imageio

            self.renderer = mujoco.Renderer(self.m, 720, 960)
            self.writer = imageio.get_writer(str(video), fps=30)
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.camera)
            self.camera.lookat[:] = [0.25, -0.4, 0.2]
            self.camera.distance = 4.0
            self.camera.azimuth = 125
            self.camera.elevation = -50
        try:
            return self._execute(max_attempts, out)
        finally:
            if self.writer:
                self.writer.close()
            if self.renderer:
                self.renderer.close()
            self.writer = self.renderer = None

    def _execute(self, max_attempts=3, out="results/logs/heap_execution"):
        m, d, c = self.m, self.d, self.cfg
        p = self.params
        t = p["timing"]
        rejected = set()
        for _ in range(max_attempts):
            if self.slot_cursor >= len(targets(c)):
                break
            candidates = [v for v in self.candidates() if v[1] not in rejected]
            if not candidates:
                break
            _, b, face, R = candidates[0]
            print("Attempt", m.body(b).name, flush=True)
            self.selected = b
            name = m.body(b).name
            self.vacuum = Vacuum(m, c, name)
            neighbors = {i: d.xpos[i].copy() for i in self.bodies if i != b}
            self.neighbor_reference = neighbors
            self.neighbor_max = 0.0
            initial = d.xpos[b].copy()
            n = R[:, 2]
            offset = (
                c["ingot"]["thickness"] / 2
                - c["vacuum"]["pad_center_z"]
                + c["vacuum"]["pad_half_thickness"]
                - c["fixture"]["preload"]
            )
            contact = face + n * (offset - c["ingot"]["thickness"] / 2)
            self.phase = "APPROACH_CLEARANCE"
            self.advance(np.array([contact[0], contact[1], p["safe_height"]]), R, t["clearance"])
            self.phase = "APPROACH"
            self.advance(contact, R, t["approach"], vac=True)
            self.phase = "CHECK_VACUUM"
            self.advance(contact, R, t["seal_wait"], vac=True)
            sealed = (
                np.count_nonzero(self.vacuum.quality > p["minimum_seal_quality"])
                >= p["minimum_live_cups"]
                and np.mean(self.vacuum.pressures) > p["minimum_pressure_pa"]
            )
            self.phase = "EXTRACT"
            lift = contact + [0, 0, p["lift_height"]]
            if sealed:
                self.advance(lift, R, t["extract"], vac=True)
            shift = max(
                (float(np.linalg.norm(d.xpos[i] - p)) for i, p in neighbors.items()), default=0
            )
            lifted = sealed and d.xpos[b, 2] - initial[2] > p["minimum_lift"]
            result = dict(
                selected=name,
                sealed=bool(sealed),
                lifted=bool(lifted),
                extraction_passed=bool(lifted and shift < p["maximum_neighbor_shift"]),
                neighbor_displacement_m=shift,
                success=False,
            )
            if not lifted or shift > p["maximum_neighbor_shift"]:
                result["failure"] = "seal/lift failure" if not lifted else "neighbor disturbance"
                self.phase = "ABORT"
                self.advance(self.command_pos + [0, 0, 0.15], R, t["abort"], vac=False)
                rejected.add(b)
                self.attempts.append(result)
                continue
            self.phase = "DEPLOY_JAWS"
            self.advance(lift, R, t["deploy"], vac=True, jaws=True)
            target = targets(c)[self.slot_cursor]
            self.slot_cursor += 1
            result["pallet_slot"] = self.slot_cursor - 1
            x, y, z = target["position"]
            yaw = -np.pi / 2 if target["row"] == 0 else np.pi / 2
            finalR = Rotation.from_euler("z", yaw).as_matrix()
            self.phase = "RAISE"
            self.advance(
                np.array([lift[0], lift[1], p["safe_height"]]), R, t["raise"], vac=True, jaws=True
            )
            self.phase = "TRANSPORT"
            self.advance(
                np.array([x, y, p["safe_height"]]), finalR, t["transport"], vac=True, jaws=True
            )
            result["transport_passed"] = bool(
                np.linalg.norm(d.xpos[b] - np.array([x, y, p["safe_height"] - offset]))
                < p["maximum_transport_error"]
            )
            self.phase = "PLACE"
            place = np.array([x, y, z + offset])
            self.advance(place, finalR, t["lower"], vac=True, jaws=False)
            supported = any(
                m.geom("pallet").id in [cc.geom1, cc.geom2]
                and b in m.geom_bodyid[[cc.geom1, cc.geom2]]
                for cc in d.contact
            )
            self.phase = "RELEASE"
            self.advance(place, finalR, t["release"], vac=False, jaws=False)
            self.phase = "RETRACT"
            self.advance(np.array([x, y, p["safe_height"]]), finalR, t["retract"], vac=False)
            self.phase = "SETTLE"
            self.advance(self.command_pos, finalR, t["settle"], vac=False)
            error = float(np.linalg.norm(d.xpos[b] - target["position"]))
            orientation = float(
                np.rad2deg(
                    Rotation.from_matrix(
                        (finalR @ np.diag([1, -1, -1])).T @ d.xmat[b].reshape(3, 3)
                    ).magnitude()
                )
            )
            placement_passed = bool(
                supported
                and error < c["experiment"]["positioning_tolerance"]
                and orientation < c["experiment"]["orientation_tolerance_deg"]
            )
            result.update(
                placement_passed=placement_passed,
                max_neighbor_displacement_m=self.neighbor_max,
                supported=bool(supported),
                position_error_m=error,
                orientation_error_deg=orientation,
                success=bool(
                    placement_passed
                    and result["transport_passed"]
                    and self.neighbor_max < p["maximum_neighbor_shift"]
                ),
            )
            physically_deposited = bool(
                abs(d.xpos[b, 0] - x) < 0.1
                and abs(d.xpos[b, 1] - y) < 0.1
                and d.xpos[b, 2] > c["pallet"]["height"]
            )
            result["physically_deposited"] = physically_deposited
            if physically_deposited:
                self.removed.add(b)
            if result["success"]:
                self.placed.add(b)
            else:
                rejected.add(b)
            self.attempts.append(result)
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".xml").write_text(xml(self.tree))
        save(m, d, path.with_suffix(".npz"), c)
        report = dict(
            attempts=self.attempts,
            successes=sum(r["success"] for r in self.attempts),
            physically_deposited=len(self.removed),
            remaining=len(self.bodies) - len(self.removed),
            scope="dynamic fixture extraction and placement from a physically deposited detailed heap; not KUKA heap execution",
            warnings=[int(w.number) for w in d.warning],
        )
        path.with_suffix(".json").write_text(json.dumps(report, indent=2))
        if self.rows:
            with path.with_suffix(".csv").open("w") as f:
                w = csv.DictWriter(f, fieldnames=self.rows[0])
                w.writeheader()
                w.writerows(self.rows)
        return report
