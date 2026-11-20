"""Autonomous physical KUKA cell. Perception, IK and servo drivers are mocks.

Only actuator controls change during execution. All loads retain six free DOFs.
Ground-truth perception is deliberately separated from contact dynamics.
"""

import copy
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from . import ROOT
from .heap_execution import HeapExecution
from .scene import root, xml
from .robot import append_robot
from .planning import Kinematics
from .vacuum import Vacuum
from .heap import save
from .pallet import targets, target_rotation
from .batch import ingot_config
from .geometry import components
from .cross_packing import choose_target, find_target


class MotionFailure(RuntimeError):
    pass


class ViewerClosed(Exception):
    pass


class PhysicalCell(HeapExecution):
    def __init__(
        self,
        source="results/logs/heap10_detailed",
        layers=1,
        mode="hybrid",
        cfg=None,
        prefill_layers=0,
    ):
        super().__init__(source, cfg)
        self.source = str(source)
        self.settings = self.cfg["cell"]
        self.cfg["solver"]["sleep"] = self.settings.get("sleep", False)
        self.params["safe_height"] = self.settings["safe_height"]
        self.cfg["pallet"]["center"] = self.settings["pallet_center"]
        self.cfg["robot"]["position_kp"] = self.settings["mock_position_kp"]
        self.cfg["robot"]["position_kv"] = self.settings["mock_position_kv"]
        self.mode = mode
        profile = self.cfg.get("gripper", {})
        if profile.get("mode") and mode != profile["mode"]:
            raise ValueError(f"This gripper requires --mode {profile['mode']}")
        self.display_mode = profile.get("label", "KUKA / MOCK CONTROL / " + mode.upper())
        self.layers = layers
        self.slots = targets(self.cfg, layers=layers)
        if not 0 <= prefill_layers < layers:
            raise ValueError("prefill-layers must be nonnegative and smaller than layers")
        if prefill_layers and self.cfg["pallet"].get("layout") in ("brick", "cross"):
            raise ValueError("Variable brick batch requires an empty pallet")
        self.prefill_count = len(self.slots) // layers * prefill_layers
        # Rebuild the very same settled loads with the real articulated robot.
        r = root(self.cfg)
        for asset in self.tree.find("asset"):
            r.find("asset").append(copy.deepcopy(asset))
        for b in self.bodies:
            body = copy.deepcopy(self.tree.find(f"worldbody/body[@name='{self.m.body(b).name}']"))
            body.set("pos", " ".join(map(str, self.d.xpos[b])))
            body.set("quat", " ".join(map(str, self.d.xquat[b])))
            r.find("worldbody").append(body)
        from .scene import ingot

        for i, target in enumerate(self.slots[: self.prefill_count]):
            rotation = Rotation.from_matrix(target_rotation(target)) * Rotation.from_euler(
                "x", np.pi
            )
            q = rotation.as_quat()
            ingot(
                r,
                self.cfg["ingot"],
                name=f"ingot_prefill_{i}",
                pos=np.asarray(target["position"]) + [0, 0, 0.001],
                quat=[q[3], *q[:3]],
            )
        append_robot(r, cfg=self.cfg, base=self.cfg["robot"]["base"])
        casing = r.find(".//geom[@name='tool_case']")
        casing.set("contype", "1")
        casing.set("conaffinity", "1")
        self.tree = r
        self.m = mujoco.MjModel.from_xml_string(xml(r))
        self.d = mujoco.MjData(self.m)
        self.bodies = [b for b in range(self.m.nbody) if self.m.body(b).name.startswith("ingot_")]
        self.removed = {b for b in self.bodies if self.m.body(b).name.startswith("ingot_prefill_")}
        self.slot_cursor = self.prefill_count
        # Collision-only planning copy: suppress load-load and load-floor pairs
        # here, while retaining every robot/load/environment collision shape.
        # The dynamic model above keeps every original contact pair.
        planner_model = mujoco.MjModel.from_xml_string(xml(r))
        self_clearance = self.settings.get("tool_robot_planning_clearance", 0.0)
        for gid in range(planner_model.ngeom):
            if planner_model.geom(gid).name.startswith("tool_case"):
                # Planning only: near contacts, without changing physical forces.
                planner_model.geom_margin[gid] = max(planner_model.geom_margin[gid], self_clearance)
                planner_model.geom_gap[gid] = max(planner_model.geom_gap[gid], self_clearance)
        for gid in range(planner_model.ngeom):
            body_name = planner_model.body(planner_model.geom_bodyid[gid]).name
            if planner_model.geom_contype[gid] == 0 and planner_model.geom_conaffinity[gid] == 0:
                continue
            category = (
                2
                if body_name.startswith("ingot_")
                else 4
                if planner_model.geom_bodyid[gid] == 0
                else 1
            )
            planner_model.geom_contype[gid] = category
            planner_model.geom_conaffinity[gid] = 7 if category == 1 else 1
        self.k = Kinematics(planner_model)
        self.axes = []  # No Cartesian fixture exists in this model.
        self.command_pos = np.array(self.settings["home_tool_position"])
        self.command_R = np.eye(3)
        self.next_ik = 0.0
        self.q_command = None
        flange, R = self.flange_target(self.command_pos, self.command_R)
        for seed in range(30):
            q = self.k.ik(
                flange,
                R,
                seed=seed,
                start=np.array(self.settings["initial_joint_seed"]) if seed == 0 else None,
            )
            if q is not None:
                contacts = self.k.collisions(q)
                if not any(
                    any(
                        n.startswith("kuka_") or n.startswith("tool_case") or n == "vmg_adapter"
                        for n in pair[:2]
                    )
                    for pair in contacts
                ):
                    self.q_command = q
                    break
        if self.q_command is None:
            raise MotionFailure("No collision-free initial robot configuration")
        self.d.qpos[self.k.qa] = self.q_command  # Initialization only.
        mujoco.mj_forward(self.m, self.d)
        self.viewer = None
        self.paused = False
        self.speed = 1.0
        self.next_sync = 0.0
        self.wall_start = time.monotonic()
        self.wall_anchor = self.wall_start
        self.sim_anchor = 0.0
        self.robot_collisions = set()
        self.max_effort_ratio = 0.0
        self.max_speed_ratio = 0.0
        import xml.etree.ElementTree as E
        from .robot import URDF

        u = E.parse(URDF).getroot()
        self.speed_limits = np.array(
            [float(u.find(f"joint[@name='joint_{i}']/limit").get("velocity")) for i in range(1, 7)]
        )
        self.effort_limits = np.array(
            [float(u.find(f"joint[@name='joint_{i}']/limit").get("effort")) for i in range(1, 7)]
        )
        self.robot_actuators = [self.m.actuator(f"joint_{i}_position").id for i in range(1, 7)]
        self.abort_reason = None
        self.planner_load = None
        self.deploy_only = False
        self.jaw_command = 0.0
        self.next_monitor = 0.0

    @classmethod
    def resume(cls, source, mode=None, recover_held=False):
        """Continue a saved cell; load poses and velocities come from its real state."""
        import tempfile
        import xml.etree.ElementTree as E
        from .heap import restore

        source = Path(source)
        report = json.loads(source.with_suffix(".json").read_text())
        safe = report.get(
            "safe_to_resume",
            report.get("complete", False)
            or report.get("stop_reason")
            in ("no_accessible_grasp_or_attempt_limit", "pallet_full", "heap_empty"),
        )
        empty_approach = report.get("phase") in ("APPROACH_CLEARANCE", "APPROACH")
        held_recovery = recover_held and report.get("phase") in (
            "VERIFY_JAWS_VACUUM_OFF",
            "TRANSPORT",
            "RAISE",
        )
        if not safe and not empty_approach and not held_recovery:
            raise ValueError(
                "Resume needs a released-load checkpoint. This snapshot stopped mid-grasp; inspect it with rusal-sim view or start a fresh run."
            )
        cfg = json.loads(str(np.load(source.with_suffix(".npz"))["config"]))
        original_tree = E.parse(source.with_suffix(".xml")).getroot()
        original = mujoco.MjModel.from_xml_path(str(source.with_suffix(".xml")))
        state = restore(original, source.with_suffix(".npz"))
        if held_recovery:
            name = report["attempts"][-1]["selected"]
            target = original.body(name).id
            awake = mujoco.MjData(original)
            awake.qpos[:] = state.qpos
            mujoco.mj_forward(original, awake)
            touching = set()
            for cc in awake.contact:
                for pad, other in ((cc.geom1, cc.geom2), (cc.geom2, cc.geom1)):
                    if original.geom_bodyid[other] == target and cc.dist <= 0.001:
                        touching.add(original.geom(pad).name)
            if (
                not all("jaw_pad_" + side in touching for side in ("left", "right"))
                or any(report.get("warnings", []))
                or report.get("robot_collisions")
            ):
                raise ValueError(
                    "Held recovery requires bilateral physical jaw contact and a healthy robot state"
                )
        if not safe and not held_recovery:
            # An empty-tool approach may restart at its measured pose; a held
            # load may not. Verify physical controls/joints/contacts, not a label.
            empty = all(
                state.qpos[original.joint("jaw_" + side).qposadr[0]] < 0.001
                for side in ("left", "right")
                if cfg["jaws"].get("enabled", True)
            )
            empty = empty and all(
                state.ctrl[a] < 1e-6
                for a in range(original.nu)
                if original.actuator(a).name.startswith("vacuum_")
            )
            awake = mujoco.MjData(original)
            awake.qpos[:] = state.qpos
            mujoco.mj_forward(original, awake)
            for contact in awake.contact:
                names = [original.geom(g).name for g in (contact.geom1, contact.geom2)]
                if (
                    contact.dist <= 0.001
                    and any(n.startswith(("cup_pad_", "jaw_pad_", "tool_case")) for n in names)
                    and any(
                        original.body(original.geom_bodyid[g]).name.startswith("ingot_")
                        for g in (contact.geom1, contact.geom2)
                    )
                ):
                    empty = False
            if not empty:
                raise ValueError(
                    "Approach checkpoint is not an empty tool; cannot resume mid-grasp"
                )
        clean = root(cfg)
        for asset in original_tree.find("asset"):
            if not asset.get("name", "").startswith("kuka_"):
                clean.find("asset").append(copy.deepcopy(asset))
        for body in original_tree.findall("worldbody/body"):
            name = body.get("name", "")
            if name.startswith("ingot_"):
                item = copy.deepcopy(body)
                bid = original.body(name).id
                item.set("pos", " ".join(map(str, state.xpos[bid])))
                item.set("quat", " ".join(map(str, state.xquat[bid])))
                clean.find("worldbody").append(item)
        cfg["heap_execution"]["heap_shift"] = [0, 0]
        with tempfile.TemporaryDirectory(prefix="rusal_resume_") as temporary:
            path = Path(temporary) / "source"
            path.with_suffix(".xml").write_text(xml(clean))
            model = mujoco.MjModel.from_xml_string(xml(clean))
            data = mujoco.MjData(model)
            save(model, data, path.with_suffix(".npz"), cfg)
            layers = report.get(
                "layers",
                max(1, max((a.get("layer", 0) + 1 for a in report["attempts"]), default=1)),
            )
            obj = cls(path, layers, mode or report["mode"], cfg)
        # Restore the original physical model and complete integration state.
        obj.m, obj.d, obj.tree = original, state, original_tree
        obj.source = str(source)
        obj.slots = report.get("pallet_targets", obj.slots)
        obj.attempts = report["attempts"]
        obj.max_effort_ratio = report.get("max_effort_ratio", 0.0)
        obj.max_speed_ratio = report.get("max_speed_ratio", 0.0)
        obj.removed = {
            original.body(a["selected"]).id for a in obj.attempts if a.get("physically_deposited")
        }
        obj.prefill_count = report.get("initially_on_pallet", 0)
        obj.removed.update(
            b for b in obj.bodies if original.body(b).name.startswith("ingot_prefill_")
        )
        obj.placed = {original.body(a["selected"]).id for a in obj.attempts if a.get("success")}
        obj.slot_cursor = max(
            [a.get("pallet_slot", -1) + 1 for a in obj.attempts if a.get("physically_deposited")]
            + [obj.prefill_count]
        )
        obj.q_command = state.qpos[obj.k.qa].copy()
        tb = original.body("tool").id
        obj.command_pos = state.xpos[tb].copy()
        obj.command_R = state.xmat[tb].reshape(3, 3).copy()
        obj.sim_anchor = float(state.time)
        obj.next_sample = obj.next_monitor = obj.next_sync = obj.next_frame = float(state.time)
        if held_recovery:
            obj.selected = original.body(report["attempts"][-1]["selected"]).id
            obj.vacuum = Vacuum(original, cfg, report["attempts"][-1]["selected"])
            obj.vacuum.pressures[:] = report.get("vacuum_pressures_pa", 0.0)
            obj.jaw_command = cfg["jaws"]["closed_travel"]
            obj.phase = report["phase"]
        return obj

    def flange_target(self, pos, R):
        return pos + R @ np.array([0, 0, self.cfg["robot"]["tool_mount_distance"]]), R @ np.diag(
            [1, -1, -1]
        )

    def _configure_planner_contacts(self):
        for g in range(self.k.m.ngeom):
            b = self.k.m.geom_bodyid[g]
            if self.k.m.body(b).name.startswith("ingot_") and self.k.m.geom_contype[g]:
                carried = self.planner_load is not None and b == self.selected
                self.k.m.geom_contype[g] = 1 if carried else 2
                self.k.m.geom_conaffinity[g] = 7 if carried else 1

    def _bad_planning_contact(self, contact):
        model = self.k.m
        names = [model.geom(g).name for g in [contact.geom1, contact.geom2]]
        if any(n.startswith("tool_case") for n in names) and any(
            n.startswith("kuka_") for n in names
        ):
            return contact.dist < self.settings.get(
                "tool_robot_planning_clearance", -self.settings["collision_penetration_tolerance"]
            )
        if contact.dist >= -self.settings["collision_penetration_tolerance"]:
            return False
        model = self.k.m
        geoms = [contact.geom1, contact.geom2]
        if any(
            model.geom(g).name.startswith("kuka_")
            or model.geom(g).name.startswith("tool_case")
            or model.geom(g).name == "vmg_adapter"
            for g in geoms
        ):
            return True
        bodies = model.geom_bodyid[geoms]
        if self.cfg["jaws"].get("fixed", False) and any(
            model.geom(g).name.startswith("jaw_pad_") for g in geoms
        ):
            if any(
                b != self.selected and (b == 0 or model.body(b).name.startswith("ingot_"))
                for b in bodies
            ):
                return True
        return (
            self.planner_load is not None
            and self.selected in bodies
            and any(
                b != self.selected and (b == 0 or model.body(b).name.startswith("ingot_"))
                for b in bodies
            )
        )

    def _cartesian_segment(self, pos, R, duration, start_pos=None, start_R=None, start_q=None):
        from scipy.spatial.transform import Slerp
        from .experiment import smoothstep

        p0 = self.command_pos if start_pos is None else start_pos
        R0 = self.command_R if start_R is None else start_R
        q = self.q_command.copy() if start_q is None else start_q.copy()
        self._configure_planner_contacts()
        self.k.d.qpos[:] = self.d.qpos
        interpolator = Slerp([0, 1], Rotation.from_matrix([R0, R]))
        times = np.linspace(
            0, duration, max(3, int(duration / self.settings["planner_sample_period"]) + 1)
        )
        qs = []
        for t in times:
            u = smoothstep(t / duration)
            p = p0 + (pos - p0) * u
            rr = interpolator(float(u)).as_matrix()
            fp, fr = self.flange_target(p, rr)
            q = self.k.ik(fp, fr, start=q, attempts=1)
            if q is None:
                raise MotionFailure(f"IK rejects path to {np.round(pos, 3).tolist()}")
            if self.planner_load is not None:
                rel_p, rel_R = self.planner_load
                joint = self.m.body_jntadr[self.selected]
                adr = self.m.jnt_qposadr[joint]
                self.k.d.qpos[adr : adr + 3] = p + rr @ rel_p
                quat = Rotation.from_matrix(rr @ rel_R).as_quat()
                self.k.d.qpos[adr + 3 : adr + 7] = [quat[3], *quat[:3]]
            contacts = self.k.collisions(q)
            bad = [
                (self.k.m.geom(cc.geom1).name, self.k.m.geom(cc.geom2).name, float(cc.dist))
                for cc in self.k.d.contact
                if self._bad_planning_contact(cc)
            ]
            if bad:
                raise MotionFailure(f"Collision rejects planned path: {bad[0]}")
            resolution = self.settings.get("planner_joint_edge_resolution")
            if qs and resolution:
                # Check the joint interpolation actually executed between IK
                # samples; collision-free endpoint poses alone are insufficient.
                for probe in np.linspace(
                    qs[-1], q, max(2, int(np.max(np.abs(q - qs[-1])) / resolution) + 2)
                )[1:-1]:
                    self.k.collisions(probe)
                    for cc in self.k.d.contact:
                        names = [self.k.m.geom(g).name for g in [cc.geom1, cc.geom2]]
                        if (
                            any(n.startswith("tool_case") for n in names)
                            and any(n.startswith("kuka_") for n in names)
                            and self._bad_planning_contact(cc)
                        ):
                            raise MotionFailure("Tool/robot clearance violated between IK samples")
                self.k.collisions(q)
            qs.append(q.copy())
        return times, np.asarray(qs)

    def _joint_fallback(self, pos, R, duration, q0):
        from .mock_planning import joint_route, timed_route

        planner = self.k
        self._configure_planner_contacts()
        planner.d.qpos[:] = self.d.qpos

        def collision(q):
            planner.d.qpos[planner.qa] = q
            mujoco.mj_forward(planner.m, planner.d)
            if self.planner_load is not None:
                body = planner.m.body("tool").id
                tool_R = planner.d.xmat[body].reshape(3, 3)
                rel_p, rel_R = self.planner_load
                adr = planner.m.jnt_qposadr[planner.m.body_jntadr[self.selected]]
                planner.d.qpos[adr : adr + 3] = planner.d.xpos[body] + tool_R @ rel_p
                quat = Rotation.from_matrix(tool_R @ rel_R).as_quat()
                planner.d.qpos[adr + 3 : adr + 7] = [quat[3], *quat[:3]]
                mujoco.mj_forward(planner.m, planner.d)
            return any(self._bad_planning_contact(cc) for cc in planner.d.contact)

        if collision(q0):
            raise MotionFailure("Planning start is in collision")
        fp, fr = self.flange_target(pos, R)
        rng = np.random.default_rng(self.cfg["seed"])
        for seed in range(8):
            goal = planner.ik(fp, fr, start=q0 if seed == 0 else None, seed=seed, attempts=1)
            if goal is None:
                continue
            for i in range(6):
                equivalent = [
                    goal[i] + n * 2 * np.pi
                    for n in [-1, 0, 1]
                    if planner.limits[i, 0] <= goal[i] + n * 2 * np.pi <= planner.limits[i, 1]
                ]
                goal[i] = min(equivalent, key=lambda value: abs(value - q0[i]))
            if collision(goal):
                continue
            route = joint_route(
                q0,
                goal,
                planner.limits,
                collision,
                rng,
                edge_resolution=self.settings.get("planner_joint_edge_resolution", 0.04),
            )
            if route is not None:
                return timed_route(
                    route,
                    self.speed_limits,
                    duration,
                    self.settings.get("planner_speed_fraction", 0.4),
                )
        raise MotionFailure("Mock joint-space planner found no collision-free route")

    def plan_motion(self, pos, R, duration, start_pos=None, start_R=None, start_q=None):
        p0 = self.command_pos if start_pos is None else start_pos
        R0 = self.command_R if start_R is None else start_R
        q0 = self.q_command if start_q is None else start_q
        try:
            return self._cartesian_segment(pos, R, duration, p0, R0, q0)
        except MotionFailure as original:
            if min(p0[2], pos[2]) < self.params["safe_height"] - 0.01:
                raise
            # A deterministic Cartesian roadmap mock: move through the central
            # clearance pose and rotate there instead of folding the wrist in transit.
            home = np.array(self.settings["home_tool_position"], dtype=float)
            home[2] = max(p0[2], pos[2], home[2])
            pp, rr, qq = np.array(p0), np.array(R0), q0.copy()
            times, qs, elapsed = [], [], 0.0
            for destination, rotation in [(home, R0), (home, R), (pos, R)]:
                try:
                    tt, path = self._cartesian_segment(
                        destination, rotation, max(2.0, duration / 2), pp, rr, qq
                    )
                except MotionFailure:
                    try:
                        return self._joint_fallback(pos, R, duration, q0)
                    except MotionFailure:
                        raise original
                times.extend((tt[1:] + elapsed).tolist())
                qs.extend(path[1:])
                elapsed += tt[-1]
                pp, rr, qq = destination, rotation, path[-1]
            return np.r_[0.0, times], np.vstack([q0, qs])

    def _drive_pose(self, pos, R):
        t = self.d.time - self.plan_start
        i = min(len(self.plan_times) - 2, max(0, np.searchsorted(self.plan_times, t) - 1))
        f = np.clip((t - self.plan_times[i]) / (self.plan_times[i + 1] - self.plan_times[i]), 0, 1)
        self.q_command = self.plan_q[i] * (1 - f) + self.plan_q[i + 1] * f
        self.d.ctrl[self.robot_actuators] = (
            self.q_command + self.d.qfrc_bias[self.k.va] / self.cfg["robot"]["position_kp"]
        )

    def _jaw_setpoint(self, jaws):
        target = self.cfg["jaws"]["closed_travel"] if jaws and not self.deploy_only else 0.0
        delta = self.cfg["jaws"]["max_speed"] * self.m.opt.timestep
        self.jaw_command += np.clip(target - self.jaw_command, -delta, delta)
        return self.jaw_command

    def _deploy_setpoint(self, jaws, side):
        travel = self.cfg["jaws"].get("command_deploy_travel", self.cfg["jaws"]["deploy_travel"])
        if jaws:
            return travel
        if self.cfg["jaws"].get("retract_interlock", False):
            adr = self.m.joint("jaw_" + side).qposadr[0]
            if self.d.qpos[adr] > 0.001:
                return travel
        return 0.0

    def jaw_clearance(self):
        """Check deployment and closing swept volumes against non-target objects."""
        planner = self.k
        planner.d.qpos[:] = self.d.qpos
        deploy = (
            []
            if self.cfg["jaws"].get("fixed", False)
            else [int(planner.m.joint("deploy_" + side).qposadr[0]) for side in ["left", "right"]]
        )
        jaw = [int(planner.m.joint("jaw_" + side).qposadr[0]) for side in ["left", "right"]]
        travel = self.cfg["jaws"].get("command_deploy_travel", self.cfg["jaws"]["deploy_travel"])
        closed = self.cfg["jaws"]["closed_travel"]
        for dep, squeeze in [(v, 0.0) for v in np.linspace(0, travel, 16)] + [
            (travel, v) for v in np.linspace(0, closed, 16)
        ]:
            planner.d.qpos[deploy] = dep
            planner.d.qpos[jaw] = squeeze
            mujoco.mj_forward(planner.m, planner.d)
            for contact in planner.d.contact:
                if contact.dist >= -self.settings["collision_penetration_tolerance"]:
                    continue
                for pad, other in [(contact.geom1, contact.geom2), (contact.geom2, contact.geom1)]:
                    if not planner.m.geom(pad).name.startswith("jaw_pad_"):
                        continue
                    body = planner.m.geom_bodyid[other]
                    if body != self.selected and (
                        body == 0 or planner.m.body(body).name.startswith("ingot_")
                    ):
                        return False
        return True

    def deploy_open(self, pos, R, vacuum):
        clear = self.jaw_clearance()
        self.attempts[-1]["jaw_clearance_passed"] = clear
        if not clear:
            return False
        self.deploy_only = True
        self.move(
            "OPEN_JAWS" if self.cfg["jaws"].get("fixed", False) else "DEPLOY_OPEN_JAWS",
            pos,
            R,
            self.settings["jaw_deploy_time"],
            vacuum,
            True,
        )
        self.deploy_only = False
        return True

    def _key(self, key):
        if key == 32:
            self.paused = not self.paused
        elif key in (43, 61, 334, 45, 333):
            # ASCII +/= and GLFW keypad +/-; the viewer reports physical keys.
            faster = key in (43, 61, 334)
            self.speed = float(np.clip(self.speed * (2 if faster else 0.5), 0.125, 8))
            print(f"Simulation speed requested: x{self.speed:g}", flush=True)
        self.wall_anchor = time.monotonic()
        self.sim_anchor = self.d.time

    def _after_step(self):
        if self.d.time >= self.next_monitor:
            self.max_speed_ratio = max(
                self.max_speed_ratio,
                float(np.max(np.abs(self.d.qvel[self.k.va]) / self.speed_limits)),
            )
            self.max_effort_ratio = max(
                self.max_effort_ratio,
                float(
                    np.max(np.abs(self.d.actuator_force[self.robot_actuators]) / self.effort_limits)
                ),
            )
            if self.attempts:
                current = self.attempts[-1]
                current["max_speed_ratio"] = max(
                    current.get("max_speed_ratio", 0.0),
                    float(np.max(np.abs(self.d.qvel[self.k.va]) / self.speed_limits)),
                )
                current["max_effort_ratio"] = max(
                    current.get("max_effort_ratio", 0.0),
                    float(
                        np.max(
                            np.abs(self.d.actuator_force[self.robot_actuators]) / self.effort_limits
                        )
                    ),
                )
            speed_ratio = float(np.max(np.abs(self.d.qvel[self.k.va]) / self.speed_limits))
            if speed_ratio > 1.001:
                raise MotionFailure(
                    f"Robot velocity limit exceeded in {self.phase}: {speed_ratio:.3f} of nominal"
                )
            for contact in self.d.contact:
                names = (self.m.geom(contact.geom1).name, self.m.geom(contact.geom2).name)
                bodies = self.m.geom_bodyid[[contact.geom1, contact.geom2]]
                if (
                    self.cfg["jaws"].get("fixed", False)
                    and contact.dist < -self.settings["collision_penetration_tolerance"]
                    and any(n.startswith("jaw_pad_") for n in names)
                ):
                    if any(
                        b != self.selected and (b == 0 or self.m.body(b).name.startswith("ingot_"))
                        for b in bodies
                    ):
                        self.robot_collisions.add(names)
                        raise MotionFailure(
                            f"Finger collision with neighbor/support in {self.phase}: {names}"
                        )
                if contact.dist < -self.settings["collision_penetration_tolerance"] and any(
                    n.startswith("kuka_") or n.startswith("tool_case") or n == "vmg_adapter"
                    for n in names
                ):
                    self.robot_collisions.add(names)
                    raise MotionFailure(f"Robot collision in {self.phase}: {names}")
            if any(w.number for w in self.d.warning) or not np.isfinite(self.d.qpos).all():
                raise MotionFailure("Physics engine warning or non-finite state")
            if getattr(self, "grasp_watch", False) and self.selected is not None and self.attempts:
                jaw_hold = self.grasp_watch_jaws and self.jaw_contacts()
                vac_hold = (
                    self.grasp_watch_vacuum
                    and self.vacuum is not None
                    and np.any(self.vacuum.quality > 0.5)
                    and self.vacuum.mean_pressure > self.params["minimum_pressure_pa"]
                )
                if jaw_hold or vac_hold:
                    self.grasp_missing_since = None
                else:
                    if self.grasp_missing_since is None:
                        self.grasp_missing_since = self.d.time
                    tool = self.m.body("tool").id
                    expected = (
                        self.d.xpos[tool]
                        + self.d.xmat[tool].reshape(3, 3) @ self.grasp_watch_relative
                    )
                    if (
                        self.d.time - self.grasp_missing_since >= 0.03
                        and np.linalg.norm(self.d.xpos[self.selected] - expected) > 0.008
                    ):
                        current = self.attempts[-1]
                        current.setdefault("grasp_loss_phase", self.phase)
                        current.setdefault("grasp_loss_time_s", float(self.d.time))
                        current["unexpected_grasp_loss"] = True
            self.next_monitor = self.d.time + self.settings["monitor_period"]
        if self.viewer and self.d.time >= self.next_sync:
            while self.paused and self.viewer.is_running():
                self.viewer.sync()
                time.sleep(0.03)
            if not self.viewer.is_running():
                raise ViewerClosed()
            label = f"{self.phase} | {self.m.body(self.selected).name if self.selected else '—'} | {len(self.removed)}/{len(self.bodies)} | x{self.speed:g}"
            pressure = (
                ", ".join(f"{v / 1000:.1f}" for v in self.vacuum.pressures)
                if self.vacuum
                else "off"
            )
            seal = ", ".join(f"{v:.2f}" for v in self.vacuum.quality) if self.vacuum else "—"
            jaw = (
                ", ".join(
                    f"{self.d.qpos[self.m.joint('jaw_' + side).qposadr[0]] * 1000:.1f}"
                    for side in ["left", "right"]
                )
                if self.cfg["jaws"].get("enabled", True)
                else "absent"
            )
            load_info = ""
            if self.selected and self.cfg.get("ingots"):
                item = ingot_config(self.cfg, self.m.body(self.selected).name)
                load_info = (
                    f" | {item['mass']:.2f} kg | "
                    + " x ".join(f"{item[k] * 1000:.0f}" for k in ("length", "width", "thickness"))
                    + " mm"
                )
            self.viewer.set_texts(
                [
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                        "Pressure, kPa\nSeal quality\nJaw travel, mm",
                        f"{pressure}\n{seal}\n{jaw}",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "RUSAL / PHYSICAL KUKA / MOCK CONTROL",
                        label,
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                        "Space: pause   +/-: speed   Esc: close",
                        f"t = {self.d.time:.1f} s" + load_info,
                    ),
                ]
            )
            self.viewer.sync()
            deadline = self.wall_anchor + (self.d.time - self.sim_anchor) / self.speed
            time.sleep(max(0, min(0.04, deadline - time.monotonic())))
            self.next_sync = self.d.time + max(1.0, self.speed) / 60

    def run(self, max_attempts=30, out="results/logs/sim", gui=True, speed=1.0, video=None):
        self.speed = speed
        self.checkpoint_path = out
        self.run_in_progress = True
        if gui:
            from mujoco import viewer as mjviewer

            self.viewer = mjviewer.launch_passive(self.m, self.d, key_callback=self._key)
            self.viewer.cam.lookat[:] = [0.1, -0.1, 0.4]
            self.viewer.cam.distance = 4.8
            self.viewer.cam.azimuth = 125
            self.viewer.cam.elevation = -35
        if video:
            import imageio.v2 as imageio

            Path(video).parent.mkdir(parents=True, exist_ok=True)
            self.renderer = mujoco.Renderer(self.m, 720, 960)
            self.writer = imageio.get_writer(str(video), fps=30)
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.camera)
            self.camera.lookat[:] = [0.1, -0.1, 0.35]
            self.camera.distance = 4.8
            self.camera.azimuth = 125
            self.camera.elevation = -40
        self.wall_start = self.wall_anchor = time.monotonic()
        self.sim_anchor = float(self.d.time)
        try:
            self.cycle(max_attempts)
        except KeyboardInterrupt:
            self.abort_reason = "Interrupted by user"
        except ViewerClosed:
            self.abort_reason = "Viewer closed by user"
        except MotionFailure as exc:
            self.abort_reason = str(exc)
            print(self.abort_reason, flush=True)
        finally:
            self.run_in_progress = False
            if self.writer:
                self.writer.close()
            if self.renderer:
                self.renderer.close()
            self.writer = self.renderer = None
            if self.viewer:
                self.viewer.close()
                time.sleep(0.3)  # Let the native GUI thread release its OpenGL resources.
        if self.abort_reason and self.attempts and "physically_deposited" not in self.attempts[-1]:
            self.attempts[-1]["failure"] = self.abort_reason
        return self.write_result(out)

    def candidates(self):
        # Both broad faces are candidates; the physical seal decides feasibility.
        result = []
        m, d = self.m, self.d
        radius = self.cfg["vacuum"]["diameter"] / 2
        footprint = [
            [x + dx, dy, 0]
            for x in self.cfg["vacuum"]["cup_x"]
            for dx, dy in [(0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius)]
        ]
        if "cell_centers" in self.cfg["vacuum"]:
            sx, sy = np.asarray(self.cfg["vacuum"]["rectangle"]) / 2
            footprint = [
                [x + dx, y + dy, 0]
                for x, y in self.cfg["vacuum"]["cell_centers"]
                for dx, dy in [(0, 0), (sx, sy), (sx, -sy), (-sx, sy), (-sx, -sy)]
            ]
        footprint = self.cfg.get("gripper", {}).get("pick_footprint", footprint)
        gid = np.zeros(1, dtype=np.int32)
        # Ground-truth visibility checks clutter, not the robot's own casing.
        # Use the private planning copy; physical contacts/groups stay untouched.
        ray_model, ray_data = self.k.m, self.k.d
        ray_data.qpos[:] = d.qpos
        ray_model.geom_group[:] = [
            0 if ray_model.body(ray_model.geom_bodyid[g]).name.startswith("ingot_") else 1
            for g in range(ray_model.ngeom)
        ]
        mujoco.mj_forward(ray_model, ray_data)
        ray_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        for b in self.bodies:
            if b in self.removed:
                continue
            h = ingot_config(self.cfg, m.body(b).name)["thickness"]
            body_R = d.xmat[b].reshape(3, 3)
            smooth = body_R[2, 2] < 0
            head_R = body_R @ (np.diag([1, -1, -1]) if smooth else np.eye(3))
            n = head_R[:, 2]
            if n[2] < self.params["minimum_face_cosine"]:
                continue
            for dx, dy in self.settings.get("grasp_offsets", [[0, 0]]):
                face = d.xpos[b] + n * h / 2 + head_R @ np.array([dx, dy, 0])
                clear = True
                for off in footprint:
                    distance = mujoco.mj_ray(
                        ray_model,
                        ray_data,
                        face + head_R @ off + n * self.settings["ray_clearance"],
                        -n,
                        ray_groups,
                        1,
                        -1,
                        gid,
                    )
                    if distance < 0 or m.geom_bodyid[gid[0]] != b:
                        clear = False
                        break
                if clear:
                    result.append((float(face[2]), b, face, head_R, smooth))
        # Preserve central-first offset order within the same body.
        return sorted(result, key=lambda item: (item[4], float(d.xpos[item[1], 2])), reverse=True)

    def grasp_key(self, candidate):
        _, body, face, rotation, _ = candidate
        offset = rotation.T @ (face - self.d.xpos[body])
        return (body, round(float(offset[0]), 4), round(float(offset[1]), 4))

    def move(self, phase, pos, R, duration, vacuum=False, jaws=False):
        if phase != self.phase:
            print(f"{self.d.time:7.2f}s {phase}", flush=True)
        self.plan_times, self.plan_q = self.plan_motion(np.asarray(pos), R, duration)
        if "planner_speed_fraction" in self.settings:
            # IK can change joint configuration sharply while Cartesian motion
            # stays short. Time the actual joint path, retaining force limits.
            minimum = np.max(
                np.abs(np.diff(self.plan_q, axis=0))
                / (self.settings["planner_speed_fraction"] * self.speed_limits),
                axis=1,
            )
            self.plan_times = np.r_[0.0, np.cumsum(np.maximum(np.diff(self.plan_times), minimum))]
        self.plan_start = self.d.time
        self.phase = phase
        self.grasp_watch = phase in ("RAISE", "TRANSPORT", "PLACE", "VERIFY_JAWS_VACUUM_OFF") and (
            vacuum or jaws
        )
        self.grasp_watch_vacuum = vacuum
        self.grasp_watch_jaws = jaws
        self.grasp_missing_since = None
        if self.grasp_watch and self.selected is not None:
            tool = self.m.body("tool").id
            self.grasp_watch_relative = self.d.xmat[tool].reshape(3, 3).T @ (
                self.d.xpos[self.selected] - self.d.xpos[tool]
            )
        self.advance(np.asarray(pos), R, float(self.plan_times[-1]), vacuum, jaws)

    def supported(self, body):
        # Sleeping contact islands are omitted from d.contact. Evaluate an awake
        # copy to inspect support without waking/moving the actual simulation.
        audit = mujoco.MjData(self.m)
        audit.qpos[:] = self.d.qpos
        audit.qvel[:] = self.d.qvel
        mujoco.mj_forward(self.m, audit)
        pallet = self.m.geom("pallet").id
        return any(
            body in self.m.geom_bodyid[[cc.geom1, cc.geom2]]
            and (
                pallet in [cc.geom1, cc.geom2]
                or any(i in self.removed for i in self.m.geom_bodyid[[cc.geom1, cc.geom2]])
            )
            for cc in audit.contact
        )

    def jaw_contacts(self):
        touching = set()
        for contact in self.d.contact:
            a, b = contact.geom1, contact.geom2
            if self.m.geom_bodyid[a] == self.selected:
                touching.add(self.m.geom(b).name)
            if self.m.geom_bodyid[b] == self.selected:
                touching.add(self.m.geom(a).name)
        return all("jaw_pad_" + side in touching for side in ["left", "right"])

    def body_bounds(self, body):
        item = ingot_config(self.cfg, self.m.body(body).name)
        vertices = np.concatenate(
            [
                part.mesh.vertices
                for part in components(item, item.get("geometry_seed", self.cfg["seed"]))
            ]
        )
        points = vertices @ self.d.xmat[body].reshape(3, 3).T + self.d.xpos[body]
        return points.min(axis=0), points.max(axis=0)

    def packed_rectangles(self):
        layers = {
            self.m.body(a["selected"]).id: a["layer"]
            for a in self.attempts
            if a.get("physically_deposited")
        }
        occupied = []
        for body in self.removed:
            lo, hi = self.body_bounds(body)
            occupied.append(
                dict(
                    body=body,
                    layer=layers.get(body, 0),
                    x=float(lo[0]),
                    y=float(lo[1]),
                    w=float(hi[0] - lo[0]),
                    h=float(hi[1] - lo[1]),
                )
            )
        return occupied

    def prepare_slot(self, item):
        if self.cfg["pallet"].get("layout") not in ("brick", "cross"):
            return
        target = self.slots[self.slot_cursor]
        x, y, _ = target["position"]
        # Project the actual dimensional envelope through the target yaw.
        extent = (
            np.abs(target_rotation(target))
            @ np.array([item["length"], item["width"], item["thickness"]])
            / 2
        )
        z = self.cfg["pallet"]["height"]
        supports = []
        for body in self.removed:
            lo, hi = self.body_bounds(body)
            if (
                hi[0] > x - extent[0]
                and lo[0] < x + extent[0]
                and hi[1] > y - extent[1]
                and lo[1] < y + extent[1]
            ):
                z = max(z, float(hi[2]))
                supports.append(self.m.body(body).name)
        if target["layer"] and not supports:
            raise MotionFailure("Pallet target lacks physical support")
        target["position"][2] = z + item["thickness"] / 2
        target["support_bodies"] = supports
        target["assigned_dimensions_m"] = [item[k] for k in ("length", "width", "thickness")]

    def release_clearance(self, place, rotation):
        """Keep opening fingers above adjacent loads; seat the free part by gravity."""
        clearance = self.settings["mechanical_release_clearance"]
        jaws = self.cfg["jaws"]
        half = np.abs(rotation) @ np.asarray(jaws["pad_half_sizes"])
        for side, sign in [("left", 1), ("right", -1)]:
            deploy = (
                float(self.d.qpos[self.m.joint("deploy_" + side).qposadr[0]])
                if not jaws.get("fixed", False)
                else 0.0
            )
            travel = float(self.d.qpos[self.m.joint("jaw_" + side).qposadr[0]])
            ends = [
                np.asarray(place)
                + rotation
                @ np.array(
                    [0, sign * (jaws["open_half_gap"] - q), jaws.get("mount_z", 0.035) - deploy]
                )
                for q in (0.0, travel)
            ]
            lo = np.min(ends, axis=0) - half
            hi = np.max(ends, axis=0) + half
            for body in self.removed:
                a, b = self.body_bounds(body)
                if hi[0] > a[0] and lo[0] < b[0] and hi[1] > a[1] and lo[1] < b[1]:
                    clearance = max(clearance, float(b[2] + 0.01 - lo[2]))
        return clearance

    def return_held_load(self, initial, rotation):
        """Lower a poorly seated but still clamped part, using its measured pose."""
        tool = self.m.body("tool").id
        tool_rotation = self.d.xmat[tool].reshape(3, 3)
        relative = tool_rotation.T @ (self.d.xpos[self.selected] - self.d.xpos[tool])
        relative_rotation = tool_rotation.T @ self.d.xmat[self.selected].reshape(3, 3)
        self.planner_load = (relative, relative_rotation)
        goal = (
            np.asarray(initial)
            + [0, 0, self.settings["mechanical_release_clearance"]]
            - rotation @ relative
        )
        self.move(
            "RETURN_FOR_REGRASP", goal, rotation, self.params["timing"]["approach"], False, True
        )
        self.planner_load = None
        self.move("ABORT_OPEN", goal, rotation, self.params["timing"]["release"])
        high = goal.copy()
        high[2] = max(high[2], self.params["safe_height"])
        self.move("ABORT_RETRACT", high, rotation, self.params["timing"]["retract"])
        self.attempts[-1]["returned_for_regrasp"] = True

    def inspect_drop_placement(self, body, target, stable):
        cfg = self.cfg
        item = ingot_config(cfg, self.m.body(body).name)
        points = (
            np.concatenate(
                [
                    part.mesh.vertices
                    for part in components(item, item.get("geometry_seed", cfg["seed"]))
                ]
            )
            @ self.d.xmat[body].reshape(3, 3).T
            + self.d.xpos[body]
        )
        rotation = self.d.xmat[body].reshape(3, 3)
        desired = target_rotation(target)
        axis = rotation[:2, 0]
        goal = desired[:2, 0]
        yaw = float(
            np.rad2deg(
                np.arccos(
                    np.clip(
                        np.dot(axis, goal)
                        / max(1e-12, np.linalg.norm(axis) * np.linalg.norm(goal)),
                        -1,
                        1,
                    )
                )
            )
        )
        tilt = float(np.rad2deg(np.arccos(np.clip(abs(rotation[2, 2]), 0, 1))))
        delta = self.d.xpos[body] - target["position"]
        within = bool(
            np.all(
                np.abs(points[:, :2] - cfg["pallet"]["center"])
                <= np.array([cfg["pallet"]["length"], cfg["pallet"]["width"]]) / 2 + 0.001
            )
        )
        awake = mujoco.MjData(self.m)
        awake.qpos[:] = self.d.qpos
        mujoco.mj_forward(self.m, awake)
        penetration = max(
            (
                max(0.0, -cc.dist)
                for cc in awake.contact
                if body in self.m.geom_bodyid[[cc.geom1, cc.geom2]]
            ),
            default=0.0,
        )
        checks = dict(
            stable=bool(stable),
            supported=self.supported(body),
            within_pallet=within,
            xy_error_m=float(np.linalg.norm(delta[:2])),
            z_error_m=float(abs(delta[2])),
            yaw_error_deg=yaw,
            tilt_deg=tilt,
            penetration_m=float(penetration),
        )
        accepted = bool(
            stable
            and checks["supported"]
            and within
            and checks["xy_error_m"] <= self.settings.get("drop_max_xy_error_m", 0.025)
            and checks["z_error_m"] <= self.settings.get("drop_max_z_error_m", 0.02)
            and yaw <= self.settings.get("drop_max_yaw_error_deg", 5.0)
            and tilt <= self.settings.get("drop_max_tilt_deg", 8.0)
            and penetration <= 0.001
        )
        return accepted, checks

    def recover_dropped_load(self, body, result, rejected):
        home = np.asarray(self.settings["home_tool_position"]).copy()
        home[2] = max(home[2], self.command_pos[2], self.params["safe_height"])
        self.move("RECOVERY_CLEARANCE", home, self.command_R, self.params["timing"]["clearance"])
        stable = False
        for _ in range(10):
            before = self.d.xpos[body].copy()
            self.move("WAIT_DROPPED_LOAD", home, self.command_R, 0.5)
            if (
                np.linalg.norm(self.d.cvel[body]) < 0.02
                and np.linalg.norm(self.d.xpos[body] - before) < 0.002
            ):
                stable = True
                break
        name = self.m.body(body).name
        losses = sum(a.get("dropped", False) and a["selected"] == name for a in self.attempts)
        result.update(
            drop_settled=stable,
            on_pallet_after_drop=bool(
                self.supported(body) and self.d.xpos[body, 2] > self.cfg["pallet"]["height"]
            ),
            drop_position_m=self.d.xpos[body].tolist(),
            regrasp_number=losses,
            regrasp_limit=self.settings.get("max_regrasp_attempts", 3),
        )
        accepted, checks = self.inspect_drop_placement(body, self.slots[self.slot_cursor], stable)
        result["drop_placement_checks"] = checks
        if self.settings.get("accept_well_placed_drops", True) and accepted:
            result.update(
                accepted_after_drop=True,
                physically_deposited=True,
                retry_required=False,
                supported=True,
                position_error_m=float(
                    np.linalg.norm(self.d.xpos[body] - self.slots[self.slot_cursor]["position"])
                ),
                layer=self.slots[self.slot_cursor]["layer"],
            )
            self.removed.add(body)
            self.slot_cursor += 1
            rejected.clear()
            print(
                f"Dropped {name}: stable correct placement accepted; next slot {self.slot_cursor}",
                flush=True,
            )
            self.checkpoint()
            return True
        result["accepted_after_drop"] = False
        if not stable or losses >= result["regrasp_limit"]:
            rejected.add(body)
            result["retry_blocked_reason"] = (
                "drop did not settle" if not stable else "regrasp limit reached"
            )
        else:
            rejected.discard(body)
            rejected.difference_update(
                [key for key in rejected if isinstance(key, tuple) and key[0] == body]
            )
        print(
            f"Dropped {name}: settled={stable}, regrasp {losses}/{result['regrasp_limit']}; slot remains {self.slot_cursor}",
            flush=True,
        )

    def cycle(self, max_attempts):
        c, p, m, d = self.cfg, self.params, self.m, self.d
        t = p["timing"]
        rejected = set()
        mechanical_retry = {
            m.body(a["selected"]).id for a in self.attempts if a.get("returned_for_regrasp")
        }
        for _ in range(max_attempts):
            if self.slot_cursor >= len(self.slots):
                self.abort_reason = "pallet_full"
                break
            loss_counts = {}
            for attempt in self.attempts:
                if attempt.get("dropped"):
                    name = attempt["selected"]
                    loss_counts[name] = loss_counts.get(name, 0) + 1
            candidates = [
                v
                for v in self.candidates()
                if v[1] not in rejected
                and self.grasp_key(v) not in rejected
                and loss_counts.get(m.body(v[1]).name, 0)
                < self.settings.get("max_regrasp_attempts", 3)
            ]
            pending = {
                a["selected"]: a
                for a in self.attempts
                if (a.get("dropped") or a.get("placement_retry_required"))
                and m.body(a["selected"]).id not in self.removed
            }
            recovery_candidates = [v for v in candidates if m.body(v[1]).name in pending]
            if recovery_candidates:
                candidates = recovery_candidates
            elif any(a.get("on_pallet_after_drop") for a in pending.values()):
                self.abort_reason = "fallen pallet load blocks placement; no accessible regrasp or retry limit reached"
                break
            if not candidates:
                break
            if c["pallet"].get("layout") == "cross":
                occupied = self.packed_rectangles()
                remaining = {
                    m.body(b).name: ingot_config(c, m.body(b).name)
                    for b in self.bodies
                    if b not in self.removed
                }
                feasible = {
                    name: find_target(c, remaining[name], occupied, self.layers)
                    for name in dict.fromkeys(m.body(v[1]).name for v in candidates)
                }
                feasible = {name: plan for name, plan in feasible.items() if plan is not None}
                if not feasible:
                    self.abort_reason = "no supported packing space within the configured layers"
                    break
                lowest = min(plan["layer"] for plan in feasible.values())
                candidates = [
                    v
                    for v in candidates
                    if m.body(v[1]).name in feasible
                    and feasible[m.body(v[1]).name]["layer"] == lowest
                ]
                # Prefer seal-friendly exposed faces within the lowest feasible
                # packing layer; retries of a fallen pallet load retain priority.
                smooth_candidates = [v for v in candidates if v[4]]
                if smooth_candidates and not recovery_candidates:
                    candidates = smooth_candidates
                names = list(dict.fromkeys(m.body(v[1]).name for v in candidates))
                cached = getattr(self, "cross_assignment", None)
                signature = (
                    self.slot_cursor,
                    tuple((o["body"], round(o["x"], 3), round(o["y"], 3)) for o in occupied),
                )
                if cached and cached[0] == signature and cached[1][0] in names:
                    assignment = copy.deepcopy(cached[1])
                else:
                    assignment = choose_target(
                        c, names, remaining, occupied, self.layers, c["seed"] + len(self.attempts)
                    )
                    self.cross_assignment = (signature, copy.deepcopy(assignment))
                if assignment is None:
                    self.abort_reason = "no supported packing space within the configured layers"
                    break
                chosen, target = assignment
                candidates = [v for v in candidates if m.body(v[1]).name == chosen]
                self.slots[self.slot_cursor] = target
            grasp_key = self.grasp_key(candidates[0])
            _, b, face, R, smooth = candidates[0]
            self.selected = b
            self.planner_load = None
            name = m.body(b).name
            self.vacuum = Vacuum(m, c, name)
            neighbors = {i: d.xpos[i].copy() for i in self.bodies if i != b}
            self.neighbor_reference = neighbors
            self.neighbor_max = 0.0
            initial = d.xpos[b].copy()
            result = dict(
                pickup_position_m=initial.tolist(),
                selected=name,
                surface="smooth" if smooth else "ribbed",
                success=False,
                grasp_offset_m=list(grasp_key[1:]),
            )
            if loss_counts.get(name, 0):
                result["regrasp_after_drop"] = loss_counts[name]
            self.attempts.append(result)
            print(f"Pick {name} ({result['surface']}), slot {self.slot_cursor}", flush=True)
            ic = ingot_config(c, name)
            self.prepare_slot(ic)
            if c["pallet"].get("layout") == "cross":
                print(
                    f"Packing layer {self.slots[self.slot_cursor]['layer'] + 1}, target {self.slots[self.slot_cursor]['position']}, lookahead {self.slots[self.slot_cursor]['lookahead_fitted_remaining']}/{len(remaining)}",
                    flush=True,
                )
            result["dimensions_m"] = [ic[k] for k in ("length", "width", "thickness")]
            result["mass_kg"] = ic["mass"]
            offset = (
                ic["thickness"] / 2
                - c["vacuum"]["pad_center_z"]
                + c["vacuum"]["pad_half_thickness"]
                - c["fixture"]["preload"]
            )
            contact = face + R[:, 2] * (offset - ic["thickness"] / 2)
            clearance = np.array([contact[0], contact[1], p["safe_height"]])
            vac = self.mode != "mechanical"
            # Both broad faces may seal; decide from measured seal, not the side label.
            mechanical = self.mode == "mechanical" or b in mechanical_retry
            if b in mechanical_retry:
                vac = False
                result["mechanical_retry_after_failed_lift"] = True
            # Reject an infeasible complete nominal route before touching the load.
            target_preview = self.slots[self.slot_cursor]
            tx, ty, tz = target_preview["position"]
            viable = False
            R_candidates = [R, R @ Rotation.from_euler("z", np.pi).as_matrix()]
            orientations = [(target_rotation(target_preview), False)]
            if c["pallet"].get("layout") == "cross" and c["pallet"].get(
                "allow_marking_conflict", False
            ):
                orientations.append(
                    (
                        target_rotation(target_preview)
                        @ Rotation.from_euler("z", np.pi).as_matrix(),
                        True,
                    )
                )
            previews = [
                (rotation, conflict, grasp_R)
                for rotation, conflict in orientations
                for grasp_R in R_candidates
            ]
            for nominal_R, marking_conflict, grasp_R in previews:
                relative_preview = grasp_R.T @ d.xmat[b].reshape(3, 3)
                desired_body = nominal_R @ (np.diag([1, -1, -1]) if smooth else np.eye(3))
                end_R = desired_body @ relative_preview.T
                preview = [
                    (clearance, grasp_R, t["clearance"]),
                    (contact, grasp_R, t["approach"]),
                    (contact + [0, 0, p["lift_height"]], grasp_R, t["extract"]),
                    (clearance, grasp_R, t["raise"]),
                    (np.array([tx, ty, p["safe_height"]]), end_R, t["transport"]),
                    (
                        np.array(
                            [
                                tx,
                                ty,
                                tz
                                + offset
                                + (
                                    self.settings["mechanical_release_clearance"]
                                    if mechanical
                                    else 0
                                ),
                            ]
                        ),
                        end_R,
                        t["lower"],
                    ),
                ]
                pp, rr, qq = self.command_pos.copy(), self.command_R.copy(), self.q_command.copy()
                try:
                    for destination, rotation, duration in preview:
                        _, path_q = self.plan_motion(destination, rotation, duration, pp, rr, qq)
                        pp, rr, qq = destination, rotation, path_q[-1]
                except MotionFailure as exc:
                    result["failure"] = str(exc)
                    continue
                R = grasp_R
                if marking_conflict:
                    q = Rotation.from_matrix(nominal_R).as_quat()
                    target_preview["quaternion_wxyz"] = [float(q[3]), *map(float, q[:3])]
                    target_preview["marked_end_outward"] = False
                    target_preview["marking_conflict"] = (
                        "Nominal end direction unreachable; same layer axis retained with a 180 degree reversal"
                    )
                    result["marking_conflict"] = target_preview["marking_conflict"]
                viable = True
                result.pop("failure", None)
                break
            if not viable:
                rejected.add(grasp_key)
                print(result["failure"], flush=True)
                continue
            self.move("APPROACH_CLEARANCE", clearance, R, t["clearance"])
            self.move("APPROACH", contact, R, t["approach"], vac)
            if mechanical and not self.deploy_open(contact, R, vac):
                result["failure"] = "jaw deployment/closing obstructed by a neighbor or support"
                self.move("ABORT_OPEN", contact, R, t["release"])
                self.move("ABORT_RETRACT", clearance, R, t["retract"])
                rejected.add(grasp_key)
                continue
            self.move(
                "SEAL_OR_CLAMP",
                contact,
                R,
                t["deploy"] if mechanical else t["seal_wait"],
                vac,
                mechanical,
            )
            sealed = bool(
                np.count_nonzero(self.vacuum.quality > p["minimum_seal_quality"])
                >= p["minimum_live_cups"]
                and self.vacuum.mean_pressure > p["minimum_pressure_pa"]
                and sum(self.vacuum.capacity)
                >= c.get("gripper", {}).get("holding_safety_factor", 0) * m.body_mass[b] * 9.81
            )
            clamped = self.jaw_contacts() if mechanical else False
            result.update(sealed=sealed, clamped=clamped)
            print(
                f"Seal={sealed}, bilateral clamp={clamped}, pressure={self.vacuum.mean_pressure / 1000:.1f} kPa",
                flush=True,
            )
            if not sealed and not mechanical and self.mode == "hybrid":
                mechanical = self.deploy_open(contact, R, vac)
                if mechanical:
                    self.move("MECHANICAL_FALLBACK", contact, R, t["deploy"], vac, True)
                clamped = self.jaw_contacts() if mechanical else False
                result["clamped"] = clamped
            if not (sealed or clamped):
                result["failure"] = "no seal / no bilateral jaw contact"
                self.move("ABORT_OPEN", contact, R, t["release"])
                self.move("ABORT_RETRACT", clearance, R, t["retract"])
                rejected.add(grasp_key)
                continue
            lift = contact + [0, 0, p["lift_height"]]
            self.move("EXTRACT", lift, R, t["extract"], vac, mechanical)
            lifted = bool(d.xpos[b, 2] - initial[2] > p["minimum_lift"])
            result.update(
                lifted=lifted,
                neighbor_displacement_m=self.neighbor_max,
                extraction_passed=lifted and self.neighbor_max < p["maximum_neighbor_shift"],
            )
            if not lifted:
                result["failure"] = "load slipped / extraction obstructed"
                # Return the actual tool toward the pickup before opening.
                self.move("ABORT_LOWER", contact, R, t["approach"], vac, mechanical)
                self.move("ABORT_OPEN", contact, R, t["release"])
                self.move("ABORT_RETRACT", clearance, R, t["retract"])
                if self.mode == "hybrid" and not mechanical:
                    # A pressure/seal test alone does not prove a successful lift.
                    # Retry once with real bilateral jaw contact, after lowering.
                    mechanical_retry.add(b)
                else:
                    rejected.add(grasp_key)
                continue
            if self.mode == "hybrid" and not mechanical:
                mechanical = self.deploy_open(lift, R, vac)
                if mechanical:
                    self.move("DEPLOY_JAWS", lift, R, t["deploy"], vac, True)
                    result["clamped_after_lift"] = self.jaw_contacts()
                if c.get("gripper", {}).get("require_post_lift_clamp"):
                    if not result.get("clamped_after_lift", False):
                        result["failure"] = "post-lift mechanical clamp not confirmed"
                        self.move("ABORT_LOWER", contact, R, t["approach"], vac, False)
                        self.move("ABORT_OPEN", contact, R, t["release"])
                        self.move("ABORT_RETRACT", clearance, R, t["retract"])
                        rejected.add(grasp_key)
                        continue
                    before = d.xpos[b].copy()
                    self.move("VERIFY_JAWS_VACUUM_OFF", lift, R, 1.0, False, True)
                    held = bool(
                        self.jaw_contacts()
                        and np.linalg.norm(d.xpos[b] - before) < 0.01
                        and d.xpos[b, 2] - initial[2] > p["minimum_lift"]
                    )
                    result["vacuum_off_hold_passed"] = held
                    result["vacuum_off_hold_drift_m"] = float(np.linalg.norm(d.xpos[b] - before))
                    # Carry and place mechanically after a real vacuum-off hold.
                    vac = False
                    sealed = False
                    if not held:
                        result["failure"] = "mechanical hold failed with vacuum off"
                        if result.get("unexpected_grasp_loss") and not self.jaw_contacts():
                            self.move("ABORT_OPEN", self.command_pos, self.command_R, t["release"])
                            result.update(
                                physically_deposited=False,
                                dropped=True,
                                retry_required=True,
                                pallet_slot=self.slot_cursor,
                            )
                            self.recover_dropped_load(b, result, rejected)
                            continue
                        self.return_held_load(initial, R)
                        mechanical_retry.add(b)
                        continue
            # Measured relative grasp pose, not an object attachment.
            tool_b = m.body("tool").id
            tool_R = d.xmat[tool_b].reshape(3, 3)
            relative_pos = tool_R.T @ (d.xpos[b] - d.xpos[tool_b])
            relative_R = tool_R.T @ d.xmat[b].reshape(3, 3)
            self.planner_load = (relative_pos, relative_R)
            target = self.slots[self.slot_cursor]
            goal_body_R = target_rotation(target) @ (np.diag([1, -1, -1]) if smooth else np.eye(3))
            final_R = goal_body_R @ relative_R.T
            place = np.asarray(target["position"]) - final_R @ relative_pos
            high = place.copy()
            high[2] = max(p["safe_height"], place[2] + 0.25)
            self.move(
                "RAISE", np.array([lift[0], lift[1], high[2]]), R, t["raise"], vac, mechanical
            )
            self.move("TRANSPORT", high, final_R, t["transport"], vac, mechanical)
            result["transport_passed"] = bool(
                not result.get("unexpected_grasp_loss")
                and np.linalg.norm(d.xpos[b] - (high + final_R @ relative_pos))
                < p["maximum_transport_error"]
            )
            if not result["transport_passed"]:
                still_held = self.jaw_contacts() or (
                    np.count_nonzero(self.vacuum.quality > p["minimum_seal_quality"])
                    >= p["minimum_live_cups"]
                    and np.mean(self.vacuum.pressures) > p["minimum_pressure_pa"]
                )
                result["failure"] = (
                    "excessive grasp slip in transport" if still_held else "load lost in transport"
                )
                if still_held and not result.get("unexpected_grasp_loss"):
                    # Preserve physical support and place slowly using the newly
                    # measured grasp transform. The failed transport stays failed.
                    result["controlled_recovery"] = True
                else:
                    self.planner_load = None
                    self.move("ABORT_OPEN", self.command_pos, self.command_R, t["release"])
                    self.move("ABORT_RETRACT", self.command_pos, self.command_R, t["abort"])
                    error = float(np.linalg.norm(d.xpos[b] - target["position"]))
                    deposited = bool(
                        self.supported(b)
                        and error < 0.1
                        and d.xpos[b, 2] > c["pallet"]["height"]
                        and np.linalg.norm(d.cvel[b]) < 0.02
                    )
                    result.update(
                        physically_deposited=False,
                        on_pallet_after_drop=deposited,
                        dropped=True,
                        retry_required=True,
                        pallet_slot=self.slot_cursor,
                        layer=target["layer"],
                        position_error_m=error,
                        placement_passed=False,
                    )
                    # A settled, correctly seated fall may complete placement.
                    self.recover_dropped_load(b, result, rejected)
                    continue
            # Ground-truth mock perception corrects the accumulated physical grasp slip.
            tool_R = d.xmat[tool_b].reshape(3, 3)
            relative_pos = tool_R.T @ (d.xpos[b] - d.xpos[tool_b])
            relative_R = tool_R.T @ d.xmat[b].reshape(3, 3)
            self.planner_load = (relative_pos, relative_R)
            final_R = goal_body_R @ relative_R.T
            place = np.asarray(target["position"]) - final_R @ relative_pos
            # The clamp opens only after reaching the destination above the pallet.
            if mechanical and sealed:
                self.move("RETRACT_JAWS", high, final_R, t["deploy"], vac, False)
                mechanical = False
            if mechanical and not sealed:
                seating = self.release_clearance(place, final_R)
                place = place + np.array([0, 0, seating])
                result["gravity_seating_height_m"] = seating
            self.move("PLACE", place, final_R, t["lower"], vac, mechanical)
            supported = self.supported(b)
            self.planner_load = None
            self.move("RELEASE", place, final_R, t["release"])
            self.move("RETRACT", high, final_R, t["retract"])
            self.move("SETTLE", high, final_R, t["settle"])
            supported = self.supported(b)
            error = float(np.linalg.norm(d.xpos[b] - target["position"]))
            angle = float(
                np.rad2deg(
                    Rotation.from_matrix(goal_body_R.T @ d.xmat[b].reshape(3, 3)).magnitude()
                )
            )
            if result.get("unexpected_grasp_loss"):
                result.update(
                    physically_deposited=False,
                    on_pallet_after_drop=bool(supported),
                    dropped=True,
                    retry_required=True,
                    pallet_slot=self.slot_cursor,
                    layer=target["layer"],
                    position_error_m=error,
                    placement_passed=False,
                    failure="load lost before commanded release",
                )
                self.recover_dropped_load(b, result, rejected)
                continue
            deposited = bool(
                supported
                and abs(d.xpos[b, 0] - target["position"][0]) < 0.1
                and abs(d.xpos[b, 1] - target["position"][1]) < 0.1
                and d.xpos[b, 2] > c["pallet"]["height"]
                and np.linalg.norm(d.cvel[b]) < 0.02
            )
            if c["pallet"].get("layout") == "cross":
                deposited, checks = self.inspect_drop_placement(
                    b, target, np.linalg.norm(d.cvel[b]) < 0.02
                )
                result["placement_checks"] = checks
                if not deposited:
                    result.update(
                        placement_retry_required=True, on_pallet_after_drop=bool(supported)
                    )
            placement = bool(
                supported
                and error < c["experiment"]["positioning_tolerance"]
                and angle < c["experiment"]["orientation_tolerance_deg"]
            )
            result.update(
                pallet_slot=self.slot_cursor,
                layer=target["layer"],
                supported=bool(supported),
                physically_deposited=deposited,
                position_error_m=error,
                orientation_error_deg=angle,
                placement_passed=placement,
                max_neighbor_displacement_m=self.neighbor_max,
                success=bool(
                    placement
                    and result["transport_passed"]
                    and result["extraction_passed"]
                    and self.neighbor_max < p["maximum_neighbor_shift"]
                    and result.get("max_speed_ratio", 0) <= 1.0
                    and result.get("max_effort_ratio", 0) <= 1.000001
                ),
            )
            if deposited:
                self.removed.add(b)
                self.slot_cursor += 1
                rejected.clear()  # Removing a top load can expose previously rejected candidates.
            else:
                rejected.add(grasp_key)
            if result["success"]:
                self.placed.add(b)
            print(
                f"Placed={deposited}, error={error * 1000:.2f} mm, quality={result['success']}",
                flush=True,
            )
            if deposited:
                self.checkpoint()

    def checkpoint(self):
        if self.settings.get("checkpoint_after_placement", False) and getattr(
            self, "checkpoint_path", None
        ):
            self.write_result(self.checkpoint_path)

    def write_result(self, out):
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".xml").write_text(xml(self.tree))
        save(self.m, self.d, path.with_suffix(".npz"), self.cfg)
        final_audit = []
        for attempt in self.attempts:
            if not attempt.get("physically_deposited"):
                continue
            body = self.m.body(attempt["selected"]).id
            target = self.slots[attempt["pallet_slot"]]
            error = float(np.linalg.norm(self.d.xpos[body] - target["position"]))
            item = ingot_config(self.cfg, attempt["selected"])
            vertices = np.concatenate(
                [
                    part.mesh.vertices
                    for part in components(item, item.get("geometry_seed", self.cfg["seed"]))
                ]
            )
            points = vertices @ self.d.xmat[body].reshape(3, 3).T + self.d.xpos[body]
            inside = bool(
                np.all(
                    np.abs(points[:, :2] - self.cfg["pallet"]["center"])
                    <= np.array([self.cfg["pallet"]["length"], self.cfg["pallet"]["width"]]) / 2
                    + 0.001
                )
            )
            final_audit.append(
                dict(
                    selected=attempt["selected"],
                    slot=attempt["pallet_slot"],
                    position_error_m=error,
                    supported=self.supported(body),
                    within_pallet=inside,
                    world_bounds_m=[points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                    mass_kg=float(self.m.body_mass[body]),
                )
            )
        packing_summary = []
        for layer in range(self.layers):
            names = {
                a["selected"]
                for a in self.attempts
                if a.get("physically_deposited") and a.get("layer") == layer
            }
            area = sum(
                ingot_config(self.cfg, name)["length"] * ingot_config(self.cfg, name)["width"]
                for name in names
            )
            packing_summary.append(
                dict(
                    layer=layer,
                    count=len(names),
                    nominal_footprint_m2=area,
                    nominal_area_fraction=area
                    / (self.cfg["pallet"]["length"] * self.cfg["pallet"]["width"]),
                )
            )
        report = dict(
            run_in_progress=getattr(self, "run_in_progress", False),
            packing_summary=packing_summary,
            final_placement_audit=final_audit,
            scope="KUKA articulated dynamics, ground-truth perception and mock IK/servo/task controllers",
            source=self.source,
            configuration=self.cfg,
            pallet_targets=self.slots,
            layers=self.layers,
            phase=self.phase,
            safe_to_resume=self.phase
            in ("IDLE", "SETTLE", "ABORT_RETRACT", "WAIT_DROPPED_LOAD", "RECOVERY_CLEARANCE"),
            vacuum_pressures_pa=self.vacuum.pressures.tolist() if self.vacuum else [],
            command_position=self.command_pos.tolist(),
            command_rotation=self.command_R.tolist(),
            mode=self.mode,
            attempts=self.attempts,
            initially_on_pallet=self.prefill_count,
            transferred=len(self.removed) - self.prefill_count,
            physically_deposited=len(self.removed),
            remaining=len(self.bodies) - len(self.removed),
            robot_limits_pass=self.max_speed_ratio <= 1.0 and self.max_effort_ratio <= 1.000001,
            quality_passes=len(self.placed),
            complete=len(self.removed) == len(self.bodies)
            and all(a["supported"] and a["within_pallet"] for a in final_audit),
            stop_reason=(
                "running" if getattr(self, "run_in_progress", False) else self.abort_reason
            )
            or (
                "heap_empty"
                if len(self.removed) == len(self.bodies)
                else "no_accessible_grasp_or_attempt_limit"
            ),
            video_initial_still_s=0.0,
            simulation_time_s=float(self.d.time),
            wall_time_s=time.monotonic() - self.wall_start,
            max_speed_ratio=self.max_speed_ratio,
            max_effort_ratio=self.max_effort_ratio,
            robot_collisions=sorted(self.robot_collisions),
            warnings=[int(w.number) for w in self.d.warning],
            mocks=[
                "ground-truth perception",
                "Cartesian IK trajectory planner",
                "effort-limited position servo drivers",
                "vacuum/jaw command interfaces",
            ],
            physical=[
                "robot links/inertia",
                "free 6-DOF ingots",
                "contact/friction",
                "bounded vacuum adhesion",
                "jaw actuator forces",
                "gravity and pallet support",
            ],
        )
        path.with_suffix(".json").write_text(json.dumps(report, indent=2))
        if self.rows:
            import csv

            with path.with_suffix(".csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.rows[0])
                writer.writeheader()
                writer.writerows(self.rows)
        return report
