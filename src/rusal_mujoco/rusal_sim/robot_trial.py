"""Single-load KUKA dynamic attempt with numerical IK and bounded torque servos.
Research diagnostic: sampled pose collision checks, not a certified motion planner.
"""

import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from . import ROOT, configuration
from .planning import Kinematics
from .vacuum import Vacuum
from .experiment import smoothstep


def run(video=None, out=None):
    from .robot import build
    from .scene import xml

    c = configuration()
    m = mujoco.MjModel.from_xml_string(xml(build(c)[0]))
    d = mujoco.MjData(m)
    k = Kinematics(m)
    v = Vacuum(m, c)
    R = np.diag([1, -1, -1])
    initial = np.array([0.3, 0, 0.345])
    q = None
    for seed in range(20):
        trial = k.ik(initial, R, seed=seed)
        if trial is not None and not k.collisions(trial):
            q = trial
            break
    if q is None:
        raise RuntimeError("No collision-free initial pose")
    d.qpos[k.qa] = q
    mujoco.mj_forward(m, d)
    waypoints = [
        (0, initial, 0),
        (1, np.array([0.3, 0, 0.264]), 0),
        (2, np.array([0.3, 0, 0.264]), 0),
        (4, np.array([0.3, 0, 0.7]), 0),
        (8, np.array([0.3, 0.8, 0.7]), np.pi / 2),
        (10, np.array([0.3, 0.8, 0.424]), np.pi / 2),
        (11, np.array([0.3, 0.8, 0.424]), np.pi / 2),
        (13, np.array([0.3, 0.8, 0.7]), np.pi / 2),
    ]
    collisions = set()
    rows = []
    next_ik = 0
    next_sample = 0
    next_frame = 0
    failed = None
    renderer = writer = None
    if video:
        camera = m.camera("overview").id
        m.cam_pos[camera] = [3, -4, 3]
        xx = np.array([0.8, 0.6, 0])
        yy = np.array([-0.3, 0.4, np.sqrt(0.75)])
        zz = np.cross(xx, yy)
        rq = Rotation.from_matrix(np.column_stack([xx, yy, zz])).as_quat()
        m.cam_quat[camera] = [rq[3], *rq[:3]]
        import imageio.v2 as imageio

        renderer = mujoco.Renderer(m, 720, 960)
        writer = imageio.get_writer(str(video), fps=30)
    try:
        while d.time < 14:
            t = d.time
            if t >= next_ik:
                seg = next(
                    (i for i in range(len(waypoints) - 1) if t <= waypoints[i + 1][0]),
                    len(waypoints) - 2,
                )
                t0, p0, y0 = waypoints[seg]
                t1, p1, y1 = waypoints[seg + 1]
                f = smoothstep((t - t0) / (t1 - t0))
                p = p0 + (p1 - p0) * f
                y = y0 + (y1 - y0) * f
                desired = k.ik(p, Rotation.from_euler("z", y).as_matrix() @ R, start=q, attempts=1)
                if desired is None:
                    failed = "IK failure"
                    break
                q = desired
                next_ik += 0.02
            for i, n in enumerate([f"joint_{j}" for j in range(1, 7)]):
                # Gravity compensation goes through the same effort-limited actuator.
                d.ctrl[m.actuator(n + "_position").id] = (
                    q[i] + d.qfrc_bias[k.va[i]] / c["robot"]["position_kp"]
                )
            v.update(m, d, on=0.7 < t < 10.5)
            for side in ["left", "right"]:
                d.ctrl[m.actuator(f"deploy_motor_{side}").id] = 0
                j = m.joint(f"jaw_{side}")
                d.ctrl[m.actuator(f"jaw_motor_{side}").id] = (
                    -c["jaws"]["kp"] * d.qpos[j.qposadr[0]]
                    - c["jaws"]["damping"] * d.qvel[j.dofadr[0]]
                )
            mujoco.mj_step(m, d)
            if t >= next_sample:
                for contact in d.contact:
                    names = (m.geom(contact.geom1).name, m.geom(contact.geom2).name)
                    if contact.dist < -0.001 and any(n.startswith("kuka_") for n in names):
                        collisions.add(names)
                rows.append(
                    dict(
                        time=t,
                        load=d.xpos[m.body("ingot").id].tolist(),
                        qpos=d.qpos[k.qa].tolist(),
                        qvel=d.qvel[k.va].tolist(),
                        qacc=d.qacc[k.va].tolist(),
                        torque=d.qfrc_actuator[k.va].tolist(),
                        pressure=v.pressures.tolist(),
                    )
                )
                next_sample += 0.02
            if renderer and t >= next_frame:
                from .visualization import status_frame

                renderer.update_scene(d, camera="overview")
                writer.append_data(
                    status_frame(
                        renderer.render(),
                        m,
                        d,
                        v,
                        "RELEASE" if t > 10.5 else "TRANSFER",
                        "KUKA VACUUM",
                    )
                )
                next_frame += 1 / 30
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    import xml.etree.ElementTree as E
    from .robot import URDF

    u = E.parse(URDF).getroot()
    limits = [float(u.find(f"joint[@name='joint_{i}']/limit").get("velocity")) for i in range(1, 7)]
    speed_ratio = float(np.max(np.abs([r["qvel"] for r in rows]) / limits))
    effort_limits = [
        float(u.find(f"joint[@name='joint_{i}']/limit").get("effort")) for i in range(1, 7)
    ]
    torque_ratio = float(np.max(np.abs([r["torque"] for r in rows]) / effort_limits))
    goal_rotation = Rotation.from_euler("z", np.pi / 2).as_matrix() @ np.diag([1, -1, -1])
    orientation_error = Rotation.from_matrix(
        goal_rotation.T @ d.xmat[m.body("ingot").id].reshape(3, 3)
    ).magnitude()
    pos = d.xpos[m.body("ingot").id]
    error = float(np.linalg.norm(pos - [0.3, 0.8, 0.2]))
    report = dict(
        success=error < 0.005
        and orientation_error < np.deg2rad(3)
        and speed_ratio <= 1
        and torque_ratio <= 1.001
        and not collisions,
        position_pass=error < 0.005,
        orientation_error_deg=float(np.rad2deg(orientation_error)),
        max_speed_ratio=speed_ratio,
        max_effort_ratio=torque_ratio,
        unexpected_penetrations=list(collisions),
        final_position=pos.tolist(),
        position_error_m=error,
        failure=failed,
        max_z=max((r["load"][2] for r in rows), default=0),
        warnings=[int(w.number) for w in d.warning],
        scope="KUKA dynamic single-ingot attempt; no heap selection or MoveIt execution",
    )
    samples = (
        Path(out).with_name(Path(out).name + "_samples").with_suffix(".json")
        if out is not None
        else ROOT / "results/logs/robot_trial_samples.json"
    )
    result = (
        Path(out).with_suffix(".json")
        if out is not None
        else ROOT / "results/reports/robot_trial.json"
    )
    for path in (samples, result):
        path.parent.mkdir(parents=True, exist_ok=True)
    samples.write_text(json.dumps(rows))
    result.write_text(json.dumps(report, indent=2))
    return report
