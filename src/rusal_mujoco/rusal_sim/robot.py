"""Official KUKA URDF conversion with explicit fixed flange and independent FK check."""

from pathlib import Path
import subprocess
import os
import shutil
import xml.etree.ElementTree as E
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from . import ROOT
from .scene import add, s, root, tool, ingot, xml

REPO = ROOT / "src/third_party/kuka_robot_descriptions"
URDF = ROOT / "src/rusal_description/urdf/kr50_r2500.urdf"


def vec(text, default="0 0 0"):
    return np.fromstring(text or default, sep=" ")


def origin(el):
    o = el.find("origin")
    return (
        (vec(o.get("xyz")), Rotation.from_euler("xyz", vec(o.get("rpy"))))
        if o is not None
        else (np.zeros(3), Rotation.identity())
    )


def quat(rot):
    x, y, z, w = rot.as_quat()
    return s([w, x, y, z])


def expand():
    """Regenerate the checked-in URDF using the optional ROS xacro tool."""
    macro = REPO / "kuka_iontec_support/urdf/kr50_r2500_macro.xacro"
    wrapper = URDF.with_suffix(".xacro")
    include = os.path.relpath(macro, URDF.parent)
    wrapper.write_text(
        f'<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="kr50_r2500"><xacro:include filename="{include}"/><xacro:kuka_kr50_r2500_robot prefix="" package_name="kuka_iontec_support"/></robot>\n'
    )
    executable = shutil.which("xacro")
    if executable is None and Path("/opt/ros/jazzy/bin/xacro").exists():
        executable = "/opt/ros/jazzy/bin/xacro"
    if executable is None:
        raise RuntimeError("URDF regeneration requires xacro. Source the ROS environment first.")
    result = subprocess.run(
        [executable, wrapper.name], cwd=URDF.parent, check=True, capture_output=True, text=True
    )
    URDF.write_text(result.stdout)
    return E.parse(URDF).getroot()


def append_robot(r, urdf=None, base=(-1.2, 0, 0), gripper=True, cfg=None):
    cfg = __import__("rusal_sim").configuration() if cfg is None else cfg
    u = urdf if urdf is not None else E.parse(URDF).getroot()
    links = {x.get("name"): x for x in u.findall("link")}
    a = r.find("asset")
    w = r.find("worldbody")
    aact = r.find("actuator")
    if aact is None:
        aact = add(r, "actuator")
    base_body = add(w, "body", name="base_link", pos=s(base))
    bodies = {"base_link": base_body}
    pending = list(u.findall("joint"))
    while pending:
        progress = False
        for j in pending[:]:
            parent = j.find("parent").get("link")
            child = j.find("child").get("link")
            if parent not in bodies:
                continue
            pos, rot = origin(j)
            b = add(bodies[parent], "body", name=child, pos=s(pos), quat=quat(rot))
            bodies[child] = b
            if j.get("type") != "fixed":
                lim = j.find("limit")
                effort = float(lim.get("effort"))
                add(
                    b,
                    "joint",
                    name=j.get("name"),
                    type="hinge",
                    axis=j.find("axis").get("xyz"),
                    range=f"{lim.get('lower')} {lim.get('upper')}",
                    damping=cfg["robot"]["joint_damping"],
                    armature=0.1,
                )
                add(
                    aact,
                    "position",
                    name=j.get("name") + "_position",
                    joint=j.get("name"),
                    kp=cfg["robot"]["position_kp"],
                    kv=cfg["robot"]["position_kv"],
                    forcerange=s([-effort, effort]),
                )
            pending.remove(j)
            progress = True
        if not progress:
            raise ValueError("Disconnected URDF tree")
    for name, b in bodies.items():
        link = links[name]
        inert = link.find("inertial")
        if inert is not None:
            pos, rot = origin(inert)
            ii = inert.find("inertia")
            I = np.array(
                [
                    [float(ii.get("ixx")), float(ii.get("ixy")), float(ii.get("ixz"))],
                    [float(ii.get("ixy")), float(ii.get("iyy")), float(ii.get("iyz"))],
                    [float(ii.get("ixz")), float(ii.get("iyz")), float(ii.get("izz"))],
                ]
            )
            R = rot.as_matrix()
            I = R @ I @ R.T
            add(
                b,
                "inertial",
                mass=inert.find("mass").get("value"),
                pos=s(pos),
                fullinertia=s([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]),
            )
        for k, col in enumerate(link.findall("collision")):
            mesh = col.find("geometry/mesh")
            path = mesh.get("filename").replace("package://", str(REPO) + "/")
            mn = f"kuka_{name}_{k}"
            add(a, "mesh", name=mn, file=path)
            pos, rot = origin(col)
            add(
                b,
                "geom",
                name=mn,
                type="mesh",
                mesh=mn,
                pos=s(pos),
                quat=quat(rot),
                rgba=".94 .46 .04 1",
                mass=0,
            )
    contact = r.find("contact")
    if contact is None:
        contact = add(r, "contact")
    add(contact, "exclude", body1="base_link", body2="link_1")
    if gripper:
        # Tool local -Z points out of flange +Z; rigid calibrated mount, no object attachment.
        mount = add(bodies["flange"], "body", name="vmg_mount", quat="0 1 0 0")
        rc = cfg["robot"]
        add(
            mount,
            "geom",
            name="vmg_adapter",
            type="cylinder",
            size=s([rc["adapter_radius"], rc["adapter_length"] / 2]),
            pos=s([0, 0, -rc["adapter_length"] / 2]),
            density=rc["adapter_density"],
            rgba=".4 .43 .46 1",
        )
        tb = tool(r, cfg, parent=mount, fixture=False)
        tb.set("pos", s([0, 0, -cfg["robot"]["tool_mount_distance"]]))
    return bodies


def fk(u, q, base=(-1.2, 0, 0)):
    transforms = {"base_link": np.eye(4)}
    transforms["base_link"][:3, 3] = base
    joints = list(u.findall("joint"))
    while joints:
        for j in joints[:]:
            pa = j.find("parent").get("link")
            ch = j.find("child").get("link")
            if pa not in transforms:
                continue
            p, R = origin(j)
            T = np.eye(4)
            T[:3, :3] = R.as_matrix()
            T[:3, 3] = p
            if j.get("type") != "fixed":
                A = np.eye(4)
                A[:3, :3] = Rotation.from_rotvec(
                    vec(j.find("axis").get("xyz")) * q[j.get("name")]
                ).as_matrix()
                T = T @ A
            transforms[ch] = transforms[pa] @ T
            joints.remove(j)
    return transforms["flange"]


def build(cfg):
    u = E.parse(URDF).getroot()
    r = root(cfg)
    append_robot(r, u, base=cfg["robot"]["base"], cfg=cfg)
    ingot(r, cfg["ingot"], pos=(0.3, 0, cfg["ingot"]["thickness"] / 2 + 0.001))
    return r, u


def validate(cfg, n=100):
    r, u = build(cfg)
    m = mujoco.MjModel.from_xml_string(xml(r))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(cfg["seed"])
    errors = []
    for _ in range(n):
        q = {}
        for i in range(1, 7):
            j = m.joint(f"joint_{i}")
            value = rng.uniform(*m.jnt_range[j.id])
            d.qpos[j.qposadr[0]] = value
            q[j.name] = value
        mujoco.mj_forward(m, d)
        T = fk(u, q)
        b = m.body("flange").id
        errors.append(
            [
                np.linalg.norm(T[:3, 3] - d.xpos[b]),
                Rotation.from_matrix(T[:3, :3].T @ d.xmat[b].reshape(3, 3)).magnitude(),
            ]
        )
    target = ROOT / "src/rusal_description/mjcf/cell.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(xml(r))
    return dict(
        samples=n,
        max_position_error_m=float(np.max(errors, axis=0)[0]),
        max_orientation_error_rad=float(np.max(errors, axis=0)[1]),
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
    )
