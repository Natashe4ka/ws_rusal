"""MJCF construction. Kinematic fixture drives tool only; load is always free."""

import xml.etree.ElementTree as E
import numpy as np
from .geometry import components, properties


def s(v):
    return " ".join(f"{float(x):.12g}" for x in v)


def add(parent, tag, **kw):
    return E.SubElement(parent, tag, {k: str(v) for k, v in kw.items()})


def root(cfg):
    r = E.Element("mujoco", model="rusal_contact_research")
    add(r, "compiler", angle="radian", autolimits="true")
    q = cfg["solver"]
    opt = add(
        r,
        "option",
        solver=q.get("algorithm", "Newton"),
        sleep_tolerance=q.get("sleep_tolerance", 0.001),
        timestep=q["timestep"],
        iterations=q["iterations"],
        tolerance=q["tolerance"],
        integrator="implicitfast",
        cone="elliptic",
        gravity="0 0 -9.81",
    )
    add(
        opt,
        "flag",
        multiccd="enable",
        sleep="enable" if cfg["solver"].get("sleep", False) else "disable",
    )
    d = add(r, "default")
    c = cfg["contacts"]
    add(
        d,
        "geom",
        friction=s([c["aluminum_aluminum"], c["torsional"], c["rolling"]]),
        solref=s(c["solref"]),
        solimp=s(c["solimp"]),
        condim=4,
    )
    v = add(r, "visual")
    add(v, "global", offwidth=960, offheight=720)
    add(r, "asset")
    w = add(r, "worldbody")
    add(w, "light", pos="0 -1 3", dir="0 0 -1", diffuse=".9 .9 .9")
    add(
        w,
        "geom",
        name="floor",
        type="plane",
        size="4 4 .1",
        rgba=".22 .25 .28 1",
        friction=s([c["aluminum_floor"], 0.001, 0.0001]),
    )
    p = cfg["pallet"]
    x, y = p["center"]
    add(
        w,
        "geom",
        name="pallet",
        type="box",
        pos=s([x, y, p["height"] / 2]),
        size=s([p["length"] / 2, p["width"] / 2, p["height"] / 2]),
        rgba=".48 .30 .13 1",
    )
    for name, pos, target in [
        ("overview", [1.4, -1.5, 1.1], [0, 0, 0.15]),
        ("heap", [0, -2, 2], [0, 0, 0.2]),
        ("gripper", [0.65, -0.7, 0.55], [0, 0, 0.15]),
        ("pallet", [1.6, -0.8, 1.3], [x, y, 0.3]),
        ("side", [0, -1, 0.25], [0, 0, 0.18]),
    ]:
        z = np.array(pos) - target
        z = z / np.linalg.norm(z)
        xx = np.cross([0, 0, 1], z)
        xx /= np.linalg.norm(xx)
        yy = np.cross(z, xx)
        add(w, "camera", name=name, pos=s(pos), xyaxes=s([*xx, *yy]))
    return r


def ingot(r, cfg, name="ingot", pos=(0, 0, 0.021), quat=(0, 1, 0, 0), seed=42):
    w = r.find("worldbody")
    a = r.find("asset")
    b = add(w, "body", name=name, pos=s(pos), quat=s(quat))
    add(b, "freejoint", name=f"{name}_free")
    p = properties(cfg, seed)
    I = np.array(p["inertia"])
    add(
        b,
        "inertial",
        mass=cfg["mass"],
        pos=s(p["com"]),
        fullinertia=s([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]),
    )
    for c in components(cfg, seed):
        mn = f"{name}_{c.name}"
        add(
            a,
            "mesh",
            name=mn,
            vertex=s(c.mesh.vertices.ravel()),
            face=" ".join(str(int(v)) for v in c.mesh.faces.ravel()),
        )
        add(b, "geom", name=mn, type="mesh", mesh=mn, rgba=".62 .67 .72 1", mass=0)
    add(
        b,
        "geom",
        name=f"{name}_mark",
        type="box",
        pos=s([cfg["length"] / 2 + 0.0001, 0, 0]),
        size=".0002 .035 .008",
        rgba=".9 .13 .04 1",
        contype=0,
        conaffinity=0,
        mass=0,
    )
    return b


def tool(r, cfg, parent=None, fixture=True):
    w = r.find("worldbody") if parent is None else parent
    b = add(w, "body", name="tool", pos=s([0, 0, cfg["fixture"]["initial_height"]]))
    if fixture:
        stage = cfg["fixture"]
        for axis, vector in [("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")]:
            add(
                b,
                "joint",
                name=f"fixture_{axis}",
                type="slide",
                axis=vector,
                damping=stage["damping"],
            )
        for axis, vector in [("rx", "1 0 0"), ("ry", "0 1 0"), ("rz", "0 0 1")]:
            add(
                b,
                "joint",
                name=f"fixture_{axis}",
                type="hinge",
                axis=vector,
                damping=stage["angular_damping"],
            )
    profile = cfg.get("gripper", {})
    add(
        b,
        "geom",
        name="tool_case",
        type="box",
        size=s(profile.get("case_half_sizes", [0.2, 0.055, 0.025])),
        pos=s(profile.get("case_center", [0, 0, 0.04])),
        mass=profile.get("case_mass", 6),
        rgba=s(profile.get("color", [0.9, 0.55, 0.06, 1])),
        contype=0,
        conaffinity=0,
    )
    for extra in profile.get("extra_cases", []):
        add(
            b,
            "geom",
            name="tool_case_" + extra["name"],
            type="box",
            size=s(extra["half_sizes"]),
            pos=s(extra["center"]),
            mass=extra["mass"],
            rgba=".25 .32 .42 1",
        )
    add(b, "site", name="tcp", pos="0 0 -.025", size=".004")
    v = cfg["vacuum"]
    a = r.find("actuator")
    if a is None:
        a = add(r, "actuator")
    if fixture:
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            add(
                a,
                "position",
                name=f"fixture_motor_{axis}",
                joint=f"fixture_{axis}",
                kp=cfg["fixture"]["kp"],
                kv=cfg["fixture"]["kv"],
                forcerange=s([-cfg["fixture"]["max_force"], cfg["fixture"]["max_force"]]),
            )
    centers = (
        v.get(
            "cell_centers", [[x, 0] for x in (v["cup_x"] if v["layout"] == "V1" else [-0.09, 0.09])]
        )
        if v.get("enabled", True)
        else []
    )
    for i, (x, y) in enumerate(centers):
        c = add(b, "body", name=f"cup_{i}", pos=s([x, y, 0]))
        add(
            c,
            "joint",
            name=f"cup_slide_{i}",
            type="slide",
            axis="0 0 1",
            range=s([-v["compliance"], v["compliance"]]),
            stiffness=v["spring_stiffness"],
            damping=v["spring_damping"],
        )
        # Cylinder contact is a finite pad proxy, not a resolved elastomer membrane.
        size = (
            s([v["diameter"] / 2, v["pad_half_thickness"]])
            if v["layout"] == "V1"
            else s([v["rectangle"][0] / 2, v["rectangle"][1] / 2, v["pad_half_thickness"]])
        )
        add(
            c,
            "geom",
            name=f"cup_pad_{i}",
            type="cylinder" if v["layout"] == "V1" else "box",
            size=size,
            pos=s([0, 0, v["pad_center_z"]]),
            gap=v["adhesion_gap"],
            mass=v.get("pad_mass", 0.15),
            rgba=".08 .14 .17 1",
            friction=s([cfg["contacts"]["aluminum_rubber"], 0.001, 0.0001]),
            priority=1,
        )
        add(a, "adhesion", name=f"vacuum_{i}", body=f"cup_{i}", gain=1, ctrlrange="0 2000")
    j = cfg["jaws"]
    for side, sign in [("left", 1), ("right", -1)] if j.get("enabled", True) else []:
        dep = add(
            b,
            "body",
            name=f"deploy_{side}",
            pos=s([0, sign * j["open_half_gap"], j.get("mount_z", 0.035)]),
        )
        if not j.get("fixed", False):
            add(
                dep,
                "joint",
                name=f"deploy_{side}",
                type="slide",
                axis="0 0 -1",
                range=s([0, j["deploy_travel"]]),
                damping=30,
            )
            add(dep, "inertial", pos="0 0 0", mass=0.5, diaginertia=".001 .001 .001")
        jaw = add(dep, "body", name=f"jaw_{side}")
        add(
            jaw,
            "joint",
            name=f"jaw_{side}",
            type="slide",
            axis=s([0, -sign, 0]),
            range=s([0, j["closed_travel"]]),
            damping=j["damping"],
        )
        add(
            jaw,
            "geom",
            name=f"jaw_pad_{side}",
            type="box",
            size=s(j.get("pad_half_sizes", [0.14, 0.008, 0.025])),
            mass=j.get("finger_mass", 0.6),
            rgba=".22 .24 .26 1",
            priority=1,
            friction=s([cfg["contacts"]["aluminum_steel"], 0.001, 0.0001]),
        )
        if j["tip"] == "support":
            add(
                jaw,
                "geom",
                name=f"jaw_support_{side}",
                type="box",
                pos=s([0, -sign * 0.012, -0.023]),
                size=".13 .02 .003",
                mass=0.1,
                rgba=".3 .3 .3 1",
            )
        add(
            a,
            "motor",
            name=f"jaw_motor_{side}",
            joint=f"jaw_{side}",
            ctrlrange=s([-j["force_per_jaw"], j["force_per_jaw"]]),
        )
        if not j.get("fixed", False):
            add(
                a,
                "position",
                name=f"deploy_motor_{side}",
                joint=f"deploy_{side}",
                kp=j.get("deploy_kp", 5000),
                kv=j.get("deploy_kv", 80),
                ctrlrange=s([0, j["deploy_travel"]]),
                forcerange="-300 300",
            )
    return b


def scene(cfg, surface="smooth", fixture=True):
    r = root(cfg)
    ingot(
        r,
        cfg["ingot"],
        seed=cfg["seed"],
        pos=(0, 0, cfg["ingot"]["thickness"] / 2 + 0.001),
        quat=(0, 1, 0, 0) if surface == "smooth" else (1, 0, 0, 0),
    )
    if fixture:
        tool(r, cfg)
    return r


def xml(r):
    return E.tostring(r, encoding="unicode")
