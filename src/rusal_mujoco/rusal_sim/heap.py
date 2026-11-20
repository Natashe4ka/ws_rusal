"""Sequential gravitational deposition; save complete MuJoCo integration state."""

import json
import time
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from .scene import root, ingot, xml
from .geometry import components, properties
from .batch import make_batch, ingot_config


def kinetic(m, d):
    mujoco.mj_energyVel(m, d)
    return float(d.energy[1])


def save(m, d, path, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, state, spec)
    np.savez(
        path,
        state=state,
        spec=int(spec),
        config=json.dumps(cfg),
        sleeping_tree_counters=d.tree_asleep.copy(),
    )


def restore(m, path):
    z = np.load(path)
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, z["state"], int(z["spec"]))
    mujoco.mj_forward(m, d)
    return d


def generate(cfg, count=10, seed=42, out="results/logs/heap", variant="A"):
    import copy

    wall_start = time.monotonic()
    cfg = copy.deepcopy(cfg)
    cfg["pallet"]["center"] = [3, 3]
    cfg["ingot"]["variant"] = variant
    r = root(cfg)
    rng = np.random.default_rng(seed)
    m = d = None
    maxz = 0
    for i in range(count):
        old = d
        xy = rng.uniform(
            -np.array(cfg["heap"]["area"]) / 2 + 0.25, np.array(cfg["heap"]["area"]) / 2 - 0.25
        )
        q = Rotation.random(random_state=rng).as_quat()
        ingot(
            r,
            cfg["ingot"],
            name=f"ingot_{i}",
            pos=(*xy, maxz + 0.4),
            quat=[q[3], *q[:3]],
            seed=seed + i,
        )
        m = mujoco.MjModel.from_xml_string(xml(r))
        d = mujoco.MjData(m)
        if old is not None:
            d.qpos[: len(old.qpos)] = old.qpos
            d.qvel[: len(old.qvel)] = old.qvel
            d.time = old.time
        for _ in range(round(cfg["heap"]["drop_interval"] / m.opt.timestep)):
            mujoco.mj_step(m, d)
        if count >= 50 and (i + 1) % 25 == 0:
            print(
                f"deposited {i + 1}/{count}, wall={time.monotonic() - wall_start:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        maxz = max(d.xpos[m.body(f"ingot_{k}").id, 2] for k in range(i + 1)) + 0.225
    for _ in range(round(cfg["heap"]["settle_time"] / m.opt.timestep)):
        mujoco.mj_step(m, d)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".xml").write_text(xml(r))
    save(m, d, path.with_suffix(".npz"), cfg)
    energy = kinetic(m, d)
    points = []
    for i in range(count):
        b = m.body(f"ingot_{i}").id
        vertices = np.concatenate([p.mesh.vertices for p in components(cfg["ingot"], seed + i)])
        points.append(vertices @ d.xmat[b].reshape(3, 3).T + d.xpos[b])
    points = np.concatenate(points)
    report = dict(
        wall_seconds=time.monotonic() - wall_start,
        sleep_enabled=cfg["solver"].get("sleep", False),
        count=count,
        seed=seed,
        variant=variant,
        kinetic_energy_j=energy,
        settled=energy < cfg["heap"]["kinetic_energy_tolerance"],
        time=d.time,
        height=float(points[:, 2].max()),
        world_bounds=[points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        max_penetration_m=None
        if cfg["solver"].get("sleep", False)
        else max((max(0, -c.dist) for c in d.contact), default=0),
        warnings=[int(w.number) for w in d.warning],
    )
    path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def generate_compact(
    cfg, count=5, seed=42, out="results/logs/compact5", surface="mixed", initial_source=None
):
    """Low-height random deposition into one overlapping heap, without walls.

    Initial broad-face orientations are randomly flipped, yawed and tilted.
    Poses are transferred only when adding a new body before the next episode;
    every subsequent displacement is produced by mj_step.
    """
    import copy

    if count < 1:
        raise ValueError("count must be positive")
    if surface not in ("mixed", "smooth", "ribbed"):
        raise ValueError(surface)
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    make_batch(cfg, count, seed)
    cfg["pallet"]["center"] = [3, 3]
    cfg["solver"]["sleep"] = False  # Validate actual residual movement.
    p = cfg["compact_heap"]
    rng = np.random.default_rng(seed)
    # Spread the first three drops across a triangular base. Later drops bridge
    # that base, rather than repeatedly landing over the same tiny XY patch.
    layout_angle = rng.uniform(-np.pi, np.pi) if count > 1 else 0.0
    r = root(cfg)
    m = d = None
    top = 0.0
    start = time.monotonic()

    def world_vertices(model, data):
        return [
            np.concatenate(
                [
                    part.mesh.vertices
                    for part in components(ingot_config(cfg, f"ingot_{i}"), seed + i)
                ]
            )
            @ data.xmat[model.body(f"ingot_{i}").id].reshape(3, 3).T
            + data.xpos[model.body(f"ingot_{i}").id]
            for i in range((model.nq) // 7)
        ]

    if initial_source is None:
        for i in range(count):
            old = d
            tilt = rng.uniform(-p["maximum_tilt_deg"], p["maximum_tilt_deg"], 2)
            flip = np.pi if rng.integers(2) else 0.0
            if surface != "mixed":
                flip = np.pi if surface == "smooth" else 0.0
            rotation = (
                Rotation.from_euler("z", rng.uniform(-np.pi, np.pi))
                * Rotation.from_euler("xy", np.deg2rad(tilt))
                * Rotation.from_euler("x", flip)
            )
            vertices = (
                np.concatenate(
                    [
                        part.mesh.vertices
                        for part in components(ingot_config(cfg, f"ingot_{i}"), seed + i)
                    ]
                )
                @ rotation.as_matrix().T
            )
            xy = rng.uniform(-np.array(p["xy_span"]) / 2, np.array(p["xy_span"]) / 2)
            if count > 1:
                base_count = min(3, count)
                angle = layout_angle + (
                    2 * np.pi * i / base_count
                    if i < base_count
                    else np.pi * (i - base_count) + np.pi / 3
                )
                radius = p["base_radius"] if i < base_count else p["upper_radius"]
                xy += radius * np.array([np.cos(angle), np.sin(angle)])
                if count > 5:
                    # Larger heaps need a broad random deposition area, rather
                    # than repeatedly filling the two upper positions of heap5.
                    spread = p.get("large_heap_radius", 0.26)
                    angle = rng.uniform(-np.pi, np.pi)
                    radius = spread * np.sqrt(rng.uniform())
                    xy = radius * np.array([np.cos(angle), np.sin(angle)])
            height = top - vertices[:, 2].min() + p["drop_clearance"]
            q = rotation.as_quat()
            ingot(
                r,
                ingot_config(cfg, f"ingot_{i}"),
                name=f"ingot_{i}",
                pos=[*xy, height],
                quat=[q[3], *q[:3]],
                seed=seed + i,
            )
            m = mujoco.MjModel.from_xml_string(xml(r))
            d = mujoco.MjData(m)
            if old is not None:
                d.qpos[: old.qpos.size] = old.qpos
                d.qvel[: old.qvel.size] = old.qvel
                d.time = old.time
            for _ in range(round(p["drop_interval"] / m.opt.timestep)):
                mujoco.mj_step(m, d)
            top = max(points[:, 2].max() for points in world_vertices(m, d))
            if count > 5 and (i + 1) % 5 == 0:
                print(
                    f"Gravity-deposited {i + 1}/{count} ingots; wall={time.monotonic() - start:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
    else:
        import xml.etree.ElementTree as E

        source = Path(initial_source)
        r = E.parse(source.with_suffix(".xml")).getroot()
        m = mujoco.MjModel.from_xml_path(str(source.with_suffix(".xml")))
        if m.nq != count * 7:
            raise ValueError("Deposition snapshot body count differs")
        d = restore(m, source.with_suffix(".npz"))
    for _ in range(round(p["settle_time"] / m.opt.timestep)):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    bodies = [m.body(f"ingot_{i}").id for i in range(count)]
    validation_history = []
    extra_settle = 0.0
    while True:
        reference = d.xpos[bodies].copy()
        reference_quaternions = d.xquat[bodies].copy()
        bounding_radius = max(
            np.linalg.norm(
                [ingot_config(cfg, f"ingot_{i}")[k] for k in ("length", "width", "thickness")]
            )
            / 2
            for i in range(count)
        )
        max_drift = 0.0
        max_energy = 0.0
        max_penetration = 0.0
        for _ in range(round(p["validation_window"] / m.opt.timestep)):
            mujoco.mj_step(m, d)
            angles = 2 * np.arccos(
                np.clip(np.abs(np.sum(d.xquat[bodies] * reference_quaternions, axis=1)), 0, 1)
            )
            max_drift = max(
                max_drift,
                float(
                    np.max(
                        np.linalg.norm(d.xpos[bodies] - reference, axis=1)
                        + bounding_radius * angles
                    )
                ),
            )
            max_energy = max(max_energy, kinetic(m, d))
            max_penetration = max(
                max_penetration, max((max(0.0, -c.dist) for c in d.contact), default=0.0)
            )
        warnings = [int(w.number) for w in d.warning]
        stable = bool(
            max_drift < p["maximum_drift"]
            and max_energy < p["maximum_kinetic_energy"]
            and max_penetration < p["maximum_penetration"]
            and not any(warnings)
        )
        validation_history.append(
            dict(
                time=float(d.time),
                drift_m=max_drift,
                energy_j=max_energy,
                penetration_m=max_penetration,
                stable=stable,
            )
        )
        if stable or extra_settle >= p.get("maximum_extra_settle_time", 0):
            break
        seconds = min(
            p.get("settle_retry_interval", 10.0),
            p.get("maximum_extra_settle_time", 0) - extra_settle,
        )
        for _ in range(round(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)
        extra_settle += seconds
        print(
            f"Heap still settling; extra physics time={extra_settle:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    points = np.concatenate(world_vertices(m, d))
    bounds = np.array([points.min(axis=0), points.max(axis=0)])
    footprint = bounds[1, :2] - bounds[0, :2]
    warnings = [int(w.number) for w in d.warning]
    stable = bool(
        max_drift < p["maximum_drift"]
        and max_energy < p["maximum_kinetic_energy"]
        and max_penetration < p["maximum_penetration"]
        and not any(warnings)
    )
    edges = {
        tuple(sorted((int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2]))))
        for c in d.contact
        if c.dist <= 0
        and m.geom_bodyid[c.geom1] in bodies
        and m.geom_bodyid[c.geom2] in bodies
        and m.geom_bodyid[c.geom1] != m.geom_bodyid[c.geom2]
    }
    connected = {bodies[0]}
    while True:
        expanded = (
            connected
            | {b for a, b in edges if a in connected}
            | {a for a, b in edges if b in connected}
        )
        if expanded == connected:
            break
        connected = expanded
    contact_connected = len(connected) == count
    floor_id = m.geom("floor").id
    floor_supported = sorted(
        {
            int(m.geom_bodyid[c.geom2 if c.geom1 == floor_id else c.geom1])
            for c in d.contact
            if floor_id in (c.geom1, c.geom2) and c.dist <= 0
        }
        & set(bodies)
    )
    # A connected narrow tower must not pass as a spread-out pile.
    spread = len(floor_supported) >= min(3, count)
    component_fraction = len(connected) / count
    compact = bool(np.all(footprint <= p["maximum_footprint"]) and contact_connected and spread)
    report = dict(
        count=count,
        seed=seed,
        surface=surface,
        variant=cfg["ingot"]["variant"],
        scenario="compact random gravity-deposited heap; no walls or pose corrections",
        settled=stable and compact,
        validated_equilibrium=stable,
        compact=compact,
        contact_graph_connected=contact_connected,
        ingot_contact_edges=[list(e) for e in sorted(edges)],
        floor_supported_count=len(floor_supported),
        floor_supported_bodies=floor_supported,
        contact_component_fraction=component_fraction,
        validation_history=validation_history,
        initial_source=str(initial_source) if initial_source else None,
        kinetic_energy_j=kinetic(m, d),
        maximum_window_energy_j=max_energy,
        maximum_window_drift_m=max_drift,
        max_penetration_m=max_penetration,
        footprint_m=footprint.tolist(),
        world_bounds=bounds.tolist(),
        height=float(bounds[1, 2]),
        time=float(d.time),
        wall_seconds=time.monotonic() - start,
        warnings=warnings,
    )
    if cfg.get("ingots"):
        report["ingots"] = {
            name: dict(
                dimensions_m=[v[k] for k in ("length", "width", "thickness")],
                mass_kg=v["mass"],
                properties=properties(v, v["geometry_seed"]),
            )
            for name, v in cfg["ingots"].items()
        }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".xml").write_text(xml(r))
    save(m, d, path.with_suffix(".npz"), cfg)
    path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    if not report["settled"]:
        raise RuntimeError(
            f"Compact heap failed equilibrium/footprint checks; inspect {path.with_suffix('.json')}"
        )
    return report
