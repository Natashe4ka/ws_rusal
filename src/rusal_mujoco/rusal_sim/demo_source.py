"""Controlled stacked infeed for complete palletizing regression, not a random heap."""

import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from . import configuration
from .scene import root, ingot, xml
from .heap import save


def generate_stack(count=12, out="results/logs/stack12", seed=42, surface="smooth"):
    if count < 1:
        raise ValueError("count must be positive")
    c = configuration()
    c["pallet"]["center"] = [4, 4]  # Kept clear while the source physically settles.
    r = root(c)
    rng = np.random.default_rng(seed)
    for i in range(count):
        pile = i % 6
        layer = i // 6
        x = (pile % 3 - 1) * 0.45
        y = (pile // 3 - 0.5) * 0.54
        yaw = rng.uniform(-0.025, 0.025)
        smooth = surface == "smooth" or (surface == "mixed" and i % 2 == 0)
        q = (
            Rotation.from_euler("z", yaw) * Rotation.from_euler("x", np.pi if smooth else 0)
        ).as_quat()
        ingot(
            r,
            c["ingot"],
            name=f"ingot_{i}",
            pos=[x, y, (layer + 0.5) * c["ingot"]["thickness"] + 0.002],
            quat=[q[3], *q[:3]],
            seed=seed + i,
        )
    m = mujoco.MjModel.from_xml_string(xml(r))
    d = mujoco.MjData(m)
    for _ in range(round(2 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    positions = d.xpos.copy()
    max_drift = 0.0
    for _ in range(round(1 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        max_drift = max(max_drift, float(np.max(np.linalg.norm(d.xpos - positions, axis=1))))
    mujoco.mj_energyVel(m, d)
    kinetic = float(d.energy[1])
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".xml").write_text(xml(r))
    save(m, d, path.with_suffix(".npz"), c)
    report = dict(
        count=count,
        scenario="controlled six-pile stacked infeed; not random heap",
        seed=seed,
        surface=surface,
        kinetic_energy_j=kinetic,
        max_drift_m=max_drift,
        validated_equilibrium=kinetic < 0.001 and max_drift < 0.005,
        settled=kinetic < 0.001 and max_drift < 0.005,
        warnings=[int(w.number) for w in d.warning],
    )
    path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report
