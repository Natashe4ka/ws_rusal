"""Compute actual world-space envelope, including inclined/vertical pieces."""

import json
import numpy as np
import mujoco
from rusal_sim import ROOT
from rusal_sim.geometry import components
from rusal_sim.heap import restore

for stem in ["heap5", "heap50", "heap250", "heap10_detailed"]:
    base = ROOT / "results/logs" / stem
    if not base.with_suffix(".npz").exists():
        continue
    z = np.load(base.with_suffix(".npz"))
    cfg = json.loads(str(z["config"]))
    report = json.loads(base.with_suffix(".json").read_text())
    m = mujoco.MjModel.from_xml_path(str(base.with_suffix(".xml")))
    d = restore(m, base.with_suffix(".npz"))
    points = []
    for i in range(report["count"]):
        b = m.body(f"ingot_{i}").id
        verts = np.concatenate(
            [p.mesh.vertices for p in components(cfg["ingot"], report["seed"] + i)]
        )
        points.append(verts @ d.xmat[b].reshape(3, 3).T + d.xpos[b])
    points = np.concatenate(points)
    report["height"] = float(points[:, 2].max())
    report["world_bounds"] = [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
    base.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(stem, report["height"], report["settled"])
