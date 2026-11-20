"""Validate a coarsely generated heap using CG/1ms fully awake dynamics with tighter convergence controls.
The prepared trajectory is not used as evidence about gripping performance.
"""

import json, time
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import mujoco
from rusal_sim import ROOT
from rusal_sim.heap import restore, save, kinetic
from rusal_sim.geometry import components

source = ROOT / "results/logs/heap250_cg"
out = ROOT / "results/logs/heap250_refined"
if not source.with_suffix(".npz").exists():
    raise SystemExit("Coarse generation has not finished")
z = np.load(source.with_suffix(".npz"))
cfg = json.loads(str(z["config"]))
preparation = cfg["solver"].copy()
cfg["solver"].update(algorithm="CG", timestep=0.001, iterations=200, tolerance=1e-8, sleep=False)
r = E.parse(source.with_suffix(".xml")).getroot()
opt = r.find("option")
for key, value in [
    ("solver", "CG"),
    ("timestep", ".001"),
    ("iterations", "200"),
    ("tolerance", "1e-8"),
]:
    opt.set(key, value)
opt.find("flag").set("sleep", "disable")
out.with_suffix(".xml").write_text(E.tostring(r, encoding="unicode"))
m = mujoco.MjModel.from_xml_path(str(out.with_suffix(".xml")))
d = restore(m, source.with_suffix(".npz"))
initial = d.qpos.reshape(-1, 7)[:, :3].copy()
start = time.monotonic()
last_displacement = float("inf")
windows = 0
for window in range(5):
    prior = d.qpos.reshape(-1, 7)[:, :3].copy()
    for _ in range(1000):
        mujoco.mj_step(m, d)
    windows += 1
    last_displacement = float(np.max(np.linalg.norm(d.qpos.reshape(-1, 7)[:, :3] - prior, axis=1)))
    print(
        f"refinement window {windows}: max drift={last_displacement:.6g} m, KE={kinetic(m, d):.6g} J",
        flush=True,
    )
    if last_displacement < 0.005 and kinetic(m, d) < 0.1:
        break
positions = d.qpos.reshape(-1, 7)[:, :3]
displacement = float(np.max(np.linalg.norm(positions - initial, axis=1)))
energy = kinetic(m, d)
penetration = max((max(0, -c.dist) for c in d.contact), default=0)
points = []
vertices = np.concatenate([p.mesh.vertices for p in components(cfg["ingot"])])
for i in range(250):
    b = m.body(f"ingot_{i}").id
    points.append(vertices @ d.xmat[b].reshape(3, 3).T + d.xpos[b])
points = np.concatenate(points)
report = dict(
    count=250,
    seed=42,
    variant="A",
    sleep_enabled=False,
    kinetic_energy_j=energy,
    settled=energy < 0.1,
    max_penetration_m=penetration,
    height=float(points[:, 2].max()),
    world_bounds=[points.min(axis=0).tolist(), points.max(axis=0).tolist()],
    max_relaxation_displacement_m=displacement,
    refinement_sim_seconds=windows,
    last_window_displacement_m=last_displacement,
    refinement_wall_seconds=time.monotonic() - start,
    preparation_solver=preparation,
    final_solver=cfg["solver"],
    solver_iterations_last_step=int(np.max(d.solver_niter)),
    warnings=[int(w.number) for w in d.warning],
    scope="250-body heap stability test; no extraction performance inference",
)
report["validated_equilibrium"] = (
    energy < 0.1
    and last_displacement < 0.005
    and penetration < 0.001
    and not any(report["warnings"])
)
report["energy_settled"] = report["settled"]
report["settled"] = report["validated_equilibrium"]
save(m, d, out.with_suffix(".npz"), cfg)
out.with_suffix(".json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
