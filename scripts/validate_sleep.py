"""Wake accelerated deposited heaps and check that they remain stable under full dynamics."""

import json, time
import numpy as np
import mujoco
from rusal_sim import ROOT
from rusal_sim.heap import restore, kinetic

reports = []
for n in [5, 250]:
    p = ROOT / f"results/logs/heap{n}_sleep"
    if not p.with_suffix(".npz").exists():
        continue
    m = mujoco.MjModel.from_xml_path(str(p.with_suffix(".xml")))
    d = restore(m, p.with_suffix(".npz"))
    initial = d.qpos.copy()
    m.opt.enableflags = int(m.opt.enableflags) & ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    start = time.monotonic()
    for _ in range(round(1 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    displacements = [
        np.linalg.norm(d.qpos[7 * i : 7 * i + 3] - initial[7 * i : 7 * i + 3]) for i in range(n)
    ]
    r = dict(
        count=n,
        wake_time_s=1,
        max_displacement_m=float(max(displacements)),
        kinetic_energy_j=kinetic(m, d),
        wall_seconds=time.monotonic() - start,
        max_penetration_m=max((max(0, -c.dist) for c in d.contact), default=0),
        warnings=[int(w.number) for w in d.warning],
    )
    r["stable_after_wake"] = (
        r["max_displacement_m"] < 0.005 and r["kinetic_energy_j"] < 0.1 and not any(r["warnings"])
    )
    reports.append(r)
    print(r, flush=True)
(ROOT / "results/reports/sleep_validation.json").write_text(json.dumps(reports, indent=2))
