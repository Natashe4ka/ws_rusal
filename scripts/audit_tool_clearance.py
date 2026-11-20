"""Read-only 50 Hz pose replay to measure tool/KUKA proximity after a run."""

import argparse
import csv
import json
from pathlib import Path
import mujoco

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
a = p.parse_args()
m = mujoco.MjModel.from_xml_path(str(a.source.with_suffix(".xml")))
d = mujoco.MjData(m)
radius = 0.05
for g in range(m.ngeom):
    if m.geom(g).name.startswith("tool_case"):
        m.geom_margin[g] = radius
        m.geom_gap[g] = radius
minimum = radius
closest = None
count = 0
with a.source.with_suffix(".csv").open() as stream:
    for row in csv.DictReader(stream):
        d.qpos[:] = [float(row[f"qpos_{i}"]) for i in range(m.nq)]
        mujoco.mj_forward(m, d)  # Queries saved poses; does not integrate a world.
        for c in d.contact:
            names = [m.geom(g).name for g in [c.geom1, c.geom2]]
            if (
                any(n.startswith("tool_case") for n in names)
                and any(n.startswith("kuka_") for n in names)
                and c.dist < minimum
            ):
                minimum = float(c.dist)
                closest = dict(pair=names, time=float(row["time"]), phase=row["phase"])
        count += 1
report = dict(
    minimum_sampled_gap_m=minimum,
    closest=closest,
    samples=count,
    query_radius_m=radius,
    scope="Recorded pose replay, 20 ms samples; physical execution monitor remains independent",
)
a.source.with_name(a.source.name + "_clearance").with_suffix(".json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
