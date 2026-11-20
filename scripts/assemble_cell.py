"""Embed settled detailed heap into a separate KUKA cell without changing its LOD."""

import copy
import xml.etree.ElementTree as E
import mujoco
from rusal_sim import ROOT, configuration
from rusal_sim.scene import root, xml
from rusal_sim.robot import append_robot
from rusal_sim.heap import restore

c = configuration()
source = ROOT / "results/logs/heap10_detailed"
h = E.parse(source.with_suffix(".xml")).getroot()
m = mujoco.MjModel.from_xml_path(str(source.with_suffix(".xml")))
d = restore(m, source.with_suffix(".npz"))
r = root(c)
append_robot(r, cfg=c, base=c["robot"]["base"])
for asset in h.find("asset"):
    r.find("asset").append(copy.deepcopy(asset))
for body in h.findall("worldbody/body"):
    name = body.get("name")
    if not name.startswith("ingot"):
        continue
    b = copy.deepcopy(body)
    bid = m.body(name).id
    b.set("pos", " ".join(str(x) for x in d.xpos[bid] + [0.25, -0.9, 0]))
    b.set("quat", " ".join(str(x) for x in d.xquat[bid]))
    r.find("worldbody").append(b)
# Wide cell camera retains exact original geometry and settled poses.
cam = r.find("worldbody/camera[@name='overview']")
cam.set("pos", "3 -4 3")
cam.set("xyaxes", ".8 .6 0 -.3 .4 .8660254")
p = ROOT / "src/rusal_description/mjcf/cell_heap.xml"
p.write_text(xml(r))
compiled = mujoco.MjModel.from_xml_path(str(p))
print("Compiled full cell:", compiled.nbody, "bodies,", compiled.ngeom, "geoms")
