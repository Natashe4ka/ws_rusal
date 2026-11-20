"""Narrow fingers entering two real rib grooves. No lower hooks or attachment."""

import copy
import mujoco
import numpy as np
from .experiment import Experiment
from .scene import xml, s
from .vacuum import Vacuum


def experiment(cfg, force=0):
    cfg = copy.deepcopy(cfg)
    cfg["jaws"]["force_per_jaw"] = force
    e = Experiment(cfg, "mechanical", "ribbed")
    for side, sign in [("left", 1), ("right", -1)]:
        dep = e.tree.find(f"worldbody/body[@name='tool']/body[@name='deploy_{side}']")
        dep.set(
            "pos",
            s(
                [
                    sign * cfg["rib_fingers"]["groove_center_x"],
                    0,
                    cfg["rib_fingers"]["retracted_height"],
                ]
            ),
        )
        j = dep.find(f"body/joint[@name='jaw_{side}']")
        j.set("axis", s([-sign, 0, 0]))
        g = dep.find(f"body/geom[@name='jaw_pad_{side}']")
        g.set("size", s(cfg["rib_fingers"]["half_sizes"]))
        # XML cannot specify a degenerate control interval; zero force is clamped in controller.
        a = e.tree.find(f"actuator/motor[@name='jaw_motor_{side}']")
        a.set("ctrlrange", s([-max(force, 0.001), max(force, 0.001)]))
    # This is a separate finger cartridge: remove vacuum pads, which otherwise
    # physically obstruct the same grooves. No contact exclusions hide that interference.
    head = e.tree.find("worldbody/body[@name='tool']")
    for cup in list(head.findall("body")):
        if cup.get("name", "").startswith("cup_"):
            head.remove(cup)
    actuators = e.tree.find("actuator")
    for actuator in list(actuators):
        if actuator.tag == "adhesion":
            actuators.remove(actuator)
    e.m = mujoco.MjModel.from_xml_string(xml(e.tree))
    e.d = mujoco.MjData(e.m)
    e.vacuum = InactiveVacuum()
    return e


class InactiveVacuum:
    def __init__(self):
        self.pressures = np.zeros(3)
        self.quality = np.zeros(3)
        self.capacity = np.zeros(3)

    def update(self, *args, **kwargs):
        pass
