"""Pressure dynamics and geometric seal gating above bounded contact adhesion."""

import numpy as np
from .geometry import components
from .batch import ingot_config


class Vacuum:
    def __init__(self, m, cfg, target_name="ingot"):
        self.target = m.body(target_name).id
        self.ingot = ingot_config(cfg, target_name)
        self.cfg = cfg
        v = cfg["vacuum"]
        self.xs = (
            v.get("cell_centers", v["cup_x"] if v["layout"] == "V1" else [-0.09, 0.09])
            if v.get("enabled", True)
            else []
        )
        self.pressures = np.zeros(len(self.xs))
        self.quality = self.pressures.copy()
        self.ids = [m.actuator(f"vacuum_{i}").id for i in range(len(self.xs))]
        self.pads = [m.geom(f"cup_pad_{i}").id for i in range(len(self.xs))]
        self.area = (
            np.pi * (v["diameter"] / 2) ** 2 if v["layout"] == "V1" else np.prod(v["rectangle"])
        )
        self.capacity = np.zeros(len(self.xs))
        self.bounds = np.array(
            [
                c.mesh.bounds
                for c in components(self.ingot, self.ingot.get("geometry_seed", cfg["seed"]))
                if c.name != "base"
            ]
        )
        angles = np.linspace(0, 2 * np.pi, v["sample_count"], endpoint=False)
        if v["layout"] == "V1":
            self.offsets = (
                np.c_[np.cos(angles), np.sin(angles), np.zeros(len(angles))] * v["diameter"] / 2
            )
        else:
            # Uniformly sample all four edges of the rectangular seal perimeter.
            n = max(2, v["sample_count"] // 4)
            x, y = np.array(v["rectangle"]) / 2
            self.offsets = np.array(
                [[xx, yy, 0] for xx in np.linspace(-x, x, n) for yy in [-y, y]]
                + [[xx, yy, 0] for yy in np.linspace(-y, y, n) for xx in [-x, x]]
            )

    @property
    def mean_pressure(self):
        return float(np.mean(self.pressures)) if len(self.pressures) else 0.0

    @property
    def mean_quality(self):
        return float(np.mean(self.quality)) if len(self.quality) else 0.0

    def update(self, m, d, on=True, disabled=(), leak=1):
        v = self.cfg["vacuum"]
        ic = self.ingot
        b = self.target
        R = d.xmat[b].reshape(3, 3)
        p = d.xpos[b]
        contactpads = set()
        if on:
            for c in d.contact:
                if m.geom_bodyid[c.geom1] == b:
                    contactpads.add(c.geom2)
                if m.geom_bodyid[c.geom2] == b:
                    contactpads.add(c.geom1)
        for i, (aid, gid) in enumerate(zip(self.ids, self.pads)):
            eta = 0
            if on and gid in contactpads and i not in disabled:
                Rc = d.geom_xmat[gid].reshape(3, 3)
                normal = Rc[:, 2]
                alignment = abs(np.dot(R[:, 2], normal))
                smooth = np.dot(R[:, 2], normal) < 0
                if alignment >= np.cos(np.deg2rad(v["angle_tolerance_deg"])):
                    center = d.geom_xpos[gid] - normal * v["pad_half_thickness"]
                    local = (self.offsets @ Rc.T + center - p) @ R
                    ribheight = ic["rib_height"] if ic["variant"] != "A" else 0
                    taper = (
                        0
                        if smooth
                        else (ic["thickness"] - ribheight)
                        * np.tan(np.deg2rad(ic["taper_angle_deg"]))
                    )
                    valid = (np.abs(local[:, 0]) <= ic["length"] / 2 - taper) & (
                        np.abs(local[:, 1]) <= ic["width"] / 2 - taper
                    )
                    heights = np.full(len(local), ic["thickness"] / 2 - ribheight)
                    if not smooth and len(self.bounds):
                        hits = np.all(
                            (local[:, None, :2] >= self.bounds[None, :, 0, :2])
                            & (local[:, None, :2] <= self.bounds[None, :, 1, :2]),
                            axis=2,
                        )
                        heights = np.max(
                            np.where(hits, self.bounds[None, :, 1, 2], heights[:, None]), axis=1
                        )
                    if smooth:
                        heights[:] = -ic["thickness"] / 2
                    rim_contact = np.mean(
                        valid & (np.abs(local[:, 2] - heights) <= v["seal_height_tolerance"])
                    )
                    relief = 0 if smooth else np.ptp(heights[valid]) if np.any(valid) else np.inf
                    eff = (
                        v["smooth_efficiency"]
                        if relief <= v["seal_height_tolerance"] and rim_contact >= 0.95
                        else v["ribbed_efficiency"]
                    )
                    eta = eff * np.mean(valid) * (1 - v["contamination"]) * leak
            self.quality[i] = np.clip(eta, 0, 1)
            target = v["pressure_pa"] if on and eta > 0 else 0
            tau = v["build_tau"] if target > self.pressures[i] else v["loss_tau"]
            self.pressures[i] += (target - self.pressures[i]) * (-np.expm1(-m.opt.timestep / tau))
            self.capacity[i] = self.pressures[i] * self.area * self.quality[i] if on else 0
            d.ctrl[aid] = self.capacity[i]
