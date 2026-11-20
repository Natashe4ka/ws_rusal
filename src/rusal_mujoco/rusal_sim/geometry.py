"""Convex tapered casting plus physically separate ribs; no collision hull over grooves."""

from dataclasses import dataclass
import numpy as np
import trimesh


@dataclass
class Component:
    name: str
    mesh: trimesh.Trimesh


def components(cfg, seed=42):
    a, b, h = (cfg[k] for k in ("length", "width", "thickness"))
    if min(a, b, h, cfg["mass"]) <= 0 or not 0 <= cfg["rib_height"] < h:
        raise ValueError("Invalid ingot dimensions/mass")
    ribbed = cfg["variant"] != "A"
    r = cfg["rib_height"] if ribbed else 0
    z0, z1 = -h / 2, h / 2 - r
    t = (z1 - z0) * np.tan(np.deg2rad(cfg["taper_angle_deg"]))
    vertices = [
        [sx * (a / 2 - (t if z == z1 else 0)), sy * (b / 2 - (t if z == z1 else 0)), z]
        for z in (z0, z1)
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    base = trimesh.convex.convex_hull(np.array(vertices))
    result = [Component("base", base)]
    rng = np.random.default_rng(seed)
    if ribbed:
        n = int((a - 2 * t) // cfg["rib_pitch"])
        for i in range(n):
            rh = r * (rng.uniform(0.7, 1) if cfg["variant"] == "C" else 1)
            x = (i - (n - 1) / 2) * cfg["rib_pitch"]
            rows = cfg.get("rib_rows", 3)
            gap = cfg.get("rib_row_gap", 0.012)
            rw = (b - 2 * t - (rows - 1) * gap) / rows
            for row in range(rows):
                mesh = trimesh.creation.box([cfg["rib_width"], rw, rh])
                mesh.apply_translation([x, (row - (rows - 1) / 2) * (rw + gap), z1 + rh / 2])
                result.append(Component(f"rib_{i}_{row}", mesh))
    # Variant C has bounded rib height and taper, not unbounded decorative noise.
    return result


def properties(cfg, seed=42):
    parts = components(cfg, seed)
    volumes = np.array([c.mesh.volume for c in parts])
    volume = float(volumes.sum())
    centers = np.array([c.mesh.center_mass for c in parts])
    com = np.average(centers, axis=0, weights=volumes)
    rho = cfg["mass"] / volume
    tensor = np.zeros((3, 3))
    for c, v, p in zip(parts, volumes, centers):
        d = p - com
        tensor += rho * c.mesh.moment_inertia + rho * v * (
            np.dot(d, d) * np.eye(3) - np.outer(d, d)
        )
    a, b, h = (cfg[k] for k in ("length", "width", "thickness"))
    box = cfg["mass"] / 12 * np.array([b * b + h * h, a * a + h * h, a * a + b * b])
    return dict(
        volume=volume,
        geometric_mass=volume * cfg["density"],
        mass=cfg["mass"],
        effective_density=rho,
        unusual_density=abs(rho / cfg["density"] - 1) > 0.15,
        com=com.tolist(),
        inertia=tensor.tolist(),
        box_inertia=box.tolist(),
    )


def surface_height(cfg, x, y):
    """Local +Z surface, including open grooves. Outside footprint is NaN."""
    a, b, h = (cfg[k] for k in ("length", "width", "thickness"))
    r = cfg["rib_height"] if cfg["variant"] != "A" else 0
    t = (h - r) * np.tan(np.deg2rad(cfg["taper_angle_deg"]))
    if abs(x) > a / 2 - t or abs(y) > b / 2 - t:
        return np.nan
    z = h / 2 - r
    for c in components(cfg):
        if (
            c.name != "base"
            and c.mesh.bounds[0, 0] <= x <= c.mesh.bounds[1, 0]
            and c.mesh.bounds[0, 1] <= y <= c.mesh.bounds[1, 1]
        ):
            z = max(z, c.mesh.bounds[1, 2])
    return z
