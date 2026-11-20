"""Dense, spaced and staggered batch layouts; reject dimensional overfill."""

import numpy as np
from scipy.spatial.transform import Rotation


def targets(cfg, layers=1, clearance=0, marking="visible_from_outside"):
    if marking not in ("off", "visible_from_outside"):
        raise ValueError(marking)
    i, p = cfg["ingot"], cfg["pallet"]
    a, b, h = [i[k] for k in ("length", "width", "thickness")]
    if p.get("layout") == "cross":
        # Coordinates/layers are measured and assigned online at each pickup.
        total = len(cfg.get("ingots", {})) or cfg.get("simulation", {}).get("count", 24)
        return [
            dict(
                layer=-1,
                row=0,
                col=0,
                position=[*p["center"], p["height"] + h / 2],
                quaternion_wxyz=[1, 0, 0, 0],
                unassigned=True,
            )
            for _ in range(total)
        ]
    if p.get("layout") == "brick":
        # Reserve the batch's actual largest envelope. Alternating 5/4 rows
        # overlap the joints below; Z is updated from physical support at pickup.
        items = list(cfg.get("ingots", {}).values()) or [i]
        width = max(v["width"] for v in items)
        length = max(v["length"] for v in items)
        height = max(v["thickness"] for v in items)
        count = p.get("brick_columns", 5)
        if count < 2:
            raise ValueError("Brick layout needs at least two columns")
        pitch = (p["length"] - 2 * p.get("edge_margin", 0.01) - width) / (count - 1)
        if pitch - width < p.get("minimum_gap", 0.025) or length > p["width"]:
            raise ValueError("Brick envelopes/jaw clearance do not fit pallet")
        result = []
        cx, cy = p["center"]
        cy += p.get("brick_y_offset", -0.17)
        if abs(cy - p["center"][1]) + length / 2 > p["width"] / 2:
            raise ValueError("Brick row extends beyond pallet")
        total = len(cfg.get("ingots", {})) or p.get("brick_count", 10)
        for layer in range(layers):
            n = count - (layer % 2)
            remaining = min(n, total - len(result))
            start = (n - remaining) // 2
            for col in range(start, start + remaining):
                if len(result) >= total:
                    return result
                yaw = -np.pi / 2
                q = Rotation.from_euler("z", yaw).as_quat()
                result.append(
                    dict(
                        layer=layer,
                        row=0,
                        col=col,
                        position=[
                            cx + (col - (n - 1) / 2) * pitch,
                            cy,
                            p["height"] + layer * height + height / 2,
                        ],
                        quaternion_wxyz=[q[3], *q[:3]],
                        marked_end_outward=True,
                    )
                )
        if len(result) < total:
            raise ValueError("Insufficient brick layers for this batch")
        return result
    if p.get("layout") == "spaced":
        # No later compaction is simulated: these are the actual seated targets.
        nx, ny = p.get("grid", [2, 3])
        gx, gy = p.get("gaps", [0.06, 0.06])
        ox, oy = p.get("offset", [0, 0])
        along_y = p.get("long_axis", "x") == "y"
        w, d = (b, a) if along_y else (a, b)
        if (
            layers < 1
            or min(nx, ny) < 1
            or min(gx, gy) < 0
            or nx * w + (nx - 1) * gx + 2 * abs(ox) > p["length"] + 1e-10
            or ny * d + (ny - 1) * gy + 2 * abs(oy) > p["width"] + 1e-10
        ):
            raise ValueError("Spaced layout does not fit the pallet")
        cx, cy = p["center"]
        result = []
        for layer in range(layers):
            for row in range(ny):
                for col in range(nx):
                    yaw = (
                        (-np.pi / 2 if row == 0 else np.pi / 2)
                        if along_y
                        else (np.pi if col == 0 else 0.0)
                    )
                    q = Rotation.from_euler("z", yaw).as_quat()
                    result.append(
                        dict(
                            layer=layer,
                            row=row,
                            col=col,
                            position=[
                                cx + ox + (col - (nx - 1) / 2) * (w + gx),
                                cy + oy + (row - (ny - 1) / 2) * (d + gy),
                                p["height"] + (layer + 0.5) * h,
                            ],
                            quaternion_wxyz=[q[3], *q[:3]],
                            marked_end_outward=True,
                        )
                    )
        return result
    if 6 * b + 5 * clearance > p["length"] + 1e-10 or 2 * a + clearance > p["width"] + 1e-10:
        raise ValueError("12 ingots do not fit: nominal packing has zero assembly clearance")
    if layers < 1:
        raise ValueError("layers must be positive")
    result = []
    cx, cy = p["center"]
    for layer in range(layers):
        for row in range(2):
            yaw = -np.pi / 2 if row == 0 else np.pi / 2
            q = Rotation.from_euler("z", yaw).as_quat()
            result.append(None) if False else None
            for col in range(6):
                result.append(
                    dict(
                        layer=layer,
                        row=row,
                        col=col,
                        position=[
                            cx + (col - 2.5) * (b + clearance),
                            cy + (row - 0.5) * (a + clearance),
                            p["height"] + (layer + 0.5) * h,
                        ],
                        quaternion_wxyz=[q[3], *q[:3]],
                        marked_end_outward=True,
                    )
                )
    return result


def target_rotation(target):
    q = target["quaternion_wxyz"]
    return Rotation.from_quat([*q[1:], q[0]]).as_matrix()
