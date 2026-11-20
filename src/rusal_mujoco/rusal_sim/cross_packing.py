"""Online fixed-orientation rectangle packing, rotating each successive layer."""

import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull


def intersects(a, b):
    return (
        a[0] < b[0] + b[2] - 1e-9
        and a[0] + a[2] > b[0] + 1e-9
        and a[1] < b[1] + b[3] - 1e-9
        and a[1] + a[3] > b[1] + 1e-9
    )


def free_rectangles(bounds, obstacles):
    free = [bounds]
    for obstacle in obstacles:
        next_free = []
        for r in free:
            if not intersects(r, obstacle):
                next_free.append(r)
                continue
            x, y, w, h = r
            ox, oy, ow, oh = obstacle
            if ox > x:
                next_free.append((x, y, ox - x, h))
            if ox + ow < x + w:
                next_free.append((ox + ow, y, x + w - ox - ow, h))
            if oy > y:
                next_free.append((x, y, w, oy - y))
            if oy + oh < y + h:
                next_free.append((x, oy + oh, w, y + h - oy - oh))
        free = []
        for r in next_free:
            if r[2] < 1e-8 or r[3] < 1e-8 or r in free:
                continue
            if any(
                r != s
                and r[0] >= s[0] - 1e-9
                and r[1] >= s[1] - 1e-9
                and r[0] + r[2] <= s[0] + s[2] + 1e-9
                and r[1] + r[3] <= s[1] + s[3] + 1e-9
                for s in next_free
            ):
                continue
            free.append(r)
    return free


def find_target(cfg, item, occupied, layers):
    p = cfg["pallet"]
    cx, cy = p["center"]
    margin = p.get("edge_margin", 0.002)
    gap = p.get("packing_gap", 0.002)
    bounds = (
        cx - p["length"] / 2 + margin,
        cy - p["width"] / 2 + margin,
        p["length"] - 2 * margin,
        p["width"] - 2 * margin,
    )
    items = list(cfg.get("ingots", {}).values()) or [item]
    for layer in range(layers):
        min_w = min(i["length" if layer % 2 == 0 else "width"] for i in items)
        min_h = min(i["width" if layer % 2 == 0 else "length"] for i in items)
        w, h = (
            (item["length"], item["width"]) if layer % 2 == 0 else (item["width"], item["length"])
        )
        blockers = [
            (o["x"] - gap, o["y"] - gap, o["w"] + 2 * gap, o["h"] + 2 * gap)
            for o in occupied
            if o["layer"] >= layer
        ]
        free = free_rectangles(bounds, blockers)
        best = None
        for x, y, rw, rh in free:
            if w > rw + 1e-9 or h > rh + 1e-9:
                continue
            xs = {x, x + rw - w}
            ys = {y, y + rh - h}
            if layer:
                for o in occupied:
                    if o["layer"] == layer - 1:
                        xs.update((o["x"], o["x"] + o["w"] - w))
                        ys.update((o["y"], o["y"] + o["h"] - h))
            for px, py in (
                (px, py)
                for px in sorted(xs)
                for py in sorted(ys)
                if px >= x - 1e-9
                and py >= y - 1e-9
                and px + w <= x + rw + 1e-9
                and py + h <= y + rh + 1e-9
            ):
                rect = (px, py, w, h)
                support = 1.0
                if layer:
                    supports = [
                        o
                        for o in occupied
                        if o["layer"] == layer - 1
                        and intersects(rect, (o["x"], o["y"], o["w"], o["h"]))
                    ]
                    support = min(
                        1.0,
                        sum(
                            max(0, min(px + w, o["x"] + o["w"]) - max(px, o["x"]))
                            * max(0, min(py + h, o["y"] + o["h"]) - max(py, o["y"]))
                            for o in supports
                        )
                        / (w * h),
                    )
                    if support < p.get("minimum_support_fraction", 0.55):
                        continue
                    if not supports:
                        continue
                    corners = []
                    for o in supports:
                        xa, xb = max(px, o["x"]), min(px + w, o["x"] + o["w"])
                        ya, yb = max(py, o["y"]), min(py + h, o["y"] + o["h"])
                        corners.extend([(xa, ya), (xa, yb), (xb, ya), (xb, yb)])
                    hull = ConvexHull(corners)
                    if np.any(
                        hull.equations[:, :2] @ np.array([px + w / 2, py + h / 2])
                        + hull.equations[:, 2]
                        > -p.get("support_center_margin", 0.003)
                    ):
                        continue
                # Do not fill a row with two long castings when doing so
                # strands a strip too small for any remaining casting. Another
                # row can receive this part while short pieces are still buried.
                dx, dy = rw - w, rh - h
                waste = (dx * h if 1e-8 < dx < min_w else 0.0) + (
                    dy * w if 1e-8 < dy < min_h else 0.0
                )
                score = (waste, py, px, -support, min(rw - w, rh - h), max(rw - w, rh - h))
                if best is None or score < best[0]:
                    best = (score, rect)
        if best:
            score, (x, y, w, h) = best
            px, py = x + w / 2, y + h / 2
            yaw = (
                (-np.pi / 2 if py <= cy else np.pi / 2)
                if layer % 2 == 1
                else (np.pi if px <= cx else 0.0)
            )
            q = Rotation.from_euler("z", yaw).as_quat()
            return dict(
                layer=layer,
                row=int(py > cy) if layer % 2 == 1 else int((py - bounds[1]) / 0.2),
                col=0,
                position=[px, py, p["height"] + item["thickness"] / 2],
                quaternion_wxyz=[q[3], *q[:3]],
                marked_end_outward=True,
                footprint=[x, y, w, h],
                support_fraction=-score[3],
                packing_score=[
                    layer,
                    *(
                        [item["length"] > 0.4, -item["width"]]
                        if layer % 2 == 1
                        else [item["width"], item["length"]]
                    ),
                    *score,
                ],
                assigned_dimensions_m=[item[k] for k in ("length", "width", "thickness")],
            )
    return None


def choose_target(cfg, available, remaining, occupied, layers, seed=42):
    """Bounded mock lookahead avoids consuming every small part in the first layer.

    Only the first pickup must currently be accessible. Future free-body poses
    remain unknown; the simulation replans against measured landings each cycle.
    """
    rng = np.random.default_rng(seed)
    initial = {name: find_target(cfg, remaining[name], occupied, layers) for name in available}
    valid = [name for name in available if initial[name] is not None]
    if not valid:
        return None
    lowest = min(initial[name]["layer"] for name in valid)
    valid = [name for name in valid if initial[name]["layer"] == lowest]
    best = None
    started = time.monotonic()
    for trial in range(cfg["pallet"].get("packing_lookahead_trials", 32)):
        name = (
            min(valid, key=lambda n: initial[n]["packing_score"])
            if trial == 0
            else valid[int(rng.integers(len(valid)))]
        )
        first = initial[name]
        pending = dict(remaining)
        del pending[name]
        rectangles = list(occupied)

        def append(target):
            x, y, w, h = target["footprint"]
            rectangles.append(dict(x=x, y=y, w=w, h=h, layer=target["layer"]))

        append(first)
        count = 1
        layer_sum = first["layer"]
        while pending:
            plans = [(find_target(cfg, item, rectangles, layers), n) for n, item in pending.items()]
            plans = [(p, n) for p, n in plans if p is not None]
            if not plans:
                break
            low = min(p["layer"] for p, n in plans)
            plans = [(p, n) for p, n in plans if p["layer"] == low]
            target, next_name = (
                min(plans, key=lambda v: v[0]["packing_score"])
                if trial == 0
                else plans[int(rng.integers(len(plans)))]
            )
            append(target)
            del pending[next_name]
            count += 1
            layer_sum += target["layer"]
        score = (-count, layer_sum, available.index(name))
        if best is None or score < best[0]:
            best = (score, name, first)
        if count == len(remaining) or time.monotonic() - started >= cfg["pallet"].get(
            "packing_lookahead_seconds", 12.0
        ):
            break
    _, name, target = best
    target["lookahead_fitted_remaining"] = -best[0][0]
    return name, target
