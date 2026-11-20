"""Packing bounds, true crosswise layers, and recovery placement decisions."""

from collections import Counter
import numpy as np
from rusal_sim import ROOT, configuration
from rusal_sim.batch import make_batch
from rusal_sim.cross_packing import find_target, intersects, free_rectangles
from rusal_sim.pallet import target_rotation


def test_maxrects_never_offers_space_intersecting_obstacles():
    bounds = (0.0, 0.0, 1.2, 0.8)
    obstacles = [(0.1, 0.1, 0.3, 0.2), (0.5, 0.0, 0.2, 0.4), (0.0, 0.6, 0.8, 0.2)]
    free = free_rectangles(bounds, obstacles)
    assert free
    for rect in free:
        assert not any(intersects(rect, o) for o in obstacles)
        x, y, w, h = rect
        assert x >= 0 and y >= 0 and x + w <= 1.2 + 1e-9 and y + h <= 0.8 + 1e-9


def test_random_batch_packing_preserves_dimensions_and_uses_cross_layers():
    cfg = configuration(ROOT / "configs/hybrid_cross24.yaml")
    cfg["pallet"]["center"] = [0.0, 0.0]
    make_batch(cfg, 24, 42)
    assert sum(i["length"] * i["width"] for i in cfg["ingots"].values()) > 2 * 1.2 * 0.8
    left = dict(cfg["ingots"])
    occupied = []
    targets = []
    while left:
        choices = [(find_target(cfg, item, occupied, 3), name) for name, item in left.items()]
        choices = [(target, name) for target, name in choices if target is not None]
        if not choices:
            break
        target, name = min(choices, key=lambda v: v[0]["packing_score"])
        item = left.pop(name)
        x, y, w, h = target["footprint"]
        assert x >= -0.6 and y >= -0.4 and x + w <= 0.6 + 1e-9 and y + h <= 0.4 + 1e-9
        assert not any(
            intersects((x, y, w, h), (o["x"], o["y"], o["w"], o["h"]))
            for o in occupied
            if o["layer"] == target["layer"]
        )
        extent = np.abs(target_rotation(target)) @ np.array(
            [item["length"], item["width"], item["thickness"]]
        )
        np.testing.assert_allclose(extent[:2], [w, h], atol=1e-12)
        if target["layer"]:
            assert target["support_fraction"] >= cfg["pallet"]["minimum_support_fraction"]
        occupied.append(dict(x=x, y=y, w=w, h=h, layer=target["layer"]))
        targets.append(target)
    # Greedy alone need not fit every piece; the runtime adds bounded lookahead.
    counts = Counter(t["layer"] for t in targets)
    assert sum(counts.values()) >= 22 and set(counts) == {0, 1, 2}
    assert counts[0] >= 8
    assert max(t["position"][1] for t in targets if t["layer"] == 0) > 0
    first = [
        target_rotation(next(t for t in targets if t["layer"] == layer))[:, 0] for layer in range(3)
    ]
    assert abs(np.dot(first[0], first[1])) < 1e-9 and abs(np.dot(first[1], first[2])) < 1e-9


def test_upper_layer_rejects_support_whose_hull_misses_center():
    cfg = configuration(ROOT / "configs/hybrid_cross24.yaml")
    cfg["pallet"]["center"] = [0.0, 0.0]
    item = dict(cfg["ingot"], length=0.4, width=0.2)
    # Nearly the entire lower footprint is unavailable. A narrow edge cannot
    # support a freely resting upper piece whose center lies outside its hull.
    occupied = [dict(x=-0.6, y=-0.4, w=1.2, h=0.02, layer=0)]
    target = find_target(cfg, item, occupied, 2)
    assert target is not None and target["layer"] == 0
    occupied = [dict(x=-0.6, y=-0.4, w=1.2, h=0.8, layer=1)]
    assert find_target(cfg, item, occupied, 2) is None
