"""Twenty-ingot scenario presets and physical pallet capacity."""

import numpy as np
from rusal_sim import ROOT, configuration
from rusal_sim.pallet import targets, target_rotation


def test_twenty_ingots_have_twenty_in_bounds_slots_without_dimensional_overfill():
    cfg = configuration(ROOT / "configs/hybrid_heap20.yaml")
    assert cfg["simulation"]["count"] == 20
    slots = targets(cfg, layers=cfg["simulation"]["layers"])
    assert len(slots) == 20
    assert [s["layer"] for s in slots] == sum(([i] * 5 for i in range(4)), [])
    corners = np.array([[x, y, 0] for x in [-0.2, 0.2] for y in [-0.1, 0.1]])
    for s in slots:
        xy = (corners @ target_rotation(s).T + np.asarray(s["position"]))[:, :2]
        assert np.all(np.abs(xy - cfg["pallet"]["center"]) <= np.array([1.2, 0.8]) / 2 + 1e-10)
    for layer in range(1, 4):
        np.testing.assert_allclose(
            np.array([s["position"] for s in slots[layer * 5 : (layer + 1) * 5]])
            - np.array([s["position"] for s in slots[:5]]),
            np.tile([0, 0, layer * 0.04], (5, 1)),
            atol=1e-12,
        )
