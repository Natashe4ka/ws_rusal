"""Catalog stroke/force conventions and mutually exclusive physical tool models."""

import mujoco
import numpy as np
from rusal_sim import ROOT, configuration
from rusal_sim.scene import scene, xml
from rusal_sim.vacuum import Vacuum
from rusal_sim.pallet import targets, target_rotation


def test_both_profiles_use_identical_spaced_targets_inside_pallet():
    configs = [configuration(ROOT / f"configs/{p}.yaml") for p in ("piab_kvg150", "schunk_pgn200")]
    positions = []
    for c in configs:
        slots = targets(c)
        assert len(slots) == 5
        xy = np.array([s["position"][:2] for s in slots])
        np.testing.assert_allclose(np.diff(xy[:, 0]) - 0.2, 0.04)
        corners = np.array([[x, y, 0] for x in [-0.2, 0.2] for y in [-0.1, 0.1]])
        for s in slots:
            world = corners @ target_rotation(s).T + np.array(s["position"])
            assert np.all(
                np.abs(world[:, :2] - c["pallet"]["center"]) <= np.array([1.2, 0.8]) / 2 + 1e-10
            )
        positions.append(xy)
    np.testing.assert_array_equal(*positions)


def test_schunk_catalog_stroke_and_total_force_are_not_doubled():
    c = configuration(ROOT / "configs/schunk_pgn200.yaml")
    m = mujoco.MjModel.from_xml_string(xml(scene(c)))
    assert m.neq == m.nmocap == 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "deploy_left") == -1
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_pad_0") == -1
    for side in ("left", "right"):
        np.testing.assert_allclose(m.jnt_range[m.joint("jaw_" + side).id], [0, 0.025])
        np.testing.assert_allclose(
            m.actuator_ctrlrange[m.actuator("jaw_motor_" + side).id], [-1000, 1000]
        )
    assert sum(m.body_mass[m.body("tool").id :]) == 8.5
    assert 2 * c["jaws"]["force_per_jaw"] < 3800
    vac = Vacuum(m, c)
    vac.update(m, mujoco.MjData(m), True)
    assert vac.mean_pressure == 0 and vac.mean_quality == 0
    assert len(vac.ids) == 0


def test_piabs_segmented_foam_preserves_area_and_cannot_clamp():
    c = configuration(ROOT / "configs/piab_kvg150.yaml")
    m = mujoco.MjModel.from_xml_string(xml(scene(c)))
    assert m.neq == m.nmocap == 0
    vac = Vacuum(m, c)
    assert len(vac.ids) == 15
    assert np.isclose(vac.area * len(vac.ids), 0.3 * 0.15)
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jaw_left") == -1
    d = mujoco.MjData(m)
    vac.pressures[:] = c["vacuum"]["pressure_pa"]
    vac.update(m, d, False)
    assert not np.any(vac.capacity) and not np.any(d.ctrl[vac.ids])
    # Grooves have not silently been declared sealable to improve this demo.
    assert c["vacuum"]["seal_height_tolerance"] == 0.0015
    assert c["vacuum"]["ribbed_efficiency"] == 0.015
