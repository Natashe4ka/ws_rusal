"""Individual inertias, seal geometry and pallet envelopes survive serialization."""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from rusal_sim import ROOT, configuration
from rusal_sim.batch import make_batch
from rusal_sim.geometry import properties
from rusal_sim.pallet import targets, target_rotation
from rusal_sim.scene import root, ingot, xml
from rusal_sim.heap import save
from rusal_sim.cell import PhysicalCell
from rusal_sim.vacuum import Vacuum


def test_variable_geometry_mass_and_seal_use_same_individual_config(tmp_path):
    cfg = configuration(ROOT / "configs/hybrid_batch10.yaml")
    make_batch(cfg, 10, 42)
    again = configuration(ROOT / "configs/hybrid_batch10.yaml")
    make_batch(again, 10, 42)
    assert cfg["ingots"] == again["ingots"]
    tree = root(cfg)
    for index, (name, item) in enumerate(cfg["ingots"].items()):
        ingot(tree, item, name=name, pos=[index * 0.5, 0, 0.3], seed=item["geometry_seed"])
    model = mujoco.MjModel.from_xml_string(xml(tree))
    data = mujoco.MjData(model)
    source = tmp_path / "individuals"
    source.with_suffix(".xml").write_text(xml(tree))
    save(model, data, source.with_suffix(".npz"), cfg)
    cell = PhysicalCell(source, layers=3, cfg=cfg)
    for name, item in cfg["ingots"].items():
        b = cell.m.body(name).id
        assert 12 <= cell.m.body_mass[b] <= 14
        np.testing.assert_allclose(cell.m.body_mass[b], item["mass"])
        props = properties(item, item["geometry_seed"])
        np.testing.assert_allclose(cell.m.body_ipos[b], props["com"])
        q = cell.m.body_iquat[b]
        rotation = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
        np.testing.assert_allclose(
            rotation @ np.diag(cell.m.body_inertia[b]) @ rotation.T, props["inertia"], atol=1e-12
        )
        vacuum = Vacuum(cell.m, cell.cfg, name)
        assert vacuum.ingot == item
        if item["variant"] != "A":
            assert vacuum.bounds[:, :, 2].max() <= item["thickness"] / 2 + 1e-10
    assert cell.cfg["ingots"] == cfg["ingots"]
    for side in ("left", "right"):
        aid = cell.m.actuator("deploy_motor_" + side).id
        np.testing.assert_allclose(cell.m.actuator_forcerange[aid], [-300, 300])
        assert cell.m.actuator_gainprm[aid, 0] == 50000
        assert cell._deploy_setpoint(True, side) == 0.08
    assert cell.m.neq == 0 and cell.m.nmocap == 0


def test_brick_envelopes_are_staggered_with_jaw_gaps_and_no_overhang():
    cfg = configuration(ROOT / "configs/hybrid_batch10.yaml")
    make_batch(cfg, 10, 42)
    slots = targets(cfg, 3)
    assert [sum(s["layer"] == layer for s in slots) for layer in range(3)] == [5, 4, 1]
    for slot in slots:
        for item in cfg["ingots"].values():
            a, b = item["length"] / 2, item["width"] / 2
            corners = np.array([[x, y, 0] for x in [-a, a] for y in [-b, b]])
            xy = (corners @ target_rotation(slot).T + slot["position"])[:, :2]
            assert np.all(np.abs(xy - cfg["pallet"]["center"]) <= np.array([1.2, 0.8]) / 2 + 1e-10)
    lower = [s["position"][0] for s in slots if s["layer"] == 0]
    upper = [s["position"][0] for s in slots if s["layer"] == 1]
    np.testing.assert_allclose(upper, (np.array(lower[:-1]) + lower[1:]) / 2)


def test_narrow_heavy_ingot_is_carried_with_bounded_force_control(tmp_path):
    from rusal_sim.heap import generate_compact

    cfg = configuration(ROOT / "configs/hybrid_batch10.yaml")
    cfg.pop("ingot_variation")
    cfg["ingot"].update(length=0.41604, width=0.19118, thickness=0.030697, mass=13.59)
    cfg["pallet"]["brick_count"] = 1
    source = tmp_path / "narrow_heavy"
    generate_compact(cfg, 1, 42, source, surface="ribbed")
    cell = PhysicalCell(source, layers=3, cfg=cfg)
    report = cell.run(12, tmp_path / "narrow_result", gui=False)
    assert report["complete"], report
    assert not any(a.get("dropped") for a in report["attempts"]), report["attempts"]
    assert any(a.get("transport_passed") for a in report["attempts"])
    for side in ("left", "right"):
        aid = cell.m.actuator("jaw_motor_" + side).id
        forces = [
            abs(row[f"actuator_force_{aid}"]) for row in cell.rows if row["phase"] == "TRANSPORT"
        ]
        assert forces and max(forces) <= 1000.000001
        assert np.mean(forces) > 950
