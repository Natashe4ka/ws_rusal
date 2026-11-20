import copy
import numpy as np
import mujoco
import pytest
from rusal_sim import configuration
from rusal_sim.geometry import components, properties
from rusal_sim.scene import scene, xml, root, ingot
from rusal_sim.experiment import Experiment
from rusal_sim.pallet import targets
from rusal_sim.heap import generate, restore


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_geometry_mass_inertia(variant):
    c = configuration()
    c["ingot"]["variant"] = variant
    parts = components(c["ingot"])
    p = properties(c["ingot"])
    bounds = np.array([x.mesh.bounds for x in parts])
    assert bounds[:, :, 2].max() - bounds[:, :, 2].min() <= c["ingot"]["thickness"] + 1e-10
    assert all(x.mesh.is_convex and x.mesh.volume > 0 for x in parts)
    I = np.array(p["inertia"])
    assert np.min(np.linalg.eigvalsh(I)) > 0
    assert p["com"][2] <= 0 if variant != "A" else abs(p["com"][2]) < 0.001
    m = mujoco.MjModel.from_xml_string(xml(scene(c, fixture=False)))
    b = m.body("ingot").id
    assert m.body_mass[b] == pytest.approx(7.5)
    assert np.sort(m.body_inertia[b]) == pytest.approx(np.linalg.eigvalsh(I))
    cc = copy.deepcopy(c["ingot"])
    cc["mass"] = 15
    assert np.array(properties(cc)["inertia"]) == pytest.approx(I * 2)


def test_grooves_are_real_voids():
    c = configuration()["ingot"]
    ribs = [p for p in components(c) if p.name != "base"]
    assert len(ribs) > 20
    assert ribs[0].mesh.bounds[1, 0] < ribs[3].mesh.bounds[0, 0]
    assert ribs[0].mesh.bounds[1, 1] < ribs[1].mesh.bounds[0, 1]


def test_free_drop():
    c = configuration()
    r = root(c)
    ingot(r, c["ingot"], pos=(0, 0, 0.8))
    m = mujoco.MjModel.from_xml_string(xml(r))
    d = mujoco.MjData(m)
    for _ in range(2500):
        mujoco.mj_step(m, d)
    assert 0.018 < d.qpos[2] < 0.023
    assert np.linalg.norm(d.qvel) < 0.001
    assert not any(w.number for w in d.warning)


@pytest.mark.parametrize("degrees,sliding", [(10, False), (40, True)])
def test_inclined_plane(degrees, sliding):
    from scipy.spatial.transform import Rotation

    c = configuration()
    c["ingot"]["variant"] = "A"
    for key in ["aluminum_aluminum", "aluminum_floor"]:
        c["contacts"][key] = 0.3
    r = root(c)
    floor = r.find("worldbody/geom[@name='floor']")
    angle = np.deg2rad(degrees)
    floor.set("euler", f"0 {angle} 0")
    rot = Rotation.from_euler("y", angle)
    p = rot.apply([0, 0, 0.022])
    q = rot.as_quat()
    ingot(r, c["ingot"], pos=p, quat=[q[3], *q[:3]])
    m = mujoco.MjModel.from_xml_string(xml(r))
    d = mujoco.MjData(m)
    for _ in range(1500):
        mujoco.mj_step(m, d)
    displacement = abs(d.qpos[0] - p[0])
    assert displacement > 0.2 if sliding else displacement < 0.01


def test_real_hold_and_release():
    e = Experiment(configuration())
    result = e.run()
    assert result["success"]
    assert result["final_z"] < 0.025
    assert e.m.neq == 0 and e.m.nmocap == 0
    assert max(r["vacuum_capacity_n"] for r in e.rows) <= 3 * np.pi * 0.04**2 * 60000 * 0.85 + 1e-8
    assert not result["warnings"]


@pytest.mark.parametrize("setting", ["off", "insufficient", "unsealed", "ribbed"])
def test_expected_failure(setting):
    c = configuration()
    mode = "off" if setting == "off" else "vacuum"
    if setting == "insufficient":
        c["vacuum"]["pressure_pa"] = 2000
    e = Experiment(
        c,
        mode,
        "ribbed" if setting == "ribbed" else "smooth",
        leak=0 if setting == "unsealed" else 1,
    )
    r = e.run()
    assert not r["success"]
    assert r["final_z"] < 0.025


def test_mass_and_friction_capacity():
    # Independent analytical sanity checks, not a claim about contact simulation.
    c = configuration()
    mu = c["contacts"]["aluminum_steel"]
    n = c["jaws"]["force_per_jaw"]
    assert 2 * mu * n / (15 * 9.81) < 2 * mu * n / (5 * 9.81)
    assert 2 * 0.2 * n < 2 * 0.8 * n


def test_pallet_bounds_and_marking():
    c = configuration()
    ps = targets(c, 3)
    assert len(ps) == 36
    for p in ps:
        assert abs(p["position"][0] - 0.8) + 0.1 <= 0.6 + 1e-9
        assert abs(p["position"][1] - 0.6) + 0.2 <= 0.4 + 1e-9
        assert p["marked_end_outward"]
    with pytest.raises(ValueError):
        targets(c, clearance=0.001)


def test_heap_replay(tmp_path):
    path = tmp_path / "heap"
    c = configuration()
    report = generate(c, 5, 42, path)
    assert report["settled"]
    m = mujoco.MjModel.from_xml_path(str(path.with_suffix(".xml")))
    a = restore(m, path.with_suffix(".npz"))
    b = restore(m, path.with_suffix(".npz"))
    for _ in range(100):
        mujoco.mj_step(m, a)
        mujoco.mj_step(m, b)
    np.testing.assert_array_equal(a.qpos, b.qpos)


def test_official_kuka_fk():
    from rusal_sim.robot import fk, URDF, build
    import xml.etree.ElementTree as E

    m = mujoco.MjModel.from_xml_string(xml(build(configuration())[0]))
    d = mujoco.MjData(m)
    q = {f"joint_{i}": v for i, v in enumerate([0.2, -1.2, 0.7, 0.4, -0.5, 0.8], 1)}
    for name, v in q.items():
        d.qpos[m.joint(name).qposadr[0]] = v
    mujoco.mj_forward(m, d)
    T = fk(E.parse(URDF).getroot(), q)
    np.testing.assert_allclose(T[:3, 3], d.xpos[m.body("flange").id], atol=1e-9)
    rng = np.random.default_rng(42)
    for _ in range(100):
        q = {}
        for i in range(1, 7):
            joint = m.joint(f"joint_{i}")
            q[joint.name] = rng.uniform(*m.jnt_range[joint.id])
            d.qpos[joint.qposadr[0]] = q[joint.name]
        mujoco.mj_forward(m, d)
        T = fk(E.parse(URDF).getroot(), q)
        np.testing.assert_allclose(T[:3, 3], d.xpos[m.body("flange").id], atol=1e-9)
    expected = T[:3, 3] + T[:3, :3] @ np.array([0, 0, 0.225])
    np.testing.assert_allclose(expected, d.site_xpos[m.site("tcp").id], atol=1e-9)


def test_mechanical_contacts_lift_and_release():
    e = Experiment(configuration(), "mechanical", "ribbed")
    r = e.run()
    assert r["success"] and r["final_z"] < 0.025
    assert max(row["jaw_left_n"] for row in e.rows) > 100
    assert max(row["jaw_right_n"] for row in e.rows) > 100


def test_local_cycle_checks_support_and_neighbors(tmp_path):
    from rusal_sim.pick_place import PickPlace

    r = PickPlace(configuration()).run_cycle(tmp_path / "cycle")
    assert r["success"] and r["stable"] and r["orientation_error_deg"] < 3
    assert r["neighbor_displacement_m"] < 0.005
    states = [v["state"] for v in r["history"]]
    assert states.index("WAIT_SUPPORT") < states.index("RELEASE") < states.index("DONE")


def test_kuka_dynamic_transfer_with_limits(tmp_path):
    from rusal_sim.robot_trial import run

    r = run(out=tmp_path / "robot_trial")
    assert r["success"] and r["max_z"] > 0.4
    assert not r["unexpected_penetrations"] and not any(r["warnings"])


@pytest.mark.parametrize("thickness", [0.03, 0.05])
def test_lift_across_thickness_range(thickness):
    c = configuration()
    c["ingot"]["thickness"] = thickness
    assert Experiment(c).run()["success"]


def test_extract_from_deposited_heap_and_track_occupied_slot(tmp_path):
    from rusal_sim.heap_execution import HeapExecution
    from pathlib import Path

    h = HeapExecution(Path(__file__).parent / "fixtures/heap10")
    assert h.candidates()
    report = h.execute(1, tmp_path / "heap_cycle")
    a = report["attempts"][0]
    assert a["sealed"] and a["lifted"] and a["extraction_passed"]
    assert a["transport_passed"] and a["placement_passed"]
    assert a["physically_deposited"] and report["remaining"] == 9
    assert (
        h.slot_cursor == 1
    )  # A physically occupied slot stays reserved even if a quality criterion fails.
    b = h.m.body(a["selected"]).id
    assert h.d.xpos[b, 2] == pytest.approx(0.2, abs=0.005)
    assert h.m.neq == 0 and h.m.nmocap == 0
