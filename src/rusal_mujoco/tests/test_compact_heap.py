"""Compact physical deposition and geometry-dependent vacuum seal checks."""

import mujoco
import numpy as np
from rusal_sim import configuration
from rusal_sim.heap import generate_compact, restore
from rusal_sim.experiment import Experiment


def test_five_ingots_form_one_stable_contact_connected_heap(tmp_path):
    path = tmp_path / "compact"
    report = generate_compact(configuration(), 5, 42, path)
    assert report["settled"] and report["validated_equilibrium"]
    assert report["contact_graph_connected"]
    assert 0.6 < max(report["footprint_m"]) < 0.8
    assert report["floor_supported_count"] >= 3
    assert 0.12 < report["height"] < 0.3
    assert report["maximum_window_drift_m"] < 0.005
    assert not any(report["warnings"])
    m = mujoco.MjModel.from_xml_path(str(path.with_suffix(".xml")))
    d = restore(m, path.with_suffix(".npz"))
    assert m.neq == 0 and m.nmocap == 0 and m.njnt == 5
    assert np.all(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    assert m.body_mass[1:].tolist() == [7.5] * 5
    assert d.ncon > 0


def test_ribbed_face_is_not_categorically_banned_for_vacuum(tmp_path):
    # Synthetic wide flat rib crowns, NOT a measured profile of the user's casting.
    # Each 80 mm rim fits on a 90 mm land. Efficiencies and suction forces are unchanged.
    c = configuration()
    c["ingot"].update(rib_width=0.09, rib_pitch=0.12, rib_rows=1)
    experiment = Experiment(c, "vacuum", "ribbed")
    result = experiment.run(tmp_path / "wide_rib_crowns")
    assert result["success"], result
    assert max(row["vacuum_capacity_n"] for row in experiment.rows) > 700
    assert experiment.m.neq == 0


def test_hybrid_attempts_real_vacuum_on_sealable_rib_crowns(tmp_path):
    from rusal_sim.cell import PhysicalCell

    c = configuration()
    c["ingot"].update(rib_width=0.09, rib_pitch=0.12, rib_rows=1)
    source = tmp_path / "wide_crowns"
    generate_compact(c, 1, 42, source)
    cell = PhysicalCell(source, mode="hybrid", cfg=c)
    report = cell.run(1, tmp_path / "robot_wide_crowns", gui=False)
    attempt = report["attempts"][0]
    assert attempt["surface"] == "ribbed"
    assert attempt["sealed"] and attempt["lifted"], report
    phases = list(dict.fromkeys(row["phase"] for row in cell.rows))
    assert "MECHANICAL_FALLBACK" not in phases
    assert phases.index("EXTRACT") < phases.index("DEPLOY_OPEN_JAWS")
