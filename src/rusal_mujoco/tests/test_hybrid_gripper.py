"""Actual vacuum-to-jaw load handoff, including vacuum-off physical holding."""

from rusal_sim import ROOT, configuration
from rusal_sim.heap import generate_compact
from rusal_sim.cell import PhysicalCell


def test_vacuum_pick_then_jaws_hold_with_adhesion_disabled(tmp_path):
    cfg = configuration(ROOT / "configs/hybrid_foam_jaws.yaml")
    source = tmp_path / "heap"
    generate_compact(cfg, 1, 42, source, surface="smooth")
    cell = PhysicalCell(source, mode="hybrid", cfg=cfg)
    r = cell.run(1, tmp_path / "hybrid", gui=False)
    a = r["attempts"][0]
    assert a.get("sealed") and a.get("lifted"), a
    assert a.get("clamped_after_lift") and a.get("vacuum_off_hold_passed"), a
    assert a["vacuum_off_hold_drift_m"] < 0.01
    assert a.get("physically_deposited") and r["robot_limits_pass"], r
    phases = list(dict.fromkeys(row["phase"] for row in cell.rows))
    assert (
        phases.index("EXTRACT")
        < phases.index("DEPLOY_OPEN_JAWS")
        < phases.index("DEPLOY_JAWS")
        < phases.index("VERIFY_JAWS_VACUUM_OFF")
        < phases.index("TRANSPORT")
    )
    transport = [row for row in cell.rows if row["phase"] == "TRANSPORT"]
    assert transport and all(row["vacuum_capacity_n"] == 0 for row in transport)
    assert cell.m.neq == 0 and cell.m.nmocap == 0
