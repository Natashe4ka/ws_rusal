"""Physical end-to-end checks for the mocked-controller KUKA cell."""

import numpy as np
import mujoco
import pytest
from rusal_sim.cell import PhysicalCell
from rusal_sim.demo_source import generate_stack


def test_robot_cell_actually_picks_and_deposits(tmp_path):
    source = tmp_path / "source"
    assert generate_stack(1, source)["validated_equilibrium"]
    cell = PhysicalCell(source, mode="hybrid")
    assert cell.m.neq == 0
    assert cell.m.nmocap == 0
    assert all(not cell.m.joint(i).name.startswith("fixture_") for i in range(cell.m.njnt))
    assert cell.m.jnt_type[cell.m.body_jntadr[cell.bodies[0]]] == mujoco.mjtJoint.mjJNT_FREE
    report = cell.run(1, tmp_path / "result", gui=False)
    assert report["complete"], report
    assert not report["robot_collisions"]
    assert not any(report["warnings"])
    assert report["max_effort_ratio"] <= 1.000001
    assert report["max_speed_ratio"] <= 1
    attempt = report["attempts"][0]
    assert attempt["lifted"] and attempt["transport_passed"] and attempt["supported"]
    assert attempt["placement_passed"], attempt
    assert max(row["load_z"] for row in cell.rows) > 0.4
    assert np.linalg.norm(cell.d.xpos[cell.bodies[0]] - [0.05, 0.4, 0.2]) < 0.005
    resumed = PhysicalCell.resume(tmp_path / "result")
    assert np.array_equal(resumed.d.qpos, cell.d.qpos)
    assert np.array_equal(resumed.d.qvel, cell.d.qvel)
    assert resumed.d.time == cell.d.time
    assert resumed.removed == cell.removed


def test_ribbed_vacuum_failure_is_not_reported_as_pick(tmp_path):
    source = tmp_path / "ribbed"
    generate_stack(1, source, surface="ribbed")
    cell = PhysicalCell(source, mode="vacuum")
    report = cell.run(1, tmp_path / "failed", gui=False)
    assert not report["complete"]
    assert report["physically_deposited"] == 0
    assert not report["attempts"][0]["sealed"]
    assert cell.d.xpos[cell.bodies[0], 2] < 0.03
    assert not any(report["warnings"])


def test_joint_fallback_routes_around_obstacle():
    from rusal_sim.mock_planning import joint_route

    def blocked(q):
        return abs(q[0]) < 0.2 and q[1] < 0.5

    route = joint_route(
        np.array([-1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        blocked,
        np.random.default_rng(42),
    )
    assert route is not None
    for a, b in zip(route, route[1:]):
        assert all(not blocked(q) for q in np.linspace(a, b, 500))


def test_jaw_deployment_rejects_neighbor_without_moving_world(tmp_path):
    source = tmp_path / "neighbor"
    generate_stack(2, source)
    cell = PhysicalCell(source)
    cell.selected = cell.m.body("ingot_0").id
    neighbor = cell.m.body("ingot_1").id
    R = cell.d.xmat[cell.selected].reshape(3, 3) @ np.diag([1, -1, -1])
    contact = cell.d.xpos[cell.selected] + R[:, 2] * 0.044
    fp, fr = cell.flange_target(contact, R)
    q = cell.k.ik(fp, fr, start=cell.q_command)
    assert q is not None
    cell.d.qpos[cell.k.qa] = q  # Set up a stationary initial pre-grasp scene.
    adr = cell.m.jnt_qposadr[cell.m.body_jntadr[neighbor]]
    cell.d.qpos[adr : adr + 3] = cell.d.xpos[cell.selected] + R @ np.array([0, 0.24, 0])
    mujoco.mj_forward(cell.m, cell.d)
    before = cell.d.qpos.copy()
    assert not cell.jaw_clearance()
    assert np.array_equal(cell.d.qpos, before)
    cell.d.qpos[adr : adr + 3] = [3, 3, 0.02]
    mujoco.mj_forward(cell.m, cell.d)
    assert cell.jaw_clearance()


@pytest.mark.parametrize("failure_phase,loss_after", [("TRANSPORT", 3.8), ("PLACE", 0.4)])
@pytest.mark.parametrize("misplaced", [False, True])
def test_dropped_load_is_accepted_only_when_well_placed(
    tmp_path, monkeypatch, failure_phase, loss_after, misplaced
):
    from rusal_sim.vacuum import Vacuum

    source = tmp_path / "loss"
    generate_stack(1, source)
    from rusal_sim import configuration

    cfg = configuration()
    cfg["pallet"].update(layout="spaced", grid=[1, 1], long_axis="y", gaps=[0, 0], offset=[0, 0])
    cell = PhysicalCell(source, mode="vacuum", cfg=cfg)
    original = Vacuum.update
    fault = {"active": False, "time": None}

    def update(vacuum, model, data, on=True, *args, **kwargs):
        if (
            len(cell.attempts) == 1
            and cell.phase == failure_phase
            and data.time - cell.plan_start > loss_after
        ):
            fault["active"] = True
            if fault["time"] is None:
                fault["time"] = data.time
        # A bounded lateral disturbance produces a genuinely misplaced landing.
        data.xfrc_applied[vacuum.target, 0] = (
            80.0
            if misplaced
            and len(cell.attempts) == 1
            and fault["time"] is not None
            and data.time - fault["time"] < 0.05
            else 0.0
        )
        return original(
            vacuum,
            model,
            data,
            on and (len(cell.attempts) > 1 or not fault["active"]),
            *args,
            **kwargs,
        )

    monkeypatch.setattr(Vacuum, "update", update)
    report = cell.run(1, tmp_path / "loss_result", gui=False)
    assert fault["active"]
    attempt = report["attempts"][0]
    assert attempt["grasp_loss_phase"] == failure_phase
    if failure_phase == "TRANSPORT":
        assert not attempt["transport_passed"]
    assert attempt["failure"] == (
        "load lost in transport"
        if failure_phase == "TRANSPORT"
        else "load lost before commanded release"
    )
    if not misplaced:
        assert attempt["accepted_after_drop"], attempt
        assert attempt["physically_deposited"] and not attempt["retry_required"]
        assert report["transferred"] == 1 and report["complete"]
        assert cell.slot_cursor == 1 and cell.bodies[0] in cell.removed
        assert report["quality_passes"] == 0  # Correct landing does not erase grasp failure.
        return
    assert not attempt["physically_deposited"], attempt
    assert not attempt["accepted_after_drop"]
    assert attempt["on_pallet_after_drop"] and attempt["dropped"]
    assert attempt["retry_required"] and attempt["drop_settled"]
    assert report["quality_passes"] == 0
    assert cell.slot_cursor == 0
    assert cell.bodies[0] not in cell.removed
    assert any(v[1] == cell.bodies[0] for v in cell.candidates())
    assert cell.supported(cell.bodies[0])
    assert cell.d.xpos[cell.bodies[0], 2] < 0.23
    retried = cell.run(2, tmp_path / "regrasped", gui=False)
    assert retried["complete"], retried
    assert retried["transferred"] == 1 and cell.slot_cursor == 1
    assert retried["attempts"][1]["physically_deposited"]
    assert not retried["attempts"][1].get("dropped", False)
    assert retried["attempts"][1]["regrasp_after_drop"] == 1


def test_velocity_limit_stops_physics_without_clamping_state(tmp_path):
    import pytest
    from rusal_sim.cell import MotionFailure

    source = tmp_path / "overspeed"
    generate_stack(1, source)
    cell = PhysicalCell(source)
    cell.d.qvel[cell.k.va[5]] = 1.1 * cell.speed_limits[5]
    before = cell.d.qvel.copy()
    with pytest.raises(MotionFailure, match="velocity limit exceeded"):
        cell._after_step()
    assert np.array_equal(cell.d.qvel, before)


def test_dense_joint_edges_reject_a_thin_obstacle_between_waypoints():
    from rusal_sim.mock_planning import joint_route

    # In one dimension this obstacle cannot be bypassed. Coarse endpoint
    # sampling would miss the small forbidden interval at +0.013 rad.
    blocked = lambda q: abs(q[0] - 0.013) < 0.003
    route = joint_route(
        np.array([-0.1]),
        np.array([0.1]),
        np.array([[-0.2, 0.2]]),
        blocked,
        np.random.default_rng(42),
        iterations=40,
        edge_resolution=0.001,
    )
    assert route is None


def test_resume_empty_approach_checks_physical_vacuum_controls(tmp_path):
    source = tmp_path / "approach_source"
    generate_stack(1, source)
    cell = PhysicalCell(source)
    cell.phase = "APPROACH"
    out = tmp_path / "approach_checkpoint"
    cell.write_result(out)
    resumed = PhysicalCell.resume(out)
    np.testing.assert_array_equal(resumed.d.qpos, cell.d.qpos)
    np.testing.assert_array_equal(resumed.d.qvel, cell.d.qvel)
    # Merely labelling a state as an approach must never authorize resuming
    # with an active vacuum attachment.
    cell.d.ctrl[cell.m.actuator("vacuum_0").id] = 1.0
    cell.write_result(out)
    with pytest.raises(ValueError, match="not an empty tool"):
        PhysicalCell.resume(out)
