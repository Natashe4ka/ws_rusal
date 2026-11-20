import argparse
import json
from pathlib import Path
from . import ROOT, configuration


def main():
    p = argparse.ArgumentParser(description="RUSAL contact physics research")
    p.add_argument("--config")
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("experiment")
    e.add_argument("--mode", choices=["vacuum", "mechanical", "hybrid", "off"], default="vacuum")
    e.add_argument("--surface", choices=["smooth", "ribbed"], default="smooth")
    e.add_argument("--mass", type=float)
    e.add_argument("--pressure", type=float)
    e.add_argument("--layout", choices=["V1", "V2"])
    e.add_argument("--video")
    e.add_argument("--out", default="results/logs/trial")
    h = sub.add_parser("heap")
    h.add_argument("--count", type=int, choices=[5, 10, 25, 50, 100, 250], default=10)
    h.add_argument("--sleep", action="store_true")
    h.add_argument("--seed", type=int, default=42)
    h.add_argument("--variant", choices=["A", "B", "C"], default="A")
    h.add_argument("--out", default="results/logs/heap")
    sim = sub.add_parser("sim")
    sim.add_argument("--config", dest="sim_config")
    sim.add_argument("--resume")
    sim.add_argument("--source")
    sim.add_argument("--scenario", choices=["heap", "stack", "compact"], default="compact")
    sim.add_argument("--count", type=int)
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--surface", choices=["smooth", "ribbed", "mixed"])
    sim.add_argument("--headless", action="store_true")
    sim.add_argument("--speed", type=float, default=1.0)
    sim.add_argument("--max-attempts", type=int)
    sim.add_argument("--layers", type=int)
    sim.add_argument("--prefill-layers", type=int, default=0)
    sim.add_argument("--mode", choices=["vacuum", "mechanical", "hybrid"], default="hybrid")
    sim.add_argument("--out", default="results/logs/sim")
    sim.add_argument("--video")
    rt = sub.add_parser("robot-trial")
    rt.add_argument("--video")
    sub.add_parser("robot")
    sub.add_parser("suite")
    sub.add_parser("plot")
    sub.add_parser("reachability")
    v = sub.add_parser("view")
    v.add_argument("model", nargs="?", default="src/rusal_description/mjcf/cell.xml")
    v.add_argument("--state")
    he = sub.add_parser("heap-extract")
    he.add_argument("--source", default="results/logs/heap10_detailed")
    he.add_argument("--max-attempts", type=int, default=3)
    he.add_argument("--out", default="results/logs/heap_execution")
    he.add_argument("--video")
    pp = sub.add_parser("pick-place")
    pp.add_argument("--video")
    pp.add_argument("--out", default="results/logs/pick_place")
    sub.add_parser("ros")
    a = p.parse_args()
    cfg = configuration(getattr(a, "sim_config", None) or a.config)
    if a.command == "sim":
        from .cell import PhysicalCell

        if not 0 < a.speed <= 8:
            p.error("--speed must be in (0,8]")
        if a.source is not None and a.scenario == "compact":
            a.scenario = "heap"
        if a.scenario == "heap" and a.count is not None and not a.resume:
            p.error("--count applies to generated heaps: use --scenario compact --count 5")
        if a.count is None:
            a.count = cfg.get("simulation", {}).get("count", 5 if a.scenario == "compact" else 12)
        if a.layers is None:
            a.layers = cfg.get("simulation", {}).get("layers", 1)
        if a.max_attempts is None:
            a.max_attempts = cfg.get("simulation", {}).get("max_attempts", 30)
        if a.layers < 1 or a.max_attempts < 0 or a.count < 1:
            p.error("layers/count must be positive; max-attempts must be nonnegative")
        if a.scenario == "compact" and not a.resume:
            from .heap import generate_compact

            a.source = f"results/logs/compact{a.count}_seed{a.seed}" + (
                "" if a.surface in (None, "mixed") else "_" + a.surface
            )
            print(
                json.dumps(
                    generate_compact(cfg, a.count, a.seed, a.source, surface=a.surface or "mixed"),
                    indent=2,
                ),
                flush=True,
            )
        if a.scenario == "stack" and not a.resume:
            from .demo_source import generate_stack

            a.source = f"results/logs/stack{a.count}"
            generate_stack(a.count, a.source, surface=a.surface or "smooth")
        if a.source is None:
            a.source = "results/logs/heap10_detailed"
        cell = (
            PhysicalCell.resume(a.resume)
            if a.resume
            else PhysicalCell(a.source, a.layers, a.mode, cfg, a.prefill_layers)
        )
        print(
            json.dumps(cell.run(a.max_attempts, a.out, not a.headless, a.speed, a.video), indent=2)
        )
    elif a.command == "experiment":
        from .experiment import Experiment

        if a.mass is not None:
            cfg["ingot"]["mass"] = a.mass
        if a.pressure is not None:
            cfg["vacuum"]["pressure_pa"] = a.pressure
        if a.layout:
            cfg["vacuum"]["layout"] = a.layout
        print(json.dumps(Experiment(cfg, a.mode, a.surface).run(a.out, a.video), indent=2))
    elif a.command == "heap":
        from .heap import generate

        if a.sleep:
            cfg["solver"]["sleep"] = True
        print(json.dumps(generate(cfg, a.count, a.seed, a.out, a.variant), indent=2))
    elif a.command == "robot-trial":
        from .robot_trial import run

        print(json.dumps(run(a.video), indent=2))
    elif a.command == "robot":
        from .robot import validate

        result = validate(cfg)
        (ROOT / "results/reports").mkdir(parents=True, exist_ok=True)
        (ROOT / "results/reports/kuka_fk.json").write_text(json.dumps(result, indent=2))
        print(result)
    elif a.command == "suite":
        from .suite import run

        run(cfg)
    elif a.command == "plot":
        from .suite import plot

        plot()
    elif a.command == "view":
        import mujoco
        import mujoco.viewer
        from .heap import restore
        import time

        m = mujoco.MjModel.from_xml_path(a.model)
        d = restore(m, a.state) if a.state else mujoco.MjData(m)
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running():
                tick = time.monotonic()
                mujoco.mj_step(m, d)
                viewer.sync()
                time.sleep(max(0, m.opt.timestep - (time.monotonic() - tick)))
    elif a.command == "heap-extract":
        from .heap_execution import HeapExecution

        print(json.dumps(HeapExecution(a.source).execute(a.max_attempts, a.out, a.video), indent=2))
    elif a.command == "pick-place":
        from .pick_place import PickPlace

        print(json.dumps(PickPlace(cfg).run_cycle(a.out, a.video), indent=2))
    elif a.command == "ros":
        from .ros_bridge import main as ros_main

        ros_main()
    elif a.command == "reachability":
        import mujoco
        from .planning import reachability
        import itertools

        m = mujoco.MjModel.from_xml_path(str(ROOT / "src/rusal_description/mjcf/cell.xml"))
        points = list(itertools.product([-0.5, 0.3, 1.1], [-0.8, 0, 0.8], [0.05, 0.5, 1.0, 1.5]))
        result = reachability(m, points)
        (ROOT / "results/reports/reachability.json").write_text(json.dumps(result, indent=2))
        print("collision-free", sum(x["collision_free"] for x in result), "/", len(result))


if __name__ == "__main__":
    main()
