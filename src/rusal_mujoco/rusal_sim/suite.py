"""Parameterized physical runs. Aggregates are not industrial reliability estimates."""

import copy
import json
import csv
from pathlib import Path
import numpy as np
from . import ROOT
from .experiment import Experiment


def run(cfg):
    cases = []
    for mass in [5, 7.5, 8, 10, 15]:
        for mode in ["vacuum", "mechanical", "hybrid"]:
            for surface in ["smooth", "ribbed"]:
                cases.append(
                    (f"{mode}_{surface}_{mass}", dict(mass=mass, mode=mode, surface=surface))
                )
    for mu in [0.2, 0.3, 0.4, 0.6, 0.8]:
        cases.append((f"friction_{mu}", dict(mode="mechanical", surface="ribbed", mu=mu, mass=15)))
    for pressure in [0, 2000, 20000, 40000, 70000]:
        cases.append((f"pressure_{pressure}", dict(pressure=pressure, mass=15)))
    for leak in [0, 0.05, 0.2, 0.5, 1]:
        cases.append((f"seal_{leak}", dict(leak=leak, mass=15)))
    cases.extend(
        [
            ("one_cup_failed", dict(disabled=[0], mass=15)),
            ("V2", dict(layout="V2", mass=15)),
            ("all_off", dict(mode="off")),
        ]
    )
    for dt in [0.0005, 0.002]:
        for mode in ["vacuum", "mechanical"]:
            cases.append((f"dt_{dt}_{mode}", dict(dt=dt, mode=mode)))
    results = []
    for name, args in cases:
        c = copy.deepcopy(cfg)
        for key, section, param in [
            ("mass", "ingot", "mass"),
            ("mu", "contacts", "aluminum_steel"),
            ("pressure", "vacuum", "pressure_pa"),
            ("dt", "solver", "timestep"),
            ("layout", "vacuum", "layout"),
        ]:
            if key in args:
                c[section][param] = args[key]
        exp = Experiment(
            c,
            args.get("mode", "vacuum"),
            args.get("surface", "smooth"),
            args.get("disabled", ()),
            args.get("leak", 1),
        )
        result = exp.run(ROOT / "results/logs" / name)
        result["case"] = name
        results.append(result)
        print(name, result["success"], round(result["max_z"], 4), flush=True)
        (ROOT / "results/reports/suite.json").write_text(json.dumps(results, indent=2))
    run_extra(cfg)


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in [
        "vacuum_smooth_7.5",
        "vacuum_ribbed_7.5",
        "mechanical_ribbed_7.5",
        "hybrid_smooth_7.5",
        "pressure_2000",
    ]:
        path = ROOT / "results/logs" / f"{name}.csv"
        if not path.exists():
            continue
        with path.open() as f:
            rows = list(csv.DictReader(f))
        t = [float(r["time"]) for r in rows]
        fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        panels = [
            (["z", "tool_z"], "Height, m"),
            (["pressure_pa"], "Vacuum pressure, Pa"),
            (["vacuum_capacity_n"], "Vacuum capacity, N"),
            (["jaw_left_n", "jaw_right_n"], "Jaw normal force, N"),
            (["relative_displacement"], "Relative displacement, m"),
            (["contact_force_n"], "Sum contact force norms, N"),
        ]
        for ax, (keys, title) in zip(axs.flat, panels):
            for key in keys:
                ax.plot(t, [float(r[key]) for r in rows], label=key)
            ax.set_ylabel(title)
            ax.grid()
            ax.legend(fontsize=8)
            ax.set_xlabel("Time, s")
        fig.suptitle(name + " — driven fixture, free load")
        fig.tight_layout()
        fig.savefig(ROOT / "results/plots" / f"{name}.png")
        plt.close(fig)
    report = ROOT / "results/reports/suite.json"
    if report.exists():
        data = json.loads(report.read_text())
        fig, ax = plt.subplots()
        for mode in ["vacuum", "mechanical", "hybrid"]:
            masses = sorted(set(r["mass"] for r in data if r["case"].startswith(mode + "_")))
            ax.plot(
                masses,
                [
                    np.mean(
                        [
                            r["success"]
                            for r in data
                            if r["mode"] == mode
                            and r["mass"] == m
                            and r["case"].startswith(mode + "_")
                        ]
                    )
                    for m in masses
                ],
                "-o",
                label=mode,
            )
        ax.set(
            xlabel="Mass, kg",
            ylabel="Pass fraction across two surface scenarios",
            ylim=(-0.05, 1.05),
        )
        ax.legend()
        ax.grid()
        fig.tight_layout()
        fig.savefig(ROOT / "results/plots/mode_comparison.png")
        plt.close(fig)


def run_extra(cfg):
    path = ROOT / "results/reports/suite.json"
    results = [
        r for r in json.loads(path.read_text()) if not r["case"].startswith(("tilt_", "accel_"))
    ]
    for key, values in [("tilt", [0, 15, 30, 45, 60]), ("accel", [0, 2, 8, 20])]:
        for value in values:
            for mode in ["vacuum", "mechanical"]:
                c = copy.deepcopy(cfg)
                c["ingot"]["mass"] = 15
                c["experiment"][
                    "transport_tilt_deg" if key == "tilt" else "transport_acceleration"
                ] = value
                # Acceleration comparison uses a reduced vacuum reserve to expose shear failure.
                if key == "accel":
                    c["vacuum"]["pressure_pa"] = 20000
                name = f"{key}_{value}_{mode}"
                r = Experiment(c, mode).run(ROOT / "results/logs" / name)
                r["case"] = name
                results.append(r)
                path.write_text(json.dumps(results, indent=2))
                print(name, r["success"], flush=True)
    plot()
    plot_transport()


def plot_transport():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((ROOT / "results/reports/suite.json").read_text())
    for prefix, label in [
        ("tilt_", "Commanded transport tilt, deg"),
        ("accel_", "Commanded horizontal acceleration, m/s²"),
    ]:
        fig, ax = plt.subplots()
        for mode in ["vacuum", "mechanical"]:
            rows = [r for r in data if r["case"].startswith(prefix) and r["mode"] == mode]
            xs = [float(r["case"].split("_")[1]) for r in rows]
            ax.plot(xs, [int(r["success"]) for r in rows], "-o", label=mode)
        ax.set(xlabel=label, ylabel="Scenario pass (0/1)", ylim=(-0.05, 1.05))
        ax.legend()
        ax.grid()
        fig.tight_layout()
        fig.savefig(ROOT / "results/plots" / f"{prefix}transport.png")
        plt.close(fig)
