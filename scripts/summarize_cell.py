"""Collect measured physical-cell results without conflating transfer and quality."""

import json
from pathlib import Path
from rusal_sim import ROOT

names = [
    "palletizing12_automatic",
    "palletizing12_verified",
    "physical_heap",
    "physical_heap_final",
    "physical_heap_guarded",
    "physical_heap_validated",
    "physical_heap_checked",
    "second_layer",
    "cell_mechanical_final",
    "gui_cycle",
    "cell_standard",
    "cell_fine",
]
summary = {}
for name in names:
    path = ROOT / "results/logs" / f"{name}.json"
    if not path.exists():
        continue
    result = json.loads(path.read_text())
    successful = [a for a in result["attempts"] if a.get("physically_deposited")]
    summary[name] = dict(
        report=str(path.relative_to(ROOT)),
        complete=result["complete"],
        transferred=result.get("transferred", result["physically_deposited"]),
        initially_on_pallet=result.get("initially_on_pallet", 0),
        quality_passes=result["quality_passes"],
        remaining=result["remaining"],
        worst_placement_error_mm=max(
            (a["position_error_m"] * 1000 for a in successful), default=None
        ),
        worst_final_error_mm=max(
            (a["position_error_m"] * 1000 for a in result.get("final_placement_audit", [])),
            default=None,
        ),
        max_speed_ratio=result["max_speed_ratio"],
        max_effort_ratio=result["max_effort_ratio"],
        robot_limits_pass=result["max_speed_ratio"] <= 1 and result["max_effort_ratio"] <= 1.000001,
        robot_collisions=result["robot_collisions"],
        warnings=result["warnings"],
    )
output = ROOT / "results/reports/physical_cell_summary.json"
output.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
