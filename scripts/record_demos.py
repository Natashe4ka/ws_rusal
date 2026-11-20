from rusal_sim import configuration, ROOT
from rusal_sim.experiment import Experiment

for name, mode, surface in [
    ("vacuum_smooth", "vacuum", "smooth"),
    ("vacuum_ribbed_failure", "vacuum", "ribbed"),
    ("mechanical_ribbed", "mechanical", "ribbed"),
    ("hybrid_grasp", "hybrid", "smooth"),
]:
    print(
        name,
        Experiment(configuration(), mode, surface).run(
            ROOT / "results/logs" / f"video_{name}", ROOT / "results/videos" / f"{name}.mp4"
        ),
        flush=True,
    )
