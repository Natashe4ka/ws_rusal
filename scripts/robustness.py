"""Repeated, independently seeded rib-height deviations. Not a heap reliability estimate."""

import copy, json
from rusal_sim import configuration, ROOT
from rusal_sim.experiment import Experiment

results = []
for seed in [17, 42, 91]:
    for mode in ["vacuum", "mechanical", "hybrid"]:
        c = configuration()
        c["seed"] = seed
        c["ingot"]["variant"] = "C"
        e = Experiment(c, mode, "ribbed")
        r = e.run(ROOT / "results/logs" / f"robustness_{seed}_{mode}")
        r["seed"] = seed
        r["variant"] = "C"
        results.append(r)
        print(seed, mode, r["success"], flush=True)
(ROOT / "results/reports/robustness.json").write_text(json.dumps(results, indent=2))
