"""Reproducible individual geometry; prescribed mass overrides material density."""

import copy
import numpy as np


def ingot_config(cfg, name):
    return cfg.get("ingots", {}).get(name, cfg["ingot"])


def make_batch(cfg, count, seed):
    variation = cfg.get("ingot_variation")
    if not variation:
        return
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1301]))
    cfg["ingots"] = {}
    for i in range(count):
        item = copy.deepcopy(cfg["ingot"])
        for key in ("length", "width", "thickness", "mass"):
            item[key] = float(rng.uniform(*variation[key]))
        item["geometry_seed"] = seed + i
        cfg["ingots"][f"ingot_{i}"] = item
