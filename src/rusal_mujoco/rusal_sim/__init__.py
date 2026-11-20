"""SI-unit research models. MuJoCo is the only dynamics engine."""

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]


def configuration(path=None):
    with (ROOT / "configs/default.yaml").open() as stream:
        defaults = yaml.safe_load(stream)
    if path is None:
        return defaults
    with open(path) as stream:
        overrides = yaml.safe_load(stream) or {}

    def merge(base, changes):
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value

    merge(defaults, overrides)
    return defaults
