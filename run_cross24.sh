#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
.venv/bin/python - <<'PY'
from pathlib import Path
import json
import numpy as np
from rusal_sim import configuration
from rusal_sim.batch import make_batch
from rusal_sim.heap import generate_compact
cfg=configuration('configs/hybrid_cross24.yaml')
make_batch(cfg,24,42)
p=Path('results/logs/cross24_seed42')
valid=all(p.with_suffix(s).exists() for s in ('.npz','.xml','.json'))
if valid:
    r=json.loads(p.with_suffix('.json').read_text())
    saved=json.loads(str(np.load(p.with_suffix('.npz'))['config']))
    valid=r.get('settled') and r.get('validated_equilibrium') and r.get('count')==24 and saved.get('ingots')==cfg['ingots']
if not valid:
    generate_compact(cfg,24,42,str(p))
PY
exec ./run_sim.sh --config configs/hybrid_cross24.yaml --source results/logs/cross24_seed42 --mode hybrid --out results/logs/cross24_gui "$@"
