#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if ! .venv/bin/python - <<'PY'
from pathlib import Path
import json,sys
p=Path('results/logs/compact20_seed42')
valid=all(p.with_suffix(s).exists() for s in ('.npz','.xml','.json'))
if valid:
    r=json.loads(p.with_suffix('.json').read_text())
    valid=r.get('settled',False) and r.get('validated_equilibrium',False) and r.get('count')==20
sys.exit(0 if valid else 1)
PY
then
  .venv/bin/python - <<'PY'
from rusal_sim import configuration
from rusal_sim.heap import generate_compact
generate_compact(configuration('configs/hybrid_heap20.yaml'),20,42,'results/logs/compact20_seed42')
PY
fi
exec ./run_sim.sh --config configs/hybrid_heap20.yaml --source results/logs/compact20_seed42 --mode hybrid --out results/logs/heap20_gui "$@"
