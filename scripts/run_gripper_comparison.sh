#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/share/gripper_comparison
echo 'Generating one shared, physically settled five-ingot heap...'
.venv/bin/python - <<'PY'
from rusal_sim import configuration
from rusal_sim.heap import generate_compact
r=generate_compact(configuration(),5,42,'results/logs/compact5_seed42')
print('Stable:',r['settled'],'Footprint:',r['footprint_m'])
PY
echo 'Recording Piab approximation; progress: results/share/gripper_comparison/piab_console.txt'
./run_sim.sh --config configs/piab_kvg150.yaml --source results/logs/compact5_seed42 --mode vacuum --headless --max-attempts 15 --out results/share/gripper_comparison/piab_run --video results/share/gripper_comparison/piab_raw.mp4 > results/share/gripper_comparison/piab_console.txt 2>&1
echo 'Recording SCHUNK approximation; progress: results/share/gripper_comparison/schunk_console.txt'
./run_sim.sh --config configs/schunk_pgn200.yaml --source results/logs/compact5_seed42 --mode mechanical --headless --max-attempts 15 --out results/share/gripper_comparison/schunk_run --video results/share/gripper_comparison/schunk_raw.mp4 > results/share/gripper_comparison/schunk_console.txt 2>&1
echo 'Adding close-ups and Russian result cards...'
MUJOCO_GL=egl .venv/bin/python scripts/package_gripper_videos.py
echo 'Ready: results/share/gripper_comparison.zip'
