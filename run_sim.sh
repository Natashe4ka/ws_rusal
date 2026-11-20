#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing .venv. Run: python3 -m venv --system-site-packages .venv && .venv/bin/pip install -e .' >&2
  exit 1
fi
headless_run=false
for arg in "$@"; do
  if [[ "$arg" == --headless ]]; then
    headless_run=true
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
  fi
done
# The remote desktop uses Mesa; software GL avoids DRI3 buffer allocation errors.
if [[ "$headless_run" == false ]]; then
  export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
  export LP_NUM_THREADS="${LP_NUM_THREADS:-2}"
fi
exec .venv/bin/python -m rusal_sim.cli sim "$@"
