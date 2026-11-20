#!/usr/bin/env python3
"""Fetch the pinned KUKA dependency for a source archive or verify a checkout."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / "robot-model.lock.json").read_text())
    target = ROOT / lock["path"]
    if target.exists() and any(target.iterdir()):
        revision = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != lock["revision"]:
            raise SystemExit(
                f"KUKA revision mismatch: {revision}; expected {lock['revision']}. "
                "Existing files were not changed."
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                lock["repository"],
                str(target),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--detach", lock["revision"]], check=True
        )
    print(f"KUKA dependency verified: {lock['revision']}")


if __name__ == "__main__":
    main()
