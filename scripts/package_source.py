#!/usr/bin/env python3
"""Build a source ZIP from the Git index without local results or submodules."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/share/rusal-sim-source.zip")
    args = parser.parse_args()
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode:
        raise SystemExit("Stage source changes before packaging: git add .")
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT)
    files = []
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode = metadata.split()[0]
        if mode == b"160000":
            continue  # Fetched using the revision lock, not copied into the ZIP.
        if mode not in (b"100644", b"100755"):
            raise SystemExit("Unsupported source file mode: " + name.decode())
        rel = Path(name.decode())
        if rel.parts[0] in {"results", ".git", ".venv", "build", "install", "log"}:
            raise SystemExit("Local output staged for release: " + str(rel))
        data = (ROOT / rel).read_bytes()
        if len(data) > 10 * 1024 * 1024:
            raise SystemExit("Source file exceeds 10 MiB: " + str(rel))
        if str(ROOT).encode() in data:
            raise SystemExit("Workstation path in staged source: " + str(rel))
        files.append((rel.as_posix(), data, mode))
    if not files:
        raise SystemExit("No staged source files.")
    manifest = {
        "format": 1,
        "robot_dependency": json.loads((ROOT / "robot-model.lock.json").read_text()),
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data, _ in files},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data, mode in files:
            info = zipfile.ZipInfo("rusal-sim/" + name)
            info.external_attr = (0o100755 if mode == b"100755" else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
        archive.writestr("rusal-sim/SOURCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(f"{args.out}: {len(files)} source files; {args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
