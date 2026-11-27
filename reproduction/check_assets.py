"""Verify the byte identities required by one archived inference profile."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    config_root = Path(__file__).resolve().parent
    profiles = json.loads((config_root / "profiles.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=profiles)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    source = config_root.parent / "profiles" / args.profile
    spec = profiles[args.profile]["file_arguments"]
    roots = {"model": args.model_root.expanduser().resolve(), "repository": source.resolve()}
    failures = 0
    verified = set()
    for flag in ("--draft-artifact-spec", "--target-artifact-spec"):
        assets = json.loads((source / "experiments" / spec[flag]).read_text())["artifacts"]
        for artifact in assets:
            root = roots[artifact["root"]]
            relative = Path(artifact["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Asset path escapes its declared root: {artifact['name']}")
            path = (root / relative).resolve()
            identity = (path, artifact["sha256"])
            if identity in verified:
                continue
            if not path.is_file():
                print(f"Missing {artifact['name']} ({relative})")
                failures += 1
                continue
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
            if checksum.hexdigest() != artifact["sha256"]:
                print(f"SHA256 mismatch for {artifact['name']} ({relative})")
                failures += 1
                continue
            verified.add(identity)
    print(f"Verified {len(verified)} unique files. {failures} failed checks.")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
