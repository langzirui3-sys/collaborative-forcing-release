from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
from pathlib import Path

from . import __version__
from .host import (
    HostConfigError,
    configure_hf_environment,
    load_host_config,
    resolve_hf_token,
)
from .manifest import ManifestError, load_manifest, manifest_hash, validate_manifest
from .reconcile import load_publication_patterns, reconcile_runs
from .runner import ExperimentError, build_plan, run_experiment
from .storage import StorageError, publish_run


def _host_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host-config",
        type=Path,
        default=Path("~/.config/cfexp/host.toml"),
        help="host-local TOML config (default: ~/.config/cfexp/host.toml)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfexp")
    parser.add_argument("--version", action="version", version=f"cfexp {__version__}")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect tools, paths, and storage config")
    _host_argument(doctor)

    validate = subparsers.add_parser("validate", help="validate one experiment manifest")
    validate.add_argument("manifest", type=Path)

    plan = subparsers.add_parser("plan", help="resolve a manifest without starting work")
    plan.add_argument("manifest", type=Path)
    plan.add_argument("--gpus", default=None)
    plan.add_argument("--allow-dirty", action="store_true")
    _host_argument(plan)

    run = subparsers.add_parser("run", help="run and publish one experiment")
    run.add_argument("manifest", type=Path)
    run.add_argument("--gpus", default=None)
    run.add_argument("--allow-dirty", action="store_true")
    run.add_argument(
        "--local-only",
        action="store_true",
        help="do not publish; intended only for smoke tests",
    )
    gate_receipts = run.add_mutually_exclusive_group()
    gate_receipts.add_argument(
        "--insight2-gate-receipt",
        type=Path,
        default=None,
        help="read-only policy-bound gate receipt required by Insight 2(b)",
    )
    gate_receipts.add_argument(
        "--insight2a-acceptance",
        type=Path,
        default=None,
        help=(
            "legacy flag for a sealed cfexp 2(a)-acceptance receipt; use "
            "--insight2-gate-receipt for new launch commands"
        ),
    )
    _host_argument(run)

    publish = subparsers.add_parser("publish", help="publish an existing local run")
    publish.add_argument("run_dir", type=Path)
    publish.add_argument("destination")
    publish.add_argument(
        "--marker",
        choices=["_SUCCESS.json", "_FAILED.json"],
        required=True,
    )
    _host_argument(publish)

    reconcile = subparsers.add_parser(
        "reconcile",
        help="retry publication for completed local runs",
    )
    reconcile.add_argument(
        "--run-id",
        default=None,
        help="reconcile only this run (default: scan all local runs)",
    )
    _host_argument(reconcile)
    return parser


def _disk(path: Path) -> dict[str, object]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {
        "path": str(path),
        "exists": path.exists(),
        "probe": str(probe),
        "free_gib": round(usage.free / (1024**3), 2),
        "used_percent": round(100 * usage.used / usage.total, 2),
    }


def _doctor_uv() -> str | None:
    """Report the launcher-pinned uv even when the hardened PATH omits it."""

    configured = os.environ.get("CFEXP_UV_BIN")
    if not configured:
        return shutil.which("uv")
    path = Path(configured)
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if (
        not path.is_absolute()
        or resolved != path
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid not in {0, os.getuid()}
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or not os.access(path, os.X_OK)
    ):
        return None
    return str(path)


def doctor_command(args: argparse.Namespace) -> int:
    host = load_host_config(args.host_config)
    tools = {
        name: shutil.which(name)
        for name in ("git", "rclone", "docker", "nvidia-smi")
    }
    tools["uv"] = _doctor_uv()
    report = {
        "version": __version__,
        "tools": tools,
        "paths": {
            "code_root": _disk(host.code_root),
            "model_root": _disk(host.model_root),
            "cache_root": _disk(host.cache_root),
            "run_root": _disk(host.run_root),
            "tmp_root": _disk(host.tmp_root),
        },
        "artifact_remote_configured": bool(host.artifact_remote),
        "secrets_file_exists": host.secrets_file.exists(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if tools["git"] and tools["uv"] else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.subcommand == "doctor":
            return doctor_command(args)
        if args.subcommand == "validate":
            manifest = load_manifest(args.manifest)
            validate_manifest(manifest)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "name": manifest["metadata"]["name"],
                        "spec_hash": manifest_hash(manifest),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.subcommand == "plan":
            host = load_host_config(args.host_config)
            plan = build_plan(
                args.manifest,
                host,
                args.gpus,
                allow_dirty=args.allow_dirty,
            )
            print(json.dumps(plan.serializable(host), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.subcommand == "run":
            host = load_host_config(args.host_config)
            result = run_experiment(
                args.manifest,
                host,
                args.gpus,
                allow_dirty=args.allow_dirty,
                local_only=args.local_only,
                insight2_gate_receipt=(
                    args.insight2_gate_receipt or args.insight2a_acceptance
                ),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return int(result["exit_code"])
        if args.subcommand == "publish":
            host = load_host_config(args.host_config)
            if args.destination.startswith("hf://"):
                configure_hf_environment(host)
            hf_token = (
                resolve_hf_token(host)
                if args.destination.startswith("hf://")
                else None
            )
            include_patterns, exclude_patterns = load_publication_patterns(args.run_dir)
            result = publish_run(
                args.run_dir,
                args.destination,
                args.marker,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                hf_token=hf_token,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.subcommand == "reconcile":
            host = load_host_config(args.host_config)
            result = reconcile_runs(host, only_run_id=args.run_id)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 1 if result["failures"] else 0
    except (ExperimentError, HostConfigError, ManifestError, StorageError, RuntimeError) as error:
        print(f"cfexp: error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
