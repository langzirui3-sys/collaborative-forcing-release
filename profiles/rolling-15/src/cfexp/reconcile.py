"""Retry publication for completed runs left below a host's run root.

The reconciler is intentionally conservative: only a run with exactly one
terminal marker is publishable.  Runs without a marker may still be active and
are therefore reported as skipped, while one bad completed run never prevents
the remaining completed runs from being retried.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .host import (
    FORMAL_HF_ARTIFACT_REMOTE,
    HostConfig,
    configure_hf_environment,
    resolve_hf_token,
)
from .storage import (
    TERMINAL_MARKERS,
    StorageError,
    configure_hf_publisher_environment,
    publish_run,
)


_PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")


def _safe_error(error: BaseException, token: str | None) -> str:
    """Return an error message that cannot disclose the resolved HF token."""

    message = str(error) or type(error).__name__
    if token:
        message = message.replace(token, "<redacted>")
    return message


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _terminal_marker(run_dir: Path) -> tuple[str, Path] | None:
    present = [
        name
        for name in sorted(TERMINAL_MARKERS)
        if (run_dir / name).exists() or (run_dir / name).is_symlink()
    ]
    if not present:
        return None
    if len(present) != 1:
        raise ValueError(f"run has conflicting terminal markers: {present}")
    marker_path = run_dir / present[0]
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError(f"terminal marker is not a regular file: {marker_path}")
    return present[0], marker_path


def _marker_month(marker_path: Path) -> tuple[int, int]:
    marker = _load_json_object(marker_path, "terminal marker")
    created_at = marker.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError(f"terminal marker has no valid created_at: {marker_path}")
    try:
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"terminal marker has invalid created_at {created_at!r}: {marker_path}"
        ) from error
    return timestamp.year, timestamp.month


def _publication_spec(
    run_dir: Path,
) -> tuple[dict[str, Any], str, list[str], list[str]]:
    resolved = _load_json_object(run_dir / "resolved-manifest.json", "resolved manifest")
    manifest = resolved.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("resolved manifest is missing manifest")
    spec = manifest.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("resolved manifest is missing manifest.spec")
    storage = spec.get("storage")
    outputs = spec.get("outputs")
    if not isinstance(storage, dict) or not isinstance(outputs, dict):
        raise ValueError("resolved manifest is missing storage or outputs")

    project = storage.get("project")
    if not isinstance(project, str) or not _PROJECT_RE.fullmatch(project):
        raise ValueError("resolved manifest storage.project is not path-safe")

    def patterns(name: str, *, required: bool) -> list[str]:
        value = outputs.get(name)
        if value is None and not required:
            return []
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise ValueError(f"resolved manifest outputs.{name} must be a string list")
        if required and not value:
            raise ValueError(f"resolved manifest outputs.{name} must not be empty")
        return list(value)

    return (
        resolved,
        project,
        patterns("include", required=True),
        patterns("exclude", required=False),
    )


def load_publication_patterns(run_dir: Path) -> tuple[list[str], list[str]]:
    """Load the original output selection for a safe low-level retry."""

    try:
        _, _, includes, excludes = _publication_spec(run_dir)
    except ValueError as error:
        raise StorageError(f"cannot recover publication patterns: {error}") from error
    return includes, excludes


def _destination(
    artifact_remote: str,
    project: str,
    year: int,
    month: int,
    run_id: str,
) -> str:
    remote = artifact_remote.strip()
    if not remote:
        raise ValueError("host artifact_remote is not configured")
    if any(character in run_id for character in ("\x00", "\n", "\r", "/", "\\")):
        raise ValueError(f"unsafe run id: {run_id!r}")
    separator = "" if remote.endswith(("/", ":")) else "/"
    return f"{remote}{separator}{project}/runs/{year:04d}/{month:02d}/{run_id}"


def _resolved_destination(
    resolved: dict[str, Any],
    artifact_remote: str,
    project: str,
    run_id: str,
    marker_path: Path,
) -> str:
    """Recover the exact start-time destination, including for month-crossing runs."""

    remote = artifact_remote.strip()
    if not remote:
        raise ValueError("host artifact_remote is not configured")
    separator = "" if remote.endswith(("/", ":")) else "/"
    prefix = f"{remote}{separator}{project}/runs/"
    saved = resolved.get("artifact_destination")
    if saved is not None:
        if not isinstance(saved, str) or not re.fullmatch(
            rf"{re.escape(prefix)}\d{{4}}/\d{{2}}/{re.escape(run_id)}", saved
        ):
            raise ValueError("resolved manifest artifact_destination is not valid for this run")
        return saved

    # Backward compatibility for runs produced before artifact_destination was
    # recorded: run IDs contain their UTC start timestamp. Synthetic/legacy
    # IDs without it fall back to the terminal marker completion time.
    match = re.search(r"(?:^|-)(\d{8})T\d{6}Z(?:-|$)", run_id)
    if match:
        started = datetime.strptime(match.group(1), "%Y%m%d")
        return _destination(remote, project, started.year, started.month, run_id)
    year, month = _marker_month(marker_path)
    return _destination(remote, project, year, month, run_id)


def _publication_token(host: HostConfig) -> str | None:
    """Resolve the publisher token without ever returning it in public results."""

    if not host.artifact_remote.startswith("hf://"):
        return None
    return resolve_hf_token(host)


def reconcile_runs(
    host: HostConfig,
    *,
    only_run_id: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Retry immutable publication for completed local runs.

    The returned lists are suitable for JSON output and deliberately contain
    no secret values.  Publication exceptions are isolated per run so later
    runs are still reconciled.
    """

    report: dict[str, list[dict[str, object]]] = {
        "successes": [],
        "skipped": [],
        "failures": [],
    }
    if host.artifact_remote.startswith("hf://"):
        configure_hf_environment(host)
        configure_hf_publisher_environment()
    run_root = host.run_root
    if not run_root.is_dir():
        report["failures"].append(
            {
                "run_id": only_run_id,
                "error": f"run root is not a directory: {run_root}",
            }
        )
        return report
    try:
        candidates = sorted(
            (
                path
                for path in run_root.iterdir()
                if path.is_dir() and not path.is_symlink()
            ),
            key=lambda path: path.name,
        )
    except OSError as error:
        report["failures"].append(
            {
                "run_id": only_run_id,
                "error": _safe_error(error, None),
            }
        )
        return report

    if only_run_id is not None:
        candidates = [path for path in candidates if path.name == only_run_id]
        if not candidates:
            report["failures"].append(
                {
                    "run_id": only_run_id,
                    "error": "requested run was not found",
                }
            )
            return report

    token: str | None = None
    token_error: BaseException | None = None
    try:
        token = _publication_token(host)
    except Exception as error:  # Keep an unsafe global credential from aborting the scan.
        token_error = error

    for run_dir in candidates:
        run_id = run_dir.name
        try:
            marker = _terminal_marker(run_dir)
            if marker is None:
                report["skipped"].append(
                    {
                        "run_id": run_id,
                        "reason": "no_terminal_marker",
                    }
                )
                continue
            marker_name, marker_path = marker
            resolved, project, includes, excludes = _publication_spec(run_dir)
            publication_requested = resolved.get("publication_requested")
            if publication_requested is not True:
                report["skipped"].append(
                    {
                        "run_id": run_id,
                        "reason": (
                            "local_only"
                            if publication_requested is False
                            else "unknown_publication_intent"
                        ),
                    }
                )
                continue
            if (
                host.artifact_remote.startswith("hf://")
                and host.artifact_remote != FORMAL_HF_ARTIFACT_REMOTE
            ):
                raise ValueError(
                    "formal reconciliation requires the pinned private Hugging Face "
                    f"dataset {FORMAL_HF_ARTIFACT_REMOTE}"
                )
            if token_error is not None:
                raise token_error
            destination = _resolved_destination(
                resolved,
                host.artifact_remote,
                project,
                run_id,
                marker_path,
            )
            publication = publish_run(
                run_dir,
                destination,
                marker_name,
                include_patterns=includes,
                exclude_patterns=excludes,
                hf_token=token,
            )
            report["successes"].append(
                {
                    "run_id": run_id,
                    "marker": marker_name,
                    "destination": destination,
                    "publication": publication,
                }
            )
        except Exception as error:
            report["failures"].append(
                {
                    "run_id": run_id,
                    "error": _safe_error(error, token),
                }
            )
    return report


__all__ = ["load_publication_patterns", "reconcile_runs"]
