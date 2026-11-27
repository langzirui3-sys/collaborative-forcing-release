"""Auditable human gate receipts for Insight 2(b).

The default and exploratory policies bind an explicit human decision to sealed
cfexp 2(a) evidence.  The external-prior policy is intentionally different: it
binds only a human compute authorization to the exact 2(b) manifest/commit and
states that prior 2(a) evidence is external and was not revalidated here.  It
does not manufacture a source run or a statistical claim.

Acceptance receipts live in the host-local ``approvals`` directory beside the
configured run root.  They are inputs to a later run, never outputs smuggled
into either run directory.  The formal runner copies the validated bytes into
the new attempt's provenance directory so publication preserves the decision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence

from .host import FORMAL_HF_ARTIFACT_REMOTE, HostConfig, load_host_config
from .manifest import load_manifest, manifest_hash, validate_manifest
from .metadata import collect_git_metadata, find_git_root


SIGNAL_ACCEPTANCE_SCHEMA = "insight2a_signal_acceptance_v1"
SIGNAL_ACCEPTANCE_PROVENANCE_PATH = "provenance/insight2a-signal-acceptance.json"
DEFAULT_SIGNAL_ACCEPTANCE_POLICY = "sealed_2a_eval_same_revision"
EXPLORATORY_SMOKE_ACCEPTANCE_POLICY = (
    "sealed_2a_smoke_cross_revision_engineering_smoke_only"
)
EXTERNAL_PRIOR_2B_EVAL_ACCEPTANCE_POLICY = (
    "human_external_prior_2b_eval_compute_only"
)
EXTERNAL_PRIOR_2B_AUTHORIZATION_SCHEMA = (
    "insight2b_external_prior_human_authorization_v1"
)
EXTERNAL_PRIOR_2B_AUTHORIZATION_PROVENANCE_PATH = (
    "provenance/insight2b-external-prior-human-authorization.json"
)
_EXPLORATORY_SMOKE_LIMITATIONS = {
    "authorization_scope": "engineering_smoke_only",
    "correction_gain_signal": "not_statistically_significant_in_bound_2a_smoke",
    "paper_claim_authorized": False,
}
_EXTERNAL_PRIOR_2B_EVAL_LIMITATIONS = {
    "authorization_scope": "2b_computation_only",
    "prior_2a_evidence": "external_to_cfexp_not_revalidated",
    "cfexp_2a_result_generation_authorized": False,
    "cfexp_2a_claim_authorized": False,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_RFC3339_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")


class SignalAcceptanceError(ValueError):
    """An Insight 2 gate receipt or any evidence it claims is invalid."""


@dataclass(frozen=True)
class SignalAcceptance:
    """Validated immutable receipt and the exact bytes copied into provenance."""

    source_path: Path
    sha256: str
    raw_bytes: bytes
    payload: dict[str, Any]

    @property
    def acceptance_policy(self) -> str:
        return str(
            self.payload.get("acceptance_policy", DEFAULT_SIGNAL_ACCEPTANCE_POLICY)
        )

    @property
    def provenance_path(self) -> str:
        if self.acceptance_policy == EXTERNAL_PRIOR_2B_EVAL_ACCEPTANCE_POLICY:
            return EXTERNAL_PRIOR_2B_AUTHORIZATION_PROVENANCE_PATH
        return SIGNAL_ACCEPTANCE_PROVENANCE_PATH

    def audit_record(self) -> dict[str, Any]:
        return {
            "schema": self.payload["schema"],
            "receipt_sha256": self.sha256,
            "receipt": self.payload,
            "copied_to": self.provenance_path,
        }


def approval_root(host: HostConfig) -> Path:
    """Return the one controlled cross-run input directory for this host."""

    return host.run_root.expanduser().resolve().parent / "approvals"


def _fail(message: str) -> None:
    raise SignalAcceptanceError(message)


def _duplicate_checked_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            _fail(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _invalid_constant(value: str) -> None:
    _fail(f"invalid JSON constant {value!r}")


def _load_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_invalid_constant,
        )
    except SignalAcceptanceError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SignalAcceptanceError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        _fail(f"{label} must contain a JSON object")
    return value


def _read_regular(path: Path, *, label: str, maximum_size: int = 16 * 1024 * 1024) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SignalAcceptanceError(f"cannot inspect {label}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        _fail(f"{label} must be a regular non-symlink file")
    if metadata.st_size > maximum_size:
        _fail(f"{label} is unexpectedly large")
    try:
        return path.read_bytes()
    except OSError as error:
        raise SignalAcceptanceError(f"cannot read {label}: {error}") from error


def _load_json_file(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, label=label)
    return _load_json_bytes(raw, label=label), raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _nonempty_text(value: Any, *, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        _fail(f"{label} must be a non-empty, trimmed string")
    if len(value) > maximum or any(ord(character) < 32 for character in value):
        _fail(f"{label} contains invalid text")
    return value


def _safe_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail(f"{label} must be a canonical relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in posix.parts)
        or posix.as_posix() != value
    ):
        _fail(f"{label} must be a canonical relative path")
    return value


def _resolve_run_file(run_dir: Path, value: Any, *, label: str) -> Path:
    relative = _safe_relative_path(value, label=label)
    candidate = run_dir / relative
    cursor = run_dir
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"{label} may not traverse a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(run_dir)
    except (OSError, ValueError) as error:
        raise SignalAcceptanceError(f"{label} is unavailable below the source run") from error
    return resolved


def _manifest_phase(manifest: Mapping[str, Any]) -> str:
    try:
        spec = manifest["spec"]
        parameters = spec["parameters"]
        command = spec["command"]
        parameter_phase = parameters["phase"]
        argv = command["argv"]
    except (KeyError, TypeError) as error:
        raise SignalAcceptanceError("Insight 2 manifest has no frozen phase") from error
    if not isinstance(parameter_phase, str) or not isinstance(argv, list):
        _fail("Insight 2 manifest phase is invalid")
    command_phases = [
        argv[index + 1]
        for index, item in enumerate(argv[:-1])
        if item == "--phase"
    ]
    if command_phases != [parameter_phase]:
        _fail("Insight 2 manifest command and parameters disagree on phase")
    return parameter_phase


def requires_signal_acceptance(manifest: Mapping[str, Any]) -> bool:
    """Validate the manifest's visible gate flag and return whether it is gated."""

    phase = _manifest_phase(manifest)
    try:
        parameters = manifest["spec"]["parameters"]
    except (KeyError, TypeError) as error:
        raise SignalAcceptanceError("Insight 2 manifest parameters are invalid") from error
    gate = parameters.get("run_only_after_2a_signal_check")
    if phase.startswith("2b-"):
        if gate is not True:
            _fail("every Insight 2(b) manifest must enable the 2(a) signal gate")
        signal_acceptance_policy(manifest)
        return True
    if gate is not None or "signal_acceptance_policy" in parameters:
        _fail("the 2(a) signal gate policy is only valid for Insight 2(b) manifests")
    return False


def _validate_target_policy(*, target_phase: str, policy: str) -> str:
    if target_phase not in {"2b-smoke", "2b-eval"}:
        _fail("signal acceptance can authorize only an Insight 2(b) phase")
    if policy == DEFAULT_SIGNAL_ACCEPTANCE_POLICY:
        return policy
    if policy == EXPLORATORY_SMOKE_ACCEPTANCE_POLICY:
        if target_phase != "2b-smoke":
            _fail("exploratory 2(a)-smoke evidence can authorize only 2b-smoke")
        return policy
    if policy == EXTERNAL_PRIOR_2B_EVAL_ACCEPTANCE_POLICY:
        if target_phase != "2b-eval":
            _fail("external-prior human authorization can authorize only 2b-eval")
        return policy
    _fail(f"unsupported signal acceptance policy: {policy!r}")


def signal_acceptance_policy(manifest: Mapping[str, Any]) -> str:
    """Return the explicit evidence policy for a gated Insight 2(b) manifest."""

    phase = _manifest_phase(manifest)
    try:
        parameters = manifest["spec"]["parameters"]
    except (KeyError, TypeError) as error:
        raise SignalAcceptanceError("Insight 2 manifest parameters are invalid") from error
    if not isinstance(parameters, Mapping):
        _fail("Insight 2 manifest parameters are invalid")
    value = parameters.get(
        "signal_acceptance_policy", DEFAULT_SIGNAL_ACCEPTANCE_POLICY
    )
    if not isinstance(value, str):
        _fail("signal_acceptance_policy must be a string")
    return _validate_target_policy(target_phase=phase, policy=value)


def _source_run_directory(run_root: Path, run_id: str) -> Path:
    if _SAFE_COMPONENT_RE.fullmatch(run_id) is None or run_id in {".", ".."}:
        _fail("source_2a.run_id is not a safe run identifier")
    root = run_root.expanduser().resolve(strict=True)
    candidate = root / run_id
    if candidate.is_symlink():
        _fail("source 2(a) run directory may not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise SignalAcceptanceError("source 2(a) run is unavailable") from error
    if resolved.parent != root or not resolved.is_dir():
        _fail("source 2(a) run must be a direct run_root child")
    return resolved


def _index_entries(index: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    _require_keys(index, {"schema_version", "algorithm", "files"}, label="artifact index")
    if index["schema_version"] != 1 or index["algorithm"] != "sha256":
        _fail("source 2(a) artifact index schema is unsupported")
    files = index["files"]
    if not isinstance(files, list):
        _fail("source 2(a) artifact index files must be a list")
    result: dict[str, dict[str, Any]] = {}
    for offset, item in enumerate(files):
        if not isinstance(item, dict):
            _fail(f"artifact index entry {offset} must be an object")
        _require_keys(item, {"path", "size", "sha256"}, label=f"artifact index entry {offset}")
        path = _safe_relative_path(item["path"], label=f"artifact index entry {offset}.path")
        size = item["size"]
        digest = item["sha256"]
        if (
            path in result
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            _fail(f"artifact index entry {offset} is invalid or duplicated")
        result[path] = dict(item)
    return result


def inspect_source_2a_run(
    *,
    run_root: Path,
    run_id: str,
    expected_artifact_remote: str,
    required_phase: str = "2a-eval",
) -> dict[str, Any]:
    """Validate a sealed 2(a) run and return receipt-bound evidence."""

    if required_phase not in {"2a-smoke", "2a-eval"}:
        _fail("source 2(a) required phase is invalid")

    run_dir = _source_run_directory(run_root, run_id)
    failed_marker = run_dir / "_FAILED.json"
    if failed_marker.exists() or failed_marker.is_symlink():
        _fail("source 2(a) run contains a failure marker")
    marker, marker_raw = _load_json_file(run_dir / "_SUCCESS.json", label="success marker")
    index, index_raw = _load_json_file(
        run_dir / "artifact-index.json", label="artifact index"
    )
    index_digest = _sha256(index_raw)
    if marker.get("run_id") != run_id or marker.get("status") != "SUCCEEDED":
        _fail("source 2(a) terminal marker is not a successful marker for this run")
    if marker.get("artifact_index_sha256") != index_digest:
        _fail("source 2(a) terminal marker is not bound to the artifact index bytes")
    publication = marker.get("artifact_publication")
    if not isinstance(publication, dict):
        _fail("source 2(a) terminal marker has no publication policy")
    destination = publication.get("destination")
    expected_prefix = f"{expected_artifact_remote.rstrip('/')}/collaborative-forcing/runs/"
    if (
        not isinstance(destination, str)
        or not destination.startswith(expected_prefix)
        or not destination.endswith(f"/{run_id}")
    ):
        _fail("source 2(a) run is not bound to the formal immutable artifact destination")

    resolved, _ = _load_json_file(
        run_dir / "resolved-manifest.json", label="source resolved manifest"
    )
    source_manifest = resolved.get("manifest")
    if not isinstance(source_manifest, dict):
        _fail("source resolved manifest has no experiment manifest")
    try:
        validate_manifest(source_manifest)
    except Exception as error:
        raise SignalAcceptanceError(f"source experiment manifest is invalid: {error}") from error
    phase = _manifest_phase(source_manifest)
    if phase != required_phase:
        _fail(f"Insight 2(b) acceptance requires a completed {required_phase} run")
    source_spec_hash = resolved.get("spec_hash")
    if (
        not isinstance(source_spec_hash, str)
        or _SHA256_RE.fullmatch(source_spec_hash) is None
        or source_spec_hash != manifest_hash(source_manifest)
        or marker.get("spec_hash") != source_spec_hash
    ):
        _fail("source 2(a) manifest digest binding is invalid")
    git = resolved.get("git")
    source_revision = git.get("sha") if isinstance(git, dict) else None
    if (
        not isinstance(source_revision, str)
        or _GIT_SHA_RE.fullmatch(source_revision) is None
        or marker.get("git_sha") != source_revision
    ):
        _fail("source 2(a) Git revision binding is invalid")
    if resolved.get("artifact_destination") != destination:
        _fail("source 2(a) resolved destination disagrees with its terminal marker")

    run_state, _ = _load_json_file(run_dir / "run.json", label="source run state")
    attempt_id = run_state.get("latest_attempt_id")
    if (
        run_state.get("run_id") != run_id
        or run_state.get("execution_status") != "SUCCEEDED"
        or not isinstance(attempt_id, str)
        or not attempt_id
        or marker.get("attempt_id") != attempt_id
    ):
        _fail("source 2(a) run state is not a successful terminal attempt")

    result_relative = f"attempts/{attempt_id}/result.json"
    result_path = _resolve_run_file(run_dir, result_relative, label="source result")
    result, _ = _load_json_file(result_path, label="source result")
    analysis = result.get("analysis")
    if (
        result.get("status") != "succeeded"
        or result.get("phase") != required_phase
        or not isinstance(analysis, dict)
    ):
        _fail(f"source 2(a) result is not a successful {required_phase} result")
    summary_inner = _safe_relative_path(
        analysis.get("summary"), label="source result analysis.summary"
    )
    figure_inner = _safe_relative_path(
        analysis.get("plot_data"), label="source result analysis.plot_data"
    )
    summary_relative = f"attempts/{attempt_id}/{summary_inner}"
    figure_relative = f"attempts/{attempt_id}/{figure_inner}"

    entries = _index_entries(index)
    evidence: dict[str, tuple[str, str]] = {}
    for label, relative in (
        ("summary", summary_relative),
        ("figure_data", figure_relative),
        ("result", result_relative),
    ):
        entry = entries.get(relative)
        if entry is None:
            _fail(f"source 2(a) artifact index omits {label}")
        path = _resolve_run_file(run_dir, relative, label=f"source {label}")
        raw = _read_regular(path, label=f"source {label}")
        digest = _sha256(raw)
        if len(raw) != entry["size"] or digest != entry["sha256"]:
            _fail(f"source 2(a) {label} no longer matches the sealed artifact index")
        evidence[label] = (relative, digest)

    if required_phase == "2a-smoke":
        summary, _ = _load_json_file(
            _resolve_run_file(
                run_dir, evidence["summary"][0], label="source smoke summary"
            ),
            label="source smoke summary",
        )
        spearman = summary.get("spearman")
        ci95 = spearman.get("ci95") if isinstance(spearman, dict) else None
        if (
            not isinstance(ci95, list)
            or len(ci95) != 2
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in ci95
            )
            or not float(ci95[0]) <= 0.0 <= float(ci95[1])
        ):
            _fail(
                "exploratory smoke policy requires a bound 2a-smoke whose primary "
                "risk-gain Spearman 95% CI includes zero"
            )

    return {
        "run_id": run_id,
        "phase": required_phase,
        "code_revision": source_revision,
        "manifest_spec_sha256": source_spec_hash,
        "artifact_destination": destination,
        "success_marker_sha256": _sha256(marker_raw),
        "artifact_index_sha256": index_digest,
        "summary_path": evidence["summary"][0],
        "summary_sha256": evidence["summary"][1],
        "figure_data_path": evidence["figure_data"][0],
        "figure_data_sha256": evidence["figure_data"][1],
    }


def _validate_approval_location(path: Path, root: Path) -> Path:
    unresolved = path.expanduser()
    if not unresolved.is_absolute() or unresolved.name in {"", ".", ".."}:
        _fail("signal acceptance path must be an absolute receipt file path")
    if unresolved.parent != root:
        _fail("signal acceptance receipt must be a direct child of the host approvals root")
    try:
        root_metadata = root.lstat()
        path_metadata = unresolved.lstat()
    except OSError as error:
        raise SignalAcceptanceError(f"cannot inspect signal acceptance path: {error}") from error
    if root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        _fail("host approvals root must be a regular non-symlink directory")
    if root_metadata.st_uid != os.geteuid() or stat.S_IMODE(root_metadata.st_mode) & 0o022:
        _fail("host approvals root must be owned by the runner and not group/world writable")
    if unresolved.is_symlink() or not stat.S_ISREG(path_metadata.st_mode):
        _fail("signal acceptance receipt must be a regular non-symlink file")
    if path_metadata.st_uid != os.geteuid() or stat.S_IMODE(path_metadata.st_mode) & 0o222:
        _fail("signal acceptance receipt must be runner-owned and read-only")
    try:
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise SignalAcceptanceError("signal acceptance receipt escapes approvals root") from error
    if resolved.parent != root:
        _fail("signal acceptance receipt must be a direct child of approvals root")
    return resolved


def load_signal_acceptance(
    path: Path,
    *,
    host: HostConfig,
    target_phase: str,
    target_spec_hash: str,
    target_code_revision: str,
    target_acceptance_policy: str = DEFAULT_SIGNAL_ACCEPTANCE_POLICY,
) -> SignalAcceptance:
    """Load and fully bind a read-only receipt to source evidence and target run."""

    target_acceptance_policy = _validate_target_policy(
        target_phase=target_phase, policy=target_acceptance_policy
    )
    if _SHA256_RE.fullmatch(target_spec_hash) is None:
        _fail("target manifest spec hash is invalid")
    if _GIT_SHA_RE.fullmatch(target_code_revision) is None:
        _fail("target code revision is invalid")
    root = approval_root(host)
    receipt_path = _validate_approval_location(path, root)
    raw = _read_regular(receipt_path, label="signal acceptance receipt", maximum_size=64 * 1024)
    payload = _load_json_bytes(raw, label="signal acceptance receipt")
    receipt_policy = payload.get(
        "acceptance_policy", DEFAULT_SIGNAL_ACCEPTANCE_POLICY
    )
    if not isinstance(receipt_policy, str):
        _fail("signal acceptance receipt policy must be a string")
    receipt_policy = _validate_target_policy(
        target_phase=target_phase, policy=receipt_policy
    )
    base_keys = {
        "schema",
        "decision",
        "accepted_at",
        "accepted_by",
        "rationale",
        "accepted_for",
    }
    if receipt_policy == DEFAULT_SIGNAL_ACCEPTANCE_POLICY:
        expected_receipt_keys = base_keys | {"source_2a"}
    elif receipt_policy == EXPLORATORY_SMOKE_ACCEPTANCE_POLICY:
        expected_receipt_keys = base_keys | {
            "source_2a",
            "acceptance_policy",
            "limitations",
        }
    else:
        expected_receipt_keys = base_keys | {"acceptance_policy", "limitations"}
    _require_keys(payload, expected_receipt_keys, label="signal acceptance receipt")
    if receipt_policy != target_acceptance_policy:
        _fail("signal acceptance receipt policy does not match the target manifest")
    if receipt_policy == EXPLORATORY_SMOKE_ACCEPTANCE_POLICY:
        if payload["limitations"] != _EXPLORATORY_SMOKE_LIMITATIONS:
            _fail("exploratory smoke receipt must preserve the required limitations")
    if receipt_policy == EXTERNAL_PRIOR_2B_EVAL_ACCEPTANCE_POLICY:
        if payload["limitations"] != _EXTERNAL_PRIOR_2B_EVAL_LIMITATIONS:
            _fail("external-prior receipt must preserve the required limitations")
        expected_schema = EXTERNAL_PRIOR_2B_AUTHORIZATION_SCHEMA
    else:
        expected_schema = SIGNAL_ACCEPTANCE_SCHEMA
    if payload["schema"] != expected_schema or payload["decision"] != "accept":
        _fail("signal acceptance receipt must contain an explicit accept decision")
    accepted_at = payload["accepted_at"]
    if not isinstance(accepted_at, str) or _RFC3339_UTC_RE.fullmatch(accepted_at) is None:
        _fail("signal acceptance accepted_at must be an RFC3339 UTC timestamp")
    try:
        parsed_at = datetime.fromisoformat(accepted_at[:-1] + "+00:00")
    except ValueError as error:
        raise SignalAcceptanceError("signal acceptance accepted_at is invalid") from error
    if parsed_at.tzinfo != timezone.utc:
        _fail("signal acceptance accepted_at must use UTC")
    _nonempty_text(payload["accepted_by"], label="accepted_by", maximum=200)
    _nonempty_text(payload["rationale"], label="rationale", maximum=4000)

    accepted_for = payload["accepted_for"]
    if not isinstance(accepted_for, dict):
        _fail("accepted_for must be an object")
    _require_keys(
        accepted_for,
        {"phase", "code_revision", "manifest_spec_sha256"},
        label="accepted_for",
    )
    expected_target = {
        "phase": target_phase,
        "code_revision": target_code_revision,
        "manifest_spec_sha256": target_spec_hash,
    }
    if accepted_for != expected_target:
        _fail("signal acceptance is not bound to this exact 2(b) phase, commit, and manifest")

    if receipt_policy == EXTERNAL_PRIOR_2B_EVAL_ACCEPTANCE_POLICY:
        return SignalAcceptance(
            source_path=receipt_path,
            sha256=_sha256(raw),
            raw_bytes=raw,
            payload=payload,
        )

    source = payload["source_2a"]
    if not isinstance(source, dict):
        _fail("source_2a must be an object")
    expected_source_keys = {
        "run_id",
        "phase",
        "code_revision",
        "manifest_spec_sha256",
        "artifact_destination",
        "success_marker_sha256",
        "artifact_index_sha256",
        "summary_path",
        "summary_sha256",
        "figure_data_path",
        "figure_data_sha256",
    }
    _require_keys(source, expected_source_keys, label="source_2a")
    run_id = source.get("run_id")
    if not isinstance(run_id, str):
        _fail("source_2a.run_id must be a string")
    inspected = inspect_source_2a_run(
        run_root=host.run_root,
        run_id=run_id,
        expected_artifact_remote=host.artifact_remote,
        required_phase=(
            "2a-smoke"
            if receipt_policy == EXPLORATORY_SMOKE_ACCEPTANCE_POLICY
            else "2a-eval"
        ),
    )
    if (
        receipt_policy == DEFAULT_SIGNAL_ACCEPTANCE_POLICY
        and inspected["code_revision"] != target_code_revision
    ):
        _fail("source 2(a) and target 2(b) must use the same published code revision")
    if source != inspected:
        _fail("signal acceptance source_2a evidence disagrees with the sealed source run")
    return SignalAcceptance(
        source_path=receipt_path,
        sha256=_sha256(raw),
        raw_bytes=raw,
        payload=payload,
    )


def write_signal_acceptance(
    destination: Path,
    *,
    host: HostConfig,
    target_phase: str,
    target_spec_hash: str,
    target_code_revision: str,
    accepted_by: str,
    rationale: str,
    target_acceptance_policy: str = DEFAULT_SIGNAL_ACCEPTANCE_POLICY,
    source_run_id: str | None = None,
) -> SignalAcceptance:
    """Create a policy-bound, read-only human gate receipt for Insight 2(b)."""

    reviewer = _nonempty_text(accepted_by, label="accepted_by", maximum=200)
    reason = _nonempty_text(rationale, label="rationale", maximum=4000)
    target_acceptance_policy = _validate_target_policy(
        target_phase=target_phase, policy=target_acceptance_policy
    )
    root = approval_root(host)
    destination = destination.expanduser()
    try:
        if root.exists() or root.is_symlink():
            root_metadata = root.lstat()
            if root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
                _fail("host approvals root must be a regular non-symlink directory")
            if root_metadata.st_uid != os.geteuid():
                _fail("host approvals root must be owned by the runner")
        else:
            root.mkdir(mode=0o700, parents=False, exist_ok=False)
        root.chmod(0o700)
    except OSError as error:
        raise SignalAcceptanceError(f"cannot prepare host approvals root: {error}") from error
    if destination.parent != root:
        _fail("signal acceptance output must be a direct child of the host approvals root")
    source: dict[str, Any] | None = None
    if target_acceptance_policy == EXTERNAL_PRIOR_2B_EVAL_ACCEPTANCE_POLICY:
        if source_run_id is not None:
            _fail("external-prior human authorization must not claim a source 2(a) run")
    else:
        if not isinstance(source_run_id, str) or not source_run_id:
            _fail("this signal acceptance policy requires --source-run-id")
        source = inspect_source_2a_run(
            run_root=host.run_root,
            run_id=source_run_id,
            expected_artifact_remote=host.artifact_remote,
            required_phase=(
                "2a-smoke"
                if target_acceptance_policy == EXPLORATORY_SMOKE_ACCEPTANCE_POLICY
                else "2a-eval"
            ),
        )
        if (
            target_acceptance_policy == DEFAULT_SIGNAL_ACCEPTANCE_POLICY
            and source["code_revision"] != target_code_revision
        ):
            _fail("source 2(a) and target 2(b) must use the same published code revision")
    payload = {
        "schema": (
            EXTERNAL_PRIOR_2B_AUTHORIZATION_SCHEMA
            if target_acceptance_policy == EXTERNAL_PRIOR_2B_EVAL_ACCEPTANCE_POLICY
            else SIGNAL_ACCEPTANCE_SCHEMA
        ),
        "decision": "accept",
        "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "accepted_by": reviewer,
        "rationale": reason,
        "accepted_for": {
            "phase": target_phase,
            "code_revision": target_code_revision,
            "manifest_spec_sha256": target_spec_hash,
        },
    }
    if source is not None:
        payload["source_2a"] = source
    if target_acceptance_policy == EXPLORATORY_SMOKE_ACCEPTANCE_POLICY:
        payload["acceptance_policy"] = target_acceptance_policy
        payload["limitations"] = dict(_EXPLORATORY_SMOKE_LIMITATIONS)
    elif target_acceptance_policy == EXTERNAL_PRIOR_2B_EVAL_ACCEPTANCE_POLICY:
        payload["acceptance_policy"] = target_acceptance_policy
        payload["limitations"] = dict(_EXTERNAL_PRIOR_2B_EVAL_LIMITATIONS)
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(destination, flags, 0o400)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        destination.chmod(0o444)
        directory_descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise SignalAcceptanceError(f"cannot create signal acceptance receipt: {error}") from error
    return load_signal_acceptance(
        destination,
        host=host,
        target_phase=target_phase,
        target_spec_hash=target_spec_hash,
        target_code_revision=target_code_revision,
        target_acceptance_policy=target_acceptance_policy,
    )


def _target_binding(manifest_path: Path) -> tuple[str, str, str, str]:
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest)
    if manifest["spec"]["parameters"].get("template_only") is True:
        _fail("target manifest is a non-runnable template")
    if not requires_signal_acceptance(manifest):
        _fail("target manifest is not a gated Insight 2(b) manifest")
    phase = _manifest_phase(manifest)
    repo_root = find_git_root(manifest_path.resolve().parent)
    git = collect_git_metadata(repo_root)
    if git.get("dirty"):
        _fail("acceptance must be created from a clean published checkout")
    revision = str(manifest["spec"]["code"]["revision"])
    process = subprocess.run(
        ["git", "rev-parse", f"{revision}^{{commit}}"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.returncode != 0 or process.stdout.strip() != git.get("sha"):
        _fail("target manifest revision does not resolve to the current checkout")
    return (
        phase,
        manifest_hash(manifest),
        str(git["sha"]),
        signal_acceptance_policy(manifest),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an immutable, policy-bound human gate receipt for Insight 2(b)"
    )
    parser.add_argument(
        "--source-run-id",
        required=False,
        help=(
            "sealed cfexp 2(a) run required by sealed-evidence policies; omit for "
            "the external-prior 2b-eval compute-only policy"
        ),
    )
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--accepted-by", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--host-config",
        type=Path,
        default=Path("~/.config/cfexp/host.toml"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        host = load_host_config(args.host_config)
        if host.artifact_remote != FORMAL_HF_ARTIFACT_REMOTE:
            _fail("acceptance receipts require the formal private Hugging Face dataset")
        phase, spec_hash, revision, acceptance_policy = _target_binding(
            args.target_manifest
        )
        receipt = write_signal_acceptance(
            args.output.expanduser(),
            host=host,
            source_run_id=(
                None if args.source_run_id is None else str(args.source_run_id)
            ),
            target_phase=phase,
            target_spec_hash=spec_hash,
            target_code_revision=revision,
            accepted_by=str(args.accepted_by),
            rationale=str(args.rationale),
            target_acceptance_policy=acceptance_policy,
        )
    except (SignalAcceptanceError, OSError, RuntimeError, ValueError) as error:
        print(f"insight2-signal-acceptance: error: {error}", file=os.sys.stderr)
        return 2
    report = {
        "created": True,
        "path": str(receipt.source_path),
        "sha256": receipt.sha256,
        "acceptance_policy": receipt.acceptance_policy,
        "accepted_for": receipt.payload["accepted_for"],
    }
    source = receipt.payload.get("source_2a")
    if isinstance(source, Mapping):
        report["source_run_id"] = source["run_id"]
    else:
        report["authorization_basis"] = "external_prior_not_revalidated"
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SIGNAL_ACCEPTANCE_PROVENANCE_PATH",
    "SIGNAL_ACCEPTANCE_SCHEMA",
    "DEFAULT_SIGNAL_ACCEPTANCE_POLICY",
    "EXPLORATORY_SMOKE_ACCEPTANCE_POLICY",
    "EXTERNAL_PRIOR_2B_EVAL_ACCEPTANCE_POLICY",
    "EXTERNAL_PRIOR_2B_AUTHORIZATION_PROVENANCE_PATH",
    "EXTERNAL_PRIOR_2B_AUTHORIZATION_SCHEMA",
    "SignalAcceptance",
    "SignalAcceptanceError",
    "approval_root",
    "inspect_source_2a_run",
    "load_signal_acceptance",
    "main",
    "requires_signal_acceptance",
    "signal_acceptance_policy",
    "write_signal_acceptance",
]
