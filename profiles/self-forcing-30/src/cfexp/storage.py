"""Immutable experiment artifact publication.

The storage layer deliberately has a very small contract:

* artifact contents are described by a deterministic SHA-256 index;
* publishing is copy-only and idempotent;
* an existing path may be reused only when its contents are identical; and
* a terminal marker is made visible only after every other file.

``file://`` destinations are implemented directly with the standard library,
``hf://datasets/...`` destinations use the Hugging Face Hub API, and every
other destination is treated as an rclone remote. Remote publication never
invokes sync or delete.
"""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
import re
import stat as stat_module
import subprocess
import tempfile
from contextlib import contextmanager
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import quote, unquote, urlsplit


ARTIFACT_INDEX_NAME = "artifact-index.json"
TERMINAL_MARKERS = frozenset({"_SUCCESS.json", "_FAILED.json"})
VIDEO_SUFFIXES = frozenset(
    {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}
)
HF_VIDEO_PATTERNS = (
    "**/*.[mM][pP]4",
    "**/*.[wW][eE][bB][mM]",
    "**/*.[mM][oO][vV]",
    "**/*.[mM][kK][vV]",
    "**/*.[aA][vV][iI]",
    "**/*.[mM]4[vV]",
    "**/*.[mM][pP][eE][gG]",
    "**/*.[mM][pP][gG]",
    "**/*.[wW][mM][vV]",
    "**/*.[fF][lL][vV]",
)
HF_CONTROL_PATTERNS = ("resolved-manifest.json", "run.json")
_INDEX_VERSION = 1
_HF_COMMIT_BATCH_SIZE = 512
_HF_COMMIT_BATCH_BYTES = 64 * 1024 * 1024
_SHA256_RE = re.compile(r"^([0-9a-fA-F]{64})(?:\s|$)")
_CANONICAL_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HF_ENDPOINT = "https://huggingface.co"
_HF_DISABLE_XET_ENV = "HF_HUB_DISABLE_XET"


class StorageError(RuntimeError):
    """Raised when artifacts cannot be published without breaking immutability."""


def _hf_commit_batch(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Bound atomic commits by count and total source bytes, preserving order.

    The Hub permits 25k LFS files and 1GB of regular-file payload per commit.
    Counting all source bytes conservatively also bounds regular-file encoding
    overhead for grouped files. An oversized individual artifact travels alone
    (large videos/archives use LFS); the server still enforces its own limits.
    No folder uploader may split our compare-and-swap operation implicitly.
    """
    batch: list[Mapping[str, Any]] = []
    total = 0
    for entry in entries:
        size = entry["size"]
        if type(size) is not int or size < 0:
            raise StorageError("invalid indexed artifact size for Hub commit")
        if batch and (len(batch) >= _HF_COMMIT_BATCH_SIZE
                      or total + size > _HF_COMMIT_BATCH_BYTES):
            break
        batch.append(entry)
        total += size
        if total >= _HF_COMMIT_BATCH_BYTES:
            break
    return batch


def configure_hf_publisher_environment() -> None:
    """Pin the Hub publisher to the legacy HTTP transfer implementation.

    This assignment is deliberately unconditional: an ambient value must not
    re-enable Xet for either a formal publication or a reconciliation retry.
    Callers invoke it before the first lazy ``huggingface_hub`` import so the
    SDK's import-time constants observe the pinned value.
    """

    os.environ[_HF_DISABLE_XET_ENV] = "1"


def _safe_hf_error(error: BaseException, token: str | None) -> str:
    """Render a third-party error without ever echoing the active credential."""

    message = str(error) or type(error).__name__
    if token:
        message = message.replace(token, "<redacted>")
    return message


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_root(root: Path | str) -> Path:
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise StorageError(f"artifact root is not a directory: {path}")
    return path


def _validate_relative_path(relative: str) -> None:
    # Newlines make rclone's line-oriented listing ambiguous.  NUL is invalid
    # for local filenames already, but checking it keeps the remote contract
    # explicit.
    # The Hub client normalizes backslashes to slashes, so a literal POSIX
    # backslash cannot be represented as an exact remote path either.
    if any(character in relative for character in ("\x00", "\n", "\r", "\\")):
        raise StorageError(f"unsupported control character in artifact path: {relative!r}")


def _patterns(
    values: Sequence[str] | None, *, default: tuple[str, ...], name: str
) -> tuple[str, ...]:
    if values is None:
        return default
    if isinstance(values, (str, bytes)):
        raise StorageError(f"{name} must be a sequence of glob patterns")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise StorageError(f"{name} must contain non-empty string glob patterns")
        _validate_relative_path(value)
        normalized.append(value)
    return tuple(normalized)


def _matches(relative: str, pattern: str) -> bool:
    """Match a POSIX artifact path with a small, predictable glob dialect.

    ``fnmatch`` treats ``*`` as matching ``/``, which is useful for manifest
    output trees.  A leading ``**/`` is additionally allowed to match zero
    directories, so ``**/*.tmp`` also excludes a root-level ``result.tmp``.
    """

    if fnmatchcase(relative, pattern):
        return True
    return pattern.startswith("**/") and fnmatchcase(relative, pattern[3:])


def _selected(
    relative: str, include_patterns: tuple[str, ...], exclude_patterns: tuple[str, ...]
) -> bool:
    return any(_matches(relative, pattern) for pattern in include_patterns) and not any(
        _matches(relative, pattern) for pattern in exclude_patterns
    )


def _strict_tree_entries(root: Path) -> list[tuple[Path, int]]:
    """Enumerate an artifact tree without links or silently skipped paths."""

    entries: list[tuple[Path, int]] = []

    def scan_error(error: OSError) -> None:
        raise StorageError("could not completely scan the artifact tree") from error

    for current, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=scan_error,
    ):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for name in [*directory_names, *file_names]:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            _validate_relative_path(relative)
            try:
                entry_stat = path.lstat()
            except OSError as error:
                raise StorageError(
                    f"could not inspect artifact path: {relative}"
                ) from error
            if stat_module.S_ISLNK(entry_stat.st_mode):
                raise StorageError(
                    f"symbolic links are not publishable artifacts: {relative}"
                )
            if not (
                stat_module.S_ISDIR(entry_stat.st_mode)
                or stat_module.S_ISREG(entry_stat.st_mode)
            ):
                raise StorageError(f"artifact is not a regular file: {relative}")
            entries.append((path, entry_stat.st_mode))
    return sorted(entries, key=lambda item: item[0].relative_to(root).as_posix())


def _iter_indexed_files(
    root: Path,
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
    force_include_patterns: tuple[str, ...] = (),
) -> Iterable[tuple[str, Path]]:
    for path, mode in _strict_tree_entries(root):
        relative = path.relative_to(root).as_posix()
        if relative == ARTIFACT_INDEX_NAME or relative in TERMINAL_MARKERS:
            continue
        normally_selected = _selected(relative, include_patterns, exclude_patterns)
        force_selected = any(
            _matches(relative, pattern) for pattern in force_include_patterns
        )
        if not normally_selected and not force_selected:
            continue
        if stat_module.S_ISDIR(mode):
            continue
        yield relative, path


def build_artifact_index(
    root: Path | str,
    *,
    include_patterns: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
    force_include_patterns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic content index for files below *root*.

    The index itself and terminal marker files are control objects, so they are
    excluded.  Paths always use POSIX separators and are sorted lexicographically.
    """

    source_root = _source_root(root)
    includes = _patterns(include_patterns, default=("**",), name="include_patterns")
    excludes = _patterns(exclude_patterns, default=(), name="exclude_patterns")
    force_includes = _patterns(
        force_include_patterns,
        default=(),
        name="force_include_patterns",
    )
    files: list[dict[str, Any]] = []
    for relative, path in _iter_indexed_files(
        source_root,
        includes,
        excludes,
        force_includes,
    ):
        file_stat = path.stat()
        files.append(
            {
                "path": relative,
                "size": file_stat.st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": _INDEX_VERSION,
        "algorithm": "sha256",
        "files": files,
    }


def write_artifact_index(
    root: Path | str,
    *,
    include_patterns: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
    force_include_patterns: Sequence[str] | None = None,
) -> Path:
    """Atomically write ``artifact-index.json`` below *root*."""

    source_root = _source_root(root)
    index = build_artifact_index(
        source_root,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        force_include_patterns=force_include_patterns,
    )
    payload = (json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    target = source_root / ARTIFACT_INDEX_NAME

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{ARTIFACT_INDEX_NAME}.", dir=source_root, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def _publication_patterns(
    destination: str | None,
    include_patterns: Sequence[str] | None,
    exclude_patterns: Sequence[str] | None,
) -> tuple[Sequence[str] | None, Sequence[str] | None, Sequence[str] | None]:
    """Return the exact selection policy shared by sealing and publication."""

    if destination is not None and _hf_destination(destination) is not None:
        return (
            include_patterns,
            [*(exclude_patterns or ()), "**/*.part"],
            (*HF_VIDEO_PATTERNS, *HF_CONTROL_PATTERNS),
        )
    return include_patterns, exclude_patterns, None


def _publication_policy(
    destination: str | None,
    include_patterns: Sequence[str] | None,
    exclude_patterns: Sequence[str] | None,
    force_include_patterns: Sequence[str] | None,
) -> dict[str, Any]:
    return {
        # None deliberately means an unbound local-only/test seal. Formal runs
        # always bind the exact immutable HF run prefix selected before launch.
        "destination": destination,
        "include_patterns": list(
            _patterns(include_patterns, default=("**",), name="include_patterns")
        ),
        "exclude_patterns": list(
            _patterns(exclude_patterns, default=(), name="exclude_patterns")
        ),
        "force_include_patterns": list(
            _patterns(
                force_include_patterns,
                default=(),
                name="force_include_patterns",
            )
        ),
    }


@contextmanager
def _exclusive_seal_lock(source_root: Path) -> Iterator[None]:
    """Serialize terminal-marker transactions for one artifact directory."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(source_root, flags)
    except OSError as error:
        raise StorageError(f"could not open artifact root for sealing: {source_root}") from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as error:
            raise StorageError(
                f"could not exclusively lock artifact root for sealing: {source_root}"
            ) from error
        yield
    finally:
        os.close(descriptor)


def _atomic_json_create(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically create a JSON object without replacing an existing path."""

    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise StorageError(f"terminal marker already exists: {path}") from error
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def seal_run_artifacts(
    root: Path | str,
    marker_name: str,
    marker_payload: Mapping[str, Any],
    *,
    destination: str | None,
    include_patterns: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
) -> tuple[Path, Path]:
    """Freeze a completed run index and atomically write its bound marker last.

    A terminal marker is a one-way transition: an existing marker is never
    overwritten or rebound. Hugging Face runs use the same mandatory video and
    partial-file rules here as they do during upload.
    """

    source_root = _source_root(root)
    if marker_name not in TERMINAL_MARKERS:
        allowed = ", ".join(sorted(TERMINAL_MARKERS))
        raise StorageError(f"marker_name must be one of {allowed}; got {marker_name!r}")
    with _exclusive_seal_lock(source_root):
        for candidate in TERMINAL_MARKERS:
            marker_path = source_root / candidate
            if marker_path.exists() or marker_path.is_symlink():
                raise StorageError(f"terminal marker already exists: {marker_path}")

        selected_includes, selected_excludes, force_includes = _publication_patterns(
            destination,
            include_patterns,
            exclude_patterns,
        )
        index_path = write_artifact_index(
            source_root,
            include_patterns=selected_includes,
            exclude_patterns=selected_excludes,
            force_include_patterns=force_includes,
        )
        payload = dict(marker_payload)
        payload["artifact_index_sha256"] = _sha256(index_path)
        payload["artifact_publication"] = _publication_policy(
            destination,
            selected_includes,
            selected_excludes,
            force_includes,
        )
        marker_path = source_root / marker_name
        _atomic_json_create(marker_path, payload)
        return index_path, marker_path


def _validate_index_path(relative: object) -> str:
    if not isinstance(relative, str) or not relative:
        raise StorageError("frozen artifact index path must be a non-empty string")
    _validate_relative_path(relative)
    parts = relative.split("/")
    if relative.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise StorageError(f"unsafe path in frozen artifact index: {relative!r}")
    if relative == ARTIFACT_INDEX_NAME or relative in TERMINAL_MARKERS:
        raise StorageError(f"reserved path in frozen artifact index: {relative!r}")
    return relative


def _load_frozen_index(index_path: Path) -> tuple[dict[str, Any], str]:
    if index_path.is_symlink() or not index_path.is_file():
        raise StorageError(f"frozen artifact index is not a regular file: {index_path}")
    try:
        raw = index_path.read_bytes()
        index = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StorageError(f"could not read frozen artifact index: {error}") from error
    if not isinstance(index, dict) or set(index) != {
        "schema_version",
        "algorithm",
        "files",
    }:
        raise StorageError("frozen artifact index has an invalid top-level schema")
    if type(index["schema_version"]) is not int or index["schema_version"] != _INDEX_VERSION:
        raise StorageError("frozen artifact index has an unsupported schema version")
    if index["algorithm"] != "sha256" or not isinstance(index["files"], list):
        raise StorageError("frozen artifact index has an invalid algorithm or files list")

    paths: list[str] = []
    for entry in index["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise StorageError("frozen artifact index contains an invalid file entry")
        relative = _validate_index_path(entry["path"])
        size = entry["size"]
        sha256 = entry["sha256"]
        if type(size) is not int or size < 0:
            raise StorageError(f"invalid size in frozen artifact index: {relative!r}")
        if not isinstance(sha256, str) or not _CANONICAL_SHA256_RE.fullmatch(sha256):
            raise StorageError(f"invalid SHA-256 in frozen artifact index: {relative!r}")
        paths.append(relative)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise StorageError("frozen artifact index paths must be unique and sorted")
    return index, hashlib.sha256(raw).hexdigest()


def _load_bound_marker(
    source_root: Path,
    marker_name: str,
    expected_index_sha256: str,
    expected_policy: Mapping[str, Any],
) -> dict[str, Any]:
    marker_path = source_root / marker_name
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StorageError(f"could not read terminal marker: {error}") from error
    if not isinstance(marker, dict):
        raise StorageError("terminal marker must contain a JSON object")
    bound_sha256 = marker.get("artifact_index_sha256")
    if not isinstance(bound_sha256, str) or not _CANONICAL_SHA256_RE.fullmatch(
        bound_sha256
    ):
        raise StorageError("terminal marker is not bound to a valid artifact index")
    if bound_sha256 != expected_index_sha256:
        raise StorageError("terminal marker artifact index binding does not match")
    actual_policy = marker.get("artifact_publication")
    if not isinstance(actual_policy, dict) or set(actual_policy) != {
        "destination",
        "include_patterns",
        "exclude_patterns",
        "force_include_patterns",
    }:
        raise StorageError("terminal marker has no valid artifact publication policy")
    policy_to_compare = dict(expected_policy)
    if actual_policy.get("destination") is None:
        policy_to_compare["destination"] = None
    if actual_policy != policy_to_compare:
        raise StorageError("terminal marker artifact publication policy does not match")
    allowed_statuses = (
        {"SUCCEEDED"} if marker_name == "_SUCCESS.json" else {"FAILED", "INTERRUPTED"}
    )
    if marker.get("status") not in allowed_statuses:
        raise StorageError(
            f"terminal marker status does not match {marker_name}: {marker.get('status')!r}"
        )
    if marker.get("run_id") != source_root.name:
        raise StorageError("terminal marker run_id does not match its run directory")
    return marker


def _validate_frozen_tree(
    source_root: Path,
    marker_name: str,
    *,
    destination: str | None,
    include_patterns: Sequence[str] | None,
    exclude_patterns: Sequence[str] | None,
    force_include_patterns: Sequence[str] | None = None,
) -> tuple[dict[str, Any], str]:
    _validate_marker(source_root, marker_name)
    index_path = source_root / ARTIFACT_INDEX_NAME
    index, index_sha256 = _load_frozen_index(index_path)
    policy = _publication_policy(
        destination,
        include_patterns,
        exclude_patterns,
        force_include_patterns,
    )
    _load_bound_marker(source_root, marker_name, index_sha256, policy)
    current = build_artifact_index(
        source_root,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        force_include_patterns=force_include_patterns,
    )
    if current != index:
        raise StorageError(
            "completed run tree no longer matches its frozen artifact index"
        )
    return index, index_sha256


def _planned_files(
    source_root: Path,
    marker_name: str,
    *,
    destination: str | None = None,
    include_patterns: Sequence[str] | None,
    exclude_patterns: Sequence[str] | None,
    force_include_patterns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    index, index_sha256 = _validate_frozen_tree(
        source_root,
        marker_name,
        destination=destination,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        force_include_patterns=force_include_patterns,
    )
    index_path = source_root / ARTIFACT_INDEX_NAME

    planned: list[dict[str, Any]] = []
    for entry in index["files"]:
        planned.append(
            {
                "path": entry["path"],
                "source": source_root / entry["path"],
                "size": entry["size"],
                "sha256": entry["sha256"],
            }
        )

    index_stat = index_path.stat()
    planned.append(
        {
            "path": ARTIFACT_INDEX_NAME,
            "source": index_path,
            "size": index_stat.st_size,
            "sha256": index_sha256,
        }
    )
    planned.sort(key=lambda entry: entry["path"])

    marker_path = source_root / marker_name
    if marker_path.is_symlink() or not marker_path.is_file():
        raise StorageError(f"terminal marker does not exist as a regular file: {marker_path}")
    marker_stat = marker_path.stat()
    planned.append(
        {
            "path": marker_name,
            "source": marker_path,
            "size": marker_stat.st_size,
            "sha256": _sha256(marker_path),
        }
    )
    return planned


def _validate_marker(source_root: Path, marker_name: str) -> None:
    if marker_name not in TERMINAL_MARKERS:
        allowed = ", ".join(sorted(TERMINAL_MARKERS))
        raise StorageError(f"marker_name must be one of {allowed}; got {marker_name!r}")
    selected = source_root / marker_name
    if selected.is_symlink() or not selected.is_file():
        raise StorageError(f"terminal marker does not exist as a regular file: {selected}")
    other_markers = TERMINAL_MARKERS - {marker_name}
    for other in other_markers:
        other_path = source_root / other
        if other_path.exists() or other_path.is_symlink():
            raise StorageError(
                f"source contains conflicting terminal marker {other!r} "
                f"while publishing {marker_name!r}"
            )


def _file_destination(destination: str) -> Path | None:
    if not destination.startswith("file://"):
        return None
    parsed = urlsplit(destination)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise StorageError(f"unsupported file destination: {destination!r}")
    if parsed.query or parsed.fragment:
        raise StorageError("file destinations may not contain a query or fragment")
    path = Path(unquote(parsed.path)).expanduser()
    if not path.is_absolute():
        raise StorageError(f"file destination must be absolute: {destination!r}")
    if any(character in str(path) for character in ("\x00", "\n", "\r")):
        raise StorageError(f"unsupported control character in file destination: {destination!r}")
    # Normalize lexically, but deliberately do not resolve() here.  Resolving
    # would hide a symbolic-link ancestor before we have a chance to reject it.
    normalized = Path(os.path.abspath(os.path.normpath(path)))
    if normalized.parent == normalized:
        raise StorageError("file destination must not be the filesystem root")
    return normalized


def _hf_destination(destination: str) -> tuple[str, str] | None:
    """Parse an immutable Hugging Face dataset destination.

    Destinations use ``hf://datasets/<namespace>/<repo>/<prefix>``. The prefix
    is mandatory so a publication can never accidentally target the repository
    root.
    """

    try:
        parsed = urlsplit(destination)
    except ValueError as error:
        if destination.startswith("hf:"):
            raise StorageError(f"invalid Hugging Face destination: {destination!r}") from error
        return None
    if parsed.scheme != "hf":
        return None
    if parsed.netloc != "datasets" or parsed.query or parsed.fragment:
        raise StorageError(
            "Hugging Face destinations must use "
            "hf://datasets/<namespace>/<repo>/<prefix>"
        )
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 3:
        raise StorageError(
            "Hugging Face destinations require a repository and a non-empty prefix"
        )
    if any(
        part in {".", ".."}
        or any(character in part for character in ("\x00", "\n", "\r", "/", "\\"))
        for part in parts
    ):
        raise StorageError(f"unsafe Hugging Face destination: {destination!r}")
    repo_id = "/".join(parts[:2])
    prefix = "/".join(parts[2:]).strip("/")
    return repo_id, prefix


def _new_hf_api(token: str):
    configure_hf_publisher_environment()
    try:
        from huggingface_hub import HfApi
    except ImportError:  # pragma: no cover - dependency failure
        raise StorageError(
            "huggingface_hub is required for hf:// dataset destinations"
        ) from None
    return HfApi(token=token, endpoint=_HF_ENDPOINT)


def _hf_account_names(whoami: object) -> set[str]:
    if not isinstance(whoami, dict):
        return set()
    names: set[str] = set()
    value = whoami.get("name")
    if isinstance(value, str) and value:
        names.add(value.casefold())
    organizations = whoami.get("orgs", [])
    if isinstance(organizations, list):
        for organization in organizations:
            if not isinstance(organization, dict):
                continue
            # Only the canonical organization slug is an identity claim.
            # displayName/fullname fields are mutable presentation text and
            # must never authorize a token for a namespace.
            value = organization.get("name")
            if isinstance(value, str) and value:
                names.add(value.casefold())
    return names


def _hf_check_write_access(client: object, repo_id: str, *, token: str) -> None:
    """Check dataset write access across pinned Hugging Face Hub versions.

    ``huggingface-hub==0.28.1`` exposes only the read form of
    :meth:`HfApi.auth_check`; newer releases add ``write=True`` by calling the
    same server-side endpoint with a ``/write`` suffix. Keep the public SDK
    path when it is available and reproduce that exact authenticated request
    for the pinned legacy runtime.
    """

    try:
        auth_check = getattr(client, "auth_check")
        parameters = inspect.signature(auth_check).parameters
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Hugging Face client has no inspectable auth_check"
        ) from error

    if "write" in parameters:
        auth_check(repo_id, repo_type="dataset", write=True)
        return

    try:
        from huggingface_hub.utils import (
            build_hf_headers,
            get_session,
            hf_raise_for_status,
        )
    except ImportError as error:  # pragma: no cover - dependency failure
        raise RuntimeError("Hugging Face write-check helpers are unavailable") from error

    repository_parts = repo_id.split("/")
    if len(repository_parts) != 2 or any(not part for part in repository_parts):
        raise RuntimeError("invalid Hugging Face repository id for write check")
    encoded_repo_id = "/".join(quote(part, safe="") for part in repository_parts)
    response = get_session().get(
        f"{_HF_ENDPOINT}/api/datasets/{encoded_repo_id}/auth-check/write",
        headers=build_hf_headers(
            token=token,
            library_name="cfexp",
            is_write_action=True,
        ),
        timeout=20,
    )
    hf_raise_for_status(response)


def _preflight_hf_dataset(
    repo_id: str, *, token: str | None, api: object | None = None
) -> tuple[object, str]:
    configure_hf_publisher_environment()
    if not token:
        raise StorageError(
            "an explicit project-isolated HF_TOKEN is required for Hugging Face publication"
        )
    client = api or _new_hf_api(token)
    namespace = repo_id.split("/", 1)[0]
    try:
        whoami = client.whoami()  # type: ignore[attr-defined]
        account_names = _hf_account_names(whoami)
        if namespace.casefold() not in account_names:
            active = "unknown"
            if isinstance(whoami, dict) and isinstance(whoami.get("name"), str):
                active = str(whoami["name"])
            raise StorageError(
                f"Hugging Face credential belongs to {active!r}, not repository "
                f"namespace {namespace!r}"
            )
        # This is intentionally a write check. A read-only fine-grained token
        # must fail before a long GPU job starts, not during final publication.
        _hf_check_write_access(client, repo_id, token=token)
        info = client.dataset_info(repo_id, expand=["private"], timeout=20)  # type: ignore[attr-defined]
    except StorageError:
        raise
    except Exception as error:
        raise StorageError(
            f"Hugging Face dataset preflight failed for {repo_id}: "
            f"{_safe_hf_error(error, token)}"
        ) from None
    if getattr(info, "private", None) is not True:
        raise StorageError(f"Hugging Face dataset must be private: {repo_id}")
    active_name = "unknown"
    if isinstance(whoami, dict) and isinstance(whoami.get("name"), str):
        active_name = str(whoami["name"])
    return client, active_name


def _hf_repo_head(api: object, repo_id: str, *, token: str) -> str:
    try:
        info = api.dataset_info(repo_id, expand=["private", "sha"], timeout=20)  # type: ignore[attr-defined]
    except Exception as error:
        raise StorageError(
            f"could not read Hugging Face dataset head for {repo_id}: "
            f"{_safe_hf_error(error, token)}"
        ) from None
    if getattr(info, "private", None) is not True:
        raise StorageError(f"Hugging Face dataset must remain private: {repo_id}")
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise StorageError(f"Hugging Face dataset returned an invalid head commit: {sha!r}")
    return sha.lower()


def _hf_remote_files(
    api: object,
    repo_id: str,
    prefix: str,
    *,
    token: str,
    revision: str | None = None,
) -> dict[str, object]:
    try:
        try:
            tree = api.list_repo_tree(  # type: ignore[attr-defined]
                repo_id=repo_id,
                path_in_repo=prefix,
                recursive=True,
                # Size, Git blob ID and LFS SHA-256 are present without
                # expansion. Last-commit/security metadata is not used by
                # verification and cuts Hub pagination from 1000 to 50.
                expand=False,
                revision=revision,
                repo_type="dataset",
            )
        except TypeError:  # pragma: no cover - compatibility with older clients
            tree = api.list_repo_tree(  # type: ignore[attr-defined]
                repo_id=repo_id,
                path_in_repo=prefix,
                recursive=True,
                revision=revision,
                repo_type="dataset",
            )
        files: dict[str, object] = {}
        prefix_with_slash = prefix.rstrip("/") + "/"
        for item in tree:
            path = getattr(item, "path", None)
            size = getattr(item, "size", None)
            if not isinstance(path, str) or not isinstance(size, int):
                continue
            if not path.startswith(prefix_with_slash):
                raise StorageError(f"Hugging Face returned a path outside prefix: {path}")
            relative = path[len(prefix_with_slash) :]
            _validate_relative_path(relative)
            if relative:
                files[relative] = item
        return files
    except StorageError:
        raise
    except Exception as error:
        try:
            from huggingface_hub.errors import EntryNotFoundError
        except ImportError:  # pragma: no cover - dependency failure
            EntryNotFoundError = ()  # type: ignore[assignment,misc]
        if isinstance(error, EntryNotFoundError):
            return {}
        raise StorageError(
            f"could not list Hugging Face dataset prefix {repo_id}/{prefix}: "
            f"{_safe_hf_error(error, token)}"
        ) from None


def _hf_lfs_sha256(item: object) -> str | None:
    lfs = getattr(item, "lfs", None)
    if isinstance(lfs, dict):
        value = lfs.get("sha256")
    else:
        value = getattr(lfs, "sha256", None)
    if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
        return value.lower()
    return None


def _git_blob_sha1(path: Path, size: int) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hf_object_hash(item: object, entry: dict[str, Any], remote_name: str) -> None:
    lfs_sha256 = _hf_lfs_sha256(item)
    if lfs_sha256 is not None:
        if lfs_sha256 != entry["sha256"]:
            raise StorageError(f"Hugging Face SHA-256 mismatch for {remote_name}")
        return
    blob_id = getattr(item, "blob_id", None)
    if isinstance(blob_id, str) and re.fullmatch(r"[0-9a-fA-F]{40}", blob_id):
        local_blob_id = _git_blob_sha1(Path(entry["source"]), int(entry["size"]))
        if local_blob_id != blob_id.lower():
            raise StorageError(f"Hugging Face Git blob mismatch for {remote_name}")
        return
    raise StorageError(f"Hugging Face returned no verifiable hash metadata for {remote_name}")


def _hf_download_bytes(
    repo_id: str,
    remote_path: str,
    *,
    token: str,
    revision: str | None = None,
) -> bytes:
    try:
        from huggingface_hub import hf_hub_download

        with tempfile.TemporaryDirectory(prefix="cfexp-hf-verify-") as directory:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                token=token,
                endpoint=_HF_ENDPOINT,
                revision=revision,
                local_dir=directory,
                force_download=True,
            )
            return Path(local_path).read_bytes()
    except Exception as error:
        raise StorageError(
            f"could not verify Hugging Face object {repo_id}/{remote_path}: "
            f"{_safe_hf_error(error, token)}"
        ) from None


def _verify_hf_files(
    api: object,
    repo_id: str,
    prefix: str,
    expected: dict[str, dict[str, Any]],
    *,
    token: str,
    allow_missing_marker: str | None = None,
    revision: str | None = None,
) -> dict[str, object]:
    remote = _hf_remote_files(
        api,
        repo_id,
        prefix,
        token=token,
        revision=revision,
    )
    expected_paths = set(expected)
    if allow_missing_marker and allow_missing_marker not in remote:
        expected_paths.remove(allow_missing_marker)
    actual_paths = set(remote)
    unexpected = sorted(actual_paths - expected_paths)
    missing = sorted(expected_paths - actual_paths)
    if unexpected or missing:
        raise StorageError(
            f"Hugging Face artifact set mismatch; unexpected={unexpected}, missing={missing}"
        )
    for relative in sorted(actual_paths):
        entry = expected[relative]
        item = remote[relative]
        if getattr(item, "size", None) != entry["size"]:
            raise StorageError(
                f"Hugging Face size mismatch for {repo_id}/{prefix}/{relative}"
            )
        _verify_hf_object_hash(item, entry, f"{repo_id}/{prefix}/{relative}")
    return remote


def _publish_hf(
    destination: str,
    repo_id: str,
    prefix: str,
    planned: list[dict[str, Any]],
    marker_name: str,
    *,
    token: str | None,
    validate_source: Callable[[], None],
) -> tuple[int, int]:
    api, _ = _preflight_hf_dataset(repo_id, token=token)
    assert token is not None  # _preflight_hf_dataset rejects implicit SDK credentials.
    expected = {entry["path"]: entry for entry in planned}
    source_root = Path(expected[marker_name]["source"]).parent
    index_remote_path = f"{prefix}/{ARTIFACT_INDEX_NAME}"
    non_marker = [entry for entry in planned if entry["path"] != marker_name]
    local_index = Path(expected[ARTIFACT_INDEX_NAME]["source"]).read_bytes()
    marker_path = Path(expected[marker_name]["source"])
    initial_paths: set[str] | None = None
    last_race: Exception | None = None

    # A Hub commit is optimistic-concurrency controlled with parent_commit.
    # This loop also makes an interrupted response recoverable: if the server
    # accepted a commit but the client did not receive the response, the next
    # immutable snapshot observes and verifies the newly present objects.
    for _attempt in range(max(8, len(non_marker) * 3 + 8)):
        head = _hf_repo_head(api, repo_id, token=token)
        remote = _hf_remote_files(
            api,
            repo_id,
            prefix,
            token=token,
            revision=head,
        )
        if initial_paths is None:
            initial_paths = set(remote)

        existing_markers = set(remote) & TERMINAL_MARKERS
        if existing_markers - {marker_name}:
            raise StorageError(
                "destination is already finalized with "
                + ", ".join(sorted(existing_markers - {marker_name}))
            )
        unexpected = sorted(set(remote) - set(expected))
        if unexpected:
            raise StorageError(f"Hugging Face prefix contains unexpected files: {unexpected}")
        for relative, item in remote.items():
            entry = expected[relative]
            if getattr(item, "size", None) != entry["size"]:
                raise StorageError(
                    f"immutable destination collision: {repo_id}/{prefix}/{relative}"
                )
            try:
                _verify_hf_object_hash(item, entry, f"{repo_id}/{prefix}/{relative}")
            except StorageError as error:
                raise StorageError(
                    f"immutable destination collision: {repo_id}/{prefix}/{relative}"
                ) from error

        if ARTIFACT_INDEX_NAME in remote:
            remote_index = _hf_download_bytes(
                repo_id,
                index_remote_path,
                token=token,
                revision=head,
            )
            if remote_index != local_index:
                raise StorageError(
                    f"immutable destination collision: {repo_id}/{index_remote_path}"
                )

        if marker_name in remote:
            validate_source()
            remote_marker = _hf_download_bytes(
                repo_id,
                f"{prefix}/{marker_name}",
                token=token,
                revision=head,
            )
            if remote_marker != marker_path.read_bytes():
                raise StorageError(
                    f"immutable destination collision: {repo_id}/{prefix}/{marker_name}"
                )
            _verify_hf_files(
                api,
                repo_id,
                prefix,
                expected,
                token=token,
                revision=head,
            )
            assert initial_paths is not None
            skipped = len(initial_paths)
            return len(planned) - skipped, skipped

        missing = [entry for entry in non_marker if entry["path"] not in remote]
        if missing:
            # ``create_commit`` applies this bounded group atomically against
            # one parent SHA.  That preserves the compare-and-swap contract
            # without consuming one Hub repository commit per artifact.  In
            # contrast, ``upload_folder`` may silently split a directory into
            # several commits, with the parent applying only to the first.
            batch = _hf_commit_batch(missing)
            for entry in batch:
                source_sha256 = _sha256(Path(entry["source"]))
                if source_sha256 != entry["sha256"]:
                    raise StorageError(
                        f"source changed while publishing {entry['source']}: "
                        f"expected {entry['sha256']}, got {source_sha256}"
                    )
            try:
                from huggingface_hub import CommitOperationAdd

                operations = [
                    CommitOperationAdd(
                        path_in_repo=f"{prefix}/{entry['path']}",
                        path_or_fileobj=str(entry["source"]),
                    )
                    for entry in batch
                ]
                api.create_commit(  # type: ignore[attr-defined]
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=operations,
                    parent_commit=head,
                    num_threads=1,
                    commit_message=(
                        f"Upload immutable run {source_root.name}: "
                        f"{len(batch)} artifact(s)"
                    ),
                )
            except Exception as error:
                # Retry only if the repository head moved. That covers both a
                # genuine parent race and an accepted commit whose response was
                # lost, without retrying ordinary auth/network failures.
                try:
                    current_head = _hf_repo_head(api, repo_id, token=token)
                except StorageError:
                    current_head = head
                if current_head == head:
                    raise StorageError(
                        f"Hugging Face upload failed for {destination}: "
                        f"{_safe_hf_error(error, token)}"
                    ) from None
                last_race = error
            continue

        _verify_hf_files(
            api,
            repo_id,
            prefix,
            expected,
            token=token,
            allow_missing_marker=marker_name,
            revision=head,
        )
        remote_index = _hf_download_bytes(
            repo_id,
            index_remote_path,
            token=token,
            revision=head,
        )
        if remote_index != local_index:
            raise StorageError(
                f"Hugging Face index verification failed: {repo_id}/{index_remote_path}"
            )
        if _sha256(marker_path) != expected[marker_name]["sha256"]:
            raise StorageError(f"source changed while publishing {marker_path}")
        validate_source()
        try:
            api.upload_file(  # type: ignore[attr-defined]
                repo_id=repo_id,
                repo_type="dataset",
                path_or_fileobj=str(marker_path),
                path_in_repo=f"{prefix}/{marker_name}",
                parent_commit=head,
                commit_message=f"Finalize immutable run {source_root.name}",
            )
        except Exception as error:
            try:
                current_head = _hf_repo_head(api, repo_id, token=token)
            except StorageError:
                current_head = head
            if current_head == head:
                raise StorageError(
                    f"Hugging Face final marker upload failed for {destination}: "
                    f"{_safe_hf_error(error, token)}"
                ) from None
            last_race = error
        # Always re-read an immutable snapshot and verify the marker rather
        # than trusting a successful or ambiguous client response.

    detail = f": {_safe_hf_error(last_race, token)}" if last_race else ""
    raise StorageError(
        f"Hugging Face publication kept racing with another writer for {destination}{detail}"
    )


def _path_prefixes(path: Path) -> Iterable[Path]:
    yield from reversed(path.parents)
    yield path


def _reject_symlink_ancestors(path: Path) -> None:
    """Reject every existing symlink component in an absolute local path."""

    for candidate in _path_prefixes(path):
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as error:
            raise StorageError(
                f"cannot inspect file destination component {candidate}: {error}"
            ) from error
        if stat_module.S_ISLNK(mode):
            raise StorageError(
                f"file destination traverses symbolic link component: {candidate}"
            )


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path
    while True:
        if candidate.exists():
            return candidate
        if candidate.is_symlink():  # includes broken links
            raise StorageError(
                f"file destination traverses symbolic link component: {candidate}"
            )
        parent = candidate.parent
        if parent == candidate:  # pragma: no cover - an absolute root always exists
            raise StorageError(f"file destination has no existing ancestor: {path}")
        candidate = parent


def _preflight_local_destination(destination: Path) -> None:
    _reject_symlink_ancestors(destination)
    nearest = _nearest_existing_ancestor(destination)
    if destination.exists() and not destination.is_dir():
        raise StorageError(f"file destination is not a directory: {destination}")
    if not nearest.is_dir():
        raise StorageError(
            f"nearest existing file destination ancestor is not a directory: {nearest}"
        )
    if not os.access(nearest, os.W_OK | os.X_OK):
        raise StorageError(f"file destination ancestor is not writable: {nearest}")


def _list_local_files(destination: Path) -> set[str]:
    """Return regular files below *destination*, rejecting unsafe objects."""

    def scan_error(error: OSError) -> None:
        raise StorageError(
            "could not completely scan the file destination"
        ) from error

    files: set[str] = set()
    for directory, dirnames, filenames in os.walk(
        destination,
        topdown=True,
        followlinks=False,
        onerror=scan_error,
    ):
        base = Path(directory)
        try:
            base_stat = base.lstat()
        except OSError as error:
            raise StorageError(
                "could not inspect a file destination directory"
            ) from error
        if not stat_module.S_ISDIR(base_stat.st_mode):
            relative = base.relative_to(destination).as_posix()
            raise StorageError(
                f"destination contains a non-directory tree entry: {relative}"
            )
        dirnames.sort()
        filenames.sort()
        for name in dirnames:
            path = base / name
            relative = path.relative_to(destination).as_posix()
            _validate_relative_path(relative)
            try:
                entry_stat = path.lstat()
            except OSError as error:
                raise StorageError(
                    f"could not inspect file destination path: {relative}"
                ) from error
            if not stat_module.S_ISDIR(entry_stat.st_mode):
                raise StorageError(
                    f"destination contains a non-directory tree entry: {relative}"
                )
        for name in filenames:
            path = base / name
            relative = path.relative_to(destination).as_posix()
            _validate_relative_path(relative)
            try:
                entry_stat = path.lstat()
            except OSError as error:
                raise StorageError(
                    f"could not inspect file destination path: {relative}"
                ) from error
            if not stat_module.S_ISREG(entry_stat.st_mode):
                raise StorageError(
                    f"destination contains a non-regular file: {relative}"
                )
            files.add(relative)
    return files


def _reject_unexpected_files(existing: set[str], expected: set[str]) -> None:
    unexpected = sorted(existing - expected)
    if unexpected:
        raise StorageError(
            "destination contains unexpected artifact(s): " + ", ".join(unexpected)
        )


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        common = Path(os.path.commonpath((str(first), str(second))))
    except ValueError:
        return False
    return common == first or common == second


def _install_local_file(source: Path, target: Path, expected_sha256: str) -> None:
    """Copy and exclusively install one object after checking its content."""

    _reject_symlink_ancestors(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(target.parent)
    temporary: Path | None = None
    digest = hashlib.sha256()
    try:
        with source.open("rb") as source_handle, tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", dir=target.parent, delete=False
        ) as target_handle:
            temporary = Path(target_handle.name)
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                digest.update(chunk)
                target_handle.write(chunk)
            target_handle.flush()
            os.fsync(target_handle.fileno())

        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise StorageError(
                f"source changed while publishing {source}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        _reject_symlink_ancestors(target.parent)
        try:
            # A hard link gives us atomic create-if-absent semantics: unlike
            # os.replace(), it can never overwrite an object won by a race.
            os.link(temporary, target)
        except FileExistsError:
            if not target.is_file() or target.is_symlink() or _sha256(target) != expected_sha256:
                raise StorageError(f"immutable destination collision: {target}")
        temporary.unlink(missing_ok=True)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _copy_local_immutable(
    source: Path, target: Path, expected_sha256: str, expected_size: int
) -> bool:
    """Copy one local object, returning True when a new object was installed."""

    _reject_symlink_ancestors(target.parent)
    if target.exists() or target.is_symlink():
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_size != expected_size
            or _sha256(target) != expected_sha256
        ):
            raise StorageError(f"immutable destination collision: {target}")
        return False
    _install_local_file(source, target, expected_sha256)
    return True


def _publish_local(
    source_root: Path,
    destination_root: Path,
    planned: list[dict[str, Any]],
    marker_name: str,
    *,
    validate_source: Callable[[], None],
) -> tuple[int, int]:
    if _paths_overlap(source_root, destination_root):
        raise StorageError("source and file destination must not overlap")

    _preflight_local_destination(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(destination_root)
    existing_markers = {
        marker
        for marker in TERMINAL_MARKERS
        if (destination_root / marker).exists()
        or (destination_root / marker).is_symlink()
    }
    if existing_markers - {marker_name}:
        raise StorageError(
            "destination is already finalized with "
            + ", ".join(sorted(existing_markers - {marker_name}))
        )

    existing = _list_local_files(destination_root)
    expected = {entry["path"] for entry in planned}
    _reject_unexpected_files(existing, expected)

    marker_exists = marker_name in existing_markers
    # Preflight all collisions before writing anything.  This makes a failed
    # attempt leave the destination exactly as it was.
    for entry in planned:
        target = destination_root / entry["path"]
        _reject_symlink_ancestors(target.parent)
        if target.exists() or target.is_symlink():
            if (
                target.is_symlink()
                or not target.is_file()
                or target.stat().st_size != entry["size"]
                or _sha256(target) != entry["sha256"]
            ):
                raise StorageError(f"immutable destination collision: {target}")
        elif marker_exists:
            raise StorageError(
                f"finalized destination is missing expected artifact: {entry['path']}"
            )

    uploaded = 0
    skipped = 0
    for entry in planned:
        if entry["path"] == marker_name:
            continue
        target = destination_root / entry["path"]
        if _copy_local_immutable(
            entry["source"], target, entry["sha256"], entry["size"]
        ):
            uploaded += 1
        else:
            skipped += 1
    validate_source()
    marker_entry = next(entry for entry in planned if entry["path"] == marker_name)
    marker_target = destination_root / marker_name
    if _copy_local_immutable(
        marker_entry["source"],
        marker_target,
        marker_entry["sha256"],
        marker_entry["size"],
    ):
        uploaded += 1
    else:
        skipped += 1
    return uploaded, skipped


def _run_rclone(
    arguments: list[str], *, missing_ok: bool = False
) -> subprocess.CompletedProcess[str]:
    command = ["rclone", *arguments]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise StorageError("rclone is required for non-file destinations") from error
    if result.returncode != 0 and not (missing_ok and result.returncode in (3, 4)):
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise StorageError(f"rclone failed ({result.returncode}): {detail}")
    return result


def preflight_destination(
    destination: str, *, hf_token: str | None = None
) -> dict[str, str]:
    """Check that a destination is usable without creating or changing data.

    A missing final prefix is valid for both local filesystems and object
    stores.  Authentication, configuration, permission, and unsafe local-path
    failures are reported before an expensive experiment starts.
    """

    if not destination or not destination.strip():
        raise StorageError(
            "destination must be a file:// URI, hf:// dataset URI, or an rclone remote"
        )
    hf_destination = _hf_destination(destination)
    if hf_destination is not None:
        repo_id, prefix = hf_destination
        _, account = _preflight_hf_dataset(repo_id, token=hf_token)
        return {
            "destination": destination,
            "backend": "huggingface",
            "repo_id": repo_id,
            "prefix": prefix,
            "account": account,
        }
    local_destination = _file_destination(destination)
    if local_destination is not None:
        _preflight_local_destination(local_destination)
        return {"destination": destination, "backend": "file"}
    if ":" not in destination:
        raise StorageError(
            "non-file destination must be an rclone remote such as remote:bucket/path"
        )
    _run_rclone(["lsf", "--max-depth", "1", destination], missing_ok=True)
    return {"destination": destination, "backend": "rclone"}


def _remote_join(destination: str, relative: str) -> str:
    if destination.endswith(("/", ":")):
        return destination + relative
    return destination + "/" + relative


def _list_remote_files(destination: str) -> set[str]:
    result = _run_rclone(
        ["lsf", "--recursive", "--files-only", "--format", "p", destination],
        missing_ok=True,
    )
    if result.returncode in (3, 4):
        return set()
    files = set()
    for line in result.stdout.splitlines():
        relative = line.rstrip("/")
        _validate_relative_path(relative)
        if relative:
            files.add(relative)
    return files


def _remote_sha256(target: str) -> str:
    # Most backends can answer this without transferring the object.
    try:
        result = _run_rclone(["hashsum", "SHA-256", target], missing_ok=False)
    except StorageError:
        result = None
    if result is not None:
        for line in result.stdout.splitlines():
            match = _SHA256_RE.match(line)
            if match:
                return match.group(1).lower()

    # Some object stores expose only MD5/ETag.  Stream the object through rclone
    # as an exact, backend-independent fallback rather than weakening the
    # immutable comparison to size or modification time.
    digest = hashlib.sha256()
    with tempfile.TemporaryFile(mode="w+b") as error_output:
        try:
            process = subprocess.Popen(
                ["rclone", "cat", target],
                stdout=subprocess.PIPE,
                stderr=error_output,
            )
        except FileNotFoundError as error:
            raise StorageError("rclone is required for non-file destinations") from error
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
            digest.update(chunk)
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            error_output.seek(0)
            detail = error_output.read().decode("utf-8", errors="replace").strip()
            raise StorageError(
                f"rclone failed while hashing {target!r} ({return_code}): "
                f"{detail or 'no diagnostic output'}"
            )
    return digest.hexdigest()


def _publish_remote(
    destination: str,
    planned: list[dict[str, Any]],
    marker_name: str,
    *,
    validate_source: Callable[[], None],
) -> tuple[int, int]:
    existing = _list_remote_files(destination)
    existing_markers = existing & TERMINAL_MARKERS
    if existing_markers - {marker_name}:
        raise StorageError(
            "destination is already finalized with "
            + ", ".join(sorted(existing_markers - {marker_name}))
        )

    marker_exists = marker_name in existing_markers
    expected_by_path = {entry["path"]: entry for entry in planned}
    _reject_unexpected_files(existing, set(expected_by_path))
    # Check every collision before starting transfers.
    for relative, entry in expected_by_path.items():
        if relative in existing:
            actual = _remote_sha256(_remote_join(destination, relative))
            if actual != entry["sha256"]:
                raise StorageError(
                    f"immutable destination collision: "
                    f"{_remote_join(destination, relative)}"
                )
        elif marker_exists:
            raise StorageError(
                f"finalized destination is missing expected artifact: {relative}"
            )

    uploaded = 0
    skipped = 0
    for entry in planned:
        relative = entry["path"]
        if relative == marker_name:
            continue
        if relative in existing:
            skipped += 1
            continue
        source_sha256 = _sha256(entry["source"])
        if source_sha256 != entry["sha256"]:
            raise StorageError(
                f"source changed while publishing {entry['source']}: "
                f"expected {entry['sha256']}, got {source_sha256}"
            )
        target = _remote_join(destination, relative)
        _run_rclone(
            [
                "copyto",
                "--immutable",
                "--checksum",
                str(entry["source"]),
                target,
            ]
        )
        remote_sha256 = _remote_sha256(target)
        if remote_sha256 != entry["sha256"]:
            raise StorageError(
                f"remote checksum mismatch after publishing {target}: "
                f"expected {entry['sha256']}, got {remote_sha256}"
            )
        uploaded += 1
    validate_source()
    marker_entry = expected_by_path[marker_name]
    if marker_name in existing:
        skipped += 1
    else:
        if _sha256(marker_entry["source"]) != marker_entry["sha256"]:
            raise StorageError(
                f"source changed while publishing {marker_entry['source']}"
            )
        marker_target = _remote_join(destination, marker_name)
        _run_rclone(
            [
                "copyto",
                "--immutable",
                "--checksum",
                str(marker_entry["source"]),
                marker_target,
            ]
        )
        if _remote_sha256(marker_target) != marker_entry["sha256"]:
            raise StorageError(
                f"remote checksum mismatch after publishing {marker_target}"
            )
        uploaded += 1
    return uploaded, skipped


def publish_run(
    source_dir: Path,
    destination: str,
    marker_name: str,
    *,
    include_patterns: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Publish one completed run/attempt to an immutable destination.

    ``destination`` is the already-expanded run/attempt URI. ``file://`` URIs
    use the local implementation, ``hf://datasets/...`` uses the private Hub
    dataset implementation, and other strings are handed to rclone. The
    selected terminal marker must exist in ``source_dir`` and is always the
    final object considered for publication.
    """

    if not destination or not destination.strip():
        raise StorageError(
            "destination must be a file:// URI, hf:// dataset URI, or an rclone remote"
        )
    source_root = _source_root(source_dir)
    _validate_marker(source_root, marker_name)
    hf_destination = _hf_destination(destination)
    local_destination = _file_destination(destination)
    if hf_destination is None and local_destination is None and ":" not in destination:
        raise StorageError(
            "non-file destination must be an rclone remote such as remote:bucket/path"
        )
    if local_destination is not None and _paths_overlap(source_root, local_destination):
        raise StorageError("source and file destination must not overlap")
    selected_includes, selected_excludes, force_includes = _publication_patterns(
        destination,
        include_patterns,
        exclude_patterns,
    )
    def validate_source() -> None:
        _validate_frozen_tree(
            source_root,
            marker_name,
            destination=destination,
            include_patterns=selected_includes,
            exclude_patterns=selected_excludes,
            force_include_patterns=force_includes,
        )
    planned = _planned_files(
        source_root,
        marker_name,
        destination=destination,
        include_patterns=selected_includes,
        exclude_patterns=selected_excludes,
        force_include_patterns=force_includes,
    )
    if hf_destination is not None:
        repo_id, prefix = hf_destination
        uploaded, skipped = _publish_hf(
            destination,
            repo_id,
            prefix,
            planned,
            marker_name,
            token=hf_token,
            validate_source=validate_source,
        )
        backend = "huggingface"
    elif local_destination is not None:
        uploaded, skipped = _publish_local(
            source_root,
            local_destination,
            planned,
            marker_name,
            validate_source=validate_source,
        )
        backend = "file"
    else:
        uploaded, skipped = _publish_remote(
            destination,
            planned,
            marker_name,
            validate_source=validate_source,
        )
        backend = "rclone"

    return {
        "destination": destination,
        "backend": backend,
        "marker": marker_name,
        "files": len(planned),
        "uploaded": uploaded,
        "skipped": skipped,
        "bytes": sum(entry["size"] for entry in planned),
    }


__all__ = [
    "ARTIFACT_INDEX_NAME",
    "StorageError",
    "VIDEO_SUFFIXES",
    "build_artifact_index",
    "configure_hf_publisher_environment",
    "preflight_destination",
    "publish_run",
    "seal_run_artifacts",
    "write_artifact_index",
]
