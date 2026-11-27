"""Auditable wrapper around the pinned official VBench-Long evaluator.

This module does not implement or approximate a video metric.  It validates a
clean checkout of the pinned upstream source, invokes that source in
``long_custom_input`` mode, and accepts scores only from its result JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence


VBENCH_LONG_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
VBENCH_LONG_REPOSITORY = "https://github.com/Vchitect/VBench"
VBENCH_LONG_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
MEASUREMENT_SCHEMA = "cfexp_official_vbench_long_measurement_v2"
ATTEMPT_SCHEMA = "cfexp_official_vbench_long_attempt_v1"
CONFIG_SCHEMA = "cfexp_official_vbench_long_config_v3"
BATCH_MEASUREMENT_SCHEMA = "cfexp_official_vbench_long_batch_measurement_v1"
BATCH_ATTEMPT_SCHEMA = "cfexp_official_vbench_long_batch_attempt_v1"
BATCH_CONFIG_SCHEMA = "cfexp_official_vbench_long_batch_config_v1"
BATCH_ROOT_RELATIVE = "vbench-long/_batches"
OFFLINE_ASSET_SCHEMA = "cfexp_vbench_long_offline_assets_v1"
OFFLINE_TREE_HASH_ALGORITHM = "sha256_path_nul_size_nul_file_sha256_lf_v1"
CHECKOUT_TREE_HASH_ALGORITHM = (
    "sha256_path_nul_mode_nul_git_blob_nul_size_nul_file_sha256_lf_v1"
)
STAGED_ALIAS_ALGORITHM = (
    "sha256_canonical_json_prompt_id_measurement_id_source_video_v1"
)
OFFLINE_NETWORK_POLICY = "python_inet_fail_closed_bound_ffmpeg_probe_allowlist_v3"
# Every scorer is a single-process worker. Port zero lets TCPStore bind an
# OS-assigned port atomically, without a probe/close/bind race between scorers.
VBENCH_DISTRIBUTED_ENVIRONMENT = {
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "0",
    "RANK": "0",
    "LOCAL_RANK": "0",
    "WORLD_SIZE": "1",
    "TORCHELASTIC_USE_AGENT_STORE": "False",
}
VBENCH_LONG_SUBJECT_CLIP2CLIP_EXTRACTOR = "dinov2"
VBENCH_LONG_BACKGROUND_CLIP2CLIP_EXTRACTOR = "dreamsim"
VBENCH_LONG_CREDENTIAL_ENVIRONMENT_VARIABLES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_TOKEN_PATH",
    "HF_STORED_TOKENS_PATH",
)
VBENCH_LONG_SCRUBBED_ENVIRONMENT_VARIABLES = (
    *VBENCH_LONG_CREDENTIAL_ENVIRONMENT_VARIABLES,
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HUGGING_FACE_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "HF_ASSETS_CACHE",
    "HUGGINGFACE_ASSETS_CACHE",
    "HF_MODULES_CACHE",
    "TRANSFORMERS_CACHE",
    "DIFFUSERS_CACHE",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "HF_ENDPOINT",
    "HUGGINGFACE_CO_RESOLVE_ENDPOINT",
    "PYTHONPYCACHEPREFIX",
)

VBENCH_LONG_OFFLINE_FILE_PATHS = {
    "dino_v1_weights": "vbench-cache/dino_model/dino_vitbase16_pretrain.pth",
    "dinov2_weights": "checkpoints/dinov2_vitb14_pretrain.pth",
    "dreamsim_dino_weights": "home/.cache/dino_vitb16_pretrain.pth",
    "dreamsim_clip_weights": "home/.cache/clip_vitb16_pretrain.pth.tar",
    "dreamsim_open_clip_weights": "home/.cache/open_clip_vitb16_pretrain.pth.tar",
    "dreamsim_adapter_config": "home/.cache/ensemble_lora/adapter_config.json",
    "dreamsim_adapter_weights": "home/.cache/ensemble_lora/adapter_model.safetensors",
    "clip_vit_b32_weights": "vbench-cache/clip_model/ViT-B-32.pt",
    "clip_vit_l14_weights": "vbench-cache/clip_model/ViT-L-14.pt",
    "amt_weights": "vbench-cache/amt_model/amt-s.pth",
    "raft_weights": "vbench-cache/raft_model/models/raft-things.pth",
    "aesthetic_weights": (
        "vbench-cache/aesthetic_model/emb_reader/"
        "sa_0_4_vit_l_14_linear.pth"
    ),
    "musiq_spaq_weights": (
        "vbench-cache/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
    ),
}
VBENCH_LONG_OFFLINE_TREE_PATHS = {
    "dino_v1_source": "vbench-cache/dino_model/facebookresearch_dino_main",
    "dinov2_source": "model-sources/facebookresearch_dinov2_main",
}
_MEASUREMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_PROMPT_SUFFIX_RE = re.compile(r"(?:-Scene-\d+|-\d+)(?:_\d+)?$")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_STAGED_ALIAS_RE = re.compile(r"cfvb-[0-9a-f]{64}")
_BATCH_ID_RE = re.compile(r"batch-[0-9a-f]{64}")


class VBenchLongError(RuntimeError):
    """Raised when official VBench-Long cannot produce valid evidence."""


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VBenchLongError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _portable_relative_path(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} path is invalid")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and value == path.as_posix()
        and not any(part in {"", ".", ".."} for part in path.parts),
        f"{label} path is not a portable relative path",
    )
    return value


def _tree_binding(root: Path) -> dict[str, Any]:
    """Hash all source files that can affect a local torch-hub import."""

    _require(root.is_dir() and not root.is_symlink(), f"asset tree is invalid: {root}")
    records: list[tuple[str, int, str]] = []
    total_size = 0
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if candidate.is_symlink():
            raise VBenchLongError(f"asset tree contains a symlink: {candidate}")
        if any(part == ".git" for part in relative.parts):
            continue
        if any(part == "__pycache__" for part in relative.parts) or candidate.suffix == ".pyc":
            raise VBenchLongError(
                f"asset tree contains executable Python bytecode: {candidate}"
            )
        if candidate.is_dir():
            continue
        _require(candidate.is_file(), f"asset tree contains a special file: {candidate}")
        sha256, nbytes = _sha256_file(candidate)
        records.append((relative.as_posix(), nbytes, sha256))
        total_size += nbytes
    _require(bool(records), f"asset tree contains no source files: {root}")
    digest = hashlib.sha256()
    for relative, nbytes, sha256 in sorted(records):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(nbytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(records),
        "nbytes": total_size,
        "algorithm": OFFLINE_TREE_HASH_ALGORITHM,
    }


def vbench_long_staged_alias(
    *, prompt_id: str, measurement_id: str, source_video_sha256: str
) -> str:
    """Return the deterministic underscore-free group name seen by VBench.

    Frozen upstream derives clip2clip fusion keys with ``basename.split('_')``.
    Passing an experiment prompt id through as the filename can therefore
    silently zero the cross-clip term.  The evaluator only sees this bound
    opaque alias; evidence retains the original prompt id.
    """

    _require(isinstance(prompt_id, str) and bool(prompt_id), "prompt_id is empty")
    _require(
        isinstance(measurement_id, str)
        and _MEASUREMENT_RE.fullmatch(measurement_id) is not None,
        "measurement_id is not portable",
    )
    _require(
        isinstance(source_video_sha256, str)
        and _SHA256_RE.fullmatch(source_video_sha256) is not None,
        "source video SHA-256 is invalid",
    )
    digest = hashlib.sha256(
        _canonical_bytes(
            {
                "measurement_id": measurement_id,
                "prompt_id": prompt_id,
                "source_video_sha256": source_video_sha256,
            }
        )
    ).hexdigest()
    alias = f"cfvb-{digest}"
    _require(
        _STAGED_ALIAS_RE.fullmatch(alias) is not None and "_" not in alias,
        "internal VBench staged alias is invalid",
    )
    _require(
        _PROMPT_SUFFIX_RE.search(alias) is None,
        "internal VBench staged alias would be rewritten upstream",
    )
    return alias


@dataclass(frozen=True)
class VBenchLongVideoRequest:
    """One immutable route input to a shared official evaluator process."""

    video_path: str | Path
    prompt: str
    measurement_id: str


@dataclass(frozen=True)
class VBenchLongOfflineAssets:
    manifest_path: Path
    manifest_sha256: str
    manifest_nbytes: int
    spec_path: Path
    spec_sha256: str
    spec_nbytes: int
    root: Path
    files: tuple[tuple[str, str, str, int], ...]
    trees: tuple[tuple[str, str, str, int, int], ...]

    def file_path(self, name: str) -> Path:
        for asset_name, relative, _sha256, _nbytes in self.files:
            if asset_name == name:
                return self.root / relative
        raise VBenchLongError(f"offline asset is not bound: {name}")

    def tree_path(self, name: str) -> Path:
        for asset_name, relative, _sha256, _file_count, _nbytes in self.trees:
            if asset_name == name:
                return self.root / relative
        raise VBenchLongError(f"offline asset tree is not bound: {name}")

    def config_binding(self) -> dict[str, Any]:
        return {
            "schema": OFFLINE_ASSET_SCHEMA,
            "manifest": {
                "path": str(self.manifest_path),
                "sha256": self.manifest_sha256,
                "nbytes": self.manifest_nbytes,
            },
            "git_spec": {
                "path": str(self.spec_path),
                "sha256": self.spec_sha256,
                "nbytes": self.spec_nbytes,
            },
            "root": str(self.root),
            "files": {
                name: {"path": relative, "sha256": sha256, "nbytes": nbytes}
                for name, relative, sha256, nbytes in self.files
            },
            "trees": {
                name: {
                    "path": relative,
                    "sha256": sha256,
                    "file_count": file_count,
                    "nbytes": nbytes,
                    "algorithm": OFFLINE_TREE_HASH_ALGORITHM,
                }
                for name, relative, sha256, file_count, nbytes in self.trees
            },
        }


@dataclass(frozen=True)
class VBenchLongCheckout:
    root: Path
    commit: str
    eval_script: Path
    eval_sha256: str
    eval_nbytes: int
    full_info: Path
    full_info_sha256: str
    full_info_nbytes: int
    tracked_tree_sha256: str
    tracked_file_count: int
    tracked_nbytes: int

    def config_binding(self) -> dict[str, Any]:
        return {
            "repository": VBENCH_LONG_REPOSITORY,
            "commit": self.commit,
            "checkout": str(self.root),
            "cleanliness": "tracked_untracked_ignored_empty",
            "tracked_tree": {
                "sha256": self.tracked_tree_sha256,
                "file_count": self.tracked_file_count,
                "nbytes": self.tracked_nbytes,
                "algorithm": CHECKOUT_TREE_HASH_ALGORITHM,
            },
            "eval_script": {
                "path": str(self.eval_script),
                "sha256": self.eval_sha256,
                "nbytes": self.eval_nbytes,
            },
            "full_info": {
                "path": str(self.full_info),
                "sha256": self.full_info_sha256,
                "nbytes": self.full_info_nbytes,
            },
        }


def validate_vbench_long_checkout(
    checkout: str | Path,
    *,
    runner: CommandRunner = subprocess.run,
    environment: Mapping[str, str] | None = None,
) -> VBenchLongCheckout:
    """Bind a sterile checkout of the exact official evaluator revision."""

    unresolved = Path(checkout).expanduser()
    _require(
        unresolved.is_dir() and not unresolved.is_symlink(),
        "VBench checkout must be a non-symlink directory",
    )
    root = unresolved.resolve(strict=True)
    eval_script = root / "vbench2_beta_long" / "eval_long.py"
    full_info = root / "vbench2_beta_long" / "VBench_full_info.json"
    _require(
        eval_script.is_file() and not eval_script.is_symlink(),
        "pinned VBench-Long eval_long.py is missing or is a symlink",
    )
    _require(
        full_info.is_file() and not full_info.is_symlink(),
        "pinned VBench-Long full-info JSON is missing or is a symlink",
    )

    def probe(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        effective_command = list(command)
        _require(
            effective_command[:1] == ["git"],
            "VBench checkout probe must use the pinned Git client entry point",
        )
        # Never let repository replace refs or ambient Git plumbing reinterpret
        # the pinned commit.  A replace ref can otherwise leave ``HEAD`` showing
        # the expected SHA while ``status`` and ``ls-tree`` silently operate on
        # a different evaluator tree.
        effective_command.insert(1, "--no-replace-objects")
        probe_environment = dict(
            os.environ if environment is None else environment
        )
        for variable in tuple(probe_environment):
            if variable.startswith("GIT_"):
                probe_environment.pop(variable, None)
        probe_environment.update(
            {
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        try:
            completed = runner(
                effective_command,
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
                env=probe_environment,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise VBenchLongError(
                f"cannot inspect frozen VBench checkout: {error}"
            ) from error
        _require(
            isinstance(completed.returncode, int),
            "VBench checkout probe returned no status",
        )
        return completed

    revision = probe(("git", "-C", str(root), "rev-parse", "HEAD"))
    _require(revision.returncode == 0, "cannot resolve VBench checkout revision")
    _require(
        revision.stdout.strip() == VBENCH_LONG_COMMIT,
        "VBench checkout is not at the frozen commit",
    )
    status = probe(
        (
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
        )
    )
    _require(status.returncode == 0, "cannot inspect VBench checkout status")
    _require(
        not status.stdout,
        "VBench checkout contains tracked, untracked, or ignored filesystem drift",
    )

    tree = probe(
        (
            "git",
            "-C",
            str(root),
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            VBENCH_LONG_COMMIT,
        )
    )
    _require(tree.returncode == 0, "cannot enumerate frozen VBench commit tree")
    commit_entries: dict[str, tuple[str, str]] = {}
    for raw in tree.stdout.split("\0"):
        if not raw:
            continue
        try:
            metadata, raw_relative = raw.split("\t", 1)
            mode, object_type, object_id = metadata.split(" ", 2)
        except ValueError as error:
            raise VBenchLongError("VBench commit tree record is malformed") from error
        relative = _portable_relative_path(raw_relative, label="checkout tracked file")
        _require(
            object_type == "blob"
            and mode in {"100644", "100755"}
            and re.fullmatch(r"[0-9a-f]{40}", object_id) is not None,
            f"unsupported VBench tracked entry: {relative}",
        )
        _require(relative not in commit_entries, "duplicate VBench commit tree path")
        commit_entries[relative] = (mode, object_id)
    _require(bool(commit_entries), "frozen VBench commit tree is empty")

    index = probe(("git", "-C", str(root), "ls-files", "--stage", "-z"))
    _require(index.returncode == 0, "cannot inspect frozen VBench index")
    index_entries: dict[str, tuple[str, str]] = {}
    for raw in index.stdout.split("\0"):
        if not raw:
            continue
        try:
            metadata, raw_relative = raw.split("\t", 1)
            mode, object_id, stage = metadata.split(" ", 2)
        except ValueError as error:
            raise VBenchLongError("VBench index record is malformed") from error
        relative = _portable_relative_path(raw_relative, label="checkout index file")
        _require(stage == "0", f"VBench index has an unmerged entry: {relative}")
        _require(relative not in index_entries, "duplicate VBench index path")
        index_entries[relative] = (mode, object_id)
    _require(
        index_entries == commit_entries,
        "VBench index entries differ from the frozen commit tree",
    )

    flags = probe(("git", "-C", str(root), "ls-files", "-v", "-z"))
    _require(flags.returncode == 0, "cannot inspect frozen VBench index flags")
    flag_entries: dict[str, str] = {}
    for raw in flags.stdout.split("\0"):
        if not raw:
            continue
        _require(
            len(raw) >= 3 and raw[1] == " ",
            "VBench index flag record is malformed",
        )
        tag = raw[0]
        relative = _portable_relative_path(raw[2:], label="checkout index flag file")
        _require(relative not in flag_entries, "duplicate VBench index flag path")
        flag_entries[relative] = tag
    _require(
        set(flag_entries) == set(commit_entries),
        "VBench index flag paths differ from the frozen commit tree",
    )
    flagged = sorted(path for path, tag in flag_entries.items() if tag != "H")
    _require(
        not flagged,
        "VBench index contains assume-unchanged, skip-worktree, or other flags",
    )

    tracked_records: list[tuple[str, str, str, int, str]] = []
    tracked_nbytes = 0
    for relative, (mode, object_id) in sorted(commit_entries.items()):
        path = root.joinpath(*PurePosixPath(relative).parts)
        _require(
            path.is_file() and not path.is_symlink(),
            f"VBench tracked file is missing or is a symlink: {relative}",
        )
        payload = path.read_bytes()
        git_blob = hashlib.sha1(
            f"blob {len(payload)}\0".encode("ascii") + payload
        ).hexdigest()
        _require(
            git_blob == object_id,
            f"VBench tracked bytes differ from the frozen commit: {relative}",
        )
        executable = bool(path.stat().st_mode & 0o111)
        _require(
            executable == (mode == "100755"),
            f"VBench tracked executable mode drifted: {relative}",
        )
        file_sha256 = hashlib.sha256(payload).hexdigest()
        tracked_records.append(
            (relative, mode, object_id, len(payload), file_sha256)
        )
        tracked_nbytes += len(payload)
    tracked_digest = hashlib.sha256()
    for relative, mode, object_id, nbytes, file_sha256 in tracked_records:
        tracked_digest.update(relative.encode("utf-8"))
        tracked_digest.update(b"\0")
        tracked_digest.update(mode.encode("ascii"))
        tracked_digest.update(b"\0")
        tracked_digest.update(object_id.encode("ascii"))
        tracked_digest.update(b"\0")
        tracked_digest.update(str(nbytes).encode("ascii"))
        tracked_digest.update(b"\0")
        tracked_digest.update(file_sha256.encode("ascii"))
        tracked_digest.update(b"\n")

    eval_sha256, eval_nbytes = _sha256_file(eval_script)
    full_info_sha256, full_info_nbytes = _sha256_file(full_info)
    return VBenchLongCheckout(
        root=root,
        commit=VBENCH_LONG_COMMIT,
        eval_script=eval_script,
        eval_sha256=eval_sha256,
        eval_nbytes=eval_nbytes,
        full_info=full_info,
        full_info_sha256=full_info_sha256,
        full_info_nbytes=full_info_nbytes,
        tracked_tree_sha256=tracked_digest.hexdigest(),
        tracked_file_count=len(tracked_records),
        tracked_nbytes=tracked_nbytes,
    )


def validate_vbench_long_offline_assets(
    manifest_path: str | Path,
    *,
    spec_path: str | Path,
) -> VBenchLongOfflineAssets:
    """Validate the complete, project-private VBench-Long offline bundle."""

    candidate = Path(manifest_path).expanduser()
    _require(candidate.is_file() and not candidate.is_symlink(), "offline asset manifest is missing")
    manifest = candidate.resolve(strict=True)
    unresolved_spec = Path(spec_path).expanduser()
    _require(
        unresolved_spec.is_file() and not unresolved_spec.is_symlink(),
        "offline asset Git specification is missing",
    )
    spec = unresolved_spec.resolve(strict=True)
    manifest_bytes = manifest.read_bytes()
    spec_bytes = spec.read_bytes()
    _require(
        manifest_bytes == spec_bytes,
        "offline asset manifest differs from the Git specification",
    )
    value = _load_object(manifest)
    _require(
        manifest_bytes == _canonical_bytes(value) + b"\n",
        "offline asset manifest is not canonical JSON",
    )
    _require(
        set(value) == {"schema", "files", "trees"}
        and value.get("schema") == OFFLINE_ASSET_SCHEMA,
        "offline asset manifest schema drifted",
    )
    raw_files = value.get("files")
    raw_trees = value.get("trees")
    _require(
        isinstance(raw_files, Mapping)
        and set(raw_files) == set(VBENCH_LONG_OFFLINE_FILE_PATHS),
        "offline asset file set drifted",
    )
    _require(
        isinstance(raw_trees, Mapping)
        and set(raw_trees) == set(VBENCH_LONG_OFFLINE_TREE_PATHS),
        "offline asset source-tree set drifted",
    )
    root = manifest.parent.resolve(strict=True)
    file_bindings: list[tuple[str, str, str, int]] = []
    for name, expected_relative in VBENCH_LONG_OFFLINE_FILE_PATHS.items():
        record = raw_files[name]
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "sha256", "nbytes"},
            f"offline asset record is invalid: {name}",
        )
        relative = _portable_relative_path(record.get("path"), label=name)
        _require(relative == expected_relative, f"offline asset path drifted: {name}")
        path = root / relative
        _require(path.is_file() and not path.is_symlink(), f"offline asset is missing: {name}")
        sha256, nbytes = _sha256_file(path)
        _require(
            record.get("sha256") == sha256 and record.get("nbytes") == nbytes,
            f"offline asset bytes drifted: {name}",
        )
        file_bindings.append((name, relative, sha256, nbytes))
    tree_bindings: list[tuple[str, str, str, int, int]] = []
    for name, expected_relative in VBENCH_LONG_OFFLINE_TREE_PATHS.items():
        record = raw_trees[name]
        _require(
            isinstance(record, Mapping)
            and set(record)
            == {"path", "sha256", "file_count", "nbytes", "algorithm"},
            f"offline asset tree record is invalid: {name}",
        )
        relative = _portable_relative_path(record.get("path"), label=name)
        _require(relative == expected_relative, f"offline asset tree path drifted: {name}")
        binding = _tree_binding(root / relative)
        _require(
            record.get("sha256") == binding["sha256"]
            and record.get("file_count") == binding["file_count"]
            and record.get("nbytes") == binding["nbytes"]
            and record.get("algorithm") == OFFLINE_TREE_HASH_ALGORITHM,
            f"offline asset tree drifted: {name}",
        )
        tree_bindings.append(
            (
                name,
                relative,
                str(binding["sha256"]),
                int(binding["file_count"]),
                int(binding["nbytes"]),
            )
        )
    manifest_sha256, manifest_nbytes = _sha256_file(manifest)
    spec_sha256, spec_nbytes = _sha256_file(spec)
    return VBenchLongOfflineAssets(
        manifest_path=manifest,
        manifest_sha256=manifest_sha256,
        manifest_nbytes=manifest_nbytes,
        spec_path=spec,
        spec_sha256=spec_sha256,
        spec_nbytes=spec_nbytes,
        root=root,
        files=tuple(file_bindings),
        trees=tuple(tree_bindings),
    )


def _write_bound_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        _require(path.read_bytes() == payload, f"existing evidence conflicts: {path}")
        return
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VBenchLongError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _finite_unit(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    result = float(value)
    _require(0.0 <= result <= 1.0, f"{label} must be in [0,1]")
    return result


def _detail_score(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        return float(value)
    _require(
        isinstance(value, (int, float)) and math.isfinite(float(value)),
        f"{label} must be a finite number or boolean",
    )
    return float(value)


def _expected_split_detail_paths(
    *,
    measurement_root: Path,
    staged_videos: Mapping[str, Path],
) -> dict[str, dict[str, dict[str, str]]]:
    """Bind every accepted result row to an actual official split clip."""

    videos = (measurement_root / "videos").resolve(strict=True)
    expected: dict[str, dict[str, dict[str, str]]] = {}
    for alias, staged_video in staged_videos.items():
        _require(
            _STAGED_ALIAS_RE.fullmatch(alias) is not None
            and "_" not in alias
            and staged_video.is_file()
            and not staged_video.is_symlink()
            and staged_video.parent.resolve(strict=True) == videos
            and staged_video.name == f"{alias}.mp4",
            "official staged video alias is invalid",
        )
        split_root = videos / "split_clip" / alias
        _require(
            split_root.is_dir()
            and not split_root.is_symlink()
            and split_root.parent.resolve(strict=True) == videos / "split_clip",
            "official split clip directory is invalid",
        )
        clip_name = re.compile(rf"^{re.escape(alias)}_[0-9]{{3}}\.mp4$")
        entries = list(split_root.iterdir())
        _require(bool(entries), "official split clip set is empty")
        by_clip: dict[str, dict[str, str]] = {}
        for entry in entries:
            _require(
                entry.is_file()
                and not entry.is_symlink()
                and clip_name.fullmatch(entry.name) is not None,
                "official split clip set is invalid",
            )
            resolved = entry.resolve(strict=True)
            _require(
                resolved.parent == split_root.resolve(strict=True),
                "official split clip escapes its staged alias",
            )
            relative = resolved.relative_to(videos).as_posix()
            _require(entry.name not in by_clip, "official split clip set is invalid")
            by_clip[entry.name] = {
                "clip": relative,
                "aggregate": f"{relative}/{alias}.mp4",
            }
        expected[alias] = by_clip
    return expected


def _validate_detail_video_path(
    value: Any,
    *,
    measurement_root: Path,
    staged_video: Path,
    staged_alias: str,
    expected_kind: str,
) -> tuple[str, str]:
    _require(
        isinstance(value, str) and bool(value) and Path(value).is_absolute(),
        "official detail video_path must be absolute",
    )
    candidate = Path(value)
    _require(".." not in candidate.parts, "official detail video_path contains '..'")
    videos = (measurement_root / "videos").resolve(strict=True)
    resolved = candidate.resolve(strict=False)
    _require(
        resolved.is_relative_to(videos),
        "official detail video_path escapes the current measurement videos",
    )
    relative = resolved.relative_to(videos)
    _require(
        _STAGED_ALIAS_RE.fullmatch(staged_alias) is not None
        and "_" not in staged_alias
        and staged_video.name == f"{staged_alias}.mp4",
        "official staged video alias is invalid",
    )
    parts = relative.parts
    _require(
        expected_kind in {"clip", "aggregate"},
        "official detail path kind is invalid",
    )
    clip_name = re.compile(rf"^{re.escape(staged_alias)}_[0-9]{{3}}\.mp4$")
    if (
        expected_kind == "clip"
        and len(parts) == 3
        and parts[0] == "split_clip"
        and parts[1] == staged_alias
        and clip_name.fullmatch(parts[2]) is not None
    ):
        return parts[2], relative.as_posix()
    # Frozen long_custom_input preprocessing emits one additional, non-existent
    # aggregate path from reorganize_clips_results: it appends the alias below
    # the concrete clip filename.  Accept that exact upstream shape, but no
    # arbitrary deeper nesting or ``_full`` path (30/60-second inputs always
    # contain at least one complete two-second clip).
    if (
        expected_kind == "aggregate"
        and len(parts) == 4
        and parts[0] == "split_clip"
        and parts[1] == staged_alias
        and clip_name.fullmatch(parts[2]) is not None
        and parts[3] == f"{staged_alias}.mp4"
    ):
        return parts[2], relative.as_posix()
    raise VBenchLongError(
        "official detail video_path is not derived from the unique staged prompt"
    )


def validate_vbench_long_official_result(
    path: str | Path,
    *,
    measurement_root: str | Path,
    staged_video: str | Path,
    staged_alias: str,
) -> dict[str, float]:
    """Validate provenance and independently aggregate an official result."""

    result_path = Path(path).resolve(strict=True)
    root = Path(measurement_root).resolve(strict=True)
    staged = Path(staged_video).resolve(strict=True)
    _require(
        result_path.parent == root / "results"
        and result_path.name.endswith("_eval_results.json"),
        "official result path is outside the current measurement results",
    )
    value = _load_object(result_path)
    _require(
        set(value) == set(VBENCH_LONG_DIMENSIONS),
        "official result does not contain exactly the six frozen dimensions",
    )
    expected_details = _expected_split_detail_paths(
        measurement_root=root,
        staged_videos={staged_alias: staged},
    )[staged_alias]
    scores: dict[str, float] = {}
    for dimension in VBENCH_LONG_DIMENSIONS:
        item = value[dimension]
        consistency = dimension in {
            "subject_consistency",
            "background_consistency",
        }
        detail_kinds = (
            ("aggregate",)
            if consistency
            else ("clip", "aggregate")
        )
        _require(
            isinstance(item, list) and len(item) == 1 + len(detail_kinds),
            f"official {dimension} result has an unknown structure",
        )
        scores_by_kind: dict[str, dict[str, float]] = {}
        for detail_index, (details, expected_kind) in enumerate(
            zip(item[1:], detail_kinds, strict=True), start=1
        ):
            _require(
                isinstance(details, list) and bool(details),
                f"official {dimension} detail {detail_index} is empty",
            )
            seen_detail_paths: set[str] = set()
            detail_scores: dict[str, float] = {}
            for detail in details:
                _require(
                    isinstance(detail, Mapping),
                    f"official {dimension} detail is not a mapping",
                )
                _require(
                    "video_path" in detail and "video_results" in detail,
                    f"official {dimension} detail lacks video provenance or score",
                )
                split_clip, relative = _validate_detail_video_path(
                    detail["video_path"],
                    measurement_root=root,
                    staged_video=staged,
                    staged_alias=staged_alias,
                    expected_kind=expected_kind,
                )
                _require(
                    relative not in seen_detail_paths,
                    f"official {dimension} contains a duplicate detail path",
                )
                seen_detail_paths.add(relative)
                detail_scores[split_clip] = _detail_score(
                    detail["video_results"],
                    label=f"official {dimension} detail score",
                )
            _require(
                seen_detail_paths
                == {
                    paths[expected_kind]
                    for paths in expected_details.values()
                },
                f"official {dimension} {expected_kind} details do not exactly cover split clips",
            )
            scores_by_kind[expected_kind] = detail_scores
        if not consistency:
            primary_scores = scores_by_kind["clip"]
            aggregate_scores = scores_by_kind["aggregate"]
            _require(
                set(primary_scores) == set(aggregate_scores)
                and all(
                    math.isclose(
                        primary_scores[split_clip],
                        aggregate_scores[split_clip],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for split_clip in primary_scores
                ),
                f"official {dimension} aggregate disagrees with clip details",
            )
        primary_scores = scores_by_kind[detail_kinds[0]]
        recomputed = math.fsum(primary_scores.values()) / len(primary_scores)
        if dimension == "imaging_quality":
            recomputed /= 100.0
        overall = _finite_unit(item[0], f"{dimension} overall score")
        _require(
            math.isclose(overall, recomputed, rel_tol=0.0, abs_tol=1e-12),
            f"official {dimension} overall disagrees with detailed measurements",
        )
        scores[dimension] = overall
    return scores


def _batch_detail_alias(
    value: Any,
    *,
    measurement_root: Path,
    staged_videos: Mapping[str, Path],
    expected_kind: str,
) -> tuple[str, str, str]:
    """Return the exact staged alias, split clip, and relative detail path."""

    _require(
        isinstance(value, str) and bool(value) and Path(value).is_absolute(),
        "official batch detail video_path must be absolute",
    )
    candidate = Path(value)
    _require(".." not in candidate.parts, "official batch detail video_path contains '..'")
    videos = (measurement_root / "videos").resolve(strict=True)
    resolved = candidate.resolve(strict=False)
    _require(
        resolved.is_relative_to(videos),
        "official batch detail video_path escapes the current batch videos",
    )
    relative = resolved.relative_to(videos)
    parts = relative.parts
    _require(
        expected_kind in {"clip", "aggregate"},
        "official batch detail path kind is invalid",
    )
    _require(
        len(parts) in {3, 4}
        and parts[0] == "split_clip"
        and parts[1] in staged_videos,
        "official batch detail contains a foreign staged alias",
    )
    alias = parts[1]
    staged = staged_videos[alias]
    _require(
        _STAGED_ALIAS_RE.fullmatch(alias) is not None
        and "_" not in alias
        and staged.parent.resolve(strict=True) == videos
        and staged.name == f"{alias}.mp4",
        "official batch staged video alias is invalid",
    )
    clip_name = re.compile(rf"^{re.escape(alias)}_[0-9]{{3}}\.mp4$")
    valid_clip = (
        expected_kind == "clip"
        and len(parts) == 3
        and clip_name.fullmatch(parts[2]) is not None
    )
    valid_aggregate = (
        expected_kind == "aggregate"
        and len(parts) == 4
        and clip_name.fullmatch(parts[2]) is not None
        and parts[3] == f"{alias}.mp4"
    )
    _require(
        valid_clip or valid_aggregate,
        "official batch detail path is not derived from its staged alias",
    )
    return alias, parts[2], relative.as_posix()


def validate_vbench_long_official_batch_result(
    path: str | Path,
    *,
    measurement_root: str | Path,
    staged_videos: Mapping[str, str | Path],
) -> dict[str, dict[str, float]]:
    """Rebuild every video's six scores from official per-alias details.

    The official top-level score is only a batch consistency check.  It is
    never copied into an individual measurement.
    """

    result_path = Path(path).resolve(strict=True)
    root = Path(measurement_root).resolve(strict=True)
    _require(
        result_path.parent == root / "results"
        and result_path.name.endswith("_eval_results.json"),
        "official batch result path is outside the current batch results",
    )
    _require(bool(staged_videos), "official batch has no staged videos")
    normalized: dict[str, Path] = {}
    for alias, value in staged_videos.items():
        _require(
            isinstance(alias, str)
            and _STAGED_ALIAS_RE.fullmatch(alias) is not None
            and alias not in normalized,
            "official batch staged alias set is invalid",
        )
        staged = Path(value).resolve(strict=True)
        _require(
            staged.is_file()
            and not staged.is_symlink()
            and staged.parent == (root / "videos").resolve(strict=True)
            and staged.name == f"{alias}.mp4",
            "official batch staged video is invalid",
        )
        normalized[alias] = staged

    expected_details = _expected_split_detail_paths(
        measurement_root=root,
        staged_videos=normalized,
    )

    value = _load_object(result_path)
    _require(
        set(value) == set(VBENCH_LONG_DIMENSIONS),
        "official batch result does not contain exactly the six frozen dimensions",
    )
    by_alias: dict[str, dict[str, float]] = {alias: {} for alias in normalized}
    expected_aliases = set(normalized)
    for dimension in VBENCH_LONG_DIMENSIONS:
        item = value[dimension]
        consistency = dimension in {"subject_consistency", "background_consistency"}
        _require(
            isinstance(item, list) and len(item) == (2 if consistency else 3),
            f"official batch {dimension} result has an unknown structure",
        )
        primary = item[1]
        _require(
            isinstance(primary, list) and bool(primary),
            f"official batch {dimension} primary details are empty",
        )
        primary_by_alias: dict[str, dict[str, float]] = {}
        seen_primary_paths: set[str] = set()
        primary_kind = "aggregate" if consistency else "clip"
        for detail in primary:
            _require(
                isinstance(detail, Mapping)
                and "video_path" in detail
                and "video_results" in detail,
                f"official batch {dimension} primary detail is invalid",
            )
            alias, split_clip, relative = _batch_detail_alias(
                detail["video_path"],
                measurement_root=root,
                staged_videos=normalized,
                expected_kind=primary_kind,
            )
            _require(
                relative not in seen_primary_paths,
                f"official batch {dimension} contains a duplicate detail path",
            )
            seen_primary_paths.add(relative)
            primary_by_alias.setdefault(alias, {})[split_clip] = _detail_score(
                detail["video_results"],
                label=f"official batch {dimension} detail score",
            )
        _require(
            seen_primary_paths
            == {
                paths[primary_kind]
                for by_clip in expected_details.values()
                for paths in by_clip.values()
            },
            f"official batch {dimension} details do not exactly cover split clips",
        )
        _require(
            set(primary_by_alias) == expected_aliases,
            f"official batch {dimension} details do not exactly cover staged aliases",
        )

        aggregate_by_alias: dict[str, dict[str, float]] = {}
        if not consistency:
            aggregate = item[2]
            _require(
                isinstance(aggregate, list) and bool(aggregate),
                f"official batch {dimension} aggregate details are empty",
            )
            seen_aggregate_paths: set[str] = set()
            for detail in aggregate:
                _require(
                    isinstance(detail, Mapping)
                    and "video_path" in detail
                    and "video_results" in detail,
                    f"official batch {dimension} aggregate detail is invalid",
                )
                alias, split_clip, relative = _batch_detail_alias(
                    detail["video_path"],
                    measurement_root=root,
                    staged_videos=normalized,
                    expected_kind="aggregate",
                )
                _require(
                    relative not in seen_aggregate_paths,
                    f"official batch {dimension} contains a duplicate aggregate path",
                )
                seen_aggregate_paths.add(relative)
                aggregate_by_alias.setdefault(alias, {})[
                    split_clip
                ] = _detail_score(
                    detail["video_results"],
                    label=f"official batch {dimension} aggregate score",
                )
            _require(
                seen_aggregate_paths
                == {
                    paths["aggregate"]
                    for by_clip in expected_details.values()
                    for paths in by_clip.values()
                },
                f"official batch {dimension} aggregates do not exactly cover split clips",
            )
            _require(
                set(aggregate_by_alias) == expected_aliases,
                f"official batch {dimension} aggregates do not exactly cover staged aliases",
            )
            for alias in sorted(expected_aliases):
                _require(
                    set(primary_by_alias[alias])
                    == set(aggregate_by_alias[alias])
                    and all(
                        math.isclose(
                            primary_by_alias[alias][split_clip],
                            aggregate_by_alias[alias][split_clip],
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        for split_clip in primary_by_alias[alias]
                    ),
                    f"official batch {dimension} aggregate disagrees with clip details",
                )

        final_scores: dict[str, float] = {}
        for alias in sorted(expected_aliases):
            recomputed_raw = math.fsum(primary_by_alias[alias].values()) / len(
                primary_by_alias[alias]
            )
            normalized_score = (
                recomputed_raw / 100.0
                if dimension == "imaging_quality"
                else recomputed_raw
            )
            final_scores[alias] = _finite_unit(
                normalized_score,
                f"official batch {dimension} alias score",
            )
            by_alias[alias][dimension] = final_scores[alias]
        all_primary_scores = [
            score
            for alias in sorted(expected_aliases)
            for _, score in sorted(primary_by_alias[alias].items())
        ]
        recomputed_overall = math.fsum(all_primary_scores) / len(all_primary_scores)
        if dimension == "imaging_quality":
            recomputed_overall /= 100.0
        overall = _finite_unit(item[0], f"official batch {dimension} overall score")
        _require(
            math.isclose(overall, recomputed_overall, rel_tol=0.0, abs_tol=1e-12),
            f"official batch {dimension} overall disagrees with detailed measurements",
        )
    return by_alias


def validate_vbench_long_success_attempt(
    measurement_root: str | Path,
    *,
    run_dir: str | Path,
    config_sha256: str,
    command: Sequence[str],
) -> bool:
    """Return true only for canonical successful evidence bound to this run."""

    root = Path(measurement_root).resolve(strict=True)
    run_root = Path(run_dir).resolve(strict=True)
    _require(
        root.is_relative_to(run_root),
        "VBench-Long attempt root escapes run_dir",
    )
    attempts = root / "attempts"
    if not attempts.is_dir():
        return False
    command_list = list(command)
    command_sha256 = hashlib.sha256(_canonical_bytes(command_list)).hexdigest()
    valid = False
    for path in sorted(attempts.glob("*.json")):
        value = _load_object(path)
        _require(
            path.read_bytes() == _canonical_bytes(value) + b"\n",
            "VBench-Long attempt evidence is not canonical JSON",
        )
        attempt_id = path.stem
        expected_keys = {
            "schema",
            "attempt",
            "config_sha256",
            "command",
            "command_sha256",
            "returncode",
            "stdout",
            "stderr",
        }
        _require(
            set(value) == expected_keys
            and value.get("schema") == ATTEMPT_SCHEMA
            and isinstance(value.get("attempt"), int)
            and not isinstance(value.get("attempt"), bool)
            and int(value["attempt"]) >= 1
            and attempt_id == f"{int(value['attempt']):04d}",
            "VBench-Long attempt identity is invalid",
        )
        for stream in ("stdout", "stderr"):
            binding = value.get(stream)
            expected_relative = (
                root / "attempts" / f"{attempt_id}.{stream}.log"
            ).relative_to(run_root).as_posix()
            _require(
                isinstance(binding, Mapping)
                and set(binding) == {"path", "sha256", "nbytes"}
                and binding.get("path") == expected_relative,
                f"VBench-Long attempt {stream} binding is invalid",
            )
            stream_path = run_root / expected_relative
            _require(
                stream_path.is_file()
                and not stream_path.is_symlink()
                and _sha256_file(stream_path)
                == (binding.get("sha256"), binding.get("nbytes")),
                f"VBench-Long attempt {stream} bytes drifted",
            )
        returncode = value.get("returncode")
        _require(
            isinstance(returncode, int) and not isinstance(returncode, bool),
            "VBench-Long attempt returncode is invalid",
        )
        if returncode == 0:
            _require(
                value.get("config_sha256") == config_sha256
                and value.get("command") == command_list
                and value.get("command_sha256") == command_sha256,
                "successful VBench-Long attempt is not bound to current config and command",
            )
            valid = True
    return valid


def _file_binding(path: Path, *, run_dir: Path) -> dict[str, Any]:
    sha256, nbytes = _sha256_file(path)
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": sha256,
        "nbytes": nbytes,
    }


def _validate_file_binding(
    value: Any,
    *,
    run_dir: Path,
    expected_path: Path | None,
    label: str,
) -> Path:
    _require(
        isinstance(value, Mapping)
        and set(value) == {"path", "sha256", "nbytes"},
        f"{label} binding is invalid",
    )
    relative = _portable_relative_path(value.get("path"), label=label)
    path = (run_dir / relative).resolve(strict=True)
    _require(path.is_relative_to(run_dir), f"{label} escapes run_dir")
    if expected_path is not None:
        _require(path == expected_path.resolve(strict=True), f"{label} path drifted")
    _require(
        path.is_file()
        and not path.is_symlink()
        and _sha256_file(path) == (value.get("sha256"), value.get("nbytes")),
        f"{label} bytes drifted",
    )
    return path


def validate_vbench_long_batch_success_attempt(
    batch_root: str | Path,
    *,
    run_dir: str | Path,
    batch_id: str,
    config_sha256: str,
    command: Sequence[str],
    inputs: Sequence[Mapping[str, Any]],
) -> Path | None:
    """Return the result bound by a canonical successful shared attempt."""

    root = Path(batch_root).resolve(strict=True)
    run_root = Path(run_dir).resolve(strict=True)
    _require(root.is_relative_to(run_root), "VBench-Long batch root escapes run_dir")
    _require(
        _BATCH_ID_RE.fullmatch(batch_id) is not None and root.name == batch_id,
        "VBench-Long batch identity is invalid",
    )
    attempts = root / "attempts"
    if not attempts.is_dir():
        return None
    command_list = list(command)
    command_sha256 = hashlib.sha256(_canonical_bytes(command_list)).hexdigest()
    frozen_inputs = [dict(item) for item in inputs]
    valid_results: list[Path] = []
    for path in sorted(attempts.glob("*.json")):
        value = _load_object(path)
        _require(
            path.read_bytes() == _canonical_bytes(value) + b"\n",
            "VBench-Long batch attempt evidence is not canonical JSON",
        )
        attempt_id = path.stem
        _require(
            set(value)
            == {
                "schema",
                "attempt",
                "batch_id",
                "config_sha256",
                "inputs",
                "command",
                "command_sha256",
                "returncode",
                "stdout",
                "stderr",
                "official_result",
            }
            and value.get("schema") == BATCH_ATTEMPT_SCHEMA
            and isinstance(value.get("attempt"), int)
            and not isinstance(value.get("attempt"), bool)
            and int(value["attempt"]) >= 1
            and attempt_id == f"{int(value['attempt']):04d}",
            "VBench-Long batch attempt identity is invalid",
        )
        _require(
            value.get("batch_id") == batch_id
            and value.get("config_sha256") == config_sha256
            and value.get("inputs") == frozen_inputs
            and value.get("command") == command_list
            and value.get("command_sha256") == command_sha256,
            "VBench-Long batch attempt binding drifted",
        )
        for stream in ("stdout", "stderr"):
            expected = root / "attempts" / f"{attempt_id}.{stream}.log"
            _validate_file_binding(
                value.get(stream),
                run_dir=run_root,
                expected_path=expected,
                label=f"VBench-Long batch attempt {stream}",
            )
        returncode = value.get("returncode")
        _require(
            isinstance(returncode, int) and not isinstance(returncode, bool),
            "VBench-Long batch attempt returncode is invalid",
        )
        if returncode == 0:
            result_path = _validate_file_binding(
                value.get("official_result"),
                run_dir=run_root,
                expected_path=None,
                label="VBench-Long batch official result",
            )
            _require(
                result_path.parent == root / "results"
                and result_path.name.endswith("_eval_results.json"),
                "VBench-Long batch result path is invalid",
            )
            valid_results.append(result_path)
        else:
            _require(
                value.get("official_result") is None,
                "failed VBench-Long batch attempt unexpectedly binds a result",
            )
    if not valid_results:
        return None
    reference = valid_results[0]
    reference_hash = _sha256_file(reference)
    _require(
        all(_sha256_file(path) == reference_hash for path in valid_results),
        "successful VBench-Long batch attempts disagree on result bytes",
    )
    return valid_results[-1]


@dataclass(frozen=True)
class VBenchLongMeasurement:
    """Official six-dimension result with QualityEvaluator-compatible aliases."""

    dimension_scores: tuple[tuple[str, float], ...]
    macro_mean: float
    motion_smoothness: float
    evidence_path: str

    @property
    def primary(self) -> float:
        return self.macro_mean

    @property
    def temporal_consistency(self) -> float:
        return self.motion_smoothness

    def scores(self) -> dict[str, float]:
        return dict(self.dimension_scores)


def validate_vbench_long_batch_measurement(
    sidecar_path: str | Path,
    *,
    run_dir: str | Path,
    measurement_id: str,
    prompt: str,
    video_path: str | Path,
) -> VBenchLongMeasurement:
    """Validate one route sidecar and its complete shared batch provenance."""

    run_root = Path(run_dir).resolve(strict=True)
    sidecar = Path(sidecar_path).resolve(strict=True)
    source = Path(video_path).resolve(strict=True)
    _require(sidecar.is_relative_to(run_root), "batch measurement escapes run_dir")
    value = _load_object(sidecar)
    _require(
        sidecar.read_bytes() == _canonical_bytes(value) + b"\n",
        "batch measurement evidence is not canonical JSON",
    )
    expected_keys = {
        "schema",
        "measurement_id",
        "batch_id",
        "batch_config",
        "batch_attempt",
        "official_result",
        "dimension_scores",
        "aggregation",
        "macro_mean",
        "motion_smoothness",
        "staging",
        "source_video",
    }
    _require(
        set(value) == expected_keys
        and value.get("schema") == BATCH_MEASUREMENT_SCHEMA
        and value.get("measurement_id") == measurement_id
        and value.get("aggregation")
        == "unweighted_arithmetic_mean_of_six_official_dimension_scores",
        "batch measurement schema, identity, or aggregation drifted",
    )
    batch_id = value.get("batch_id")
    _require(
        isinstance(batch_id, str) and _BATCH_ID_RE.fullmatch(batch_id) is not None,
        "batch measurement has an invalid batch identity",
    )
    batch_root = run_root / BATCH_ROOT_RELATIVE / batch_id
    config_path = _validate_file_binding(
        value.get("batch_config"),
        run_dir=run_root,
        expected_path=batch_root / "config.json",
        label="VBench-Long batch config",
    )
    config = _load_object(config_path)
    _require(
        config_path.read_bytes() == _canonical_bytes(config) + b"\n",
        "VBench-Long batch config is not canonical JSON",
    )
    _require(
        set(config)
        == {
            "schema",
            "batch_id",
            "upstream",
            "python_executable",
            "launcher",
            "visible_device",
            "mode",
            "dev_flag",
            "dimensions",
            "offline_assets",
            "offline_policy",
            "offline_environment",
            "inputs",
            "command",
        }
        and config.get("schema") == BATCH_CONFIG_SCHEMA
        and config.get("batch_id") == batch_id
        and config.get("mode") == "long_custom_input"
        and config.get("dev_flag") is True
        and config.get("dimensions") == list(VBENCH_LONG_DIMENSIONS),
        "VBench-Long batch config fields drifted",
    )
    inputs = config.get("inputs")
    _require(isinstance(inputs, list) and bool(inputs), "VBench-Long batch input set is empty")
    aliases: set[str] = set()
    measurement_ids: set[str] = set()
    staged_videos: dict[str, Path] = {}
    selected: Mapping[str, Any] | None = None
    for item in inputs:
        _require(
            isinstance(item, Mapping)
            and set(item)
            == {"alias", "measurement_id", "prompt", "source_video", "staged_video"},
            "VBench-Long batch input record is invalid",
        )
        alias = item.get("alias")
        item_measurement = item.get("measurement_id")
        item_prompt = item.get("prompt")
        _require(
            isinstance(alias, str)
            and _STAGED_ALIAS_RE.fullmatch(alias) is not None
            and alias not in aliases,
            "VBench-Long batch contains a duplicate or invalid alias",
        )
        _require(
            isinstance(item_measurement, str)
            and _MEASUREMENT_RE.fullmatch(item_measurement) is not None
            and item_measurement not in measurement_ids,
            "VBench-Long batch contains a duplicate or invalid measurement ID",
        )
        _require(
            isinstance(item_prompt, str) and bool(item_prompt),
            "VBench-Long batch prompt is invalid",
        )
        aliases.add(alias)
        measurement_ids.add(item_measurement)
        source_binding = item.get("source_video")
        _require(
            isinstance(source_binding, Mapping)
            and set(source_binding) == {"path", "sha256", "nbytes"}
            and isinstance(source_binding.get("path"), str)
            and Path(str(source_binding["path"])).is_absolute(),
            "VBench-Long batch source binding is invalid",
        )
        item_source = Path(str(source_binding["path"])).resolve(strict=True)
        _require(
            item_source.is_file()
            and _sha256_file(item_source)
            == (source_binding.get("sha256"), source_binding.get("nbytes")),
            "VBench-Long batch source bytes drifted",
        )
        staged_relative = _portable_relative_path(
            item.get("staged_video"), label="VBench-Long batch staged video"
        )
        staged = (run_root / staged_relative).resolve(strict=True)
        _require(
            staged.is_relative_to(batch_root / "videos")
            and staged.name == f"{alias}.mp4"
            and staged.is_file()
            and not staged.is_symlink()
            and staged.samefile(item_source)
            and _sha256_file(staged)
            == (source_binding.get("sha256"), source_binding.get("nbytes")),
            "VBench-Long batch staged video differs from its source",
        )
        staged_videos[alias] = staged
        if item_measurement == measurement_id:
            _require(selected is None, "VBench-Long batch repeats the requested measurement")
            selected = item
    _require(selected is not None, "VBench-Long batch omits the requested measurement")
    selected_source = selected["source_video"]
    selected_alias = str(selected["alias"])
    _require(
        selected.get("prompt") == prompt
        and Path(str(selected_source["path"])).resolve(strict=True) == source
        and value.get("source_video") == selected_source
        and value.get("staging")
        == {
            "algorithm": STAGED_ALIAS_ALGORITHM,
            "alias": selected_alias,
            "original_prompt_id": prompt,
            "staged_video": selected["staged_video"],
        },
        "batch measurement route binding drifted",
    )
    command = config.get("command")
    _require(
        isinstance(command, list) and all(isinstance(item, str) for item in command),
        "VBench-Long batch command is invalid",
    )
    config_sha256 = str(value["batch_config"]["sha256"])
    result_path = validate_vbench_long_batch_success_attempt(
        batch_root,
        run_dir=run_root,
        batch_id=batch_id,
        config_sha256=config_sha256,
        command=command,
        inputs=inputs,
    )
    _require(result_path is not None, "batch measurement has no successful shared attempt")
    attempt_path = _validate_file_binding(
        value.get("batch_attempt"),
        run_dir=run_root,
        expected_path=None,
        label="VBench-Long shared batch attempt",
    )
    _require(
        attempt_path.parent == batch_root / "attempts"
        and attempt_path.suffix == ".json",
        "batch measurement attempt path is invalid",
    )
    attempt = _load_object(attempt_path)
    _require(
        attempt.get("schema") == BATCH_ATTEMPT_SCHEMA
        and attempt.get("returncode") == 0
        and attempt.get("batch_id") == batch_id
        and attempt.get("config_sha256") == config_sha256
        and attempt.get("inputs") == inputs
        and attempt.get("command") == command,
        "batch measurement does not bind a successful matching attempt",
    )
    bound_result = _validate_file_binding(
        value.get("official_result"),
        run_dir=run_root,
        expected_path=result_path,
        label="VBench-Long batch official result",
    )
    _require(
        attempt.get("official_result") == value.get("official_result"),
        "batch measurement and shared attempt bind different result bytes",
    )
    all_scores = validate_vbench_long_official_batch_result(
        bound_result,
        measurement_root=batch_root,
        staged_videos=staged_videos,
    )
    scores = all_scores[selected_alias]
    _require(
        value.get("dimension_scores") == scores,
        "batch measurement scores differ from official alias details",
    )
    macro = math.fsum(scores.values()) / len(scores)
    _require(
        value.get("macro_mean") == macro
        and value.get("motion_smoothness") == scores["motion_smoothness"],
        "batch measurement aggregate scores drifted",
    )
    return VBenchLongMeasurement(
        dimension_scores=tuple((name, scores[name]) for name in VBENCH_LONG_DIMENSIONS),
        macro_mean=macro,
        motion_smoothness=scores["motion_smoothness"],
        evidence_path=sidecar.relative_to(run_root).as_posix(),
    )


class OfficialVBenchLongScorer:
    """Run and recover one-video official VBench-Long measurements."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        checkout: str | Path,
        python_executable: str | Path,
        asset_manifest: str | Path,
        asset_spec: str | Path,
        visible_device: str | None = None,
        runner: CommandRunner = subprocess.run,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve(strict=True)
        _require(self.run_dir.is_dir(), "run_dir must be an existing directory")
        executable_text = os.fspath(python_executable)
        resolved_python = (
            shutil.which(executable_text)
            if not Path(executable_text).is_absolute()
            else executable_text
        )
        _require(resolved_python is not None, "VBench Python executable was not found")
        self.python_executable = str(Path(resolved_python).expanduser().resolve(strict=True))
        self.assets = validate_vbench_long_offline_assets(
            asset_manifest,
            spec_path=asset_spec,
        )
        self.launcher = Path(__file__).with_name("vbench_offline_entry.py").resolve(strict=True)
        _require(self.launcher.is_file(), "offline VBench launcher is missing")
        self.launcher_sha256, self.launcher_nbytes = _sha256_file(self.launcher)
        runtime_cache = self.run_dir / "tmp" / "vbench-long-runtime-cache"
        runtime_cache.mkdir(parents=True, exist_ok=True)
        _require(
            runtime_cache.is_dir() and not runtime_cache.is_symlink(),
            "VBench runtime cache is invalid",
        )
        cache_directories = {
            "MPLCONFIGDIR": runtime_cache / "matplotlib",
            "XDG_CACHE_HOME": runtime_cache / "xdg",
            "HF_HOME": runtime_cache / "huggingface",
            "HF_HUB_CACHE": runtime_cache / "huggingface" / "hub",
            "HUGGINGFACE_HUB_CACHE": runtime_cache / "huggingface" / "hub",
            "HUGGING_FACE_HUB_CACHE": runtime_cache / "huggingface" / "hub",
            "HF_DATASETS_CACHE": runtime_cache / "huggingface" / "datasets",
            "HF_ASSETS_CACHE": runtime_cache / "huggingface" / "assets",
            "HUGGINGFACE_ASSETS_CACHE": runtime_cache / "huggingface" / "assets",
            "HF_MODULES_CACHE": runtime_cache / "huggingface" / "modules",
            "TRANSFORMERS_CACHE": runtime_cache / "transformers",
            "DIFFUSERS_CACHE": runtime_cache / "diffusers",
            "TORCH_HOME": runtime_cache / "torch",
            "TORCH_EXTENSIONS_DIR": runtime_cache / "torch-extensions",
            # Do not read generated bytecode beside shared Python packages.
            # Disabling writes alone still reads those cold/stale .pyc files.
            "PYTHONPYCACHEPREFIX": runtime_cache / "python-bytecode",
        }
        for cache in set(cache_directories.values()):
            cache.mkdir(parents=True, exist_ok=True)
            _require(
                cache.is_dir() and not cache.is_symlink(),
                f"VBench runtime cache directory is invalid: {cache}",
            )
        _require(
            visible_device is None
            or (
                isinstance(visible_device, str)
                and bool(visible_device.strip())
                and "," not in visible_device
            ),
            "visible_device must name exactly one CUDA device",
        )
        self.visible_device = visible_device
        self.environment = dict(os.environ)
        for variable in VBENCH_LONG_SCRUBBED_ENVIRONMENT_VARIABLES:
            self.environment.pop(variable, None)
        if visible_device is not None:
            self.environment["CUDA_VISIBLE_DEVICES"] = visible_device
        self.offline_environment = {
            **VBENCH_DISTRIBUTED_ENVIRONMENT,
            "HOME": str(self.assets.root / "home"),
            "VBENCH_CACHE_DIR": str(self.assets.root / "vbench-cache"),
            **{name: str(path) for name, path in cache_directories.items()},
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "http_proxy": "http://127.0.0.1:9",
            "https_proxy": "http://127.0.0.1:9",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "socks5://127.0.0.1:9",
        }
        self.environment.update(self.offline_environment)
        self.runner = runner
        self.checkout_binding = validate_vbench_long_checkout(
            checkout,
            runner=self.runner,
            environment=self.environment,
        )
        self.checkout = self.checkout_binding.root
        self.eval_script = self.checkout_binding.eval_script
        self.full_info = self.checkout_binding.full_info
        self.runtime_cache = runtime_cache
        self._validate_checkout_and_python()

    def _run_probe(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            completed = self.runner(
                list(command),
                cwd=str(self.checkout),
                capture_output=True,
                text=True,
                check=False,
                env=self.environment,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise VBenchLongError(f"cannot execute preflight command: {error}") from error
        _require(isinstance(completed.returncode, int), "runner returned no status")
        return completed

    def _validate_checkout_and_python(self) -> None:
        python = self._run_probe(
            (
                self.python_executable,
                "-c",
                (
                    "import sys;"
                    f"sys.path.insert(0,{str(self.checkout)!r});"
                    "import torch,vbench,vbench2_beta_long;"
                    "print('vbench-long-python-ok')"
                ),
            )
        )
        _require(python.returncode == 0, "VBench Python environment preflight failed")
        _require(
            python.stdout.strip().splitlines()[-1:] == ["vbench-long-python-ok"],
            "VBench Python environment returned an invalid preflight marker",
        )
        # The dependency import must not have created ignored caches or any
        # other checkout drift.  The launcher performs this validation again
        # immediately before every official evaluation.
        self.checkout_binding = validate_vbench_long_checkout(
            self.checkout,
            runner=self.runner,
            environment=self.environment,
        )

    @staticmethod
    def _validate_prompt(prompt: str) -> str:
        _require(isinstance(prompt, str) and bool(prompt.strip()), "prompt is empty")
        _require(prompt == prompt.strip(), "prompt may not have outer whitespace")
        _require(
            len(prompt.encode("utf-8")) <= 180
            and prompt not in {".", ".."}
            and Path(prompt).name == prompt
            and not any(ord(character) < 32 for character in prompt),
            "prompt cannot be represented losslessly as a VBench-Long filename",
        )
        _require(
            _PROMPT_SUFFIX_RE.search(prompt) is None,
            "prompt suffix would be rewritten by official VBench-Long",
        )
        return prompt

    def _command(self, measurement_root: Path) -> list[str]:
        return [
            self.python_executable,
            str(self.launcher),
            "--asset-manifest",
            str(self.assets.manifest_path),
            "--asset-spec",
            str(self.assets.spec_path),
            "--checkout",
            str(self.checkout),
            "--eval-script",
            str(self.eval_script),
            "--eval-sha256",
            self.checkout_binding.eval_sha256,
            "--eval-nbytes",
            str(self.checkout_binding.eval_nbytes),
            "--full-info",
            str(self.full_info),
            "--full-info-sha256",
            self.checkout_binding.full_info_sha256,
            "--full-info-nbytes",
            str(self.checkout_binding.full_info_nbytes),
            "--runtime-cache-root",
            str(self.runtime_cache),
            "--",
            "--videos_path",
            str(measurement_root / "videos"),
            "--dimension",
            *VBENCH_LONG_DIMENSIONS,
            "--mode",
            "long_custom_input",
            "--dev_flag",
            "--load_ckpt_from_local",
            "True",
            "--sb_clip2clip_feat_extractor",
            VBENCH_LONG_SUBJECT_CLIP2CLIP_EXTRACTOR,
            "--bg_clip2clip_feat_extractor",
            VBENCH_LONG_BACKGROUND_CLIP2CLIP_EXTRACTOR,
            "--num_of_samples_per_prompt",
            "1",
            "--output_path",
            str(measurement_root / "results"),
            "--full_json_dir",
            str(self.full_info),
        ]

    @staticmethod
    def _recover_result(
        results_dir: Path,
        *,
        measurement_root: Path,
        staged_video: Path,
        staged_alias: str,
    ) -> tuple[Path, dict[str, float]] | None:
        if not results_dir.is_dir():
            return None
        valid: list[tuple[Path, dict[str, float]]] = []
        for path in sorted(results_dir.glob("*_eval_results.json")):
            try:
                valid.append(
                    (
                        path,
                        validate_vbench_long_official_result(
                            path,
                            measurement_root=measurement_root,
                            staged_video=staged_video,
                            staged_alias=staged_alias,
                        ),
                    )
                )
            except VBenchLongError:
                continue
        if not valid:
            return None
        reference = valid[0][1]
        _require(
            all(scores == reference for _, scores in valid),
            "multiple official result files disagree",
        )
        return valid[-1]

    def _successful_attempt_exists(
        self,
        root: Path,
        *,
        config_sha256: str,
        command: Sequence[str],
    ) -> bool:
        return validate_vbench_long_success_attempt(
            root,
            run_dir=self.run_dir,
            config_sha256=config_sha256,
            command=command,
        )

    def _measurement_from_sidecar(
        self,
        sidecar_path: Path,
        *,
        config_sha256: str,
        command: Sequence[str],
        measurement_root: Path,
        staged_video: Path,
        prompt: str,
        staged_alias: str,
    ) -> VBenchLongMeasurement:
        value = _load_object(sidecar_path)
        _require(
            sidecar_path.read_bytes() == _canonical_bytes(value) + b"\n",
            "measurement evidence is not canonical JSON",
        )
        _require(
            set(value)
            == {
                "schema",
                "measurement_id",
                "config_sha256",
                "official_result",
                "dimension_scores",
                "aggregation",
                "macro_mean",
                "motion_smoothness",
                "staging",
            }
            and value.get("schema") == MEASUREMENT_SCHEMA
            and value.get("measurement_id") == measurement_root.name
            and value.get("staging")
            == {
                "algorithm": STAGED_ALIAS_ALGORITHM,
                "alias": staged_alias,
                "original_prompt_id": prompt,
            }
            and value.get("aggregation")
            == "unweighted_arithmetic_mean_of_six_official_dimension_scores",
            "measurement schema, identity, or aggregation drifted",
        )
        _require(value.get("config_sha256") == config_sha256, "measurement config binding drifted")
        _require(
            self._successful_attempt_exists(
                measurement_root,
                config_sha256=config_sha256,
                command=command,
            ),
            "measurement has no successful attempt bound to current config and command",
        )
        scores_raw = value.get("dimension_scores")
        _require(isinstance(scores_raw, dict), "measurement has no dimension scores")
        _require(set(scores_raw) == set(VBENCH_LONG_DIMENSIONS), "measurement dimensions drifted")
        scores = {
            dimension: _finite_unit(scores_raw[dimension], dimension)
            for dimension in VBENCH_LONG_DIMENSIONS
        }
        macro = math.fsum(scores.values()) / len(scores)
        recorded_macro = _finite_unit(value.get("macro_mean"), "measurement macro mean")
        _require(abs(recorded_macro - macro) <= 1e-12, "measurement macro mean drifted")
        recorded_motion = _finite_unit(
            value.get("motion_smoothness"), "measurement motion smoothness"
        )
        _require(
            recorded_motion == scores["motion_smoothness"],
            "measurement motion smoothness drifted",
        )
        official_result = value.get("official_result")
        _require(isinstance(official_result, Mapping), "measurement official result is invalid")
        result_relative = official_result.get("path")
        _require(
            isinstance(result_relative, str)
            and bool(result_relative)
            and not Path(result_relative).is_absolute()
            and ".." not in Path(result_relative).parts,
            "measurement official result path is invalid",
        )
        result_path = (self.run_dir / result_relative).resolve()
        _require(
            result_path.is_relative_to(self.run_dir),
            "measurement official result escapes run_dir",
        )
        _require(result_path.is_file(), "measurement official result is missing")
        result_sha256, result_size = _sha256_file(result_path)
        _require(
            official_result.get("sha256") == result_sha256
            and official_result.get("nbytes") == result_size,
            "official result bytes drifted",
        )
        _require(
            validate_vbench_long_official_result(
                result_path,
                measurement_root=measurement_root,
                staged_video=staged_video,
                staged_alias=staged_alias,
            )
            == scores,
            "official scores drifted",
        )
        return VBenchLongMeasurement(
            dimension_scores=tuple((name, scores[name]) for name in VBENCH_LONG_DIMENSIONS),
            macro_mean=macro,
            motion_smoothness=scores["motion_smoothness"],
            evidence_path=sidecar_path.relative_to(self.run_dir).as_posix(),
        )

    def score_video(
        self,
        *,
        video_path: str | Path,
        prompt: str,
        measurement_id: str,
    ) -> VBenchLongMeasurement:
        """Score one complete video, resuming only from bound official output."""

        _require(
            isinstance(measurement_id, str)
            and _MEASUREMENT_RE.fullmatch(measurement_id) is not None,
            "measurement_id is not portable",
        )
        frozen_prompt = self._validate_prompt(prompt)
        source_video = Path(video_path).expanduser().resolve(strict=True)
        _require(source_video.is_file(), "video_path must be a regular file")
        video_sha256, video_size = _sha256_file(source_video)
        staged_alias = vbench_long_staged_alias(
            prompt_id=frozen_prompt,
            measurement_id=measurement_id,
            source_video_sha256=video_sha256,
        )
        root = self.run_dir / "vbench-long" / measurement_id
        videos_dir = root / "videos"
        results_dir = root / "results"
        videos_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        staged_video = videos_dir / f"{staged_alias}.mp4"
        other_staged_videos = [
            path for path in videos_dir.glob("*.mp4") if path != staged_video
        ]
        _require(
            not other_staged_videos,
            "VBench staging contains a video outside the bound alias",
        )
        if staged_video.exists():
            _require(
                staged_video.is_file()
                and not staged_video.is_symlink()
                and staged_video.samefile(source_video),
                "staged VBench video conflicts with the requested source",
            )
        else:
            try:
                os.link(source_video, staged_video)
            except OSError as error:
                raise VBenchLongError(
                    "cannot hard-link the full video into VBench-Long staging"
                ) from error
        command = self._command(root)
        config = {
            "schema": CONFIG_SCHEMA,
            "measurement_id": measurement_id,
            "upstream": self.checkout_binding.config_binding(),
            "python_executable": self.python_executable,
            "launcher": {
                "path": str(self.launcher),
                "sha256": self.launcher_sha256,
                "nbytes": self.launcher_nbytes,
            },
            "visible_device": self.visible_device,
            "mode": "long_custom_input",
            "dev_flag": True,
            "dimensions": list(VBENCH_LONG_DIMENSIONS),
            "offline_assets": self.assets.config_binding(),
            "offline_policy": {
                "network": OFFLINE_NETWORK_POLICY,
                "load_ckpt_from_local": True,
                "subject_clip2clip_extractor": (
                    VBENCH_LONG_SUBJECT_CLIP2CLIP_EXTRACTOR
                ),
                "background_clip2clip_extractor": (
                    VBENCH_LONG_BACKGROUND_CLIP2CLIP_EXTRACTOR
                ),
                "dev_flag_slow_fast_weights": "frozen_upstream_yaml",
                "scrubbed_parent_environment": list(
                    VBENCH_LONG_SCRUBBED_ENVIRONMENT_VARIABLES
                ),
            },
            "offline_environment": self.offline_environment,
            "prompt": frozen_prompt,
            "staging": {
                "algorithm": STAGED_ALIAS_ALGORITHM,
                "alias": staged_alias,
                "original_prompt_id": frozen_prompt,
            },
            "source_video": {
                "path": str(source_video),
                "sha256": video_sha256,
                "nbytes": video_size,
            },
            "staged_video": staged_video.relative_to(self.run_dir).as_posix(),
            "command": command,
        }
        config_path = root / "config.json"
        _write_bound_json(config_path, config)
        config_sha256, _ = _sha256_file(config_path)
        sidecar_path = root / "measurement.json"
        if sidecar_path.exists():
            return self._measurement_from_sidecar(
                sidecar_path,
                config_sha256=config_sha256,
                command=command,
                measurement_root=root,
                staged_video=staged_video,
                prompt=frozen_prompt,
                staged_alias=staged_alias,
            )

        successful_attempt = self._successful_attempt_exists(
            root,
            config_sha256=config_sha256,
            command=command,
        )
        if successful_attempt:
            recovered = self._recover_result(
                results_dir,
                measurement_root=root,
                staged_video=staged_video,
                staged_alias=staged_alias,
            )
            _require(
                recovered is not None,
                "successful VBench-Long attempt has no valid bound result JSON",
            )
        else:
            _require(
                not any(results_dir.glob("*_eval_results.json")),
                "unbound pre-existing VBench-Long result JSON is forbidden",
            )
            attempt_number = len(list((root / "attempts").glob("*.json"))) + 1
            attempt_id = f"{attempt_number:04d}"
            try:
                completed = self.runner(
                    command,
                    cwd=str(self.checkout),
                    capture_output=True,
                    text=True,
                    check=False,
                    env=self.environment,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise VBenchLongError(f"official VBench-Long launch failed: {error}") from error
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            stdout_path = root / "attempts" / f"{attempt_id}.stdout.log"
            stderr_path = root / "attempts" / f"{attempt_id}.stderr.log"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")
            stdout_sha256, stdout_size = _sha256_file(stdout_path)
            stderr_sha256, stderr_size = _sha256_file(stderr_path)
            _write_bound_json(
                root / "attempts" / f"{attempt_id}.json",
                {
                    "schema": ATTEMPT_SCHEMA,
                    "attempt": attempt_number,
                    "config_sha256": config_sha256,
                    "command": command,
                    "command_sha256": hashlib.sha256(
                        _canonical_bytes(command)
                    ).hexdigest(),
                    "returncode": completed.returncode,
                    "stdout": {"path": stdout_path.relative_to(self.run_dir).as_posix(), "sha256": stdout_sha256, "nbytes": stdout_size},
                    "stderr": {"path": stderr_path.relative_to(self.run_dir).as_posix(), "sha256": stderr_sha256, "nbytes": stderr_size},
                },
            )
            _require(
                completed.returncode == 0,
                f"official VBench-Long exited with status {completed.returncode}",
            )
            recovered = self._recover_result(
                results_dir,
                measurement_root=root,
                staged_video=staged_video,
                staged_alias=staged_alias,
            )
            _require(recovered is not None, "official VBench-Long produced no valid result JSON")

        result_path, scores = recovered
        current_sha256, current_size = _sha256_file(source_video)
        _require(
            (current_sha256, current_size) == (video_sha256, video_size),
            "source video changed during VBench-Long evaluation",
        )
        result_sha256, result_size = _sha256_file(result_path)
        macro = math.fsum(scores.values()) / len(scores)
        sidecar = {
            "schema": MEASUREMENT_SCHEMA,
            "measurement_id": measurement_id,
            "staging": {
                "algorithm": STAGED_ALIAS_ALGORITHM,
                "alias": staged_alias,
                "original_prompt_id": frozen_prompt,
            },
            "config_sha256": config_sha256,
            "official_result": {
                "path": result_path.relative_to(self.run_dir).as_posix(),
                "sha256": result_sha256,
                "nbytes": result_size,
            },
            "dimension_scores": scores,
            "aggregation": "unweighted_arithmetic_mean_of_six_official_dimension_scores",
            "macro_mean": macro,
            "motion_smoothness": scores["motion_smoothness"],
        }
        _write_bound_json(sidecar_path, sidecar)
        return self._measurement_from_sidecar(
            sidecar_path,
            config_sha256=config_sha256,
            command=command,
            measurement_root=root,
            staged_video=staged_video,
            prompt=frozen_prompt,
            staged_alias=staged_alias,
        )

    def score_videos(
        self,
        requests: Sequence[VBenchLongVideoRequest],
    ) -> dict[str, VBenchLongMeasurement]:
        """Score a lane's videos in one official process and split by detail alias."""

        _require(bool(requests), "VBench-Long batch request is empty")
        identities: list[dict[str, Any]] = []
        measurement_ids: set[str] = set()
        aliases: set[str] = set()
        for request in requests:
            _require(
                isinstance(request, VBenchLongVideoRequest),
                "VBench-Long batch request has an invalid type",
            )
            measurement_id = request.measurement_id
            _require(
                isinstance(measurement_id, str)
                and _MEASUREMENT_RE.fullmatch(measurement_id) is not None
                and measurement_id not in measurement_ids,
                "VBench-Long batch contains a duplicate or invalid measurement_id",
            )
            prompt = self._validate_prompt(request.prompt)
            source = Path(request.video_path).expanduser().resolve(strict=True)
            _require(source.is_file(), "VBench-Long batch video_path must be a regular file")
            sha256, nbytes = _sha256_file(source)
            alias = vbench_long_staged_alias(
                prompt_id=prompt,
                measurement_id=measurement_id,
                source_video_sha256=sha256,
            )
            _require(alias not in aliases, "VBench-Long batch staged alias is duplicated")
            measurement_ids.add(measurement_id)
            aliases.add(alias)
            identities.append(
                {
                    "alias": alias,
                    "measurement_id": measurement_id,
                    "prompt": prompt,
                    "source_video": {
                        "path": str(source),
                        "sha256": sha256,
                        "nbytes": nbytes,
                    },
                }
            )
        identities.sort(key=lambda item: str(item["measurement_id"]))
        batch_id = "batch-" + hashlib.sha256(_canonical_bytes(identities)).hexdigest()
        _require(_BATCH_ID_RE.fullmatch(batch_id) is not None, "internal batch ID is invalid")
        # Keep shared evidence below the already-published ``vbench-long/**``
        # namespace.  Existing manifests therefore include config/result/log
        # evidence and continue excluding the hard-linked ``videos/**`` bytes.
        root = self.run_dir / BATCH_ROOT_RELATIVE / batch_id
        videos_dir = root / "videos"
        results_dir = root / "results"
        videos_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        inputs: list[dict[str, Any]] = []
        staged_videos: dict[str, Path] = {}
        for identity in identities:
            alias = str(identity["alias"])
            source = Path(str(identity["source_video"]["path"])).resolve(strict=True)
            staged = videos_dir / f"{alias}.mp4"
            if staged.exists():
                _require(
                    staged.is_file()
                    and not staged.is_symlink()
                    and staged.samefile(source)
                    and _sha256_file(staged)
                    == (
                        identity["source_video"]["sha256"],
                        identity["source_video"]["nbytes"],
                    ),
                    "staged VBench batch video conflicts with its requested source",
                )
            else:
                try:
                    os.link(source, staged)
                except OSError as error:
                    raise VBenchLongError(
                        "cannot hard-link a full video into VBench-Long batch staging"
                    ) from error
            record = dict(identity)
            record["staged_video"] = staged.relative_to(self.run_dir).as_posix()
            inputs.append(record)
            staged_videos[alias] = staged
        _require(
            {path.name for path in videos_dir.glob("*.mp4")}
            == {f"{alias}.mp4" for alias in aliases},
            "VBench batch staging contains a video outside the bound alias set",
        )

        command = self._command(root)
        config = {
            "schema": BATCH_CONFIG_SCHEMA,
            "batch_id": batch_id,
            "upstream": self.checkout_binding.config_binding(),
            "python_executable": self.python_executable,
            "launcher": {
                "path": str(self.launcher),
                "sha256": self.launcher_sha256,
                "nbytes": self.launcher_nbytes,
            },
            "visible_device": self.visible_device,
            "mode": "long_custom_input",
            "dev_flag": True,
            "dimensions": list(VBENCH_LONG_DIMENSIONS),
            "offline_assets": self.assets.config_binding(),
            "offline_policy": {
                "network": OFFLINE_NETWORK_POLICY,
                "load_ckpt_from_local": True,
                "subject_clip2clip_extractor": VBENCH_LONG_SUBJECT_CLIP2CLIP_EXTRACTOR,
                "background_clip2clip_extractor": VBENCH_LONG_BACKGROUND_CLIP2CLIP_EXTRACTOR,
                "dev_flag_slow_fast_weights": "frozen_upstream_yaml",
                "scrubbed_parent_environment": list(
                    VBENCH_LONG_SCRUBBED_ENVIRONMENT_VARIABLES
                ),
            },
            "offline_environment": self.offline_environment,
            "inputs": inputs,
            "command": command,
        }
        config_path = root / "config.json"
        _write_bound_json(config_path, config)
        config_sha256, _config_nbytes = _sha256_file(config_path)

        result_path = validate_vbench_long_batch_success_attempt(
            root,
            run_dir=self.run_dir,
            batch_id=batch_id,
            config_sha256=config_sha256,
            command=command,
            inputs=inputs,
        )
        if result_path is None:
            _require(
                not any(results_dir.glob("*_eval_results.json")),
                "unbound pre-existing VBench-Long batch result JSON is forbidden",
            )
            attempt_number = len(list((root / "attempts").glob("*.json"))) + 1
            attempt_id = f"{attempt_number:04d}"
            try:
                completed = self.runner(
                    command,
                    cwd=str(self.checkout),
                    capture_output=True,
                    text=True,
                    check=False,
                    env=self.environment,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise VBenchLongError(
                    f"official VBench-Long batch launch failed: {error}"
                ) from error
            attempts_dir = root / "attempts"
            attempts_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = attempts_dir / f"{attempt_id}.stdout.log"
            stderr_path = attempts_dir / f"{attempt_id}.stderr.log"
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")
            candidates = sorted(results_dir.glob("*_eval_results.json"))
            bound_result: dict[str, Any] | None = None
            if completed.returncode == 0:
                _require(
                    len(candidates) == 1,
                    "official VBench-Long batch did not produce exactly one result JSON",
                )
                result_path = candidates[0]
                bound_result = _file_binding(result_path, run_dir=self.run_dir)
            attempt_path = attempts_dir / f"{attempt_id}.json"
            _write_bound_json(
                attempt_path,
                {
                    "schema": BATCH_ATTEMPT_SCHEMA,
                    "attempt": attempt_number,
                    "batch_id": batch_id,
                    "config_sha256": config_sha256,
                    "inputs": inputs,
                    "command": command,
                    "command_sha256": hashlib.sha256(
                        _canonical_bytes(command)
                    ).hexdigest(),
                    "returncode": completed.returncode,
                    "stdout": _file_binding(stdout_path, run_dir=self.run_dir),
                    "stderr": _file_binding(stderr_path, run_dir=self.run_dir),
                    "official_result": bound_result,
                },
            )
            _require(
                completed.returncode == 0,
                f"official VBench-Long batch exited with status {completed.returncode}",
            )
            assert result_path is not None

        scores_by_alias = validate_vbench_long_official_batch_result(
            result_path,
            measurement_root=root,
            staged_videos=staged_videos,
        )
        successful_result = validate_vbench_long_batch_success_attempt(
            root,
            run_dir=self.run_dir,
            batch_id=batch_id,
            config_sha256=config_sha256,
            command=command,
            inputs=inputs,
        )
        _require(
            successful_result is not None
            and successful_result.resolve(strict=True) == result_path.resolve(strict=True),
            "official VBench-Long batch result is not bound to a successful attempt",
        )
        successful_attempts = []
        for candidate in sorted((root / "attempts").glob("*.json")):
            attempt_value = _load_object(candidate)
            if attempt_value.get("returncode") == 0:
                successful_attempts.append(candidate)
        _require(bool(successful_attempts), "VBench-Long batch has no successful attempt file")
        attempt_path = successful_attempts[-1]

        measurements: dict[str, VBenchLongMeasurement] = {}
        for item in inputs:
            measurement_id = str(item["measurement_id"])
            alias = str(item["alias"])
            source = Path(str(item["source_video"]["path"])).resolve(strict=True)
            _require(
                _sha256_file(source)
                == (item["source_video"]["sha256"], item["source_video"]["nbytes"]),
                "source video changed during VBench-Long batch evaluation",
            )
            scores = scores_by_alias[alias]
            macro = math.fsum(scores.values()) / len(scores)
            measurement_root = self.run_dir / "vbench-long" / measurement_id
            measurement_root.mkdir(parents=True, exist_ok=True)
            sidecar_path = measurement_root / "measurement.json"
            sidecar = {
                "schema": BATCH_MEASUREMENT_SCHEMA,
                "measurement_id": measurement_id,
                "batch_id": batch_id,
                "batch_config": _file_binding(config_path, run_dir=self.run_dir),
                "batch_attempt": _file_binding(attempt_path, run_dir=self.run_dir),
                "official_result": _file_binding(result_path, run_dir=self.run_dir),
                "dimension_scores": scores,
                "aggregation": "unweighted_arithmetic_mean_of_six_official_dimension_scores",
                "macro_mean": macro,
                "motion_smoothness": scores["motion_smoothness"],
                "staging": {
                    "algorithm": STAGED_ALIAS_ALGORITHM,
                    "alias": alias,
                    "original_prompt_id": item["prompt"],
                    "staged_video": item["staged_video"],
                },
                "source_video": item["source_video"],
            }
            _write_bound_json(sidecar_path, sidecar)
            measurements[measurement_id] = validate_vbench_long_batch_measurement(
                sidecar_path,
                run_dir=self.run_dir,
                measurement_id=measurement_id,
                prompt=str(item["prompt"]),
                video_path=source,
            )
        _require(
            set(measurements) == measurement_ids,
            "VBench-Long batch measurements do not exactly cover the request",
        )
        return measurements


def score_vbench_long_video(
    *,
    run_dir: str | Path,
    checkout: str | Path,
    python_executable: str | Path,
    asset_manifest: str | Path,
    asset_spec: str | Path,
    visible_device: str | None = None,
    video_path: str | Path,
    prompt: str,
    measurement_id: str,
    runner: CommandRunner = subprocess.run,
) -> VBenchLongMeasurement:
    """Functional entry point suitable for a quality-evaluator adapter."""

    return OfficialVBenchLongScorer(
        run_dir=run_dir,
        checkout=checkout,
        python_executable=python_executable,
        asset_manifest=asset_manifest,
        asset_spec=asset_spec,
        visible_device=visible_device,
        runner=runner,
    ).score_video(
        video_path=video_path,
        prompt=prompt,
        measurement_id=measurement_id,
    )


__all__ = [
    "ATTEMPT_SCHEMA",
    "BATCH_ATTEMPT_SCHEMA",
    "BATCH_CONFIG_SCHEMA",
    "BATCH_MEASUREMENT_SCHEMA",
    "BATCH_ROOT_RELATIVE",
    "CHECKOUT_TREE_HASH_ALGORITHM",
    "CONFIG_SCHEMA",
    "MEASUREMENT_SCHEMA",
    "OFFLINE_ASSET_SCHEMA",
    "OFFLINE_NETWORK_POLICY",
    "OFFLINE_TREE_HASH_ALGORITHM",
    "STAGED_ALIAS_ALGORITHM",
    "OfficialVBenchLongScorer",
    "VBENCH_LONG_COMMIT",
    "VBENCH_LONG_DIMENSIONS",
    "VBENCH_LONG_BACKGROUND_CLIP2CLIP_EXTRACTOR",
    "VBENCH_LONG_CREDENTIAL_ENVIRONMENT_VARIABLES",
    "VBENCH_LONG_OFFLINE_FILE_PATHS",
    "VBENCH_LONG_OFFLINE_TREE_PATHS",
    "VBENCH_LONG_REPOSITORY",
    "VBENCH_LONG_SCRUBBED_ENVIRONMENT_VARIABLES",
    "VBENCH_LONG_SUBJECT_CLIP2CLIP_EXTRACTOR",
    "VBenchLongError",
    "VBenchLongMeasurement",
    "VBenchLongOfflineAssets",
    "VBenchLongCheckout",
    "VBenchLongVideoRequest",
    "score_vbench_long_video",
    "vbench_long_staged_alias",
    "validate_vbench_long_offline_assets",
    "validate_vbench_long_checkout",
    "validate_vbench_long_official_result",
    "validate_vbench_long_official_batch_result",
    "validate_vbench_long_batch_measurement",
    "validate_vbench_long_batch_success_attempt",
    "validate_vbench_long_success_attempt",
]
