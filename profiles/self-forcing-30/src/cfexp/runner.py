from __future__ import annotations

import errno
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlsplit

from .host import (
    FORMAL_HF_ARTIFACT_REMOTE,
    HostConfig,
    configure_hf_environment,
    ensure_host_directories,
    load_secrets,
    resolve_hf_token,
)
from .manifest import load_manifest, manifest_hash, validate_manifest
from .metadata import (
    collect_git_metadata,
    collect_gpu_metadata,
    collect_machine_metadata,
    find_git_root,
    safe_component,
    secret_presence,
    sha256_file,
    utc_now,
    write_json,
)
from .storage import (
    VIDEO_SUFFIXES,
    StorageError,
    preflight_destination,
    publish_run,
    seal_run_artifacts,
)
from .signal_acceptance import (
    EXTERNAL_PRIOR_2B_EVAL_ACCEPTANCE_POLICY,
    SignalAcceptance,
    SignalAcceptanceError,
    approval_root,
    load_signal_acceptance,
    requires_signal_acceptance,
    signal_acceptance_policy,
)


class ExperimentError(RuntimeError):
    """Raised for reproducibility or execution failures."""


class _UnsafeRuntimeCleanup(ExperimentError):
    """Raised when a runtime root may still be in use by an owned process group."""


_RUNTIME_CLEANUP_ATTEMPTS = 6
_RUNTIME_CLEANUP_BACKOFF_SECONDS = 0.5


_INHERITED_ENVIRONMENT_NAMES = frozenset(
    {
        "ALL_PROXY",
        "CURL_CA_BUNDLE",
        "DOCKER_CERT_PATH",
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "DOCKER_TLS_VERIFY",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "LOGNAME",
        "NO_PROXY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TZ",
        "USER",
        "XDG_RUNTIME_DIR",
        "all_proxy",
        "https_proxy",
        "http_proxy",
        "no_proxy",
    }
)

_MANAGED_ENVIRONMENT_NAMES = frozenset(
    {
        "CFEXP_ATTEMPT_ID",
        "CFEXP_CACHE_DIR",
        "CFEXP_MODEL_ROOT",
        "CFEXP_RUN_DIR",
        "CFEXP_UV_BIN",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_ENDPOINT",
        "HF_TOKEN",
        "HF_XET_CACHE",
        "MODEL_ROOT",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "TMPDIR",
        "TORCH_HOME",
        "UV_CACHE_DIR",
        "UV_PROJECT_ENVIRONMENT",
    }
)


@dataclass(frozen=True)
class Plan:
    manifest: dict[str, object]
    spec_hash: str
    repo_root: Path
    git: dict[str, object]
    lock_sha256: str | None
    gpu_ids: list[str]
    command: list[str]
    artifact_remote: str
    repository_inputs: list[dict[str, object]] = field(default_factory=list)

    def serializable(self, host: HostConfig) -> dict[str, object]:
        spec = self.manifest["spec"]
        assert isinstance(spec, dict)
        gate_required = _requires_signal_acceptance(self.manifest)
        serialized: dict[str, object] = {
            "name": self.manifest["metadata"]["name"],  # type: ignore[index]
            "spec_hash": self.spec_hash,
            "repo_root": str(self.repo_root),
            "git": self.git,
            "lock_sha256": self.lock_sha256,
            "gpu_ids": self.gpu_ids,
            "command": self.command,
            "artifact_remote": self.artifact_remote or None,
            "repository_inputs": self.repository_inputs,
            "insight2_gate_receipt_required": gate_required,
            "host": host.serializable(),
            "resources": spec["resources"],
        }
        if (
            not gate_required
            or signal_acceptance_policy(self.manifest)
            != EXTERNAL_PRIOR_2B_EVAL_ACCEPTANCE_POLICY
        ):
            serialized["insight2a_signal_acceptance_required"] = gate_required
        return serialized


def _requires_signal_acceptance(manifest: dict[str, object]) -> bool:
    """Return the validated gate state without imposing Insight 2 on other jobs."""

    spec = manifest.get("spec")
    if not isinstance(spec, dict):
        return False
    parameters = spec.get("parameters")
    if not isinstance(parameters, dict):
        return False
    phase = parameters.get("phase")
    if phase not in {"2b-smoke", "2b-eval"} and "run_only_after_2a_signal_check" not in parameters:
        return False
    try:
        return requires_signal_acceptance(manifest)
    except SignalAcceptanceError as error:
        raise ExperimentError(str(error)) from error


def _git_resolve(repo_root: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{revision}^{{commit}}"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise ExperimentError(f"cannot resolve Git revision {revision!r}: {result.stdout.strip()}")
    return result.stdout.strip()


def _gpu_ids(value: str | None) -> list[str]:
    if value is None:
        value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if any(not item.isdigit() for item in ids):
        raise ExperimentError("GPU ids must be a comma-separated list of non-negative integers")
    if len(ids) != len(set(ids)):
        raise ExperimentError("GPU ids must not contain duplicates")
    return ids


def _artifact_remote(manifest: dict[str, object], host: HostConfig) -> str:
    spec = manifest["spec"]
    assert isinstance(spec, dict)
    storage = spec["storage"]
    assert isinstance(storage, dict)
    remote = str(storage.get("remote", "")).strip()
    if not remote or "${" in remote:
        remote = host.artifact_remote.strip()
    return remote


def _experiment_command(
    manifest: dict[str, object],
    repo_root: Path,
    host: HostConfig,
    gpu_ids: list[str],
    run_dir: Path | None = None,
) -> list[str]:
    spec = manifest["spec"]
    assert isinstance(spec, dict)
    runtime = spec["runtime"]
    command = spec["command"]
    assert isinstance(runtime, dict) and isinstance(command, dict)
    argv = [str(item) for item in command["argv"]]
    backend = runtime["backend"]
    if backend == "uv":
        uv = _uv_executable()
        return [
            uv,
            "--project",
            str(repo_root),
            "--no-config",
            "run",
            "--no-env-file",
            "--python",
            str(runtime["python"]),
            "--frozen",
            *argv,
        ]

    if run_dir is None:
        return ["docker", "run", "<runtime mounts>", str(runtime["image"]), *argv]
    engine = runtime.get("engine", "docker")
    if engine != "docker":
        raise ExperimentError(f"unsupported image engine {engine!r}; only docker is implemented")
    docker = shutil.which("docker")
    if not docker:
        raise ExperimentError("docker is required for runtime.backend=image")
    cwd = str(command.get("cwd", ".")).strip("/")
    docker_command = [
        docker,
        "run",
        "--rm",
        "--name",
        f"cfexp-{run_dir.name[-24:]}",
        "--mount",
        f"type=bind,src={repo_root},dst=/workspace,readonly",
        "--mount",
        f"type=bind,src={run_dir},dst=/run",
        "--mount",
        f"type=bind,src={host.model_root},dst=/models,readonly",
        "--mount",
        f"type=bind,src={host.cache_root},dst=/cache",
        "--mount",
        f"type=bind,src={host.tmp_root},dst=/scratch",
        "--workdir",
        f"/workspace/{cwd}" if cwd and cwd != "." else "/workspace",
    ]
    if host.hf_proxy and (urlsplit(host.hf_proxy).hostname or "").lower() in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        # A loopback proxy belongs to the host. Linux host networking preserves
        # that address inside the pinned execution image.
        docker_command.extend(["--network", "host"])
    if gpu_ids:
        docker_command.extend(["--gpus", f"device={','.join(gpu_ids)}"])
    docker_command.extend(
        [
            "--env",
            "CFEXP_RUN_DIR=/run",
            "--env",
            f"CFEXP_ATTEMPT_ID={run_dir.name}",
            "--env",
            "CFEXP_CACHE_DIR=/cache",
            "--env",
            "CFEXP_MODEL_ROOT=/models",
            "--env",
            "MODEL_ROOT=/models",
            "--env",
            "HF_HOME=/cache/huggingface",
            "--env",
            "HF_HUB_CACHE=/cache/huggingface/hub",
            "--env",
            "HF_ENDPOINT=https://huggingface.co",
            "--env",
            "HF_XET_CACHE=/cache/huggingface/xet",
            "--env",
            "UV_CACHE_DIR=/cache/uv",
            "--env",
            "TORCH_HOME=/cache/torch",
            "--env",
            "PYTHONNOUSERSITE=1",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONHASHSEED=0",
            "--env",
            "TMPDIR=/run/tmp",
        ]
    )
    if host.hf_proxy:
        for name in (
            "ALL_PROXY",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "all_proxy",
            "https_proxy",
            "http_proxy",
        ):
            docker_command.extend(["--env", f"{name}={host.hf_proxy}"])
    for name, value in command.get("env", {}).items():
        docker_command.extend(["--env", f"{name}={value}"])
    for name in command.get("secret_env", []):
        docker_command.extend(["--env", str(name)])
    docker_command.extend([str(runtime["image"]), *argv])
    return docker_command


def _uv_executable() -> str:
    """Resolve a launcher-pinned uv binary without widening PATH."""

    configured = os.environ.get("CFEXP_UV_BIN")
    if not configured:
        return shutil.which("uv") or "uv"
    path = Path(configured)
    if not path.is_absolute():
        raise ExperimentError("CFEXP_UV_BIN must be an absolute path")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ExperimentError("CFEXP_UV_BIN is not a usable executable") from error
    if resolved != path or not stat.S_ISREG(metadata.st_mode):
        raise ExperimentError("CFEXP_UV_BIN must be a canonical regular file")
    if metadata.st_uid not in {0, os.getuid()} or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise ExperimentError("CFEXP_UV_BIN has unsafe ownership or permissions")
    if not os.access(path, os.X_OK):
        raise ExperimentError("CFEXP_UV_BIN is not executable")
    return str(path)


def _verify_repository_inputs(
    inputs: object,
    repo_root: Path,
) -> list[dict[str, object]]:
    """Verify immutable files already staged by the checked-out repository.

    ``cfexp`` does not fetch arbitrary manifest inputs.  The only executable
    input form is ``repository://PATH``: PATH must exactly match the declared
    target, already be tracked by Git, stay below the repository without
    traversing symlinks, and match its pinned SHA-256.  Returning the verified
    size and digest gives planning and terminal provenance a concrete binding.
    """

    if inputs in (None, []):
        return []
    if not isinstance(inputs, list):
        raise ExperimentError("spec.inputs must be an array")
    try:
        root = repo_root.resolve(strict=True)
    except OSError as error:
        raise ExperimentError("cannot resolve the Git repository for inputs") from error
    verified: list[dict[str, object]] = []
    for index, raw in enumerate(inputs):
        label = f"spec.inputs[{index}]"
        if not isinstance(raw, dict):
            raise ExperimentError(f"{label} must be an object")
        name = raw.get("name")
        uri = raw.get("uri")
        target = raw.get("target")
        expected_sha256 = raw.get("sha256")
        if not all(isinstance(value, str) for value in (name, uri, target)):
            raise ExperimentError(f"{label} has invalid name, uri, or target")
        assert isinstance(name, str) and isinstance(uri, str) and isinstance(target, str)
        if uri != f"repository://{target}":
            raise ExperimentError(
                f"{label} is not an exact repository:// binding to its target"
            )
        if not isinstance(expected_sha256, str) or "revision" in raw:
            raise ExperimentError(f"{label} repository input requires only sha256")
        posix = PurePosixPath(target)
        if (
            target != str(posix)
            or posix.is_absolute()
            or not posix.parts
            or any(part in {"", ".", ".."} for part in posix.parts)
        ):
            raise ExperimentError(f"{label} target is not a safe repository path")
        candidate = root.joinpath(*posix.parts)
        current = root
        try:
            for part in posix.parts:
                current = current / part
                metadata = current.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise ExperimentError(f"{label} target traverses a symlink")
        except FileNotFoundError as error:
            raise ExperimentError(f"{label} target does not exist") from error
        except OSError as error:
            raise ExperimentError(f"cannot inspect {label} target") from error
        if not stat.S_ISREG(metadata.st_mode):
            raise ExperimentError(f"{label} target is not a regular file")
        try:
            candidate.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as error:
            raise ExperimentError(f"{label} target escapes the Git repository") from error
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", target],
            cwd=root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tracked.returncode != 0:
            raise ExperimentError(f"{label} target is not tracked by Git")
        actual_sha256 = sha256_file(candidate)
        if actual_sha256 != expected_sha256.casefold():
            raise ExperimentError(f"{label} target SHA-256 does not match its pin")
        verified.append(
            {
                "name": name,
                "uri": uri,
                "target": target,
                "sha256": actual_sha256,
                "size": metadata.st_size,
            }
        )
    return verified


def build_plan(
    manifest_path: Path,
    host: HostConfig,
    gpus: str | None,
    *,
    allow_dirty: bool = False,
) -> Plan:
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest)
    spec = manifest["spec"]
    assert isinstance(spec, dict)
    parameters = spec.get("parameters", {})
    if isinstance(parameters, dict) and parameters.get("template_only") is True:
        raise ExperimentError(
            "template_only manifests are non-runnable; derive and commit a formal manifest first"
        )
    _requires_signal_acceptance(manifest)
    repo_root = find_git_root(manifest_path.resolve().parent)
    git = collect_git_metadata(repo_root)
    code = spec["code"]
    runtime = spec["runtime"]
    resources = spec["resources"]
    assert isinstance(code, dict) and isinstance(runtime, dict) and isinstance(resources, dict)

    inputs = spec.get("inputs", [])
    checkpoint = spec.get("checkpoint", {})
    if isinstance(checkpoint, dict) and checkpoint.get("enabled"):
        raise ExperimentError(
            "enabled checkpoint/resume policy is not implemented; set spec.checkpoint.enabled=false"
        )

    revision_sha = _git_resolve(repo_root, str(code["revision"]))
    if revision_sha != git["sha"]:
        raise ExperimentError(
            f"working tree HEAD {git['sha']} does not match requested revision {revision_sha}"
        )
    if code.get("require_clean", True) and git["dirty"] and not allow_dirty:
        raise ExperimentError(
            "Git working tree is dirty; commit the experiment code or pass "
            "--allow-dirty for a local smoke test"
        )
    repository_inputs = _verify_repository_inputs(inputs, repo_root)

    selected_gpus = _gpu_ids(gpus)
    requested_gpus = int(resources["gpus"])
    if requested_gpus != len(selected_gpus):
        raise ExperimentError(
            f"manifest requests {requested_gpus} GPU(s), but "
            f"{len(selected_gpus)} id(s) were selected"
        )
    if requested_gpus:
        available = {str(item["index"]): item for item in collect_gpu_metadata()}
        missing = [gpu_id for gpu_id in selected_gpus if gpu_id not in available]
        if missing:
            raise ExperimentError(f"selected GPU ids are not visible to nvidia-smi: {missing}")
        minimum_gb = resources.get("min_gpu_memory_gb")
        if minimum_gb is not None:
            insufficient = []
            for gpu_id in selected_gpus:
                total_mib = available[gpu_id].get("memory_total_mib")
                if not isinstance(total_mib, int) or total_mib < float(minimum_gb) * 1024:
                    insufficient.append(gpu_id)
            if insufficient:
                raise ExperimentError(
                    f"GPU ids {insufficient} do not meet min_gpu_memory_gb={minimum_gb}"
                )

    lock_hash = None
    if runtime["backend"] == "uv":
        python_request = runtime.get("python")
        if not isinstance(python_request, str) or not python_request.strip():
            raise ExperimentError("runtime.backend=uv requires a non-empty runtime.python")
        lock_path = (repo_root / str(runtime["lockfile"])).resolve()
        try:
            lock_path.relative_to(repo_root)
        except ValueError as error:
            raise ExperimentError("lockfile must stay inside the Git repository") from error
        if not lock_path.is_file():
            raise ExperimentError(f"lockfile does not exist: {lock_path}")
        lock_hash = sha256_file(lock_path)

    return Plan(
        manifest=manifest,
        spec_hash=manifest_hash(manifest),
        repo_root=repo_root,
        git=git,
        lock_sha256=lock_hash,
        gpu_ids=selected_gpus,
        command=_experiment_command(manifest, repo_root, host, selected_gpus),
        artifact_remote=_artifact_remote(manifest, host),
        repository_inputs=repository_inputs,
    )


def _run_id(plan: Plan) -> str:
    metadata = plan.manifest["metadata"]
    assert isinstance(metadata, dict)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "-".join(
        [
            safe_component(str(metadata["name"])),
            timestamp,
            str(plan.git["sha"])[:8],
            plan.spec_hash[:8],
            uuid.uuid4().hex[:8],
        ]
    )


def _append_event(path: Path, event: str, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": utc_now(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _destination(remote: str, project: str, run_id: str) -> str:
    now = datetime.now(timezone.utc)
    separator = "" if remote.endswith(("/", ":")) else "/"
    return f"{remote}{separator}{project}/runs/{now:%Y}/{now:%m}/{run_id}"


def _has_durable_origin(remote: object) -> bool:
    """Return whether *remote* names a non-local Git transport.

    Formal runs must be recoverable from another machine. Local paths, bundle
    files, and file:// remotes do not provide that guarantee.
    """

    if not isinstance(remote, str) or not remote.strip():
        return False
    value = remote.strip()
    lowered = value.lower()
    if lowered.startswith("file://"):
        return False
    if value.startswith(("/", "./", "../", "~", "\\\\")):
        return False
    if lowered.endswith(".bundle"):
        return False
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if lowered.startswith(("https://", "http://", "ssh://", "git://")):
        try:
            parsed = urlsplit(value)
        except ValueError:
            return False
        scheme = parsed.scheme.lower()
        if parsed.password or parsed.query or parsed.fragment:
            return False
        if scheme in {"http", "https", "git"} and parsed.username is not None:
            return False
        if scheme == "ssh" and parsed.username not in {None, "git"}:
            return False
        hostname = parsed.hostname
        return bool(hostname and hostname.lower() not in local_hosts)
    # Git's SCP-like syntax, e.g. git@github.com:org/repository.git.  Do not
    # mistake a Windows drive letter for a network host.
    if len(value) >= 3 and value[1:3] in {":\\", ":/"}:
        return False
    host, separator, path = value.partition(":")
    if "@" in host:
        username, raw_hostname = host.rsplit("@", 1)
        if username != "git":
            return False
    else:
        raw_hostname = host
    hostname = raw_hostname.strip("[]").lower()
    return bool(
        separator
        and hostname
        and hostname not in local_hosts
        and path
        and "/" in path
        and " " not in host
    )


def _github_repository_slug(remote: object) -> str | None:
    """Return the canonical GitHub owner/repository named by a Git remote."""

    if not isinstance(remote, str) or not remote.strip():
        return None
    value = remote.strip()
    hostname = ""
    repository_path = ""
    if value.lower().startswith(("https://", "http://", "ssh://", "git://")):
        try:
            parsed = urlsplit(value)
        except ValueError:
            return None
        if parsed.query or parsed.fragment or parsed.password:
            return None
        if parsed.scheme.lower() == "ssh" and parsed.username != "git":
            return None
        hostname = (parsed.hostname or "").lower()
        repository_path = parsed.path
    else:
        host, separator, repository_path = value.partition(":")
        if not separator:
            return None
        if "@" not in host:
            return None
        username, hostname = host.rsplit("@", 1)
        if username != "git":
            return None
        hostname = hostname.strip("[]").lower()
    if hostname != "github.com":
        return None
    parts = repository_path.strip("/").split("/")
    if len(parts) != 2:
        return None
    owner, repository = parts
    if repository.lower().endswith(".git"):
        repository = repository[:-4]
    if not owner or not repository:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(character not in allowed for character in owner + repository):
        return None
    return f"{owner}/{repository}".casefold()


def _verify_github_repository_private(repository: str) -> None:
    """Fail unless GitHub hides an existing repository from anonymous API access."""

    api_url = f"https://api.github.com/repos/{repository}"
    request = urlrequest.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "cfexp-private-repository-check",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    # Ignore ambient proxies (including the host-local HF SOCKS relay): GitHub
    # is checked directly and without credentials so HTTP 404 proves it is not
    # anonymously visible. Authenticated git ls-remote separately proves the
    # same repository exists and contains this commit.
    opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))
    try:
        with opener.open(request, timeout=20) as response:
            status = response.status
    except urlerror.HTTPError as error:
        if error.code == 404:
            return
        raise ExperimentError(
            "could not verify that the configured GitHub repository is private"
        ) from None
    except (OSError, urlerror.URLError):
        raise ExperimentError(
            "could not verify that the configured GitHub repository is private"
        ) from None
    if status == 200:
        raise ExperimentError("configured GitHub repository is publicly readable")
    raise ExperimentError("could not verify that the configured GitHub repository is private")


def _git_execution_state(metadata: dict[str, object]) -> tuple[object, ...]:
    return (
        metadata.get("sha"),
        tuple(metadata.get("status", [])),
        metadata.get("remote"),
        metadata.get("remote_has_embedded_credentials"),
        metadata.get("remote_url_rewrite_configured"),
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.expanduser().resolve()
    right = right.expanduser().resolve()
    return left == right or left in right.parents or right in left.parents


def _ensure_formal_path_isolation(plan: Plan, host: HostConfig) -> None:
    """Keep generated data and publisher credentials outside the code tree."""

    signal_approval_root = approval_root(host)
    managed_roots = {
        "model_root": host.model_root,
        "cache_root": host.cache_root,
        "run_root": host.run_root,
        "tmp_root": host.tmp_root,
        "Insight 2 signal approval root": signal_approval_root,
    }
    for name, path in managed_roots.items():
        if _paths_overlap(plan.repo_root, path):
            raise ExperimentError(
                f"formal runs require {name} to be outside the Git repository"
            )
    for name, path in managed_roots.items():
        if name == "Insight 2 signal approval root":
            continue
        if _paths_overlap(signal_approval_root, path):
            raise ExperimentError(
                "formal runs require the Insight 2 signal approval root to be "
                f"isolated from {name}"
            )
    protected_roots = {"Git repository": plan.repo_root, **managed_roots}
    secret_path = host.secrets_file.expanduser().resolve()
    for name, path in protected_roots.items():
        root = path.expanduser().resolve()
        if secret_path == root or root in secret_path.parents:
            raise ExperimentError(
                f"publisher secrets_file must be outside the experiment-visible {name}"
            )


def _repository_video_violations(repo_root: Path) -> list[str]:
    """Find generated videos or directory symlinks without following links."""

    root = repo_root.expanduser().resolve()
    failures: list[OSError] = []

    def on_error(error: OSError) -> None:
        failures.append(error)

    violations: list[str] = []
    for current, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=on_error,
    ):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in directory_names:
            path = current_path / name
            try:
                if path.is_symlink():
                    if stat.S_ISDIR(path.stat().st_mode):
                        resolved_target = path.resolve()
                        if resolved_target != root and root not in resolved_target.parents:
                            relative = path.relative_to(root).as_posix()
                            violations.append(f"{relative}/ (external directory symlink)")
                    continue
            except OSError as error:
                failures.append(error)
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in file_names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            try:
                is_symlink = path.is_symlink()
                path.lstat()
                if is_symlink:
                    resolved_target = path.resolve()
                    target_is_external = (
                        resolved_target != root and root not in resolved_target.parents
                    )
                    if target_is_external:
                        violations.append(f"{relative} (external file symlink)")
            except OSError as error:
                failures.append(error)
                continue
            if path.suffix.casefold() in VIDEO_SUFFIXES:
                violations.append(relative)
    if failures:
        raise ExperimentError(
            "could not completely scan the Git repository for escaped video outputs"
        )
    return sorted(set(violations))


def _verify_remote_commit(repo_root: Path, sha: str, remote: str = "origin") -> None:
    """Prove that *sha* is the tip of a branch or tag advertised by origin."""

    try:
        result = subprocess.run(
            ["git", "ls-remote", "--heads", "--tags", remote],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise ExperimentError(f"could not verify Git commit on origin: {error}") from error
    if result.returncode != 0:
        raise ExperimentError(
            f"could not verify current Git commit {sha} against origin; "
            "check the remote and authentication, then publish the code first"
        )
    advertised = {
        line.split(None, 1)[0]
        for line in result.stdout.splitlines()
        if line.strip() and len(line.split(None, 1)[0]) == 40
    }
    if sha not in advertised:
        raise ExperimentError(
            f"current Git commit {sha} is not advertised by an origin branch or tag; "
            "publish the current branch or create a durable tag first"
        )


def _publication_hf_token(host: HostConfig) -> str | None:
    """Resolve the HF publishing token without exposing it to experiment code."""

    return resolve_hf_token(host)


def _command_environment(
    plan: Plan,
    host: HostConfig,
    attempt_dir: Path,
    attempt_id: str,
) -> tuple[dict[str, str], list[str]]:
    spec = plan.manifest["spec"]
    assert isinstance(spec, dict)
    command = spec["command"]
    assert isinstance(command, dict)
    environment = {
        name: os.environ[name]
        for name in _INHERITED_ENVIRONMENT_NAMES
        if os.environ.get(name) is not None
    }
    names = [str(name) for name in command.get("secret_env", [])]
    literal_names = {str(name) for name in command.get("env", {})}
    protected_literals = sorted(literal_names & _MANAGED_ENVIRONMENT_NAMES)
    if protected_literals:
        raise ExperimentError(
            f"command.env may not override cfexp-managed variables: {protected_literals}"
        )
    protected_secrets = sorted(set(names) & _MANAGED_ENVIRONMENT_NAMES)
    if protected_secrets:
        raise ExperimentError(
            f"secret_env may not override cfexp-managed variables: {protected_secrets}"
        )
    secrets = load_secrets(host.secrets_file)
    for name in names:
        if name in secrets:
            environment[name] = secrets[name]
        # A scheduler may inject a declared secret directly. Ambient values
        # for undeclared names are never inherited.
        if name in os.environ:
            environment[name] = os.environ[name]
    environment.update({str(key): str(value) for key, value in command.get("env", {}).items()})
    environment.update(
        {
            "CFEXP_RUN_DIR": str(attempt_dir),
            "CFEXP_ATTEMPT_ID": attempt_id,
            "CFEXP_CACHE_DIR": str(host.cache_root),
            "CFEXP_MODEL_ROOT": str(host.model_root),
            "MODEL_ROOT": str(host.model_root),
            "HF_HOME": str(host.cache_root / "huggingface"),
            "HF_HUB_CACHE": str(host.cache_root / "huggingface" / "hub"),
            "HF_ENDPOINT": "https://huggingface.co",
            "HF_XET_CACHE": str(host.cache_root / "huggingface" / "xet"),
            "UV_CACHE_DIR": str(host.cache_root / "uv"),
            "TORCH_HOME": str(host.cache_root / "torch"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            # The experiment's uv environment must never be created in the
            # checkout. The parent is made with mode 0700 immediately before
            # execution and removed after the entire process group exits.
            "UV_PROJECT_ENVIRONMENT": str(
                host.tmp_root / f"cfexp-runtime-{attempt_id}" / "venv"
            ),
            # Ordinary scratch output belongs to this run too; this prevents a
            # finished video written to $TMPDIR from bypassing publication.
            "TMPDIR": str(attempt_dir / "tmp"),
        }
    )
    # An explicit empty value keeps CPU manifests from accidentally seeing all
    # host GPUs when CUDA_VISIBLE_DEVICES was absent in the parent shell.
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(plan.gpu_ids)
    missing = [name for name in names if not environment.get(name)]
    if missing:
        raise ExperimentError(f"required secret environment variables are missing: {missing}")
    return environment, names


def _parse_linux_proc_stat(raw: str, *, expected_pid: int) -> dict[str, object]:
    """Extract only non-argument identity fields from one Linux proc stat row."""
    opening, closing = raw.find("("), raw.rfind(")")
    if opening < 1 or closing <= opening:
        raise ValueError("invalid proc stat delimiters")
    pid = int(raw[:opening].strip())
    # comm may contain spaces, newlines and parentheses; the final ')' is the
    # delimiter because all remaining stat fields are scalar values.
    fields = raw[closing + 1:].split()
    if pid != expected_pid or pid <= 0 or len(fields) < 20 or len(fields[0]) != 1:
        raise ValueError("invalid proc stat identity")
    ppid, pgid, sid = (int(fields[index]) for index in (1, 2, 3))
    starttime = int(fields[19])  # stat field 22, after state at field 3.
    if min(ppid, pgid, sid, starttime) < 0:
        raise ValueError("invalid proc stat counters")
    return {
        "pid": pid, "ppid": ppid, "pgid": pgid, "sid": sid,
        "state": fields[0], "starttime": starttime,
        "comm": raw[opening + 1:closing],
    }


def _snapshot_process_group(
    process_group: int, *, proc_root: Path = Path("/proc"), platform: str | None = None,
) -> dict[str, object]:
    """Best-effort Linux evidence, never a replacement for killpg inspection.

    No command lines or environments are read.  The scan is non-atomic: an
    empty member list is not evidence that a group is absent.  Entries whose
    stat disappears with ENOENT/ESRCH are counted as exit races; other read or
    parse failures make the result partial.  Failure details contain exception
    types only, never exception messages or proc contents.
    """
    members: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    result: dict[str, object] = {
        "schema": "linux_process_group_snapshot_v1", "status": "available",
        "pgid": process_group, "members": members, "exited_during_scan": 0,
        "errors": errors, "non_atomic": True,
    }
    if (sys.platform if platform is None else platform) != "linux":
        result["status"] = "unsupported"
        return result
    try:
        entries = sorted(
            (entry for entry in proc_root.iterdir()
             if entry.name.isascii() and entry.name.isdigit()),
            key=lambda entry: int(entry.name),
        )
        for entry in entries:
            pid = int(entry.name)
            try:
                member = _parse_linux_proc_stat(
                    (entry / "stat").read_text(encoding="utf-8", errors="replace"),
                    expected_pid=pid,
                )
            except (FileNotFoundError, ProcessLookupError):
                result["exited_during_scan"] = int(result["exited_during_scan"]) + 1
                continue
            except Exception as error:
                errors.append({"pid": pid, "error_type": type(error).__name__})
                result["status"] = "partial"
                continue
            if member["pgid"] == process_group:
                members.append(member)
    except Exception as error:
        errors.append({"pid": None, "error_type": type(error).__name__})
        result["status"] = "error"
    return result


def _execute(
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    event_path: Path,
    redactions: list[str] | None = None,
) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _append_event(event_path, "process_started", argv=argv)
    interrupted = False
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as error:
            log.write(f"cfexp could not start process: {error}\n")
            _append_event(event_path, "process_start_failed", error=str(error), exit_code=127)
            return 127, "FAILED"

        handled_signals = [signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            handled_signals.append(signal.SIGHUP)
        previous_handlers = {item: signal.getsignal(item) for item in handled_signals}
        forwarded_signal = signal.SIGINT
        process_group_permission_denied = False

        def _interrupt_from_signal(signum: int, _frame: object) -> None:
            nonlocal forwarded_signal
            forwarded_signal = signum
            raise KeyboardInterrupt

        def _record_process_group_permission_denied() -> None:
            nonlocal process_group_permission_denied
            if process_group_permission_denied:
                return
            process_group_permission_denied = True
            log.write(
                "cfexp could not inspect or terminate the experiment process group; "
                "failing the attempt so it is never treated as safely completed\n"
            )
            _append_event(event_path, "process_group_permission_denied")

        def _signal_process_group(signum: int) -> bool:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                return False
            except PermissionError:
                _record_process_group_permission_denied()
                return False
            return True

        for item in handled_signals:
            signal.signal(item, _interrupt_from_signal)
        try:
            assert process.stdout is not None
            secret_values = sorted(
                {value for value in (redactions or []) if value},
                key=len,
                reverse=True,
            )
            for line in process.stdout:
                safe_line = line
                for value in secret_values:
                    safe_line = safe_line.replace(value, "<redacted>")
                sys.stdout.write(safe_line)
                log.write(safe_line)
            return_code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            _signal_process_group(forwarded_signal)
            try:
                return_code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                _signal_process_group(signal.SIGTERM)
                try:
                    return_code = process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    _signal_process_group(signal.SIGKILL)
                    return_code = process.wait()
        finally:
            if process.stdout is not None:
                process.stdout.close()
            for item, previous_handler in previous_handlers.items():
                signal.signal(item, previous_handler)
        residual_process_group = False
        if hasattr(os, "killpg"):
            try:
                os.killpg(process.pid, 0)
                residual_process_group = True
            except ProcessLookupError:
                pass
            except PermissionError:
                residual_process_group = True
                _record_process_group_permission_denied()
        if residual_process_group:
            log.write(
                "cfexp detected that the process group still exists after the experiment "
                "leader exited; terminating remaining members before artifact inspection\n"
            )
            _append_event(
                event_path, "residual_process_group_detected",
                process_group_snapshot=_snapshot_process_group(process.pid),
            )
            _signal_process_group(signal.SIGTERM)
            if not process_group_permission_denied:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        break
                    except PermissionError:
                        _record_process_group_permission_denied()
                        break
                    time.sleep(0.1)
                else:
                    _signal_process_group(signal.SIGKILL)
                    if not process_group_permission_denied:
                        deadline = time.monotonic() + 5
                        while time.monotonic() < deadline:
                            try:
                                os.killpg(process.pid, 0)
                            except ProcessLookupError:
                                break
                            except PermissionError:
                                _record_process_group_permission_denied()
                                break
                            time.sleep(0.1)
            if return_code == 0:
                return_code = 78
        process_group_still_present = process_group_permission_denied
        if not process_group_still_present and hasattr(os, "killpg"):
            try:
                os.killpg(process.pid, 0)
                process_group_still_present = True
            except ProcessLookupError:
                pass
            except PermissionError:
                process_group_still_present = True
                _record_process_group_permission_denied()
        if process_group_still_present:
            log.write(
                "cfexp could not prove that the experiment process group exited; "
                "refusing to remove its runtime environment\n"
            )
            _append_event(event_path, "process_group_cleanup_incomplete")
            _append_event(
                event_path,
                "process_finished",
                status="FAILED",
                exit_code=78,
            )
            raise _UnsafeRuntimeCleanup(
                "experiment process group remains or could not be inspected after "
                "bounded TERM/KILL; runtime cleanup was refused"
            )
    status = "INTERRUPTED" if interrupted else ("SUCCEEDED" if return_code == 0 else "FAILED")
    _append_event(event_path, "process_finished", status=status, exit_code=return_code)
    return return_code, status


def _remove_runtime_environment(
    runtime_environment_root: Path,
    expected_runtime_environment_root: Path,
) -> None:
    """Remove one exact runtime root, retrying only transient filesystem races."""

    if runtime_environment_root != expected_runtime_environment_root:
        raise ExperimentError("refusing an unexpected uv runtime environment path")
    retryable_errors = {errno.ENOTEMPTY}
    stale_errno = getattr(errno, "ESTALE", None)
    if stale_errno is not None:
        retryable_errors.add(stale_errno)
    last_error: OSError | None = None
    for attempt in range(_RUNTIME_CLEANUP_ATTEMPTS):
        try:
            root_stat = os.lstat(runtime_environment_root)
        except FileNotFoundError:
            return
        except OSError as error:
            if error.errno not in retryable_errors:
                raise ExperimentError(
                    f"could not inspect uv runtime environment before cleanup: {error}"
                ) from error
            last_error = error
        else:
            if stat.S_ISLNK(root_stat.st_mode):
                raise ExperimentError(
                    "uv runtime environment became a symlink; refusing cleanup"
                )
            if not stat.S_ISDIR(root_stat.st_mode):
                raise ExperimentError(
                    "uv runtime environment is not a directory; refusing cleanup"
                )
            try:
                shutil.rmtree(runtime_environment_root)
            except FileNotFoundError:
                pass
            except OSError as error:
                if error.errno not in retryable_errors:
                    raise ExperimentError(
                        f"could not remove uv runtime environment: {error}"
                    ) from error
                last_error = error

        try:
            remaining_stat = os.lstat(runtime_environment_root)
        except FileNotFoundError:
            return
        except OSError as error:
            if error.errno not in retryable_errors:
                raise ExperimentError(
                    f"could not verify uv runtime environment cleanup: {error}"
                ) from error
            last_error = error
        else:
            if stat.S_ISLNK(remaining_stat.st_mode):
                raise ExperimentError(
                    "uv runtime environment became a symlink; refusing cleanup"
                )
            if not stat.S_ISDIR(remaining_stat.st_mode):
                raise ExperimentError(
                    "uv runtime environment is not a directory; refusing cleanup"
                )
            if last_error is None:
                last_error = OSError(
                    errno.ENOTEMPTY,
                    "runtime environment still exists after cleanup",
                    str(runtime_environment_root),
                )
        if attempt + 1 < _RUNTIME_CLEANUP_ATTEMPTS:
            time.sleep(_RUNTIME_CLEANUP_BACKOFF_SECONDS * (attempt + 1))

    raise ExperimentError(
        "uv runtime environment cleanup remained incomplete after bounded retries"
    ) from last_error


def _persist_signal_acceptance(
    attempt_dir: Path,
    acceptance: SignalAcceptance,
) -> Path:
    """Create the published provenance copy without re-reading mutable input."""

    destination = attempt_dir / acceptance.provenance_path
    destination.parent.mkdir(parents=True, exist_ok=False)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(destination, flags, 0o444)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(acceptance.raw_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ExperimentError(f"could not preserve Insight 2 gate receipt: {error}") from error
    if sha256_file(destination) != acceptance.sha256:
        raise ExperimentError("preserved Insight 2 gate receipt digest drifted")
    return destination


def run_experiment(
    manifest_path: Path,
    host: HostConfig,
    gpus: str | None,
    *,
    allow_dirty: bool = False,
    local_only: bool = False,
    insight2a_acceptance: Path | None = None,
    insight2_gate_receipt: Path | None = None,
) -> dict[str, object]:
    if not local_only and FORMAL_HF_ARTIFACT_REMOTE is None:
        raise ExperimentError("This distribution supports local-only runner checks.")
    plan = build_plan(manifest_path, host, gpus, allow_dirty=allow_dirty)
    acceptance_required = _requires_signal_acceptance(plan.manifest)
    if insight2a_acceptance is not None and insight2_gate_receipt is not None:
        raise ExperimentError("specify only one Insight 2 gate receipt")
    gate_receipt = insight2_gate_receipt or insight2a_acceptance
    if acceptance_required and gate_receipt is None:
        raise ExperimentError(
            "Insight 2(b) is blocked until --insight2-gate-receipt names an "
            "immutable policy-bound receipt"
        )
    if not acceptance_required and gate_receipt is not None:
        raise ExperimentError(
            "--insight2-gate-receipt is valid only for a gated Insight 2(b) manifest"
        )
    if plan.artifact_remote.startswith("hf://"):
        configure_hf_environment(host)
    if not local_only and plan.git["dirty"]:
        raise ExperimentError(
            "formal runs require a clean Git working tree; --allow-dirty is limited to "
            "--local-only smoke tests"
        )
    if not local_only and plan.git.get("remote_has_embedded_credentials"):
        raise ExperimentError(
            "formal runs reject credential-bearing origin URLs; use a Git credential "
            "helper, SSH agent, or deploy key instead"
        )
    if not local_only and plan.git.get("remote_url_rewrite_configured"):
        raise ExperimentError(
            "formal runs reject Git url.*.insteadOf/pushInsteadOf configuration"
        )
    if not local_only:
        expected_repository = host.github_repository.casefold()
        actual_repository = _github_repository_slug(plan.git.get("remote"))
        if actual_repository != expected_repository:
            raise ExperimentError(
                "formal runs require origin to be the host-configured GitHub repository"
            )
    if not local_only and not _has_durable_origin(plan.git.get("remote")):
        raise ExperimentError(
            "formal runs require remote.origin.url to use a non-local Git transport; "
            "push this revision to GitHub or another durable Git server first"
        )
    if not local_only and host.artifact_remote != FORMAL_HF_ARTIFACT_REMOTE:
        raise ExperimentError(
            "formal runs require the pinned private Hugging Face dataset "
            f"{FORMAL_HF_ARTIFACT_REMOTE}"
        )
    if not local_only and plan.artifact_remote != host.artifact_remote.strip():
        raise ExperimentError(
            "manifest artifact remote must resolve to the exact host-configured "
            "Hugging Face dataset"
        )
    if not local_only:
        _ensure_formal_path_isolation(plan, host)
    shared_hf_credentials = [
        host.cache_root / "huggingface" / name for name in ("token", "stored_tokens")
    ]
    if any(path.exists() or path.is_symlink() for path in shared_hf_credentials):
        raise ExperimentError(
            "shared experiment HF cache contains token state; move publisher credentials "
            "to the mode-600 secrets_file before running"
        )

    ensure_host_directories(host)
    acceptance: SignalAcceptance | None = None
    if acceptance_required:
        assert gate_receipt is not None
        try:
            acceptance = load_signal_acceptance(
                gate_receipt,
                host=host,
                target_phase=str(plan.manifest["spec"]["parameters"]["phase"]),  # type: ignore[index]
                target_spec_hash=plan.spec_hash,
                target_code_revision=str(plan.git["sha"]),
                target_acceptance_policy=signal_acceptance_policy(plan.manifest),
            )
        except SignalAcceptanceError as error:
            raise ExperimentError(f"Insight 2 gate receipt failed: {error}") from error
    run_id = _run_id(plan)
    attempt_id = f"attempt-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex}"
    run_dir = host.run_root / run_id
    attempt_dir = run_dir / "attempts" / attempt_id
    event_path = attempt_dir / "events.jsonl"

    # Resolve secrets and all failure-prone paths before creating an attempt so
    # a missing credential cannot leave an empty, apparently-started run.
    environment, secret_names = _command_environment(plan, host, attempt_dir, attempt_id)
    hf_token = (
        _publication_hf_token(host)
        if plan.artifact_remote.startswith("hf://")
        else None
    )
    command = _experiment_command(
        plan.manifest, plan.repo_root, host, plan.gpu_ids, run_dir=attempt_dir
    )
    spec = plan.manifest["spec"]
    assert isinstance(spec, dict)
    command_spec = spec["command"]
    assert isinstance(command_spec, dict)
    command_cwd = (plan.repo_root / str(command_spec.get("cwd", "."))).resolve()
    try:
        command_cwd.relative_to(plan.repo_root)
    except ValueError as error:
        raise ExperimentError("command.cwd must stay inside the Git repository") from error
    if not command_cwd.is_dir():
        raise ExperimentError(f"command.cwd is not a directory: {command_cwd}")

    storage_spec = spec["storage"]
    outputs_spec = spec["outputs"]
    assert isinstance(storage_spec, dict) and isinstance(outputs_spec, dict)
    include_patterns = [str(item) for item in outputs_spec.get("include", [])]
    exclude_patterns = [str(item) for item in outputs_spec.get("exclude", [])]
    destination = None
    if not local_only:
        destination = _destination(
            plan.artifact_remote,
            str(storage_spec["project"]),
            run_id,
        )
        preflight_destination(destination, hf_token=hf_token)

    if not local_only:
        _verify_remote_commit(
            plan.repo_root,
            str(plan.git["sha"]),
            str(plan.git["remote"]),
        )
        _verify_github_repository_private(host.github_repository)

    git_before = collect_git_metadata(plan.repo_root)
    if _git_execution_state(git_before) != _git_execution_state(plan.git):
        raise ExperimentError(
            "Git state changed after planning and before execution; build a new plan and retry"
        )
    if not local_only and git_before["dirty"]:
        raise ExperimentError("Git working tree became dirty before formal execution")
    if not local_only and git_before.get("remote_has_embedded_credentials"):
        raise ExperimentError("Git origin gained an embedded credential before execution")
    if not local_only and git_before.get("remote_url_rewrite_configured"):
        raise ExperimentError("Git URL rewrite configuration appeared before execution")
    repository_inputs_before = _verify_repository_inputs(
        spec.get("inputs", []), plan.repo_root
    )
    if repository_inputs_before != plan.repository_inputs:
        raise ExperimentError(
            "repository inputs changed after planning and before execution"
        )
    if not local_only:
        repository_videos_before = _repository_video_violations(plan.repo_root)
        if repository_videos_before:
            preview = ", ".join(repr(path) for path in repository_videos_before[:5])
            raise ExperimentError(
                "formal runs require video outputs to be written below CFEXP_RUN_DIR; "
                f"the Git repository already contains video paths: {preview}"
            )

    attempt_dir.mkdir(parents=True, exist_ok=False)
    (attempt_dir / "tmp").mkdir()
    acceptance_audit: dict[str, object] | None = None
    if acceptance is not None:
        _persist_signal_acceptance(attempt_dir, acceptance)
        acceptance_audit = acceptance.audit_record()

    resolved = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "spec_hash": plan.spec_hash,
        "resolved_at": utc_now(),
        "manifest": plan.manifest,
        "git": plan.git,
        "git_before_execution": git_before,
        "lock_sha256": plan.lock_sha256,
        "repository_inputs": plan.repository_inputs,
        "selected_gpu_ids": plan.gpu_ids,
        "argv": command,
        "host_paths": host.serializable(),
        # Persist the exact immutable target selected before execution. A run
        # can cross a UTC month boundary, so reconciliation must not rebuild
        # this path from its completion timestamp.
        "artifact_destination": destination,
        "publication_requested": not local_only,
        "secrets": secret_presence(secret_names, environment),
        "publication_secrets": {"HF_TOKEN": bool(hf_token)},
    }
    if (
        acceptance is not None
        and acceptance.acceptance_policy == EXTERNAL_PRIOR_2B_EVAL_ACCEPTANCE_POLICY
    ):
        resolved["insight2b_human_authorization"] = acceptance_audit
    else:
        resolved["insight2a_signal_acceptance"] = acceptance_audit
    write_json(run_dir / "resolved-manifest.json", resolved)
    acceptance_metadata: dict[str, object] | None = None
    if acceptance is not None:
        acceptance_metadata = {
            "receipt_sha256": acceptance.sha256,
            "accepted_by": acceptance.payload["accepted_by"],
            "acceptance_policy": acceptance.acceptance_policy,
            "copied_to": acceptance.provenance_path,
        }
        source = acceptance.payload.get("source_2a")
        if isinstance(source, dict):
            acceptance_metadata["source_run_id"] = source["run_id"]
        else:
            acceptance_metadata["limitations"] = acceptance.payload["limitations"]
    metadata = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "started_at": utc_now(),
        "machine": collect_machine_metadata(),
        "git": plan.git,
        "git_before_execution": git_before,
        "lock_sha256": plan.lock_sha256,
        "repository_inputs": plan.repository_inputs,
        "spec_hash": plan.spec_hash,
        "argv": command,
        "secret_presence": secret_presence(secret_names, environment),
    }
    if (
        acceptance is not None
        and acceptance.acceptance_policy == EXTERNAL_PRIOR_2B_EVAL_ACCEPTANCE_POLICY
    ):
        metadata["insight2b_human_authorization"] = acceptance_metadata
    else:
        metadata["insight2a_signal_acceptance"] = acceptance_metadata
    write_json(attempt_dir / "metadata.json", metadata)
    _append_event(event_path, "resolved")
    if acceptance is not None:
        event_fields: dict[str, object] = {
            "receipt_sha256": acceptance.sha256,
            "accepted_by": acceptance.payload["accepted_by"],
            "acceptance_policy": acceptance.acceptance_policy,
            "copied_to": acceptance.provenance_path,
        }
        source = acceptance.payload.get("source_2a")
        if isinstance(source, dict):
            event_fields["source_run_id"] = source["run_id"]
            event_name = "insight2a_signal_acceptance_verified"
        else:
            event_fields["limitations"] = acceptance.payload["limitations"]
            event_name = "insight2b_human_authorization_verified"
        _append_event(event_path, event_name, **event_fields)

    runtime_environment_root = Path(environment["UV_PROJECT_ENVIRONMENT"]).parent
    expected_runtime_environment_root = host.tmp_root / f"cfexp-runtime-{attempt_id}"
    if runtime_environment_root != expected_runtime_environment_root:
        raise ExperimentError("refusing an unexpected uv runtime environment path")
    runtime_environment_root.mkdir(mode=0o700, exist_ok=False)
    runtime_cleanup_allowed = False
    try:
        exit_code, status = _execute(
            command,
            cwd=command_cwd,
            environment=environment,
            log_path=attempt_dir / "logs" / "stdout.log",
            event_path=event_path,
            redactions=[
                environment[name] for name in secret_names if environment.get(name)
            ],
        )
        runtime_cleanup_allowed = True
    finally:
        if runtime_cleanup_allowed:
            _remove_runtime_environment(
                runtime_environment_root,
                expected_runtime_environment_root,
            )
    try:
        git_after: dict[str, object] | None = collect_git_metadata(plan.repo_root)
        git_after_error = None
    except RuntimeError as error:
        git_after = None
        git_after_error = str(error)
    git_changed = (
        git_after is None
        or _git_execution_state(git_after) != _git_execution_state(git_before)
    )
    if git_changed:
        _append_event(
            event_path,
            "git_state_changed",
            before=git_before,
            after=git_after,
            error=git_after_error,
        )
        status = "FAILED"
        if exit_code == 0:
            exit_code = 78
    repository_output_violations: list[str] = []
    repository_output_scan_error: str | None = None
    if not local_only:
        try:
            repository_output_violations = _repository_video_violations(plan.repo_root)
        except ExperimentError as error:
            repository_output_scan_error = str(error)
        if repository_output_violations or repository_output_scan_error:
            _append_event(
                event_path,
                "repository_output_violation",
                paths=repository_output_violations,
                scan_error=repository_output_scan_error,
            )
            status = "FAILED"
            if exit_code == 0:
                exit_code = 78
    metadata.update(
        {
            "finished_at": utc_now(),
            "status": status,
            "exit_code": exit_code,
            "git_after_execution": git_after,
            "git_after_error": git_after_error,
            "git_state_changed": git_changed,
            "repository_output_violations": repository_output_violations,
            "repository_output_scan_error": repository_output_scan_error,
        }
    )
    write_json(attempt_dir / "metadata.json", metadata)
    write_json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "latest_attempt_id": attempt_id,
            "execution_status": status,
            "exit_code": exit_code,
            "updated_at": utc_now(),
        },
    )
    marker_name = "_SUCCESS.json" if status == "SUCCEEDED" else "_FAILED.json"
    seal_run_artifacts(
        run_dir,
        marker_name,
        {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "status": status,
            "exit_code": exit_code,
            "spec_hash": plan.spec_hash,
            "git_sha": plan.git["sha"],
            "created_at": utc_now(),
        },
        destination=destination,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    publication = None
    if not local_only:
        assert destination is not None
        try:
            publication = publish_run(
                run_dir,
                destination,
                marker_name,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                hf_token=hf_token,
            )
        except StorageError as error:
            raise ExperimentError(
                f"experiment finished with status {status}, but publication failed: {error}; "
                f"local run is preserved at {run_dir}"
            ) from error

    result = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "status": status,
        "exit_code": exit_code,
        "run_dir": str(run_dir),
        "publication": publication,
    }
    return result
