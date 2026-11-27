from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(argv: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return 127, str(error)
    return result.returncode, result.stdout.strip()


def _git(cwd: Path, *args: str) -> str:
    code, output = _command(["git", *args], cwd=cwd)
    if code != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {output}")
    return output


def find_git_root(start: Path) -> Path:
    return Path(_git(start, "rev-parse", "--show-toplevel"))


def _redact_remote(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<invalid remote URL>"
    if parsed.scheme and parsed.netloc:
        # Strip both userinfo and URL parameters. Query strings on HTTPS Git
        # remotes are occasionally used for access tokens and must never enter
        # run metadata. Preserve SSH's conventional, non-secret `git` account
        # because repository normalization requires it and the remote runner
        # explicitly accepts that form.
        host = parsed.netloc.rsplit("@", 1)[-1]
        if (
            parsed.scheme.lower() == "ssh"
            and parsed.username == "git"
            and parsed.password is None
        ):
            host = f"git@{host}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    # SCP-like Git remotes can put arbitrary text before `@`. Preserve only
    # GitHub's conventional non-secret `git` account name; redact every other
    # username before metadata is serialized.
    value = value.split("?", 1)[0].split("#", 1)[0]
    host, separator, path = value.partition(":")
    if separator and "@" in host:
        username, hostname = host.rsplit("@", 1)
        prefix = "git@" if username == "git" else ""
        return f"{prefix}{hostname}:{path}"
    return value


def _remote_has_embedded_credentials(value: str) -> bool:
    """Detect credential-bearing Git URL channels without retaining the secret."""

    value = value.strip()
    if not value:
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        # An invalid URL is not safe enough for a formal run.  Record only the
        # boolean finding so malformed credential text never reaches metadata.
        return True
    if parsed.query or parsed.fragment:
        return True
    if parsed.scheme.lower() in {"http", "https"}:
        return parsed.username is not None or parsed.password is not None
    if parsed.scheme.lower() == "ssh":
        return parsed.password is not None or parsed.username not in {None, "git"}
    if parsed.password is not None:
        return True
    if not parsed.scheme:
        host, separator, _path = value.partition(":")
        if separator and "@" in host:
            username, _hostname = host.rsplit("@", 1)
            return username != "git"
    return False


def collect_git_metadata(repo_root: Path) -> dict[str, object]:
    sha = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    # `remote get-url` applies url.*.insteadOf. Recording the raw config value
    # would let a visually correct github.com URL be silently redirected to a
    # different server during ls-remote/fetch/push.
    code, remote = _command(["git", "remote", "get-url", "origin"], cwd=repo_root)
    rewrite_code, _ = _command(
        [
            "git",
            "config",
            "--name-only",
            "--get-regexp",
            r"^url\..*\.(insteadOf|pushInsteadOf)$",
        ],
        cwd=repo_root,
    )
    raw_remote = remote if code == 0 else ""
    return {
        "sha": sha,
        "branch": branch,
        "dirty": bool(status),
        "status": status.splitlines(),
        "remote": _redact_remote(raw_remote) if code == 0 else None,
        "remote_has_embedded_credentials": (
            _remote_has_embedded_credentials(raw_remote) if code == 0 else False
        ),
        # Fail closed on both a positive match and an inspection error. The
        # matching keys/values are deliberately not persisted because rewrite
        # targets may themselves contain credentials.
        "remote_url_rewrite_configured": rewrite_code != 1,
    }


def collect_gpu_metadata() -> list[dict[str, object]]:
    fields = "index,name,uuid,memory.total,driver_version"
    code, output = _command(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    if code != 0 or not output:
        return []
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            memory_mib: int | str = int(parts[3])
        except ValueError:
            memory_mib = parts[3]
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "uuid": parts[2],
                "memory_total_mib": memory_mib,
                "driver_version": parts[4],
            }
        )
    return rows


def collect_machine_metadata() -> dict[str, object]:
    memory_bytes = None
    if hasattr(os, "sysconf"):
        try:
            memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "gpus": collect_gpu_metadata(),
    }


def secret_presence(names: list[str], environment: dict[str, str]) -> dict[str, bool]:
    return {name: bool(environment.get(name)) for name in names}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")


def safe_component(value: str) -> str:
    cleaned = _SAFE_NAME.sub("-", value).strip("-.")
    if not cleaned:
        raise ValueError("empty path component after sanitization")
    return cleaned
