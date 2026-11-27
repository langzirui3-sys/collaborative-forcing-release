from __future__ import annotations

import os
import stat
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit


class HostConfigError(ValueError):
    """Raised when a host configuration is missing or unsafe."""


_OFFICIAL_HF_ENDPOINT = "https://huggingface.co"
FORMAL_HF_ARTIFACT_REMOTE = None


@dataclass(frozen=True)
class HostConfig:
    code_root: Path
    model_root: Path
    cache_root: Path
    run_root: Path
    tmp_root: Path
    artifact_remote: str
    runtime_preference: str
    secrets_file: Path
    hf_proxy: str = ""
    github_repository: str = "langzirui3-sys/collaborative-forcing-release"

    def serializable(self) -> dict[str, str]:
        values = asdict(self)
        return {key: str(value) for key, value in values.items()}


_DEFAULTS = {
    "code_root": "~/work/collaborative-forcing",
    "model_root": "~/.cache/collaborative-forcing/models",
    "cache_root": "~/.cache/collaborative-forcing/cache",
    "run_root": "~/.cache/collaborative-forcing/runs",
    "tmp_root": "~/.cache/collaborative-forcing/tmp",
    "artifact_remote": "",
    "runtime_preference": "uv",
    "secrets_file": "~/.config/cfexp/secrets.env",
    "hf_proxy": "",
    "github_repository": "langzirui3-sys/collaborative-forcing-release",
}

_ENV_OVERRIDES = {
    "code_root": "CFEXP_CODE_ROOT",
    "model_root": "CFEXP_MODEL_ROOT",
    "cache_root": "CFEXP_CACHE_ROOT",
    "run_root": "CFEXP_RUN_ROOT",
    "tmp_root": "CFEXP_TMP_ROOT",
    "artifact_remote": "CFEXP_ARTIFACT_REMOTE",
    "runtime_preference": "CFEXP_RUNTIME_PREFERENCE",
    "secrets_file": "CFEXP_SECRETS_FILE",
    "hf_proxy": "CFEXP_HF_PROXY",
    "github_repository": "CFEXP_GITHUB_REPOSITORY",
}

_PATH_FIELDS = {
    "code_root",
    "model_root",
    "cache_root",
    "run_root",
    "tmp_root",
    "secrets_file",
}


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def load_host_config(path: str | Path | None = None) -> HostConfig:
    config_path = _expand_path(str(path or "~/.config/cfexp/host.toml"))
    values: dict[str, str] = dict(_DEFAULTS)
    if config_path.exists():
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        unknown = set(raw) - set(_DEFAULTS)
        if unknown:
            raise HostConfigError(f"unknown host config keys: {sorted(unknown)}")
        for key, value in raw.items():
            if not isinstance(value, str):
                raise HostConfigError(f"host config field {key!r} must be a string")
            values[key] = value

    for key, variable in _ENV_OVERRIDES.items():
        if os.environ.get(variable):
            values[key] = os.environ[variable]

    if values["runtime_preference"] not in {"uv", "image"}:
        raise HostConfigError("runtime_preference must be 'uv' or 'image'")
    repository_parts = values["github_repository"].split("/")
    if (
        len(repository_parts) != 2
        or any(
            not part
            or part in {".", ".."}
            or not all(character.isalnum() or character in "._-" for character in part)
            for part in repository_parts
        )
    ):
        raise HostConfigError("github_repository must use the owner/repository form")
    if values["hf_proxy"]:
        try:
            proxy = urlsplit(values["hf_proxy"])
            proxy_port = proxy.port
        except ValueError as error:
            raise HostConfigError(f"hf_proxy is invalid: {error}") from error
        if (
            proxy.scheme not in {"http", "https", "socks5", "socks5h"}
            or not proxy.hostname
            or proxy_port is None
            or proxy.username
            or proxy.password
            or proxy.query
            or proxy.fragment
        ):
            raise HostConfigError(
                "hf_proxy must be an unauthenticated http(s) or socks5(h) URL with a port"
            )

    converted: dict[str, object] = {}
    for key, value in values.items():
        converted[key] = _expand_path(value) if key in _PATH_FIELDS else value
    return HostConfig(**converted)  # type: ignore[arg-type]


def ensure_host_directories(config: HostConfig) -> None:
    for path in (config.model_root, config.cache_root, config.run_root, config.tmp_root):
        path.mkdir(parents=True, exist_ok=True)


def load_secrets(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    if os.name == "posix":
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise HostConfigError(
                f"secrets file {path} must not be readable by group/others (use chmod 600)"
            )

    secrets: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise HostConfigError(f"invalid secrets line {number}: expected NAME=value")
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "A").isalnum() or name[0].isdigit():
            raise HostConfigError(f"invalid environment variable name on secrets line {number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        secrets[name] = value
    return secrets


def resolve_hf_token(config: HostConfig) -> str | None:
    """Resolve a publisher token kept outside the experiment's shared HF cache."""

    configured = load_secrets(config.secrets_file).get("HF_TOKEN")
    if configured:
        return configured
    environment = os.environ.get("HF_TOKEN")
    if environment:
        return environment
    return None


def configure_hf_environment(config: HostConfig) -> None:
    """Pin Hub caches and any proxy to this host's project-isolated paths."""

    hf_home = config.cache_root / "huggingface"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_XET_CACHE"] = str(hf_home / "xet")
    os.environ["HF_ENDPOINT"] = _OFFICIAL_HF_ENDPOINT
    if config.hf_proxy:
        for name in (
            "ALL_PROXY",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "all_proxy",
            "https_proxy",
            "http_proxy",
        ):
            os.environ[name] = config.hf_proxy
