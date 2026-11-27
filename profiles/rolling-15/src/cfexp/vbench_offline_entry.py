"""Fail-closed offline entry point for the frozen official VBench-Long code.

The upstream evaluator contains automatic download paths and two long-video
feature extractors which call ``torch.hub`` with GitHub repository names.  A
formal run must not exercise either behavior.  This entry point verifies the
complete asset bundle, blocks network-capable Python APIs and all unbound
processes, and loads the two DINO models from local source and weight files.
"""

from __future__ import annotations

import _socket
import argparse
import hashlib
import os
from pathlib import Path
import runpy
import socket
import stat
import subprocess
import sys
from typing import Any, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfexp.vbench_long import (  # noqa: E402
    OFFLINE_NETWORK_POLICY,
    VBENCH_DISTRIBUTED_ENVIRONMENT,
    VBENCH_LONG_CREDENTIAL_ENVIRONMENT_VARIABLES,
    VBenchLongError,
    validate_vbench_long_checkout,
    _sha256_file,
    validate_vbench_long_offline_assets,
)


class OfflineNetworkError(RuntimeError):
    """Raised if metric code attempts to leave the immutable local bundle."""


def _blocked_network(*_args: Any, **_kwargs: Any) -> Any:
    raise OfflineNetworkError(
        f"outbound network access is forbidden by {OFFLINE_NETWORK_POLICY}"
    )


def _blocked_process(*_args: Any, **_kwargs: Any) -> Any:
    raise OfflineNetworkError(
        f"process creation is forbidden by {OFFLINE_NETWORK_POLICY}"
    )


def _bound_ffmpeg_executable() -> Path:
    """Bind the bundled imageio binary without running its PATH probes."""
    import imageio_ffmpeg
    from imageio_ffmpeg._definitions import FNAME_PER_PLATFORM, get_platform

    name = FNAME_PER_PLATFORM.get(get_platform())
    if not name:
        raise OfflineNetworkError("no bundled ffmpeg for this platform")
    package = Path(imageio_ffmpeg.__file__).absolute().parent
    candidate = package / "binaries" / name
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not candidate.is_file()
        or not os.access(candidate, os.X_OK)
        or not candidate.resolve(strict=True).is_relative_to(
            Path(sys.prefix).resolve(strict=True)
        )
    ):
        raise OfflineNetworkError("bundled ffmpeg executable is not bound")
    return candidate.resolve(strict=True)


def _devnull_stream(value: Any) -> bool:
    try:
        actual = os.fstat(value.fileno())
        expected = os.stat(os.devnull)
    except (AttributeError, OSError, ValueError):
        return False
    return (
        stat.S_ISCHR(actual.st_mode)
        and (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)
    )


def _install_network_guard(ffmpeg_executable: Path | None = None) -> None:
    """Block network access and all processes except bound ffmpeg probes."""

    # ``ssl`` captures ``socket.socket`` as a base class at import time.  The
    # launcher deliberately installs this guard before importing any TLS/HTTP
    # stack so no pre-guard alias to the original Internet-capable class can
    # survive.  A polluted launcher process is rejected instead of attempting
    # to patch an already-created class hierarchy incompletely.
    if "ssl" in sys.modules:
        raise OfflineNetworkError("ssl was imported before the offline guard")

    original_socket = socket.socket
    raw_socket_init = _socket.socket.__init__

    class OfflineSocket(original_socket):  # type: ignore[misc,valid-type]
        def __init__(
            self,
            family: int = socket.AF_INET,
            type: int = socket.SOCK_STREAM,
            proto: int = 0,
            fileno: int | None = None,
        ) -> None:
            if family in {socket.AF_INET, socket.AF_INET6}:
                _blocked_network()
            raw_socket_init(self, family, type, proto, fileno)
            # Mirror ``socket.socket.__init__`` without its dynamic reference
            # to ``_socket.socket`` (which is also replaced below).
            self._io_refs = 0
            self._closed = False

        def _require_local(self) -> None:
            if self.family in {socket.AF_INET, socket.AF_INET6}:
                _blocked_network()

        def connect(self, *args: Any, **kwargs: Any) -> Any:
            self._require_local()
            return super().connect(*args, **kwargs)

        def connect_ex(self, *args: Any, **kwargs: Any) -> Any:
            self._require_local()
            return super().connect_ex(*args, **kwargs)

        def send(self, *args: Any, **kwargs: Any) -> Any:
            self._require_local()
            return super().send(*args, **kwargs)

        def sendall(self, *args: Any, **kwargs: Any) -> Any:
            self._require_local()
            return super().sendall(*args, **kwargs)

        def sendto(self, *args: Any, **kwargs: Any) -> Any:
            self._require_local()
            return super().sendto(*args, **kwargs)

        def sendmsg(self, *args: Any, **kwargs: Any) -> Any:
            self._require_local()
            return super().sendmsg(*args, **kwargs)

    socket.socket = OfflineSocket  # type: ignore[assignment,misc]
    socket.SocketType = OfflineSocket  # type: ignore[assignment,misc]
    _socket.socket = OfflineSocket  # type: ignore[assignment,misc]
    socket.create_connection = _blocked_network  # type: ignore[assignment]
    socket.getaddrinfo = _blocked_network  # type: ignore[assignment]
    socket.gethostbyname = _blocked_network  # type: ignore[assignment]
    socket.gethostbyname_ex = _blocked_network  # type: ignore[assignment]
    socket.gethostbyaddr = _blocked_network  # type: ignore[assignment]
    socket.getnameinfo = _blocked_network  # type: ignore[assignment]

    # Import the HTTP/TLS stack only after replacing both public and C-extension
    # socket constructors.  This makes any socket class aliases it captures
    # inherit the same AF_INET/AF_INET6 rejection.
    import urllib.request as urllib_request

    urllib_request.urlopen = _blocked_network  # type: ignore[assignment]
    urllib_request.urlretrieve = _blocked_network  # type: ignore[assignment]
    if ffmpeg_executable is not None:
        ffmpeg = Path(ffmpeg_executable)
        if not ffmpeg.is_absolute() or ffmpeg.is_symlink() or not ffmpeg.is_file():
            raise OfflineNetworkError("ffmpeg executable is not bound")
        ffmpeg = ffmpeg.resolve(strict=True)
        ffmpeg_digest = hashlib.sha256(ffmpeg.read_bytes()).digest()
        original_popen = subprocess.Popen

        def guarded_popen(*args: Any, **kwargs: Any) -> Any:
            positional_command = len(args) == 1 and "args" not in kwargs
            keyword_command = not args and "args" in kwargs
            command = args[0] if positional_command else None
            if keyword_command:
                command = kwargs["args"]
            # PySceneDetect catches OSError and then tries imageio's absolute
            # path.  Do not let its PATH lookup bind an ambient executable.
            if command == ["ffmpeg", "-v", "quiet"]:
                raise FileNotFoundError("unbound ffmpeg on PATH")
            tail = tuple(command[1:]) if isinstance(command, (list, tuple)) else ()
            quiet_probe = positional_command and tail == ("-v", "quiet") and not kwargs
            version_probe = (
                positional_command
                and tail == ("-version",)
                and set(kwargs) == {
                    "stdout", "stderr", "startupinfo", "creationflags", "preexec_fn"
                }
                and _devnull_stream(kwargs["stdout"])
                and kwargs["stderr"] == subprocess.STDOUT
                and kwargs["startupinfo"] is None
                and kwargs["creationflags"] == 0
                and kwargs["preexec_fn"] is None
            )
            version_output = (
                keyword_command
                and tail == ("-version",)
                and set(kwargs) == {"args", "stdout", "text"}
                and kwargs["stdout"] == subprocess.PIPE
                and kwargs["text"] is True
            )
            if (
                not isinstance(command, (list, tuple))
                or not all(isinstance(part, str) for part in command)
                or not command
                or command[0] != str(ffmpeg)
                or not (quiet_probe or version_probe or version_output)
                or not ffmpeg.is_file()
                or hashlib.sha256(ffmpeg.read_bytes()).digest() != ffmpeg_digest
            ):
                _blocked_process()
            return original_popen(*args, **kwargs)

        subprocess.Popen = guarded_popen  # type: ignore[assignment,misc]
    else:
        subprocess.Popen = _blocked_process  # type: ignore[assignment,misc]
    for name in (
        "getoutput",
        "getstatusoutput",
    ):
        if hasattr(subprocess, name):
            setattr(subprocess, name, _blocked_process)
    # run/call/check_call/check_output all delegate to the guarded Popen.
    for name in (
        "system",
        "popen",
        "fork",
        "forkpty",
        "posix_spawn",
        "posix_spawnp",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
    ):
        if hasattr(os, name):
            setattr(os, name, _blocked_process)


def _unwrap_state_dict(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("state_dict", "model"):
            nested = value.get(key)
            if isinstance(nested, dict) and nested:
                return nested
    return value


def _install_torch_hub_shim(assets: Any) -> None:
    import torch

    original_load = torch.hub.load
    dino_v1_source = assets.tree_path("dino_v1_source")
    dinov2_source = assets.tree_path("dinov2_source")
    dino_v1_weights = assets.file_path("dino_v1_weights")
    dinov2_weights = assets.file_path("dinov2_weights")

    def build_local(source: Path, model_name: str, weights: Path) -> Any:
        model = original_load(
            str(source),
            model_name,
            source="local",
            pretrained=False,
        )
        state_dict = _unwrap_state_dict(
            torch.load(str(weights), map_location="cpu", weights_only=True)
        )
        if not isinstance(state_dict, dict) or not state_dict:
            raise VBenchLongError(f"bound {model_name} checkpoint is not a state dict")
        model.load_state_dict(state_dict, strict=True)
        return model

    def offline_load(repo_or_dir: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
        repo = os.fspath(repo_or_dir)
        model_name = str(model)
        if args:
            raise VBenchLongError("unexpected positional torch.hub model arguments")
        if (
            model_name == "dino_vitb16"
            and Path(repo).resolve() == dino_v1_source.resolve()
        ):
            expected = {
                "path": str(dino_v1_weights),
                "source": "local",
                "read_frame": None,
            }
            if kwargs != expected:
                raise VBenchLongError(
                    "local VBench DINO v1 torch.hub signature drifted"
                )
            return build_local(dino_v1_source, model_name, dino_v1_weights)
        if model_name == "dino_vitb16" and repo == "facebookresearch/dino:main":
            if kwargs:
                raise VBenchLongError(
                    "DreamSim DINO v1 torch.hub signature drifted"
                )
            return build_local(dino_v1_source, model_name, dino_v1_weights)
        if model_name == "dinov2_vitb14" and repo == "facebookresearch/dinov2":
            if kwargs:
                raise VBenchLongError("DINOv2 torch.hub signature drifted")
            return build_local(dinov2_source, model_name, dinov2_weights)
        raise OfflineNetworkError(
            f"unbound torch.hub model request is forbidden: {repo!r} {model_name!r}"
        )

    torch.hub.load = offline_load
    torch.hub.download_url_to_file = _blocked_network
    torch.hub.load_state_dict_from_url = _blocked_network


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--benchmark", choices=("vbench", "vbench-long"), default="vbench-long")
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--asset-spec", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--eval-script", required=True)
    parser.add_argument("--eval-sha256", required=True)
    parser.add_argument("--eval-nbytes", required=True, type=int)
    parser.add_argument("--full-info", required=True)
    parser.add_argument("--full-info-sha256", required=True)
    parser.add_argument("--full-info-nbytes", required=True, type=int)
    parser.add_argument("--runtime-cache-root", required=True)
    parsed, remaining = parser.parse_known_args(argv)
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    if not remaining:
        parser.error("official VBench-Long arguments are missing")
    return parsed, remaining


def main(argv: Sequence[str] | None = None) -> int:
    parsed, upstream_args = _parse_args(argv)
    assets = validate_vbench_long_offline_assets(
        parsed.asset_manifest,
        spec_path=parsed.asset_spec,
    )
    checkout = validate_vbench_long_checkout(parsed.checkout)
    if parsed.benchmark == "vbench":
        from dataclasses import replace
        short_eval = checkout.root / "evaluate.py"
        short_info = checkout.root / "vbench" / "VBench_full_info.json"
        if short_eval.is_symlink() or short_info.is_symlink():
            raise VBenchLongError("short VBench entry files must not be symlinks")
        eval_sha, eval_size = _sha256_file(short_eval)
        info_sha, info_size = _sha256_file(short_info)
        checkout = replace(checkout, eval_script=short_eval, full_info=short_info,
                           eval_sha256=eval_sha, eval_nbytes=eval_size,
                           full_info_sha256=info_sha, full_info_nbytes=info_size)
        positions = [i for i, value in enumerate(upstream_args) if value == "--mode"]
        if len(positions) != 1 or upstream_args[positions[0] + 1:positions[0] + 2] != ["custom_input"]:
            raise VBenchLongError("short VBench requires exactly custom_input mode")
    eval_script = Path(parsed.eval_script).expanduser().resolve(strict=True)
    full_info = Path(parsed.full_info).expanduser().resolve(strict=True)
    if (
        eval_script != checkout.eval_script
        or full_info != checkout.full_info
        or parsed.eval_sha256 != checkout.eval_sha256
        or parsed.eval_nbytes != checkout.eval_nbytes
        or parsed.full_info_sha256 != checkout.full_info_sha256
        or parsed.full_info_nbytes != checkout.full_info_nbytes
    ):
        raise VBenchLongError("official VBench-Long checkout file binding drifted")

    full_info_positions = [
        index for index, value in enumerate(upstream_args) if value == "--full_json_dir"
    ]
    if (
        full_info_positions != [len(upstream_args) - 2]
        or upstream_args[-1] != str(full_info)
    ):
        raise VBenchLongError("official full-info argument is not exactly bound")

    runtime_cache = Path(parsed.runtime_cache_root).expanduser().resolve(strict=True)
    expected_caches = {
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
        "PYTHONPYCACHEPREFIX": runtime_cache / "python-bytecode",
    }
    for variable, expected in expected_caches.items():
        value = os.environ.get(variable)
        cache = Path(value) if value else None
        if (
            cache is None
            or not cache.is_absolute()
            or cache.resolve(strict=True) != expected
            or not cache.is_dir()
            or cache.is_symlink()
        ):
            raise VBenchLongError(
                f"{variable} is not the bound run-local cache directory"
            )
    for variable in VBENCH_LONG_CREDENTIAL_ENVIRONMENT_VARIABLES:
        if variable in os.environ:
            raise VBenchLongError(
                f"ambient credential variable was not removed: {variable}"
            )

    os.environ.update(
        {
            **VBENCH_DISTRIBUTED_ENVIRONMENT,
            "HOME": str(assets.root / "home"),
            "VBENCH_CACHE_DIR": str(assets.root / "vbench-cache"),
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
    )
    _install_network_guard(_bound_ffmpeg_executable())
    _install_torch_hub_shim(assets)
    if any(
        name == "vbench"
        or name.startswith("vbench.")
        or name == "vbench2_beta_long"
        or name.startswith("vbench2_beta_long.")
        for name in sys.modules
    ):
        raise VBenchLongError(
            "official VBench modules were imported before checkout pinning"
        )
    checkout_text = str(checkout.root)
    sys.path[:] = [item for item in sys.path if item != checkout_text]
    sys.path.insert(0, checkout_text)
    sys.argv = [str(eval_script), *upstream_args]
    runpy.run_path(str(eval_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
