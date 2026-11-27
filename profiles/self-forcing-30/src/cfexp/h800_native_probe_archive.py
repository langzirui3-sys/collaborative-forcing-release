"""Open the pinned H800 Self-Forcing archive for native parity probes."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import tarfile


ARCHIVE_RELATIVE = Path(
    "source/Self-Forcing-33593df3e81fa3ec10239271dd2c100facac6de1.github-codeload.tar.gz"
)
ARCHIVE_SHA = "9c8ea88fccca61b6421035add7605cb3a96a939b384dad7d9a9e5ff32beeae26"


def load_function(model_root, *, module_path, module_sha, name, function_sha, scope):
    archive_path = (model_root / ARCHIVE_RELATIVE).resolve(strict=True)
    if not archive_path.is_relative_to(model_root):
        raise ValueError("H800 native probe archive must be inside the model root")
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError("H800 native probe archive differs")
    with tarfile.open(archive_path) as archive:
        matches = [member for member in archive.getmembers()
                   if member.name.endswith("/" + module_path)]
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError("native probe module missing or ambiguous")
        source = archive.extractfile(matches[0]).read()
    if hashlib.sha256(source).hexdigest() != module_sha:
        raise ValueError("native probe module differs")
    text = source.decode()
    functions = [node for node in ast.parse(text).body
                 if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1 or functions[0].decorator_list:
        raise ValueError("native probe function missing, ambiguous or decorated")
    function = ast.get_source_segment(text, functions[0])
    if hashlib.sha256(function.encode()).hexdigest() != function_sha:
        raise ValueError("native probe function differs")
    exec(compile(function, "<pinned-h800-native-probe>", "exec"), scope)
    return scope[name]
