"""Paper-ready figures from the strict 72-row main-table measurement set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping, Sequence
import uuid

from .main_table_analysis import MainTableAnalysisError, RAW_SCHEMA, analyze_rows
from .main_table_v1 import METHOD_IDS, QUALITY_DIMENSIONS


FIGURE_DATA_SCHEMA = "main_table_paper_figure_data_v1"
RESULT_SCHEMA = "main_table_paper_figure_suite_v1"
FIGURE_NAMES = (
    "quality_dimensions_heatmap",
    "quality_speed_pareto",
    "runtime_and_speedup",
    "quality_gap_vs_full_target",
)
METHOD_LABELS = {
    "sdvg": "SDVG",
    "t_stitch": "T-Stitch",
    "light_forcing": "Light Forcing",
    "dummy_forcing": "Dummy Forcing",
    "ours": "Ours",
    "full_target": "Full 14B Target",
}
METHOD_COLORS = {
    "sdvg": "#7C3AED",
    "t_stitch": "#2563EB",
    "light_forcing": "#0D9488",
    "dummy_forcing": "#65A30D",
    "ours": "#DC2626",
    "full_target": "#475569",
}


class MainTablePaperFigureError(RuntimeError):
    """Raised when raw evidence cannot support the frozen figure suite."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MainTablePaperFigureError(f"value is not canonical JSON: {error}") from error


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
        size = handle.tell()
    return digest.hexdigest(), size


def load_raw_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise MainTablePaperFigureError("raw-row input must be a regular file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                MainTablePaperFigureError(f"invalid JSON constant {item!r}")
            ),
        )
    except MainTablePaperFigureError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MainTablePaperFigureError(f"cannot read raw rows: {error}") from error
    if not isinstance(value, dict) or set(value) != {"schema", "rows"}:
        raise MainTablePaperFigureError("raw-row input field schema drifted")
    if value.get("schema") != RAW_SCHEMA or not isinstance(value.get("rows"), list):
        raise MainTablePaperFigureError("raw-row input schema is invalid")
    rows = value["rows"]
    if any(not isinstance(row, Mapping) for row in rows):
        raise MainTablePaperFigureError("every raw row must be an object")
    return rows


def _binding_matches(path: Path, binding: Mapping[str, Any], *, label: str) -> None:
    if set(binding) != {"sha256", "nbytes"}:
        raise MainTablePaperFigureError(f"{label} binding schema drifted")
    sha256, nbytes = _sha256_file(path)
    if binding.get("sha256") != sha256 or binding.get("nbytes") != nbytes:
        raise MainTablePaperFigureError(f"{label} differs from its strict merge binding")


def validate_merge_directory(merge_dir: Path) -> Path:
    """Revalidate the strict merge and every sealed source before rendering."""

    from .main_table_merge import (
        ANALYSIS_FILENAME,
        RAW_FILENAME,
        RESULT_FILENAME,
        RESULT_SCHEMA as MERGE_RESULT_SCHEMA,
        SOURCE_FILENAME,
        SOURCE_SCHEMA,
        validate_method_attempt,
    )

    try:
        merge_dir = merge_dir.expanduser().resolve(strict=True)
    except OSError as error:
        raise MainTablePaperFigureError(f"merge directory is unavailable: {error}") from error
    if not merge_dir.is_dir() or merge_dir.is_symlink():
        raise MainTablePaperFigureError("merge directory must be a regular directory")
    result = _load_object(merge_dir / RESULT_FILENAME, label="merge result")
    expected_result_fields = {
        "schema",
        "status",
        "experiment_git_sha",
        "method_count",
        "prompt_count",
        "seed_count",
        "row_count",
        "method_ids",
        "official_vbench_long_verified",
        "shard_means_used",
        "artifacts",
    }
    if (
        set(result) != expected_result_fields
        or result.get("schema") != MERGE_RESULT_SCHEMA
        or result.get("status") != "succeeded"
        or result.get("method_count") != len(METHOD_IDS)
        or result.get("prompt_count") != 12
        or result.get("seed_count") != 1
        or result.get("row_count") != 72
        or result.get("method_ids") != list(METHOD_IDS)
        or result.get("official_vbench_long_verified") is not True
        or result.get("shard_means_used") is not False
    ):
        raise MainTablePaperFigureError("strict merge result contract drifted")
    artifacts = result.get("artifacts")
    expected_names = {RAW_FILENAME, ANALYSIS_FILENAME, SOURCE_FILENAME}
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_names:
        raise MainTablePaperFigureError("strict merge artifact set drifted")
    for name in expected_names:
        path = merge_dir / name
        if not path.is_file() or path.is_symlink():
            raise MainTablePaperFigureError(f"strict merge artifact is invalid: {name}")
        binding = artifacts[name]
        if not isinstance(binding, Mapping):
            raise MainTablePaperFigureError(f"strict merge binding is invalid: {name}")
        _binding_matches(path, binding, label=name)

    sources = _load_object(merge_dir / SOURCE_FILENAME, label="merge sources")
    if (
        set(sources) != {"schema", "experiment_git_sha", "methods"}
        or sources.get("schema") != SOURCE_SCHEMA
        or sources.get("experiment_git_sha") != result.get("experiment_git_sha")
        or not isinstance(sources.get("methods"), list)
        or len(sources["methods"]) != len(METHOD_IDS)
    ):
        raise MainTablePaperFigureError("strict merge source document drifted")
    for method_id, source in zip(METHOD_IDS, sources["methods"]):
        if not isinstance(source, Mapping) or not isinstance(source.get("attempt_dir"), str):
            raise MainTablePaperFigureError(f"{method_id} source binding is invalid")
        try:
            validated = validate_method_attempt(Path(source["attempt_dir"]), method_id)
        except Exception as error:
            raise MainTablePaperFigureError(
                f"{method_id} sealed source revalidation failed: {error}"
            ) from error
        if dict(validated.source) != dict(source):
            raise MainTablePaperFigureError(f"{method_id} source differs from strict merge")

    raw_path = merge_dir / RAW_FILENAME
    rows = load_raw_rows(raw_path)
    try:
        recomputed = analyze_rows(rows)
    except MainTableAnalysisError as error:
        raise MainTablePaperFigureError(f"strict merge raw rows failed analysis: {error}") from error
    recorded = _load_object(merge_dir / ANALYSIS_FILENAME, label="merge analysis")
    if recorded != recomputed:
        raise MainTablePaperFigureError("strict merge analysis differs from recomputation")
    return raw_path


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise MainTablePaperFigureError(f"{label} is not a regular file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                MainTablePaperFigureError(f"{label} contains invalid constant {item!r}")
            ),
        )
    except MainTablePaperFigureError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MainTablePaperFigureError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise MainTablePaperFigureError(f"{label} must contain one object")
    return value


def _duplicate_checked_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise MainTablePaperFigureError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def build_figure_data(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        analysis = analyze_rows(rows)
    except MainTableAnalysisError as error:
        raise MainTablePaperFigureError(f"strict main-table analysis failed: {error}") from error
    methods = analysis["methods"]
    if not isinstance(methods, list):
        raise MainTablePaperFigureError("strict analysis omitted methods")
    normalized: list[dict[str, Any]] = []
    for expected, method in zip(METHOD_IDS, methods):
        if not isinstance(method, Mapping) or method.get("method_id") != expected:
            raise MainTablePaperFigureError("strict analysis method order drifted")
        dimension_means = method.get("dimension_means")
        if not isinstance(dimension_means, Mapping) or tuple(dimension_means) != QUALITY_DIMENSIONS:
            raise MainTablePaperFigureError("strict analysis dimension order drifted")
        normalized.append(
            {
                "method_id": expected,
                "label": METHOD_LABELS[expected],
                "color": METHOD_COLORS[expected],
                "dimension_means": {
                    dimension: float(dimension_means[dimension])
                    for dimension in QUALITY_DIMENSIONS
                },
                "quality_macro_mean": float(method["quality_macro_mean"]),
                "generation_wall_seconds_arithmetic_mean": float(
                    method["generation_wall_seconds_arithmetic_mean"]
                ),
                "paired_speedup_geometric_mean": float(
                    method["paired_speedup_geometric_mean"]
                ),
                "paired_speedups": [float(value) for value in method["paired_speedups"]],
                "paired_quality_difference_vs_full_target_mean": float(
                    method["paired_quality_difference_vs_full_target_mean"]
                ),
            }
        )
    return {
        "schema": FIGURE_DATA_SCHEMA,
        "row_count": analysis["row_count"],
        "prompt_count": analysis["prompt_count"],
        "method_count": analysis["method_count"],
        "seed_count": analysis["seed_count"],
        "reference_method_id": analysis["reference_method_id"],
        "speedup_aggregation": analysis["speedup_aggregation"],
        "shard_means_used": analysis["shard_means_used"],
        "quality_dimensions": list(QUALITY_DIMENSIONS),
        "methods": normalized,
    }


def _render_matplotlib(data: Mapping[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as error:
        raise MainTablePaperFigureError(
            "paper figures require the pinned Linux CPython 3.11 environment"
        ) from error

    methods = data["methods"]
    dimensions = data["quality_dimensions"]
    labels = [item["label"] for item in methods]
    colors = [item["color"] for item in methods]
    plt.rcParams.update({"font.size": 10, "figure.dpi": 180, "savefig.dpi": 300})

    matrix = np.array(
        [[item["dimension_means"][dimension] for dimension in dimensions] for item in methods]
    )
    fig, ax = plt.subplots(figsize=(9.4, 4.5), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(dimensions)), [item.replace("_", "\n") for item in dimensions])
    ax.set_yticks(range(len(labels)), labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color="white" if matrix[row, column] < matrix.mean() else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Official VBench-Long score")
    _save_pair(fig, output_dir, FIGURE_NAMES[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for item in methods:
        ax.scatter(item["paired_speedup_geometric_mean"], item["quality_macro_mean"], s=85, color=item["color"], label=item["label"], edgecolor="white", linewidth=0.7)
        ax.annotate(item["label"], (item["paired_speedup_geometric_mean"], item["quality_macro_mean"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Paired speedup vs Full 14B Target (geometric mean, ×)")
    ax.set_ylabel("Six-dimension quality macro mean")
    ax.grid(True, alpha=0.25)
    _save_pair(fig, output_dir, FIGURE_NAMES[1])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)
    positions = np.arange(len(methods))
    axes[0].barh(positions, [item["generation_wall_seconds_arithmetic_mean"] for item in methods], color=colors)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Generation wall time (s)")
    axes[1].barh(positions, [item["paired_speedup_geometric_mean"] for item in methods], color=colors)
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].axvline(1.0, color="#111827", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Paired speedup (×)")
    _save_pair(fig, output_dir, FIGURE_NAMES[2])

    fig, ax = plt.subplots(figsize=(7.6, 4.5), constrained_layout=True)
    gaps = [item["paired_quality_difference_vs_full_target_mean"] for item in methods]
    ax.barh(positions, gaps, color=colors)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.axvline(0.0, color="#111827", linewidth=1)
    ax.set_xlabel("Mean paired quality difference vs Full 14B Target")
    _save_pair(fig, output_dir, FIGURE_NAMES[3])


def _save_pair(figure: Any, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    figure.clear()


def generate_figure_suite(
    *,
    raw_rows_path: Path,
    output_dir: Path,
    renderer: Callable[[Mapping[str, Any], Path], None] = _render_matplotlib,
) -> dict[str, Any]:
    raw_rows_path = raw_rows_path.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise MainTablePaperFigureError("output directory must be new and absent")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    try:
        rows = load_raw_rows(raw_rows_path)
        data = build_figure_data(rows)
        _write_json(staging / "figure-data.json", data)
        os.environ.setdefault("MPLCONFIGDIR", str(staging / "matplotlib-cache"))
        renderer(data, staging)
        figures: list[dict[str, Any]] = []
        for name in FIGURE_NAMES:
            files: dict[str, Any] = {}
            for extension in ("png", "pdf"):
                path = staging / f"{name}.{extension}"
                if not path.is_file() or path.is_symlink() or path.stat().st_size < 1:
                    raise MainTablePaperFigureError(f"renderer omitted {path.name}")
                sha256, nbytes = _sha256_file(path)
                files[extension] = {"path": path.name, "sha256": sha256, "nbytes": nbytes}
            figures.append({"name": name, "files": files})
        raw_sha256, raw_nbytes = _sha256_file(raw_rows_path)
        figure_data_sha256, figure_data_nbytes = _sha256_file(staging / "figure-data.json")
        result = {
            "schema": RESULT_SCHEMA,
            "status": "succeeded",
            "row_count": data["row_count"],
            "prompt_count": data["prompt_count"],
            "method_count": data["method_count"],
            "figure_count": len(figures),
            "raw_rows": {"path": str(raw_rows_path), "sha256": raw_sha256, "nbytes": raw_nbytes},
            "figure_data": {"path": "figure-data.json", "sha256": figure_data_sha256, "nbytes": figure_data_nbytes},
            "figures": figures,
        }
        _write_json(staging / "result.json", result)
        os.replace(staging, output_dir)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def generate_figure_suite_from_merge(
    *,
    merge_dir: Path,
    output_dir: Path,
    renderer: Callable[[Mapping[str, Any], Path], None] = _render_matplotlib,
) -> dict[str, Any]:
    raw_rows_path = validate_merge_directory(merge_dir)
    return generate_figure_suite(
        raw_rows_path=raw_rows_path,
        output_dir=output_dir,
        renderer=renderer,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cfexp.main_table_paper_figures")
    parser.add_argument("--merge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = generate_figure_suite_from_merge(
            merge_dir=args.merge_dir,
            output_dir=args.output_dir,
        )
    except (MainTablePaperFigureError, OSError) as error:
        print(f"main-table-paper-figures: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FIGURE_NAMES",
    "MainTablePaperFigureError",
    "RAW_SCHEMA",
    "RESULT_SCHEMA",
    "build_figure_data",
    "generate_figure_suite",
    "generate_figure_suite_from_merge",
    "load_raw_rows",
    "validate_merge_directory",
]
