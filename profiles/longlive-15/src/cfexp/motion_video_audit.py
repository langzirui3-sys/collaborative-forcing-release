"""Read sealed development videos with an independent CPU decoder.

The audit is a separate formal run. Neither the source index nor its excluded,
but hash-bound, lossless frame tensors are ever modified. Missing transient
frames are explicitly reported, never regenerated or treated as matching. CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct
import subprocess
import zlib

from .main_table_merge import _load_frozen_index
from .motion_repair_probe import ARMS, PROMPT_IDS

SOURCE_SHA = "cd34826187ac6b43291ddbb81ff7cde5173f1b48"
SOURCE_INDEX = "5410681ed55b7f15a939c8aaf69ccb1064e3324aa8e5031c9e7434d20bd7b96e"
FRAME_INDICES = (0, 120, 240, 360, 480, 720, 1080, 1436)
SHAPE = (1437, 3, 480, 832)
STRIDE = 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest(), path.stat().st_size


def bound_path(root, binding):
    rel = binding["path"]
    p = PurePosixPath(rel)
    require(rel == str(p) and not p.is_absolute() and bool(p.parts)
            and all(x not in (".", "..") for x in p.parts), "unsafe evidence path")
    path = root
    for part in p.parts:
        path = path / part
        require(not path.is_symlink(), "evidence traverses symlink")
    require(path.resolve(strict=True).is_relative_to(root.resolve()), "evidence escapes root")
    require(digest(path) == (binding["sha256"], binding["nbytes"]), "evidence hash mismatch")
    return path


def rgb_samples(path, indices=FRAME_INDICES, shape=SHAPE, stride=STRIDE):
    """Parse only the documented dense U8 TCHW safetensors layout; no pickle."""
    with path.open("rb") as f:
        raw = f.read(8)
        require(len(raw) == 8, "truncated safetensors length")
        n = struct.unpack("<Q", raw)[0]
        require(2 <= n <= 1024 * 1024, "invalid safetensors header size")
        header = json.loads(f.read(n))
        require(set(header) <= {"frames", "__metadata__"} and "frames" in header,
                "unexpected safetensors keys")
        meta = header["frames"]
        total = shape[0] * shape[1] * shape[2] * shape[3]
        require(meta == {"dtype": "U8", "shape": list(shape), "data_offsets": [0, total]},
                "unexpected lossless frame layout")
        require(path.stat().st_size == 8 + n + total, "truncated or trailing tensor bytes")
        require(shape[1] == 3 and shape[2] % stride == shape[3] % stride == 0,
                "unsupported RGB dimensions")
        require(all(type(i) is int and 0 <= i < shape[0] for i in indices), "invalid frame index")
        plane = shape[2] * shape[3]
        frames = []
        for index in indices:
            f.seek(8 + n + index * 3 * plane)
            raw_frame = f.read(3 * plane)
            frame = bytearray()
            for y in range(0, shape[2], stride):
                for x in range(0, shape[3], stride):
                    pos = y * shape[3] + x
                    frame.extend((raw_frame[pos], raw_frame[plane + pos], raw_frame[2 * plane + pos]))
            frames.append(bytes(frame))
    return frames


# Isolated media interpreter is the existing experiment's PyAV environment.
# It decodes every frame in order, checks count and dimensions, and returns only
# the same nearest-neighbor RGB samples as rgb_samples (no interpolation).
MEDIA_CODE = '''import av,json,sys
indices=json.loads(sys.argv[2]); selected=[]; count=0
with av.open(sys.argv[1]) as container:
 assert len(container.streams.video)==1
 for i,frame in enumerate(container.decode(video=0)):
  assert (frame.width,frame.height)==(832,480)
  if i in indices: selected.append(frame.to_ndarray(format="rgb24")[::4,::4,:].tobytes())
  count=i+1
assert count==1437 and len(selected)==len(indices),(count,len(selected))
sys.stdout.buffer.write(b"".join(selected))
'''


def video_samples(path, python):
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([str(python), "-I", "-c", MEDIA_CODE, str(path), json.dumps(FRAME_INDICES)],
                            capture_output=True, env=env, timeout=300, check=True)
    size = (SHAPE[2] // STRIDE) * (SHAPE[3] // STRIDE) * 3
    require(len(result.stdout) == size * len(FRAME_INDICES), "wrong MP4 sample payload")
    return [result.stdout[i:i + size] for i in range(0, len(result.stdout), size)]


def png(path, width, height, rgb):
    require(len(rgb) == width * height * 3, "wrong RGB size")
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    scanlines = b"".join(b"\0" + rgb[i:i + width * 3] for i in range(0, len(rgb), width * 3))
    payload = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))
    with path.open("xb") as f:
        f.write(payload)


def sheet(path, rows):
    width, height = SHAPE[3] // STRIDE, SHAPE[2] // STRIDE
    require(all(len(row) == len(FRAME_INDICES) for row in rows), "wrong sheet columns")
    data = bytearray()
    for row in rows:
        for y in range(height):
            for frame in row:
                data.extend(frame[y * width * 3:(y + 1) * width * 3])
    png(path, width * len(FRAME_INDICES), height * len(rows), data)


def run(source, output, media_python):
    require(source.is_absolute() and output.is_absolute(), "absolute run paths required")
    source = source.resolve(strict=True)
    output = output.resolve(strict=True)
    require(not output.is_relative_to(source) and not source.is_relative_to(output), "source/output overlap")
    require(not any((p / name).exists() for p in (output, *output.parents)
                    for name in ("_SUCCESS.json", "_FAILED.json", "artifact-index.json")), "sealed output")
    require(not (output / "visual-audit").exists() and not (output / "result.json").exists(), "audit exists")
    frozen, marker = _load_frozen_index(source)
    require(marker["git_sha"] == SOURCE_SHA and marker["artifact_index_sha256"] == SOURCE_INDEX,
            "unexpected diagnostic source seal")
    attempt = source / "attempts" / marker["attempt_id"]
    result_path = frozen.file(str((attempt / "result.json").relative_to(source)), label="source result")
    original = json.loads(result_path.read_text())
    rows = original["rows"]
    require(len(rows) == 16 and {(r["arm"], r["prompt_id"]) for r in rows}
            == {(a, p) for a in ARMS for p in PROMPT_IDS}, "diagnostic population changed")
    dest = output / "visual-audit"
    dest.mkdir()
    summaries, panels = [], {}
    for row in rows:
        cell = attempt / "cells" / row["arm"]
        video = bound_path(attempt, row["video"])
        evidence_path = bound_path(attempt, row["decode_evidence"])
        evidence = json.loads(evidence_path.read_text())
        require(evidence["expected_frames"] == 1437 and evidence["frames"]["shape"] == list(SHAPE)
                and evidence["frames"]["dtype"] == "uint8", "decoded layout drift")
        require(evidence["video"]["sha256"] == row["video"]["sha256"], "video evidence mismatch")
        # prepare_full normally removes these transient tensors after encoding.
        # Missing RGB is an explicit limitation, not proof of encoding parity.
        raw = None
        relative_frames = evidence["frames"]["path"]
        expected_relative = "decoded/" + evidence["history_manifest_sha256"] + ".safetensors"
        require(relative_frames == expected_relative, "unexpected transient frame path")
        if (cell / relative_frames).exists() or (cell / relative_frames).is_symlink():
            frame_path = bound_path(cell, evidence["frames"])
            raw = rgb_samples(frame_path)
        encoded = video_samples(video, media_python)
        per_frame = []
        for slot, (index, mp4) in enumerate(zip(FRAME_INDICES, encoded, strict=True)):
            entry = {"frame": index, "seconds": index / 24,
                "mp4_clipped_channel_fraction": sum(v in (0, 255) for v in mp4) / len(mp4)}
            if raw is not None:
                rgb = raw[slot]
                differences = [abs(a - b) for a, b in zip(rgb, mp4, strict=True)]
                entry.update({"sampled_rgb_mean_absolute_mp4_error": sum(differences) / len(differences),
                    "sampled_rgb_maximum_mp4_error": max(differences),
                    "lossless_clipped_channel_fraction": sum(v in (0, 255) for v in rgb) / len(rgb)})
            per_frame.append(entry)
        panels[row["prompt_id"], row["arm"]] = encoded
        summaries.append({"arm": row["arm"], "prompt_id": row["prompt_id"],
            "video": row["video"], "decode_evidence": row["decode_evidence"],
            "lossless_frames": evidence["frames"], "lossless_frames_available": raw is not None,
            "samples": per_frame,
            "accepted_chunks": row["trajectory"]["accepted_chunks"]})
        print("AUDITED", row["arm"], row["prompt_id"], flush=True)
    sheets = []
    for prompt in PROMPT_IDS:
        path = dest / (prompt + ".png")
        sheet(path, [panels[prompt, arm] for arm in ARMS])
        sha, size = digest(path)
        sheets.append({"prompt_id": prompt, "path": str(path.relative_to(output)), "sha256": sha, "nbytes": size})
    report = {"schema": "motion_video_readonly_audit_v1", "source_run": str(source),
        "source_marker": marker, "source_result_sha256": digest(result_path)[0],
        "rows": summaries, "contact_sheets": sheets,
        "sheet_rows": [a + ":MP4_CPU_decode" for a in ARMS],
        "sheet_columns_frame_indices": FRAME_INDICES,
        "sampling": "nearest_neighbor_stride_4_no_color_adjustment", "regeneration_performed": False,
        "interpretation": "Visual diagnostic only; no automatic quality winner or inferred causal repair."}
    with (output / "result.json").open("x") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--media-python", type=Path, required=True)
    args = p.parse_args()
    output = Path(os.environ["CFEXP_RUN_DIR"])
    run(args.source_run, output, args.media_python)


if __name__ == "__main__":
    main()
