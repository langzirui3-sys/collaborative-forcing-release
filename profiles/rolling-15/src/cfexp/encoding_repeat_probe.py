"""CPU-only encoder repeatability audit; never changes a source run or scores.

Repeated encoding uses RGB decoded from an existing MP4, not original VAE RGB.
It can test encoder repeatability, but cannot assign a corrected VBench score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .erank_acceptance_report import load_run
from .main_table_merge import _load_frozen_index

PROMPT_ID = "017_temporal_flickering_vbl0042"
SHAPE = (357, 480, 832, 3)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compare_rows(left, right):
    """Separate generated-latent equality from encoded-video equality."""
    require(len(left) == len(right) == 12, "expected frozen first12 source rows")
    ids = [r["prompt_id"] for r in left]
    require(len(set(ids)) == 12 and ids == [r["prompt_id"] for r in right],
            "source prompt populations/order differ")
    result = []
    for a, b in zip(left, right):
        require(a["prompt"] == b["prompt"] and a["seed"] == b["seed"] == 20260911,
                "source prompt or seed differs")
        ca, cb = a["trajectory"]["committed"], b["trajectory"]["committed"]
        require([x["chunk_index"] for x in ca] == list(range(30))
                and [x["chunk_index"] for x in cb] == list(range(30)),
                "source committed horizon is incomplete")
        result.append({"prompt_id": a["prompt_id"],
            "latent_hashes_equal": all(x["latent_sha256"] == y["latent_sha256"]
                                       for x, y in zip(ca, cb)),
            "accepted_positions_equal": a["trajectory"]["accepted_chunks"] == b["trajectory"]["accepted_chunks"],
            "video_hashes_equal": a["video"]["sha256"] == b["video"]["sha256"]})
    return result


def read_source(root):
    data, marker = load_run(root, target=False)
    frozen, _ = _load_frozen_index(root)
    prefix = "attempts/" + marker["attempt_id"] + "/"
    frames, videos = {}, {}
    for row in data["rows"]:
        evidence = json.loads(frozen.file(prefix + row["decode_evidence"]["path"],
                                         label="decode evidence").read_text())
        require(evidence["frames"]["shape"] == [357, 3, 480, 832]
                and evidence["frames"]["dtype"] == "uint8", "unexpected source frame geometry")
        frames[row["prompt_id"]] = evidence["frames"]["sha256"]
        videos[row["prompt_id"]] = frozen.file(prefix + row["video"]["path"], label="video")
    return data, marker, frames, videos


def decode_rgb(path, av, np):
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1, "source must have one video stream")
        values = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    frames = np.stack(values)
    require(tuple(frames.shape) == SHAPE and str(frames.dtype) == "uint8", "decoded video geometry differs")
    return frames


def encode_rgb(path, frames, *, options, av):
    require(not path.exists(), "encoder output already exists")
    # Exact torchvision 0.20.1 write_video video-only path. Preserve its default
    # ABR/bitrate; do not improve CRF or quality as part of a repeatability test.
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 832, 480, "yuv420p"
        stream.options = options
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pict_type = "NONE"
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def run(left_root, right_root, output):
    import av
    import numpy as np
    require(av.__version__ == "13.1.0", "audit requires the frozen PyAV version")
    left, lm, lf, lv = read_source(left_root)
    right, rm, rf, _ = read_source(right_root)
    require(left["code_revision"] == "38c03d2d0ebeff7c6504597140456b2ef20337ec"
            and right["code_revision"] == "b2a412cf9915262e58e1746d5f16cc94bda78fa8",
            "unexpected frozen source revisions")
    comparisons = compare_rows(left["rows"], right["rows"])
    for row in comparisons:
        row["recorded_lossless_frame_hashes_equal"] = lf[row["prompt_id"]] == rf[row["prompt_id"]]
    require(all(r["latent_hashes_equal"] and r["recorded_lossless_frame_hashes_equal"] for r in comparisons),
            "sources no longer isolate post-generation encoding differences")
    frames = decode_rgb(lv[PROMPT_ID], av, np)
    frame_hash = hashlib.sha256(frames.tobytes()).hexdigest()
    directory = output / "encoding-repeats"
    directory.mkdir(exist_ok=False)
    rows = []
    for kind, options in (("legacy_default", {}), ("serial_candidate", {"threads": "1"})):
        for repeat in range(3):
            path = directory / f"{kind}-{repeat}.mp4"
            started = time.perf_counter()
            encode_rgb(path, frames, options=options, av=av)
            elapsed = time.perf_counter() - started
            raw = path.read_bytes()
            decoded = decode_rgb(path, av, np)
            offset = raw.find(b"x264 - core")
            rows.append({"kind": kind, "repeat": repeat, "options": options,
                "path": str(path.relative_to(output)), "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "decoded_rgb_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
                "encode_wall_seconds": elapsed,
                "encoder_sei": raw[offset:offset+1200].split(b"\0")[0].decode("ascii", "replace") if offset >= 0 else None})
            del decoded
            require(hashlib.sha256(frames.tobytes()).hexdigest() == frame_hash, "encoder mutated its input RGB")
    report = {"schema": "encoding_repeat_probe_v1", "source_runs": [str(left_root), str(right_root)],
        "source_index_sha256": [lm["artifact_index_sha256"], rm["artifact_index_sha256"]],
        "source_comparisons": comparisons, "prompt_id": PROMPT_ID, "av_version": av.__version__,
        "av_libraries": av.library_versions, "input_rgb_sha256": frame_hash, "repeats": rows,
        "repeatability": {kind: {
            "distinct_mp4_hashes": len({r["sha256"] for r in rows if r["kind"] == kind}),
            "distinct_decoded_rgb_hashes": len({r["decoded_rgb_sha256"] for r in rows if r["kind"] == kind})}
            for kind in ("legacy_default", "serial_candidate")},
        "original_lossless_frames_reverified": False,
        "input_representation": "RGB decoded from the sealed legacy MP4, not original VAE frames",
        "quality_scores_recomputed": False, "generation_performed": False,
        "interpretation": "Encoder diagnostic only. No corrected quality, speedup, or method-success claim."}
    with (output / "result.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-run", type=Path, required=True)
    parser.add_argument("--right-run", type=Path, required=True)
    args = parser.parse_args()
    run(args.left_run, args.right_run, Path(os.environ["CFEXP_RUN_DIR"]))


if __name__ == "__main__":
    main()
