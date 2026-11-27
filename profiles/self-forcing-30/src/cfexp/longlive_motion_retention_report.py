"""Measure motion persistence on the original scored LongLive continuation pair."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from statistics import mean, median
import struct
from types import SimpleNamespace

from . import native_ready_report as audit
from .correction_state import LatentHistory
from .longlive_motion import _BF16, _FRAME_ELEMENTS, _SHAPE, latent_motion_energy
from .longlive_motion_continuation import CONTROL_ARM, GAIN_BY_LAG, GUIDANCE_ARMS, GUIDED_ARM, TARGET_POSITIONS
from .longlive_spacing_report import verify_route
from .main_table_v1 import QUALITY_DIMENSIONS

LAST_TARGET_BLOCKS = TARGET_POSITIONS[1::2]
LAGS = tuple(range(7))
AMPLITUDE_SPATIAL_STRIDE = 4


def _frame_block(frame):
    if frame < 9:
        return 0
    return 1 + (frame - 9) // 12


def decoded_gray_motion(path):
    """Measure only adjacent frames within each native 30-second block."""
    import av

    changes = [[] for _ in range(60)]
    previous = None
    previous_block = None
    frames = 0
    with av.open(str(path)) as video:
        stream, = video.streams.video
        audit._require(stream.average_rate == 24, 'motion measurement requires the original 24 fps video')
        for index, frame in enumerate(video.decode(stream)):
            block = _frame_block(index)
            audit._require(block < len(changes), 'motion measurement has excess frames')
            scaled = frame.reformat(width=64, height=36, format='gray')
            plane, = scaled.planes
            data = bytes(plane)
            stride = plane.line_size
            pixels = b''.join(data[row*stride:row*stride+64] for row in range(36))
            audit._require(len(pixels) == 2304, 'invalid gray motion sample')
            if block == previous_block:
                changes[block].append(sum(abs(a-b) for a, b in zip(previous, pixels)) / len(pixels))
            previous, previous_block = pixels, block
            frames += 1
    audit._require(frames == 717 and [len(values) for values in changes] == [8]+[11]*59,
                   'motion frames do not match the original native block geometry')
    return [mean(values) for values in changes]


def _motion_profile(rows, field, lags):
    rng = random.Random(20261112)
    draws = [rng.choices(range(len(rows)), k=len(rows)) for _ in range(10_000)]
    profile = []
    for lag in lags:
        values = [row[field][str(lag)] for row in rows]
        distribution = sorted(median(values[i] for i in ids) for ids in draws)
        profile.append(dict(lag=lag, median=median(values), mean=mean(values),
                            paired_bootstrap95=[distribution[249], distribution[9749]]))
    return profile


def _per_lag(control, guided, anchors, lags):
    return {str(lag): mean(guided[n+lag] - control[n+lag] for n in anchors) for lag in lags}


def _audited_cohort(run, quality_report, revision, guidance_arm=GUIDED_ARM):
    audit._require(guidance_arm in GUIDANCE_ARMS, 'unknown LongLive guidance arm')
    audit._require(run.is_dir() and not run.is_symlink(), 'invalid original motion run directory')
    frozen, marker = audit._load_frozen_index(run)
    run = frozen.run_dir
    quality = json.loads(quality_report.read_bytes())
    audit._require(quality['schema'] == 'longlive_spacing_verified_report_v1'
        and quality['status'] == 'verified_original_local_seal'
        and quality['nominal_seconds'] == 30
        and quality['full_target_positions'] == {arm: list(TARGET_POSITIONS)
                                                  for arm in (CONTROL_ARM, guidance_arm)}
        and quality['run_id'] == marker['run_id'] == run.name
        and quality['attempt_id'] == marker['attempt_id']
        and quality['generation_revision'] == marker['git_sha'] == revision
        and quality['marker_sha256'] == audit._sha256_file(run/'_SUCCESS.json')[0]
        and quality['artifact_index_sha256'] == marker['artifact_index_sha256'],
        'motion report must bind the independently audited original execution seal')

    prefix = 'attempts/' + marker['attempt_id'] + '/'
    src = SimpleNamespace(frozen=frozen, prefix=prefix, run=run)
    result = audit.read_json(src, 'result.json')
    original = {(row['prompt_id'], row['arm']): row for row in result['rows']}
    checked = {(row['prompt_id'], row['arm']): row for row in quality['rows']}
    prompts = sorted({prompt for prompt, _ in original})
    score_fields = {'arm', 'prompt_id', 'prompt', 'quality_scores', 'video_ready_wall_seconds', 'video'}
    audit._require(result['git']['sha'] == revision and result['status'] == 'succeeded'
        and len(original) == len(checked) == len(result['rows']) == len(quality['rows']) == 96
        and len(prompts) == 24 and set(original) == set(checked)
        and set(original) == {(prompt, arm) for prompt in prompts
                              for arm in ('target', 'draft_root', CONTROL_ARM, guidance_arm)}
        and all(set(row) == score_fields and all(original[key][field] == value for field, value in row.items())
                for key, row in checked.items()),
        'motion report requires the complete original four-arm cohort')
    return src, marker, original, prompts


def _paired_outcomes(cells, prompts, guidance_arm):
    rng = random.Random(20261112)
    full = tuple(range(len(prompts)))
    draws = [rng.choices(full, k=len(full)) for _ in range(10_000)]

    def interval(estimate):
        distribution = sorted(estimate(ids) for ids in draws)
        return dict(estimate=estimate(full), paired_bootstrap95=[distribution[249], distribution[9749]])

    def sampled_mean(values, ids):
        return math.fsum(values[i] for i in ids) / len(ids)

    def qualities(row):
        return 100 * math.fsum(row['quality_scores'][d] for d in QUALITY_DIMENSIONS) / len(QUALITY_DIMENSIONS)

    guided = [cells[(p, guidance_arm)] for p in prompts]
    out = {}
    for reference in ('target', 'draft_root', CONTROL_ARM):
        control = [cells[(p, reference)] for p in prompts]
        quality_delta = [qualities(a) - qualities(b) for a, b in zip(guided, control, strict=True)]
        dimensions = {d: [100 * (a['quality_scores'][d] - b['quality_scores'][d])
                          for a, b in zip(guided, control, strict=True)] for d in QUALITY_DIMENSIONS}
        guided_time = [x['video_ready_wall_seconds'] for x in guided]
        control_time = [x['video_ready_wall_seconds'] for x in control]
        dynamic_delta = dimensions['dynamic_degree']
        out[reference] = dict(
            quality_delta_pp=interval(lambda ids: sampled_mean(quality_delta, ids)),
            dimension_delta_pp={d: interval(lambda ids: sampled_mean(values, ids))
                                for d, values in dimensions.items()},
            reference_to_guided_ready_ratio=interval(lambda ids:
                math.fsum(control_time[i] for i in ids) / math.fsum(guided_time[i] for i in ids)),
            dynamic_degree_improved=sum(delta > 0 for delta in dynamic_delta),
            dynamic_degree_declined=sum(delta < 0 for delta in dynamic_delta))
    return out


def paired_outcomes_report(run, quality_report, *, revision, guidance_arm=GUIDED_ARM):
    """Compute matched quality and video-ready uncertainty from sealed scores."""
    src, marker, cells, prompts = _audited_cohort(run, quality_report, revision, guidance_arm)
    return dict(
        schema='longlive_motion_paired_outcomes_v1', status='verified',
        run_id=src.run.name, attempt_id=marker['attempt_id'], generation_revision=revision,
        original_artifact_index_sha256=marker['artifact_index_sha256'],
        quality_report_sha256=audit._sha256_file(quality_report)[0],
        guidance_arm=guidance_arm, prompts=len(prompts), bootstrap_seed=20261112, bootstrap_draws=10_000,
        comparisons=_paired_outcomes(cells, prompts, guidance_arm))


def report(run, quality_report, *, revision, guidance_arm=GUIDED_ARM):
    """Bind both diagnostics to the audited videos and accepted native latents."""
    src, marker, original, prompts = _audited_cohort(run, quality_report, revision, guidance_arm)
    run, prefix, frozen = src.run, src.prefix, src.frozen

    rows = []
    for prompt in prompts:
        decoded, latent, histories = {}, {}, {}
        cells = [original[(prompt, arm)] for arm in (CONTROL_ARM, guidance_arm)]
        for cell in cells:
            arm = cell['arm']
            trace = audit._bound(frozen, prefix, cell['native_trajectory'], 'motion trajectory')
            history = histories[arm] = LatentHistory.from_dict(trace['history'])
            verify_route(cell, trace, history, 60)
            video = audit.referenced_file(src, cell['video'], 'motion report scored video')
            decoded[arm] = decoded_gray_motion(video)
            latent[arm] = {
                block: latent_motion_energy(run/prefix/trace['input_root'], history.segments[block])
                for block in {n+lag for n in LAST_TARGET_BLOCKS for lag in LAGS[1:]}
            }
        control, guided = cells
        before = histories[CONTROL_ARM].segments
        after = histories[guidance_arm].segments
        differences = [i for i, (a, b) in enumerate(zip(before, after, strict=True))
                       if a.latent_sha256 != b.latent_sha256]
        audit._require(control['prompt'] == guided['prompt']
            and control['initial_noise_sha256'] == guided['initial_noise_sha256']
            and (not differences or differences[0] > LAST_TARGET_BLOCKS[0]),
            'guide and control did not share the original Target prefix and noise')

        rows.append(dict(
            prompt_id=prompt,
            first_changed_latent_chunk=differences[0] if differences else None,
            video_sha256={row['arm']: row['video']['sha256'] for row in cells},
            dynamic_degree={row['arm']: row['quality_scores']['dynamic_degree'] for row in cells},
            motion_smoothness={row['arm']: row['quality_scores']['motion_smoothness'] for row in cells},
            first_decoded_gray=_per_lag(decoded[CONTROL_ARM], decoded[guidance_arm],
                                       LAST_TARGET_BLOCKS[:1], LAGS),
            all_decoded_gray=_per_lag(decoded[CONTROL_ARM], decoded[guidance_arm],
                                     LAST_TARGET_BLOCKS, LAGS),
            first_latent_energy=_per_lag(latent[CONTROL_ARM], latent[guidance_arm],
                                        LAST_TARGET_BLOCKS[:1], LAGS[1:]),
            all_latent_energy=_per_lag(latent[CONTROL_ARM], latent[guidance_arm],
                                      LAST_TARGET_BLOCKS, LAGS[1:])))
    transitions = {
        f'{old:g}_to_{new:g}': sum(row['dynamic_degree'][CONTROL_ARM] == old and
                                  row['dynamic_degree'][guidance_arm] == new for row in rows)
        for old in (0, 1) for new in (0, 1)
    }
    audit._require(sum(transitions.values()) == len(rows), 'motion report dynamic labels are not binary')
    return dict(
        schema='longlive_motion_retention_report_v1', status='verified',
        run_id=run.name, attempt_id=marker['attempt_id'], generation_revision=revision,
        original_artifact_index_sha256=marker['artifact_index_sha256'],
        quality_report_sha256=audit._sha256_file(quality_report)[0],
        guidance_arm=guidance_arm, prompts=len(rows), final_target_chunks=list(LAST_TARGET_BLOCKS),
        guidance_lags=[1, 2, 3], bootstrap_seed=20261112, bootstrap_draws=10_000,
        dynamic_degree_transitions=transitions,
        effects={metric: _motion_profile(rows, metric, LAGS if metric.endswith('gray') else LAGS[1:])
                 for metric in ('first_decoded_gray', 'all_decoded_gray',
                                'first_latent_energy', 'all_latent_energy')},
        rows=rows,
        scope='First window has identical accepted Target history. Later windows include accumulated '
              'trajectory changes. Gray variation measures appearance and motion together. '
              'Latent energy measures adjacent-frame differences normalized by latent magnitude.')


def _bf16_payload(src, input_root, segment):
    phase = SimpleNamespace(frozen=src.frozen, prefix=src.prefix+input_root+'/', run=src.run)
    path = audit.referenced_file(phase, dict(path=segment.latent_path,
        sha256=segment.latent_sha256, nbytes=segment.latent_nbytes), 'amplitude latent')
    raw = path.read_bytes()
    audit._require(len(raw) == segment.latent_nbytes and sha256(raw).hexdigest() == segment.latent_sha256,
                   'amplitude bytes differ from the accepted latent')
    header_size = struct.unpack_from('<Q', raw)[0]
    audit._require(8 < header_size < 4096, 'invalid amplitude latent header')
    start = 8 + header_size
    tensor = json.loads(raw[8:start])['latent']
    audit._require(tensor['shape'] == _SHAPE and tensor['dtype'] == 'BF16'
        and tensor['data_offsets'] == [0, 6*_FRAME_ELEMENTS]
        and len(raw) == start + 6*_FRAME_ELEMENTS, 'amplitude latent must have the native layout')
    return raw[start:]


def _position_amplitude(reference, control, guided, gain):
    """Compare temporal changes across channels at one spatial position."""
    direction = tuple((b-a)*0.5 for a, b in zip(reference[0], reference[2], strict=True))
    norm_squared = math.fsum(x*x for x in direction)
    for step in (0, 1):
        delta = tuple(b-a for a, b in zip(control[step], control[step+1], strict=True))
        actual = tuple(b-a-base for a, b, base in
                       zip(guided[step], guided[step+1], delta, strict=True))
        projection = math.fsum(a*b for a, b in zip(delta, direction, strict=True)) / max(norm_squared, 1e-8)
        aligned = min(1.0, max(0.0, projection))
        factor = min(gain*aligned, 1-aligned)
        yield (projection, math.fsum(x*x for x in delta), factor*factor*norm_squared,
               math.fsum(x*x for x in actual))


def amplitude(reference, control, guided):
    """Summarize the prescribed and realized change on a fixed spatial grid."""
    audit._require(len(reference) == len(control) == len(guided) == 6*_FRAME_ELEMENTS,
                   'amplitude payload has an invalid frame count')
    audit._require(control[:2*_FRAME_ELEMENTS] == guided[:2*_FRAME_ELEMENTS],
                   'first guided frame changed at the shared-prefix intervention')
    _, _, channels, height, width = _SHAPE
    buckets = dict(nonpositive=0, inside=0, saturated=0)
    control_energy = prescribed_energy = realized_energy = 0.0
    changed = 0

    def sample(payload, spatial):
        return tuple(tuple(_BF16[struct.unpack_from('<H', payload,
            2*(frame*_FRAME_ELEMENTS + channel*height*width + spatial))[0]]
            for channel in range(channels)) for frame in range(3))

    for y in range(0, height, AMPLITUDE_SPATIAL_STRIDE):
        for x in range(0, width, AMPLITUDE_SPATIAL_STRIDE):
            r, c, g = (sample(payload, y*width+x) for payload in (reference, control, guided))
            audit._require(all(math.isfinite(value) for frames in (r, c, g)
                               for frame in frames for value in frame), 'nonfinite amplitude sample')
            changed += sum(a != b for old, new in zip(c[1:], g[1:], strict=True)
                           for a, b in zip(old, new, strict=True))
            for projection, base, prescribed, realized in _position_amplitude(r, c, g, GAIN_BY_LAG[0]):
                bucket = 'nonpositive' if projection <= 0 else 'saturated' if projection >= 1 else 'inside'
                buckets[bucket] += 1
                control_energy += base
                prescribed_energy += prescribed
                realized_energy += realized
    transitions = sum(buckets.values())
    samples = transitions*channels
    return dict(sampled_transitions=transitions,
                projection_fractions={name: count/transitions for name, count in buckets.items()},
                changed_output_fraction=changed/samples,
                control_delta_rms=math.sqrt(control_energy/samples),
                prescribed_increment_rms=math.sqrt(prescribed_energy/samples),
                realized_increment_rms=math.sqrt(realized_energy/samples))


def first_block_amplitude_report(run, quality_report, *, revision):
    """Measure the first guided block, where both histories still match."""
    src, marker, original, prompts = _audited_cohort(run, quality_report, revision)
    anchor = LAST_TARGET_BLOCKS[0]
    first = anchor + 1
    rows = []
    for prompt in prompts:
        cells = [original[(prompt, arm)] for arm in (CONTROL_ARM, GUIDED_ARM)]
        traces, histories = [], []
        for cell in cells:
            trace = audit._bound(src.frozen, src.prefix, cell['native_trajectory'], 'amplitude trajectory')
            history = LatentHistory.from_dict(trace['history'])
            verify_route(cell, trace, history, 60)
            traces.append(trace)
            histories.append(history.segments)
        before, after = histories
        audit._require([s.latent_sha256 for s in before[:first]] == [s.latent_sha256 for s in after[:first]]
            and cells[0]['initial_noise_sha256'][first] == cells[1]['initial_noise_sha256'][first],
            'amplitude requires the exact shared accepted prefix and first Draft noise')
        inputs = (_bf16_payload(src, traces[0]['input_root'], before[anchor]),
                  _bf16_payload(src, traces[0]['input_root'], before[first]),
                  _bf16_payload(src, traces[1]['input_root'], after[first]))
        stats = amplitude(*inputs)
        rows.append(dict(prompt_id=prompt, **stats,
                         reference_sha256=before[anchor].latent_sha256,
                         control_sha256=before[first].latent_sha256,
                         guided_sha256=after[first].latent_sha256))
    return dict(schema='longlive_first_guided_amplitude_v1', status='verified',
        run_id=marker['run_id'], attempt_id=marker['attempt_id'], generation_revision=revision,
        original_artifact_index_sha256=marker['artifact_index_sha256'],
        quality_report_sha256=audit._sha256_file(quality_report)[0],
        reference_chunk=anchor, first_guided_chunk=first, spatial_stride=AMPLITUDE_SPATIAL_STRIDE,
        rows=rows,
        median_changed_output_fraction=median(row['changed_output_fraction'] for row in rows),
        median_aligned_fraction=median(row['projection_fractions']['inside'] for row in rows),
        median_prescribed_increment_rms=median(row['prescribed_increment_rms'] for row in rows),
        median_realized_increment_rms=median(row['realized_increment_rms'] for row in rows),
        scope='CPU projection on fixed spatial samples and measured BF16 changes at the first '
              'guided block. This diagnostic does not measure perceptual video motion.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--quality-report', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--first-block-amplitude', action='store_true')
    modes.add_argument('--paired-outcomes', action='store_true')
    parser.add_argument('--guidance-arm', choices=GUIDANCE_ARMS, default=GUIDED_ARM)
    args = parser.parse_args()
    if args.first_block_amplitude:
        if args.guidance_arm != GUIDED_ARM:
            parser.error('first-block amplitude uses the original aligned-motion rule')
        result = first_block_amplitude_report(args.run, args.quality_report, revision=args.revision)
    else:
        reader = paired_outcomes_report if args.paired_outcomes else report
        result = reader(args.run, args.quality_report, revision=args.revision, guidance_arm=args.guidance_arm)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
