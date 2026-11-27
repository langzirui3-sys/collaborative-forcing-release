"""Compare Rolling K4 against genuine native Draft on a common Target cohort."""
from __future__ import annotations

import math
import random

from . import native_ready_report as ready
from . import native_reference_report as reference
from .main_table_v1 import QUALITY_DIMENSIONS

MODEL = 'rolling'
SECONDS = 60
PROMPTS = 24


def _source_identity(row):
    return tuple(row[key] for key in ('run_id', 'attempt_id', 'artifact_index_sha256'))


def summarize(draft, ours):
    """Join two verified comparisons on the same immutable Target output."""
    ready._require(draft['status'] == ours['status'] == 'verified'
        and draft['model'] == MODEL and ours['model_pair'] == 'krea_rolling_forcing'
        and draft['nominal_seconds'] == ours['nominal_seconds'] == SECONDS
        and draft['prompt_count'] == ours['prompt_count'] == PROMPTS
        and draft['seed'] == ours['seed'] == 20260911
        and draft['generation_code_revision'] == ours['generation_code_revision']
        and draft['timing_scope'] == ours['timing_scope'] == ready.OVERLAP_READY_SCOPE
        and ours['non_root_target_burst_length'] == 4 and ours['observation_horizon'] == 8
        and draft['common_codec_options'] == ours['actual_x264_options']
        and _source_identity(draft['sources'][1]) == _source_identity(ours['sources'][1]),
        'Rolling comparison must share the exact Target seal, source and timing scope')
    fields = ('prompt_id', 'target_quality_100', 'target_dimensions',
              'target_ready_seconds', 'target_video_sha256')
    pairs = []
    for d, o in zip(draft['pairs'], ours['pairs'], strict=True):
        ready._require(all(d[key] == o[key] for key in fields),
            'Rolling comparison changed the scored Target video or prompt')
        pairs.append(dict(prompt_id=d['prompt_id'],
            quality_delta=o['ours_quality_100'] - d['draft_quality_100'],
            dynamic_delta=100*(o['ours_dimensions']['dynamic_degree'] -
                               d['draft_dimensions']['dynamic_degree']),
            ready_delta=o['ours_ready_seconds'] - d['draft_ready_seconds']))
    ready._require(len(pairs) == len({row['prompt_id'] for row in pairs}) == PROMPTS,
        'Rolling comparison requires all 24 unique paired prompts')

    rng = random.Random(20261112)
    indices = [rng.choices(range(PROMPTS), k=PROMPTS) for _ in range(10_000)]
    effects = {}
    for field in ('quality_delta', 'dynamic_delta', 'ready_delta'):
        values = [row[field] for row in pairs]
        distribution = sorted(math.fsum(values[i] for i in ids) / PROMPTS for ids in indices)
        effects[field] = dict(paired_mean=math.fsum(values) / PROMPTS,
            prompt_resampling_95_percentile_interval=[distribution[249], distribution[9749]])
    dimensions = {
        key: 100*math.fsum(o['ours_dimensions'][key] - d['draft_dimensions'][key]
                        for d, o in zip(draft['pairs'], ours['pairs'])) / PROMPTS
        for key in QUALITY_DIMENSIONS
    }
    return dict(schema='native_rolling_draft_value_report_v1', status='verified',
        generation_code_revision=draft['generation_code_revision'],
        nominal_seconds=SECONDS, prompt_count=PROMPTS, seed=draft['seed'],
        bootstrap_seed=20261112, bootstrap_draws=len(indices),
        paired_effects_k4_minus_draft=effects, dimension_mean_deltas_100=dimensions,
        pairs=pairs, draft_vs_target=draft, ours_vs_target=ours)


def compare(draft, ours, target, *, repo):
    """Run the existing seal-bound pair auditors before joining their results."""
    pure = reference.compare(draft, target, repo=repo, model=MODEL, seconds=SECONDS)
    mixed = ready.compare(ours, target, repo=repo, model=MODEL, seconds=SECONDS, overlap=True)
    return summarize(pure, mixed)
