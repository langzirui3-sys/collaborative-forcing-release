"""Analysis of canonical, already seal-verified four-pair main-table rows.

The caller must verify source SUCCESS markers, indexes, raw files and metric
proofs before normalization. This arithmetic layer is not a seal verifier.
Partial cells expose coverage only, never a provisional quality or speedup.
"""
from __future__ import annotations
import math
from collections.abc import Mapping

PAIRS=('krea_self_forcing','krea_causal_forcing','krea_longlive','lingbot_world_2')
METHODS=('draft_only','full_target','sdvg','t_stitch','light_forcing','dummy_forcing','collaborative_forcing')
DURATIONS=(15,30,60)
from .vbench_long import VBENCH_LONG_DIMENSIONS as DIMENSIONS


def number(value, *, positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (positive and value<=0):
        raise ValueError('nonfinite or invalid measurement')
    return float(value)


def analyze(rows, *, prompt_ids):
    """Keep all 84 cells visible; aggregate only complete 24-prompt cells."""
    ids=tuple(prompt_ids)
    if len(ids)!=24 or len(set(ids))!=24 or any(not isinstance(p,str) or not p for p in ids):
        raise ValueError('requires the 24 distinct frozen prompt IDs')
    groups={}
    for raw in rows:
        if not isinstance(raw,Mapping):raise ValueError('row is not an object')
        pair=raw.get('model_pair');method=raw.get('method_id');duration=raw.get('duration_seconds')
        prompt=raw.get('prompt_id')
        if pair not in PAIRS or method not in METHODS or type(duration) is not int or duration not in DURATIONS or prompt not in ids:
            raise ValueError('row outside frozen four-pair matrix')
        if raw.get('seed')!=20260911 or type(raw.get('seed')) is not int:
            raise ValueError('seed drift')
        if raw.get('run_kind')!='formal_prompt_shard':raise ValueError('smoke/development rows cannot fill the main table')
        if raw.get('benchmark')!=('VBench-Long' if duration==60 else 'VBench'):
            raise ValueError('wrong quality evaluator for horizon')
        latency=number(raw.get('generation_wall_seconds'),positive=True)
        quality=raw.get('quality_scores')
        if not isinstance(quality,Mapping) or set(quality)!=set(DIMENSIONS):
            raise ValueError('requires all six quality dimensions')
        scores={d:number(quality[d]) for d in DIMENSIONS}
        if any(not 0<=x<=1 for x in scores.values()):raise ValueError('quality outside unit interval')
        key=(pair,method,duration);group=groups.setdefault(key,{})
        if prompt in group:raise ValueError('duplicate model/method/duration/prompt row')
        group[prompt]=(latency,scores)
    cells=[]
    for pair in PAIRS:
        for duration in DURATIONS:
            reference=groups.get((pair,'full_target',duration),{})
            for method in METHODS:
                group=groups.get((pair,method,duration),{})
                cell={'model_pair':pair,'method_id':method,'duration_seconds':duration,
                    'benchmark':'VBench-Long' if duration==60 else 'VBench',
                    'prompt_count':len(group),'missing_prompt_ids':[p for p in ids if p not in group],
                    'status':'complete' if len(group)==24 else 'incomplete',
                    'quality_dimension_means':None,'six_dimension_macro_mean':None,
                    'mean_generation_seconds':None,'speedup_vs_target':None}
                if len(group)==24:
                    means={d:math.fsum(group[p][1][d] for p in ids)/24 for d in DIMENSIONS}
                    latency=math.fsum(group[p][0] for p in ids)/24
                    cell.update(quality_dimension_means=means,
                        six_dimension_macro_mean=math.fsum(means.values())/6,
                        mean_generation_seconds=latency)
                    if len(reference)==24:
                        cell['speedup_vs_target']=(math.fsum(reference[p][0] for p in ids)/24)/latency
                cells.append(cell)
    complete=sum(c['status']=='complete' for c in cells)
    return {'schema':'four_pair_analysis_v1','expected_cells':84,'expected_rows':2016,
            'observed_rows':sum(len(g) for g in groups.values()),'complete_cells':complete,
            'status':'complete' if complete==84 else 'incomplete','cells':cells,
            'quality_scope':'six shared dimensions on 24 custom prompts; macro mean is not the official benchmark total',
            'speedup_scope':'ratio of paired-population mean generation latencies within the same model pair and duration'}
