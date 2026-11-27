"""Fixed cross-model outcomes; visual labels must be frozen before ranks open.

No visual label is inferred here. Model families keep their own clock, Target
budget and reference latency; a censored trajectory never gets an invented
collapse time. These functions operate on already verified source artifacts.
"""
from __future__ import annotations
import math

from .cross_model_delay import ARMS, geometry, require, schedule, video_interval
from .prevention_endpoints import validate_labels, visual_endpoint
from .prevention_cohort_analysis import restricted_onset_bounds

INTERVENTIONS = ARMS[2:]


def first_return_sample(pair, arm):
    require(arm in INTERVENTIONS, 'intervention required')
    return math.ceil(video_interval(pair, schedule(pair, arm)[-1])[1])


def window_counts(labels, start):
    validate_labels(labels)
    require(type(start) is int and 0 <= start < 60, 'invalid review window')
    window=labels[start:]
    return {'first_sample_seconds':start, 'samples':len(window),
        'noncollapsed':window.count('noncollapsed'), 'uncertain':window.count('uncertain'),
        'severe':window.count('severe_artifact')}


def warm_latency(row):
    # Do not sum chunk times: those miss conditioning, transport and cache setup.
    pipeline='pipeline_with_monitoring_wall_seconds' if 'pipeline_with_monitoring_wall_seconds' in row else 'pipeline_wall_seconds'
    names=(pipeline,'encode_seconds')
    require(all(type(row.get(k)) in (int,float) and math.isfinite(row[k]) and row[k]>0
                for k in names), 'complete measured warm pipeline and encode required')
    return sum(row[k] for k in names)


def validate_trace(pair, arm, trace):
    g=geometry(pair)
    native=(trace.get('native_pipeline') is True if pair=='longlive_krea'
        else trace.get('native_source_loop') is True and trace.get('independent_model_kv') is True)
    require(arm in ARMS and native and trace.get('rank_routes_generation') is False,
            'native ungated trace required')
    rows=trace.get('chunks',[])
    require(len(rows)==g['chunk_count'] and [x['chunk'] for x in rows]==list(range(g['chunk_count'])),
            'complete ordered chunk coverage required')
    targets=set(schedule(pair,arm))
    for i,row in enumerate(rows):
        producer='target' if i in targets else 'draft'
        require(row['producer']==producer and row['video_interval_seconds']==video_interval(pair,i),
                'producer or video clock drift')
        observation=row['erank_observation']
        expected={'longlive_krea':{'draft':'longlive_1.3b','target':'krea_14b'},
                  'lingbot_world_2':{'draft':'lingbot_1.3b','target':'lingbot_14b'}}[pair][producer]
        require(observation['chunk']==i and observation['producer']==expected,
                'rank producer binding drift')
        require(observation['layer_index_zero_based']==14 and observation['token_count']==256
            and observation['centered'] is False and observation['gram_dtype']=='float32'
            and observation['relative_gram_cutoff']==1e-7, 'rank definition drift')
        require(type(observation['rank']) in (int,float) and math.isfinite(observation['rank'])
            and 1 <= observation['rank'] <= 256.000001, 'invalid rank')
    return rows


def paired_outcome(pair, arm, labels, native_draft, timing, native_target_timing):
    require(arm in INTERVENTIONS,'intervention required')
    base=visual_endpoint(native_draft);current=visual_endpoint(labels)
    b=restricted_onset_bounds(native_draft);c=restricted_onset_bounds(labels)
    bounds=[c[0]-b[1],c[1]-b[0]]
    start=first_return_sample(pair,arm)
    bp=window_counts(native_draft,start);cp=window_counts(labels,start)
    seconds=warm_latency(timing);target=warm_latency(native_target_timing)
    unambiguous=all(x['possible_onset_seconds']==x['confirmed_onset_seconds']
        and x['confirmed_onset_seconds'] is not None for x in (base,current))
    positive=bounds[0]>0
    return {'arm':arm,'pair':pair,'native_draft_visual':base,'intervention_visual':current,
        'confirmed_onset_difference_seconds':current['confirmed_onset_seconds']-base['confirmed_onset_seconds']
            if unambiguous else None,
        'restricted_onset_cap_seconds':57,'restricted_onset_gain_bounds_seconds':bounds,
        'post_return':{'draft':bp,'intervention':cp,
            'noncollapsed_gain_bounds_samples':[cp['noncollapsed']-bp['noncollapsed']-bp['uncertain'],
                cp['noncollapsed']+cp['uncertain']-bp['noncollapsed']]},
        'warm_pipeline_plus_encode_seconds':seconds,'native_target_seconds':target,
        'speedup_vs_native_target':target/seconds,'saves_time_vs_native_target':seconds<target,
        'demonstrated_delay_within_observed_window':positive,
        'meets_joint_criterion':positive and seconds<target,
        'interpretation':'restricted observed onset; censored time is not an actual collapse time'}
