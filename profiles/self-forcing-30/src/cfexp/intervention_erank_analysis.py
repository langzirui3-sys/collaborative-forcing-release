"""Read-only, sealed-run eRank alignment; no automatic quality labels."""
from __future__ import annotations
import argparse
import json
import math
import hashlib
from pathlib import Path
from .motion_video_audit import require
from .single_target_recovery import CHUNKS, video_time, write_json
from .self_forcing_frame_adapter import validate_effective_rank_evidence


def observations(trace):
    rows = trace['chunks']
    require(len(rows) == CHUNKS, 'incomplete rank trajectory')
    scheduled = set(trace['intervention_chunks'])
    points = []
    for i, row in enumerate(rows):
        require(row['chunk'] == i, 'rank chunk order drift')
        target = i in scheduled
        require(row['producer'] == ('target' if target else 'draft'), 'producer schedule drift')
        item = row['draft_erank_observation']
        if target:
            require(item is None, 'Target block must not carry Draft rank')
            points.append(None)
            continue
        require(item['model'] == 'draft' and item['layer_index_zero_based'] == 14
                and item['timestep'] == 1000 and item['video_seconds'] == video_time(i),
                'rank protocol or timestamp drift')
        points.append(validate_effective_rank_evidence(item['evidence'],
            expected_rank=item['rank'], expected_eigenvalue_count=256))
    return points


def aligned(traces):
    require('draft_only' in traces, 'missing matched baseline')
    require(not traces['draft_only']['intervention_chunks'], 'baseline contains Target')
    ranks = {name: observations(trace) for name, trace in traces.items()}
    base = ranks['draft_only']
    return {'video_seconds': [video_time(i) for i in range(CHUNKS)],
            'ranks': ranks,
            'delta_from_baseline': {name: [None if x is None else x-b for x,b in zip(values,base)]
                                    for name,values in ranks.items()},
            'target_intervals': {name: [[video_time(i), video_time(i)+.5]
                                for i in trace['intervention_chunks']] for name,trace in traces.items()},
            'quality_conclusion': 'not inferred from rank; requires independent visual review'}


def visual_intervals(review, *, run_id, prompt_id, arms):
    require(review.get('run_id') == run_id and review.get('prompt_id') == prompt_id,
            'visual review belongs to another run or prompt')
    require(isinstance(review.get('reviewer'), str) and bool(review['reviewer'].strip()),
            'missing visual reviewer')
    require(review.get('rank_hidden_during_review') is True,
            'visual review must be performed without rank curves')
    rows = review.get('intervals')
    require(isinstance(rows,list) and bool(rows), 'missing visual intervals')
    for row in rows:
        require(row.get('arm') in arms, 'unknown visual review arm')
        require(row.get('label') in {'severe_artifact','noncollapsed','uncertain'},
                'invalid visual review label')
        a,b=row.get('start_seconds'),row.get('end_seconds')
        require(all(type(x) in (int,float) and math.isfinite(x) for x in (a,b))
                and 0 <= a < b <= 60, 'invalid visual review interval')
        require(isinstance(row.get('evidence'),str) and bool(row['evidence'].strip()),
                'visual label needs frame/video evidence')
    return rows


def run(source, output, review_path, *, source_index=None, review_sha256=None):
    from .main_table_merge import _load_frozen_index
    source = source.resolve(strict=True)
    from .stage_timing import _unsealed_directory
    _unsealed_directory(output.parent)
    require(not output.resolve().is_relative_to(source), 'analysis cannot write into source run')
    frozen, marker = _load_frozen_index(source)
    if source_index is not None:
        require(marker['artifact_index_sha256'] == source_index, 'source index drift')
    attempt = source/'attempts'/marker['attempt_id']
    result_path = attempt/'result.json'
    frozen.file(result_path.relative_to(source).as_posix(), label='rank result')
    result = json.loads(result_path.read_text())
    traces = {}
    for arm in result['comparison_row_order']:
        path = attempt/'arms'/arm/'trace.json'
        frozen.file(path.relative_to(source).as_posix(), label='rank trace')
        traces[arm] = json.loads(path.read_text())
    data = aligned(traces)
    data['source'] = {'run_id':source.name,'index_sha256':marker['artifact_index_sha256'],
                      'code_sha':marker['git_sha'],'prompt_id':result['prompt_id']}
    review_bytes=review_path.read_bytes()
    if review_sha256 is not None:
        require(hashlib.sha256(review_bytes).hexdigest() == review_sha256, 'visual review digest drift')
    review=json.loads(review_bytes)
    intervals=visual_intervals(review,run_id=source.name,prompt_id=result['prompt_id'],arms=traces)
    data['visual_review']=review
    data['visual_review_sha256']=hashlib.sha256(review_bytes).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'aligned-erank.json',data)
    (output/'visual-review.json').write_bytes(review_bytes)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,1,figsize=(14,10),sharex=True)
    for name, values in data['ranks'].items():
        y = [float('nan') if x is None else x for x in values]
        axes[0].plot(data['video_seconds'],y,label=name)
        axes[1].plot(data['video_seconds'],[float('nan') if x is None else x
                     for x in data['delta_from_baseline'][name]],label=name)
    axes[0].set_ylabel('Draft eRank, layer 15, t=1000')
    axes[1].set_ylabel('eRank minus matched Draft baseline')
    axes[2].set_xlabel('Video time (seconds)')
    names=list(traces)
    colors={'severe_artifact':'#c62828','noncollapsed':'#2e7d32','uncertain':'#777777'}
    for row in intervals:
        axes[2].broken_barh([(row['start_seconds'],row['end_seconds']-row['start_seconds'])],
            (names.index(row['arm'])-.3,.6),facecolors=colors[row['label']])
    axes[2].set_yticks(range(len(names)),names,fontsize=8)
    axes[2].set_title('Rank-hidden frame review: green noncollapsed / red severe artifact / grey uncertain; ticks = samples, blank = unreviewed',fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
        for start,end in sorted({tuple(v) for intervals in data['target_intervals'].values() for v in intervals}):
            ax.axvspan(start,end,color='grey',alpha=.12)
    axes[0].legend(fontsize=8)
    fig.suptitle(result['prompt_id']+' — gaps = Target blocks; shading = union of interventions')
    fig.tight_layout()
    fig.savefig(output/'aligned-erank.png',dpi=160)
    fig.savefig(output/'aligned-erank.pdf')
    plt.close(fig)
    return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--visual-review',type=Path,required=True)
    parser.add_argument('--source-index',required=True)
    parser.add_argument('--visual-review-sha256',required=True)
    args=parser.parse_args()
    from .main_table_baseline_campaign import _run_dir
    run(args.source_run,_run_dir()/'erank-analysis',args.visual_review,
        source_index=args.source_index,review_sha256=args.visual_review_sha256)

if __name__ == '__main__':main()
