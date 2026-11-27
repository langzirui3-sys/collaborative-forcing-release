"""Analyze the complete48-video study after binding all rank-hidden labels."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

from .cross_model_delay import ARMS,PAIRS,geometry,load_plan,require,schedule,sha,video_interval,write_new
from .cross_model_outcomes import INTERVENTIONS,paired_outcome,warm_latency
from .cross_model_review import GENERATION,media_paths,source_list,validate_case
from .prevention_endpoints import validate_labels,visual_endpoint


def validate_review(review,manifest,manifest_sha,sources,sources_sha):
    require(review['schema']=='cross_model_delay_visual_review_v1'
        and review['review_manifest_sha256']==manifest_sha,'review manifest binding drift')
    require(review['new_rank_data_hidden'] is True and review['prior_rank_exposure_cases']==[],
            'all new ranks must remain hidden until all labels are frozen')
    require(all(isinstance(review.get(k),str) and review[k].strip()
        for k in ('reviewer','reviewed_at_utc')),'review provenance missing')
    require(manifest['schema']=='cross_model_delay_rank_hidden_media_v1'
        and manifest['sources_sha256']==sources_sha and manifest['rank_data_excluded'] is True
        and manifest['arm_identity_hidden'] is False
        and manifest['generation_cases']==12 and manifest['video_count']==48,
        'complete rank-hidden media population required')
    wanted={(p,i) for p in PAIRS for i in range(6)}
    cases=manifest['cases'];reviews=review['cases']
    require(len(cases)==len(reviews)==12
        and {(x['pair'],x['prompt_index']) for x in cases}==wanted
        and {(x['pair'],x['prompt_index']) for x in reviews}==wanted,'incomplete or duplicate reviewed case')
    bound={(x['pair'],x['prompt_index']):x for x in sources}
    case_by={(x['pair'],x['prompt_index']):x for x in cases}
    files={x['path']:x for x in manifest['files']}
    wanted_paths={f'{p}/prompt-{i:02d}/'+m for p,i in wanted for m in media_paths()}
    require(len(manifest['files'])==len(files)==120 and set(files)==wanted_paths,
            'rank-hidden media whitelist drift')
    labels={}
    for row in reviews:
        key=row['pair'],row['prompt_index'];m=case_by[key];s=bound[key]
        prefix=f'{key[0]}/prompt-{key[1]:02d}/'
        require(row['run_id']==m['run_id']==Path(s['run_dir']).name
            and row['index_sha256']==m['index_sha256']==s['index_sha256']
            and m['generation_sha']==GENERATION[key[0]] and m['media_prefix']==prefix,
            'visual labels belong to another source')
        require(set(row['arms'])==set(ARMS),'all4 reviewed arms required')
        for arm,evidence in row['arms'].items():
            validate_labels(evidence['labels'])
            require(evidence['video_sha256']==files[prefix+f'arms/{arm}/video.mp4']['sha256']
                and evidence['evidence_panels']==[prefix+f'comparison-seconds-{i:02d}-{i+9:02d}.png'
                    for i in range(0,60,10)],'review does not bind exact full-horizon RGB evidence')
            require(isinstance(evidence['notes'],str) and evidence['notes'].strip(),'visual evidence notes required')
            labels[key+(arm,)]=evidence['labels']
    return labels


def draft_rank_comparison(traces,arm,pair):
    """Same producer/chunk after return, never compare Draft rank with Target."""
    start=max(schedule(pair,arm))+1;stop=min(start+10,geometry(pair)['chunk_count'])
    indices=list(range(start,stop));native=traces['draft_only']['chunks'];routed=traces[arm]['chunks']
    require(all(native[i]['producer']==routed[i]['producer']=='draft' for i in indices),
            'post-return rank must use the same Draft producer')
    def mean(rows):return sum(rows[i]['erank_observation']['rank'] for i in indices)/len(indices)
    base=mean(native);current=mean(routed)
    return {'chunk_indices':indices,'video_interval_seconds':[video_interval(pair,start)[0],video_interval(pair,stop-1)[1]],
        'native_draft_mean':base,'intervention_draft_mean':current,'mean_difference':current-base,
        'interpretation':'descriptive same-Draft hidden rank; no quality or causal mechanism claim'}


def summarize(rows,pair,arm):
    selected=[x for x in rows if x['pair']==pair]
    require(len(selected)==6 and {x['prompt_index'] for x in selected}==set(range(6)),
            'all6 fixed prompts required for family summary')
    values=[x['paired'][arm] for x in selected]
    mean=lambda fn:sum(fn(x) for x in values)/6
    return {'prompt_count':6,'joint_criterion_count':sum(x['meets_joint_criterion'] for x in values),
        'observed_delay_lower_bound_positive_count':sum(x['demonstrated_delay_within_observed_window'] for x in values),
        'time_saving_count':sum(x['saves_time_vs_native_target'] for x in values),
        'mean_speedup':mean(lambda x:x['speedup_vs_native_target']),
        'pooled_time_speedup':sum(x['native_target_seconds'] for x in values)/sum(x['warm_pipeline_plus_encode_seconds'] for x in values),
        'mean_restricted_onset_gain_bounds_seconds':[mean(lambda x:x['restricted_onset_gain_bounds_seconds'][j]) for j in (0,1)],
        'censored_native_draft_count':sum(x['native_draft_visual']['right_censored_no_confirmed_event'] for x in values),
        'censored_intervention_count':sum(x['intervention_visual']['right_censored_no_confirmed_event'] for x in values),
        'conclusion_scope':'all6 fixed prompts / one seed; exploratory within-family observations'}


def plot_case(case,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pair=case['pair'];i=case['prompt_index']
    colors={'draft':'#1565c0','target':'#b85c00'}
    fig,axes=plt.subplots(4,2,figsize=(15,11),sharex=True,sharey='col')
    for row,arm in enumerate(ARMS):
        chunks=case['arms'][arm]['chunks'];times=[sum(x['video_interval_seconds'])/2 for x in chunks]
        for col,producer in enumerate(('draft','target')):
            ax=axes[row,col]
            ranks=[x['erank_observation']['rank'] if x['producer']==producer else float('nan') for x in chunks]
            ax.plot(times,ranks,color=colors[producer],marker='.',markersize=3)
            if not any(x['producer']==producer for x in chunks):
                ax.text(.5,.5,f'No {producer} chunks in this arm',transform=ax.transAxes,ha='center',color='#666666')
            ax.set_title(f'{arm} / {producer} producer',fontsize=10);ax.set_ylabel('Hidden-state eRank')
            for chunk in schedule(pair,arm) if arm in INTERVENTIONS else ():
                a,b=video_interval(pair,chunk);ax.axvspan(a,b,color='#ffb74d',alpha=.18)
            ax.set_xlim(0,60);ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Video time (seconds)')
    fig.suptitle(f'{pair} / prompt {i:02d}: first-denoise block15 singular-entropy rank\n'
        '256 tokens; raw ranks across the two producer columns are not quality-comparable; gaps are not interpolated')
    fig.tight_layout();prefix=output/f'{pair}-prompt-{i:02d}-erank'
    fig.savefig(prefix.with_suffix('.png'),dpi=180);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(15,5),sharex=True)
    label_colors={'noncollapsed':'#2e7d32','severe_artifact':'#c62828','uncertain':'#888888'}
    for ax,arm in zip(axes,ARMS,strict=True):
        for second,label in enumerate(case['arms'][arm]['labels']):
            ax.broken_barh([(second,.85)],(0,.7),color=label_colors[label])
        for chunk in schedule(pair,arm) if arm in INTERVENTIONS else ():
            a,b=video_interval(pair,chunk);ax.broken_barh([(a,b-a)],(-.07,.84),facecolors='none',edgecolors='black')
        ax.set_yticks([]);ax.set_ylabel(arm,rotation=0,ha='right',fontsize=8);ax.set_xlim(0,60)
    axes[-1].set_xlabel('Video time (seconds)')
    fig.suptitle('Frozen 1Hz RGB review: green normal / red severe / grey uncertain; samples, not continuous labels\n'
        'Black outlines: the actual three Target chunks; labels were frozen before rank inspection',fontsize=10)
    fig.tight_layout();prefix=output/f'{pair}-prompt-{i:02d}-visual'
    fig.savefig(prefix.with_suffix('.png'),dpi=180);plt.close(fig)


def run(bundle,bundle_index,bundle_generation,review_path,review_sha,protocol_path,protocol_sha,output):
    from .main_table_merge import _load_frozen_index
    from .stage_timing import _unsealed_directory
    _unsealed_directory(output)
    require(not output.resolve().is_relative_to(bundle.resolve())
        and not bundle.resolve().is_relative_to(output.resolve()),'analysis output overlaps review bundle')
    frozen,marker=_load_frozen_index(bundle)
    require(marker['artifact_index_sha256']==bundle_index and marker['git_sha']==bundle_generation,
            'review bundle sealed identity drift')
    frozen.verify_all();attempt=bundle/'attempts'/marker['attempt_id']
    def read(root,index,base,relative):
        path=base/relative;index.file(path.relative_to(root).as_posix(),label='cross-model analysis input')
        return json.loads(path.read_text())
    manifest=read(bundle,frozen,attempt,'review-manifest.json');manifest_sha=sha(attempt/'review-manifest.json')
    require(manifest['protocol_sha256']==protocol_sha,'review protocol drift')
    read(bundle,frozen,attempt,'sources.json')
    sources=source_list(attempt/'sources.json',manifest['sources_sha256'])
    require(sha(review_path)==review_sha,'visual review digest drift')
    review=json.loads(review_path.read_text())
    # Validate ALL2880 visual labels and all source/media bindings before opening any new rank scalar.
    labels=validate_review(review,manifest,manifest_sha,sources,manifest['sources_sha256'])
    write_new(output/'accepted-review-binding.json',{'review_sha256':review_sha,'review_manifest_sha256':manifest_sha,
        'review_bundle_index_sha256':bundle_index,'review_bundle_generation':bundle_generation,
        'sources_sha256':manifest['sources_sha256'],'new_rank_data_hidden_until_all_labels_frozen':True})
    cases=[]
    for binding in sources:
        pair=binding['pair'];i=binding['prompt_index'];_,item=load_plan(protocol_path,protocol_sha,pair,i)
        root=Path(binding['run_dir']);source,seal=_load_frozen_index(root)
        require(seal['git_sha']==GENERATION[pair] and seal['artifact_index_sha256']==binding['index_sha256'],
                'generation source identity drift')
        require(not output.resolve().is_relative_to(root.resolve()) and not root.resolve().is_relative_to(output.resolve()),
                'analysis output overlaps source')
        source.verify_all();base=root/'attempts'/seal['attempt_id']
        result=read(root,source,base,'result.json');traces={a:read(root,source,base,f'arms/{a}/trace.json') for a in ARMS}
        timing=validate_case(result,traces,pair=pair,prompt_index=i,item=item,protocol_sha=protocol_sha)
        case={'pair':pair,'prompt_index':i,'prompt':item,'run_id':root.name,'index_sha256':binding['index_sha256'],
            'arms':{},'paired':{}}
        for arm in ARMS:
            case['arms'][arm]={'labels':labels[pair,i,arm],'visual':visual_endpoint(labels[pair,i,arm]),
                'timing':timing[arm],'chunks':traces[arm]['chunks']}
        for arm in INTERVENTIONS:
            case['paired'][arm]=paired_outcome(pair,arm,labels[pair,i,arm],labels[pair,i,'draft_only'],timing[arm],timing['target_only'])
            case['paired'][arm]['post_return_same_draft_rank']=draft_rank_comparison(traces,arm,pair)
        cases.append(case);plot_case(case,output)
        print('CROSS_MODEL_CASE_ANALYZED',pair,i,'all4 arms retained',flush=True)
    summary={pair:{arm:summarize(cases,pair,arm) for arm in INTERVENTIONS} for pair in PAIRS}
    write_new(output/'analysis.json',{'schema':'cross_model_delay_analysis_v1','case_count':12,'video_count':48,
        'protocol_sha256':protocol_sha,'visual_review_sha256':review_sha,'cases':cases,'family_summaries':summary,
        'limitations':['one seed and six fixed prompts per family; no population guarantee',
            'LingBot official image/action inputs are compared within that family only',
            'censoring and uncertainty retained; no invented collapse time',
            'eRank is producer-specific hidden-state diversity, not KV error or a quality score',
            'arm identity visible during rank-hidden visual review; annotations are not fully blinded',
            'timings include observation, native setup and VAE/MP4 work; model loading and warmup excluded']})
    with (output/'outcomes.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['pair','prompt_index','arm','warm_seconds','speedup_vs_target',
            'confirmed_onset_seconds','possible_onset_seconds','right_censored',
            'restricted_delay_lower_seconds','restricted_delay_upper_seconds','joint_criterion'])
        count=0
        for case in cases:
            target=warm_latency(case['arms']['target_only']['timing'])
            for arm,data in case['arms'].items():
                seconds=warm_latency(data['timing']);v=data['visual'];effect=case['paired'].get(arm)
                writer.writerow([case['pair'],case['prompt_index'],arm,seconds,target/seconds,
                    v['confirmed_onset_seconds'],v['possible_onset_seconds'],v['right_censored_no_confirmed_event'],
                    *(effect['restricted_onset_gain_bounds_seconds'] if effect else ['', '']),
                    effect['meets_joint_criterion'] if effect else ''])
                count+=1
        require(count==48,'all48 outcome rows required')
    with (output/'erank.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['pair','prompt_index','arm','chunk','start_seconds','end_seconds',
            'producer','layer_index_zero_based','token_count','hidden_erank'])
        count=0
        for case in cases:
            for arm,data in case['arms'].items():
                for chunk in data['chunks']:
                    rank=chunk['erank_observation']
                    writer.writerow([case['pair'],case['prompt_index'],arm,chunk['chunk'],
                        *chunk['video_interval_seconds'],rank['producer'],rank['layer_index_zero_based'],
                        rank['token_count'],rank['rank']]);count+=1
        require(count==4320,'all4320 native chunk ranks required')
    require(sha(review_path)==review_sha,'review changed after rank inspection')
    with (output/'frozen-visual-review.json').open('xb') as f:f.write(review_path.read_bytes())
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','review','protocol'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('bundle-index-sha256','bundle-generation-sha','review-sha256','protocol-sha256'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    from .main_table_baseline_campaign import _run_dir
    run(a.bundle,a.bundle_index_sha256,a.bundle_generation_sha,a.review,a.review_sha256,
        a.protocol,a.protocol_sha256,_run_dir())

if __name__=='__main__':main()
