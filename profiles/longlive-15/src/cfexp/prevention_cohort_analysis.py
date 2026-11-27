"""Analyze all 36 paired arms only after a bound rank-hidden visual review."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from .motion_video_audit import require
from .paired_prevention import ARMS
from .prevention_review_bundle import GENERATION_SHA, source_list, validate_pair, media_paths
from .prevention_endpoints import validate_labels, visual_endpoint, post_return_counts, predictor_summary
from .prevention_rank_predictors import FEATURES, causal_features
from .single_target_recovery import write_json


def bound_json(path,sha):
    payload=path.read_bytes()
    require(hashlib.sha256(payload).hexdigest()==sha,'analysis input digest drift')
    return json.loads(payload),payload


def frozen_json(root,frozen,marker,relative):
    p=root/'attempts'/marker['attempt_id']/relative
    frozen.file(p.relative_to(root).as_posix(),label='cohort analysis evidence')
    return json.loads(p.read_text())


def validate_review(review,manifest,manifest_sha,sources,sources_sha):
    require(review['schema']=='paired_prevention_visual_review_v1'
        and review['review_manifest_sha256']==manifest_sha,'visual review manifest binding drift')
    require(review['new_rank_data_hidden'] is True and review['prior_rank_exposure_anchors']==[0,2],
            'rank-hidden review or prior exposure disclosure missing')
    require(isinstance(review['reviewer'],str) and review['reviewer'].strip()
        and isinstance(review['reviewed_at_utc'],str) and review['reviewed_at_utc'].strip(),
        'visual reviewer provenance missing')
    require(manifest['schema']=='paired_prevention_rank_hidden_media_v1'
        and manifest['sources_sha256']==sources_sha and manifest['rank_data_excluded'] is True,
            'rank-hidden media provenance drift')
    source_by={x['pair_index']:x for x in sources}
    require(len(manifest['pairs'])==12 and {x['pair_index'] for x in manifest['pairs']}==set(range(12)),
            'incomplete review media population')
    pairs={x['pair_index']:x for x in manifest['pairs']}
    files={x['path']:x for x in manifest['files']}
    require(len(manifest['files'])==len(files)==108
        and set(files)=={f'pair-{i:02d}/'+p for i in range(12) for p in media_paths()},
            'review media whitelist drift')
    require(len(review['pairs'])==12 and {x['pair_index'] for x in review['pairs']}==set(range(12)),
            'incomplete or duplicate visual labels')
    labels={}
    for p in review['pairs']:
        i=p['pair_index'];m=pairs[i];s=source_by[i]
        require(type(i) is int and p['run_id']==m['run_id']==Path(s['run_dir']).name
            and p['index_sha256']==m['index_sha256']==s['index_sha256'],
            'visual labels belong to another source')
        require(set(p['arms'])==set(ARMS),'missing or extra reviewed arm')
        for arm,row in p['arms'].items():
            validate_labels(row['labels'])
            prefix=f'pair-{i:02d}/';video=prefix+f'arms/{arm}/video.mp4'
            sheets=[prefix+f'comparison-seconds-{j:02d}-{j+9:02d}.png' for j in range(0,60,10)]
            require(row['video_sha256']==files[video]['sha256'] and row['evidence_panels']==sheets,
                    'visual evidence does not cover exact fixed 1Hz panels')
            require(isinstance(row['notes'],str) and row['notes'].strip(),'visual evidence notes missing')
            labels[i,arm]=row['labels']
    return labels


def restricted_onset_bounds(labels):
    """Bounds for min(first collapse onset,57); never impute actual failure at57."""
    e=visual_endpoint(labels)
    return [min(e['possible_onset_seconds'] if e['possible_onset_seconds'] is not None else 57,57),
            min(e['confirmed_onset_seconds'] if e['confirmed_onset_seconds'] is not None else 57,57)]


def timing_row(row):
    names=('generation_with_instrumentation_seconds','decode_seconds','encode_seconds')
    require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>0 for k in names),
            'missing or invalid instrumented latency')
    d={k:row[k] for k in names};d['warm_generation_decode_encode_seconds']=sum(d.values())
    return d


def paired_outcome(base_labels,labels,arm,base_timing,timing):
    require(arm!='draft_only','intervention arm required')
    b=visual_endpoint(base_labels);e=visual_endpoint(labels)
    bb=restricted_onset_bounds(base_labels);ee=restricted_onset_bounds(labels)
    # Use the intervention's same post-return window on BOTH trajectories.
    bp=post_return_counts(base_labels,arm);ep=post_return_counts(labels,arm)
    return {'confirmed_onsets_seconds':[b['confirmed_onset_seconds'],e['confirmed_onset_seconds']],
        'right_censored':[b['right_censored_no_confirmed_event'],e['right_censored_no_confirmed_event']],
        'confirmed_onset_difference_seconds':e['confirmed_onset_seconds']-b['confirmed_onset_seconds']
            if all(x['confirmed_onset_seconds'] is not None and
                x['possible_onset_seconds']==x['confirmed_onset_seconds'] for x in (b,e)) else None,
        'restricted_onset_gain_bounds_seconds':[ee[0]-bb[1],ee[1]-bb[0]],
        'restricted_onset_cap_seconds':57,
        'noncollapsed_sample_gain':e['noncollapsed_samples']-b['noncollapsed_samples'],
        'uncertain_samples':[b['uncertain_samples'],e['uncertain_samples']],
        'post_return':{'first_sample_seconds':ep['first_sample_seconds'],'samples':ep['samples'],
            'baseline':bp,'intervention':ep,
            'noncollapsed_sample_gain':ep['noncollapsed_samples']-bp['noncollapsed_samples'],
            'gain_bounds_samples':[ep['noncollapsed_samples']-bp['noncollapsed_samples']-bp['uncertain_samples'],
                ep['noncollapsed_samples']+ep['uncertain_samples']-bp['noncollapsed_samples']]},
        'latency_extra_seconds':{k:timing[k]-base_timing[k] for k in timing},
        'warm_latency_ratio':timing['warm_generation_decode_encode_seconds']/base_timing['warm_generation_decode_encode_seconds']}


def grouped_mean(rows,value):
    by={}
    for row in rows:
        v=value(row)
        if v is not None:by.setdefault(row['prompt_id'],[]).append(v)
    means={k:sum(v)/len(v) for k,v in by.items()}
    return {'prompt_means':means,'prompt_count':len(means),'trajectory_count':sum(map(len,by.values())),
        'mean_of_prompt_means':sum(means.values())/len(means) if means else None}


def predictor_comparisons(rows):
    result={}
    for population,selected in [('all12',rows),('new10_excluding_prior_anchors',[r for r in rows if r['pair_index'] not in (0,2)]),
        ('prior2_anchors',[r for r in rows if r['pair_index'] in (0,2)])]:
        primary=[dict(r,forecast=r['arms']['draft_only']['prediction']) for r in selected]
        comparison={}
        for long in ('sustained_fall_5s','drawdown_10s'):
            for single in FEATURES[:3]:
                def difference(r):
                    a=r['forecast']['common'][long]['roc_auc'];b=r['forecast']['common'][single]['roc_auc']
                    return a-b if a is not None and b is not None else None
                comparison[long+' minus '+single]=grouped_mean(primary,difference)
        result[population]={'trajectory_count':len(primary),'common_auc':{
            k:grouped_mean(primary,lambda r,k=k:r['forecast']['common'][k]['roc_auc']) for k in FEATURES},
            'paired_common_auc_differences':comparison}
    return result


def plot_pair(row,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(14,11),sharex=True)
    colors={'noncollapsed':'#2e7d32','severe_artifact':'#c62828','uncertain':'#888888'}
    for j,(arm,data) in enumerate(row['arms'].items()):
        times=[r['video_seconds'] for r in data['features']]
        axes[0].plot(times,[float('nan') if x is None else x for x in data['ranks']],label=arm)
        for ax,key in zip(axes[1:3],('sustained_fall_5s','drawdown_10s')):
            ax.plot(times,[float('nan') if r[key] is None else r[key] for r in data['features']],label=arm)
        for t,label in enumerate(data['labels']):axes[3].broken_barh([(t,.85)],(j-.3,.6),color=colors[label])
        for chunk in ARMS[arm]:
            from .single_target_recovery import video_time
            axes[3].broken_barh([(video_time(chunk),.5)],(j-.36,.72),
                facecolors='none',edgecolors='black',linewidths=1.1)
    axes[0].set_ylabel('Draft eRank\nlayer15 / t1000');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Causal 5s fall\nrank / second');axes[2].set_ylabel('Causal 10s\ndrawdown fraction')
    axes[3].set_yticks(range(3),list(ARMS),fontsize=8);axes[3].set_xlabel('Video time (seconds)')
    axes[3].set_title('Fixed 1Hz review: green noncollapsed / red severe / grey uncertain; bars are samples, not continuous labels\nBlack outlines: Target intervals in the corresponding intervention arm only',fontsize=9)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f"Pair {row['pair_index']:02d}: {row['prompt_id']} / seed {row['seed']} — gaps never interpolated")
    fig.tight_layout();fig.savefig(path.with_suffix('.png'),dpi=140);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)


def run(sources_path,sources_sha,bundle,bundle_index,review_path,review_sha,output):
    from .main_table_merge import _load_frozen_index
    from .stage_timing import _unsealed_directory
    _unsealed_directory(output)
    sources=source_list(sources_path,sources_sha)
    for p in [bundle]+[Path(s['run_dir']) for s in sources]:
        require(not output.resolve().is_relative_to(p.resolve()) and not p.resolve().is_relative_to(output.resolve()),
                'analysis/source output overlap')
    bf,bm=_load_frozen_index(bundle)
    require((bundle/'_SUCCESS.json').is_file() and bm['artifact_index_sha256']==bundle_index,'review bundle seal drift')
    bf.verify_all();manifest=frozen_json(bundle,bf,bm,'review-manifest.json')
    manifest_bytes=(bundle/'attempts'/bm['attempt_id']/'review-manifest.json').read_bytes()
    manifest_sha=hashlib.sha256(manifest_bytes).hexdigest()
    review,review_bytes=bound_json(review_path,review_sha)
    # Validate the ENTIRE population before reading a single new rank scalar.
    labels=validate_review(review,manifest,manifest_sha,sources,sources_sha)
    write_json(output/'accepted-review-binding.json',{'review_sha256':review_sha,'review_bundle_run':bundle.name,
        'review_bundle_index_sha256':bundle_index,'review_manifest_sha256':manifest_sha,'sources_sha256':sources_sha})
    payloads=[]
    for s in sources:
        i=s['pair_index'];root=Path(s['run_dir']);f,m=_load_frozen_index(root)
        require((root/'_SUCCESS.json').is_file() and m['git_sha']==GENERATION_SHA
            and m['artifact_index_sha256']==s['index_sha256'],'generation seal drift')
        f.verify_all();result=frozen_json(root,f,m,'result.json')
        traces={a:frozen_json(root,f,m,f'arms/{a}/trace.json') for a in ARMS}
        native=frozen_json(root,f,m,'native-control/native_draft/trace.json')['noise'];validate_pair(result,traces,native,i)
        media={x['path']:x for x in manifest['files']}
        for relative in media_paths():
            key=f"attempts/{m['attempt_id']}/{relative}"
            f.file(key,label='review/source media match')
            size,sha=f.entries[key];entry=media[f'pair-{i:02d}/'+relative]
            require((size,sha)==(entry['size'],entry['sha256']),'review media differs from sealed source')
        payloads.append((s,result,traces))
    rows=[]
    for s,result,traces in payloads:
        i=s['pair_index']
        from .intervention_erank_analysis import aligned
        aligned_data=aligned(traces)
        timings={x['arm']:timing_row(x) for x in result['arms']}
        row={'pair_index':i,'prompt_id':result['prompt_id'],'seed':result['seed'],'source':s,
            'initialization_seconds':result['initialization_seconds'],
            'native_control_validation_seconds':result['native_control_validation_seconds'],'arms':{},'paired':{}}
        for arm in ARMS:
            ranks=aligned_data['ranks'][arm];features=causal_features(ranks);visual=labels[i,arm]
            row['arms'][arm]={'labels':visual,'visual':visual_endpoint(visual),'timing':timings[arm],
                'ranks':ranks,'features':features,'prediction':predictor_summary(features,visual)}
            if arm!='draft_only':row['paired'][arm]=paired_outcome(labels[i,'draft_only'],visual,arm,timings['draft_only'],timings[arm])
        write_json(output/f'pair-{i:02d}.json',row);plot_pair(row,output/f'pair-{i:02d}');rows.append(row)
        print('COHORT_ANALYSIS_PAIR_COMPLETE',i,flush=True)
    quality={a:{'post_return_noncollapsed_gain':grouped_mean(rows,lambda r,a=a:r['paired'][a]['post_return']['noncollapsed_sample_gain']),
        'noncollapsed_gain':grouped_mean(rows,lambda r,a=a:r['paired'][a]['noncollapsed_sample_gain']),
        'post_return_gain_lower_bound':grouped_mean(rows,lambda r,a=a:r['paired'][a]['post_return']['gain_bounds_samples'][0]),
        'post_return_gain_upper_bound':grouped_mean(rows,lambda r,a=a:r['paired'][a]['post_return']['gain_bounds_samples'][1]),
        'restricted_onset_gain_lower_bound_seconds':grouped_mean(rows,lambda r,a=a:r['paired'][a]['restricted_onset_gain_bounds_seconds'][0]),
        'restricted_onset_gain_upper_bound_seconds':grouped_mean(rows,lambda r,a=a:r['paired'][a]['restricted_onset_gain_bounds_seconds'][1]),
        'warm_latency_extra_seconds':grouped_mean(rows,lambda r,a=a:r['paired'][a]['latency_extra_seconds']['warm_generation_decode_encode_seconds'])}
        for a in ARMS if a!='draft_only'}
    data={'schema':'paired_prevention_cohort_analysis_v1','pairs':rows,'quality_summary':quality,
        'primary_draft_predictor_comparison':predictor_comparisons(rows),
        'limitations':['Development cohort,6 prompts x2 seeds; no fitted trigger threshold or external validation.',
            'Prior rank exposure anchors pairs0,2 disclosed separately; arm identities visible during visual review.',
            'Fixed1Hz severe-artifact labels; ordinary motion not collapse, uncertain labels not imputed.',
            'Restricted onset bounds cap at57s and do not impute an actual failure time for censored videos.',
            'Inclusive instrumented generation latency; initialization/native validation separate; not pure inference.',
            'Prompt-grouped descriptive AUCs only; no frame-independent significance or classifier accuracy claim.']}
    write_json(output/'cohort-analysis.json',data)
    (output/'visual-review.json').write_bytes(review_bytes)
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--sources-sha256',required=True)
    p.add_argument('--review-bundle-run',type=Path,required=True);p.add_argument('--review-bundle-index',required=True)
    p.add_argument('--visual-review',type=Path,required=True);p.add_argument('--visual-review-sha256',required=True)
    a=p.parse_args()
    from .main_table_baseline_campaign import _run_dir
    run(a.sources,a.sources_sha256,a.review_bundle_run,a.review_bundle_index,a.visual_review,a.visual_review_sha256,_run_dir())

if __name__=='__main__':main()
