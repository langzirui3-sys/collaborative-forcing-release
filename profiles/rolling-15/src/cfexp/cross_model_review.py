"""Export all48 verified cross-model videos for a rank-hidden visual review.

This stage does not infer labels or select successful examples. It refuses a
partial population, a failed seal, altered indexed bytes, a missing rank sample
or a different generation protocol. Only explicit RGB media enter the ZIP.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from .cross_model_delay import ARMS,PAIRS,SEED,geometry,load_plan,require,schedule,sha,write_new
from .cross_model_outcomes import validate_trace,warm_latency
from .prevention_review_bundle import copy_bound

GENERATION={
    'longlive_krea':'b60fe73858bb339c0afc1240125192e0a09c7ce0',
    'lingbot_world_2':'f1cffce2a90fe8fb60c2130b87e6bd1eae32e6c3',
}


def source_list(path,expected):
    require(sha(path)==expected,'source-list digest drift')
    data=json.loads(Path(path).read_text())
    require(data['schema']=='cross_model_delay_sources_v1','source-list schema drift')
    rows=data['sources']
    require(len(rows)==12,'all12 model/prompt cases required')
    expected_cases={(p,i) for p in PAIRS for i in range(6)}
    require(all(type(x.get('prompt_index')) is int for x in rows),'invalid prompt index')
    require({(x.get('pair'),x['prompt_index']) for x in rows}==expected_cases,
            'missing, duplicate or unexpected model/prompt case')
    require(len({x['run_dir'] for x in rows})==12,'source run reused')
    for row in rows:
        require(Path(row['run_dir']).is_absolute() and re.fullmatch(r'[0-9a-f]{64}',row['index_sha256']) is not None,
                'invalid immutable source binding')
    return sorted(rows,key=lambda x:(PAIRS.index(x['pair']),x['prompt_index']))


def media_paths():
    return [f'arms/{a}/video.mp4' for a in ARMS]+[
        f'comparison-seconds-{i:02d}-{i+9:02d}.png' for i in range(0,60,10)]



def review_template(review,manifest_sha):
    """Checksum-bound empty labels; deliberately invalid until independently filled."""
    require(len(review['cases'])==12 and len(review['files'])==120,'complete review population required')
    files={x['path']:x for x in review['files']};cases=[]
    for case in review['cases']:
        prefix=case['media_prefix']
        arms={arm:{'video_sha256':files[prefix+f'arms/{arm}/video.mp4']['sha256'],
            'evidence_panels':[prefix+f'comparison-seconds-{i:02d}-{i+9:02d}.png' for i in range(0,60,10)],
            'labels':[None]*60,'notes':''} for arm in ARMS}
        cases.append({'pair':case['pair'],'prompt_index':case['prompt_index'],
            'run_id':case['run_id'],'index_sha256':case['index_sha256'],'arms':arms})
    return {'schema':'cross_model_delay_visual_review_v1','review_manifest_sha256':manifest_sha,
        'new_rank_data_hidden':True,'prior_rank_exposure_cases':[],'reviewer':'','reviewed_at_utc':'',
        'template_status':'unreviewed; empty labels must never be treated as normal', 'cases':cases}


def panel_archive_manifest(review,manifest_sha):
    files=[x for x in review['files'] if x['path'].endswith('.png')]
    expected={case['media_prefix']+f'comparison-seconds-{i:02d}-{i+9:02d}.png'
        for case in review['cases'] for i in range(0,60,10)}
    require(len(files)==72 and {x['path'] for x in files}==expected,'all72 fixed RGB panels required')
    require(review['video_count']==48 and review['generation_cases']==12,'full study verification required')
    return {'schema':'cross_model_delay_panel_archive_v1','review_manifest_sha256':manifest_sha,
        'rank_data_excluded':True,'video_files_in_this_archive':False,'verified_source_video_count':48,
        'full_video_archive':'rank-hidden-review.zip','included_files':files,
        'scope':'All48 videos remain independently verified; this companion contains their fixed 1Hz RGB panels only.'}


def validate_case(result,traces,*,pair,prompt_index,item,protocol_sha):
    require(result['schema']=='cross_model_delay_result_v1'
        and result['status']=='generation_complete_quality_pending','incomplete generation result')
    require(result['pair']==pair and result['prompt_index']==prompt_index
        and result['prompt']==item and result['seed']==SEED
        and result['protocol_sha256']==protocol_sha,'frozen case identity drift')
    require(result['all_actual_noise_checks_passed'] is True,'noise parity not complete')
    rows=result['arms']
    require(len(rows)==4 and {x['arm'] for x in rows}==set(ARMS),'all4 unique arms required')
    require(result['execution_order'][0]=='draft_only' and len(result['execution_order'])==4
        and set(result['execution_order'])==set(ARMS),'native reference order drift')
    g=geometry(pair);bound={x['arm']:x for x in rows}
    for arm in ARMS:
        chunks=validate_trace(pair,arm,traces[arm]);row=bound[arm]
        require(row['chunk_count']==g['chunk_count'] and row['video']['frames']==g['rgb_frames']
            and row['video']['fps']==g['fps'],'video geometry drift')
        require(row['warm_pipeline_plus_encode_seconds']==warm_latency(row),'recorded warm latency drift')
        if arm.startswith('target_three'):
            first=min(schedule(pair,arm))
            require(all(x['native_prefix_exact'] is True for x in chunks[:first])
                and [x['latent_sha256'] for x in chunks[:first]]==[
                    x['latent_sha256'] for x in traces['draft_only']['chunks'][:first]],'native prefix parity missing')
            require(all(x['native_clean_cache_update'] is True for x in chunks), 'native clean commit missing')
            require(traces[arm]['renoise']==traces['draft_only']['renoise'],'actual Draft RNG parity drift')
        if pair=='lingbot_world_2':
            require(traces[arm]['renoise']==traces['draft_only']['renoise'],'actual native RNG parity drift')
            require(traces[arm]['initial_chunk_noise_sha256']==traces['draft_only']['initial_chunk_noise_sha256'],
                    'actual initial noise parity drift')
            require([x['conditioning_sha256'] for x in chunks]==[
                x['conditioning_sha256'] for x in traces['draft_only']['chunks']], 'native conditioning parity drift')
    return bound


def run(sources_path,sources_sha,protocol_path,protocol_sha,output):
    from .main_table_merge import _load_frozen_index
    from .stage_timing import _unsealed_directory
    from .cross_model_delay import audit_video
    _unsealed_directory(output)
    sources=source_list(sources_path,sources_sha)
    review={'schema':'cross_model_delay_rank_hidden_media_v1','sources_sha256':sources_sha,
        'protocol_sha256':protocol_sha,'rank_data_excluded':True,'arm_identity_hidden':False,
        'new_rank_data_viewed_by_reviewer':False,'visual_sampling_seconds':list(range(60)),
        'generation_cases':12,'video_count':48,'cases':[],'files':[]}
    with (zipfile.ZipFile(output/'rank-hidden-review.zip','x',compression=zipfile.ZIP_STORED,allowZip64=True) as z,
          zipfile.ZipFile(output/'rank-hidden-panels.zip','x',compression=zipfile.ZIP_STORED,allowZip64=True) as panels):
        for binding in sources:
            pair=binding['pair'];i=binding['prompt_index']
            _,item=load_plan(protocol_path,protocol_sha,pair,i)
            root=Path(binding['run_dir']).resolve(strict=True)
            require(not output.resolve().is_relative_to(root) and not root.is_relative_to(output.resolve()),
                    'source and review output overlap')
            frozen,marker=_load_frozen_index(root)
            require(marker['git_sha']==GENERATION[pair] and marker['artifact_index_sha256']==binding['index_sha256'],
                    'generation commit or sealed index drift')
            frozen.verify_all();attempt=root/'attempts'/marker['attempt_id']
            def read(relative):
                p=attempt/relative;frozen.file(p.relative_to(root).as_posix(),label='cross-model review evidence')
                return json.loads(p.read_text())
            result=read('result.json');traces={a:read(f'arms/{a}/trace.json') for a in ARMS}
            bound=validate_case(result,traces,pair=pair,prompt_index=i,item=item,protocol_sha=protocol_sha)
            if pair=='longlive_krea':
                for arm in ARMS[2:]:
                    for chunk in schedule(pair,arm):
                        candidate=read(f'target-evidence/{arm}-{chunk:03d}.json')['response']['result']
                        native=read(f'target-evidence/target_only-{chunk:03d}.json')['response']['result']
                        require(candidate['renoise_tensor_sha256']==native['renoise_tensor_sha256']
                            and len(candidate['renoise_tensor_sha256'])==3,'actual Target stochastic input drift')
            prefix=f'{pair}/prompt-{i:02d}/'
            review['cases'].append({'pair':pair,'prompt_index':i,'prompt':item,'seed':SEED,
                'run_id':root.name,'attempt_id':marker['attempt_id'],'generation_sha':marker['git_sha'],
                'index_sha256':binding['index_sha256'],'media_prefix':prefix})
            for relative in media_paths():
                path=attempt/relative;key=path.relative_to(root).as_posix()
                frozen.file(key,label='cross-model rank-hidden media')
                size,digest=frozen.entries[key];audit=None
                if relative.endswith('.mp4'):
                    g=geometry(pair);audit=audit_video(path,g['rgb_frames'],g['fps'])
                    arm=Path(relative).parent.name
                    require(audit==bound[arm]['video'],'independent video audit differs')
                arc=prefix+relative;copy_bound(z,path,arc,size,digest)
                if relative.endswith('.png'):copy_bound(panels,path,arc,size,digest)
                review['files'].append({'path':arc,'size':size,'sha256':digest,'video_audit':audit})
            print('CROSS_MODEL_REVIEW_VERIFIED',pair,i,'all4 videos and native histories verified',flush=True)
        require(len(review['files'])==120,'review media population drift')
        payload=json.dumps(review,indent=2,allow_nan=False)+'\n'
        manifest_sha=hashlib.sha256(payload.encode()).hexdigest()
        template=review_template(review,manifest_sha)
        panel_manifest=panel_archive_manifest(review,manifest_sha)
        z.writestr('review-manifest.json',payload)
        z.writestr('visual-review-template.json',json.dumps(template,indent=2,allow_nan=False)+'\n')
        panels.writestr('review-manifest.json',payload)
        panels.writestr('panel-archive-manifest.json',json.dumps(panel_manifest,indent=2,allow_nan=False)+'\n')
        panels.writestr('visual-review-template.json',json.dumps(template,indent=2,allow_nan=False)+'\n')
    require(sha(sources_path)==sources_sha,'source list changed during review export')
    write_new(output/'review-manifest.json',review)
    require(sha(output/'review-manifest.json')==manifest_sha,'review manifest serialization drift')
    write_new(output/'visual-review-template.json',template)
    write_new(output/'panel-archive-manifest.json',panel_manifest)
    with (output/'sources.json').open('xb') as f:f.write(Path(sources_path).read_bytes())
    return review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--sources-sha256',required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    a=p.parse_args()
    from .main_table_baseline_campaign import _run_dir
    run(a.sources,a.sources_sha256,a.protocol,a.protocol_sha256,_run_dir())

if __name__=='__main__':main()
