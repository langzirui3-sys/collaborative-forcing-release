"""Verify all12 paired sources and export rank-free media for independent review."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from .motion_video_audit import require
from .single_target_recovery import write_json, validate_noise_calls
from .paired_prevention import ARMS, PROMPT_IDS, SEEDS

GENERATION_SHA='029632be07e82dca5e347905283af2bbb2b34ebc'
PROTOCOL_SHA='e107d5ed592a8b5b2c801a2b3f04b97a35efe594159d3ecb04b7491d8e455052'


def source_list(path, expected_sha256):
    payload=path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected_sha256, 'review source-list digest drift')
    d=json.loads(payload)
    require(d['schema']=='paired_prevention_review_sources_v1', 'source-list schema drift')
    rows=d['sources']
    require(len(rows)==12 and {x['pair_index'] for x in rows}==set(range(12)), 'incomplete or duplicate review population')
    require(all(type(x['pair_index']) is int and isinstance(x['run_dir'],str)
        and Path(x['run_dir']).is_absolute() and isinstance(x['index_sha256'],str)
        and len(x['index_sha256'])==64 for x in rows), 'invalid review source binding')
    return sorted(rows,key=lambda x:x['pair_index'])


def media_paths():
    # Explicit whitelist: never export traces, rank arrays, noise, or weights.
    return [f'arms/{arm}/video.mp4' for arm in ARMS]+[
        f'comparison-seconds-{i:02d}-{i+9:02d}.png' for i in range(0,60,10)]


def validate_pair(result, traces, native_noise, pair_index):
    require(result['pair_index']==pair_index and result['protocol_sha256']==PROTOCOL_SHA,
            'pair identity or protocol drift')
    require(result['prompt_id']==PROMPT_IDS[pair_index//2] and result['seed']==SEEDS[pair_index%2],
            'prompt or seed drift')
    require(result['comparison_row_order']==list(ARMS) and len(result['arms'])==3,
            'review arm population drift')
    require({r['arm'] for r in result['arms']}==set(ARMS), 'duplicate review arm')
    require([(r['chunk'],r['step']) for r in native_noise]==[(i,j) for i in range(120) for j in range(3)],
            'native noise coverage drift')
    for arm,schedule in ARMS.items():
        t=traces[arm];rows=t['chunks'];first=min(schedule) if schedule else 120
        require(t['intervention_chunks']==list(schedule) and len(t['target_calls'])==len(schedule),
                'forced Target budget drift')
        require(t['initial_noise_sha256']==result['initial_noise_sha256'], 'shared initial noise drift')
        require(len(rows)==120 and [r['chunk'] for r in rows]==list(range(120)), 'incomplete arm')
        require(all(r['producer']==('target' if r['chunk'] in schedule else 'draft') for r in rows),
                'producer schedule drift')
        require(all(r['native_reference']['exact'] for r in rows[:first]), 'native prefix parity failed')
        validate_noise_calls(t['noise'],native_noise,tuple(schedule))


def verify_video(path):
    import av
    from fractions import Fraction
    count=0
    with av.open(str(path)) as container:
        require(len(container.streams.video)==1, 'unexpected video stream count')
        stream=container.streams.video[0]
        require(stream.average_rate==Fraction(24,1), 'video frame rate drift')
        for frame in container.decode(stream):
            require((frame.width,frame.height)==(832,480), 'video geometry drift')
            require(frame.pts is not None and frame.time_base is not None
                and abs(float(frame.pts*frame.time_base)-count/24)<1e-6, 'video presentation timeline drift')
            count+=1
    require(count==1437, 'independent video frame count drift')
    return {'decoded_frames':count,'fps':24,'width':832,'height':480,'duration_seconds':count/24}


def copy_bound(zipper, source, archive_path, expected_size, expected_sha256):
    h=hashlib.sha256();size=0
    with source.open('rb') as reader, zipper.open(archive_path,'w') as writer:
        while True:
            block=reader.read(1024*1024)
            if not block:break
            h.update(block);size+=len(block);writer.write(block)
    require(size==expected_size and h.hexdigest()==expected_sha256, 'media changed while packaging')


def run(sources_path, sources_sha256, output):
    from .main_table_merge import _load_frozen_index
    from .stage_timing import _unsealed_directory
    _unsealed_directory(output)
    sources=source_list(sources_path,sources_sha256)
    require(not (output/'rank-hidden-review.zip').exists(), 'review bundle already started')
    review={'schema':'paired_prevention_rank_hidden_media_v1','sources_sha256':sources_sha256,
        'rank_data_excluded':True,'arm_identity_hidden':False,'review_status':'pending',
        'prior_rank_exposure_anchors':[0,2], 'comparison_row_order':list(ARMS),
        'visual_sampling_seconds':list(range(60)), 'pairs':[],'files':[]}
    with zipfile.ZipFile(output/'rank-hidden-review.zip','x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for binding in sources:
            root=Path(binding['run_dir']).resolve(strict=True)
            require(not output.resolve().is_relative_to(root) and not root.is_relative_to(output.resolve()),
                    'source and review output overlap')
            frozen,marker=_load_frozen_index(root)
            require(marker['git_sha']==GENERATION_SHA and marker['artifact_index_sha256']==binding['index_sha256'],
                    'generation code or source index drift')
            frozen.verify_all()
            attempt=root/'attempts'/marker['attempt_id']
            def read(relative):
                p=attempt/relative;frozen.file(p.relative_to(root).as_posix(),label='paired review source')
                return json.loads(p.read_text())
            result=read('result.json');traces={arm:read('arms/'+arm+'/trace.json') for arm in ARMS}
            native=read('native-control/native_draft/trace.json')['noise']
            i=binding['pair_index'];validate_pair(result,traces,native,i)
            review['pairs'].append({'pair_index':i,'run_id':root.name,'index_sha256':marker['artifact_index_sha256'],
                'prompt_id':result['prompt_id'],'prompt':result['prompt'],'seed':result['seed']})
            for relative in media_paths():
                p=attempt/relative;key=p.relative_to(root).as_posix()
                frozen.file(key,label='rank-hidden media')
                size,sha=frozen.entries[key]
                arc=f'pair-{i:02d}/'+relative
                video_audit=verify_video(p) if relative.endswith('.mp4') else None
                copy_bound(z,p,arc,size,sha)
                review['files'].append({'path':arc,'size':size,'sha256':sha,'video_audit':video_audit})
            print('REVIEW_PAIR_VERIFIED',i,'all3 videos decoded; fixed noise and prefix checks passed',flush=True)
        z.writestr('review-manifest.json',json.dumps(review,indent=2,allow_nan=False)+'\n')
    write_json(output/'review-manifest.json',review)
    payload=sources_path.read_bytes()
    require(hashlib.sha256(payload).hexdigest()==sources_sha256, 'source-list changed while packaging')
    (output/'sources.json').write_bytes(payload)
    return review


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--sources-sha256',required=True)
    a=p.parse_args()
    from .main_table_baseline_campaign import _run_dir
    run(a.sources,a.sources_sha256,_run_dir())

if __name__=='__main__':main()
