"""Read sealed LingBot latents; independently decode without regenerating them.

The original KV tensors were not saved. Window layout below is a source-derived
simulation, explicitly separate from the runtime's checked monotone cursors.
No observation in this probe changes the frozen visual labels or source seal.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
from pathlib import Path
import time

from .cross_model_delay import ARMS, require, sha, write_new
from .main_table_merge import _load_frozen_index


def cache_layout(chunks=60, width=4, capacity=18, sink=6):
    require(0 <= sink < capacity and width <= capacity-sink, 'invalid cache geometry')
    frames=[]; rows=[]
    for chunk in range(chunks):
        incoming=list(range(chunk*width,(chunk+1)*width))
        overflow=max(0,len(frames)+width-capacity)
        if overflow:frames=frames[:sink]+frames[sink+overflow:]
        frames+=incoming
        rows.append({'chunk':chunk,'global_end_latent_frame':(chunk+1)*width,
                     'local_end_latent_frame':len(frames),'evicted_frames':overflow,
                     'retained_absolute_latent_frames':list(frames)})
    return rows


def tensor_comparison(left,right,torch):
    a=left.detach().float().reshape(-1); b=right.detach().float().reshape(-1)
    require(a.shape==b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
            'comparison requires aligned finite tensors')
    return {'rmse':float((a-b).square().mean().sqrt()),
            'cosine':float(torch.nn.functional.cosine_similarity(a[None],b[None])[0]),
            'exact':bool(torch.equal(left,right))}


def latent_geometry(shape):
    require(len(shape)==4 and tuple(shape[:2])==(16,4)
            and all(type(x) is int and x>0 and x%2==0 for x in shape[2:]),
            'invalid native latent geometry')
    return (3,957,shape[2]*8,shape[3]*8)


def load_trace_bound_latent(base, arm, chunk, record, torch, *, longlive=False):
    """Recover excluded local tensors through hashes in an indexed source trace.

    These files are not themselves indexed/published. Never present them as such.
    LongLive's legacy digest rounds to BF16, so require BF16 storage to bind all bits.
    """
    from safetensors.torch import load_file
    from .cross_model_delay import tensor_digest
    from .incremental_draft import latent_digest
    require(record['chunk']==chunk, 'source trace chunk drift')
    path=base/'arms'/arm/f'{chunk:03d}.safetensors'
    require(path.resolve().is_relative_to(base.resolve()) and path.is_file(),
            'local source latent missing or escapes source attempt')
    x=load_file(str(path))['latent']
    require(not longlive or x.dtype==torch.bfloat16, 'legacy digest requires BF16 storage')
    digest=latent_digest(x,torch) if longlive else tensor_digest(x,torch)
    require(digest==record['latent_sha256'], 'local source latent differs from sealed trace')
    return x


def run(source,index_sha,output,model_root):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from PIL import Image
    from torchvision.io import write_video
    from .stage_timing import _unsealed_directory
    torch.set_num_threads(4)
    _unsealed_directory(output)
    frozen,seal=_load_frozen_index(source)
    require(seal['artifact_index_sha256']==index_sha
            and seal['git_sha']=='f1cffce2a90fe8fb60c2130b87e6bd1eae32e6c3', 'source seal drift')
    base=source/'attempts'/seal['attempt_id']
    def bound(relative):
        return frozen.file((base/relative).relative_to(source).as_posix(),label='reset-probe input')
    result=json.loads(bound('result.json').read_text())
    require(result['pair']=='lingbot_world_2' and result['prompt_index']==3,'wrong source case')
    from .lingbot_smoke import verified_file
    from .longlive_adapter import _safe_extract_tar
    repository=Path(__file__).resolve().parents[2]
    source_record=json.loads((repository/'experiments/lingbot-source-download-v1.json').read_text())['files'][0]
    archive=verified_file(model_root,source_record)
    # Runtime materializations were intentionally excluded from source seals.
    # Re-extract the independently pinned official archive, rather than import
    # unindexed files left in the original run's temporary directories.
    _safe_extract_tar(archive,output/'tmp/native-source')
    revision='lingbot-world-v2-1895d300d8ac936401689b26389f51cbd36530eb'
    official=output/'tmp/native-source'/revision
    vae_path=official/'wan/modules/vae2_1.py'
    model_path=official/'wan/modules/model_causal.py'
    native_path=official/'wan/image2video.py'
    records=json.loads((repository/'experiments/four-pair-model-downloads-v1.json').read_text())['files']
    weight_record=next(x for x in records if x['repository']=='robbyant/lingbot-world-v2-14b-causal-fast'
                       and x['filename']=='Wan2.1_VAE.pth')
    weights=verified_file(model_root,weight_record)
    input_receipt=json.loads(bound('input-binding.json').read_text())
    write_new(output/'source-binding.json',{'source_run':str(source),'source_index':index_sha,
        'generation_sha':seal['git_sha'],'vae_source_sha256':sha(vae_path),
        'model_source_sha256':sha(model_path),'native_source_sha256':sha(native_path),
        'vae_weights_sha256':sha(weights),'official_source_archive_sha256':sha(archive),
        'input_binding':input_receipt,
        'local_latent_binding':'tensor content digest in indexed trace; safetensors excluded from original index',
        'no_dit_regeneration':True,'original_kv_tensors_available':False})
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one leased CUDA device required')
    spec=importlib.util.spec_from_file_location('cf_reset_bound_vae',vae_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    vae=module.Wan2_1_VAE(vae_pth=str(weights),device=torch.device('cuda:0'))
    layout=cache_layout();write_new(output/'source-derived-cache-layout.json',{
        'kind':'simulation, not a dump of original KV tensors','capacity':18,'sink':6,
        'native_clean_cursor_checked_for_every_original_chunk':True,'chunks':layout})
    all_rows=[];reports=[];draft_reference=None
    for arm in ARMS:
        directory=output/arm;directory.mkdir()
        trace=json.loads(bound(f'arms/{arm}/trace.json').read_text())
        require(len(trace['chunks'])==60,'incomplete saved trajectory')
        chunks=[];expected_shape=None
        for i in range(60):
            x=load_trace_bound_latent(base,arm,i,trace['chunks'][i],torch)
            latent_geometry(tuple(x.shape))
            if expected_shape is None:expected_shape=tuple(x.shape)
            require(tuple(x.shape)==expected_shape and bool(torch.isfinite(x).all()),'latent shape drift')
            chunks.append(x)
        latent=torch.cat(chunks,dim=1);del chunks
        if arm=='draft_only':draft_reference=latent
        for frame in range(240):
            x=latent[:,frame]; row={'arm':arm,'latent_frame':frame,'video_end_seconds':frame/4,
                'mean':float(x.float().mean()),'std':float(x.float().std()),
                **{'first_'+k:v for k,v in tensor_comparison(x,latent[:,0],torch).items()},
                **{'native_draft_'+k:v for k,v in tensor_comparison(x,draft_reference[:,frame],torch).items()}}
            if frame:row.update({'previous_'+k:v for k,v in tensor_comparison(x,latent[:,frame-1],torch).items()})
            all_rows.append(row)
        start=time.perf_counter()
        with torch.inference_mode():video=vae.decode([latent.to('cuda:0')])[0]
        require(tuple(video.shape)==latent_geometry(expected_shape),'fresh decode geometry drift')
        frames=video.add(1).mul(127.5).round().clamp(0,255).byte().permute(1,2,3,0).cpu().contiguous()
        del video;gc.collect();torch.cuda.empty_cache()
        report={'arm':arm,'full_decode_seconds':time.perf_counter()-start,'original_png_comparison':[],
            'fresh_context_comparison':[],'tail_decode_start_latent_frame':160,
            'tail_alignment_rgb_offset':640,'note':'crop first latent produces one frame; offset=4*160, not4*160-3'}
        for second in range(60):
            current=Image.fromarray(frames[second*16].numpy()).resize((416,240))
            original=np.asarray(Image.open(bound(f'arms/{arm}/second-{second:02d}.png')).convert('RGB')).astype(np.float32)
            rgb=np.asarray(current).astype(np.float32)
            report['original_png_comparison'].append({'second':second,'mae':float(np.abs(rgb-original).mean()),
                'max_abs':float(np.abs(rgb-original).max()),'exact':bool(np.array_equal(rgb,original))})
            if 45<=second<=56:current.save(directory/f'fresh-full-{second:02d}.png')
        write_video(str(directory/'fresh-full-video.mp4'),frames,fps=16,video_codec='libx264',options={'crf':'18'})
        # A second independent VAE session starts at40s, with no prior decoder
        # feature history. By48..55s it has32..60 warmup latent frames.
        with torch.inference_mode():tail=vae.decode([latent[:,160:].to('cuda:0')])[0]
        tail_frames=tail.add(1).mul(127.5).round().clamp(0,255).byte().permute(1,2,3,0).cpu().contiguous()
        del tail
        for second in range(45,57):
            full=frames[second*16].float();crop=tail_frames[second*16-640].float()
            report['fresh_context_comparison'].append({'second':second,
                'mae':float((full-crop).abs().mean()),'max_abs':int((full-crop).abs().max()),
                'exact':bool(torch.equal(full,crop))})
            Image.fromarray(crop.byte().numpy()).resize((416,240)).save(directory/f'fresh-tail-{second:02d}.png')
        write_new(directory/'decode-audit.json',report);reports.append(report)
        print('RESET_PROBE_ARM_COMPLETE',arm,'all60 original PNGs compared',flush=True)
        del latent,frames,tail_frames;gc.collect();torch.cuda.empty_cache()
    names=['arm','latent_frame','video_end_seconds','mean','std','first_rmse','first_cosine','first_exact',
           'native_draft_rmse','native_draft_cosine','native_draft_exact',
           'previous_rmse','previous_cosine','previous_exact']
    with (output/'latent-frame-statistics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(all_rows)
    write_new(output/'decode-summary.json',{'arms':reports,'regeneration':False,
        'limitations':['source-derived cache layout is not original tensor contents',
            'latent distances are diagnostics, not semantic quality scores',
            'decoder reproduction alone does not establish which generator component caused a reset']})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--source-index-sha256',required=True);a=p.parse_args()
    from .main_table_baseline_campaign import _run_dir,_model_root
    run(a.source,a.source_index_sha256,_run_dir(),_model_root())


if __name__=='__main__':main()
