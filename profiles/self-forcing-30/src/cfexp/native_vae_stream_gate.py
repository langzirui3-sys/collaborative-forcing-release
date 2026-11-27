"""Full-prefix versus cached native VAE RGB gate on sealed accepted latents.

No generation or quality claim is made. The native model owns its own VAE
component, independently bound archive/wrapper/normalization and weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .accepted_stream import NativeStreamDecoder
from .correction_state import LatentHistory
from .main_table_baseline_campaign import _run_dir,_visible_devices,_gpu_compute_pids,_gpu_uuid,_write_new_json
from .main_table_merge import _load_frozen_index
from .main_table_ours_campaign import _assert_idle_pair
from .metadata import find_git_root,collect_git_metadata
from .native_ready_report import referenced_file
from .native_vae_component import selected_model,validate_flags,validate_identity
from .stage_timing import _unsealed_directory
from .streaming_vae_probe import to_pixels
from .video_only_decode import pixel_witness


def load_source(config, model):
    row=config['sources'][model];run=Path(row['run_dir'])
    if run.is_symlink():raise ValueError('native VAE source cannot be a symlink')
    if not (run/'_SUCCESS.json').is_file() or (run/'_FAILED.json').exists():
        raise ValueError('native VAE source must be successfully sealed')
    frozen,marker=_load_frozen_index(run);frozen.verify_all()
    if (marker['git_sha']!=row['git_sha'] or marker['artifact_index_sha256']!=row['artifact_index_sha256']
            or marker['run_id']!=run.name):raise ValueError('native VAE gate source seal differs')
    prefix='attempts/'+marker['attempt_id']+'/'
    result=json.loads(frozen.file(prefix+'result.json',label='sealed native cost result').read_text())
    if (result.get('schema')!='native_fixed_route_cost_probe_v1' or result.get('status')!='succeeded'
            or result.get('code_revision')!=row['git_sha']
            or result.get('model')!=('longlive_1.3b' if model=='longlive' else 'rolling_forcing_1.3b')
            or result.get('arm')!='fixed_burst' or result.get('duration',{}).get('chunk_count')!=60
            or result.get('target_chunks')!=[0,8,9,10,11]):
        raise ValueError('native VAE gate requires the frozen accepted 30s trajectory')
    src=SimpleNamespace(frozen=frozen,run=run,prefix=prefix)
    history_path=referenced_file(src,result['history'],'accepted source history')
    history=LatentHistory.from_dict(json.loads(history_path.read_text()))
    if history.next_chunk_index!=60:raise ValueError('accepted source history is incomplete')
    for segment in history.segments:
        referenced_file(src,dict(path=segment.latent_path,sha256=segment.latent_sha256,
                                nbytes=segment.latent_nbytes),'accepted source latent')
    return run/prefix,history,dict(run_id=marker['run_id'],attempt_id=marker['attempt_id'],
        artifact_index_sha256=marker['artifact_index_sha256'],git_sha=marker['git_sha'],
        history=result['history'],history_head_sha256=history.head_sha256)


def run(model, protocol, protocol_sha256):
    root=_unsealed_directory(_run_dir());repo=find_git_root(Path.cwd());devices=_visible_devices()
    if len(devices)!=2 or len(set(devices))!=2:raise ValueError('reserve the native pair for its RGB gate')
    _assert_idle_pair(*devices)
    if selected_model()!=model:raise ValueError('native VAE gate model differs from its manifest')
    validate_flags()
    raw=Path(protocol).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=protocol_sha256:raise ValueError('native render gate protocol changed')
    config=json.loads(raw)
    if config['schema']!='native_coupled_vae_stream_gate_v1':raise ValueError('wrong native render protocol')
    source,history,evidence=load_source(config,model)
    import torch
    from safetensors.torch import load_file
    uuids=[_gpu_uuid(x) for x in devices]
    def ownership():
        if any(not _gpu_compute_pids(u)<={os.getpid()} for u in uuids):
            raise ValueError('native VAE gate acquired an unrelated GPU process')
    cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all()
    decoder=NativeStreamDecoder(root,native_model=model,repository=repo)
    validate_identity(decoder.evidence['coupled_native_identity'],model)
    chunks=[]
    for segment in history.segments:
        values=load_file(str(source/segment.latent_path),device='cpu')
        if set(values)!={'latent'}:raise ValueError('source latent keys differ')
        value=values['latent']
        if tuple(value.shape)!=(1,3,16,60,104) or value.dtype!=torch.bfloat16:
            raise ValueError('source accepted latent geometry differs')
        chunks.append(value)
    ownership()
    try:
        decoder.clear()
        with torch.inference_mode():
            latent=torch.cat(chunks,dim=1).to('cuda:0').contiguous()
            reference=to_pixels(torch,decoder.model.decode(latent.permute(0,2,1,3,4),decoder.scale))
        decoder.clear();streamed=[]
        for index,value in enumerate(chunks):
            ownership();pixels=decoder.decode(value)
            if tuple(pixels.shape)!=(9 if index==0 else 12,3,480,832):
                raise ValueError('native cached VAE frame cursor differs')
            streamed.append(pixels)
        actual=torch.cat(streamed,dim=0)
        equal=torch.equal(reference,actual)
        after_gpu=torch.cuda.get_rng_state_all()
        unchanged=(torch.equal(cpu,torch.get_rng_state()) and len(gpu)==len(after_gpu) and
            all(torch.equal(a,b) for a,b in zip(gpu,after_gpu)))
        if not unchanged:raise ValueError('native VAE renderer changed the generation RNG tape')
        result=dict(schema='native_coupled_vae_rgb_gate_v1',status='succeeded' if equal else 'failed',
            code_revision=collect_git_metadata(repo)['sha'],model=model,source=evidence,
            decoder=decoder.evidence,protocol_sha256=protocol_sha256,chunk_count=60,
            rgb_equal=equal,rng_unchanged=unchanged,
            reference=pixel_witness(reference),streamed=pixel_witness(actual),
            quality_measured=False,formal_speedup_claim=False,
            claim_scope='one accepted 30s history, native full versus cached VAE RGB only')
        _write_new_json(root/'result.json',result)
        if not equal:raise ValueError('native full/cached VAE RGB differs; rendering candidate blocked')
        return result
    finally:decoder.clear()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=('longlive','rolling'))
    p.add_argument('--protocol',required=True)
    p.add_argument('--protocol-sha256',required=True)
    a=p.parse_args()
    try:run(a.model,a.protocol,a.protocol_sha256)
    finally:
        from .runtime_helpers import shutdown_owned_runtime_helpers
        shutdown_owned_runtime_helpers(label='native-vae-rgb-gate')

if __name__=='__main__':main()
