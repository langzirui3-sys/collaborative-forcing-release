"""Opt-in native attention metadata fast path gated by sealed CUDA evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

from .full_length_attention import ATTENTION_SHA,FLASH_FUNCTION_SHA,FullLengthAttention
from .full_attention_probe import SHAPES
from .h800_native_probe_archive import ARCHIVE_SHA as H800_ARCHIVE_SHA
from .rope_coefficients import ARCHIVE_SHA,MODULE_SHA
from .rope_worker import NativeProbeIdentity, require

ENV='CFEXP_NATIVE_FULL_LENGTH_ATTENTION'
PROBE_CODE='154275222e61ffcc341db0a75c4ab55373661094'
PROBE_SPEC='fb420a2b9e14f85b31ffd4de0abb357d011543540e2cedb87596645b19e8a683'
HELPER_SHA='93e031d8da9c7dfe7ea0554a571d1b90e7562b41760ff08ee42c1d8b2c1a4dc7'
PREFIX='native-full-attention-gpu-probe-ntu-v1-'
MODEL_SHA='f15214485a5297e6f9e79684bbad80acbcdc88c3272596b256a739a734529449'


H800_PROBE = NativeProbeIdentity(
    'native-full-attention-gpu-probe-h800-v1-',
    '72beb42e544426249f1895be5093fa5c332762fd',
    '490eb70337a408793fb73be72d4be2b917c17d434b55b2c1e77b7f4e2e07f595',
    H800_ARCHIVE_SHA,
)


def probe_identity(device_name):
    if device_name == 'NVIDIA H800':
        return H800_PROBE
    return NativeProbeIdentity(PREFIX, PROBE_CODE, PROBE_SPEC, ARCHIVE_SHA)


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_probe_manifest(resolved, *, device_name=None):
    from .manifest import manifest_hash
    proof=probe_identity(device_name)
    manifest=copy.deepcopy(resolved['manifest'])
    require(manifest['spec']['code']['revision'] in ('HEAD',proof.code),'full attention probe code differs')
    manifest['spec']['code']['revision']='HEAD'
    remote=manifest['spec']['storage']['remote'];require(isinstance(remote,str) and bool(remote),'missing probe destination')
    manifest['spec']['storage']['remote']='${CFEXP_ARTIFACT_REMOTE}'
    require(manifest_hash(manifest)==proof.spec,'full attention probe specification differs')


def validate_probe(data,*,torch_version,device_name):
    require(data['schema']=='native_full_attention_wrapper_probe_v1'
        and data['torch_version']==torch_version and data['flash_attn_version']=='2.8.3'
        and data['native_archive_sha256']==probe_identity(device_name).archive_sha and data['native_module_sha256']==ATTENTION_SHA
        and data['native_function_sha256']==FLASH_FUNCTION_SHA and data['helper_sha256']==HELPER_SHA,
        'full attention probe implementation/runtime differs')
    require(data['generation_worker_integration'] is data['video_quality_evaluated']
        is data['end_to_end_speedup_claim'] is False,'full attention probe scope differs')
    rows=data['rows']
    require(len(rows)==12 and {(r['device_index'],tuple(r['shape'])) for r in rows}
        =={(i,(1,q,k,12,128)) for i in (0,1) for q,k in SHAPES},'full attention probe coverage differs')
    for row in rows:
        require(row['device_name']==device_name and row['dtype']=='bfloat16'
            and row['bitwise_equal'] is row['inputs_unchanged'] is True
            and row['repeat_count']==20 and row['calls_per_sample']==5
            and row['execution_counts']=={'calls':100,'dense_calls':100,'fallbacks':0},
            'full attention CUDA parity/inputs/execution differ')
        require(len(row['input_sha256'])==3
            and all(isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h)
                    for h in [*row['input_sha256'],row['output_sha256']]),'full attention input/output binding missing')
        for key in ('native','dense'):
            values=row['timing_seconds_per_call'][key]
            require(len(values)==20 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values)
                and statistics.median(values)==row['median_'+key+'_seconds'],'invalid full attention timing samples')
        ratio=row['median_native_seconds']/row['median_dense_seconds']
        require(ratio==row['wrapper_speedup_ratio_of_medians'] and ratio>=1.,
                'native attention wrapper did not improve every probe case')
    require(math.fsum(r['median_native_seconds'] for r in rows)/
            math.fsum(r['median_dense_seconds'] for r in rows)>=1.02,
            'full attention wrapper aggregate probe gain below two percent')


def probe_gate(run_dir,*,torch_version,device_name):
    from .rope_worker import probe_gate as rope_gate
    from .main_table_merge import _load_frozen_index
    proof=probe_identity(device_name)
    # Reuse the existing active-attempt and sealed native-RoPE checks. This path
    # cannot silently replace the required native cache/frequency configuration.
    rope_gate(run_dir,torch_version=torch_version,device_name=device_name)
    root=next(p for p in [run_dir,*run_dir.parents] if (p/'resolved-manifest.json').is_file())
    sources=[]
    for source in sorted(root.parent.glob(proof.prefix+'*')):
        require(not source.is_symlink(),'attention probe root is a symlink')
        require(not (source/'_FAILED.json').exists(),'full attention probe failure requires a new version')
        if not (source/'_SUCCESS.json').exists():continue
        frozen,marker=_load_frozen_index(source);frozen.verify_all()
        require(marker['git_sha']==proof.code,'full attention probe source revision differs')
        validate_probe_manifest(json.loads(frozen.file('resolved-manifest.json',label='attention probe manifest').read_text()),
                                device_name=device_name)
        relative='attempts/'+marker['attempt_id']+'/result.json'
        data=json.loads(frozen.file(relative,label='attention probe result').read_text())
        validate_probe(data,torch_version=torch_version,device_name=device_name)
        sources.append({'run_dir':str(source),'marker':marker,'result_binding':frozen.entries[relative]})
    require(bool(sources),'no successful sealed full native attention probe exists')
    evidence={'schema':'native_full_attention_probe_gate_v1','probe_code':proof.code,
        'probe_spec_hash':proof.spec,'helper_sha256':HELPER_SHA,'torch_version':torch_version,
        'device_name':device_name,'sources':sources}
    return evidence,object_sha(evidence)


def validate_execution(value):
    require(isinstance(value,dict) and value.get('schema')=='native_full_attention_execution_v1'
        and value.get('helper_sha256')==HELPER_SHA
        and value.get('native_attention_parameters_and_kv_slices_preserved') is True
        and isinstance(value.get('probe_gate_sha256'),str)
        and re.fullmatch('[0-9a-f]{64}',value['probe_gate_sha256']) is not None,
        'missing full attention execution binding')
    counts=value.get('counters',{})
    require(all(type(counts.get(k)) is int and counts[k]>=0 for k in ('calls','dense_calls','fallbacks'))
        and counts['calls']==counts['dense_calls']+counts['fallbacks'] and counts['dense_calls']>0,
        'native full-length attention did not actually execute')


def native_bindings(backend):
    expected={'wan.modules.attention':ATTENTION_SHA,'wan.modules.model':MODEL_SHA,
              'wan.modules.causal_model':MODULE_SHA}
    modules={}
    for name,sha in expected.items():
        module=sys.modules.get(name);require(module is not None,'missing native attention module '+name)
        path=Path(module.__file__).resolve(strict=True)
        require(path.is_relative_to(backend._source_root.resolve(strict=True))
            and hashlib.sha256(path.read_bytes()).hexdigest()==sha,'loaded native attention source differs')
        modules[name]=module
    attention=modules['wan.modules.attention'];original=attention.flash_attention
    require(getattr(original,'__module__',None)=='wan.modules.attention'
        and getattr(original,'__name__',None)=='flash_attention'
        and modules['wan.modules.model'].flash_attention is original
        and modules['wan.modules.causal_model'].attention is attention.attention
        and attention.attention.__globals__ is attention.__dict__,
        'native attention aliases were already replaced')
    require(attention.FLASH_ATTN_2_AVAILABLE is True and attention.FLASH_ATTN_3_AVAILABLE is False
        and attention.flash_attn.__version__=='2.8.3','native attention backend selection differs')
    return [attention,modules['wan.modules.model']],original,attention.flash_attn.flash_attn_func


def install(backend,run_dir):
    from .self_forcing_frame_adapter import OfficialSelfForcingBackend
    from .main_table_baseline_campaign import _write_new_json
    from . import full_length_attention
    from .sdvg_adapter import draft_generation_backend

    backend = draft_generation_backend(backend)
    require(type(backend) is OfficialSelfForcingBackend
        and getattr(backend,'_rope_coefficients_installed',False),
        'full native attention requires explicit CPU-bound native KV and probed RoPE')
    require(not getattr(backend,'_full_length_attention_installed',False),'full native attention already installed')
    require(hashlib.sha256(Path(full_length_attention.__file__).read_bytes()).hexdigest()==HELPER_SHA,
            'full attention helper differs from the CUDA-probed implementation')
    modules,original,dense=native_bindings(backend)
    gate,gate_sha=probe_gate(run_dir,torch_version=str(backend.torch.__version__),
                           device_name=backend.torch.cuda.get_device_name(backend.device))
    _write_new_json(run_dir/'provenance/native-full-attention-probe-gate.json',dict(gate,sha256=gate_sha))
    fast=FullLengthAttention(original,dense,backend.torch);generate=backend.generate_chunk
    def generate_chunk(**kwargs):
        require(all(m.flash_attention is fast for m in modules),'native attention alias ownership changed')
        before=fast.evidence();result=generate(**kwargs);after=fast.evidence()
        evidence={'schema':'native_full_attention_execution_v1','probe_gate_sha256':gate_sha,
            'helper_sha256':HELPER_SHA,'native_attention_parameters_and_kv_slices_preserved':True,
            'counters':{k:after[k]-before[k] for k in before}}
        validate_execution(evidence)
        return dict(result,full_attention_execution_evidence=evidence)
    def restore():
        require(all(m.flash_attention is fast for m in modules),'native attention aliases changed during execution')
        for module in modules:module.flash_attention=original
        backend.generate_chunk=generate
    for module in modules:module.flash_attention=fast
    backend.generate_chunk=generate_chunk;backend._full_length_attention_installed=True
    return restore


def audit_sealed_execution(frozen,marker,data,*,target,chunk_count=30,budget=4):
    from .main_table_merge import _load_frozen_index
    prefix='attempts/'+marker['attempt_id']+'/';gates={}
    for name in frozen.entries:
        if not name.startswith(prefix) or not name.endswith('/provenance/native-full-attention-probe-gate.json'):continue
        gate=json.loads(frozen.file(name,label='full attention gate').read_text());sha=gate.pop('sha256')
        proof=probe_identity(gate['device_name'])
        require(object_sha(gate)==sha and gate['schema']=='native_full_attention_probe_gate_v1'
            and gate['probe_code']==proof.code and gate['probe_spec_hash']==proof.spec
            and gate['helper_sha256']==HELPER_SHA,'full attention gate object differs')
        require({g['name'] for g in data['hardware']['gpus']}=={gate['device_name']} and bool(gate['sources']),
                'full attention gate hardware/source differs')
        for source in gate['sources']:
            index,seal=_load_frozen_index(Path(source['run_dir']));index.verify_all()
            require(seal==source['marker'] and seal['git_sha']==proof.code,'full attention source marker changed')
            validate_probe_manifest(json.loads(index.file('resolved-manifest.json',label='attention manifest').read_text()),
                                    device_name=gate['device_name'])
            relative='attempts/'+seal['attempt_id']+'/result.json'
            require(list(index.entries[relative])==source['result_binding'],'attention result binding changed')
            validate_probe(json.loads(index.file(relative,label='full attention probe').read_text()),
                           torch_version=gate['torch_version'],device_name=gate['device_name'])
        gates[name.split('/provenance/',1)[0]]=sha
    require(bool(gates),'sealed run omitted full attention gate')
    counts={r['prompt_id']:0 for r in data['rows']};calls=dense=fallbacks=0
    seen=set()
    for name in frozen.entries:
        if not name.startswith(prefix) or '/state/rollouts/' not in name or not name.endswith('.json'):continue
        from .insight2_run import _load_audited_checkpoint
        checkpoint,history,context,segment,_=_load_audited_checkpoint(
            frozen.file(name,label='full attention checkpoint'),label='full attention checkpoint')
        generated=checkpoint['generated']
        if generated['context']['role']!='draft':continue
        require(not target,'pure Target executed Draft generation')
        item=generated['diagnostics']['full_attention_execution_evidence'];validate_execution(item)
        require(item['probe_gate_sha256']==gates.get(name.split('/state/rollouts/',1)[0]),
                'attention execution belongs to another gate')
        pid=context.noise.prompt_id;require(pid in counts,'extra prompt in full attention audit')
        key=(pid,context.next_chunk_index)
        require(key not in seen,'duplicate native Draft checkpoint position');seen.add(key)
        counts[pid]+=1;calls+=item['counters']['calls'];dense+=item['counters']['dense_calls'];fallbacks+=item['counters']['fallbacks']
    require(all(n==(0 if target else chunk_count-1-budget) for n in counts.values()),'full attention checkpoint coverage differs')
    return {'schema':'sealed_native_full_attention_audit_v1','draft_calls':sum(counts.values()),
        'attention_calls':calls,'dense_calls':dense,'native_fallback_calls':fallbacks,
        'all_probe_and_call_bindings_verified':True}
