"""Explicit, probe-gated native RoPE reuse in the SF worker only."""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import re

from .h800_native_probe_archive import ARCHIVE_SHA as H800_ARCHIVE_SHA
from .rope_coefficients import (ARCHIVE_SHA, MODULE_SHA, FUNCTION_SHA,
                                RopeCoefficientCache)

ENV = 'CFEXP_NATIVE_ROPE_COEFFICIENT_REUSE'
PROBE_CODE = '7d0e04644374fe88b2237c07f1035b6e10b9b750'
PROBE_SPEC = 'b3b05ae215ec9b5d348b67b2f1c193f4fca88a59133c011cb966bc632b99db45'
HELPER_SHA = 'e63063c9ce2f74208e98482838513941288de16f6b8520a511826b5919d1db1e'
PREFIX = 'native-rope-coefficient-gpu-probe-ntu-v1-'


@dataclass(frozen=True)
class NativeProbeIdentity:
    prefix: str
    code: str
    spec: str
    archive_sha: str


H800_PROBE = NativeProbeIdentity(
    'native-rope-coefficient-gpu-probe-h800-v1-',
    '72beb42e544426249f1895be5093fa5c332762fd',
    '68c84cf026798dc20d9832f043a7f279a26eb213981499b1c3c122d9faf24db6',
    H800_ARCHIVE_SHA,
)


def probe_identity(device_name):
    if device_name == 'NVIDIA H800':
        return H800_PROBE
    return NativeProbeIdentity(PREFIX, PROBE_CODE, PROBE_SPEC, ARCHIVE_SHA)


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_probe_manifest(resolved, *, device_name=None):
    from .manifest import manifest_hash
    proof=probe_identity(device_name)
    manifest=copy.deepcopy(resolved['manifest'])
    require(manifest['spec']['code']['revision'] in ('HEAD',proof.code), 'probe code binding differs')
    manifest['spec']['code']['revision']='HEAD'
    # The runner expands this sole non-secret destination variable. Its actual
    # immutable destination remains checked by the sealed artifact-index loader;
    # compare execution configuration against the published templated manifest.
    remote=manifest['spec']['storage']['remote']
    require(isinstance(remote,str) and bool(remote), 'probe artifact destination missing')
    manifest['spec']['storage']['remote']='${CFEXP_ARTIFACT_REMOTE}'
    require(manifest_hash(manifest)==proof.spec, 'native probe specification differs')


def validate_probe(data, *, torch_version, device_name):
    require(data['schema']=='native_rope_coefficient_cuda_probe_v1'
        and data['torch_version']==torch_version
        and data['archive_sha256']==probe_identity(device_name).archive_sha and data['module_sha256']==MODULE_SHA
        and data['function_sha256']==FUNCTION_SHA, 'wrong native RoPE probe implementation/runtime')
    require(data['generation_worker_integration'] is data['video_quality_evaluated']
        is data['end_to_end_speedup_claim'] is False, 'kernel probe scope differs')
    rows=data['rows']
    require(len(rows)==4 and {(r['device_index'],r['grid'][0][0]) for r in rows}
        =={(i,f) for i in (0,1) for f in (1,3)}, 'probe geometry/device coverage incomplete')
    require(data['repeat_count']==20 and data['calls_per_timing_sample']==30,
            'probe timing population differs')
    for row in rows:
        f=row['grid'][0][0]
        require(row['grid']==[[f,30,52]] and row['shape']==[1,f*30*52,12,128]
            and row['dtype']=='bfloat16' and row['frequency_dtype']=='complex128'
            and row['device_name']==device_name and row['inputs_unchanged'] is True
            and row['source_mutation_checked'] is True, 'native kernel inputs differ')
        require(len(row['parity'])==5
            and [p['start_frame'] for p in row['parity']]==[0,3,12,45,87]
            and all(p['bitwise_equal'] is True for p in row['parity']), 'CUDA parity failed')
        cache=row['cache']
        require(all(type(cache[k]) is int and cache[k]>=0
                    for k in ('calls','hits','misses','fallbacks','resident_coefficient_bytes',
                              'peak_coefficient_bytes','bound_bytes','coefficient_count'))
            and cache['calls']==cache['hits']+cache['misses']+cache['fallbacks']
            and cache['calls']==616*(1 if f==1 else 2)
            and cache['hits']>0 and cache['fallbacks']==0 and cache['coefficient_count']==1
            and cache['resident_coefficient_bytes']==f*30*52*64*16
            and 0<cache['resident_coefficient_bytes']<=cache['peak_coefficient_bytes']
            <=cache['bound_bytes']==8*1024**2, 'probe cache bound/reuse differs')
        for key in ('native','cached'):
            values=row['timing_seconds_per_call'][key]
            require(len(values)==20 and all(type(v) in (int,float) and math.isfinite(v)
                and v>0 for v in values), 'probe timing incomplete/nonfinite')
            require(statistics.median(values)==row['median_'+key+'_seconds'],
                    'probe summary differs from samples')
        ratio=row['median_native_seconds']/row['median_cached_seconds']
        require(ratio==row['kernel_speedup_ratio_of_medians'] and ratio>=1.05,
                'native kernel probe did not show at least 5 percent local gain')


def probe_gate(run_dir, *, torch_version, device_name):
    from .main_table_merge import _load_frozen_index
    proof=probe_identity(device_name)
    roots=[p for p in [run_dir,*run_dir.parents]
           if (p/'resolved-manifest.json').is_file()]
    require(bool(roots), 'worker is not inside a formal run')
    root=roots[0]
    # run.json is written only after execution terminates. Bind the live worker
    # to the runner's pre-execution manifest AND attempt metadata instead.
    resolved=json.loads((root/'resolved-manifest.json').read_text())
    attempt_id=resolved.get('attempt_id')
    require(isinstance(attempt_id,str) and bool(attempt_id)
            and Path(attempt_id).name==attempt_id and attempt_id not in ('.','..'),
            'invalid active formal attempt identity')
    attempt=root/'attempts'/attempt_id
    require(run_dir.is_relative_to(attempt), 'worker lies outside the active formal attempt')
    metadata=json.loads((attempt/'metadata.json').read_text())
    require(resolved.get('run_id')==root.name
        and all(key in resolved and metadata.get(key)==resolved[key]
                for key in ('run_id','attempt_id','git','spec_hash','lock_sha256')),
        'active formal attempt metadata differs')
    sources=[]
    for source in sorted(root.parent.glob(proof.prefix+'*')):
        require(not source.is_symlink(), 'probe root may not be a symlink')
        require(not (source/'_FAILED.json').exists(), 'native probe failure requires a new version')
        if not (source/'_SUCCESS.json').exists():
            continue
        frozen,marker=_load_frozen_index(source);frozen.verify_all()
        require(marker['git_sha']==proof.code, 'native probe code differs')
        resolved=json.loads(frozen.file('resolved-manifest.json',label='probe manifest').read_text())
        validate_probe_manifest(resolved, device_name=device_name)
        relative='attempts/'+marker['attempt_id']+'/result.json'
        data=json.loads(frozen.file(relative,label='probe result').read_text())
        validate_probe(data,torch_version=torch_version,device_name=device_name)
        sources.append({'run_dir':str(source),'marker':marker,'result_binding':frozen.entries[relative]})
    require(bool(sources), 'no successful sealed native CUDA RoPE probe exists')
    evidence={'schema':'native_rope_probe_gate_v1','probe_code':proof.code,
              'probe_spec_hash':proof.spec,'helper_sha256':HELPER_SHA,
              'torch_version':torch_version,'device_name':device_name,'sources':sources}
    raw=json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()
    return evidence,hashlib.sha256(raw).hexdigest()


def validate_execution(value):
    require(isinstance(value,dict) and value.get('schema')=='native_rope_execution_v2'
        and value.get('helper_sha256')==HELPER_SHA
        and value.get('native_float64_complex128_math') is True,
        'missing native RoPE execution identity')
    require(isinstance(value.get('probe_gate_sha256'),str)
        and re.fullmatch('[0-9a-f]{64}',value['probe_gate_sha256']) is not None,
        'missing native RoPE probe binding')
    require(isinstance(value.get('frequency_initialization_sha256'),str)
        and re.fullmatch('[0-9a-f]{64}',value['frequency_initialization_sha256']) is not None,
        'missing versioned native frequency binding')
    counters=value.get('counters',{})
    require(all(type(counters.get(k)) is int and counters[k]>=0
                for k in ('calls','hits','misses','fallbacks'))
        and counters['calls']==sum(counters[k] for k in ('hits','misses','fallbacks'))
        and counters['hits']>0, 'native RoPE did not actually reuse constants')
    require(type(value.get('peak_coefficient_bytes')) is int
        and 0<value['peak_coefficient_bytes']<=value.get('bound_bytes',0)==8*1024**2,
        'native RoPE memory bound differs')


def prepare_frequencies(model, *, torch, device):
    """Perform native frequency placement before inference_mode, without RNG.

    The original forward moves this plain attribute to the weight device on
    first use. Moving it inside inference_mode creates an unversioned tensor,
    correctly rejected by the coefficient cache. An equal, versioned copy lets
    that unchanged cache enforce its mutation checks in real generation.
    """
    source=model.freqs
    require(isinstance(source,torch.Tensor) and source.dtype==torch.complex128
        and source.ndim==2 and min(source.shape)>0 and not source.requires_grad
        and source.numel()*source.element_size()<=1024**2,
        'native frequency source contract differs')
    require(model.patch_embedding.weight.device==torch.device(device),
            'native frequency and model execution devices differ')
    def digest(value):
        return hashlib.sha256(value.detach().to('cpu').contiguous()
                              .view(torch.uint8).numpy().tobytes()).hexdigest()
    with torch.inference_mode(False),torch.no_grad():
        prepared=source.detach().to(device=device,copy=True)
        source_sha=digest(source);prepared_sha=digest(prepared)
    require(not torch.is_inference(prepared) and type(prepared._version) is int
        and source_sha==prepared_sha, 'frequency initialization changed values or lacks version tracking')
    evidence={'schema':'native_rope_versioned_frequency_initialization_v1',
        'dtype':str(prepared.dtype),'shape':list(prepared.shape),
        'source_device':str(source.device),'prepared_device':str(prepared.device),
        'source_sha256':source_sha,'prepared_sha256':prepared_sha,
        'prepared_is_inference':False,'prepared_version':prepared._version,
        'original_values_preserved':True,'initialization_before_generation':True}
    return prepared,evidence


def validate_frequency(value):
    require(value.get('schema')=='native_rope_versioned_frequency_initialization_v1'
        and value.get('dtype')=='torch.complex128'
        and isinstance(value.get('shape'),list) and len(value['shape'])==2
        and all(type(n) is int and n>0 for n in value['shape'])
        and math.prod(value['shape'])*16<=1024**2
        and isinstance(value.get('source_sha256'),str)
        and re.fullmatch('[0-9a-f]{64}',value['source_sha256']) is not None
        and value.get('prepared_sha256')==value['source_sha256']
        and value.get('prepared_is_inference') is False
        and type(value.get('prepared_version')) is int and value['prepared_version']>=0
        and value.get('original_values_preserved') is True
        and value.get('initialization_before_generation') is True,
        'native frequency initialization evidence differs')


def audit_sealed_execution(frozen, marker, data, *, target, chunk_count=30, budget=4):
    """Independently recheck indexed per-call evidence and its sealed CUDA gate."""
    from .main_table_merge import _load_frozen_index
    prefix='attempts/'+marker['attempt_id']+'/'
    gates={};frequencies={};counts={r['prompt_id']:0 for r in data['rows']};calls=hits=fallbacks=0
    for name in frozen.entries:
        if not name.startswith(prefix) or not name.endswith('/provenance/native-rope-probe-gate.json'):
            continue
        value=json.loads(frozen.file(name,label='native RoPE gate').read_text())
        sha=value.pop('sha256')
        proof=probe_identity(value['device_name'])
        require(hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()==sha,
                'native RoPE gate object binding differs')
        require(value['schema']=='native_rope_probe_gate_v1' and value['probe_code']==proof.code
            and value['probe_spec_hash']==proof.spec and value['helper_sha256']==HELPER_SHA,
            'native RoPE gate implementation differs')
        require({g['name'] for g in data['hardware']['gpus']}=={value['device_name']},
                'probe gate hardware class differs from generation')
        require(bool(value['sources']), 'native RoPE gate has no source')
        for source in value['sources']:
            index,source_marker=_load_frozen_index(Path(source['run_dir']));index.verify_all()
            require(source_marker==source['marker'] and source_marker['git_sha']==proof.code,
                    'sealed native RoPE source marker changed')
            resolved=json.loads(index.file('resolved-manifest.json',label='probe manifest').read_text())
            validate_probe_manifest(resolved, device_name=value['device_name'])
            relative='attempts/'+source_marker['attempt_id']+'/result.json'
            require(list(index.entries[relative])==source['result_binding'], 'probe result binding changed')
            probe=json.loads(index.file(relative,label='native CUDA probe').read_text())
            validate_probe(probe,torch_version=value['torch_version'],device_name=value['device_name'])
        cell=name.split('/provenance/',1)[0]
        gates[cell]=sha
        init=json.loads(frozen.file(cell+'/provenance/native-rope-frequency-initialization.json',
                                    label='versioned frequency initialization').read_text())
        init_sha=init.pop('sha256')
        require(hashlib.sha256(json.dumps(init,sort_keys=True,separators=(',',':')).encode()).hexdigest()==init_sha,
                'frequency initialization object binding differs')
        validate_frequency(init);frequencies[cell]=init_sha
    require(bool(gates), 'sealed run omitted the immutable native RoPE gate')
    seen=set()
    for name in frozen.entries:
        if not name.startswith(prefix) or '/state/rollouts/' not in name or not name.endswith('.json'):
            continue
        from .insight2_run import _load_audited_checkpoint
        checkpoint,history,context,segment,_=_load_audited_checkpoint(
            frozen.file(name,label='native RoPE checkpoint'),label='native RoPE checkpoint')
        generated=checkpoint['generated']
        if generated['context']['role']!='draft':
            continue
        require(not target, 'pure Target reference executed Draft generation')
        item=generated['diagnostics']['rope_execution_evidence'];validate_execution(item)
        cell=name.split('/state/rollouts/',1)[0]
        require(item['probe_gate_sha256']==gates.get(cell), 'Draft execution belongs to another CUDA gate')
        require(item['frequency_initialization_sha256']==frequencies.get(cell),
                'Draft execution belongs to another frequency initialization')
        pid=context.noise.prompt_id;require(pid in counts,'extra prompt in native RoPE execution')
        key=(pid,context.next_chunk_index)
        require(key not in seen,'duplicate native Draft checkpoint position');seen.add(key)
        counts[pid]+=1;calls+=item['counters']['calls'];hits+=item['counters']['hits']
        fallbacks+=item['counters']['fallbacks']
    require(all(v==(0 if target else chunk_count-1-budget) for v in counts.values()),
            'native RoPE checkpoint coverage differs from exact P8/Target calls')
    return {'schema':'sealed_native_rope_execution_audit_v2','draft_calls':sum(counts.values()),
            'rope_calls':calls,'coefficient_hits':hits,'native_fallback_calls':fallbacks,
            'gate_count':len(gates),'all_probe_and_call_bindings_verified':True}


def install(backend, run_dir):
    from . import rope_coefficients
    from .self_forcing_frame_adapter import OfficialSelfForcingBackend
    from .main_table_baseline_campaign import _write_new_json
    from .sdvg_adapter import draft_generation_backend

    backend = draft_generation_backend(backend)
    require(type(backend) is OfficialSelfForcingBackend
        and getattr(backend,'_cpu_history_binding_installed',False),
        'native RoPE requires the explicit native-KV CPU-bound SF path')
    require(not getattr(backend,'_rope_coefficients_installed',False), 'native RoPE already installed')
    require(hashlib.sha256(Path(rope_coefficients.__file__).read_bytes()).hexdigest()==HELPER_SHA,
            'RoPE helper differs from the CUDA-probed implementation')
    module=sys.modules.get('wan.modules.causal_model')
    require(module is not None, 'native causal attention module is not loaded')
    path=Path(module.__file__).resolve(strict=True)
    require(path.is_relative_to(backend._source_root.resolve(strict=True))
        and hashlib.sha256(path.read_bytes()).hexdigest()==MODULE_SHA,
        'loaded native causal attention source differs')
    text=path.read_text();nodes=[n for n in ast.parse(text).body
        if isinstance(n,ast.FunctionDef) and n.name=='causal_rope_apply']
    require(len(nodes)==1 and not nodes[0].decorator_list
        and hashlib.sha256(ast.get_source_segment(text,nodes[0]).encode()).hexdigest()==FUNCTION_SHA,
        'loaded native RoPE function differs')
    original=module.causal_rope_apply
    require(getattr(original,'__module__',None)==module.__name__
        and getattr(original,'__name__',None)=='causal_rope_apply', 'native RoPE was already wrapped')
    evidence,gate_sha=probe_gate(run_dir,torch_version=str(backend.torch.__version__),
        device_name=backend.torch.cuda.get_device_name(backend.device))
    model=backend.pipeline.generator.model
    original_frequencies=model.freqs
    prepared,frequency_evidence=prepare_frequencies(model,torch=backend.torch,device=backend.device)
    validate_frequency(frequency_evidence)
    frequency_sha=hashlib.sha256(json.dumps(frequency_evidence,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    _write_new_json(run_dir/'provenance/native-rope-probe-gate.json',dict(evidence,sha256=gate_sha))
    _write_new_json(run_dir/'provenance/native-rope-frequency-initialization.json',
                    dict(frequency_evidence,sha256=frequency_sha))
    cache=RopeCoefficientCache(original,backend.torch)
    generate=backend.generate_chunk

    def generate_chunk(**kwargs):
        require(model.freqs is prepared, 'native frequency ownership changed during generation')
        require(prepared._version==frequency_evidence['prepared_version'],
                'native frequency values changed after initialization')
        before=cache.evidence()
        result=generate(**kwargs)
        require(prepared._version==frequency_evidence['prepared_version'],
                'native generation changed the fixed frequency source')
        after=cache.evidence()
        item={'schema':'native_rope_execution_v2','probe_gate_sha256':gate_sha,
            'frequency_initialization_sha256':frequency_sha,
            'helper_sha256':HELPER_SHA,'native_float64_complex128_math':True,
            'counters':{k:after[k]-before[k] for k in ('calls','hits','misses','fallbacks')},
            'peak_coefficient_bytes':after['peak_coefficient_bytes'],'bound_bytes':after['bound_bytes']}
        validate_execution(item)
        return dict(result,rope_execution_evidence=item)

    def restore():
        require(module.causal_rope_apply is cache, 'native RoPE wrapper changed during execution')
        require(model.freqs is prepared, 'native frequency ownership changed during execution')
        module.causal_rope_apply=original
        model.freqs=original_frequencies
        backend.generate_chunk=generate
        cache.clear()

    model.freqs=prepared
    module.causal_rope_apply=cache
    backend.generate_chunk=generate_chunk
    backend._rope_coefficients_installed=True
    return restore
