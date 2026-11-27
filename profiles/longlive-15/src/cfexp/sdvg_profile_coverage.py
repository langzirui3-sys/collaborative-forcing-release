"""Read sealed bounded profiles before main dispatch; never fill missing data.

A sparse SDVG trajectory may never call Target after warm-up. The identical
Krea operator can be covered by another declared campaign, but its measurement
remains a separate operator profile, not that trajectory's measured latency.
"""
import argparse
import json
import math
from pathlib import Path, PurePosixPath
import statistics

from .baseline_shard_report import source_manifest
from .main_table_merge import _load_frozen_index, _load_json, _require, _sha256_file

MODELS = {'self_forcing_1.3b','longlive_1.3b','rolling_forcing_1.3b'}


def validate_sample(row):
    role = row.get('role')
    _require(role in ('draft','target') and row.get('bounded_profile_sample') is True
        and type(row.get('warmup_forward_calls_before')) is int
        and row['warmup_forward_calls_before'] >= 2
        and row.get('action') == ('generate' if role=='draft' else 'generate_persistent'),
        'sample was not a post-warmup generation request')
    for key in ('request_wall_seconds','cuda_stream_elapsed_seconds','cuda_kernel_us_sum',
                'peak_allocated_bytes','peak_reserved_bytes'):
        x=row.get(key)
        _require(isinstance(x,(float,int)) and not isinstance(x,bool)
            and math.isfinite(x) and x>0, 'missing or invalid measured sample: '+key)
    _require(row['peak_reserved_bytes'] >= row['peak_allocated_bytes'], 'allocator peaks are inconsistent')


def read_profile(run, *, repo):
    run=Path(run).resolve(strict=True);frozen,marker=_load_frozen_index(run)
    prefix='attempts/'+marker['attempt_id']+'/'
    terminal=_load_json(frozen.file('run.json',label='terminal run'),label='terminal run')
    _require(terminal.get('execution_status')=='SUCCEEDED' and terminal.get('exit_code')==0,
             'profile execution did not succeed')
    envelope=_load_json(frozen.file('resolved-manifest.json',label='profile manifest'),label='manifest')
    invocation=envelope['manifest'];relative='experiments/'+invocation['metadata']['name']+'.manifest.json'
    _require(invocation==source_manifest(repo,marker['git_sha'],relative), 'profile source manifest differs')
    spec=invocation['spec'];env=spec['command']['env'];argv=spec['command']['argv']
    _require(spec['resources']['gpus']==1 and spec['parameters'].get('smoke') is True
        and '--smoke-prompt-id' in argv and argv[2]=='cfexp.tracked_sdvg'
        and env.get('CFEXP_SDVG_BOUNDED_PROFILE')=='1'
        and env.get('CFEXP_SDVG_TRACKING')=='1' and env.get('CFEXP_STAGE_TIMING')=='1',
        'source is not a separate bounded single-device profiling campaign')
    result=_load_json(frozen.file(prefix+'result.json',label='profile result'),label='result')
    _require(result.get('status')=='succeeded' and result.get('prompt_count')==1,
             'bounded campaign did not complete its fixed prompt')
    summary=_load_json(frozen.file(prefix+'provenance/tracking/profile-summary.json',
                                   label='profile summary'),label='profile summary')
    session=_load_json(frozen.file(prefix+'provenance/tracking/session.json',label='W&B session'),label='W&B')
    _require(summary['config']==session['config'] and summary['hardware']==result['hardware']
        and summary['config']['git_sha']==marker['git_sha'] and summary['device_count']==1,
        'profile configuration/hardware/source binding differs')
    c=summary['config']; bindings={x['target']:x['sha256'] for x in spec['inputs']}
    _require(c['phase']=='bounded_profile' and c['generation_seed']==int(env['CFEXP_GENERATION_SEED'])
        and c['nominal_seconds']==spec['parameters']['nominal_duration_seconds']
        and c['model_pair']==argv[argv.index('--draft-model-id')+1]
        and c['warmup_forward_calls_per_worker']==2, 'profile config differs from source workload')
    for role in ('draft','target','sdvg'):
        arg='--'+role+'-artifact-spec'
        _require(c[role+'_artifact_spec_sha256']==bindings[argv[argv.index(arg)+1]],
                 'artifact identity differs from frozen source input')
    samples=summary['samples'];_require(len(samples)==len({x['role'] for x in samples}), 'duplicate profile role')
    worker_rows=[]
    for relative in frozen.entries:
        if relative.startswith(prefix+'provenance/tracking/') and relative.endswith('.jsonl'):
            worker_rows.extend(json.loads(x) for x in frozen.file(relative,label='worker observations').read_text().splitlines())
    _require(sorted((json.dumps(x,sort_keys=True) for x in worker_rows if x['bounded_profile_sample']))
        ==sorted(json.dumps(x,sort_keys=True) for x in samples), 'sample differs from worker observations')
    for row in samples:
        validate_sample(row)
        relative=PurePosixPath(row['trace_path'])
        _require(not relative.is_absolute() and '..' not in relative.parts
                 and str(relative).startswith('tracking/profiler/'), 'unsafe trace path')
        trace=run/prefix/str(relative)
        _require(not any(p.is_symlink() for p in (trace,*trace.parents))
            and _sha256_file(trace)[0]==row['trace_sha256'], 'server trace differs from bound hash')
    timings={}
    for role in ('draft','target'):
        action='generate' if role=='draft' else 'generate_persistent'
        observations=[x['request_wall_seconds'] for x in worker_rows
            if x['role']==role and x['action']==action and x['warmup_forward_calls_before']>=2
            and not x['bounded_profile_sample']]
        timings[role]=dict(unprofiled_post_warmup_requests=len(observations),
            request_wall_median_seconds=statistics.median(observations) if observations else None,
            scope='worker request incl history/host work, stage-timing enabled; not pure compute or E2E')
    return dict(run_id=marker['run_id'],git_sha=marker['git_sha'],
        artifact_index_sha256=marker['artifact_index_sha256'],config=summary['config'],
        hardware=summary['hardware'],samples=samples,request_timings=timings,
        wandb_mode=session['mode'],wandb_synced_evidence_verified=False)


def coverage(profiles):
    expected={(m,s) for m in MODELS for s in (15,30)}
    keys=[(x['config']['model_pair'],x['config']['nominal_seconds']) for x in profiles]
    _require(len(profiles)==6 and set(keys)==expected and len(set(keys))==6,
             'need all six declared model/horizon workloads')
    target_identity=None;revision=None;common=[]
    for p in profiles:
        c=p['config'];gpus=p['hardware'].get('gpus',[])
        _require(c['physical_device_count']==c['batch_size']==1 and c['rank']==0
            and c['output_resolution']==[480,832] and c['threshold']==-.7
            and len(gpus)==1 and gpus[0]['name']=='NVIDIA H200', 'profile hardware/batch/resolution differs')
        _require(c['method']=='sdvg' and c['phase']=='bounded_profile' and c['root_target'] is True
            and c['native_kv'] is True and c['warmup_forward_calls_per_worker']==2
            and c['source_population_sha256']=='8a2f89312402c2688cb00074194d748e264dd47e5a2f2e6fe010ca2afb0d9d94',
            'profile protocol differs from frozen main population/operator')
        identity=(c['target_artifact_spec_sha256'],json.dumps(c['target_adapter_config'],sort_keys=True),
                  c['precision'],c['native_kv'])
        if target_identity is None:target_identity,revision=identity,p['git_sha']
        _require(identity==target_identity and p['git_sha']==revision==c['git_sha'],
                 'common Target operator source/configuration is not identical')
        roles=[]
        for row in p['samples']:
            validate_sample(row);roles.append(row['role'])
            if row['role']=='target':common.append(dict(source_run_id=p['run_id'],sample=row))
        _require(roles.count('draft')==1 and roles.count('target') in (0,1)
                 and len(roles)==len(set(roles)), 'Draft coverage missing or sample roles duplicated')
    _require(common, 'no actual post-warmup Target operator sample; main dispatch remains gated')
    return dict(schema='sdvg_pre_main_profile_coverage_v1',status='verified_operator_coverage',
        profile_count=6,profiles=profiles,common_target_operator_profiles=common,
        no_imputed_trajectory_target_latencies=True,
        caveats=['Sparse Target timing remains missing for that trajectory; common operator samples stay separate.',
            'Profiler windows, stage-instrumented request latency, and main video-ready time are different scopes.',
            'Per-process allocator peaks must not be summed as simultaneous physical-device peak memory.',
            'These six diagnostic videos are not part of any384case main quality aggregate.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
    runs=json.loads(a.runs.read_text());_require(isinstance(runs,list),'runs must be a list')
    print(json.dumps(coverage([read_profile(r,repo=a.repo) for r in runs]),allow_nan=False))


if __name__=='__main__':main()
