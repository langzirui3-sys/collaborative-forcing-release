"""Diagnostic-only real CUDA observations for the native LongLive gate.

Never imported by production generation. The original native oracle still
checks every output/cache/cursor; these instrumented timings are not speedups.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .kernel_profile import summarize_events
from .longlive_joint_handoff import require
from .stage_timing import _unsealed_directory,stage

ENV='CFEXP_LONGLIVE_JOINT_PROFILE'
WARMUP=2
KINDS=('native_first_denoise','joint_first_denoise')


def classify(steps):
    require(steps and all(type(x) in (int,float) and math.isfinite(x) for x in steps),
            'profile timesteps must be actual finite native values')
    unique=set(steps)
    if unique=={0.}:return 'native_clean_b3'
    if unique=={1000.}:return 'native_first_denoise'
    if unique=={0.,1000.}:
        require(len(steps)>3 and steps[-3:]==[1000.]*3 and all(x==0. for x in steps[:-3]),
                'profile mixed timestep lanes differ')
        return 'joint_first_denoise'
    require(len(unique)==1 and next(iter(unique)) in (937.5,833.3333129882812,625.),
            'profile changed native LongLive schedule')
    return 'native_later_denoise'


class Selection:
    def __init__(self):self.counts={};self.sampled=set()
    def before(self,kind):
        count=self.counts.get(kind,0)
        return count,kind in KINDS and count>=WARMUP and kind not in self.sampled
    def commit(self,kind,selected):
        self.counts[kind]=self.counts.get(kind,0)+1
        if selected:self.sampled.add(kind)


def append(path, record):
    with path.open('a') as f:f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')


@contextmanager
def forward_observations(backend, run_dir):
    require(os.environ.get(ENV)=='1','native joint profiling requires explicit opt-in')
    root=_unsealed_directory(Path(run_dir));t=backend.torch
    require(t.cuda.is_available(),'native joint profiling requires actual CUDA; no CPU fallback')
    generator=backend.session.executor.pipeline.generator
    original=generator.forward;had_local='forward' in generator.__dict__;local=generator.__dict__.get('forward')
    directory=root/'provenance'/'tracking';directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'draft-forwards-{os.getpid()}.jsonl';path.open('x').close()
    selection=Selection()
    def observed(*args,**kwargs):
        steps=kwargs['timestep'].detach().flatten().cpu().tolist();kind=classify(steps)
        count,selected=selection.before(kind);profiler=None
        if selected:
            t.cuda.synchronize();t.cuda.reset_peak_memory_stats()
            profiler=t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU,t.profiler.ProfilerActivity.CUDA],
                record_shapes=True,profile_memory=True,with_stack=False)
            begin,end=t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True)
            begin.record()
        start=time.perf_counter()
        with stage(kind,cuda=True):
            if profiler is None:value=original(*args,**kwargs)
            else:
                with profiler:value=original(*args,**kwargs)
        if selected:end.record();end.synchronize()
        elapsed=time.perf_counter()-start
        row=dict(schema='longlive_joint_forward_observation_v1',role='draft',pid=os.getpid(),rank=0,
            kind=kind,sequence=sum(selection.counts.values()),warmup_matching_forwards_before=count,
            bounded_profile_sample=selected,current_start=kwargs['current_start'],timesteps=steps,
            latent_frames=int(kwargs['noisy_image_or_video'].shape[1]),request_wall_seconds=elapsed,
            peak_allocated_bytes=t.cuda.max_memory_allocated(),peak_reserved_bytes=t.cuda.max_memory_reserved(),
            wall_scope='instrumented generator forward with native oracle hooks; not chunk/E2E latency',
            memory_scope='own worker allocator; not simultaneous physical-device sum')
        if selected:
            events=summarize_events(profiler.events());require(any(x['kind']=='kernel' for x in events),'no actual CUDA kernel events')
            trace=root/'tracking'/'profiler'/f'{kind}-{os.getpid()}.trace.json';trace.parent.mkdir(parents=True,exist_ok=True)
            require(not trace.exists(),'profile trace already exists');profiler.export_chrome_trace(str(trace))
            row.update(cuda_stream_elapsed_seconds=begin.elapsed_time(end)/1000,
                cuda_kernel_us_sum=sum(x['cuda_us'] for x in events if x['kind']=='kernel'),
                cuda_transfer_us_sum=sum(x['cuda_us'] for x in events if x['kind']=='transfer'),
                trace_path=trace.relative_to(root).as_posix(),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                comparison_frame_equivalent_count=12)
            row['stream_window_frame_equivalents_per_second']=12/row['cuda_stream_elapsed_seconds']
        append(path,row);selection.commit(kind,selected);return value
    generator.forward=observed
    try:yield
    finally:
        if had_local:generator.forward=local
        else:delattr(generator,'forward')


def validate_forward_records(records, expected_trace):
    require(len(records)==len(expected_trace) and [x['sequence'] for x in records]==list(range(len(expected_trace))),
            'joint profile omitted or duplicated actual forwards')
    selection=Selection()
    for row,trace in zip(records,expected_trace):
        kind=classify(trace['timesteps']);count,sampled=selection.before(kind)
        require(all(row.get(k)==v for k,v in trace.items()) and row['kind']==kind
            and row['warmup_matching_forwards_before']==count and row['bounded_profile_sample'] is sampled
            and row['role']=='draft' and row['rank']==0
            and type(row['request_wall_seconds']) in (float,int) and math.isfinite(row['request_wall_seconds'])
            and row['request_wall_seconds']>0 and type(row['peak_allocated_bytes']) is int
            and row['peak_allocated_bytes']>0 and row['peak_reserved_bytes']>=row['peak_allocated_bytes'],
            'joint forward observation differs from real oracle trace or warm-up policy')
        if sampled:
            for key in ('cuda_stream_elapsed_seconds','cuda_kernel_us_sum','stream_window_frame_equivalents_per_second'):
                require(type(row.get(key)) in (float,int) and math.isfinite(row[key]) and row[key]>0,
                        'joint profile lacks a real positive CUDA observation')
        selection.commit(kind,sampled)
    require(selection.sampled==set(KINDS),'both post-warmup native and joint CUDA profiles are required')
    return [x for x in records if x['bounded_profile_sample']]
