"""Actual CUDA forward observations for the isolated Rolling case gate.

The two sampled calls have different work: a joint clean suffix plus first
noisy block, and an ordinary first noisy block. Their ratio is NOT a speedup.
Full-cache oracle hooks also add overhead. Production never selects this file.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import time

from .kernel_profile import summarize_events
from .rolling_joint_handoff import active_handoff, require
from .stage_timing import stage

WARMUP = 2
KINDS = ('joint_suffix_first_window', 'native_first_window_at12')


def profile_kinds(target_suffix_blocks):
    require(type(target_suffix_blocks) is int and target_suffix_blocks in (2,4),
            'expected a frozen B2 or B4 profiling budget')
    return KINDS[0], 'native_first_window_at'+str(8+target_suffix_blocks)


def classify(*, steps, current_start, updating_cache, handoff, target_suffix_blocks=4):
    kinds = profile_kinds(target_suffix_blocks)
    resume = 8+target_suffix_blocks
    require(type(current_start) is int and current_start >= 0
            and type(updating_cache) is bool and steps
            and all(type(x) in (int, float) and math.isfinite(x) for x in steps),
            'invalid actual Rolling forward metadata')
    if handoff is not None:
        require(handoff == (resume*4680, target_suffix_blocks) and current_start == 8*4680
                and not updating_cache and steps == [0.]*(3*target_suffix_blocks)+[1000.]*3,
                'joint physical generator call differs from the fixed-budget case')
        return kinds[0]
    if not updating_cache and current_start == resume*4680 and steps == [1000.]*3:
        return kinds[1]
    require(all(x in (0., 1000., 937.5, 833.3333129882812, 625.) for x in steps),
            'physical generator changed the native Rolling schedule')
    return 'native_clean' if updating_cache else 'native_window'


class Selection:
    def __init__(self, target_suffix_blocks=4):
        self.kinds = profile_kinds(target_suffix_blocks)
        self.counts = {}; self.sampled = set()

    def before(self, kind):
        count = self.counts.get(kind, 0)
        return count, kind in self.kinds and count >= WARMUP and kind not in self.sampled

    def commit(self, kind, sampled):
        self.counts[kind] = self.counts.get(kind, 0)+1
        if sampled: self.sampled.add(kind)


@contextmanager
def observe(backend, root, *, target_suffix_blocks=4):
    t, generator = backend.torch, backend.pipeline.generator
    require(t.cuda.is_available() and backend.device.type == 'cuda'
            and 'forward' not in vars(generator), 'actual unpatched CUDA generator required')
    original = generator.forward; selection = Selection(target_suffix_blocks); records = []
    path = root/'provenance/tracking/physical-forwards.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True); path.open('x').close()

    def forward(*args, **kwargs):
        steps = kwargs['timestep'].detach().reshape(-1).cpu().tolist()
        kind = classify(steps=steps, current_start=kwargs['current_start'],
            updating_cache=kwargs.get('updating_cache', False), handoff=active_handoff(),
            target_suffix_blocks=target_suffix_blocks)
        count, sampled = selection.before(kind); profiler = None
        if sampled:
            t.cuda.synchronize(); t.cuda.reset_peak_memory_stats()
            profiler = t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU,
                t.profiler.ProfilerActivity.CUDA], record_shapes=True,
                profile_memory=True, with_stack=False)
            begin, end = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
            begin.record()
        started = time.perf_counter()
        with stage('rolling_physical_'+kind, cuda=True):
            if profiler is None: value = original(*args, **kwargs)
            else:
                with profiler: value = original(*args, **kwargs)
        if sampled: end.record(); end.synchronize()
        row = dict(sequence=len(records), kind=kind, current_start=kwargs['current_start'],
            timesteps=steps, latent_frames=int(kwargs['noisy_image_or_video'].shape[1]),
            updating_cache=kwargs.get('updating_cache', False),
            warmup_matching_forwards_before=count, bounded_profile_sample=sampled,
            wall_seconds=time.perf_counter()-started,
            peak_allocated_bytes=t.cuda.max_memory_allocated(),
            peak_reserved_bytes=t.cuda.max_memory_reserved(),
            role='draft', rank=0, diagnostic_oracle_hooks_included=True)
        if sampled:
            events = summarize_events(profiler.events())
            require(any(x['kind']=='kernel' for x in events), 'no actual CUDA kernel events')
            trace = root/'tracking/profiler'/(kind+'.trace.json')
            trace.parent.mkdir(parents=True, exist_ok=True)
            require(not trace.exists(), 'CUDA trace already exists')
            profiler.export_chrome_trace(str(trace))
            row.update(cuda_stream_seconds=begin.elapsed_time(end)/1000,
                cuda_kernel_us_sum=sum(x['cuda_us'] for x in events if x['kind']=='kernel'),
                cuda_transfer_us_sum=sum(x['cuda_us'] for x in events if x['kind']=='transfer'),
                trace_path=trace.relative_to(root).as_posix(),
                trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
            row['noisy_latent_frames_per_cuda_stream_second'] = 3/row['cuda_stream_seconds']
        with path.open('a') as f: f.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        records.append(row); selection.commit(kind, sampled)
        return value

    generator.forward = forward
    try: yield records
    finally: del generator.forward


def validate(records, *, target_suffix_blocks=4):
    selection = Selection(target_suffix_blocks)
    resume = 8+target_suffix_blocks
    require(records and [x['sequence'] for x in records] == list(range(len(records))),
            'physical generator coverage is incomplete')
    for row in records:
        joint = row['kind'] == KINDS[0]
        kind = classify(steps=row['timesteps'], current_start=row['current_start'],
            updating_cache=row['updating_cache'], handoff=(resume*4680, target_suffix_blocks) if joint else None,
            target_suffix_blocks=target_suffix_blocks)
        count, sampled = selection.before(kind)
        require(row['kind'] == kind and row['warmup_matching_forwards_before'] == count
                and row['bounded_profile_sample'] is sampled
                and row['latent_frames'] == len(row['timesteps'])
                and row['role'] == 'draft' and row['rank'] == 0
                and row['diagnostic_oracle_hooks_included'] is True,
                'physical generator/profile selection differs')
        for k in ('wall_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes'):
            require(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>0,
                    'missing actual timing or GPU memory')
        require(row['peak_reserved_bytes'] >= row['peak_allocated_bytes'], 'invalid GPU memory')
        if sampled:
            for k in ('cuda_stream_seconds', 'cuda_kernel_us_sum',
                      'noisy_latent_frames_per_cuda_stream_second'):
                require(type(row.get(k)) in (int,float) and math.isfinite(row[k]) and row[k]>0,
                        'missing actual CUDA measurement')
        selection.commit(kind, sampled)
    require(selection.sampled == set(selection.kinds) and selection.counts[KINDS[0]] == 3,
            'three real joint calls and both post-warmup CUDA samples required')
    return [x for x in records if x['bounded_profile_sample']]
