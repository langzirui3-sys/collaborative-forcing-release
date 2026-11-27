"""Trained-model CUDA oracle for native LongLive joint execution, not timing."""
from .longlive_joint_handoff import joint_first_step, require
from .prefix_oracle import check_suffix
from .longlive_joint_trace import ModuleTrace, difference, validate_trace

CASES = ((1, 1), (1, 4), (3, 4), (7, 4), (11, 1))
SCHEDULE = (1000., 937.5, 833.3333129882812, 625.)


def run(backend, *, prompt, history, noises, report):
    t, p = backend.torch, backend.session.executor.pipeline
    # Native construction initializes kv_cache1 but creates crossattn_cache
    # lazily at its first allocation. Absence is empty, never a valid payload.
    require(not backend.session.host_cursors.enabled and p.kv_cache1 is None
            and getattr(p, 'crossattn_cache', None) is None
            and backend.session.executor.cache.resident_bytes == 0,
            'joint gate requires exclusive empty native state and device cursors')
    require(tuple(float(x) for x in p.denoising_step_list) == SCHEDULE,
            'joint gate cannot change the native LongLive schedule')
    handles = []
    trace, hidden = [], []
    capture = False
    chunk_tokens = 3 * p.frame_seq_length

    def pre(_m, _a, kw):
        trace.append(dict(current_start=kw['current_start'],
            timesteps=kw['timestep'].detach().flatten().cpu().tolist(),
            latent_frames=int(kw['noisy_image_or_video'].shape[1])))

    def layer(_m, _a, output):
        if capture:
            hidden.append(backend._native_hidden(output)[:, -chunk_tokens:].detach().clone())

    # Serialized, one-shot diagnostic: normal policy instrumentation rejects
    # mixed timesteps. It is removed, not weakened for production requests.
    for handle in backend._hook_handles:
        handle.remove()
    backend._hook_handles.clear()
    handles.append(p.generator.register_forward_pre_hook(pre, with_kwargs=True))
    handles.append(p.generator.model.blocks[14].register_forward_hook(layer))
    try:
        with t.inference_mode():
            cond = p.text_encoder(text_prompts=[prompt])
            zeros = t.zeros((1, 3), device=history.device, dtype=t.int64)

            def call(payload, video, start, timestep):
                p.kv_cache1, p.crossattn_cache = payload['self'], payload['cross']
                try:
                    _, clean = p.generator(noisy_image_or_video=video,
                        conditional_dict=cond, timestep=timestep,
                        kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                        current_start=start)
                    return clean
                finally:
                    p.kv_cache1 = p.crossattn_cache = None

            def prefix(count, probe=None):
                p._initialize_kv_cache(batch_size=1, dtype=t.bfloat16, device=history.device)
                p._initialize_crossattn_cache(batch_size=1, dtype=t.bfloat16, device=history.device)
                payload = dict(self=p.kv_cache1, cross=p.crossattn_cache)
                p.kv_cache1 = p.crossattn_cache = None
                for i in range(count):
                    if probe is not None and i == count-1:
                        with probe.native():
                            call(payload, history[:, 3*i:3*i+3], i*chunk_tokens, zeros)
                    else:
                        call(payload, history[:, 3*i:3*i+3], i*chunk_tokens, zeros)
                return payload

            def timestep(step):
                return t.ones((1, 3), device=history.device, dtype=t.int64) * step

            for retained, external in CASES:
                n = retained + external
                row = dict(retained_blocks=retained, external_target_blocks=external,
                           chunk_index=n, stages=[], following=[])
                report['rows'].append(row)
                clean_layers = []

                def observe(index, k, v, global_end, local_end):
                    expected = ordinary['self'][index]
                    exact = (t.equal(k, expected['k']) and t.equal(v, expected['v'])
                             and global_end == int(expected['global_end_index'])
                             and local_end == int(expected['local_end_index']))
                    if not exact:
                        row['first_clean_cache_difference'] = dict(layer=index,
                            k=difference(k, expected['k'], torch=t),
                            v=difference(v, expected['v'], torch=t),
                            actual_global_end=global_end,
                            expected_global_end=int(expected['global_end_index']),
                            actual_local_end=local_end,
                            expected_local_end=int(expected['local_end_index']))
                    require(exact, 'joint clean B3 cache differs from upstream at layer '+str(index))
                    clean_layers.append(index)

                row['module_trace'] = {}
                with ModuleTrace(p.generator.model, chunk_tokens=chunk_tokens,
                                 torch=t, report=row['module_trace']) as probe:
                    ordinary = prefix(n, probe=probe)
                    joint = prefix(retained)
                    start = n * chunk_tokens
                    capture = True
                    p.kv_cache1, p.crossattn_cache = joint['self'], joint['cross']
                    before = len(trace)
                    try:
                        with probe.joint(external):
                            actual = joint_first_step(p, torch=t,
                                history=history[:, 3*retained:3*n], noise=noises[0], conditional=cond,
                                history_start=retained*chunk_tokens, draft_start=start,
                                clean_observer=observe)
                    finally:
                        p.kv_cache1 = p.crossattn_cache = None
                validate_trace(row['module_trace'])
                require(len(trace) == before+1 and trace[-1] == dict(
                    current_start=retained*chunk_tokens, latent_frames=3*(external+1),
                    timesteps=[0.]*(3*external)+[1000.]*3),
                    'joint gate did not physically execute exactly one mixed forward')
                joint_hidden = hidden.pop()
                expected = call(ordinary, noises[0], start, timestep(SCHEDULE[0]))
                native_hidden = hidden.pop()
                require(len(hidden) == 0 and clean_layers == list(range(30)),
                        'joint gate omitted full clean-cache or hidden coverage')
                proof = check_suffix(actual=actual, expected=expected,
                    actual_cache=joint, expected_cache=ordinary, torch=t)
                require(t.equal(joint_hidden, native_hidden), 'joint layer15 hidden changed')
                row['stages'].append(dict(stage='first_denoise', hidden15_exact=True,
                    clean_self_cache_layers_exact=len(clean_layers), physical_joint_forwards=1, **proof))
                capture = False
                del joint_hidden, native_hidden
                for step_index, step in enumerate(SCHEDULE[1:], 1):
                    noisy = p.scheduler.add_noise(actual.flatten(0, 1), noises[step_index].flatten(0, 1),
                        step*t.ones((3,), device=history.device, dtype=t.long)).unflatten(0, actual.shape[:2])
                    actual = call(joint, noisy, start, timestep(step))
                    expected = call(ordinary, noisy, start, timestep(step))
                    row['stages'].append(dict(stage='denoise_'+str(step_index), **check_suffix(
                        actual=actual, expected=expected, actual_cache=joint, expected_cache=ordinary, torch=t)))
                call(joint, actual, start, zeros); call(ordinary, expected, start, zeros)
                row['stages'].append(dict(stage='final_clean_B3', **check_suffix(
                    actual=actual, expected=expected, actual_cache=joint, expected_cache=ordinary, torch=t)))
                for following in range(6):
                    position = (n+1+following)*chunk_tokens
                    a = b = noises[4+4*following]
                    capture = True
                    for step_index, step in enumerate(SCHEDULE):
                        a = call(joint, a, position, timestep(step))
                        ah = hidden.pop()
                        b = call(ordinary, b, position, timestep(step))
                        bh = hidden.pop()
                        require(t.equal(ah, bh), 'following native layer15 hidden changed')
                        check_suffix(actual=a, expected=b, actual_cache=joint, expected_cache=ordinary, torch=t)
                        if step_index < 3:
                            epsilon = noises[5+4*following+step_index]
                            destination = SCHEDULE[step_index+1]*t.ones((3,), device=history.device, dtype=t.long)
                            a = p.scheduler.add_noise(a.flatten(0, 1), epsilon.flatten(0, 1), destination).unflatten(0, a.shape[:2])
                            b = p.scheduler.add_noise(b.flatten(0, 1), epsilon.flatten(0, 1), destination).unflatten(0, b.shape[:2])
                    capture = False
                    call(joint, a, position, zeros); call(ordinary, b, position, zeros)
                    row['following'].append(dict(offset=following+1, hidden15_exact=True, **check_suffix(
                        actual=a, expected=b, actual_cache=joint, expected_cache=ordinary, torch=t)))
                del ordinary, joint
                print('LONGLIVE_JOINT_GATE', retained, external, 'all stages exact', flush=True)
            report['forward_trace'] = trace
            report['passed'] = True
            return report
    finally:
        for handle in handles:
            handle.remove()
        p.kv_cache1 = p.crossattn_cache = None
        report['forward_trace'] = trace
