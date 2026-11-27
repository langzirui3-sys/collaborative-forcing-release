"""Compare complete native policy evidence, not only a fused latent primitive."""
import hashlib
import json

from .branch_kv_state import payloads_alias
from .incremental_draft import IncrementalDraftExecutor, latent_digest
from .longlive_joint_executor_oracle import (
    CASES, SCHEDULE, PREFIX_BOOTSTRAP, construct_native_prefix, expected_forward_trace)
from .longlive_joint_handoff import require
from .longlive_fusion_capture import install_capture_hooks
from .longlive_policy_backend import LongLivePolicyBackend
from .prefix_oracle import check_suffix
from .self_forcing_frame_adapter import (
    StepZeroCapture, LAYER_INDICES, EFFECTIVE_RANK_LAYER)
from .stage_timing import stage


def cold_probe_trace():
    return [dict(current_start=0, latent_frames=3, timesteps=[0.] * 3)] + [
        dict(current_start=4680, latent_frames=3, timesteps=[float(step)] * 3)
        for step in (*SCHEDULE, 0.)]


def expected_observed_forward_trace():
    # Six fresh original-policy reference calls, then the unchanged 410-call
    # oracle. The earlier six normal-RPC calls have their own physical trace.
    return cold_probe_trace() + expected_forward_trace()


def compare_normal_rpc(*, torch, written, expected, actual_cache, expected_cache):
    # Both public policy results and the worker's safetensors roundtrip are
    # native BF16 CPU tensors. KV payloads retain their own native devices.
    # Moving only the written result to the model device makes equality with
    # the untouched CPU reference invalid, even when their bytes agree.
    require(isinstance(written, torch.Tensor) and isinstance(expected, torch.Tensor)
            and written.device.type == expected.device.type == 'cpu'
            and written.dtype == expected.dtype == torch.bfloat16
            and written.shape == expected.shape,
            'normal RPC/native reference must retain the BF16 CPU result contract')
    return check_suffix(actual=written, expected=expected, actual_cache=actual_cache,
                        expected_cache=expected_cache, torch=torch)


def policy_capture_hooks(backend, *, ordinary):
    require(type(ordinary) is bool and backend._active_capture is None,
            'cannot change active policy hooks or use an implicit policy mode')
    for handle in backend._hook_handles:
        handle.remove()
    backend._hook_handles.clear()
    blocks=backend.pipeline.generator.model.blocks
    if ordinary:
        # Sharing mixed-aware hooks could hide a common observation error.
        backend._install_hooks(blocks)
    else:
        install_capture_hooks(backend, blocks)


def run(backend, *, prompt, history, noises, report, normal_probe):
    t = backend.torch
    session = backend.session
    require(not session.host_cursors.enabled and backend._active_capture is None
            and session.executor.cache.resident_bytes == 0 and len(noises) == 7,
            'policy gate requires an empty exclusive native owner and seven noises')
    require(getattr(backend, "fusestep_admission", None), "normal factory was not admitted")
    joint = session.executor
    p = joint.pipeline
    reference = IncrementalDraftExecutor(pipeline=p, torch=t,
        scope_sha256=joint.scope_sha256, max_bytes=joint.cache.max_bytes, fork_enabled=False)
    require(normal_probe['request']['chunk_index'] == 1
            and normal_probe['actual_trace'] == cold_probe_trace(),
            'ordinary RPC did not execute exactly its declared cold native path')
    session.executor = reference
    policy_capture_hooks(backend, ordinary=True)
    cache = expected = actual = None
    try:
        with t.inference_mode():
            expected = LongLivePolicyBackend.generate_chunk(backend, prompt=prompt,
                history_latents=history[:, :3], initial_noise=normal_probe['noise'],
                internal_seed=normal_probe['request']['internal_seed'], chunk_index=1,
                capture_hidden=False, history_kv_widths=(3,))
        require(t.equal(normal_probe['rng_cpu'], t.get_rng_state())
                and t.equal(normal_probe['rng_cuda'], t.cuda.get_rng_state()),
                'ordinary RPC/native cold policy RNG consumption differs')
        actual = normal_probe['result']
        for key in ('rank', 'rank_evidence', 'kv_witness', 'kv_witness_sha256', 'forward_trace'):
            require(actual[key] == expected[key], 'normal RPC/native policy observation differs: ' + key)
        reuse = dict(actual['kv_reuse'])
        require(reuse.pop('history_file_sha256') == normal_probe['history_file_sha256'],
                'normal RPC omitted the worker-owned accepted file binding')
        require(reuse == expected['kv_reuse'], 'normal RPC/native reuse evidence differs')
        require(reference.cache.state_count == 1, 'ordinary cold reference retained multiple states')
        cache = next(iter(reference.cache._states.values())).payload
        require(not payloads_alias(normal_probe['actual_cache'], cache),
                'normal RPC/native cold reference caches alias')
        proof = compare_normal_rpc(written=normal_probe['latent'],
            expected=expected['latent'], actual_cache=normal_probe['actual_cache'],
            expected_cache=cache, torch=t)
        proof.update(ordinary_generate_rpc_compared=True, rank_evidence_exact=True,
            rng_state_exact=True,
            accepted_clean_witness_exact=True, reuse_and_worker_file_bindings_exact=True,
            normal_rpc_physical_calls=6, independent_reference_physical_calls=6)
        report['normal_rpc_reference'] = proof
    finally:
        reference.cache.clear()
        session.executor = joint
        policy_capture_hooks(backend, ordinary=False)
        normal_probe.clear()  # Release the retained cold payload before the 35-pair oracle.
        cache = expected = actual = None
    report['native_prefix_bootstrap_policy'] = PREFIX_BOOTSTRAP
    trace, hidden, current = [], [], {}
    tokens = 3*p.frame_seq_length

    def pre(_module, _args, kwargs):
        steps = kwargs['timestep'].detach().flatten().cpu().tolist()
        current['capture'] = 1000. in steps
        trace.append(dict(current_start=kwargs['current_start'],
            latent_frames=int(kwargs['noisy_image_or_video'].shape[1]), timesteps=steps))

    def layer(_module, _args, output):
        if current.get('capture'):
            hidden.append(backend._native_hidden(output)[:, -tokens:].detach().clone())

    handles = [p.generator.register_forward_pre_hook(pre, with_kwargs=True),
               p.generator.model.blocks[14].register_forward_hook(layer)]

    def prefix(executor, count):
        require(backend._active_capture is None, 'prefix setup overlaps a policy capture')
        # Prefix construction contains only native zero-timestep calls. Keep
        # the real hooks active rather than removing/reinstalling them.
        backend._active_capture = StepZeroCapture(layers=LAYER_INDICES,
            target_timestep=1000, sample_hidden=backend._sample_hidden,
            rank_sample=backend._rank_sample, retain_rank_layer=EFFECTIVE_RANK_LAYER,
            policy_rank_only=True)
        try:
            construct_native_prefix(executor, prompt=prompt, history=history, count=count)
            require(not backend._active_capture.events,
                    'native clean-prefix setup produced a noisy policy observation')
        finally:
            backend._active_capture = None

    def payload(executor):
        require(executor.cache.state_count == 1,
                'policy gate did not retain exactly one committed native prefix')
        return next(iter(executor.cache._states.values())).payload

    def pair(h, noise, seed, expected_fused):
        require(not hidden and session.executor is joint,
                'a different policy owner or capture survived the previous pair')
        args = dict(prompt=prompt, history_latents=h, initial_noise=noise,
            internal_seed=seed, chunk_index=int(h.shape[1])//3, capture_hidden=False,
            history_kv_widths=(3,)*(int(h.shape[1])//3))
        session.executor = reference
        policy_capture_hooks(backend,ordinary=True)
        try:
            with stage('policy_reference_chunk', cuda=True):
                expected = LongLivePolicyBackend.generate_chunk(backend, **args)
            require(len(hidden) == 1, 'ordinary policy did not expose exactly one native hidden')
            expected_hidden = hidden.pop()
            expected_gpu_rng = t.cuda.get_rng_state().clone()
            expected_cpu_rng = t.get_rng_state().clone()
        finally:
            session.executor = joint
            policy_capture_hooks(backend,ordinary=False)
        with stage('policy_joint_chunk', cuda=True):
            actual = backend.generate_chunk(**args)
        require(len(hidden) == 1, 'joint policy did not expose exactly one native hidden')
        actual_hidden = hidden.pop()
        require(t.equal(expected_gpu_rng, t.cuda.get_rng_state())
                and t.equal(expected_cpu_rng, t.get_rng_state()),
                'complete policy path changed native random-number consumption')
        require(t.equal(expected_hidden, actual_hidden),
                'complete policy first-denoise layer15 hidden differs')
        require(actual['rank'] == expected['rank']
                and actual['rank_evidence'] == expected['rank_evidence'],
                'complete native policy eRank/Gram spectrum differs')
        require(actual['kv_witness'] == expected['kv_witness']
                and actual['kv_witness_sha256'] == expected['kv_witness_sha256'],
                'joint policy did not witness the same accepted clean history')
        require(actual['kv_reuse'] == expected['kv_reuse'],
                'complete policy accepted-history/cache binding differs')
        execution = actual['longlive_joint_execution']
        require(execution['fused_history_chunks'] == expected_fused,
                'policy gate did not use its declared native joint/fallback path')
        ap, bp = payload(joint), payload(reference)
        require(not payloads_alias(ap, bp), 'independent policy caches alias storage')
        proof = check_suffix(actual=actual['latent'], expected=expected['latent'],
            actual_cache=ap, expected_cache=bp, torch=t)
        proof.update(oracle='original_incremental_native_executor_with_independent_fresh_prefix',
            hidden15_exact=True, rank_spectrum_exact=True, rng_state_exact=True,
            exact_history_binding=True, joint_execution=execution,
            self_cache_layers=len(ap['self']), cross_cache_layers=len(ap['cross']),
            effective_rank=actual['rank'],
            gram_spectrum_sha256=hashlib.sha256(json.dumps(
                actual['rank_evidence']['gram_eigenvalues'], allow_nan=False).encode()).hexdigest(),
            latent_sha256=latent_digest(actual['latent'], t),
            ordinary_policy_adapter_compared=True, policy_rank_evidence_exact=True,
            reference_original_capture_hooks=True,
            accepted_clean_kv_witness_exact=True, policy_reuse_evidence_exact=True,
            accepted_clean_kv_witness=actual['kv_witness'],
            accepted_clean_kv_witness_sha256=actual['kv_witness_sha256'])
        return actual['latent'].to(device=history.device), proof

    try:
        with t.inference_mode():
            for case, (retained, external) in enumerate(CASES):
                joint.cache.clear(); reference.cache.clear()
                prefix(joint, retained); prefix(reference, retained)
                h = history[:, :3*(retained+external)]
                row = dict(retained_blocks=retained, external_target_blocks=external,
                    chunk_index=retained+external, following=[])
                report['rows'].append(row)
                latent, row['current'] = pair(h, noises[0], 20260911+case*100, external)
                h = t.cat((h, latent), dim=1)
                for following in range(1, 7):
                    latent, proof = pair(h, noises[following],
                        20260911+case*100+following, 0)
                    row['following'].append(dict(offset=following, **proof))
                    h = t.cat((h, latent), dim=1)
                print('LONGLIVE_JOINT_POLICY_GATE', retained, external,
                      'latent/fullKV/hidden/rank/clean-witness/RNG exact', flush=True)
            require(trace == expected_forward_trace(),
                    'complete policy actual physical generator trace differs')
            report.update(passed=True, forward_trace=trace)
            return report
    finally:
        for handle in handles:
            handle.remove()
        backend._active_capture = None
        session.executor = joint
        joint.cache.clear(); reference.cache.clear()
        p.kv_cache1 = p.crossattn_cache = None
        report['forward_trace'] = trace
