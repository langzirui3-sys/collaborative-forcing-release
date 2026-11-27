"""Compare complete native policy evidence, not only a fused latent primitive."""
import hashlib
import json

from .branch_kv_state import payloads_alias
from .incremental_draft import IncrementalDraftExecutor, latent_digest
from .longlive_joint_executor_oracle import (
    CASES, SCHEDULE, PREFIX_BOOTSTRAP, construct_native_prefix, expected_forward_trace)
from .longlive_joint_handoff import require
from .longlive_joint_policy_capture import install, install_capture_hooks
from .longlive_policy_backend import LongLivePolicyBackend
from .prefix_oracle import check_suffix
from .self_forcing_frame_adapter import (
    StepZeroCapture, LAYER_INDICES, EFFECTIVE_RANK_LAYER)
from .stage_timing import stage


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


def run(backend, *, prompt, history, noises, report):
    t = backend.torch
    session = backend.session
    require(not session.host_cursors.enabled and backend._active_capture is None
            and session.executor.cache.resident_bytes == 0 and len(noises) == 7,
            'policy gate requires an empty exclusive native owner and seven noises')
    install(backend)
    joint = session.executor
    p = joint.pipeline
    reference = IncrementalDraftExecutor(pipeline=p, torch=t,
        scope_sha256=joint.scope_sha256, max_bytes=joint.cache.max_bytes, fork_enabled=False)
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
