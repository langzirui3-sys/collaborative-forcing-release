"""Diagnostic comparison on the original mixed Target/Draft sixteen-block case.

This is an oracle, not a formal entrypoint or a production selector. A driver
must independently admit the original sealed case, noise and checkpoint before
calling it. GPU execution, real profiles and complete-video measurements remain
separate requirements. Complete cache clones below are diagnostic overhead.
"""
import os

from .incremental_draft import clone_native_payload, latent_digest
from .rolling_clean_prefix_probe import compare_caches, replay_probe
from .rolling_forcing_session import RollingForcingSession
from .rolling_joint_handoff import ENV, require
from .rolling_joint_stream import JointCommittedSession

RETAINED = 8


def observation_equal(actual, expected, torch):
    """Check actual layer15 tensors and the complete spectrum, not only rank."""
    require(set(actual) == set(expected), 'Rolling observation fields differ')
    for key in ('chunk_index', 'layer', 'timestep', 'token_count',
                'effective_rank', 'gram_eigenvalues', 'kv_witness'):
        require(actual[key] == expected[key], 'Rolling observation differs: '+key)
    for key in ('hidden', 'sampled_hidden'):
        require(isinstance(actual[key], torch.Tensor)
                and isinstance(expected[key], torch.Tensor)
                and actual[key].dtype == expected[key].dtype
                and actual[key].shape == expected[key].shape
                and torch.equal(actual[key].cpu(), expected[key].cpu()),
                'Rolling layer15 tensor differs: '+key)


def run_case(*, pipeline, torch, history, noise, prompt, seed, ownership, target_suffix_blocks=4):
    """Replay the old case, keep its live Draft prefix, then test joint resume.

    History contains T0,D1..7,T8..11,D12..15 from the independently admitted
    case. Recomputed Draft1..7 must match those original bytes before accepting
    the admitted Target prefix. B4 requires the original complete trajectory.
    B2 uses only T8 and T9, then independently checks its own counterfactual
    continuation, for which no original output exists. No external KV is
    imported. The ordinary reference owns its own cache and RNG state.
    """
    t, p = torch, pipeline
    require(type(target_suffix_blocks) is int and target_suffix_blocks in (2, 4),
            'case oracle requires a frozen B2 or B4 budget')
    width = target_suffix_blocks
    resume = RETAINED + width
    target_blocks = tuple(range(RETAINED, resume))
    following = tuple(range(resume, 16))
    original_tail = width == 4
    require(os.environ.get(ENV) == '1' and not t.is_grad_enabled()
            and os.environ.get('CFEXP_ROLLING_HOST_KV_CURSORS', '0') == '0'
            and os.environ.get('CFEXP_ROLLING_HOST_WINDOW', '0') == '0',
            'case oracle requires explicit original-metadata diagnostic execution')
    require(callable(ownership) and isinstance(prompt, str) and prompt.strip()
            and type(seed) is int and 0 <= seed < 2**63
            and isinstance(history, t.Tensor) and isinstance(noise, t.Tensor)
            and history.ndim == noise.ndim == 5 and history.shape == noise.shape
            and history.shape[0] == 1 and history.shape[1] == 48
            and history.dtype == noise.dtype == t.bfloat16
            and history.device == noise.device
            and bool(t.isfinite(history).all()) and bool(t.isfinite(noise).all())
            # Upstream creates cross-attention storage lazily on its first
            # window; a missing attribute is empty, resident storage is not.
            and p.kv_cache_clean is None and getattr(p, 'crossattn_cache', None) is None,
            'case oracle needs empty native state and sixteen bound BF16 blocks')
    identities = [latent_digest(value, t) for value in (history, noise)]
    initial_rng = t.get_rng_state().clone()
    initial_cuda_rng = t.cuda.get_rng_state(noise.device).clone() if noise.is_cuda else None
    report = dict(schema='rolling_joint_original_case_oracle_v1', passed=False,
        prompt=prompt, seed=seed, executed_device_type=noise.device.type,
        retained_blocks=RETAINED, accepted_target_blocks=list(target_blocks),
        following_blocks=list(following), input_tensor_sha256=identities,
        original_resumed_latents_required=original_tail,
        prologue=[], following=[], complete_cache_snapshots_are_diagnostic_only=True,
        source_and_checkpoint_admission_not_performed_here=True,
        physical_forward_trace_and_cuda_profile_not_admitted_here=True,
        no_quality_or_speedup_claim=True)
    candidate = reference = None
    snapshots = []
    try:
        candidate = JointCommittedSession(pipeline=p, torch=t, noise=noise,
            prompt=prompt, seed=seed, observe_step_zero=True,
            retain_hidden=True, capture_cache=True, target_suffix_blocks=width)
        accepted = history[:, :3].clone()
        for index in range(1, RETAINED):
            ownership()
            value = candidate.generate(history=accepted)
            require(t.equal(value['latent'], history[:, index*3:(index+1)*3]),
                    'native Draft prologue differs from original case at '+str(index))
            require(value['chunk_index'] == index
                    and value['joint_execution']['joint_first_window_calls'] == 0,
                    'joint execution occurred before the Target suffix')
            accepted = t.cat((accepted, value['latent']), dim=1)
            report['prologue'].append(dict(chunk=index, original_latent_exact=True))
        report['retained_prefix_oracle'] = replay_probe(pipeline=p, torch=t,
            history=accepted, condition=candidate.stream.condition, ownership=ownership)
        pointers = [(row['k'].data_ptr(), row['v'].data_ptr()) for row in p.kv_cache_clean]
        accepted = t.cat((accepted, history[:, RETAINED*3:resume*3]), dim=1)
        for index in following:
            ownership()
            ambient = t.get_rng_state().clone()
            ambient_cuda = t.cuda.get_rng_state(noise.device).clone() if noise.is_cuda else None
            value = candidate.generate(history=accepted)
            if original_tail:
                require(t.equal(value['latent'], history[:, index*3:(index+1)*3]),
                        'joint resumed latent differs from original case at '+str(index))
            require(pointers == [(row['k'].data_ptr(), row['v'].data_ptr()) for row in p.kv_cache_clean],
                    'joint handoff replaced the live native prefix storage')
            require(t.equal(ambient, t.get_rng_state())
                    and (ambient_cuda is None or t.equal(ambient_cuda, t.cuda.get_rng_state(noise.device))),
                    'joint continuation changed ambient RNG')
            expected = dict(schema='rolling_joint_window_execution_v1',
                fused_clean_blocks=width if index == resume else 0,
                joint_first_window_calls=1 if index == resume else 0,
                separate_suffix_clean_calls=0,
                native_rolling_windows_and_drains_preserved=True,
                native_first_block_clean_writes_preserved=True,
                trained_cuda_parity_established=False)
            require(value['joint_execution'] == expected, 'joint call coverage differs')
            snapshots.append(dict(latent=value['latent'].detach().cpu().clone(),
                observation=clone_native_payload(value['step_zero_observation'], t),
                caches=clone_native_payload((p.kv_cache_clean, p.crossattn_cache), t),
                noisy=candidate.stream.noisy_cache.clone(), output=candidate.stream.output.clone(),
                rng=candidate.stream.rng_state.clone(), window=candidate.stream.window))
            accepted = t.cat((accepted, value['latent']), dim=1)
        require(candidate.stream.joint_calls == 1, 'expected exactly one fixed-budget handoff')
        candidate.close(); candidate = None
        reference = RollingForcingSession(pipeline=p, torch=t, noise=noise,
            prompt=prompt, seed=seed, observe_step_zero=True,
            retain_hidden=True, capture_cache=True)
        accepted = history[:, :resume*3].clone()
        for position, index in enumerate(following):
            ownership()
            expected = snapshots[position]
            ambient = t.get_rng_state().clone()
            ambient_cuda = t.cuda.get_rng_state(noise.device).clone() if noise.is_cuda else None
            value = reference.generate(history=accepted)
            require(t.equal(ambient, t.get_rng_state())
                    and (ambient_cuda is None or t.equal(ambient_cuda, t.cuda.get_rng_state(noise.device))),
                    'native reference changed ambient RNG')
            require(value['chunk_index'] == index and t.equal(value['latent'].cpu(), expected['latent']),
                    'independent native latent differs at '+str(index))
            observation_equal(value['step_zero_observation'], expected['observation'], t)
            caches = compare_caches(expected['caches'], (p.kv_cache_clean, p.crossattn_cache), torch=t)
            require(t.equal(reference.stream.rng_state, expected['rng'])
                    and reference.stream.window == expected['window']
                    and t.equal(reference.stream.noisy_cache, expected['noisy'])
                    and t.equal(reference.stream.output, expected['output']),
                    'native continuation RNG, lookahead or drain state differs')
            report['following'].append(dict(chunk=index, original_latent_exact=True if original_tail else None,
                independent_native_latent_exact=True, layer15_hidden_exact=True,
                full_spectrum_and_rank_exact=True, accepted_history_witness_exact=True,
                full_cache_equality=caches, private_rng_exact=True,
                lookahead_and_drain_exact=True, retained_cache_storage=True, ambient_rng_preserved=True,
                latent_sha256=latent_digest(value['latent'], t)))
            snapshots[position] = None
            accepted = t.cat((accepted, value['latent']), dim=1)
        report['passed'] = True
        return report
    finally:
        snapshots.clear()
        if candidate is not None: candidate.close()
        if reference is not None: reference.close()
        require(identities == [latent_digest(value, t) for value in (history, noise)],
                'oracle mutated caller-owned accepted history or noise')
        require(t.equal(initial_rng, t.get_rng_state())
                and (initial_cuda_rng is None or t.equal(initial_cuda_rng, t.cuda.get_rng_state(noise.device))),
                'case oracle changed ambient RNG')
