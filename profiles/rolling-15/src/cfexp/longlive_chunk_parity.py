"""Formal, one-GPU prerequisite for enabling LongLive chunk policy routing.

Compare the pinned native continuous inference with the chunk bridge before
claiming either speed or quality. Eight blocks cross the native local window.
A synthetic changed history tests branch isolation, not Target video quality.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .cross_model_longlive import build_longlive
from .incremental_draft import latent_digest
from .longlive_chunk_session import LongLiveChunkSession
from .longlive_policy_backend import LongLivePolicyBackend
from .main_table_baseline_campaign import _gpu_compute_pids, _gpu_uuid, _model_root, _run_dir, _write_new_json
from .metadata import find_git_root

PROMPT = 'a motorcycle accelerating to gain speed'
SEED = 20260911
BLOCKS = 8
PREFILL_ENV = 'CFEXP_LONGLIVE_ACCEPTED_PREFILL_PARITY'


def same_rng(torch, cpu, gpu):
    actual = torch.cuda.get_rng_state_all()
    return (torch.equal(torch.get_rng_state(), cpu) and len(actual) == len(gpu)
            and all(torch.equal(a, b) for a, b in zip(actual, gpu)))


def check_accepted_prefill(backend, *, output, scope, native, noise, ownership):
    """Compare split clean-prefix updates to the ordinary replay at handoff."""
    torch = backend.torch
    native = native.to(backend.device)
    rows = []
    for burst in (2, 4):
        prefix_count = BLOCKS - burst
        suffix = native[:, prefix_count * 3:] + .125
        observations = []
        for prefill in (False, True):
            ownership()
            session = LongLiveChunkSession(
                pipeline=backend.pipeline, torch=torch, scope_sha256=scope)
            try:
                torch.manual_seed(SEED + 201)
                torch.cuda.manual_seed_all(SEED + 201)
                prefix = native[:, :(prefix_count - 1) * 3]
                last = session.generate(prompt=PROMPT, history=prefix,
                                        noise=noise[:, :3])['latent']
                prefix = torch.cat((prefix, last), dim=1)
                history = torch.cat((prefix, suffix), dim=1)
                cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                receipts = []
                if prefill:
                    for count in range(prefix_count + 1, BLOCKS):
                        accepted = history[:, :count * 3]
                        receipt = session.prefill_accepted(prompt=PROMPT, history=accepted,
                                                           history_digest_source=accepted.cpu())
                        if (receipt['reused_chunks'] != count - 1 or receipt['replayed_chunks'] != 1
                                or receipt['history_kv_widths'] != (3,) * count
                                or receipt['generated_chunks'] or receipt['scheduler_updates']):
                            raise ValueError('LongLive prefill did not advance one native clean block')
                        receipts.append(receipt)
                if not same_rng(torch, cpu, gpu):
                    raise ValueError('LongLive clean prefix update consumed RNG')
                result = session.generate(prompt=PROMPT, history=history, noise=noise[:, 3:6])
                if result['replayed_chunks'] != (1 if prefill else burst):
                    raise ValueError('LongLive handoff replayed an unexpected suffix')
                observations.append((result['latent'].cpu(), torch.get_rng_state(),
                                     torch.cuda.get_rng_state_all(), receipts))
            finally:
                session.close()
        ordinary, prefetched = observations
        equal = (torch.equal(ordinary[0], prefetched[0])
                 and torch.equal(ordinary[1], prefetched[1])
                 and len(ordinary[2]) == len(prefetched[2])
                 and all(torch.equal(a, b) for a, b in zip(ordinary[2], prefetched[2])))
        row = {'burst_blocks': burst, 'prefix_blocks': prefix_count, 'equal_latent_and_rng': equal,
               'replayed_blocks_at_handoff': {'ordinary': burst, 'prefetched': 1},
               'ordinary_latent_sha256': latent_digest(ordinary[0], torch),
               'prefetched_latent_sha256': latent_digest(prefetched[0], torch),
               'prefill_receipts': prefetched[3]}
        _write_new_json(output / f'parity/accepted-prefill-k{burst}.json', row)
        if not equal:
            raise ValueError(f'LongLive native handoff diverged after prefetching a K{burst} prefix')
        rows.append(row)
    return rows


def run():
    output = _run_dir()
    repository = find_git_root(Path.cwd())
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 1 or not devices[0].strip():
        raise ValueError('LongLive parity requires exactly one assigned GPU')
    uuid = _gpu_uuid(devices[0])
    if _gpu_compute_pids(uuid):
        raise ValueError('assigned GPU is already occupied')
    prefill_parity = os.environ.get(PREFILL_ENV, '0')
    if prefill_parity not in ('0', '1'):
        raise ValueError('invalid LongLive accepted-prefill parity mode')
    backend = None
    session = None
    observed = None
    try:
        backend, records = build_longlive(output, repository, _model_root())
        t, p = backend.torch, backend.pipeline
        def ownership():
            if not _gpu_compute_pids(uuid) <= {os.getpid()}:
                raise ValueError('assigned GPU acquired an unrelated process')
        ownership()
        scope = hashlib.sha256(json.dumps(
            {'artifacts': records, 'native_load': backend.native_load}, sort_keys=True).encode()).hexdigest()
        _write_new_json(output/'provenance/parity-input.json', {
            'schema': 'longlive_chunk_parity_input_v1', 'prompt': PROMPT, 'seed': SEED,
            'blocks': BLOCKS, 'native_load': backend.native_load,
            'verified_artifacts': records, 'scope_sha256': scope,
            'purpose': 'native CUDA parity, not a quality comparison'})
        t.manual_seed(SEED); t.cuda.manual_seed_all(SEED)
        noise = t.randn((1, BLOCKS*3, 16, 60, 104), device=backend.device, dtype=t.bfloat16)
        cpu_rng, gpu_rng = t.get_rng_state(), t.cuda.get_rng_state_all()
        original_decode = p.vae.decode_to_pixel
        # Decoding cannot affect latent equivalence and is excluded from this probe.
        p.vae.decode_to_pixel = lambda latent, **kw: t.zeros((1,1,3,1,1),device=latent.device)
        try:
            with t.inference_mode():
                _, native = p.inference(noise=noise, text_prompts=[PROMPT], return_latents=True,
                                        profile=False, low_memory=False)
        finally:
            p.vae.decode_to_pixel = original_decode
        native_cpu_rng, native_gpu_rng = t.get_rng_state(), t.cuda.get_rng_state_all()
        native_cursor_devices = sorted({str(layer[name].device.type)
            for layer in p.kv_cache1 for name in ('global_end_index','local_end_index')})
        from .longlive_host_cursors import enabled as host_cursor_enabled, validate_evidence
        host_cursors = host_cursor_enabled()
        if host_cursors and native_cursor_devices != ['cuda']:
            raise ValueError('host cursor gate requires an unmodified CUDA-cursor native reference')
        native = native.detach().cpu().contiguous()
        p.kv_cache1 = None; p.crossattn_cache = None
        t.cuda.empty_cache()
        session = LongLiveChunkSession(pipeline=p, torch=t, scope_sha256=scope)
        t.set_rng_state(cpu_rng); t.cuda.set_rng_state_all(gpu_rng)
        rows, history = [], None
        for index in range(BLOCKS):
            ownership()
            with t.inference_mode():
                result = session.generate(prompt=PROMPT, history=history,
                                          noise=noise[:,index*3:index*3+3])
            latent = result['latent'].detach().cpu().contiguous()
            expected = native[:,index*3:index*3+3]
            equal = bool(t.equal(latent, expected))
            rows.append({'chunk':index, 'equal':equal,
                         'max_abs_difference':float((latent.float()-expected.float()).abs().max()),
                         'native_sha256':latent_digest(expected,t), 'chunk_sha256':latent_digest(latent,t),
                         'reused_chunks':result['reused_chunks'], 'replayed_chunks':result['replayed_chunks']})
            if host_cursors:
                validate_evidence(result.get('longlive_host_cursor_execution'))
                rows[-1]['longlive_host_cursor_execution'] = result['longlive_host_cursor_execution']
            _write_new_json(output/f'parity/chunk-{index:03d}.json', rows[-1])
            if not equal:
                _write_new_json(output/'result.json', {'status':'failed', 'reason':'native_latent_divergence',
                                                      'first_divergent_chunk':index, 'chunks':rows})
                raise ValueError(f'LongLive native/chunk latent mismatch at block {index}')
            history = result['latent'] if history is None else t.cat([history,result['latent']],dim=1)
        rng_equal = same_rng(t, native_cpu_rng, native_gpu_rng)
        if not rng_equal:
            raise ValueError('native/chunk final RNG state differs')
        # Compare a changed-prefix continuation in the populated branch store
        # with a fresh full-prefix reconstruction; then revisit the original.
        original = history
        changed = history.clone(); changed[:,-3:] += .125
        branch_rows = []
        branch_noise = noise[:,:3]
        for label, prefix in (('changed_history',changed),('original_history',original)):
            ownership()
            t.manual_seed(SEED+99); t.cuda.manual_seed_all(SEED+99)
            with t.inference_mode():
                actual = session.generate(prompt=PROMPT,history=prefix,noise=branch_noise)['latent'].detach().cpu()
            # A separate session shares the weights only. The first session
            # clears native aliases after each call and retains its own caches.
            fresh = LongLiveChunkSession(pipeline=p,torch=t,scope_sha256=scope)
            try:
                t.manual_seed(SEED+99); t.cuda.manual_seed_all(SEED+99)
                with t.inference_mode():
                    expected = fresh.generate(prompt=PROMPT,history=prefix,noise=branch_noise)['latent'].detach().cpu()
            finally:
                fresh.close()
            equal = bool(t.equal(actual,expected))
            branch_rows.append({'case':label,'equal':equal,
                                'max_abs_difference':float((actual.float()-expected.float()).abs().max())})
            _write_new_json(output/f'parity/{label}.json',branch_rows[-1])
            if not equal:
                _write_new_json(output/'result.json', {'status':'failed', 'reason':'branch_rebuild_divergence',
                                                      'chunks':rows,'branches':branch_rows})
                raise ValueError(f'LongLive branch/rebuild mismatch: {label}')
        # The policy-facing instrumentation must not alter latent generation.
        # Compute independent fresh-session references before attaching hooks.
        policy_cases = [('root', None), ('changed_history', changed), ('original_history', original)]
        references = []
        for label, prefix in policy_cases:
            fresh = LongLiveChunkSession(pipeline=p, torch=t, scope_sha256=scope)
            try:
                t.manual_seed(SEED+101); t.cuda.manual_seed_all(SEED+101)
                with t.inference_mode():
                    references.append(fresh.generate(prompt=PROMPT, history=prefix,
                        noise=branch_noise)['latent'].detach().cpu())
            finally:
                fresh.close()
        session.close()
        prefill_rows = []
        if prefill_parity == '1':
            with t.inference_mode():
                prefill_rows = check_accepted_prefill(
                    backend, output=output, scope=scope, native=native,
                    noise=noise, ownership=ownership)
        observed = LongLivePolicyBackend(native_backend=backend, scope_sha256=scope, chunk_count=120)
        cpu_binding = os.environ.get('CFEXP_LONGLIVE_CPU_HISTORY_BINDING', '0')
        if cpu_binding not in ('0', '1'):
            raise ValueError('invalid LongLive CPU binding gate mode')
        if cpu_binding == '1':
            from .cpu_history_binding import install_longlive
            install_longlive(observed)
        policy_rows = []
        for (label, prefix), expected in zip(policy_cases, references):
            ownership()
            index = 0 if prefix is None else int(prefix.shape[1]) // 3
            measurement = observed.generate_chunk(prompt=PROMPT,
                history_latents=(prefix.detach().cpu() if cpu_binding == '1' and prefix is not None else prefix),
                initial_noise=branch_noise, internal_seed=SEED+101, chunk_index=index,
                capture_hidden=True, history_kv_widths=(3,) * index)
            actual = measurement['latent']
            equal = bool(t.equal(actual, expected))
            row = {'case': label, 'equal': equal,
                   'max_abs_difference': float((actual.float()-expected.float()).abs().max()),
                   'rank': measurement['rank'], 'rank_evidence': measurement['rank_evidence'],
                   'forward_trace': measurement['forward_trace'], 'kv_reuse': measurement['kv_reuse'],
                   'sampled_hidden_shape': list(measurement['sampled_hidden'].shape),
                   'full_hidden_shape': list(measurement['full_hidden'].shape)}
            if cpu_binding == '1':
                from .cpu_history_binding import validate_longlive_evidence
                proof = measurement.get('history_binding_execution')
                validate_longlive_evidence(proof, history_chunks=index,
                    tensor_digests=measurement['kv_reuse']['history_tensor_sha256'])
                row['history_binding_execution'] = proof
            if host_cursors:
                validate_evidence(measurement.get('longlive_host_cursor_execution'))
                row['longlive_host_cursor_execution'] = measurement['longlive_host_cursor_execution']
            _write_new_json(output/f'parity/policy-{label}.json', row)
            policy_rows.append(row)
            if not equal:
                raise ValueError(f'LongLive policy instrumentation changed the latent: {label}')
        result = {'schema':'longlive_chunk_parity_result_v1', 'status':'succeeded',
                  'chunks':rows,'branches':branch_rows, 'native_load':backend.native_load,
                  'policy_observations': policy_rows,
                  'native_rng_equal': rng_equal,
                  'host_cursor_candidate': host_cursors,
                  'native_reference_cursor_devices': native_cursor_devices,
                  'claim_scope':'one prompt, eight native blocks plus synthetic history branches; no quality result'}
        if prefill_rows:
            result['accepted_prefill_parity'] = prefill_rows
        _write_new_json(output/'result.json',result)
        return result
    finally:
        if observed is not None:
            observed.close()
        if session is not None:
            session.close()
        if backend is not None:
            backend.close()


def main():
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:
        run()
        return 0
    finally:
        # Reap owned compiler/tracker helpers before the leader exits. Keep
        # the formal runner's residual-process guard unchanged.
        shutdown_owned_runtime_helpers(label="longlive-parity-controller")


if __name__ == '__main__':
    raise SystemExit(main())
