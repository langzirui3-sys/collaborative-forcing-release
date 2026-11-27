"""Measure a causal batched Self-Forcing handoff on one sealed latent fixture."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .correction_state import LatentHistory
from .main_table_baseline_campaign import (
    _gpu_compute_pids, _gpu_uuid, _model_root, _run_dir, _write_new_json,
)
from .main_table_merge import _load_frozen_index
from .metadata import find_git_root
from .stage_timing import _unsealed_directory
from .sf_joint_handoff import (
    active_handoff, batch_attention, joint_first_step, native_timestep_shapes, window_bounds,
)

ARMS = ('sequential', 'batched_prefill', 'fused_first_step')
AUDIT_CONTROLS = ('sequential_repeat', 'sequential_width3', 'sequential_custom_attention')


def _fixture(source, index_sha, load_file):
    frozen, marker = _load_frozen_index(source)
    if marker['artifact_index_sha256'] != index_sha:
        raise ValueError('sealed fixture index differs')
    attempt = f"attempts/{marker['attempt_id']}"
    result = json.loads(frozen.file(f'{attempt}/result.json', label='fixture result').read_bytes())
    row, = result['rows']
    preflight = json.loads(frozen.file(f'{attempt}/provenance/preflight.json',
                                       label='fixture preflight').read_bytes())
    target_states = row['native_cache_audit']['target_native_states']
    burst = sorted(int(i) for i in target_states if int(i))
    if len(burst) != 4 or burst != list(range(burst[0], burst[0] + 4)):
        raise ValueError('fixture must contain a four-block Target burst')
    start, next_index = burst[0], burst[-1] + 1
    cell = f"{attempt}/cells/{row['arm']}"
    checkpoints = [p for p in frozen.entries
                   if p.startswith(cell + '/state/rollouts/') and p.endswith(f'/chunk-{next_index:02d}.json')]
    checkpoint, = checkpoints
    path = frozen.file(checkpoint, label='Draft resume checkpoint')
    data = json.loads(path.read_bytes())
    full_history = LatentHistory.from_dict(data['history'])
    history = full_history.prefix_before(next_index)
    generated = data['generated']
    from .insight2_campaign import _generated_from_dict
    draft = _generated_from_dict(generated)
    draft.segment.verify_context(history, draft.context)
    if (draft.segment != full_history.segments[-1] or draft.context.role != 'draft'
            or tuple(history.segments[i].producer_role for i in burst) != ('target',) * 4):
        raise ValueError('fixture accepted route differs')
    tensors = []
    for segment in history.segments:
        segment.verify_file(source / cell)
        tensors.append(load_file(str(source / cell / segment.latent_path))['latent'])
    from .correction_runtime import DeterministicNoiseProvider
    noise = DeterministicNoiseProvider(run_dir=_run_dir()).materialize(
        prompt_id=history.prompt_id, trajectory_seed=history.trajectory_seed,
        chunk_index=next_index)
    if noise.descriptor != draft.context.noise:
        raise ValueError('fresh deterministic noise differs from sealed resume input')
    return (row, preflight, history, tensors, noise, start, next_index,
            dict(run_id=marker['run_id'], index_sha256=index_sha, checkpoint=checkpoint,
                 accepted_history_sha256=history.manifest_sha256,
                 noise_sha256=noise.descriptor.tensor_sha256))


def _kv_snapshot(pipeline):
    snapshots = []
    for layer, cross in zip(pipeline.kv_cache1, pipeline.crossattn_cache, strict=True):
        end = int(layer['local_end_index'].item())
        tensors = [layer[k][:, :end] for k in ('k', 'v')] + [cross[k] for k in ('k', 'v')]
        snapshots.append(((int(layer['global_end_index'].item()), end, bool(cross['is_init'])),
                          tuple(x.detach().cpu() for x in tensors)))
    return snapshots


def _kv_diff(pipeline, reference, torch):
    from .early_draft_diagnostic import compare

    errors = []
    cursor_exact = True
    for layer, cross, (expected, tensors) in zip(
            pipeline.kv_cache1, pipeline.crossattn_cache, reference, strict=True):
        end = int(layer['local_end_index'].item())
        cursor_exact &= (int(layer['global_end_index'].item()), end, bool(cross['is_init'])) == expected
        values = [layer[k][:, :end] for k in ('k', 'v')] + [cross[k] for k in ('k', 'v')]
        errors.extend(compare(value, ref.to(value.device), torch) for value, ref in zip(values, tensors, strict=True))
    return dict(cursors_exact=cursor_exact, tensor_count=len(errors),
                tensors_exact=sum(e['exact'] for e in errors),
                max_abs=max(e['max_abs'] for e in errors),
                max_relative_l2=max(e['relative_l2'] for e in errors))


@contextmanager
def trace_first_history_block(model, compare, torch, reference, destination, tokens):
    """Compare the first Target block before it can attend to any new block."""
    handles = []

    def record(name, tensor):
        value = tensor.detach().cpu().contiguous()
        if destination is None:
            reference[name] = value.clone()
        else:
            destination[name] = compare(value, reference[name], torch)

    def record_patch(_module, _args, output):
        record('patch_embedding', output[:, :, :3])

    def record_modulation(_module, _args, kwargs):
        modulation = kwargs['e']
        if modulation.shape[1] == 1:
            modulation = modulation.expand(-1, 3, -1, -1)
        record('time_modulation', modulation[:, :3])

    def record_attention_input(_module, args):
        record('block_00_attention_input', args[0][:, :tokens])

    def record_attention_output(_module, _args, output):
        record('block_00_attention_output', output[:, :tokens])

    def block_output(name):
        def hook(_module, _args, output):
            record(name, output[:, :tokens])
        return hook

    try:
        handles.append(model.patch_embedding.register_forward_hook(record_patch))
        first = model.blocks[0]
        handles.append(first.register_forward_pre_hook(record_modulation, with_kwargs=True))
        handles.append(first.self_attn.register_forward_pre_hook(record_attention_input))
        handles.append(first.self_attn.register_forward_hook(record_attention_output))
        for index, block in enumerate(model.blocks):
            handles.append(block.register_forward_hook(block_output(f'block_{index:02d}_output')))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _experiment(backend, prompt, history, widths, noise, start, next_index, *, repeats,
                audit=False, native_time=False, integrated=False):
    from .early_draft_diagnostic import compare
    torch, p = backend.torch, backend.pipeline
    if p.generator.uniform_timestep or p.args.context_noise != 0:
        raise ValueError('expected native per-frame timesteps and zero context noise')
    dtype, device = noise.dtype, noise.device
    tokens = p.frame_seq_length * 3
    if widths[start:] != (1,) * 4 or len(p.denoising_step_list) != 4:
        raise ValueError('fixture does not use the native four-step/B1 handoff')
    rows, parity, reference, traces, trace_reference = [], {}, {}, {}, {}
    schedule = tuple(float(step) if integrated else int(step) for step in p.denoising_step_list)
    zero = {width: torch.zeros((1, width), device=device, dtype=torch.int64) for width in (1, 3)}
    step_tensor = {
        step: (torch.ones((1, 3), device=device, dtype=torch.int64) * native_step
               if integrated else torch.full((1, 3), step, device=device, dtype=torch.int64))
        for step, native_step in zip(schedule, p.denoising_step_list, strict=True)
    }
    conditional = p.text_encoder(text_prompts=[prompt])

    def forward(video, timestep, chunk):
        generator = p.generator if integrated else p.generator.forward
        return generator(noisy_image_or_video=video, conditional_dict=conditional,
            timestep=timestep, kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
            current_start=chunk * tokens)[1]

    def trial(arm, check):
        calls = 0
        reference_arm = arm == 'sequential'
        trace = traces.setdefault(arm, {}) if audit and check and not reference_arm else None

        def call(video, timestep, chunk):
            nonlocal calls
            calls += 1
            capture = audit and check and calls == 1
            with (trace_first_history_block(p.generator.model, compare, torch,
                                            trace_reference, trace, tokens)
                  if capture else nullcontext()):
                return forward(video, timestep, chunk)

        p._initialize_kv_cache(batch_size=1, dtype=dtype, device=device)
        p._initialize_crossattn_cache(batch_size=1, dtype=dtype, device=device)
        handles = []
        witness = None
        try:
            if integrated:
                from .self_forcing_frame_adapter import (
                    EFFECTIVE_RANK_LAYER, LAYER_INDICES, StepZeroCapture, sampled_kv_cache_witness,
                )
                backend._active_capture = StepZeroCapture(
                    layers=LAYER_INDICES, target_timestep=1000,
                    sample_hidden=backend._sample_hidden, rank_sample=backend._rank_sample,
                    retain_rank_layer=EFFECTIVE_RANK_LAYER, policy_rank_only=True,
                )

                def sample_clean():
                    nonlocal witness
                    if any(layer['global_end_index'].item() != next_index * tokens
                           for layer in p.kv_cache1):
                        raise ValueError('witness does not describe the accepted prefix')
                    witness, _ = sampled_kv_cache_witness(p.kv_cache1, p.crossattn_cache)

                def before(_module, _args, kwargs):
                    if active_handoff() is None and kwargs['current_start'] == next_index * tokens:
                        if backend._active_capture._active:
                            sample_clean()

                def after(_module, _args, _output):
                    if active_handoff() is not None:
                        sample_clean()

                handles.append(p.generator.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(p.generator.register_forward_hook(after))
            for i in range(start):
                forward(history[:, i*3:(i+1)*3], zero[widths[i]], i)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            event_start, event_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin = time.perf_counter()
            event_start.record()
            suffix = history[:, start*3:next_index*3]
            if reference_arm or arm in AUDIT_CONTROLS:
                with (batch_attention(p.generator.model)
                      if arm == 'sequential_custom_attention' else nullcontext()):
                    for i in range(start, next_index):
                        width = 3 if arm == 'sequential_width3' else widths[i]
                        call(history[:, i*3:(i+1)*3], zero[width], i)
                first = call(noise, step_tensor[schedule[0]], next_index)
            elif integrated and arm == 'fused_first_step':
                calls += 1
                first = joint_first_step(p, torch=torch, history=suffix, noise=noise,
                    conditional=conditional, step=p.denoising_step_list[0],
                    history_start=start * tokens, draft_start=next_index * tokens)
            else:
                mixed = arm == 'fused_first_step'
                video = torch.cat((suffix, noise), dim=1) if mixed else suffix
                timestep = torch.zeros((1, video.shape[1]), device=device, dtype=torch.int64)
                if mixed:
                    timestep[:, -3:] = schedule[0]
                model = p.generator.model
                with batch_attention(model), (
                    native_timestep_shapes(model, suffix.shape[1], mixed=mixed)
                    if native_time else nullcontext()
                ):
                    output = call(video, timestep, start)
                first = output[:, -3:].contiguous() if mixed else call(noise, step_tensor[schedule[0]], next_index)
                del output, video, timestep
            event_end.record()
            event_end.synchronize()
            elapsed = time.perf_counter() - begin
            timing = dict(model_forward_calls=calls, wall_seconds=elapsed, cuda_stream_ms=event_start.elapsed_time(event_end),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            if integrated:
                event, = backend._active_capture.events
                if witness is None or event.current_start != next_index * tokens:
                    raise ValueError('integrated handoff omitted its rank or clean KV evidence')
                timing.update(clean_kv_witness_sha256=witness, rank=event.ranks[EFFECTIVE_RANK_LAYER])
            if check:
                if reference_arm:
                    reference.update(first=first.detach().cpu())
                    if not integrated:
                        reference['first_kv'] = _kv_snapshot(p)
                else:
                    parity[arm] = dict(first_output=compare(first, reference['first'].to(device), torch))
                    if not integrated:
                        parity[arm]['first_kv'] = _kv_diff(p, reference['first_kv'], torch)
            clean = first
            rng = torch.Generator(device=device).manual_seed(99173)
            for step in schedule[1:]:
                flat = clean.flatten(0, 1)
                innovation = torch.randn(flat.shape, generator=rng, device=device, dtype=dtype)
                noisy = p.scheduler.add_noise(flat, innovation, step_tensor[step].flatten()).unflatten(0, clean.shape[:2])
                clean = forward(noisy, step_tensor[step], next_index)
            forward(clean, zero[3], next_index)
            if check:
                if reference_arm:
                    reference.update(final=clean.detach().cpu(), final_kv=_kv_snapshot(p))
                else:
                    parity[arm].update(final_output=compare(clean, reference['final'].to(device), torch),
                                       committed_kv=_kv_diff(p, reference['final_kv'], torch))
            torch.cuda.synchronize(device)
            return timing
        finally:
            for handle in handles:
                handle.remove()
            if integrated:
                backend._active_capture = None
            p.kv_cache1 = p.crossattn_cache = None

    # The adapter's hooks require uniform timesteps and a single Draft chunk.
    # The fixed-input experiment compares native tensors directly in every arm.
    for handle in backend._hook_handles:
        handle.remove()
    backend._hook_handles.clear()
    if integrated:
        from .sf_joint_handoff_adapter import install_capture_hooks
        install_capture_hooks(backend, p.generator.model.blocks)
    with torch.inference_mode():
        order = (ARMS[0], *AUDIT_CONTROLS, *ARMS[1:]) if audit else ARMS
        for arm in order:
            trial(arm, check=True)
        reference.clear()
        trace_reference.clear()
        for repetition in range(repeats):
            order = ARMS[repetition % len(ARMS):] + ARMS[:repetition % len(ARMS)]
            for arm in order:
                rows.append(dict(arm=arm, repetition=repetition, **trial(arm, check=False)))
    return rows, parity, schedule, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--source-index-sha256', required=True)
    parser.add_argument('--draft-artifact-spec', type=Path, required=True)
    parser.add_argument('--audit-divergence', action='store_true')
    parser.add_argument('--native-timestep-shapes', action='store_true')
    parser.add_argument('--integrated', action='store_true')
    args = parser.parse_args()
    from safetensors.torch import load_file
    from .erank_frame_campaign import verify_artifact_spec
    from .self_forcing_frame_adapter import OfficialSelfForcingBackend, _materialize_source
    from .stage_timing_probe import _hardware

    root, repo = _unsealed_directory(_run_dir()), find_git_root(Path.cwd())
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.isdecimal() or os.environ.get('CFEXP_DRAFT_KV_REUSE') != '1':
        raise ValueError('probe requires one physical GPU and explicit native cache reuse')
    uuid = _gpu_uuid(gpu)
    if _gpu_compute_pids(uuid):
        raise ValueError('probe requires an idle reserved GPU')
    row, source_preflight, history, tensors, noise, start, next_index, fixture = _fixture(
        args.source_run, args.source_index_sha256, load_file)
    spec, verified, module = verify_artifact_spec(args.draft_artifact_spec,
        model_root=_model_root(), repo_root=repo)
    if module != 'cfexp.self_forcing_frame_adapter' or source_preflight['seed'] != history.trajectory_seed:
        raise ValueError('Self-Forcing fixture model or seed differs')
    artifacts = {record['name']: Path(record['resolved_path']) for record in verified}
    source, site = _materialize_source(run_dir=root, worker_index=0, artifacts=artifacts,
        config=spec['adapter_config'], scratch_suffix='sf-handoff-fusion')
    backend = OfficialSelfForcingBackend(source_root=source, python_site=site, artifacts=artifacts,
        config=spec['adapter_config'], device_name='cuda:0', chunk_count=30)
    try:
        if _gpu_compute_pids(uuid) != {os.getpid()}:
            raise ValueError('unexpected process on the leased GPU')
        torch = backend.torch
        widths = tuple(1 if segment.producer_role == 'target' else 3 for segment in history.segments)
        latent_history = torch.cat(tensors, dim=1).to(backend.device)
        initial_noise = load_file(str(root / noise.tensor_path))['noise'].to(backend.device)
        if args.integrated and (args.audit_divergence or not args.native_timestep_shapes):
            raise ValueError('integrated probe requires the native timestep shapes')
        rows, parity, steps, traces = _experiment(backend, row['prompt'], latent_history, widths,
            initial_noise, start, next_index, repeats=12 if args.integrated else 3,
            audit=args.audit_divergence, native_time=args.native_timestep_shapes,
            integrated=args.integrated)
        if _gpu_compute_pids(uuid) != {os.getpid()}:
            raise ValueError('GPU ownership changed during the probe')
    finally:
        backend.close()
    from statistics import median

    if any(not math.isfinite(r['wall_seconds']) or r['wall_seconds'] <= 0
           or r['model_forward_calls'] != (5, 2, 1)[ARMS.index(r['arm'])] for r in rows):
        raise ValueError('handoff timing or executed forward count differs')
    if args.integrated and (
        len({(r['rank'], r['clean_kv_witness_sha256']) for r in rows}) != 1
        or any(not row['first_output']['exact'] or not row['final_output']['exact']
               or row['committed_kv']['tensors_exact'] != row['committed_kv']['tensor_count']
               or not row['committed_kv']['cursors_exact'] for row in parity.values())
    ):
        raise ValueError('integrated handoff changed rank, accepted KV, or latent')
    _write_new_json(root / 'result.json', dict(schema='sf_handoff_fusion_probe_v1',
        scope='fixed_history_draft_handoff_through_first_denoise_excludes_setup',
        source=fixture, hardware=_hardware((gpu,)), prefix_chunks=start,
        resumed_chunk=next_index, history_kv_widths=widths, schedule=steps,
        target_suffix_chunks=4, model_forward_counts=dict(zip(ARMS, (5, 2, 1), strict=True)),
        attention_window_frames=21, attention_sink_frames=0, rows=rows, parity=parity,
        **({'first_history_block_trace': traces} if args.audit_divergence else {}),
        **({'native_timestep_shapes': True} if args.native_timestep_shapes else {}),
        **({'integrated_joint_forward': True, 'timing_includes_rank_and_kv_witness': True}
           if args.integrated else {}),
        median_wall_seconds={arm: median(r['wall_seconds'] for r in rows if r['arm'] == arm)
                             for arm in ARMS}))


if __name__ == '__main__':
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:
        main()
    finally:
        shutdown_owned_runtime_helpers(label='sf-handoff-fusion-probe')
