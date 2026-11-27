"""Matched 2-prompt low-noise handoff trial, never a completed main-table cell.

All arms have an identical full Target root. The partial arms use a synthetic
control solely to stop the first Draft forward at fixed positions; there is no
claim that this synthetic control is a learned or calibrated eRank router.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import time
from pathlib import Path

from .correction_state import LatentHistory
from .early_protocol import StoppedDraft
from .insight2_campaign import ChunkGenerationRequest, PromptCase

ARMS = ('target', 'draft_root', 'prefix_all', 'low625_k4', 'low625_all')
PROMPT_IDS = ('018_temporal_flickering_vbl0004', '042_aesthetic_quality_vbl0719')
SEED = 20260911
CHUNKS = 30


def budget(arm):
    if arm not in ARMS:
        raise ValueError('unknown LongLive prefix trial arm')
    return 29 if arm in ('prefix_all','low625_all') else 4 if arm == 'low625_k4' else 0


def materialize(runtime, prompt, arm, renderer, *, check=lambda: None):
    """Generate a fresh accepted chain and preserve each native cache proof."""
    from .prefix_client import target_prefix, resume_prefix
    from .target_prefix import validate_draft_suffix_evidence
    from .longlive_low_noise_client import resume_low_noise
    from .longlive_low_noise import validate_evidence
    k = budget(arm)
    if arm == 'target':
        from .main_table_full_target import generate_full_target_trajectory
        generation = generate_full_target_trajectory(prompt_id=prompt.prompt_id,prompt=prompt.prompt,
            target=runtime.target,noise_provider=runtime.noise_provider,chunk_count=CHUNKS,
            duration_seconds=15,on_commit=renderer.append,safety_check=lambda phase,i:check())
        return generation.history,dict(arm=arm,target_first_update_positions=[],full_target_chunks=CHUNKS,
            target_first_updates=0,synthetic_control_is_erank_router=False,
            full_target_native_evidence=dict(generation.result),chunks=[dict(chunk_index=i,action='full_target',
                initial_noise_sha256=x.context.noise.tensor_sha256) for i,x in enumerate(generation.generated)])
    identity = hashlib.sha256(f'longlive-low625-quality:{prompt.prompt_id}:{arm}'.encode()).hexdigest()
    history = LatentHistory(identity, prompt.prompt_id, SEED)
    ranks, rows, selected = [], [], []
    for i in range(CHUNKS):
        check()
        noise = runtime.noise_provider.materialize(prompt_id=prompt.prompt_id,
            trajectory_seed=SEED, chunk_index=i)
        req = ChunkGenerationRequest(prompt=prompt, history=history, noise=noise,
            output_latent_path=f'latents/trial/{arm}/{prompt.prompt_id}/{i:03d}.safetensors',
            capture_hidden=False)
        row = {'chunk_index': i, 'initial_noise_sha256': noise.descriptor.tensor_sha256}
        if i == 0 or arm == 'target':
            if i:
                renderer.before_generate('target')
            item = runtime.target.generate(replace(req,
                target_logical_call_id=hashlib.sha256(f'{identity}:{i}'.encode()).hexdigest(),
                target_call_phase='unscoped'))
            row.update(action='full_target', native_state=item.diagnostics['native_correction_state'])
        elif arm == 'draft_root':
            renderer.before_generate('draft')
            item = runtime.draft.generate(req)
            row.update(action='draft', reuse=item.diagnostics['kv_reuse'])
            if row['reuse']['replayed_chunks'] != (1 if i == 1 else 0):
                raise ValueError('Draft control rebuilt accepted history after the root')
        else:
            renderer.before_generate('draft')
            control = {'schema':'early_erank_control_v1', 'prior_ranks':list(ranks),
                'calibration_id':'fixed-position-mechanism-control-not-erank-calibration',
                'calibration_drop_scores':[0.], 'remaining_budget':k-len(selected),
                'remaining_slots':CHUNKS-i}
            outcome = runtime.draft.generate_early(req, session_id=identity, control=control)
            stopped = isinstance(outcome, StoppedDraft)
            if stopped is not (i <= k):
                raise ValueError('fixed-position partial Target control drifted')
            proof = outcome.evidence if stopped else outcome.diagnostics
            rank = outcome.decision['effective_rank'] if stopped else outcome.effective_rank
            ranks.append(rank)
            if stopped:
                renderer.before_generate('target')
                prefix = target_prefix(runtime.target, req,
                    output_path=f'prefix/{arm}/{prompt.prompt_id}/{i:03d}.safetensors')
                if 'diagnostic_oracle' in prefix['result']:
                    raise ValueError('trial executed Target parity oracle work')
                renderer.before_generate('draft')
                if arm.startswith('low625_'):
                    noise_625=runtime.noise_provider.materialize(prompt_id=prompt.prompt_id+'-low-noise-625',
                        trajectory_seed=SEED,chunk_index=i).tensor_path
                    item=resume_low_noise(runtime.draft,outcome,prefix,renoise_path=noise_625)
                    validate_evidence(item.diagnostics)
                    action='target_first_update_draft_last_native_update'
                else:
                    noises = [runtime.noise_provider.materialize(
                        prompt_id=prompt.prompt_id+f'-suffix-noise-{step}',
                        trajectory_seed=SEED, chunk_index=i).tensor_path for step in (1,2,3)]
                    item = resume_prefix(runtime.draft, outcome, prefix, renoise_paths=noises)
                    validate_draft_suffix_evidence(item.diagnostics)
                    action='target_first_update_native_three_update_draft_suffix'
                if ('diagnostic_oracle' in item.diagnostics
                        or item.diagnostics['draft_native_schedule'] != [1000.,937.5,833.3333129882812,625.]):
                    raise ValueError('quality trial ran an oracle or a different native schedule')
                row.update(action=action,
                    target_prefix=prefix, native_suffix=dict(item.diagnostics))
                selected.append(i)
            else:
                item = outcome
                row['action'] = 'draft'
            row.update(reuse=proof['kv_reuse'], effective_rank=rank)
            if proof['kv_reuse']['replayed_chunks'] != (1 if i == 1 else 0):
                raise ValueError('partial trial rebuilt accepted history after the root')
        item.segment.verify_context(history, item.context)
        item.segment.verify_file(runtime.draft.run_dir)
        history = history.append(item.segment)
        renderer.append(history)
        rows.append(row)
    if len(selected) != k:
        raise ValueError('partial trial did not consume the fixed budget')
    return history, {'arm':arm, 'target_first_update_positions':selected, 'chunks':rows,
        'full_target_chunks':CHUNKS if arm == 'target' else 1,
        'target_first_updates':k, 'synthetic_control_is_erank_router':False}


def summarize(cells):
    from .main_table_v1 import QUALITY_DIMENSIONS
    expected={(a,p) for a in ARMS for p in PROMPT_IDS}
    if len(cells)!=len(expected) or {(x['arm'],x['prompt_id']) for x in cells}!=expected:
        raise ValueError('incomplete low-noise development comparison')
    output={}
    for arm in ARMS:
        rows=[x for x in cells if x['arm']==arm]
        if any(set(x['quality_scores'])!=set(QUALITY_DIMENSIONS)
            or any(not math.isfinite(v) or not 0<=v<=1 for v in x['quality_scores'].values())
            or not math.isfinite(x['video_ready_wall_seconds']) or x['video_ready_wall_seconds']<=0 for x in rows):
            raise ValueError('invalid low-noise scores or ready timing')
        output[arm]=dict(quality_mean_100=math.fsum(math.fsum(x['quality_scores'].values())/6 for x in rows)/len(rows)*100,
            dimensions_100={d:math.fsum(x['quality_scores'][d] for x in rows)/len(rows)*100 for d in QUALITY_DIMENSIONS},
            ready_mean_seconds=math.fsum(x['video_ready_wall_seconds'] for x in rows)/len(rows))
    for arm,value in output.items():
        value['speedup_ratio_of_mean_ready_times']=output['target']['ready_mean_seconds']/value['ready_mean_seconds']
        value['quality_delta_vs_target']=value['quality_mean_100']-output['target']['quality_mean_100']
    return output


def run(*, gate_run, legacy_gate_run, vbench_checkout, vbench_python, vbench_asset_manifest, vbench_asset_spec):
    from .accepted_stream_client import StreamDecoderProxy
    from .longlive_low_noise_worker import process_factory
    from .baseline_population import load_baseline_population, model_pair_for_draft
    from .correction_artifacts import verify_target_artifact_spec
    from .correction_runtime import launch_correction_runtime
    from .main_table_baseline_campaign import (_run_dir, _model_root, _visible_devices,
        _sha256_file, _write_new_json, verify_artifact_spec)
    from .main_table_ours_campaign import _assert_idle_pair, _assert_worker_pair_owns_gpus
    from .metadata import find_git_root, collect_git_metadata
    from .native_vae_component import campaign_model, completed_render_binding, OVERLAP_READY_SCOPE
    from .prefix_quality_gate import verify_gate as verify_legacy_gate
    from .longlive_low_noise_quality_gate import verify_gate
    from .erank_acceptance_pilot import bound_file, QUALITY_DIMENSIONS
    from .stage_timing_probe import _hardware
    from .vbench_short import OfficialVBenchScorer
    from .vbench_long import VBenchLongVideoRequest
    root, model_root, repo = _run_dir(), _model_root(), find_git_root(Path.cwd())
    gate = verify_gate(gate_run,repo)
    legacy_gate = verify_legacy_gate(legacy_gate_run,repo,native_model='longlive')
    if campaign_model('longlive_1.3b', overlap=True) != 'longlive':
        raise ValueError('trial requires the coupled native LongLive renderer')
    prompts, population = load_baseline_population(repo, model_pair=model_pair_for_draft('longlive_1.3b'))
    selected = [p for p in prompts if p['id'] in PROMPT_IDS]
    if tuple(p['id'] for p in selected) != PROMPT_IDS:
        raise ValueError('development prompt selection drifted')
    ds, ts = (repo/'experiments'/n for n in
        ('erank3-longlive-policy-v1.artifacts.json', 'insight2-krea14b.artifacts.json'))
    _, dv, _ = verify_artifact_spec(ds, method_id='draft_only', model_root=model_root, repository_root=repo)
    _, tv, _ = verify_target_artifact_spec(ts, model_root=model_root, repo_root=repo)
    devices = _visible_devices(); _assert_idle_pair(*devices)
    hardware = _hardware(devices)
    runtime = launch_correction_runtime(run_dir=root, repo_root=repo, draft_artifact_spec_path=ds,
        draft_model_id='longlive_1.3b', draft_artifact_spec_sha256=_sha256_file(ds)[0],
        draft_verified_artifacts=dv, target_artifact_spec_path=ts,
        target_artifact_spec_sha256=_sha256_file(ts)[0], target_verified_artifacts=tv,
        devices=devices, chunk_count=CHUNKS, write_full_videos=True, process_factory=process_factory,
        conditioning_prompts=[{'prompt_id':p['id'],'prompt':p['text']} for p in selected],
        shared_precomputed_conditioning=True)
    cells, requests, roots, noises = [], [], {}, {}
    try:
        runtime.evaluator.full_quality_metric_id = 'vbench_custom_six_dimension_macro_mean'
        # Prepare both independent decoder states before generating any video.
        # Their frozen Python source may be shared; neither VAE state nor model
        # KV is shared across workers.
        renderers = {role:StreamDecoderProxy(runtime.draft,worker,root,chunk_count=CHUNKS)
            for role,worker in (('draft',runtime.draft),('target',runtime.target))}
        for arm in ARMS:
            # Pure Target overlaps its VAE on the resident Draft GPU, as in its
            # matched native reference; the mixed arms use the resident Target GPU.
            renderer = renderers['draft' if arm == 'target' else 'target']
            runtime.evaluator.client = renderer
            for p in selected:
                check = lambda: _assert_worker_pair_owns_gpus(runtime=runtime,
                    draft_device=devices[0], target_device=devices[1], phase='LongLive prefix trial')
                case = PromptCase(p['id'], p['text'], 'evaluation')
                started = time.perf_counter()
                history, trajectory = materialize(runtime, case, arm, renderer, check=check)
                generation = time.perf_counter()-started
                measurement = 'll-low625-'+arm+'-'+p['id']
                prepared = runtime.evaluator.prepare_full(history=history, measurement_id=measurement)
                ready = time.perf_counter()-started
                root_hash = history.segments[0].latent_sha256
                noise_hashes = [r['initial_noise_sha256'] for r in trajectory['chunks']]
                if roots.setdefault(p['id'], root_hash) != root_hash or noises.setdefault(p['id'], noise_hashes) != noise_hashes:
                    raise ValueError('paired trial changed its Target root or initial noises')
                path = root/'provenance'/'longlive-low625-quality'/(arm+'-'+p['id']+'.json')
                _write_new_json(path, dict(trajectory, history=history.to_dict()))
                video = bound_file(root, root/prepared.video_path)
                cell = dict(arm=arm, prompt_id=p['id'], prompt=p['text'], seed=SEED,
                    generation_wall_seconds=generation, video_ready_wall_seconds=ready,
                    timing_scope=OVERLAP_READY_SCOPE, root_latent_sha256=root_hash,
                    video=video, native_trajectory=bound_file(root,path),
                    accepted_stream=completed_render_binding(renderer,history,prepared))
                cells.append(cell)
                requests.append(VBenchLongVideoRequest(video_path=root/video['path'],
                    prompt=p['text'], measurement_id=measurement))
                _write_new_json(root/'provenance'/'longlive-low625-quality'/(measurement+'-video-ready.json'),cell)
                print('LONGLIVE_LOW625_VIDEO_READY',json.dumps(cell),flush=True)
    finally:
        runtime.close()
    _assert_idle_pair(*devices)
    scorer = OfficialVBenchScorer(run_dir=root, checkout=vbench_checkout, python_executable=vbench_python,
        asset_manifest=vbench_asset_manifest, asset_spec=vbench_asset_spec, visible_device=devices[1])
    measurements = scorer.score_videos(requests)
    if set(measurements) != {r.measurement_id for r in requests}:
        raise ValueError('incomplete development quality coverage')
    for cell, request in zip(cells,requests):
        m=measurements[request.measurement_id]; scores=dict(m.dimension_scores)
        if set(scores) != set(QUALITY_DIMENSIONS):
            raise ValueError('development VBench dimensions differ')
        cell.update(quality_scores=scores, quality_evidence=bound_file(root,root/m.evidence_path))
    result = dict(schema='longlive_low625_development_trial_v1', status='succeeded',
        formal_main_table_result=False, held_out=False, prompt_selection='failure018_control042',
        gate=gate, legacy_three_step_gate=legacy_gate, git=collect_git_metadata(repo), hardware=hardware, population=population,
        arms=list(ARMS), rows=cells, comparison_summary=summarize(cells),
        target_reference_backend='persistent_native_session',timings_are_descriptive_development_only=True)
    _write_new_json(root/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('gate-run','legacy-gate-run','vbench-checkout','vbench-python','vbench-asset-manifest','vbench-asset-spec'):
        parser.add_argument('--'+name, type=Path, required=True)
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:
        print(json.dumps(run(**vars(parser.parse_args()))))
    finally:
        shutdown_owned_runtime_helpers(label='longlive-low625-quality')


if __name__ == '__main__':
    main()
