"""Pure Krea Target with a resident native video decoder.

Times one contiguous trajectory-start through prepare_full interval on an
otherwise idle two-GPU pair. No sum of separated generation/decode intervals.
Model load, precomputed text conditioning and scoring are outside this span.
The selected native decoder configuration is matched to the mixed arm.
"""
import argparse
import json
import os
from pathlib import Path
import time

from .paper_population import generation_seed, population_label

from .accepted_stream import TIMING_SCOPE as SF_TIMING_SCOPE
from .accepted_stream_client import enabled
from .baseline_population import load_baseline_population, model_pair_for_draft
from .correction_artifacts import verify_target_artifact_spec, target_artifact_spec_sha256
from .correction_runtime import launch_correction_runtime
from .duration_sweep import duration_profile
from .erank_acceptance_pilot import OfficialVBenchScorer, bound_file, QUALITY_DIMENSIONS
from .rolling_campaign import verify_draft_spec as verify_rolling_spec, prepare_launch
from .main_table_baseline_campaign import verify_artifact_spec as verify_native_spec
from .main_table_baseline_campaign import _run_dir, _model_root, _visible_devices, _sha256_file, _write_new_json
from .main_table_full_target import generate_full_target_trajectory
from .main_table_ours_campaign import _assert_idle_pair, _assert_worker_pair_owns_gpus
from .metadata import find_git_root, collect_git_metadata
from .native_video_ready_draft import MODELS, SELF_FORCING
from .native_vae_component import (OVERLAP_READY_SCOPE, SERIAL_READY_SCOPE,
    campaign_model, completed_render_binding)
from .vbench_long import VBenchLongVideoRequest
from .stage_timing import _unsealed_directory
from .stage_timing_probe import _hardware

TIMING_SCOPE = SERIAL_READY_SCOPE


def run(*, draft_artifact_spec, target_artifact_spec, vbench_checkout, vbench_python,
        vbench_asset_manifest, vbench_asset_spec, native_draft_model_id, duration_seconds=None,
        smoke_prompt_id=None, overlap_native_vae=False, pure_draft=False, random_control=False):
    if duration_seconds not in (15, 30, 60):
        raise ValueError('native video-ready reference requires an explicit supported duration')
    if native_draft_model_id == SELF_FORCING:
        flags = ('CFEXP_DRAFT_KV_REUSE', 'CFEXP_RENDER_TARGET_PRIORITY', 'CFEXP_VIDEO_ONLY_DECODE')
        from .native_vae_component import selected_model
        from .video_encoding_policy import SERIAL, policy
        if (not overlap_native_vae or not enabled() or selected_model() is not None
                or policy() != SERIAL or any(os.environ.get(flag) != '1' for flag in flags)):
            raise ValueError('Self-Forcing reference requires native Draft KV and its accepted stream')
        native_render, timing_scope = 'self-forcing', SF_TIMING_SCOPE
    else:
        native_render = campaign_model(native_draft_model_id, overlap=overlap_native_vae)
        timing_scope = OVERLAP_READY_SCOPE if native_render else TIMING_SCOPE
    from . import native_single_gpu
    single_gpu = native_single_gpu.validate(native_draft_model_id, native_render, enforce_idle=True)
    from .paper_population import validate_execution
    validate_execution(single_gpu=single_gpu, seconds=duration_seconds)
    if single_gpu:
        timing_scope = native_single_gpu.TIMING_SCOPE
    if type(pure_draft) is not bool or (pure_draft and native_render is None):
        raise ValueError('pure Draft reference requires the coupled native renderer')
    if type(random_control) is not bool or (
            random_control and (pure_draft or native_draft_model_id != SELF_FORCING)):
        raise ValueError('native Random is a Self-Forcing mixed control')
    role = 'random' if random_control else 'draft' if pure_draft else 'target'
    if (native_draft_model_id not in MODELS
            or (native_render is None and enabled())):
        raise ValueError('native Target reference requires an explicitly matched renderer')
    root, models = _unsealed_directory(_run_dir()), _model_root()
    repo = find_git_root(Path.cwd())
    if (root/'result.json').exists():
        raise ValueError('native Target requires a fresh run')
    if native_render:
        prompts, population = load_baseline_population(repo, model_pair=model_pair_for_draft(native_draft_model_id))
    else:
        prompts, population = load_baseline_population(repo)
    profile = duration_profile(15 if duration_seconds is None else duration_seconds)
    benchmark = 'VBench-Long' if profile.nominal_seconds == 60 else 'VBench'
    population_name = population_label('main24_method_development_not_held_out')
    if duration_seconds is None:
        if smoke_prompt_id is not None:
            raise ValueError('smoke selection requires an explicit duration')
        prompts = prompts[:12]
        population_name = 'fixed_first12_development_not_held_out'
    if smoke_prompt_id is not None:
        prompts = [p for p in prompts if p['id'] == smoke_prompt_id]
        if len(prompts) != 1:
            raise ValueError('smoke prompt is outside the frozen population')
        population_name = 'main24_smoke'
    if native_draft_model_id == 'rolling_forcing_1.3b':
        _, dv, dm = verify_rolling_spec(draft_artifact_spec, model_root=models, repo_root=repo)
        expected = 'cfexp.rolling_policy_worker'
    elif native_draft_model_id == SELF_FORCING:
        from .erank_frame_campaign import verify_artifact_spec as verify_sf_spec
        _, dv, dm = verify_sf_spec(draft_artifact_spec, model_root=models, repo_root=repo)
        expected = 'cfexp.self_forcing_frame_adapter'
    else:
        _, dv, dm = verify_native_spec(draft_artifact_spec, method_id='draft_only',
                                      model_root=models, repository_root=repo)
        expected = 'cfexp.longlive_adapter'
    _, tv, tm = verify_target_artifact_spec(target_artifact_spec, model_root=models, repo_root=repo)
    if (dm, tm) != (expected, 'cfexp.krea_chunk_backend'):
        raise ValueError('native Target backend identity differs')
    devices = native_single_gpu.devices(single_gpu, _visible_devices)
    _assert_idle_pair(*devices)
    hardware = _hardware(devices[:1] if single_gpu else devices)
    context_path = root/'provenance'/(role+'-native')/'generation-context.json'
    _write_new_json(context_path, dict(schema='native_'+role+'_generation_context_v1',
        hardware=hardware, code_revision=collect_git_metadata(repo)['sha'],
        population=population_name, population_binding=population,
        benchmark=benchmark, duration=profile.metadata(), timing_scope=timing_scope,
        native_decoder_model_id=native_draft_model_id))
    context_binding = bound_file(root, context_path)
    conditioning = [{'prompt_id':p['id'], 'prompt':p['text']} for p in prompts]
    kwargs = ({'rolling': prepare_launch(run_dir=root, repo_root=repo, model_root=models,
        prompts=conditioning, seeds=(generation_seed(),), chunk_count=profile.chunk_count)}
        if native_draft_model_id == 'rolling_forcing_1.3b' else {})
    if native_render:
        from .accepted_stream_worker import process_factory
        kwargs['process_factory'] = process_factory
    if single_gpu:
        kwargs['allow_shared_device'] = True
    started = time.perf_counter()
    runtime = launch_correction_runtime(run_dir=root, repo_root=repo,
        draft_artifact_spec_path=draft_artifact_spec, draft_model_id=native_draft_model_id,
        draft_artifact_spec_sha256=_sha256_file(draft_artifact_spec)[0], draft_verified_artifacts=dv,
        target_artifact_spec_path=target_artifact_spec,
        target_artifact_spec_sha256=target_artifact_spec_sha256(target_artifact_spec),
        target_verified_artifacts=tv, devices=devices, write_full_videos=True, chunk_count=profile.chunk_count,
        conditioning_prompts=conditioning, shared_precomputed_conditioning=True, **kwargs)
    cells, requests = [], []
    try:
        renderer = None
        if native_render:
            from .accepted_stream_client import StreamDecoderProxy
            renderer = StreamDecoderProxy(runtime.draft,
                runtime.target if pure_draft or random_control else runtime.draft, root, chunk_count=profile.chunk_count)
            runtime.evaluator.client = renderer
        runtime.evaluator.full_quality_metric_id = ('vbench_long_custom_six_dimension_macro_mean'
            if benchmark == 'VBench-Long' else 'vbench_custom_six_dimension_macro_mean')
        load_seconds = time.perf_counter()-started
        for prompt in prompts:
            _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0],
                target_device=devices[1], phase='native '+role+' '+prompt['id'])
            started = time.perf_counter()
            if random_control:
                from .native_video_ready_random import generate
                generation = generate(draft=runtime.draft, target=runtime.target,
                    noise_provider=runtime.noise_provider, prompt_id=prompt['id'],
                    prompt=prompt['text'], profile=profile, renderer=renderer, run_dir=root)
            elif pure_draft:
                from .native_video_ready_draft import generate
                generation = generate(draft=runtime.draft, noise_provider=runtime.noise_provider,
                    prompt_id=prompt['id'], prompt=prompt['text'], chunk_count=profile.chunk_count,
                    model=native_draft_model_id, renderer=renderer, run_dir=root)
            else:
                generation = generate_full_target_trajectory(prompt_id=prompt['id'], prompt=prompt['text'],
                    target=runtime.target, noise_provider=runtime.noise_provider,
                    chunk_count=profile.chunk_count, duration_seconds=profile.nominal_seconds,
                    **({'on_commit': renderer.append} if renderer is not None else {}))
            generation_seconds = time.perf_counter()-started
            prepared = runtime.evaluator.prepare_full(history=generation.history,
                measurement_id='stream-'+role+'-'+prompt['id'])
            ready_seconds = time.perf_counter()-started
            native = dict(generation.result)
            _write_new_json(root/'provenance'/(role+'-native')/
                (prompt['id']+'.json'), dict(native, history=generation.history.to_dict()))
            video = bound_file(root, root/prepared.video_path)
            cell = {'arm':role, 'prompt_id':prompt['id'], 'prompt':prompt['text'],
                'seed':generation_seed(), 'video_ready_wall_seconds':ready_seconds,
                'generation_wall_seconds':generation_seconds,
                'decode_encode_and_prepare_wall_seconds':ready_seconds-generation_seconds,
                'post_generation_drain_and_prepare_wall_seconds':ready_seconds-generation_seconds,
                'decode_encode_and_prepare_wall_seconds_semantics':'tail_after_generation_not_total_render_work',
                'timing_scope':timing_scope, 'video':video,
                'root_latent_sha256':generation.history.segments[0].latent_sha256,
                'native_decoder_model_id': native_draft_model_id, 'native_generation':native,
                'model_load_and_conditioning_wall_seconds':load_seconds,
                'decode_evidence':bound_file(root, root/prepared.decode_evidence_path)}
            if single_gpu:
                cell['decode_encode_and_prepare_wall_seconds_semantics'] = 'gpu_decode_and_encoder_drain_after_generation'
                cell['accepted_video_schedule'] = 'decode_after_generation'
            if renderer is not None:
                cell['accepted_stream'] = completed_render_binding(renderer, generation.history, prepared)
            # Keep completed timing/video evidence even if later scoring fails.
            # This is written after the measured interval, like native provenance.
            ready_path = root/'provenance'/(role+'-native')/(prompt['id']+'-video-ready.json')
            _write_new_json(ready_path, dict(schema='native_'+role+'_video_ready_record_v1',
                generation_context=context_binding, **cell))
            cell['video_ready_evidence'] = bound_file(root, ready_path)
            cells.append(cell)
            requests.append(VBenchLongVideoRequest(video_path=root/video['path'],
                prompt=prompt['text'], measurement_id='stream-'+role+'-'+prompt['id']))
    finally:
        runtime.close()
    _assert_idle_pair(*devices)
    scorer_factory = OfficialVBenchScorer
    if benchmark == 'VBench-Long':
        from .vbench_long import OfficialVBenchLongScorer
        scorer_factory = OfficialVBenchLongScorer
    scorer = scorer_factory(run_dir=root, checkout=vbench_checkout, python_executable=vbench_python,
        asset_manifest=vbench_asset_manifest, asset_spec=vbench_asset_spec, visible_device=devices[1])
    measurements = scorer.score_videos(requests)
    if set(measurements) != {r.measurement_id for r in requests}:
        raise ValueError('incomplete native Target quality coverage')
    rows = []
    for cell, request in zip(cells, requests):
        value = measurements[request.measurement_id]
        scores = dict(value.dimension_scores)
        if set(scores) != set(QUALITY_DIMENSIONS) or bound_file(root,root/cell['video']['path']) != cell['video']:
            raise ValueError('Target quality/video binding changed')
        rows.append(dict(cell, quality_scores=scores,
            quality_evidence=bound_file(root,root/value.evidence_path)))
    result = {'schema': ('native_overlap_video_ready_random_v1' if random_control else
                        'native_overlap_video_ready_draft_v1' if pure_draft else
                        'native_overlap_video_ready_target_v1' if native_render
                         else 'native_serial_video_ready_target_v1'), 'status':'succeeded',
        'formal_speedup_claim':False, 'population':population_name, 'benchmark':benchmark, 'duration':profile.metadata(),
        'population_binding':population, 'code_revision':collect_git_metadata(repo)['sha'],
        'hardware':hardware, 'timing_scope':timing_scope, 'rows':rows, 'row_count':len(rows)}
    if native_render:
        result['accepted_stream_count'] = len(rows)
    if single_gpu:
        result['accepted_video_schedule'] = 'decode_after_generation'
    if pure_draft:
        result.update(method_id='draft_only', draft_model_id=native_draft_model_id,
            initialization='pure_native_draft_no_target_root',
            draft_model_calls_per_video=profile.chunk_count, target_model_calls_per_video=0)
    if random_control:
        result.update(method_id='random', draft_model_id=native_draft_model_id,
            initialization='target_root_then_fixed_random_positions', routing_repeat=0,
            draft_model_calls_per_video=profile.chunk_count-profile.proposal_budget-1,
            target_model_calls_per_video=profile.proposal_budget+1)
    _write_new_json(root/'result.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('draft-artifact-spec','target-artifact-spec','vbench-checkout','vbench-python',
                 'vbench-asset-manifest','vbench-asset-spec'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--duration-seconds', type=int, choices=(15, 30, 60))
    parser.add_argument('--smoke-prompt-id')
    parser.add_argument('--overlap-native-vae', action='store_true')
    parser.add_argument('--random-control', action='store_true')
    parser.add_argument('--native-draft-model-id', required=True, choices=MODELS)
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:
        print(json.dumps(run(**vars(parser.parse_args()))))
    finally:
        shutdown_owned_runtime_helpers(label='native-video-ready-target')


if __name__ == '__main__':
    main()
