"""Matched pure Target with accepted rendering through the Draft worker.

Both arms retain the same physical GPU allocation, frozen ABR encoder, native SF VAE,
conditioning and first12 development population. Ready time includes generation,
stream delivery, final GPU/CPU drain, video publication and evidence preparation.
"""
import argparse
import json
import os
from pathlib import Path
import time

from .paper_population import generation_seed, population_label

from .accepted_stream import TIMING_SCOPE, SINGLE_GPU_TIMING_SCOPE, single_gpu_enabled, stream_devices
from .accepted_stream_client import StreamDecoderProxy, enabled
from .accepted_stream_worker import process_factory
from .baseline_population import load_baseline_population
from .correction_artifacts import verify_target_artifact_spec, target_artifact_spec_sha256
from .correction_runtime import launch_correction_runtime
from .duration_sweep import duration_profile
from .erank_acceptance_pilot import OfficialVBenchScorer, bound_file, QUALITY_DIMENSIONS
from .erank_frame_campaign import verify_artifact_spec
from .main_table_baseline_campaign import _run_dir, _model_root, _visible_devices, _sha256_file, _write_new_json
from .main_table_full_target import generate_full_target_trajectory
from .main_table_ours_campaign import _assert_idle_pair, _assert_worker_pair_owns_gpus
from .metadata import find_git_root, collect_git_metadata
from .vbench_long import VBenchLongVideoRequest
from .stage_timing import _unsealed_directory
from .stage_timing_probe import _hardware


def run(*, draft_artifact_spec, target_artifact_spec, vbench_checkout, vbench_python,
        vbench_asset_manifest, vbench_asset_spec, duration_seconds=None, smoke_prompt_id=None,
        pure_draft=False, development_c10=False, boundary_config=None, runtime_launcher=None):
    if not enabled():
        raise ValueError('pure reference requires explicit accepted streaming')
    if type(pure_draft) is not bool or (pure_draft and not single_gpu_enabled()):
        raise ValueError('pure Draft reference requires the single-GPU timing protocol')
    if os.environ.get('CFEXP_NATIVE_VIDEO_STREAM_MODEL') is not None:
        raise ValueError('Self-Forcing reference cannot use a different native renderer')
    if pure_draft and os.environ.get('CFEXP_DRAFT_KV_REUSE') != '1':
        raise ValueError('pure Self-Forcing Draft requires native KV reuse')
    role = 'draft' if pure_draft else 'target'
    root, models = _unsealed_directory(_run_dir()), _model_root()
    repo = find_git_root(Path.cwd())
    if (root/'result.json').exists():
        raise ValueError('stream Target requires a fresh run')
    if type(development_c10) is not bool:
        raise ValueError('invalid development reference flag')
    if development_c10:
        if runtime_launcher is None:
            raise ValueError('development C10 controls require the instrumented reference launcher')
        from .native_c10_development import reference_population
        prompts,population=reference_population(repo,boundary_config,
            seconds=duration_seconds,smoke=smoke_prompt_id)
    else:
        if boundary_config is not None:
            raise ValueError('reference boundary configuration requires explicit development C10')
        prompts, population = load_baseline_population(repo)
    profile = duration_profile(15 if duration_seconds is None else duration_seconds)
    benchmark = 'VBench-Long' if profile.nominal_seconds == 60 else 'VBench'
    population_name = population_label('main24_method_development_not_held_out')
    if development_c10:
        population_name='independent_C10_development_not_held_out'
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
    _, dv, dm = verify_artifact_spec(draft_artifact_spec, model_root=models, repo_root=repo)
    _, tv, tm = verify_target_artifact_spec(target_artifact_spec, model_root=models, repo_root=repo)
    if (dm, tm) != ('cfexp.self_forcing_frame_adapter', 'cfexp.krea_chunk_backend'):
        raise ValueError('stream Target backend identity differs')
    single_gpu = single_gpu_enabled()
    from .paper_population import validate_execution
    validate_execution(single_gpu=single_gpu, seconds=duration_seconds)
    timing_scope = SINGLE_GPU_TIMING_SCOPE if single_gpu else TIMING_SCOPE
    devices = stream_devices() if single_gpu else _visible_devices()
    _assert_idle_pair(*devices)
    hardware = _hardware(tuple(dict.fromkeys(devices)))
    started = time.perf_counter()
    launch=launch_correction_runtime if runtime_launcher is None else runtime_launcher
    runtime = launch(run_dir=root, repo_root=repo,
        draft_artifact_spec_path=draft_artifact_spec,
        draft_artifact_spec_sha256=_sha256_file(draft_artifact_spec)[0], draft_verified_artifacts=dv,
        target_artifact_spec_path=target_artifact_spec,
        target_artifact_spec_sha256=target_artifact_spec_sha256(target_artifact_spec),
        target_verified_artifacts=tv, devices=devices, write_full_videos=True, chunk_count=profile.chunk_count,
        conditioning_prompts=[{'prompt_id':p['id'], 'prompt':p['text']} for p in prompts],
        shared_precomputed_conditioning=True, process_factory=process_factory,
        **({"allow_shared_device": True} if single_gpu else {}))
    cells, requests = [], []
    try:
        renderer = runtime.target if pure_draft else runtime.draft
        decoder = StreamDecoderProxy(runtime.draft, renderer, root, chunk_count=profile.chunk_count)
        runtime.evaluator.client = decoder
        runtime.evaluator.full_quality_metric_id = ('vbench_long_custom_six_dimension_macro_mean'
            if benchmark == 'VBench-Long' else 'vbench_custom_six_dimension_macro_mean')
        load_seconds = time.perf_counter()-started
        for prompt in prompts:
            _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0],
                target_device=devices[1], phase=f"stream pure {role} {prompt['id']}")
            started = time.perf_counter()
            if pure_draft:
                from .native_video_ready_draft import SELF_FORCING, generate
                generation = generate(draft=runtime.draft, noise_provider=runtime.noise_provider,
                    prompt_id=prompt['id'], prompt=prompt['text'], chunk_count=profile.chunk_count,
                    model=SELF_FORCING, renderer=decoder, run_dir=root)
            else:
                generation = generate_full_target_trajectory(prompt_id=prompt['id'], prompt=prompt['text'],
                    target=runtime.target, noise_provider=runtime.noise_provider,
                    chunk_count=profile.chunk_count, duration_seconds=profile.nominal_seconds, on_commit=decoder.append)
            generation_seconds = time.perf_counter()-started
            prepared = runtime.evaluator.prepare_full(history=generation.history,
                measurement_id=f'stream-{role}-'+prompt['id'])
            ready_seconds = time.perf_counter()-started
            native = dict(generation.result)
            if development_c10 and not pure_draft:
                native['development_actual_chunks']=[dict(
                    chunk_index=item.segment.chunk_index,producer_role=item.segment.producer_role,
                    latent_sha256=item.segment.latent_sha256,
                    initial_noise_descriptor_sha256=item.segment.initial_noise_descriptor_sha256,
                    diagnostics=dict(item.diagnostics)) for item in generation.generated]
            _write_new_json(root/'provenance'/f'{role}-native'/
                (prompt['id']+'.json'), dict(native, history=generation.history.to_dict()))
            video = bound_file(root, root/prepared.video_path)
            cell = {'arm':role, 'prompt_id':prompt['id'], 'prompt':prompt['text'],
                'seed':generation_seed(), 'video_ready_wall_seconds':ready_seconds,
                'generation_wall_seconds':generation_seconds,
                'decode_encode_and_prepare_wall_seconds':ready_seconds-generation_seconds,
                'post_generation_drain_and_prepare_wall_seconds':ready_seconds-generation_seconds,
                'decode_encode_and_prepare_wall_seconds_semantics':('gpu_decode_and_encoder_drain_after_generation' if single_gpu
                    else 'tail_after_generation_not_total_render_work'),
                'timing_scope':timing_scope, 'video':video,
                'root_latent_sha256':generation.history.segments[0].latent_sha256,
                'accepted_stream':decoder.last_evidence, 'native_generation':native,
                'model_load_and_decoder_warmup_wall_seconds':load_seconds,
                'decode_evidence':bound_file(root, root/prepared.decode_evidence_path)}
            if single_gpu:
                cell['accepted_video_schedule'] = 'decode_after_generation'
            cells.append(cell)
            requests.append(VBenchLongVideoRequest(video_path=root/video['path'],
                prompt=prompt['text'], measurement_id=f'stream-{role}-'+prompt['id']))
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
        raise ValueError('incomplete pure reference stream quality coverage')
    rows = []
    for cell, request in zip(cells, requests):
        value = measurements[request.measurement_id]
        scores = dict(value.dimension_scores)
        if set(scores) != set(QUALITY_DIMENSIONS) or bound_file(root,root/cell['video']['path']) != cell['video']:
            raise ValueError('reference quality/video binding changed')
        rows.append(dict(cell, quality_scores=scores,
            quality_evidence=bound_file(root,root/value.evidence_path)))
    result = {'schema':f'accepted_stream_{role}_reference_v1', 'status':'succeeded',
        'formal_speedup_claim':False, 'population':population_name, 'benchmark':benchmark, 'duration':profile.metadata(),
        'population_binding':population, 'code_revision':collect_git_metadata(repo)['sha'],
        'hardware':hardware, 'timing_scope':timing_scope, 'rows':rows, 'row_count':len(rows)}
    if single_gpu:
        result['accepted_video_schedule'] = 'decode_after_generation'
    if development_c10:
        result.update(development_only=True,held_out=False,may_replace_paper128_results=False)
    _write_new_json(root/'result.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('draft-artifact-spec','target-artifact-spec','vbench-checkout','vbench-python',
                 'vbench-asset-manifest','vbench-asset-spec'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--duration-seconds', type=int, choices=(15, 30, 60))
    parser.add_argument('--smoke-prompt-id')
    parser.add_argument('--pure-draft', action='store_true')
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:
        print(json.dumps(run(**vars(parser.parse_args()))))
    finally:
        shutdown_owned_runtime_helpers(label='accepted-stream-target')


if __name__ == '__main__':
    main()
