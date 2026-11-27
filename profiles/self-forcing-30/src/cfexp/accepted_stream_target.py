"""Matched pure Target with accepted rendering on the reserved Draft worker GPU.

Both arms retain the same two-GPU allocation, frozen ABR encoder, native SF VAE,
conditioning and first12 development population. Ready time includes generation,
stream delivery, final GPU/CPU drain, video publication and evidence preparation.
"""
import argparse
import json
from pathlib import Path
import time

from .accepted_stream import TIMING_SCOPE
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
        vbench_asset_manifest, vbench_asset_spec, duration_seconds=None, smoke_prompt_id=None):
    if not enabled():
        raise ValueError('pure Target reference requires explicit accepted streaming')
    root, models = _unsealed_directory(_run_dir()), _model_root()
    repo = find_git_root(Path.cwd())
    if (root/'result.json').exists():
        raise ValueError('stream Target requires a fresh run')
    prompts, population = load_baseline_population(repo)
    profile = duration_profile(15 if duration_seconds is None else duration_seconds)
    benchmark = 'VBench-Long' if profile.nominal_seconds == 60 else 'VBench'
    population_name = 'main24_method_development_not_held_out'
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
    devices = _visible_devices()
    _assert_idle_pair(*devices)
    hardware = _hardware(devices)
    started = time.perf_counter()
    runtime = launch_correction_runtime(run_dir=root, repo_root=repo,
        draft_artifact_spec_path=draft_artifact_spec,
        draft_artifact_spec_sha256=_sha256_file(draft_artifact_spec)[0], draft_verified_artifacts=dv,
        target_artifact_spec_path=target_artifact_spec,
        target_artifact_spec_sha256=target_artifact_spec_sha256(target_artifact_spec),
        target_verified_artifacts=tv, devices=devices, write_full_videos=True, chunk_count=profile.chunk_count,
        conditioning_prompts=[{'prompt_id':p['id'], 'prompt':p['text']} for p in prompts],
        shared_precomputed_conditioning=True, process_factory=process_factory)
    cells, requests = [], []
    try:
        decoder = StreamDecoderProxy(runtime.draft, runtime.draft, root, chunk_count=profile.chunk_count)
        runtime.evaluator.client = decoder
        runtime.evaluator.full_quality_metric_id = ('vbench_long_custom_six_dimension_macro_mean'
            if benchmark == 'VBench-Long' else 'vbench_custom_six_dimension_macro_mean')
        load_seconds = time.perf_counter()-started
        for prompt in prompts:
            _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0],
                target_device=devices[1], phase='stream pure Target '+prompt['id'])
            started = time.perf_counter()
            generation = generate_full_target_trajectory(prompt_id=prompt['id'], prompt=prompt['text'],
                target=runtime.target, noise_provider=runtime.noise_provider,
                chunk_count=profile.chunk_count, duration_seconds=profile.nominal_seconds, on_commit=decoder.append)
            generation_seconds = time.perf_counter()-started
            prepared = runtime.evaluator.prepare_full(history=generation.history,
                measurement_id='stream-target-'+prompt['id'])
            ready_seconds = time.perf_counter()-started
            native = dict(generation.result)
            _write_new_json(root/'provenance'/'target-native'/
                (prompt['id']+'.json'), dict(native, history=generation.history.to_dict()))
            video = bound_file(root, root/prepared.video_path)
            cell = {'arm':'target', 'prompt_id':prompt['id'], 'prompt':prompt['text'],
                'seed':20260911, 'video_ready_wall_seconds':ready_seconds,
                'generation_wall_seconds':generation_seconds,
                'decode_encode_and_prepare_wall_seconds':ready_seconds-generation_seconds,
                'post_generation_drain_and_prepare_wall_seconds':ready_seconds-generation_seconds,
                'decode_encode_and_prepare_wall_seconds_semantics':'tail_after_generation_not_total_render_work',
                'timing_scope':TIMING_SCOPE, 'video':video,
                'root_latent_sha256':generation.history.segments[0].latent_sha256,
                'accepted_stream':decoder.last_evidence, 'native_generation':native,
                'model_load_and_decoder_warmup_wall_seconds':load_seconds,
                'decode_evidence':bound_file(root, root/prepared.decode_evidence_path)}
            cells.append(cell)
            requests.append(VBenchLongVideoRequest(video_path=root/video['path'],
                prompt=prompt['text'], measurement_id='stream-target-'+prompt['id']))
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
        raise ValueError('incomplete pure Target stream quality coverage')
    rows = []
    for cell, request in zip(cells, requests):
        value = measurements[request.measurement_id]
        scores = dict(value.dimension_scores)
        if set(scores) != set(QUALITY_DIMENSIONS) or bound_file(root,root/cell['video']['path']) != cell['video']:
            raise ValueError('Target quality/video binding changed')
        rows.append(dict(cell, quality_scores=scores,
            quality_evidence=bound_file(root,root/value.evidence_path)))
    result = {'schema':'accepted_stream_target_reference_v1', 'status':'succeeded',
        'formal_speedup_claim':False, 'population':population_name, 'benchmark':benchmark, 'duration':profile.metadata(),
        'population_binding':population, 'code_revision':collect_git_metadata(repo)['sha'],
        'hardware':hardware, 'timing_scope':TIMING_SCOPE, 'rows':rows, 'row_count':len(rows)}
    _write_new_json(root/'result.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('draft-artifact-spec','target-artifact-spec','vbench-checkout','vbench-python',
                 'vbench-asset-manifest','vbench-asset-spec'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--duration-seconds', type=int, choices=(15, 30, 60))
    parser.add_argument('--smoke-prompt-id')
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:
        print(json.dumps(run(**vars(parser.parse_args()))))
    finally:
        shutdown_owned_runtime_helpers(label='accepted-stream-target')


if __name__ == '__main__':
    main()
