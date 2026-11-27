"""Native BF16 CUDA equality gate for LongLive fusion; no timing claim."""
import hashlib
import json
import os
from pathlib import Path
from dataclasses import replace

from .correction_runtime import launch_correction_runtime
from .correction_state import LatentHistory
from .early_draft_diagnostic import PROMPT
from .insight2_campaign import PromptCase, ChunkGenerationRequest
from .longlive_joint_handoff import ENV, require
from .longlive_joint_worker import process_factory, ACTION
from .longlive_joint_oracle import CASES, SCHEDULE
from .longlive_joint_trace import validate_trace
from .longlive_late_gate import SOURCE_PATHS as PREVIOUS_SOURCE_PATHS
from .runtime_helpers import shutdown_owned_runtime_helpers

SOURCE_PATHS = tuple(sorted(set(PREVIOUS_SOURCE_PATHS) | {
    'src/cfexp/longlive_joint_handoff.py', 'src/cfexp/longlive_joint_oracle.py',
    'src/cfexp/longlive_joint_worker.py', 'src/cfexp/longlive_joint_gate.py',
    'src/cfexp/longlive_joint_trace.py',
    'src/cfexp/longlive_chunk_session.py', 'src/cfexp/longlive_policy_backend.py',
    'src/cfexp/longlive_policy_worker.py', 'src/cfexp/prefix_oracle.py',
    'src/cfexp/branch_kv_state.py', 'src/cfexp/incremental_draft.py'}))


def expected_forward_trace():
    trace = []
    def add(index, steps):
        trace.append(dict(current_start=index*4680, latent_frames=len(steps), timesteps=list(steps)))
    for retained, external in CASES:
        n = retained+external
        for count in (n, retained):
            for index in range(count):
                add(index, [0.]*3)
        add(retained, [0.]*(3*external)+[1000.]*3)
        add(n, [1000.]*3)
        for step in SCHEDULE[1:]:
            for _ in range(2):
                add(n, [step]*3)
        for _ in range(2):
            add(n, [0.]*3)
        for following in range(1, 7):
            for step in SCHEDULE:
                for _ in range(2):
                    add(n+following, [step]*3)
            for _ in range(2):
                add(n+following, [0.]*3)
    return trace


def validate_report(value, *, files):
    require(value.get('schema') == 'longlive_joint_cuda_gate_v1' and value.get('passed') is True
            and value.get('input_files') == files and value.get('prompt') == PROMPT
            and value.get('cases') == [list(x) for x in CASES]
            and value.get('native_denoising_schedule') == list(SCHEDULE)
            and value.get('native_local_frames') == 12 and value.get('native_sink_frames') == 3
            and value.get('clean_update_frames') == 3
            and value.get('forward_trace') == expected_forward_trace()
            and value.get('formal_quality_result') is False and value.get('no_end_to_end_speed_claim') is True
            and len(value.get('rows', [])) == len(CASES), 'incomplete joint gate response or input binding')
    for row, (retained, external) in zip(value['rows'], CASES):
        validate_trace(row.get('module_trace', {}))
        require((row['retained_blocks'], row['external_target_blocks'], row['chunk_index'])
                == (retained, external, retained+external)
                and [x['stage'] for x in row['stages']]
                    == ['first_denoise', 'denoise_1', 'denoise_2', 'denoise_3', 'final_clean_B3']
                and [x['offset'] for x in row['following']] == list(range(1, 7)),
                'joint gate omitted a suffix, denoise, cache update or following block')
        first = row['stages'][0]
        require(first.get('hidden15_exact') is True and first.get('clean_self_cache_layers_exact') == 30
                and first.get('physical_joint_forwards') == 1, 'joint gate omitted clean-cache/hidden/physical evidence')
        for stage in row['stages'] + row['following']:
            require(stage.get('latent_exact') is True and stage.get('maximum_absolute_difference') == 0
                    and stage.get('cache_differences') == [], 'joint gate is not exactly native-equivalent')
        require(all(x.get('hidden15_exact') is True for x in row['following']),
                'joint gate omitted following layer15 hidden checks')
    return True


def gate_devices(single_gpu):
    require(type(single_gpu) is bool, 'invalid joint diagnostic hardware mode')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    visible = tuple(x.strip() for x in visible)
    count = 1 if single_gpu else 2
    require(len(visible) == count and len(set(visible)) == count and all(visible),
            'joint diagnostic requires exactly its declared visible GPUs')
    return (visible[0], visible[0]) if single_gpu else visible


def run(output, repository, model_root, devices, *, single_gpu=False):
    from .main_table_baseline_campaign import verify_artifact_spec, _sha256_file
    from .correction_artifacts import verify_target_artifact_spec
    from .main_table_ours_campaign import _assert_idle_pair, _assert_worker_pair_owns_gpus
    require(type(single_gpu) is bool and len(devices) == 2
            and len(set(devices)) == (1 if single_gpu else 2),
            'joint diagnostic physical allocation differs from its explicit mode')
    require(os.environ.get(ENV) == '1' and os.environ.get('CFEXP_LONGLIVE_HOST_CURSORS') == '0'
            and os.environ.get('CFEXP_LONGLIVE_PREFIX_DIAGNOSTIC', '0') == '0',
            'joint gate requires isolated native execution and device cursors')
    draft_spec = repository/'experiments/erank3-longlive-policy-v1.artifacts.json'
    target_spec = repository/'experiments/insight2-krea14b.artifacts.json'
    _, dv, _ = verify_artifact_spec(draft_spec, method_id='draft_only',
                                   model_root=model_root, repository_root=repository)
    _, tv, _ = verify_target_artifact_spec(target_spec, model_root=model_root, repo_root=repository)
    live = output/'live'; live.mkdir()
    prompt = PromptCase('longlive-joint-gate', PROMPT, 'evaluation')
    history = LatentHistory('longlive-joint-gate', prompt.prompt_id, 20260911)
    report = dict(schema='longlive_joint_gate_driver_v1', passed=False, formal_quality_result=False,
        no_end_to_end_speed_claim=True, prompt=PROMPT, seed=20260911,
        source_sha256={path: _sha256_file(repository/path)[0] for path in SOURCE_PATHS},
        target_history=[], real_target_history_blocks=12,
        physical_gpu_count=1 if single_gpu else 2,
        shared_precomputed_conditioning=single_gpu)
    runtime = None
    try:
        _assert_idle_pair(*devices)
        runtime = launch_correction_runtime(run_dir=live, repo_root=repository,
            draft_artifact_spec_path=draft_spec, draft_artifact_spec_sha256=_sha256_file(draft_spec)[0],
            draft_model_id='longlive_1.3b', draft_verified_artifacts=dv,
            target_artifact_spec_path=target_spec, target_artifact_spec_sha256=_sha256_file(target_spec)[0],
            target_verified_artifacts=tv, devices=devices, chunk_count=30, process_factory=process_factory,
            **(dict(allow_shared_device=True, shared_precomputed_conditioning=True,
                    conditioning_prompts=[dict(prompt_id=prompt.prompt_id, prompt=PROMPT)])
               if single_gpu else {}))
        for n in range(12):
            _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0],
                target_device=devices[1], phase='joint LongLive gate Target history')
            noise = runtime.noise_provider.materialize(prompt_id=prompt.prompt_id,
                trajectory_seed=20260911, chunk_index=n)
            request = ChunkGenerationRequest(prompt=prompt, history=history, noise=noise,
                output_latent_path=f'latents/target-{n:02d}.safetensors', capture_hidden=False)
            item = runtime.target.generate(replace(request,
                target_logical_call_id=hashlib.sha256(f'longlive-joint-target-{n}'.encode()).hexdigest(),
                target_call_phase='unscoped'))
            history = history.append(item.segment)
            report['target_history'].append(dict(path=item.segment.latent_path,
                sha256=item.segment.latent_sha256, diagnostics=item.diagnostics))
        noises = [runtime.noise_provider.materialize(prompt_id=prompt.prompt_id+'-joint-noise',
            trajectory_seed=20260911, chunk_index=n) for n in range(28)]
        files = [dict(path=x.latent_path, sha256=x.latent_sha256) for x in history.segments]
        files += [dict(path=x.tensor_path, sha256=x.descriptor.tensor_sha256) for x in noises]
        _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0],
            target_device=devices[1], phase='joint LongLive complete CUDA gate')
        report['gate'] = runtime.draft._request(ACTION, dict(prompt=PROMPT,
            history_latent_paths=[x.latent_path for x in history.segments],
            noise_paths=[x.tensor_path for x in noises],
            output_paths={'report': 'diagnostics/joint-cuda-parity.json'}))
        validate_report(report['gate'], files=files)
        report['passed'] = True
        return report
    except BaseException as error:
        report['error'] = str(error)
        raise
    finally:
        try:
            try:
                if runtime is not None:
                    runtime.close()
            finally:
                shutdown_owned_runtime_helpers(label='longlive-joint-native-gate')
        except BaseException as error:
            report.update(passed=False, cleanup_error=str(error))
            raise
        finally:
            with (output/'longlive-joint-native-gate.json').open('x') as out:
                json.dump(report, out, indent=2, allow_nan=False)


def main():
    import argparse
    from .main_table_baseline_campaign import _run_dir, _model_root
    from .metadata import find_git_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single-gpu', action='store_true')
    args = parser.parse_args()
    run(_run_dir(), find_git_root(Path.cwd()), _model_root(), gate_devices(args.single_gpu),
        single_gpu=args.single_gpu)


if __name__ == '__main__':
    main()
