"""Native BF16 CUDA equality gate for LongLive fusion; no timing claim."""
import hashlib
import json
import math
import os
from pathlib import Path
from dataclasses import replace

from .correction_runtime import launch_correction_runtime
from .correction_state import LatentHistory
from .early_draft_diagnostic import PROMPT
from .insight2_campaign import PromptCase, ChunkGenerationRequest
from .longlive_fusion_handoff import ENV, require
from .longlive_fusion_gate_worker import process_factory, ACTION
from .longlive_joint_executor_oracle import CASES, SCHEDULE, expected_forward_trace,PREFIX_BOOTSTRAP
from .longlive_late_gate import SOURCE_PATHS as PREVIOUS_SOURCE_PATHS
from .runtime_helpers import shutdown_owned_runtime_helpers
from .longlive_joint_policy_gate import validate_report as validate_original_full_policy

SOURCE_PATHS = tuple(sorted(set(PREVIOUS_SOURCE_PATHS) | {
    'src/cfexp/longlive_joint_handoff.py', 'src/cfexp/longlive_joint_oracle.py',
    'src/cfexp/longlive_joint_worker.py', 'src/cfexp/longlive_joint_gate.py',
    'src/cfexp/longlive_joint_trace.py',
    'src/cfexp/longlive_chunk_session.py', 'src/cfexp/longlive_policy_backend.py',
    'src/cfexp/longlive_policy_worker.py', 'src/cfexp/prefix_oracle.py',
    'src/cfexp/branch_kv_state.py', 'src/cfexp/incremental_draft.py',
    'src/cfexp/longlive_joint_executor.py', 'src/cfexp/longlive_joint_executor_oracle.py',
    'src/cfexp/longlive_joint_executor_worker.py', 'src/cfexp/longlive_joint_executor_gate.py',
    'src/cfexp/cpu_history_binding.py', 'src/cfexp/stage_timing.py',
    'src/cfexp/tracked_longlive_joint_executor_gate.py',
    'src/cfexp/longlive_clean_witness.py', 'src/cfexp/longlive_joint_policy_capture.py',
    'src/cfexp/longlive_joint_policy_oracle.py', 'src/cfexp/longlive_joint_policy_worker.py',
    'src/cfexp/longlive_joint_policy_gate.py', 'src/cfexp/tracked_longlive_joint_policy_gate.py',
    'src/cfexp/longlive_fusion_handoff.py', 'src/cfexp/longlive_fusion_executor.py',
    'src/cfexp/longlive_fusion_capture.py', 'src/cfexp/longlive_fusion_runtime.py',
    'src/cfexp/longlive_fusion_gate_oracle.py', 'src/cfexp/longlive_fusion_gate_worker.py',
    'src/cfexp/longlive_fusion_gate.py', 'src/cfexp/tracked_longlive_fusion_gate.py'}))


def validate_report(value, *, files):
    normal = value.get('normal_rpc_reference', {})
    require(normal.get('latent_exact') is True and normal.get('maximum_absolute_difference') == 0
            and normal.get('cache_differences') == []
            and all(normal.get(k) is True for k in ('ordinary_generate_rpc_compared',
                'rank_evidence_exact', 'accepted_clean_witness_exact',
                'reuse_and_worker_file_bindings_exact', 'rng_state_exact'))
            and normal.get('normal_rpc_physical_calls') == 6
            and normal.get('independent_reference_physical_calls') == 6,
            'normal worker RPC/cold native reference proof missing')
    return validate_original_full_policy(value, files=files)


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
    prompt = PromptCase('longlive-joint-policy-gate', PROMPT, 'evaluation')
    history = LatentHistory('longlive-joint-policy-gate', prompt.prompt_id, 20260911)
    report = dict(schema='longlive_joint_policy_gate_driver_v1', passed=False, formal_quality_result=False,
        no_end_to_end_speed_claim=True, prompt=PROMPT, seed=20260911,
        source_sha256={path: _sha256_file(repository/path)[0] for path in SOURCE_PATHS},
        normal_factory_installed=True, normal_rpc_probe_executed=False,
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
                target_logical_call_id=hashlib.sha256(f'longlive-joint-policy-target-{n}'.encode()).hexdigest(),
                target_call_phase='unscoped'))
            history = history.append(item.segment)
            report['target_history'].append(dict(path=item.segment.latent_path,
                sha256=item.segment.latent_sha256, diagnostics=item.diagnostics))
        # Exercise the ordinary serialization/client/history path before the
        # one-shot oracle. Its cold six forwards receive a separate real trace.
        probe_noise = runtime.noise_provider.materialize(prompt_id=prompt.prompt_id,
            trajectory_seed=20260911, chunk_index=1)
        probe = runtime.draft.generate(ChunkGenerationRequest(prompt=prompt,
            history=history.prefix_before(1), noise=probe_noise,
            output_latent_path='latents/ordinary-probe-01.safetensors', capture_hidden=False))
        require(probe.diagnostics['longlive_joint_execution']['fused_history_chunks'] == 0
                and probe.diagnostics['longlive_joint_execution']['expected_physical_generator_calls'] == 6,
                'ordinary cold RPC execution evidence differs')
        report['normal_rpc_probe_executed'] = True
        report['normal_rpc_probe'] = dict(latent_sha256=probe.segment.latent_sha256,
            latent_path=probe.segment.latent_path,
            initial_noise_path=probe_noise.tensor_path,
            initial_noise_tensor_sha256=probe_noise.descriptor.tensor_sha256,
            diagnostics=probe.diagnostics)
        noises = [runtime.noise_provider.materialize(prompt_id=prompt.prompt_id+'-joint-lease-noise',
            trajectory_seed=20260911, chunk_index=n) for n in range(7)]
        files = [dict(path=x.latent_path, sha256=x.latent_sha256) for x in history.segments]
        files += [dict(path=x.tensor_path, sha256=x.descriptor.tensor_sha256) for x in noises]
        _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0],
            target_device=devices[1], phase='joint LongLive complete CUDA gate')
        report['gate'] = runtime.draft._request(ACTION, dict(prompt=PROMPT,
            history_latent_paths=[x.latent_path for x in history.segments],
            noise_paths=[x.tensor_path for x in noises],
            output_paths={'report': 'diagnostics/joint-cuda-parity.json'}))
        require(report['gate']['normal_worker_probe']['result']['latent']['sha256']
                == probe.segment.latent_sha256,
                'ordinary parent/client output differs from the child gate input')
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
                shutdown_owned_runtime_helpers(label='longlive-joint-policy-native-gate')
        except BaseException as error:
            report.update(passed=False, cleanup_error=str(error))
            raise
        finally:
            with (output/'longlive-fusion-normal-worker-gate.json').open('x') as out:
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
