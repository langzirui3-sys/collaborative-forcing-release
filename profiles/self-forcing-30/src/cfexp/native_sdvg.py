"""Validate native Draft reuse along SDVG's independent proposal history."""

import os

from .native_video_ready_draft import MODELS, SELF_FORCING, validate_cache


def video_ready_protocol(draft_model_id, *, overlap):
    from .accepted_stream import TIMING_SCOPE as SF_TIMING_SCOPE
    from .accepted_stream_client import enabled
    from .native_vae_component import OVERLAP_READY_SCOPE, campaign_model, selected_model
    from .video_encoding_policy import SERIAL, policy
    from .video_only_decode import enabled as video_only_enabled

    if draft_model_id == SELF_FORCING and overlap:
        flags = ('CFEXP_DRAFT_KV_REUSE', 'CFEXP_RENDER_TARGET_PRIORITY', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
        if (selected_model() is not None or not enabled() or not video_only_enabled()
                or policy() != SERIAL or os.environ.get('CFEXP_DRAFT_KV_FORK', '0') != '0'
                or any(os.environ.get(flag) != '1' for flag in flags)):
            raise ValueError('Self-Forcing SDVG requires independent native KV and its matched video stream')
        return 'self-forcing', SF_TIMING_SCOPE
    model = campaign_model(draft_model_id, overlap=overlap)
    return model, OVERLAP_READY_SCOPE if model else None


def launch_conditioned_draft_worker(*, run_dir, repo_root, draft_artifact_spec_path,
        draft_artifact_spec_sha256, draft_verified_artifacts, sdvg_artifact_spec_path,
        sdvg_artifact_spec_sha256, sdvg_verified_artifacts, visible_device,
        draft_model_id, chunk_count, conditioning_cache, process_factory, rolling=None):
    """Use the existing immutable config and worker protocol with cached text.

    The gate-bound correction runtime and native generation implementations
    remain byte-identical. Only this optional SDVG launcher assembles a config.
    """
    from pathlib import Path
    from .correction_runtime import (JsonlWorkerClient, _spec_below_repo,
        _strict_json_file, _sha256_file, _write_backend_config, _draft_worker_factory,
        _validate_rolling_launch_config)
    root=Path(run_dir).resolve(strict=True);repo=Path(repo_root).resolve(strict=True)
    if draft_model_id not in MODELS or chunk_count not in (30,60,120):
        raise ValueError('native SDVG launcher requires native model and horizon')
    _validate_rolling_launch_config(model_id=draft_model_id,rolling=rolling,
        run_dir=root,repo_root=repo,chunk_count=chunk_count)
    draft_spec=_spec_below_repo(draft_artifact_spec_path,repo,label='Draft artifact specification')
    sdvg_spec=_spec_below_repo(sdvg_artifact_spec_path,repo,label='SDVG artifact specification')
    if _sha256_file(sdvg_spec)[0]!=sdvg_artifact_spec_sha256:
        raise ValueError('native SDVG artifact specification digest mismatch')
    spec=_strict_json_file(sdvg_spec)
    if (spec.get('schema')!='main_table_baseline_artifacts_v1' or spec.get('status')!='frozen'
            or spec.get('method_id')!='sdvg' or spec.get('adapter_module')!='cfexp.sdvg_adapter'
            or spec.get('adapter_config',{}).get('benchmark_adaptations',{}).get('chunk_count')!=chunk_count):
        raise ValueError('native SDVG artifact identity or horizon differs')
    config,digest=_write_backend_config(run_dir=root,role='draft',spec_path=draft_spec,
        spec_sha256=draft_artifact_spec_sha256,verified_artifacts=draft_verified_artifacts,
        chunk_count=chunk_count,conditioning_cache=conditioning_cache,rolling=rolling,
        sdvg=dict(artifact_spec_path=str(sdvg_spec),artifact_spec_sha256=sdvg_artifact_spec_sha256,
            adapter_config=dict(spec['adapter_config']),
            verified_artifacts=[dict(x) for x in sdvg_verified_artifacts]))
    return JsonlWorkerClient(run_dir=root,role='draft',
        backend_factory=_draft_worker_factory(draft_model_id),expected_model_id=draft_model_id,
        backend_config_path=config,backend_config_sha256=digest,visible_device=visible_device,
        process_factory=process_factory,chunk_count=chunk_count)


def launch_conditioned_target_worker(*, run_dir, repo_root, target_artifact_spec_path,
        target_artifact_spec_sha256, target_verified_artifacts, visible_device,
        chunk_count, conditioning_cache, process_factory):
    from pathlib import Path
    from .correction_runtime import JsonlWorkerClient, _spec_below_repo, _write_backend_config
    root=Path(run_dir).resolve(strict=True);repo=Path(repo_root).resolve(strict=True)
    if chunk_count not in (30,60,120):raise ValueError('native SDVG Target horizon differs')
    spec=_spec_below_repo(target_artifact_spec_path,repo,label='Target artifact specification')
    config,digest=_write_backend_config(run_dir=root,role='target',spec_path=spec,
        spec_sha256=target_artifact_spec_sha256,verified_artifacts=target_verified_artifacts,
        chunk_count=chunk_count,conditioning_cache=conditioning_cache)
    return JsonlWorkerClient(run_dir=root,role='target',
        backend_factory='cfexp.correction_runtime:build_target_backend',
        backend_config_path=config,backend_config_sha256=digest,visible_device=visible_device,
        process_factory=process_factory,chunk_count=chunk_count)


def audit_native_draft_history(result):
    model = result['draft_model_id']
    if model not in MODELS:
        raise ValueError('native SDVG audit requires a supported Draft')
    routes = result['routes']
    count = result['chunk_count']
    if (count not in (30, 60, 120) or len(routes) != count
            or [r['chunk_index'] for r in routes] != list(range(count))
            or routes[0]['route'] != 'target'
            or not all(r['same_initial_noise_verified'] is True for r in routes)):
        raise ValueError('native SDVG route coverage or root/noise differs')
    sessions = set()
    for i, row in enumerate(routes):
        try:
            validate_cache(row['draft_cache_evidence'], model=model, index=i)
        except (KeyError, ValueError) as error:
            raise ValueError('SDVG Draft failed independent proposal cache reuse') from error
        if row['route'] == 'target':
            target = row['target_cache_evidence'] or {}
            session = target.get('native_session_id')
            if (not isinstance(session, str) or not session
                    or target.get('native_session_block_index') != i):
                raise ValueError('SDVG Target session binding differs')
            sessions.add(session)
        elif row['route'] == 'draft':
            target = (row['target_append'] or {}).get('result', {})
            if (target.get('committed_external_clean_latent') is not True
                    or target.get('native_session_block_index') != i
                    or not isinstance(target.get('native_session_id'), str)):
                raise ValueError('SDVG selected Draft was not appended to Target history')
            sessions.add(target['native_session_id'])
        else:
            raise ValueError('invalid SDVG route')
    if len(sessions) != 1:
        raise ValueError('SDVG Target session changed during selected history')
    return dict(schema='native_sdvg_independent_history_cache_audit_v1',
        draft_model_id=model, candidate_count=count, replayed_draft_chunks=0,
        selected_history_is_separate=True, target_session_count=1)
