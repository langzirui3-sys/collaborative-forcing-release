"""Worker-side binding and one-shot commit for interrupted Draft requests."""
from __future__ import annotations
import hashlib
import json
from collections.abc import Mapping

from .early_protocol import validate_control, decide


def prepare(worker, request):
    from .correction_worker import WorkerProtocolError
    def check(ok, message):
        if not ok:
            raise WorkerProtocolError(message)
    check(worker.role == 'draft' and request['capture_hidden'] is False
          and 'kv_branch_action' not in request
          and getattr(worker.backend, 'supports_history_kv_widths', False),
          'early generation requires native Draft with no branch/capture action')
    check(callable(getattr(worker.backend, 'reset_early_draft', None))
          and callable(getattr(worker.backend, 'commit_early_target', None)),
          'backend lacks early generation support')
    session = request['early_session_id']
    check(isinstance(session, str) and len(session) == 64
          and all(c in '0123456789abcdef' for c in session), 'invalid early session ID')
    control = validate_control(request['early_control'], chunk_index=request['chunk_index'])
    check(control['remaining_slots'] == worker._chunk_count-request['chunk_index'],
          'early control horizon does not match worker')
    check(getattr(worker, '_early_pending', None) is None, 'early replacement commit is pending')
    current = getattr(worker, '_early_session', None)
    if current is None or current['id'] != session:
        # A fresh worker may explicitly replay a supplied prefix. A live session
        # may only switch to the start of another trajectory, never rewind itself.
        check(current is None or request['chunk_index'] == 1,
              'new early session must start after the Target root')
        worker.backend.reset_early_draft()
        current = {'id': session, 'prompt': request['prompt'], 'next': request['chunk_index'],
                   'ranks': list(control['prior_ranks']), 'remaining': control['remaining_budget'],
                   'calibration': (control['calibration_id'], control['calibration_drop_scores'])}
        worker._early_session = current
    check(current['prompt'] == request['prompt'] and current['next'] == request['chunk_index'],
          'early session prompt/cursor mismatch')
    check(control['prior_ranks'] == current['ranks']
          and control['remaining_budget'] == current['remaining']
          and (control['calibration_id'], control['calibration_drop_scores']) == current['calibration'],
          'early control changed committed decision history or calibration')
    observations = []
    def callback(measurement):
        check(not observations, 'early callback ran more than once')
        result = decide(control, float(measurement.effective_rank))
        observations.append(result)
        return result['replace_with_target']
    return callback, observations


def finish(worker, request, result, observations, history_paths, noise_path):
    from .correction_worker import WorkerExecutionError, _sha256_file, _input_path, _json_value
    def check(ok, message):
        if not ok:
            raise WorkerExecutionError(message)
    check(len(observations) == 1 and isinstance(result, Mapping), 'missing early decision')
    decision = observations[0]
    evidence = result.get('early_execution')
    check(isinstance(evidence, Mapping) and evidence.get('schema') == 'draft_early_execution_v1',
          'missing early execution evidence')
    check(evidence.get('stopped') is decision['replace_with_target']
          and evidence.get('branch_snapshot_bytes') == 0, 'early execution/decision mismatch')
    result = {**result, 'early_decision': decision}
    worker._early_session['ranks'].append(decision['effective_rank'])
    worker._early_session['remaining'] -= int(decision['replace_with_target'])
    if not decision['replace_with_target']:
        check('latent' in result and result.get('rank') == decision['effective_rank'],
              'continued Draft rank/output mismatch')
        worker._early_session['next'] += 1
        return result, None
    check(result.get('stopped') is True and 'latent' not in result, 'stopped Draft returned a latent')
    probe = result.get('probe')
    check(isinstance(probe, Mapping) and probe.get('layer_index') == 14
          and probe.get('timestep') == 1000 and probe.get('current_start') == request['chunk_index']*4680
          and probe.get('completed_native_forward') is False
          and probe.get('effective_rank') == decision['effective_rank'], 'invalid partial probe')
    files = [*history_paths, noise_path]
    file_hashes = [_sha256_file(_input_path(worker.run_dir, p, label='early input')[1])[0] for p in files]
    pending = {'session_id': request['early_session_id'], 'prompt': request['prompt'],
               'internal_seed': request['internal_seed'],
               'chunk_index': request['chunk_index'], 'history_paths': list(history_paths),
               'widths': list(request['history_kv_widths']), 'files': files, 'hashes': file_hashes}
    token = hashlib.sha256(json.dumps({'request_id': request['request_id'], **pending},
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    pending['token'] = token
    worker._early_pending = pending
    payload = {'chunk_index': request['chunk_index'], 'history_chunk_count': len(history_paths),
               'early_session_id': request['early_session_id'], 'pending_token': token,
               'stopped': True, 'result': _json_value(result, label='early result')}
    return result, payload


def commit(worker, request):
    from .correction_worker import WorkerProtocolError, _exact_keys, _input_path, _sha256_file
    def check(ok, message):
        if not ok:
            raise WorkerProtocolError(message)
    _exact_keys(request, worker._COMMON_KEYS | {'early_session_id', 'pending_token',
                'target_latent_path', 'target_latent_sha256'}, label='early commit request')
    pending = getattr(worker, '_early_pending', None)
    check(worker.role == 'draft' and pending is not None, 'no pending early Draft replacement')
    check(request['early_session_id'] == pending['session_id']
          and request['pending_token'] == pending['token'], 'early commit capability mismatch')
    for p, digest in zip(pending['files'], pending['hashes']):
        check(_sha256_file(_input_path(worker.run_dir, p, label='pending input')[1])[0] == digest,
              'pending early input changed')
    path, latent = worker._load_input_tensor(request['target_latent_path'], label='Target latent', key='latent')
    check(path not in pending['files'], 'early replacement aliases an input')
    digest = _sha256_file(_input_path(worker.run_dir, path, label='Target latent')[1])[0]
    check(digest == request['target_latent_sha256'], 'Target latent digest mismatch')
    history, _ = worker._history(pending['history_paths'], expected_count=pending['chunk_index'])
    try:
        result = worker.backend.commit_early_target(prompt=pending['prompt'], history_latents=history,
                    history_kv_widths=tuple(pending['widths']), latent=latent)
        check(result == {'external_clean_forwards': 1, 'history_replay_chunks': 0,
                         'branch_snapshot_bytes': 0, 'committed_chunks': pending['chunk_index']+1},
              'early commit did not perform exactly one native clean forward')
    except BaseException as error:
        raise worker._poison(error) from error
    worker._early_pending = None
    worker._early_session['next'] += 1
    return {'early_session_id': request['early_session_id'], 'pending_token': request['pending_token'],
            'target_latent_path': path, 'target_latent_sha256': digest, 'result': result}
