"""Fixed-schedule, shared-output-prefix Target-to-Draft handoff experiment."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import hashlib
import os
from pathlib import Path

from .correction_artifacts import target_artifact_spec_sha256, verify_target_artifact_spec
from .correction_runtime import DeterministicNoiseProvider, launch_target_generation_worker
from .erank_frame_campaign import verify_artifact_spec
from .sf_handoff_legacy_executor import IncrementalDraftExecutor, LOCAL_REENCODE_CHUNKS, latent_digest
from .main_table_baseline_campaign import (
    _gpu_compute_pids, _gpu_uuid, _model_root, _run_dir,
)
from .metadata import find_git_root
from .motion_video_audit import digest, require
from .self_forcing_frame_adapter import OfficialSelfForcingBackend, _materialize_source
from .single_target_recovery import contact_sheet, write_json

FPS = 24
BURST_START = 9
HANDOFF = 12
WINDOW = LOCAL_REENCODE_CHUNKS
ARMS = ('draft_private_history', 'accepted_append', 'accepted_local_reencode')
PROMPTS = {
    'park_walk': 'Full-body wide shot from a stationary camera. One woman with a red jacket, black pants, and white shoes walks steadily toward the camera along a straight park path. Her alternating steps stay in view. Natural daylight and one uninterrupted shot.',
    'crosswalk': 'Full-body side view from a stationary camera. One man in a blue denim jacket and tan trousers walks steadily left to right across a pedestrian crosswalk, with both shoes visible. A gray brick building stands behind him. One uninterrupted shot.',
    'hallway_walk': 'One person wearing a yellow coat and dark boots walks away down the center of a tiled hallway, carrying a small brown bag in their right hand. Static camera, full body in frame, smooth alternating steps and no cuts.',
    'brick_alley': 'Full-body static frontal shot. One man in an olive field jacket, dark jeans and brown boots walks toward the camera on a quiet brick alley. Both feet stay in view. Continuous walking, even daylight.',
    'museum_walk': 'Full-body side view. A woman in a navy sweater, cream trousers and black sneakers walks from left to right past framed pictures in a museum corridor. Static camera and uninterrupted stride.',
    'beach_boardwalk': 'A man in a rust hoodie, charcoal trousers and white shoes walks away from a stationary camera along a wooden beach boardwalk. Keep his full figure visible. Overcast daylight and a continuous shot.',
    'library_aisle': 'A woman in a dark green cardigan, blue jeans and brown flats walks steadily toward the camera through a long library aisle. Static camera, full body visible, soft consistent lighting.',
    'station_platform': 'A man wearing a red checked shirt, black jeans and dark shoes walks left to right on a quiet outdoor train platform. Fixed full-body side view, both feet visible throughout.',
    'garden_path': 'A woman wearing a blue coat, black leggings and gray trainers walks away along a garden path. The camera holds a static full-body view. One continuous shot in diffuse daylight.',
    'covered_market': 'A man in a white shirt, navy trousers and brown shoes walks toward a fixed camera along a covered market aisle. He carries a closed umbrella in one hand. Keep both feet visible.',
    'gym_walk': 'A woman in a gray sweatshirt, maroon joggers and white trainers walks from left to right across an empty gym court. Side view from a stationary camera, full body in frame, no cuts.',
    'courtyard_walk': 'A man in a mustard jacket, dark green trousers and black boots walks diagonally toward a fixed camera in a stone courtyard. Full body and alternating steps stay visible.',
    'dog_trot': 'Wide side view of one brown dog with a blue collar trotting from left to right along a dirt path. All four paws are visible, with a low wooden fence in the background. Stable camera and continuous motion in one shot.',
}
SEEDS = (20260911, 20260912)


def frame_start(chunk):
    return max(0, 12 * chunk - 3)


def run_case(*, backend, target, noise_provider, out, prompt_id, seed, chunks, ownership,
             phase):
    from safetensors.torch import load_file, save_file

    t = backend.torch if backend is not None else None
    prompt = PROMPTS[prompt_id]
    case = out / 'cases' / prompt_id / str(seed)
    case.mkdir(parents=True, exist_ok=True)
    relative_case = case.relative_to(out)
    noises = [noise_provider.materialize(prompt_id=prompt_id, trajectory_seed=seed, chunk_index=i)
              for i in range(chunks)]
    source_paths = []
    base = backend._incremental_draft if backend is not None else None
    require(phase == 'target' or (base is not None and not base.fork_enabled),
            'native single-branch Draft KV is required')

    def draft_step(history, widths, index, *, replayed=0, reused=None):
        ownership()
        noise = noises[index]
        result = backend.generate_chunk(
            prompt=prompt, history_latents=t.cat(history, dim=1) if history else None,
            initial_noise=load_file(str(out / noise.tensor_path))['noise'],
            internal_seed=noise.descriptor.derived_seed, chunk_index=index,
            capture_hidden=False, history_kv_widths=tuple(widths),
        )
        cache = result['kv_reuse']
        require(cache['replayed_chunks'] == replayed
                and cache['reused_chunks'] == (index if reused is None else reused),
                f'unexpected Draft KV reuse or replay at chunk {index}')
        require(cache['history_tensor_sha256'] == [latent_digest(x, t) for x in history],
                'Draft KV trace refers to a different conditioning history')
        forward_trace = result['forward_trace']
        replay_positions = [4680*i for i in range(index-replayed, index)]
        prefill = forward_trace[:len(replay_positions)]
        require([event['current_start'] for event in prefill] == replay_positions
                and all(event['timestep'] == 0 for event in prefill)
                and len(forward_trace) == len(replay_positions) + 5,
                'Draft replay did not encode the expected absolute positions')
        latent = result['latent']
        record = {
            'chunk': index, 'latent_sha256': latent_digest(latent, t),
            'noise_sha256': noise.descriptor.tensor_sha256,
            'sampling_seed': noise.descriptor.derived_seed,
            'kv_witness_sha256': result['kv_witness_sha256'],
            'kv_reuse': cache, 'forward_trace': forward_trace,
        }
        return latent, record

    def target_step(index):
        ownership()
        relative = (relative_case / f'target-{index:03d}.safetensors').as_posix()
        identity = hashlib.sha256(f'sf-handoff-v1:{prompt_id}:{seed}:{index}'.encode()).hexdigest()
        noise = noises[index]
        payload = {
            'prompt': prompt, 'chunk_index': index, 'internal_seed': noise.descriptor.derived_seed,
            'capture_hidden': False, 'history_latent_paths': list(source_paths),
            'initial_noise_path': noise.tensor_path, 'output_paths': {'latent': relative},
            'target_logical_call_id': identity, 'target_call_scope_id': identity,
            'target_call_phase': 'single_point_candidate',
        }
        reply = target._request('generate', payload)
        require(reply['chunk_index'] == index and reply['history_chunk_count'] == index,
                'Target consumed the wrong shared history')
        result = reply['result']
        path = out / relative
        require(result['explicit_initial_noise_verified'] is True
                and result['latent']['path'] == relative
                and result['latent']['sha256'] == digest(path)[0]
                and len(result['denoising_forward_trace']) == 4,
                'Target did not bind the shared inputs and native four-step output')
        write_json(case / f'target-{index:03d}.json', {'request': payload, 'reply': reply})
        source_paths.append(relative)
        latent = load_file(str(path))['latent']
        return latent, result['native_correction_state']

    if phase == 'prefix':
        base.cache.clear()
        private, private_trace = [], []
        for index in range(HANDOFF):
            latent, trace = draft_step(private, [3]*index, index)
            private.append(latent)
            private_trace.append(trace)
            name = 'common' if index < BURST_START else 'private'
            save_file({'latent': latent}, str(case / f'{name}-{index:03d}.safetensors'))
        write_json(case / 'private-draft-trace.json', private_trace)
        print(f'{prompt_id} {seed} native Draft prefix completed', flush=True)
        return

    source_paths.extend((relative_case / f'common-{i:03d}.safetensors').as_posix()
                        for i in range(BURST_START))
    if phase == 'target':
        target_states = [target_step(i)[1] for i in range(BURST_START, HANDOFF)]
        write_json(case / 'target-states.json', target_states)
        print(f'{prompt_id} {seed} native Target burst completed', flush=True)
        return

    private_trace = json.loads((case / 'private-draft-trace.json').read_text())
    target_states = json.loads((case / 'target-states.json').read_text())
    private = [load_file(str(case / f'{"common" if i < BURST_START else "private"}-{i:03d}.safetensors'))['latent']
               for i in range(HANDOFF)]
    shared = [*private[:BURST_START],
              *(load_file(str(case / f'target-{i:03d}.safetensors'))['latent']
                for i in range(BURST_START, HANDOFF))]

    def restore_draft_prefix(count):
        base.cache.clear()
        prefix = []
        for index in range(count):
            latent, trace = draft_step(prefix, [3]*index, index)
            require(trace['latent_sha256'] == private_trace[index]['latent_sha256'],
                    'regenerated Draft prefix is not byte-identical')
            prefix.append(latent)

    try:
        require(len({s['native_session_id'] for s in target_states}) == 1
                and all(not s['session_recreated'] for s in target_states[1:]),
                'Target did not retain one native session across the burst')
        private_hashes = [latent_digest(x, t) for x in private]
        shared_hashes = [latent_digest(x, t) for x in shared]
        require(private_hashes == [row['latent_sha256'] for row in private_trace]
                and private_hashes[:BURST_START] == shared_hashes[:BURST_START]
                and all(a != b for a, b in zip(private_hashes[BURST_START:], shared_hashes[BURST_START:])),
                'the private and emitted prefixes do not exhibit the intended divergence')

        emitted_prefix_sha = None
        rows = []
        summary = {}
        frame_cut = frame_start(HANDOFF)
        frame_indices = [frame_cut + i for i in (-25, -13, -5, -1, 0, 3, 8, 15, 23, 35)]
        restore_draft_prefix(HANDOFF)
        reference = backend.decode_latents(t.cat(private, dim=1))
        require(len(reference) == frame_cut, 'all-Draft reference has wrong frame count')
        backend.write_video(reference, len(reference), case / 'draft-before-handoff.mp4', FPS)
        del reference

        for arm in ARMS:
            if arm == 'draft_private_history':
                conditioning = private
                widths = [3]*HANDOFF
                expected_replay, expected_reuse = 0, HANDOFF
            elif arm == 'accepted_append':
                restore_draft_prefix(BURST_START)
                conditioning = list(shared)
                widths = [3]*BURST_START + [1]*(HANDOFF-BURST_START)
                expected_replay, expected_reuse = HANDOFF-BURST_START, BURST_START
            else:
                base.cache.clear()
                backend._incremental_draft = IncrementalDraftExecutor(
                    pipeline=base.pipeline, torch=t, scope_sha256=base.scope_sha256,
                    local_reencode=True,
                )
                conditioning = list(shared)
                widths = [3]*BURST_START + [1]*(HANDOFF-BURST_START)
                expected_replay, expected_reuse = WINDOW, 0

            generated = []
            trace_rows = []
            for index in range(HANDOFF, chunks):
                at_boundary = index == HANDOFF
                latent, trace = draft_step(
                    conditioning, widths, index,
                    replayed=expected_replay if at_boundary else 0,
                    reused=expected_reuse if at_boundary else index,
                )
                generated.append(latent)
                conditioning.append(latent)
                widths.append(3)
                trace_rows.append(trace)

            # Decode the emitted history, including the identical Target burst in every arm.
            frames = backend.decode_latents(t.cat([*shared, *generated], dim=1))
            require(tuple(frames.shape) == (frame_start(chunks), 3, 480, 832), 'decoded shape drift')
            prefix_sha = hashlib.sha256(frames[:frame_cut].numpy().tobytes()).hexdigest()
            require(emitted_prefix_sha is None or prefix_sha == emitted_prefix_sha,
                    'decoded RGB frames before handoff differ across arms')
            emitted_prefix_sha = prefix_sha
            arm_dir = case / arm
            video = arm_dir / 'video.mp4'
            backend.write_video(frames, len(frames), video, FPS)
            crop = frames[frame_cut-24:frame_cut+48, :, ::4, ::4].to(t.float32)
            diffs = (crop[1:]-crop[:-1]).abs().mean(dim=(1,2,3)).tolist()
            rows.append([frames[i, :, ::4, ::4].permute(1,2,0).contiguous().numpy().tobytes()
                         for i in frame_indices])
            summary[arm] = {
                'video': video.relative_to(out).as_posix(), 'video_sha256': digest(video)[0],
                'pre_handoff_rgb_sha256': prefix_sha,
                'conditioning_history_sha256': trace_rows[0]['kv_reuse']['history_tensor_sha256'],
                'first_draft_kv_sha256': trace_rows[0]['kv_witness_sha256'],
                'first_draft_latent_sha256': trace_rows[0]['latent_sha256'],
                'first_draft_replayed_chunks': expected_replay,
                'first_draft_reused_chunks': expected_reuse,
                'first_rgb_difference_later_frame': frame_cut-23,
                'adjacent_rgb_mae': diffs,
            }
            write_json(arm_dir / 'trace.json', trace_rows)
            del frames
            print(f'{prompt_id} {seed} {arm} completed', flush=True)

        contact_sheet(case / 'boundary.png', rows)
        result = {
            'prompt_id': prompt_id, 'prompt': prompt, 'seed': seed, 'fps': FPS,
            'target_chunks': list(range(BURST_START, HANDOFF)), 'first_draft_frame': frame_cut,
            'local_reencoded_chunks': list(range(HANDOFF-WINDOW, HANDOFF)),
            'local_reencode_mode': 'sequential',
            'draft_before_handoff_video': (relative_case / 'draft-before-handoff.mp4').as_posix(),
            'frame_indices': frame_indices, 'row_order': list(ARMS),
            'shared_output_latent_sha256': shared_hashes, 'private_latent_sha256': private_hashes,
            'private_draft_trace': private_trace, 'target_states': target_states,
            'comparison': (case / 'boundary.png').relative_to(out).as_posix(), 'arms': summary,
        }
        write_json(case / 'result.json', result)
        return result
    finally:
        backend._incremental_draft.cache.clear()
        backend._incremental_draft = base


def run(repo, out, model_root, draft_spec, *, smoke, shard):
    chunks = 16
    require(os.environ.get('CFEXP_DRAFT_KV_REUSE') == '1'
            and os.environ.get('CFEXP_DRAFT_KV_FORK', '0') == '0'
            and os.environ.get('CFEXP_HOST_KV_CURSORS', '0') == '0',
            'handoff study requires native Draft KV reuse without forks')
    draft_spec = repo / draft_spec
    target_spec = repo / 'experiments/insight2-krea14b.artifacts.json'
    spec, draft_files, draft_module = verify_artifact_spec(draft_spec, model_root=model_root, repo_root=repo)
    _, target_files, target_module = verify_target_artifact_spec(target_spec, model_root=model_root, repo_root=repo)
    require((draft_module, target_module) == ('cfexp.self_forcing_frame_adapter', 'cfexp.krea_chunk_backend'),
            'experiment backend drift')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    require(len(devices) == 1 and devices[0].strip(), 'handoff study requires one reserved GPU')
    device = devices[0].strip()
    uuid = _gpu_uuid(device)
    require(not _gpu_compute_pids(uuid), 'handoff study requires one idle GPU')
    provider = DeterministicNoiseProvider(run_dir=out)
    prompts = ('brick_alley',) if smoke else tuple(PROMPTS)
    seeds = SEEDS[:1] if smoke else SEEDS
    if shard is not None:
        seeds = (SEEDS[shard],)

    @contextmanager
    def draft_backend(phase):
        artifacts = {row['name']: Path(row['resolved_path']) for row in draft_files}
        source, site = _materialize_source(run_dir=out, worker_index=0, artifacts=artifacts,
            config=spec['adapter_config'], scratch_suffix=f'sf-handoff-{phase}')
        backend = OfficialSelfForcingBackend(source_root=source, python_site=site, artifacts=artifacts,
            config=spec['adapter_config'], device_name='cuda:0', chunk_count=chunks)
        # Keep this historical ablation executor separate from the production
        # executor: the third arm explicitly reconstructs its local window.
        base = backend._incremental_draft
        require(base is not None and base.cache.state_count == 0 and not base.fork_enabled,
                'handoff study requires a fresh single-branch executor')
        backend._incremental_draft = IncrementalDraftExecutor(
            pipeline=base.pipeline, torch=backend.torch, scope_sha256=base.scope_sha256,
            max_bytes=base.cache.max_bytes, fork_enabled=False,
        )
        try:
            yield backend
        finally:
            backend.close()

    def cases(phase, *, backend=None, target=None):
        allowed = {os.getpid()} if target is None else {os.getpid(), target.process.pid}

        def ownership():
            require(_gpu_compute_pids(uuid) <= allowed,
                    'unrelated GPU workload during paired handoff study')

        return [run_case(backend=backend, target=target, noise_provider=provider, out=out,
                         prompt_id=prompt, seed=seed, chunks=chunks, ownership=ownership, phase=phase)
                for prompt in prompts for seed in seeds]

    with draft_backend('prefix') as draft:
        cases('prefix', backend=draft)
        torch = draft.torch
    del draft
    gc.collect()
    torch.cuda.empty_cache()

    with launch_target_generation_worker(
        run_dir=out, repo_root=repo, target_artifact_spec_path=target_spec,
        target_artifact_spec_sha256=target_artifact_spec_sha256(target_spec),
        target_verified_artifacts=target_files, visible_device=device, chunk_count=chunks,
    ) as target:
        cases('target', target=target)

    with draft_backend('continue') as draft:
        rows = cases('continue', backend=draft)

    write_json(out / 'result.json', {
        'schema': 'sf_handoff_continuity_v1', 'status': 'generation_complete',
        'smoke': smoke, 'chunk_count': chunks, 'rows': rows,
        'study': 'fixed schedule, SDVG-style Draft history control',
        'gpu_count': 1, 'visual_review': 'pending', 'speedup_measured': False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft-artifact-spec', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--smoke', action='store_true')
    group.add_argument('--seed-shard', type=int, choices=(0, 1))
    args = parser.parse_args()
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:
        run(find_git_root(Path.cwd()), _run_dir(), _model_root(), args.draft_artifact_spec,
            smoke=args.smoke, shard=args.seed_shard)
    finally:
        shutdown_owned_runtime_helpers(label='sf-handoff-continuity')


if __name__ == '__main__':
    main()
