"""Frozen cross-model collapse-delay study; ranks are observations, never gates."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SEED = 20260911
ARMS = ('draft_only', 'target_only', 'target_three_5s', 'target_three_10s')
PAIRS = ('longlive_krea', 'lingbot_world_2')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def tensor_digest(value, torch):
    """Bind the original dtype and all bits, including float32 LingBot noise."""
    tensor = value.detach().to(device='cpu').contiguous()
    h = hashlib.sha256((str(tuple(tensor.shape))+'\0'+str(tensor.dtype)).encode('ascii'))
    h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def conditioning_digest(value, torch):
    def bind(x):
        if isinstance(x, torch.Tensor):
            return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':tensor_digest(x,torch)}
        if isinstance(x,dict):return {k:bind(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [bind(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(bind(value),sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def geometry(pair):
    require(pair in PAIRS, 'unknown model pair')
    if pair == 'longlive_krea':
        return {'fps': 24, 'latent_chunk_frames': 3, 'chunk_count': 120,
                'rgb_frames': 1437, 'target_three_5s': [10, 11, 12],
                'target_three_10s': [20, 21, 22]}
    return {'fps': 16, 'latent_chunk_frames': 4, 'chunk_count': 60,
            'rgb_frames': 957, 'target_three_5s': [5, 6, 7],
            'target_three_10s': [10, 11, 12]}


def schedule(pair, arm):
    g = geometry(pair)
    require(arm in ARMS, 'unknown arm')
    if arm == 'draft_only':
        return ()
    if arm == 'target_only':
        return tuple(range(g['chunk_count']))
    return tuple(g[arm])


def video_interval(pair, chunk):
    g = geometry(pair)
    require(type(chunk) is int and 0 <= chunk < g['chunk_count'], 'invalid chunk')
    width = g['latent_chunk_frames']
    return [max(0, 4*width*chunk-3)/g['fps'],
            (4*width*(chunk+1)-3)/g['fps']]


def load_plan(path, expected, pair, prompt_index):
    require(sha(path) == expected, 'cross-model protocol digest drift')
    plan = json.loads(Path(path).read_text())
    require(plan['schema'] == 'cross_model_delay_6x1_v1' and plan['status'] == 'frozen', 'unfrozen plan')
    require(plan['model_pairs'] == list(PAIRS) and plan['arms'] == list(ARMS), 'model/arm drift')
    require(plan['seed'] == SEED and plan['nominal_seconds'] == 60, 'seed/horizon drift')
    require(plan['geometry'] == {p: geometry(p) for p in PAIRS}, 'geometry drift')
    require(type(prompt_index) is int and 0 <= prompt_index < 6, 'invalid prompt index')
    require(len(plan['prompts']) == 6 and len({x['prompt_id'] for x in plan['prompts']}) == 6, 'prompt population drift')
    for i, item in enumerate(plan['prompts']):
        require(isinstance(item['prompt'], str) and item['prompt'].strip(), 'empty prompt')
        require(item['lingbot_example'] == f'examples/{i:02d}', 'image/action mapping drift')
    geometry(pair)
    return plan, plan['prompts'][prompt_index]


class HiddenRankRecorder:
    """256 deterministic tokens, uncentered block-output singular entropy rank.

    begin/end explicitly bind a single producer's first denoising forward.
    Replay and clean-cache forwards must call with active=False.
    """
    def __init__(self, torch):
        self.torch = torch
        self.active = False
        self.value = None
        self.identity = None

    def begin(self, *, producer, chunk):
        require(self.identity is None, 'unfinished rank observation')
        self.identity = {'producer': producer, 'chunk': chunk, 'layer_index_zero_based': 14}
        self.value = None

    def observe(self, _module, _args, output):
        if not self.active:
            return
        require(self.identity is not None and self.value is None, 'unbound or duplicate rank capture')
        t = self.torch
        from .self_forcing_frame_adapter import deterministic_token_indices, entropy_effective_rank_from_gram_eigenvalues
        layout = 'hidden_tensor'
        if isinstance(output, tuple):
            # Pinned LongLive1.0 CausalWanAttentionBlock returns (x, info):
            # info=(current_end, local_end_index, cache_update_info). Observe x
            # without returning a hook replacement or touching the deferred KV
            # update metadata consumed by its parent Wan model.
            require(self.identity['producer'] == 'longlive_1.3b' and len(output) == 2
                    and isinstance(output[1], tuple) and len(output[1]) == 3,
                    'unrecognized native block output envelope')
            output = output[0]
            layout = 'longlive_hidden_and_cache_update_info'
        require(isinstance(output, t.Tensor) and output.ndim == 3 and output.shape[0] == 1, 'rank hidden layout')
        ids = deterministic_token_indices(int(output.shape[1]), 256)
        sample = output[0].index_select(0, t.tensor(ids, dtype=t.long, device=output.device)).detach().float()
        # Native LingBot runs under CUDA autocast. A float() sample alone
        # does not prevent matmul from being autocast back to bfloat16.
        with t.autocast(device_type=sample.device.type, enabled=False):
            gram = sample @ sample.T
            require(gram.dtype == t.float32, 'Gram must remain float32 under autocast')
            eigen = t.linalg.eigvalsh(gram).detach().cpu().double().tolist()
        rank = entropy_effective_rank_from_gram_eigenvalues(eigen, relative_cutoff=1e-7)
        require(math.isfinite(rank), 'nonfinite rank')
        self.value = {**self.identity, 'rank': rank, 'gram_eigenvalues': eigen,
                      'token_count': 256, 'relative_gram_cutoff': 1e-7,
                      'centered': False, 'gram_dtype': 'float32', 'hidden_shape': list(output.shape),
                      'native_block_output_layout': layout}

    def end(self):
        require(self.value is not None and self.identity is not None, 'missing rank observation')
        result = self.value
        self.value = self.identity = None
        self.active = False
        return result


def audit_video(path, frames, fps):
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        require(s.average_rate == fps, 'encoded FPS drift')
        pts = []
        for f in c.decode(s):
            require(f.pts is not None, 'missing frame PTS')
            pts.append(f.pts * s.time_base)
        require(len(pts) == frames, 'encoded frame count drift')
        from fractions import Fraction
        require(pts == [Fraction(i, fps) for i in range(frames)], 'encoded timestamps drift')
    return {'frames': frames, 'fps': fps, 'duration_seconds': frames/fps, 'sha256': sha(path),
            'bytes': Path(path).stat().st_size}
