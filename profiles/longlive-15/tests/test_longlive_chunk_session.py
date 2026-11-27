"""CPU branch/history tests; these do not establish native CUDA parity."""
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from cfexp.longlive_chunk_session import LongLiveChunkSession
from cfexp.longlive_native import LongLiveNativeError


class Pipeline:
    def __init__(self):
        self.num_frame_per_block = 3
        self.frame_seq_length = 1
        self.local_attn_size = 12
        self.is_lora_enabled = True
        self.args = SimpleNamespace(context_noise=0, model_kwargs=SimpleNamespace(
            local_attn_size=12, timestep_shift=5.0, sink_size=3))
        self.denoising_step_list = (1000., 937.5, 833.3333129882812, 625.)
        self.kv_cache1 = self.crossattn_cache = None
        self.window = None
        self.allocations = []
        self.calls = []
        self.generator = Generator(self)
        self.scheduler = SimpleNamespace(add_noise=lambda x, n, t: x * .9 + n * .01)

    def _set_all_modules_max_attention_size(self, value):
        self.window = value

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override):
        self.allocations.append(kv_cache_size_override)
        self.kv_cache1 = [dict(k=torch.zeros(1, kv_cache_size_override, 1, dtype=dtype),
                              v=torch.zeros(1, kv_cache_size_override, 1, dtype=dtype),
                              global_end_index=torch.tensor([0]), local_end_index=torch.tensor([0]))]

    def _initialize_crossattn_cache(self, **kwargs):
        self.crossattn_cache = [dict(k=torch.zeros(1), v=torch.zeros(1), is_init=False)]

    def text_encoder(self, text_prompts):
        return len(text_prompts[0]) * .001


class Generator:
    def __init__(self, owner):
        self.owner = owner
        self.model = SimpleNamespace(local_attn_size=None)

    def __call__(self, *, noisy_image_or_video, conditional_dict, timestep,
                 kv_cache, crossattn_cache, current_start):
        p = self.owner
        assert p.window == 12 and self.model.local_attn_size == 12
        assert tuple(timestep.shape) == (1, 3)
        p.calls.append((current_start, float(timestep.flatten()[0])))
        c = kv_cache[0]
        end = current_start + 3
        old, local = int(c['global_end_index']), int(c['local_end_index'])
        if end > old and local + 3 > c['k'].shape[1]:
            for key in ('k', 'v'):
                c[key][:, :-3] = c[key][:, 3:].clone()
            local -= 3
        local += end - old
        c['k'][:, local-3:local] = noisy_image_or_video
        c['v'][:, local-3:local] = noisy_image_or_video * .5
        c['global_end_index'].fill_(end)
        c['local_end_index'].fill_(local)
        crossattn_cache[0]['is_init'] = True
        return None, noisy_image_or_video * .5 + c['v'][:, :local].mean() * .1 + conditional_dict


def reference(history, noise, seed):
    p = Pipeline()
    p.window = p.generator.model.local_attn_size = 12
    p._initialize_kv_cache(1, noise.dtype, noise.device, 12)
    p._initialize_crossattn_cache()
    cond = p.text_encoder(['prompt'])
    n = 0 if history is None else history.shape[1] // 3
    def call(x, time, index):
        return p.generator(noisy_image_or_video=x, conditional_dict=cond,
                           timestep=torch.ones(1, 3)*time, current_start=index*3,
                           kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache)[1]
    for i in range(n):
        call(history[:, i*3:i*3+3], 0, i)
    torch.manual_seed(seed)
    x = noise
    for j, time in enumerate(p.denoising_step_list):
        clean = call(x, time, n)
        if j < 3:
            x = p.scheduler.add_noise(clean.flatten(0, 1), torch.randn_like(clean.flatten(0, 1)),
                                      p.denoising_step_list[j+1]).unflatten(0, clean.shape[:2])
    call(clean, 0, n)
    return clean


@unittest.skipIf(torch is None, 'requires optional CPU torch')
class LongLiveChunkTests(unittest.TestCase):
    def test_target_root_rolling_history_and_branch_rejection(self):
        p = Pipeline()
        session = LongLiveChunkSession(pipeline=p, torch=torch, scope_sha256='a'*64, max_bytes=4096)
        noise = torch.full((1, 3, 1), .1, dtype=torch.bfloat16)
        history = torch.full((1, 3, 1), .3, dtype=torch.bfloat16)  # external Target root
        for i in range(12):
            expected = reference(history, noise, i)
            torch.manual_seed(i)
            row = session.generate(prompt='prompt', history=history, noise=noise)
            self.assertTrue(torch.equal(row['latent'], expected))
            self.assertEqual(row['history_kv_widths'], (3,) * (i+1))
            self.assertEqual(row['reused_chunks'], 0 if i == 0 else i+1)
            history = torch.cat([history, row['latent']], dim=1)
        # Preserve an input branch, evaluate a changed Target suffix, then
        # return to the original continuation: neither branch may inherit the
        # other's cached latent values or global cursor.
        original = history
        changed = history.clone(); changed[:, -3:] += .2
        for j, prefix in enumerate((original, changed, original)):
            expected = reference(prefix, noise, 30+j)
            torch.manual_seed(30+j)
            row = session.generate(prompt='prompt', history=prefix, noise=noise,
                                   kv_branch_action='preserve' if j == 0 else 'none')
            self.assertTrue(torch.equal(row['latent'], expected))
            self.assertIsNone(p.kv_cache1)
            self.assertIsNone(p.crossattn_cache)
        self.assertEqual(set(p.allocations), {12})
        session.close()
        self.assertEqual(session.executor.cache.resident_bytes, 0)
        with self.assertRaisesRegex(LongLiveNativeError, 'closed'):
            session.generate(prompt='prompt', history=None, noise=noise)

    def test_reject_non_native_config_or_external_cache(self):
        for change in (lambda p: setattr(p, 'is_lora_enabled', False),
                       lambda p: setattr(p.args.model_kwargs, 'sink_size', 0),
                       lambda p: setattr(p, 'kv_cache1', [])):
            p = Pipeline(); change(p)
            with self.assertRaises(LongLiveNativeError):
                LongLiveChunkSession(pipeline=p, torch=torch, scope_sha256='a'*64)
            self.assertEqual(p.calls, [])

    def test_forbid_single_frame_history_prefill(self):
        p = Pipeline()
        session = LongLiveChunkSession(pipeline=p, torch=torch, scope_sha256='a'*64)
        with self.assertRaisesRegex(LongLiveNativeError, 'three-frame'):
            session.generate(prompt='prompt', history=torch.zeros(1,3,1),
                             noise=torch.zeros(1,3,1), history_kv_widths=(1,))
        self.assertEqual(p.calls, [])
