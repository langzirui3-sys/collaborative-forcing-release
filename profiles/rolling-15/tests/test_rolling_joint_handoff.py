"""Differential checks against pinned verbatim upstream Rolling attention.

CPU reference uses SDPA instead of the upstream CUDA attention dispatch. It
does not establish trained-model bit equality, policy capture or video speed.
"""
import copy
import hashlib
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cfexp.rolling_joint_handoff import (ENV, active_handoff, joint_attention,
    joint_first_window, native_attention, native_timestep_shapes)

try:
    import torch
except ImportError:
    torch = None

FIXTURE = Path(__file__).parent/'fixtures/rolling_eb8705_native_attention.py.txt'
FIXTURE_SHA = '973463b424b2955d88227e5b64e76fe89cefc91981b3f8cf4620854aa3a07b5d'


class FrozenRollingReference(unittest.TestCase):
    def test_reference_fixture_is_frozen_upstream_definitions(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), FIXTURE_SHA)
        self.assertIn('eb8705b2f00b7c361193b25aa3aae9e7edc94667', FIXTURE.read_text())
        self.assertNotIn('cfexp', FIXTURE.read_text())

    def test_primitive_is_not_selected_by_any_ordinary_factory(self):
        root = Path(__file__).resolve().parents[1]
        references = []
        for file in (root/'src/cfexp').glob('*.py'):
            if file.name not in ('rolling_joint_handoff.py', 'rolling_joint_stream.py',
                                  'rolling_joint_case_oracle.py', 'rolling_joint_case_profile.py',
                                  'rolling_joint_case_gate.py') and 'rolling_joint_handoff' in file.read_text():
                references.append(file.name)
        self.assertEqual(references, [])
        for file in (root/'src/cfexp').glob('*.py'):
            if file.name not in ('rolling_joint_stream.py', 'rolling_joint_case_oracle.py'):
                self.assertNotIn('rolling_joint_stream', file.read_text(), file.name)


@unittest.skipIf(torch is None, 'actual PyTorch is optional')
class NativeRollingJointCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.env = patch.dict(os.environ, {ENV: '1'}); self.env.start(); self.addCleanup(self.env.stop)
        def attention(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        self.native = dict(torch=torch, nn=torch.nn, math=math, attention=attention)
        exec(compile(FIXTURE.read_text(), str(FIXTURE), 'exec'), self.native)
        self.module = SimpleNamespace(**self.native)
        module_patch = patch.dict(sys.modules, {'wan.modules.causal_model': self.module})
        module_patch.start(); self.addCleanup(module_patch.stop)
        torch.manual_seed(145)
        self.layer = self.native['CausalWanSelfAttention'](24, 2, local_attn_size=21).eval()
        self.layer.frame_length = 2; self.layer.block_length = 6
        self.layer.max_attention_size = 42
        phase = torch.arange(256, dtype=torch.float64)[:, None]*torch.linspace(.01, .08, 6)[None, :]
        self.freqs = torch.polar(torch.ones_like(phase), phase)

    def empty_cache(self, blocks):
        return {'k':torch.zeros((1, blocks*6, 2, 12)),
                'v':torch.zeros((1, blocks*6, 2, 12)),
                'global_end_index':torch.zeros(1, dtype=torch.int64),
                'local_end_index':torch.zeros(1, dtype=torch.int64)}

    def call(self, layer, x, cache, start, updating):
        return layer(x, torch.tensor([x.shape[1]]), torch.tensor([[x.shape[1]//2, 1, 2]]),
                     self.freqs, None, kv_cache=cache, current_start=start, updating_cache=updating)

    def populate(self, prefix, capacity):
        cache = self.empty_cache(capacity)
        for index in range(prefix):
            self.call(self.layer, torch.randn(1, 6, 24), cache, index*6, True)
        return cache

    def compare_cache(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            if key in ('global_end_index', 'local_end_index'):
                self.assertTrue(torch.equal(left[key], right[key]), key)
            else:
                self.assertTrue(torch.equal(left[key], right[key]), key)

    def compare(self, prefix, suffix, capacity):
        with torch.no_grad():
            original = self.populate(prefix, capacity)
            expected_cache = copy.deepcopy(original); actual_cache = copy.deepcopy(original)
            original_anchor = original['k'][:, :6].clone()
            x = torch.randn(1, (suffix+1)*6, 24)
            expected, witness = [], None
            for group in range(suffix+1):
                expected.append(self.call(self.layer, x[:, group*6:(group+1)*6], expected_cache,
                    (prefix+group)*6, group < suffix))
                if group == suffix-1:
                    witness = copy.deepcopy(expected_cache)
            observed = []
            def observer(k, v, global_end, local_end):
                observed.append(dict(k=k.clone(), v=v.clone(),
                    global_end_index=torch.tensor([global_end]), local_end_index=torch.tensor([local_end])))
            rng = torch.get_rng_state().clone()
            actual = joint_attention(self.layer, x, torch.tensor([x.shape[1]]),
                torch.tensor([[x.shape[1]//2, 1, 2]]), self.freqs, None,
                kv_cache=actual_cache, current_start=prefix*6, clean_observer=observer)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(torch.equal(actual, torch.cat(expected, dim=1)))
            self.compare_cache(actual_cache, expected_cache)
            self.assertEqual(len(observed), 1); self.compare_cache(observed[0], witness)
            self.assertTrue(torch.equal(actual_cache['k'][:, :6], original_anchor), 'raw anchor must survive')
            # Resume original native warmup windows, not the candidate code:
            # three overlapping positions then one next accepted-block window.
            # Every window still performs its own clean first-block update.
            for width, offset in ((1, 0), (2, 0), (3, 0), (4, 1)):
                following = torch.randn(1, width*6, 24)
                left = self.call(self.layer, following, actual_cache, (prefix+suffix+offset)*6, False)
                right = self.call(self.layer, following, expected_cache, (prefix+suffix+offset)*6, False)
                self.assertTrue(torch.equal(left, right))
                self.compare_cache(actual_cache, expected_cache)
                # Values below stand in for the rest of the model, identical
                # on both paths; this test is only native attention/cache scope.
                clean = following[:, :6]*.8
                self.call(self.layer, clean, actual_cache, (prefix+suffix+offset)*6, True)
                self.call(self.layer, clean, expected_cache, (prefix+suffix+offset)*6, True)
                self.compare_cache(actual_cache, expected_cache)

    def test_upstream_clean_noisy_and_four_following_window_cache_comparisons(self):
        for prefix, suffix, capacity in ((1, 1, 7), (2, 4, 7), (5, 2, 7), (7, 4, 7),
                                         (13, 4, 7), (6, 3, 4), (14, 4, 10)):
            with self.subTest(prefix=prefix, suffix=suffix, capacity=capacity):
                self.compare(prefix, suffix, capacity)

    def test_noisy_future_cannot_change_clean_groups_or_history_witness(self):
        with torch.no_grad():
            original = self.populate(8, 7); x = torch.randn(1, 30, 24)
            def run(value):
                cache = copy.deepcopy(original); observed = []
                result = joint_attention(self.layer, value, torch.tensor([30]), torch.tensor([[15, 1, 2]]),
                    self.freqs, None, kv_cache=cache, current_start=48,
                    clean_observer=lambda k,v,g,l:observed.append((k.clone(), v.clone(), g, l)))
                return result, observed[0]
            left, a = run(x); changed = x.clone(); changed[:, -6:] = 9*torch.randn_like(changed[:, -6:])
            right, b = run(changed)
            self.assertTrue(torch.equal(left[:, :-6], right[:, :-6]))
            for i in range(2): self.assertTrue(torch.equal(a[i], b[i]))
            self.assertEqual(a[2:], b[2:])
            self.assertFalse(torch.equal(left[:, -6:], right[:, -6:]))

    def test_invalid_owner_geometry_and_training_rejected_before_cache_mutation(self):
        with torch.no_grad():
            original = self.populate(2, 7); x = torch.randn(1, 12, 24)
            kwargs = dict(kv_cache=original, current_start=12)
            for changes in ({'current_start':6}, {'cache_start':0}, {'updating_cache':True}):
                before = copy.deepcopy(original)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    joint_attention(self.layer, x, torch.tensor([12]), torch.tensor([[6, 1, 2]]),
                                    self.freqs, None, **{**kwargs, **changes})
                self.compare_cache(original, before)
            with patch.dict(os.environ, {ENV:'0'}), self.assertRaises(ValueError):
                joint_attention(self.layer, x, torch.tensor([12]), torch.tensor([[6, 1, 2]]),
                                self.freqs, None, **kwargs)
        with self.assertRaises(ValueError):
            joint_attention(self.layer, x, torch.tensor([12]), torch.tensor([[6, 1, 2]]),
                            self.freqs, None, **kwargs)

    def test_thirty_layer_hooks_restore_on_error_and_preexisting_patch_is_retained(self):
        model = SimpleNamespace(blocks=[SimpleNamespace(self_attn=copy.deepcopy(self.layer)) for _ in range(30)])
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with native_attention(model):
                self.assertTrue(all('forward' in vars(b.self_attn) for b in model.blocks))
                raise RuntimeError('injected')
        self.assertTrue(all('forward' not in vars(b.self_attn) for b in model.blocks))
        existing = lambda *a, **k:None; model.blocks[7].self_attn.forward = existing
        with self.assertRaises(ValueError), native_attention(model): pass
        self.assertIs(model.blocks[7].self_attn.forward, existing)

    def test_native_time_geometry_and_cleanup_preserve_original_three_row_calls(self):
        model = SimpleNamespace(time_embedding=torch.nn.Linear(8, 7),
                                time_projection=torch.nn.Linear(7, 6))
        clean = torch.randn(3, 8); noisy = torch.randn(3, 8)
        rows = torch.cat((clean.repeat(4, 1), noisy))
        expected_clean = model.time_projection(model.time_embedding(clean))
        expected_noisy = model.time_projection(model.time_embedding(noisy))
        with native_timestep_shapes(model, 12):
            actual = model.time_projection(model.time_embedding(rows))
        self.assertTrue(torch.equal(actual, torch.cat((expected_clean.repeat(4, 1), expected_noisy))))
        self.assertFalse('forward' in vars(model.time_embedding))
        self.assertFalse('forward' in vars(model.time_projection))
        changed = rows.clone(); changed[3] += 1
        with self.assertRaises(ValueError), native_timestep_shapes(model, 12):
            model.time_embedding(changed)
        self.assertFalse('forward' in vars(model.time_embedding))
        self.assertFalse('forward' in vars(model.time_projection))

    def test_model_boundary_marker_and_hooks_restore_after_native_forward_failure(self):
        model = SimpleNamespace(blocks=[SimpleNamespace(self_attn=copy.deepcopy(self.layer)) for _ in range(30)],
            time_embedding=torch.nn.Linear(8, 8), time_projection=torch.nn.Linear(8, 8))
        test = self
        class FailingGenerator:
            def __init__(self): self.model = model
            def __call__(self, **kwargs):
                test.assertEqual(active_handoff(), (18, 2))
                test.assertTrue(all('forward' in vars(x.self_attn) for x in model.blocks))
                test.assertEqual(kwargs['current_start'], 6)
                test.assertEqual(tuple(kwargs['timestep'][0].tolist()), (0.,)*6+(1000.,)*3)
                test.assertFalse(kwargs['updating_cache'])
                raise RuntimeError('native forward failed')
        caches = [dict(k=torch.zeros(1, 42, 2, 12), v=torch.zeros(1, 42, 2, 12),
            global_end_index=torch.tensor([6]), local_end_index=torch.tensor([6])) for _ in range(30)]
        p = SimpleNamespace(denoising_step_list=(1000.,937.5,833.3333129882812,625.),
            num_frame_per_block=3, independent_first_frame=False, args=SimpleNamespace(context_noise=0),
            frame_seq_length=2, kv_cache_clean=caches, crossattn_cache=[], generator=FailingGenerator())
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, 'native forward failed'):
            joint_first_window(p, torch=torch, history=torch.zeros(1,6,16,2,2,dtype=torch.bfloat16),
                noise=torch.zeros(1,3,16,2,2,dtype=torch.bfloat16), conditional={},
                history_start=6, draft_start=18)
        self.assertIsNone(active_handoff())
        self.assertTrue(all('forward' not in vars(x.self_attn) for x in model.blocks))
        self.assertFalse('forward' in vars(model.time_embedding))
        self.assertFalse('forward' in vars(model.time_projection))


if __name__ == '__main__': unittest.main()
