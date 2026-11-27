"""Sampler integration scope, separate from attention and trained CUDA gates.

The first-window primitive is substituted with independent sequential native
calls here. This tests the *unchanged original sampler*, not joint-model numeric
parity. Tests in test_rolling_joint_handoff cover native attention separately.
"""
import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_rolling_prefix_transaction import NativeTinyPipeline, torch
from cfexp.rolling_forcing_stream import RollingForcingStream
from cfexp.rolling_prefix_transaction import AcceptedPrefix
from cfexp.rolling_joint_stream import (JointCommittedSession, JointRollingObserver,
                                       JointRollingStream, RollingCleanWitness)
from cfexp.rolling_committed_prefix import CommittedPrefixSession
from cfexp.rolling_joint_handoff import ENV, _active
from cfexp.self_forcing_frame_adapter import sampled_kv_cache_witness


@unittest.skipIf(torch is None, 'actual CPU PyTorch is optional')
class JointStreamTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {ENV:'1','CFEXP_ROLLING_HOST_KV_CURSORS':'0',
                                          'CFEXP_ROLLING_HOST_WINDOW':'0'})
        self.env.start(); self.addCleanup(self.env.stop)

    def populated(self, history):
        p = NativeTinyPipeline(); condition = p.text_encoder(['moving subject'])
        p._initialize_kv_cache(batch_size=1,dtype=history.dtype,device=history.device)
        p._initialize_crossattn_cache(batch_size=1,dtype=history.dtype,device=history.device)
        for i in range(history.shape[1]//3):
            p.generator(noisy_image_or_video=history[:,i*3:(i+1)*3], conditional_dict=condition,
                timestep=torch.zeros(1,3), kv_cache=p.kv_cache_clean,
                crossattn_cache=p.crossattn_cache,current_start=i*6,updating_cache=True)
        return p, condition

    @staticmethod
    def sequential_primitive(p, *, torch, history, noise, conditional,
                             history_start, draft_start, clean_observer):
        # Independent native reference: assimilate accepted clean B3 groups,
        # then the first original noisy call. No candidate attention is used.
        for i in range(history.shape[1]//3):
            p.generator(noisy_image_or_video=history[:,i*3:(i+1)*3],
                conditional_dict=conditional,timestep=torch.zeros(1,3),
                kv_cache=p.kv_cache_clean,crossattn_cache=p.crossattn_cache,
                current_start=history_start+i*6,updating_cache=True)
        return p.generator(noisy_image_or_video=noise,conditional_dict=conditional,
            timestep=torch.full((1,3),1000.),kv_cache=p.kv_cache_clean,
            crossattn_cache=p.crossattn_cache,current_start=draft_start)[1]

    def assert_caches(self, a, b):
        for left,right in zip(a.kv_cache_clean,b.kv_cache_clean):
            for key in left:self.assertTrue(torch.equal(left[key],right[key]),key)
        self.assertEqual(sampled_kv_cache_witness(a.kv_cache_clean,a.crossattn_cache),
                         sampled_kv_cache_witness(b.kv_cache_clean,b.crossattn_cache))

    def compare_sampler(self, retained, suffix, remaining, host_window):
        history = torch.arange((retained+suffix)*3,dtype=torch.bfloat16).reshape(1,-1,1,1,1)*.01
        noise = torch.arange(remaining*3,dtype=torch.bfloat16).reshape(1,-1,1,1,1)*.02
        p,condition = self.populated(history[:,:retained*3])
        q,other_condition = self.populated(history)
        original = p.generator; pointers=[x['k'].data_ptr() for x in p.kv_cache_clean]
        with patch.dict(os.environ,{'CFEXP_ROLLING_HOST_WINDOW':str(int(host_window))}), torch.no_grad():
            torch.manual_seed(133)
            candidate = JointRollingStream(pipeline=p,torch=torch,noise=noise,prompt='moving subject',
                retained_history=history[:,:retained*3],target_suffix=history[:,retained*3:],condition=condition)
            torch.manual_seed(133)
            reference = RollingForcingStream(pipeline=q,torch=torch,noise=noise,prompt='moving subject',
                history=history,accepted_prefix=AcceptedPrefix(pipeline=q,torch=torch,history=history,
                    prompt='moving subject',condition=other_condition))
            outside_rng = torch.get_rng_state().clone()
            with patch('cfexp.rolling_joint_stream.joint_first_window',self.sequential_primitive) as primitive:
                for i in range(remaining):
                    ai,a = candidate.next_block(); bi,b = reference.next_block()
                    self.assertEqual(ai,bi);self.assertEqual(ai,retained+suffix+i)
                    self.assertTrue(torch.equal(a,b));self.assert_caches(p,q)
                    self.assertTrue(torch.equal(candidate.noisy_cache,reference.noisy_cache))
                    self.assertTrue(torch.equal(candidate.rng_state,reference.rng_state))
                    self.assertTrue(torch.equal(outside_rng,torch.get_rng_state()))
                    self.assertEqual(candidate.window,reference.window)
                    self.assertEqual(candidate.joint_calls,1)
                    self.assertIsNone(candidate.joint_suffix)
                    self.assertEqual(p.generator,original)
                with self.assertRaises(StopIteration): candidate.next_block()
                with self.assertRaises(StopIteration): reference.next_block()
            self.assertEqual(pointers,[x['k'].data_ptr() for x in p.kv_cache_clean])
            # Every original window retains its own clean warmup/update call,
            # including three terminal drain windows, regardless of tail length.
            self.assertEqual(p.calls[-2*(remaining+3):][1::2],q.calls[-2*(remaining+3):][1::2])
            candidate.close();reference.close()

    def test_complete_original_windows_drains_rng_and_cache_before_and_after_eviction(self):
        for retained,suffix,remaining in ((1,1,1),(2,4,2),(5,2,3),(8,4,6),(14,3,8)):
            for host_window in (False,True):
                with self.subTest(retained=retained,suffix=suffix,remaining=remaining,host=host_window):
                    self.compare_sampler(retained,suffix,remaining,host_window)

    def test_primitive_failure_restores_generator_and_invalidates_owner_without_retry(self):
        history=torch.ones(1,6,1,1,1,dtype=torch.bfloat16);p,c=self.populated(history)
        original=p.generator
        with torch.no_grad():
            stream=JointRollingStream(pipeline=p,torch=torch,noise=history,prompt='moving subject',
                retained_history=history,target_suffix=history,condition=c)
            with patch('cfexp.rolling_joint_stream.joint_first_window',side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError,'injected'):stream.next_block()
        self.assertEqual(p.generator,original);self.assertTrue(stream.closed)
        self.assertIsNone(p.kv_cache_clean);self.assertIsNone(p.crossattn_cache)

    def test_bad_suffix_or_cursor_rejected_without_mutating_prefix(self):
        history=torch.ones(1,6,1,1,1,dtype=torch.bfloat16);p,c=self.populated(history)
        before=copy.deepcopy(p.kv_cache_clean);calls=p.calls[:]
        with torch.no_grad():
            for suffix in (history[:,:2],history.repeat(1,3,1,1,1)):
                with self.assertRaises(ValueError):
                    JointRollingStream(pipeline=p,torch=torch,noise=history,prompt='moving subject',
                        retained_history=history,target_suffix=suffix,condition=c)
        self.assertEqual(calls,p.calls)
        for a,b in zip(before,p.kv_cache_clean):
            for key in a:self.assertTrue(torch.equal(a[key],b[key]))

    def witness_pipeline(self, blocks=2):
        p,c=self.populated(torch.ones(1,3*blocks,1,1,1,dtype=torch.bfloat16))
        p.kv_cache_clean=[copy.deepcopy(p.kv_cache_clean[0]) for _ in range(30)]
        p.crossattn_cache=[copy.deepcopy(p.crossattn_cache[0]) for _ in range(30)]
        return p

    def test_clean_witness_matches_original_descriptor_despite_later_noisy_writes(self):
        p=self.witness_pipeline();expected=sampled_kv_cache_witness(p.kv_cache_clean,p.crossattn_cache,
                                                                  batched=False,reuse_indices=False)
        witness=RollingCleanWitness(pipeline=p,torch=torch,expected_start=12)
        for i,row in enumerate(p.kv_cache_clean):
            witness.observe(i,row['k'],row['v'],12,12)
            row['k']+=9;row['v']-=9;row['global_end_index'].fill_(18)
        self.assertEqual(witness.finish(),expected)
        with self.assertRaises(ValueError):witness.finish()

    def test_policy_observer_excludes_clean_tokens_and_attaches_witness_after_all_layers(self):
        # Hook binding is covered by the native observer tests; directly invoke
        # this subclass's callbacks to isolate the five-group boundary and the
        # layer15-before-layer30 ordering that was missing in the ordinary hook.
        p=self.witness_pipeline(6);observer=JointRollingObserver.__new__(JointRollingObserver)
        observer.pipeline=p;observer.torch=torch;observer.frame_tokens=2
        observer.pending={};observer.active=None;observer.closed=False
        observer.clean_witness=None;observer.cache_witness=None
        observer.capture_cache=True;observer.retain_hidden=True;observer.handles=[]
        token=_active.set((36,4))
        try:
            observer._begin(None,(),dict(timestep=torch.tensor([[0.]*12+[1000.]*3]),
                current_start=12,updating_cache=False))
            hidden=torch.randn(1,30,16)
            for i,row in enumerate(p.kv_cache_clean):
                observer.observe_clean(i,row['k'],row['v'],36,36)
                if i==14:observer._observe(None,(),hidden)
            expected=sampled_kv_cache_witness(p.kv_cache_clean,p.crossattn_cache,
                                              batched=False,reuse_indices=False)
            observer._end(None,(),None);event=observer.take(6)
            self.assertTrue(torch.equal(event['hidden'],hidden[0,-6:].to(torch.bfloat16)))
            self.assertEqual(event['token_count'],6);self.assertEqual(event['kv_witness'],expected)
            rank=observer._rank_sample(observer._sample_hidden(hidden[:,-6:]))
            self.assertEqual(event['effective_rank'],rank.effective_rank)
            self.assertEqual(event['gram_eigenvalues'],rank.gram_eigenvalues)
            self.assertIsNone(observer.active);self.assertIsNone(observer.clean_witness)
        finally:_active.reset(token);observer.close()

    def test_keep_current_session_discards_future_then_matches_four_following_native_blocks(self):
        noise=torch.arange(90,dtype=torch.bfloat16).reshape(1,90,1,1,1)*.01
        a=JointCommittedSession(pipeline=NativeTinyPipeline(),torch=torch,noise=noise,
                                prompt='moving subject',seed=9)
        b=CommittedPrefixSession(pipeline=NativeTinyPipeline(),torch=torch,noise=noise,
                                  prompt='moving subject',seed=9)
        try:
            with torch.no_grad(), patch('cfexp.rolling_joint_stream.joint_first_window',self.sequential_primitive):
                history=torch.ones(1,3,1,1,1,dtype=torch.bfloat16)
                for _ in range(7):
                    actual=a.generate(history=history);expected=b.generate(history=history)
                    self.assertTrue(torch.equal(actual['latent'],expected['latent']))
                    history=torch.cat((history,actual['latent']),dim=1)
                history=torch.cat((history,torch.full((1,12,1,1,1),7.,dtype=torch.bfloat16)),dim=1)
                for i in range(4):
                    actual=a.generate(history=history);expected=b.generate(history=history)
                    self.assertTrue(torch.equal(actual['latent'],expected['latent']))
                    self.assert_caches(a.pipeline,b.pipeline)
                    self.assertTrue(torch.equal(a.stream.rng_state,b.stream.rng_state))
                    self.assertEqual(actual['joint_execution']['fused_clean_blocks'],4 if i==0 else 0)
                    self.assertFalse(actual['joint_execution']['trained_cuda_parity_established'])
                    history=torch.cat((history,actual['latent']),dim=1)
                self.assertEqual(a.stream.joint_calls,1)
        finally:a.close();b.close()


if __name__=='__main__':unittest.main()
