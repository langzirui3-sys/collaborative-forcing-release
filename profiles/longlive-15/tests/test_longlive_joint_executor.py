"""Lease/RNG integration tests; the emulated joint call is not CUDA evidence."""
from contextlib import contextmanager
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cfexp.longlive_joint_executor import JointLongLiveExecutor
from cfexp.longlive_joint_handoff import ENV
from cfexp.longlive_chunk_session import LongLiveChunkSession
from test_longlive_chunk_session import Pipeline, torch


class JointOwnerContracts(unittest.TestCase):
    def test_diagnostic_cannot_be_enabled_with_an_unbound_or_closed_owner(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(ValueError,'opt-in'):
            JointLongLiveExecutor(None)
        with patch.dict(os.environ,{ENV:'1'},clear=True):
            for owner in (None,SimpleNamespace(executor=None),object.__new__(LongLiveChunkSession)):
                if type(owner) is LongLiveChunkSession:owner.closed=True
                with self.subTest(owner=type(owner).__name__),self.assertRaisesRegex(ValueError,'live native'):
                    JointLongLiveExecutor(owner)


@unittest.skipIf(torch is None,'optional CPU tensor/RNG/cache integration')
class JointLeaseCpu(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env=patch.dict(os.environ,{ENV:'1','CFEXP_LONGLIVE_HOST_CURSORS':'0'},clear=True)
        self.env.start();self.addCleanup(self.env.stop)
        self.p=Pipeline();self.q=Pipeline()
        self.session=LongLiveChunkSession(pipeline=self.p,torch=torch,scope_sha256='a'*64,max_bytes=4096)
        self.reference=LongLiveChunkSession(pipeline=self.q,torch=torch,scope_sha256='a'*64,max_bytes=4096)
        self.original=self.session.executor;self.cache=self.original.cache
        self.executor=JointLongLiveExecutor(self.session)
        self.emulated=[];self.contexts=[]
        def emulate(p,*,torch,history,noise,conditional,history_start,draft_start,clean_observer):
            # Emulate the primitive's *semantics* using the independent native
            # calls. Actual one-forward fusion is checked by the CUDA gate.
            self.emulated.append((history_start,draft_start,history.shape[1]//3))
            for j in range(history.shape[1]//3):
                p.generator(noisy_image_or_video=history[:,3*j:3*j+3],conditional_dict=conditional,
                    timestep=torch.zeros((1,3),dtype=torch.int64),kv_cache=p.kv_cache1,
                    crossattn_cache=p.crossattn_cache,current_start=history_start+3*j)
            return p.generator(noisy_image_or_video=noise,conditional_dict=conditional,
                timestep=torch.ones((1,3),dtype=torch.int64)*1000,kv_cache=p.kv_cache1,
                crossattn_cache=p.crossattn_cache,current_start=draft_start)[1]
        self.primitive=patch('cfexp.longlive_joint_executor.joint_first_step',side_effect=emulate)
        self.primitive.start();self.addCleanup(self.primitive.stop)

    def equal_payloads(self):
        a=self.executor.cache;b=self.reference.executor.cache
        self.assertEqual(a.retained_bindings,b.retained_bindings)
        for key in a.retained_bindings:
            sa=a._states[key];sb=b._states[key]
            self.assertEqual((sa.next_chunk,sa.next_token,sa.resident_bytes),
                             (sb.next_chunk,sb.next_token,sb.resident_bytes))
            for group in ('self','cross'):
                for x,y in zip(sa.payload[group],sb.payload[group]):
                    self.assertEqual(set(x),set(y))
                    for field in x:
                        if isinstance(x[field],torch.Tensor):self.assertTrue(torch.equal(x[field],y[field]))
                        else:self.assertEqual(x[field],y[field])

    def pair(self,history,seed=7,prompt='prompt'):
        noise=torch.full((1,3,1),.1,dtype=torch.bfloat16)
        @contextmanager
        def stages(count):
            self.contexts.append(count);yield
        torch.manual_seed(seed)
        expected=self.reference.generate(prompt=prompt,history=history,noise=noise)
        expected_rng=torch.get_rng_state().clone()
        torch.manual_seed(seed)
        actual=self.session.generate(prompt=prompt,history=history,noise=noise,forward_context=stages)
        self.assertTrue(torch.equal(actual['latent'],expected['latent']))
        self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
        for key in ('reused_chunks','replayed_chunks','history_binding','history_kv_widths','scope_sha256'):
            self.assertEqual(actual[key],expected[key])
        self.assertIsNone(self.p.kv_cache1);self.assertIsNone(self.p.crossattn_cache)
        self.equal_payloads();return actual

    def test_same_store_ownership_duplicate_install_and_close(self):
        self.assertIs(self.session.executor,self.executor);self.assertIs(self.executor.cache,self.cache)
        self.assertIs(self.original.cache,self.cache)
        with self.assertRaises(ValueError):JointLongLiveExecutor(self.session)
        self.pair(None);self.assertGreater(self.cache.resident_bytes,0)
        self.session.close();self.assertEqual(self.cache.resident_bytes,0)

    def test_native_120_block_continuation_and_two_external_four_block_bursts(self):
        history=None;fused=[]
        for index in range(120):
            if index in (8,48):
                history=torch.cat((history,torch.full((1,12,1),index/100,dtype=torch.bfloat16)),dim=1)
            row=self.pair(history,seed=index)
            if row['joint_execution']['fused_history_chunks']:fused.append(index)
            history=row['latent'] if history is None else torch.cat((history,row['latent']),dim=1)
        self.assertEqual(fused,[8,48]);self.assertEqual([x[2] for x in self.emulated],[4,4])

    def test_every_supported_suffix_and_long_suffix_native_fallback(self):
        history=self.pair(None)['latent']
        for width in (1,2,3,4,5):
            history=torch.cat((history,torch.full((1,3*width,1),.2,dtype=torch.bfloat16)),dim=1)
            before=len(self.emulated);row=self.pair(history,seed=width)
            fused=width if width<=4 else 0
            self.assertEqual(row['joint_execution']['fused_history_chunks'],fused)
            self.assertEqual(len(self.emulated)-before,int(width<=4))
            self.assertEqual(self.contexts[-1],0 if fused else width)
            history=torch.cat((history,row['latent']),dim=1)

    def test_changed_history_and_prompt_do_not_reuse_an_unrelated_prefix(self):
        root=self.pair(None)['latent'];history=torch.cat((root,torch.ones_like(root)),dim=1)
        self.pair(history)
        changed=history.clone();changed[:,:3]+=.2
        self.assertEqual(self.pair(changed)['reused_chunks'],0)
        self.assertEqual(self.pair(changed,prompt='different')['reused_chunks'],0)

    def test_joint_failure_discards_leased_state_and_clears_native_aliases(self):
        root=self.pair(None)['latent'];history=torch.cat((root,torch.ones_like(root)),dim=1)
        with patch('cfexp.longlive_joint_executor.joint_first_step',side_effect=RuntimeError('joint failed')):
            with self.assertRaisesRegex(RuntimeError,'joint failed'):
                self.session.generate(prompt='prompt',history=history,noise=torch.ones_like(root))
        self.assertIsNone(self.p.kv_cache1);self.assertIsNone(self.p.crossattn_cache)
        self.assertEqual(self.executor.cache.state_count,0)
        self.reference.executor.cache.clear();row=self.pair(history)
        self.assertEqual(row['reused_chunks'],0);self.assertEqual(row['joint_execution']['fused_history_chunks'],0)

    def test_width_one_branch_actions_and_changed_schedule_are_rejected(self):
        root=self.pair(None)['latent']
        for kw in (dict(history_kv_widths=(1,)),dict(history_kv_widths=(True,)),
                   dict(history_kv_widths=(3,),kv_branch_action='preserve')):
            with self.assertRaises(ValueError):
                self.executor.generate(prompt='prompt',history=root,noise=torch.ones_like(root),**kw)
        self.reference.executor.pipeline.denoising_step_list=(1000,750,500,250)
        with self.assertRaisesRegex(ValueError,'native LongLive'):
            JointLongLiveExecutor(self.reference)


if __name__=='__main__':unittest.main()
