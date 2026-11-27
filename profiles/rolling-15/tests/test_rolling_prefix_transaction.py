"""Synthetic CPU safety/parity; the separate trained CUDA gate is mandatory."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from cfexp.rolling_forcing_session import RollingForcingSession
from cfexp.rolling_forcing_stream import RollingForcingError
from cfexp.rolling_clean_prefix_probe import compare_caches
from cfexp.rolling_prefix_transaction import PrefixTransaction, AcceptedPrefix
from cfexp.incremental_draft import latent_digest


class NativeTinyPipeline:
    num_frame_per_block = 3
    independent_first_frame = False
    args = SimpleNamespace(context_noise=0)
    frame_seq_length = 2

    def __init__(self):
        self.kv_cache_clean = self.crossattn_cache = None
        self.denoising_step_list = torch.tensor([1000.,937.5,833.3333129882812,625.])
        self.calls = []
        self.scheduler = SimpleNamespace(add_noise=lambda x,n,t: x*.9+n*.01+t.reshape(-1,1,1,1)*.0001)

    def text_encoder(self, text_prompts):
        return len(text_prompts[0])*.001

    def _initialize_kv_cache(self, *, batch_size, dtype, device):
        self.kv_cache_clean = [dict(k=torch.zeros(1,48,1,2,dtype=dtype,device=device),
            v=torch.zeros(1,48,1,2,dtype=dtype,device=device),
            global_end_index=torch.tensor([0],device=device),
            local_end_index=torch.tensor([0],device=device)) for _ in range(2)]

    def _initialize_crossattn_cache(self, *, batch_size, dtype, device):
        self.crossattn_cache = [dict(k=torch.zeros(1,2,1,2,dtype=dtype,device=device),
            v=torch.zeros(1,2,1,2,dtype=dtype,device=device), is_init=False) for _ in range(2)]

    def generator(self, *, noisy_image_or_video, conditional_dict, timestep,
                  kv_cache, crossattn_cache, current_start, updating_cache=False):
        self.calls.append((current_start, updating_cache))
        for layer,cross in zip(kv_cache,crossattn_cache):
            g,e=int(layer['global_end_index'].item()),int(layer['local_end_index'].item())
            new=current_start+6-g
            if new>0 and new+e>48:
                lost=new+e-48
                for name in ('k','v'):layer[name][:,6:48-lost]=layer[name][:,6+lost:48].clone()
                end=48
            else:end=e+new
            values=noisy_image_or_video[:,:3].reshape(1,3,1,1).repeat_interleave(2,dim=1)
            layer['k'][:,end-6:end]=values
            layer['v'][:,end-6:end]=-values
            if new>0:
                layer['global_end_index'].fill_(current_start+6)
                layer['local_end_index'].fill_(end)
            if not cross['is_init']:
                # Upstream cross attention replaces the initial placeholder
                # tensors on its first forward, rather than filling them.
                cross['k']=torch.full_like(cross['k'],conditional_dict)
                cross['v']=torch.full_like(cross['v'],-conditional_dict)
                cross['is_init']=True
        # Depend on accepted cache so a wrong undo affects future latents.
        context=kv_cache[0]['k'][:,:end].float().mean()
        return None,noisy_image_or_video*.5+conditional_dict+context.to(noisy_image_or_video.dtype)*.01


@unittest.skipIf(torch is None, 'requires optional CPU PyTorch')
class TransactionTests(unittest.TestCase):
    def session(self, *, enabled=True):
        return RollingForcingSession(pipeline=NativeTinyPipeline(),torch=torch,
            noise=torch.arange(48,dtype=torch.bfloat16).reshape(1,48,1,1,1)*.01,
            prompt='moving subject',seed=9,prefix_transactions=enabled)

    def accepted_prefix(self, session):
        history=torch.ones(1,3,1,1,1,dtype=torch.bfloat16)
        for index in range(1,9):
            row=session.generate(history=history)
            if index<8:history=torch.cat([history,row['latent']],dim=1)
        return history

    def test_eviction_rollback_four_block_suffix_and_native_tail_match_fresh_replay(self):
        candidate=self.session();reference=self.session(enabled=False)
        try:
            history=self.accepted_prefix(candidate)
            pointers=[x['k'].data_ptr() for x in candidate.pipeline.kv_cache_clean]
            history=torch.cat([history,torch.full((1,12,1,1,1),7.,dtype=torch.bfloat16)],dim=1)
            begin=len(candidate.pipeline.calls)
            actual=candidate.generate(history=history)
            expected=reference.generate(history=history)
            self.assertTrue(torch.equal(actual['latent'],expected['latent']))
            self.assertEqual(pointers,[x['k'].data_ptr() for x in candidate.pipeline.kv_cache_clean])
            proof=actual['prefix_transaction']
            self.assertEqual(proof['action'],'retained_after_rollback')
            self.assertEqual(proof['retained_prefix_blocks'],8)
            self.assertEqual(proof['clean_suffix_blocks'],4)
            self.assertEqual(proof['full_history_replay_blocks'],0)
            self.assertEqual(proof['rollback']['evicted_tokens_per_layer'],[6,6])
            self.assertEqual(proof['rollback']['copied_undo_bytes'],2*2*6*2*2)
            self.assertEqual(sum(clean for _,clean in candidate.pipeline.calls[begin:]),8)
            self.assertEqual(sum(clean for _,clean in reference.pipeline.calls),16)
            for index in range(13,16):
                history=torch.cat([history,actual['latent']],dim=1)
                actual=candidate.generate(history=history);expected=reference.generate(history=history)
                self.assertTrue(torch.equal(actual['latent'],expected['latent']))
            self.assertTrue(compare_caches((candidate.pipeline.kv_cache_clean,candidate.pipeline.crossattn_cache),
                (reference.pipeline.kv_cache_clean,reference.pipeline.crossattn_cache),torch=torch,layers=2)['equal'])
            self.assertTrue(torch.equal(candidate.stream.rng_state,reference.stream.rng_state))
        finally:candidate.close();reference.close()

    def test_eviction_rollback_with_host_cursors_and_host_window_preserves_native_tail(self):
        import os
        with patch.dict(os.environ,{'CFEXP_ROLLING_HOST_KV_CURSORS':'1',
                                    'CFEXP_ROLLING_HOST_WINDOW':'1'}):
            self.test_eviction_rollback_four_block_suffix_and_native_tail_match_fresh_replay()

    def test_changed_earlier_history_falls_back_to_explicit_full_replay(self):
        candidate=self.session();reference=self.session(enabled=False)
        try:
            history=self.accepted_prefix(candidate);history[:,0]+=1
            actual=candidate.generate(history=history);expected=reference.generate(history=history)
            self.assertEqual(actual['prefix_transaction']['action'],'full_replay')
            self.assertEqual(actual['replayed_history_blocks'],8)
            self.assertTrue(torch.equal(actual['latent'],expected['latent']))
        finally:candidate.close();reference.close()

    def test_root_replacement_restores_uninitialized_conditioning(self):
        candidate=self.session();reference=self.session(enabled=False)
        try:
            candidate.generate()
            root=torch.full((1,3,1,1,1),3.,dtype=torch.bfloat16)
            actual=candidate.generate(history=root);expected=reference.generate(history=root)
            self.assertEqual(actual['prefix_transaction']['retained_prefix_blocks'],0)
            self.assertEqual(actual['prefix_transaction']['clean_suffix_blocks'],1)
            self.assertTrue(torch.equal(actual['latent'],expected['latent']))
        finally:candidate.close();reference.close()

    def test_undo_prevalidation_and_one_shot_ownership_fail_closed(self):
        p=NativeTinyPipeline();p._initialize_kv_cache(batch_size=1,dtype=torch.bfloat16,device='cpu')
        p._initialize_crossattn_cache(batch_size=1,dtype=torch.bfloat16,device='cpu')
        undo=PrefixTransaction(pipeline=p,torch=torch,prefix_blocks=0,history_digest=None)
        for row in p.kv_cache_clean:
            row['global_end_index'].fill_(6);row['local_end_index'].fill_(6)
        p.kv_cache_clean[-1]['global_end_index'].fill_(12)
        before=copy.deepcopy(p.kv_cache_clean)
        with self.assertRaisesRegex(RollingForcingError,'not immediately after one candidate'):undo.rollback()
        self.assertTrue(torch.equal(before[0]['global_end_index'],p.kv_cache_clean[0]['global_end_index']))
        p.kv_cache_clean[-1]['global_end_index'].fill_(6);undo.rollback()
        with self.assertRaisesRegex(RollingForcingError,'already consumed'):undo.rollback()

    def test_inference_mode_call_allocates_versioned_cache_and_preserves_rng(self):
        candidate=self.session()
        try:
            rng=torch.get_rng_state().clone()
            with torch.inference_mode():history=self.accepted_prefix(candidate)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for row in candidate.pipeline.crossattn_cache:self.assertIsInstance(row['k']._version,int)
        finally:candidate.close()

    def test_accepted_prefix_capability_is_history_owner_bound_and_one_shot(self):
        session=self.session()
        try:
            history=torch.ones(1,3,1,1,1,dtype=torch.bfloat16);out=session.generate(history=history)
            history=torch.cat([history,out['latent']],dim=1)
            lease=AcceptedPrefix(pipeline=session.pipeline,torch=torch,history=history,
                prompt='moving subject',condition=session.stream.condition)
            with self.assertRaises(RollingForcingError):lease.consume(pipeline=session.pipeline,torch=torch,history=history+1,prompt='moving subject')
            lease.consume(pipeline=session.pipeline,torch=torch,history=history,prompt='moving subject')
            with self.assertRaises(RollingForcingError):lease.consume(pipeline=session.pipeline,torch=torch,history=history,prompt='moving subject')
        finally:session.close()

    def test_pure_draft_inference_mode_keeps_new_cross_tensors_versioned(self):
        candidate=self.session();reference=self.session(enabled=False)
        history=None
        try:
            for _ in range(5):
                with torch.inference_mode():actual=candidate.generate(history=history)
                with torch.inference_mode():expected=reference.generate(history=history)
                self.assertTrue(torch.equal(actual['latent'],expected['latent']))
                for row in candidate.pipeline.crossattn_cache:self.assertIsInstance(row['k']._version,int)
                history=actual['latent'] if history is None else torch.cat([history,actual['latent']],dim=1)
        finally:candidate.close();reference.close()


class FormalGateContractTests(unittest.TestCase):
    def test_gpu_ownership_records_exact_owners_and_idle_device(self):
        from cfexp.rolling_transaction_gate import verify_gpu_ownership
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            verify_gpu_ownership(root,0,['GPU-a','GPU-b'],[101,202],
                                 query=lambda uuid: {101} if uuid=='GPU-a' else set())
            record=json.loads((root/'provenance/gpu-ownership-000.json').read_text())
            self.assertTrue(record['exclusive'])
            self.assertEqual(record['devices'][0]['actual_pids'],[101])
            self.assertEqual(record['devices'][1]['allowed_pids'],[202])

    def test_gpu_ownership_rejects_even_an_owner_child_and_records_before_failure(self):
        from cfexp.rolling_transaction_gate import verify_gpu_ownership
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('cfexp.rolling_transaction_gate.process_identity',
                       side_effect=lambda pid:dict(pid=pid,parent_pid=101)):
                with self.assertRaisesRegex(ValueError,r'actual=\[101, 303\] allowed=\[101\]'):
                    verify_gpu_ownership(root,1,['GPU-a','GPU-b'],[101,202],
                        query=lambda uuid:{101,303} if uuid=='GPU-a' else {202})
            record=json.loads((root/'provenance/gpu-ownership-001.json').read_text())
            self.assertFalse(record['exclusive'])
            self.assertEqual(record['devices'][0]['unexpected_pids'],[303])
            self.assertEqual(record['devices'][0]['identities'][1]['parent_pid'],101)

    def test_process_identity_has_no_argv_or_environment(self):
        import os
        from cfexp.rolling_transaction_gate import process_identity
        identity=process_identity(os.getpid())
        self.assertEqual(identity['pid'],os.getpid())
        self.assertLessEqual(set(identity),{'pid','name','state','parent_pid','process_group',
                                          'session','start_ticks','executable','identity_unavailable'})



if __name__ == '__main__':unittest.main()
