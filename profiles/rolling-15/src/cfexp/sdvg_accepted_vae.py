"""Explicit selected-history SDVG scoring service, independent of diffusion KV.

Existing launchers do not install this service. Native CUDA and full reward
controller validation are separate requirements before a main-table campaign.
"""
import math

from .sdvg_adapter import SdvgDraftBackend
from .sdvg_vae_transaction import AcceptedVaeTransaction,native_owner_binding,require,sha

ENV='CFEXP_SDVG_ACCEPTED_VAE'
POLICY='sdvg_scoring_vae_selected_history_v2'
ACTION='commit_sdvg_selected_vae'


class AcceptedScoringVae:
    def __init__(self,backend,*,scope):
        require(type(backend) is SdvgDraftBackend,'selected VAE requires the known native SDVG reward wrapper')
        require(type(scope) is str and bool(scope),'selected VAE scope is missing')
        self.backend=backend;self.torch=backend.torch
        self.owner=backend.base.pipeline.vae.model
        self.binding=native_owner_binding(self.owner)
        self.scope=scope;self.transaction=None;self.prompt_id=None;self.prompt=None
        self.pending=None;self.sequence=0;self.failed=False

    def start(self,*,prompt_id,prompt,chunk_count):
        require(not self.failed and self.transaction is None,'selected VAE session is active or failed')
        require(type(chunk_count) is int and chunk_count==self.backend.chunk_count and chunk_count in (30,60,120),
                'selected VAE declared horizon differs')
        result=self.backend.start_sdvg_trajectory(prompt_id=prompt_id,prompt=prompt,chunk_count=chunk_count)
        self.prompt_id,self.prompt=prompt_id,prompt
        self.sequence+=1
        identity=sha(dict(policy=POLICY,scope=self.scope,prompt_id=prompt_id,
            prompt_sha256=sha(prompt),chunk_count=chunk_count,sequence=self.sequence))
        self.transaction=AcceptedVaeTransaction(owner=self.owner,torch=self.torch,session_id=identity)
        return dict(result,scoring_vae_policy=POLICY,decoder_owner=self.binding,vae_session_id=identity)

    def score(self,*,prompt,chunk_index,candidate_sha256,clean_latent):
        tx=self.transaction
        require(not self.failed and tx is not None and self.prompt==prompt and self.pending is None,
                'selected VAE score has no matching committed session')
        require(type(chunk_index) is int and 0<=chunk_index<self.backend.chunk_count,
                'selected VAE score index is outside the declared horizon')
        try:
            result,pending=tx.probe(chunk_index=chunk_index,candidate_sha256=candidate_sha256,
                decode_and_score=lambda:self.backend.score_sdvg_candidate(
                    prompt=prompt,chunk_index=chunk_index,clean_latent=clean_latent))
            rewards=result['frame_rewards'];expected=9 if chunk_index==0 else 12
            require(result.get('schema')=='sdvg_image_reward_block_v1' and result.get('chunk_index')==chunk_index
                and result.get('rgb_frame_count')==expected and type(rewards) is list and len(rewards)==expected
                and all(type(x) in (int,float) and math.isfinite(x) for x in rewards)
                and result.get('block_reward')==min(rewards) and result.get('threshold')==-.7
                and result.get('accepted_by_threshold') is (min(rewards)>=-.7)
                and result.get('forced_reject') is (chunk_index==0),'selected VAE reward decision differs')
            accepted=chunk_index!=0 and min(rewards)>=-.7
            self.pending=dict(pending,accepted_draft=accepted,block_reward=min(rewards))
            return dict(result,scoring_vae_policy=POLICY,vae_transaction=self.pending)
        except BaseException:
            # A bad reward is as unusable as a failed decode; no continuation.
            self.failed=True
            if tx.pending is not None:
                tx.pending['snapshot'].restore(self.owner,self.torch,max_bytes=tx.max_bytes)
                tx.failed=True
            raise

    def commit(self,*,prompt_id,chunk_index,pending_token,accepted,selected_role,
               selected_sha256,selected_latent=None):
        tx=self.transaction;pending=self.pending
        require(not self.failed and tx is not None and pending is not None
            and prompt_id==self.prompt_id and type(chunk_index) is int and chunk_index==tx.next_index
            and type(accepted) is bool and accepted is pending['accepted_draft']
            and selected_role==('draft' if accepted else 'target'),
            'selected VAE commit contradicts the bound reward/root decision')
        require((accepted and selected_latent is None) or (not accepted and selected_latent is not None),
                'selected VAE accept/reject tensor differs')
        def decode():
            with self.torch.inference_mode():
                video=self.backend.base.pipeline.vae.decode_to_pixel(
                    selected_latent.to(device=self.backend.base.device,dtype=self.torch.bfloat16).contiguous(),
                    use_cache=True)
                require(video.ndim==5 and tuple(video.shape[:3])==(1,9 if chunk_index==0 else 12,3),
                        'selected VAE Target decoder frame geometry differs')
                return video
        try:
            _,receipt=tx.commit(pending_token=pending_token,accepted=accepted,
                selected_sha256=selected_sha256,decode_selected=None if accepted else decode)
        except BaseException:
            # Invalid capabilities never advance state; actual decoding errors
            # are marked failed by the underlying transaction.
            self.failed=tx.failed;raise
        self.pending=None
        return dict(receipt,scoring_vae_policy=POLICY,prompt_id=prompt_id,selected_role=selected_role,
            block_reward=pending['block_reward'],threshold=-.7,forced_reject=chunk_index==0)

    def finish(self,*,prompt_id):
        require(not self.failed and self.transaction is not None and self.prompt_id==prompt_id,
                'selected VAE finish has no matching session')
        receipt=self.transaction.finish(expected_chunks=self.backend.chunk_count)
        result=self.backend.finish_sdvg_trajectory(prompt_id=prompt_id)
        self.transaction=None;self.prompt_id=None;self.prompt=None
        return dict(result,scoring_vae_policy=POLICY,vae_transaction=receipt)
