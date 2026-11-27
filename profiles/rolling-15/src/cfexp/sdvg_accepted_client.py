"""Selected-history reward-cache coordinator around the unchanged SDVG loop."""
from dataclasses import replace

from .sdvg_accepted_vae import POLICY,ACTION
from .sdvg_vae_transaction import require,sha,is_sha,SCHEMA


class AcceptedDraftClient:
    def __init__(self,client):
        require(client.role=='draft' and client.model_state.get('sdvg_scoring_vae_policy')==POLICY,
                'worker did not advertise selected-history scoring VAE')
        self.client=client;self.pending=None;self.commits=[];self.prompt_id=None;self.head=None
        self.vae_session=None;self.cache_head=None

    def __getattr__(self,name):return getattr(self.client,name)

    def start_sdvg_trajectory(self,**kwargs):
        require(self.prompt_id is None,'selected VAE client already active')
        result=self.client.start_sdvg_trajectory(**kwargs)
        require(result.get('scoring_vae_policy')==POLICY and is_sha(result.get('vae_session_id')),
                'selected VAE startup evidence missing')
        self.prompt_id=kwargs['prompt_id'];self.pending=None;self.commits=[];self.head=None
        self.vae_session=result['vae_session_id']
        self.cache_head=sha(dict(schema=SCHEMA,session_id=self.vae_session,accepted=[]))
        return result

    def score_sdvg_candidate(self,**kwargs):
        from .correction_runtime import _run_file,_sha256_file
        require(self.prompt_id is not None and self.pending is None,'previous selected VAE result uncommitted')
        result=self.client.score_sdvg_candidate(**kwargs)
        pending=result.get('vae_transaction');index=kwargs['chunk_index']
        digest=_sha256_file(_run_file(self.run_dir,kwargs['clean_latent_path'],must_exist=True))[0]
        require(result.get('scoring_vae_policy')==POLICY and isinstance(pending,dict)
            and pending.get('chunk_index')==index==len(self.commits) and pending.get('candidate_sha256')==digest
            and pending.get('accepted_prefix_head_sha256')==self.cache_head and is_sha(pending.get('pending_token'))
            and pending.get('accepted_draft') is (index!=0 and result['block_reward']>=result['threshold']),
            'scored VAE capability binding differs')
        self.pending=pending;return result

    def commit_selected_history(self,history):
        from .correction_state import LatentHistory
        from .correction_runtime import _run_file,_sha256_file
        require(type(history) is LatentHistory and self.pending is not None
            and history.prompt_id==self.prompt_id and history.next_chunk_index==len(self.commits)+1,
            'selected VAE history coverage differs')
        segment=history.segments[-1];pending=self.pending;accepted=pending['accepted_draft']
        previous=history.root_sha256 if self.head is None else self.head
        require(segment.parent_history_sha256==previous and segment.producer_role==('draft' if accepted else 'target')
            and _sha256_file(_run_file(self.run_dir,segment.latent_path,must_exist=True))[0]==segment.latent_sha256,
            'selected VAE history role/parent/file differs')
        result=self.client._request(ACTION,dict(prompt_id=self.prompt_id,chunk_index=segment.chunk_index,
            pending_token=pending['pending_token'],accepted=accepted,selected_role=segment.producer_role,
            selected_sha256=segment.latent_sha256,selected_latent_path=segment.latent_path))
        expected_head=sha(dict(schema=SCHEMA,previous=self.cache_head,
            chunk_index=segment.chunk_index,selected_sha256=segment.latent_sha256))
        require(result.get('scoring_vae_policy')==POLICY and result.get('chunk_index')==segment.chunk_index
            and result.get('session_id')==self.vae_session and result.get('schema')==SCHEMA
            and result.get('candidate_sha256')==pending['candidate_sha256']
            and result.get('pending_token')==pending['pending_token'] and result.get('accepted_draft') is accepted
            and result.get('selected_sha256')==segment.latent_sha256 and result.get('selected_role')==segment.producer_role
            and result.get('selected_latent_path')==segment.latent_path and result.get('prompt_id')==self.prompt_id
            and result.get('committed_blocks')==history.next_chunk_index
            and result.get('accepted_prefix_head_before')==self.cache_head
            and result.get('accepted_prefix_head_after')==expected_head
            and result.get('selected_decode_calls')==int(not accepted)
            and result.get('diffusion_kv_untouched') is True
            and result.get('operation')==('keep_scored_draft_cache' if accepted else
                'restore_prefix_then_decode_selected_target'),
            'selected VAE commit receipt differs from actual accepted history')
        self.head=history.head_sha256;self.cache_head=expected_head
        self.commits.append(dict(result));self.pending=None
        return result

    def finish_sdvg_trajectory(self,**kwargs):
        require(self.pending is None and len(self.commits)==self.chunk_count,'selected VAE commits are incomplete')
        result=self.client.finish_sdvg_trajectory(**kwargs)
        require(result.get('scoring_vae_policy')==POLICY and
            result.get('vae_transaction',{}).get('committed_blocks')==self.chunk_count and
            result['vae_transaction'].get('accepted_prefix_head_sha256')==self.cache_head,
            'selected VAE finish receipt missing')
        self.prompt_id=None;return result


def generate_selected_vae_trajectory(*,original=None,**kwargs):
    """Commit inside original timing before renderer append and next Draft call."""
    from .main_table_sdvg import generate_sdvg_trajectory
    draft=kwargs['draft'];require(type(draft) is AcceptedDraftClient,'selected VAE trajectory requires bound client')
    callback=kwargs.pop('on_commit',None)
    def commit(history):
        draft.commit_selected_history(history)
        if callback is not None:callback(history)
    generation=(original or generate_sdvg_trajectory)(**kwargs,on_commit=commit)
    require(len(draft.commits)==kwargs['chunk_count'] and draft.pending is None,'selected VAE trajectory commits incomplete')
    return replace(generation,result=dict(generation.result,scoring_vae_policy=POLICY,
        scoring_vae_commits=list(draft.commits),selected_vae_commit_in_generation_timing=True,
        scoring_and_output_renderers_separate=True))
