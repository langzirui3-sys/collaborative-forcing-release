"""Experimental one-update Target handoff, never a completed Target block."""
from __future__ import annotations


def validate_prefix_trace(trace, *, history_frames):
    expected_start=min(history_frames,3)*1560
    expected=([0.0] if history_frames else [])+[1000.0]
    if (len(trace)!=len(expected)
            or [row.get('timestep') for row in trace]!=expected
            or any(row.get('current_start')!=expected_start
                   or row.get('batch_timesteps_equal') is not True for row in trace)):
        raise ValueError('Target prefix must use native context plus exactly timestep 1000')
    return [dict(trace[-1])]


def run_first_update(*, session, models, initial_noise, torch, device, history_frames):
    """Use native history preparation and one full Target forward.

    No candidate is installed in all_latents and the session cursor is not
    advanced. The next call must bind the caller's actual accepted prefix.
    """
    start=session.recompute_kv_cache(models)
    if start!=min(history_frames,3):
        raise ValueError('Target prefix context cursor drifted')
    session.conditional_dict['prompt_embeds']=session.current_prompt_embeds
    _,prediction=models.transformer(
        noisy_image_or_video=initial_noise,
        conditional_dict=session.conditional_dict,
        timestep=torch.full((initial_noise.shape[0],3),1000,device=device,dtype=torch.int64),
        kv_cache=models.pipeline.kv_cache1,
        crossattn_cache=models.pipeline.crossattn_cache,
        current_start=start*models.pipeline.frame_seq_length)
    return prediction


def native_draft_schedule(pipeline):
    """Validate without rounding or replacing the checkpoint's warped steps."""
    steps=pipeline.denoising_step_list
    values=[float(x) for x in steps]
    validate_draft_schedule(values)
    return steps,values


def validate_draft_schedule(values):
    import math
    if (not isinstance(values,list) or len(values)!=4
            or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in values)
            or values[0]!=1000 or not all(a>b>0 for a,b in zip(values,values[1:]))):
        raise ValueError('Target prefix requires four descending native Draft steps starting at 1000')


def validate_draft_suffix_evidence(result):
    values=result.get('draft_native_schedule')
    validate_draft_schedule(values)
    if result.get('draft_denoising_schedule')!=values[1:]:
        raise ValueError('Draft suffix differs from its resolved native schedule')
    return True
