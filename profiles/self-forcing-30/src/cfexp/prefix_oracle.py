"""Independent native suffix oracle; full replay is diagnostic-only."""
from __future__ import annotations


def suffix_oracle(*, pipeline, torch, prompt, history, widths, noisy_suffix, noise_second, noise_third):
    p,t=pipeline,torch
    from .target_prefix import native_draft_schedule
    steps,_=native_draft_schedule(p)
    p._initialize_kv_cache(batch_size=1,dtype=t.bfloat16,device=noisy_suffix.device)
    p._initialize_crossattn_cache(batch_size=1,dtype=t.bfloat16,device=noisy_suffix.device)
    payload={'self':p.kv_cache1,'cross':p.crossattn_cache}
    try:
        with t.inference_mode():
            cond=p.text_encoder(text_prompts=[prompt]);n=len(widths)
            for i,width in enumerate(widths):
                p.generator(noisy_image_or_video=history[:,3*i:3*i+3].to(noisy_suffix.device),
                    conditional_dict=cond,timestep=t.zeros((1,width),device=noisy_suffix.device,dtype=t.int64),
                    kv_cache=p.kv_cache1,crossattn_cache=p.crossattn_cache,
                    current_start=3*i*p.frame_seq_length)
            noisy=noisy_suffix
            for i,step in enumerate(steps[1:]):
                _,clean=p.generator(noisy_image_or_video=noisy,conditional_dict=cond,
                    timestep=t.ones((1,3),device=noisy.device,dtype=t.int64)*step,kv_cache=p.kv_cache1,
                    crossattn_cache=p.crossattn_cache,current_start=3*n*p.frame_seq_length)
                if i<2:
                    noisy=p.scheduler.add_noise(clean.flatten(0,1),(noise_second,noise_third)[i].flatten(0,1),
                        steps[i+2]*t.ones((3,),device=noisy.device,dtype=t.long)).unflatten(0,clean.shape[:2])
            p.generator(noisy_image_or_video=clean,conditional_dict=cond,
                timestep=t.zeros((1,3),device=noisy.device,dtype=t.int64),kv_cache=p.kv_cache1,
                crossattn_cache=p.crossattn_cache,current_start=3*n*p.frame_seq_length)
        return clean,payload
    finally:
        p.kv_cache1=p.crossattn_cache=None


def check_suffix(*, actual, expected, actual_cache, expected_cache, torch):
    t=torch;differences=[]
    for kind in ('self','cross'):
        if len(actual_cache[kind])!=len(expected_cache[kind]):raise ValueError('oracle cache layer count drifted')
        for i,(a,b) in enumerate(zip(actual_cache[kind],expected_cache[kind])):
            if set(a)!=set(b):raise ValueError('oracle cache schema drifted')
            for key in a:
                equal=t.equal(a[key],b[key]) if isinstance(a[key],t.Tensor) else a[key]==b[key]
                if not equal:differences.append([kind,i,key])
    exact=bool(t.equal(actual,expected))
    maximum=float((actual.float()-expected.float()).abs().max().item())
    if not exact or differences:raise ValueError('native suffix/cache parity failed: '+str(differences[:10]))
    return {'latent_exact':exact,'maximum_absolute_difference':maximum,'cache_differences':differences,
            'oracle':'independent_full_prefix_replay_diagnostic_only','no_quality_or_speed_claim':True}
