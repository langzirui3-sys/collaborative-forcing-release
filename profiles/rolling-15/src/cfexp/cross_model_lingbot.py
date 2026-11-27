"""Route the pinned LingBot native loop between independent model KV caches.

No RGB splice, VAE round trip, or cross-model KV tensor reuse is performed.
Target reconstructs the exact committed clean prefix using its own architecture
and conditioning, then denoises at its native shift. Draft commits every
accepted Target chunk with its own native clean-cache forward before resuming.
"""
from __future__ import annotations

import gc
import importlib
import os
from pathlib import Path
import sys
import time

from .cross_model_delay import (SEED, ARMS, HiddenRankRecorder, audit_video,
    require, schedule, sha, video_interval, write_new)
from .cross_model_delay import tensor_digest as latent_digest, conditioning_digest


def move(value, device, torch):
    if isinstance(value,torch.Tensor):return value.to(device=device)
    if isinstance(value,dict):return {k:move(v,device,torch) for k,v in value.items()}
    if isinstance(value,list):return [move(v,device,torch) for v in value]
    if isinstance(value,tuple):return tuple(move(v,device,torch) for v in value)
    return value


def clean_condition(kw, torch):
    return move({k:v for k,v in kw.items() if k not in ('x','t','kv_cache','crossattn_cache','cross_attn_first_call')},'cpu',torch)


def build_pair(output, repository, model_root):
    from .lingbot_smoke import materialize, strict_dit_loader
    prepared={};bound={}
    for size in ('1.3b','14b'):
        root=output/f'materialize-{size}';root.mkdir()
        source,site,paths,inputs,receipt=materialize(repository,model_root,root,size,60)
        prepared[size]=(source,site,paths,inputs,receipt)
    source,site=prepared['1.3b'][:2]
    for name in ('wan','flash_attn'):
        require(name not in sys.modules,'dependency imported before source binding')
    sys.path[:0]=[str(site),str(source)]
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    import flash_attn
    import wan.image2video as module
    from wan.configs import WAN_CONFIGS
    require(flash_attn.__version__=='2.8.3' and Path(module.__file__).resolve().is_relative_to(source),'unbound LingBot runtime')
    for index,size in enumerate(('1.3b','14b')):
        paths=prepared[size][2];loads=[];original=module.load_dit_model
        module.load_dit_model=strict_dit_loader(module,loads)
        try:
            with torch.cuda.device(index):
                pipe=module.WanI2VCausal(config=WAN_CONFIGS['i2v-1.3B' if size=='1.3b' else 'i2v-A14B'],
                    checkpoint_dir=str(paths['checkpoint']),assets_dir=str(paths['shared'] if size=='1.3b' else paths['checkpoint']),
                    device_id=index,rank=0,t5_fsdp=False,dit_fsdp=False,use_sp=False,t5_cpu=False,
                    convert_model_dtype=True,local_attn_size=18,sink_size=6,infer_mode='causal_fast')
        finally:module.load_dit_model=original
        require(len(loads)==1,'strict model loader count')
        expected=(1536,12,30) if size=='1.3b' else (5120,40,40)
        require((pipe.model.config.dim,pipe.model.config.num_heads,pipe.model.config.num_layers)==expected,'LingBot model architecture mismatch')
        bound[size]=pipe
        write_new(output/'provenance'/f'{size}-load.json',{'native_load':loads[0],
            'native_source':str(source),'materialization':str(output/f'materialize-{size}'/'provenance/materialization.json')})
    require(prepared['1.3b'][4]==prepared['14b'][4],'model input preparation differs')
    return torch,bound,prepared['1.3b'][3],prepared['1.3b'][4]


class NativeRouter:
    def __init__(self, *, driver, target, torch, arm, directory, reference, ownership):
        self.driver,self.target,self.t=driver,target,torch
        self.arm,self.directory,self.reference,self.ownership=arm,directory,reference,ownership
        self.targets=schedule('lingbot_world_2','draft_only' if arm=='target_root_only' else arm)
        self.rows=[];self.noises=[];self.history=[];self.conditions=[]
        self.index=None;self.step=0;self.recorder=HiddenRankRecorder(torch)
        self.pending=None;self.target_cache=None;self.target_cross=None;self.target_initialized=False
        self.target_prefix=[];self.actual_initial=[]

    def synchronize(self):
        self.t.cuda.synchronize(self.driver.device)
        if self.target is not self.driver:self.t.cuda.synchronize(self.target.device)

    def target_kwargs(self, kw):
        result=move(clean_condition(kw,self.t),self.target.device,self.t)
        result.update(kv_cache=self.target_cache,crossattn_cache=self.target_cross,
                      cross_attn_first_call=not self.target_initialized)
        return result

    def target_denoising_args(self, kw):
        # The native scheduler keeps its timestep table on CPU. Upstream
        # generate() explicitly transfers the stacked timestep to the DiT GPU;
        # crossing to a second model must preserve that same device contract.
        result = self.target_kwargs(kw)
        result.update(x=move(kw['x'],self.target.device,self.t),
                      t=self.target_times[self.step].reshape(1).to(self.target.device))
        require(result['t'].device == self.target.device, 'Target timestep must reside with its model')
        return result

    def initialize_target(self, kw):
        if self.target_cache is not None:return
        t,p=self.t,self.target
        config=p.model.config;head=config.dim//config.num_heads
        self.target_cache=p._initialize_self_kv_cache(num_layers=config.num_layers,
            shape=[1,kw['frame_seqlen']*p.local_attn_size,config.num_heads,head],dtype=p.pipe_dtype,device=p.device)
        self.target_cross=p._initialize_crossattn_cache(num_layers=config.num_layers,
            shape=[1,512,config.num_heads,head],dtype=p.pipe_dtype,device=p.device)
        p.scheduler.set_timesteps(p.num_train_timesteps,shift=10.0)
        self.target_times=p.scheduler.timesteps[[0,179,358,679]]
        for j,(latent,condition) in enumerate(zip(self.history,self.conditions,strict=True)):
            args=self.target_kwargs(condition)
            args.update(x=[latent.to(p.device)],t=t.zeros(1,device=p.device,dtype=self.target_times.dtype))
            self.target_forward(**args)
            self.target_initialized=True
            require(all(int(c['global_end_index'].item())==(j+1)*4*kw['frame_seqlen'] for c in self.target_cache),'Target replay cursor drift')
            self.target_prefix.append(latent_digest(latent,t))
        require(len(self.target_prefix)==self.index,'Target did not replay entire committed history')

    def __enter__(self):
        p=self.driver
        self.native_forward=p.model.forward
        self.target_forward=self.target.model.forward
        self.native_convert=p._convert_flow_pred_to_x0
        self.native_add=p.scheduler.add_noise
        self.handles=[p.model.blocks[14].register_forward_hook(self.recorder.observe)]
        if p is not self.target:self.handles.append(self.target.model.blocks[14].register_forward_hook(self.recorder.observe))
        p.model.forward=self.forward
        p._convert_flow_pred_to_x0=self.convert
        p.scheduler.add_noise=self.add_noise
        return self

    def __exit__(self, *_):
        self.driver.model.forward=self.native_forward
        self.driver._convert_flow_pred_to_x0=self.native_convert
        self.driver.scheduler.add_noise=self.native_add
        for h in self.handles:h.remove()
        self.target_cache=self.target_cross=None

    def forward(self,*args,**kw):
        t,p=self.t,self.driver
        i=kw['current_start']//(4*kw['frame_seqlen'])
        value=float(kw['t'].reshape(-1)[0])
        routed=p is not self.target and i in self.targets
        if value>0:
            if self.index!=i:
                require(len(self.rows)==i,'native LingBot chunk order drift')
                self.ownership();self.index=i;self.step=0;self.synchronize();self.started=time.perf_counter()
            require(self.step<4,'native denoising count drift')
            if self.step==0:
                self.actual_initial.append(latent_digest(kw['x'][0],t))
                self.recorder.begin(producer='lingbot_14b' if (routed or p is self.target) else 'lingbot_1.3b',chunk=i)
            if routed:
                self.initialize_target(kw)
                target_kw=self.target_denoising_args(kw)
                self.recorder.active=self.step==0
                with t.cuda.device(self.target.device),t.amp.autocast('cuda',dtype=self.target.param_dtype):
                    result=self.target_forward(**target_kw)
                self.target_initialized=True
                result=move(result,p.device,t)
            else:
                self.recorder.active=self.step==0
                result=self.native_forward(*args,**kw)
            self.recorder.active=False
            if self.step==0:self.pending=self.recorder.end()
            self.step+=1
            return result
        require(self.index==i and self.step==4,'unexpected native clean update')
        latent=kw['x'][0]
        if routed:
            target_kw=self.target_kwargs(kw)
            target_kw.update(x=move(kw['x'],self.target.device,t),t=t.zeros(1,device=self.target.device,dtype=kw['t'].dtype))
            with t.cuda.device(self.target.device),t.amp.autocast('cuda',dtype=self.target.param_dtype):
                self.target_forward(**target_kw)
            require(all(int(c['global_end_index'].item())==(i+1)*4*kw['frame_seqlen'] for c in self.target_cache),'Target committed cursor drift')
        # On routed chunks this single native Draft clean forward is the actual
        # cross-model correction of Draft's history. No old Draft proposal exists.
        result=self.native_forward(*args,**kw)
        require(all(int(c['global_end_index'].item())==(i+1)*4*kw['frame_seqlen'] for c in kw['kv_cache']),'driver clean KV cursor drift')
        clean=latent.detach().cpu().contiguous()
        require(bool(t.isfinite(clean).all()), 'nonfinite LingBot clean latent')
        from safetensors.torch import save_file,load_file
        save_file({'latent':clean},str(self.directory/f'{i:03d}.safetensors'))
        exact=None
        if self.reference is not None and i<min(self.targets,default=60):
            exact=t.equal(load_file(str(self.reference/f'{i:03d}.safetensors'))['latent'],clean)
            require(exact,'native pre-intervention prefix differs')
        self.history.append(clean);self.conditions.append(clean_condition(kw,t))
        condition_hash=conditioning_digest(self.conditions[-1],t)
        self.synchronize()
        self.rows.append({'chunk':i,'producer':'target' if i in self.targets else 'draft',
            'video_interval_seconds':video_interval('lingbot_world_2',i),
            'generation_with_instrumentation_seconds':time.perf_counter()-self.started,
            'latent_sha256':latent_digest(clean,t),'erank_observation':self.pending,
            'native_prefix_exact':exact,'native_clean_cache_update':True,'conditioning_sha256':condition_hash,
            'target_prefix_replayed_chunks':len(self.target_prefix) if routed else None,
            'target_prefix_sha256':list(self.target_prefix) if routed and i==min(self.targets) else None,
            'target_actual_timesteps':[float(x) for x in self.target_times] if routed else None})
        print('CROSS_MODEL_CHUNK','lingbot_world_2',self.arm,i,flush=True)
        return result

    def convert(self,*,flow_pred,xt,timestep,scheduler):
        if self.driver is not self.target and self.index in self.targets:
            p=self.target
            with self.t.cuda.device(p.device):
                result=p._convert_flow_pred_to_x0(flow_pred=flow_pred.to(p.device),xt=xt.to(p.device),
                    timestep=self.target_times[self.step-1],scheduler=p.scheduler)
            return result.to(self.driver.device)
        return self.native_convert(flow_pred=flow_pred,xt=xt,timestep=timestep,scheduler=scheduler)

    def add_noise(self,clean,noise,timestep):
        self.noises.append({'chunk':self.index,'step':self.step-1,'sha256':latent_digest(noise,self.t)})
        if self.driver is not self.target and self.index in self.targets:
            p=self.target
            with self.t.cuda.device(p.device):
                result=p.scheduler.add_noise(clean.to(p.device),noise.to(p.device),self.target_times[self.step])
            return result.to(self.driver.device)
        return self.native_add(clean,noise,timestep)


def generate_arm(*, pipes, torch, item, action, output, arm, reference, ownership, target_chunks=None):
    from PIL import Image
    from torchvision.io import write_video
    from .lingbot_native import _TimedCallable
    p=pipes['14b' if arm=='target_only' else '1.3b']
    target=pipes['14b'];directory=output/'arms'/arm;directory.mkdir(parents=True,exist_ok=False)
    original_text=p.text_encoder;original_encode=p.vae.encode;original_decode=p.vae.decode
    stages={'text_seconds':0.0,'image_encode_seconds':0.0,'decode_seconds':0.0}
    def measure(name,fn,*a,**kw):
        torch.cuda.synchronize(p.device);start=time.perf_counter();result=fn(*a,**kw)
        torch.cuda.synchronize(p.device);stages[name]+=time.perf_counter()-start;return result
    p.text_encoder=_TimedCallable(original_text,lambda fn,*a,**kw:measure('text_seconds',fn,*a,**kw))
    p.vae.encode=lambda *a,**kw:measure('image_encode_seconds',original_encode,*a,**kw)
    p.vae.decode=lambda *a,**kw:measure('decode_seconds',original_decode,*a,**kw)
    p._t5_cache.clear()
    try:
        with NativeRouter(driver=p,target=target,torch=torch,arm=arm,directory=directory,
                          reference=reference,ownership=ownership) as router:
            if target_chunks is not None:
                require(tuple(target_chunks)==tuple(sorted(set(target_chunks))) and all(type(i) is int and 0<=i<60 for i in target_chunks),'invalid explicit Target chunks')
                router.targets=tuple(target_chunks)
            with Image.open(action/'image.jpg') as image,torch.cuda.device(p.device):
                start=time.perf_counter()
                video=p.generate(item['prompt'],image.convert('RGB'),action_path=str(action),chunk_size=4,
                    max_area=480*832,frame_num=957,timesteps_index=[0,179,358,679],
                    shift=10.0 if arm=='target_only' else 5.0,seed=SEED,offload_model=False,max_attention_size=None)
                router.synchronize();pipeline_seconds=time.perf_counter()-start
            require(len(router.rows)==60 and len(router.noises)==180,'native LingBot trajectory coverage')
            require([r['chunk'] for r in router.rows if r['producer']=='target']==list(schedule('lingbot_world_2',arm) if target_chunks is None else target_chunks),'forced Target count drift')
            if reference is not None and arm!='target_only':
                import json
                native=json.loads((reference/'trace.json').read_text())
                require(router.noises==native['renoise'] and router.actual_initial==native['initial_chunk_noise_sha256'],'actual shared noise drift')
            write_new(directory/'trace.json',{'chunks':router.rows,'renoise':router.noises,
                'initial_chunk_noise_sha256':router.actual_initial,'native_source_loop':True,
                'independent_model_kv':True,'rank_routes_generation':False})
            rows=router.rows
        require(video.ndim==4 and video.shape[0]==3 and video.shape[1]==957,'native video horizon')
        frames=video.add(1).mul(127.5).round().clamp(0,255).to(device='cpu',dtype=torch.uint8).permute(1,2,3,0).contiguous()
        start=time.perf_counter()
        write_video(str(directory/'video.mp4'),frames,fps=16,video_codec='libx264',options={'crf':'18'})
        encode=time.perf_counter()-start
        for second in range(60):Image.fromarray(frames[second*16].numpy()).resize((416,240)).save(directory/f'second-{second:02d}.png')
        result={'arm':arm,'video':audit_video(directory/'video.mp4',957,16),'chunk_count':60,
            'generation_with_instrumentation_seconds':sum(r['generation_with_instrumentation_seconds'] for r in rows),
            'pipeline_wall_seconds':pipeline_seconds,**stages,'encode_seconds':encode,
            'visual_review_status':'pending_rank_hidden_review'}
        write_new(directory/'result.json',result)
        del frames,video
        return result
    finally:
        p.text_encoder=original_text;p.vae.encode=original_encode;p.vae.decode=original_decode
        p._t5_cache.clear();gc.collect();torch.cuda.empty_cache()


def run(plan,item,output,repository,model_root,ownership,devices):
    initialization_started=time.perf_counter()
    t,pipes,prepared,receipt=build_pair(output,repository,model_root)
    initialization_seconds=time.perf_counter()-initialization_started
    action=prepared/item['lingbot_example']
    from PIL import Image
    # Official dummy prewarm runs outside timed trajectories and does not change
    # sampling RNG (the native per-generation private generator is reseeded).
    warmup_started=time.perf_counter()
    for pipe in pipes.values():
        with Image.open(action/'image.jpg') as image,t.cuda.device(pipe.device):
            pipe.prewarm(image.convert('RGB'),max_area=480*832,frame_num=957,chunk_size=4)
    warmup_seconds=time.perf_counter()-warmup_started
    write_new(output/'input-binding.json',{'example_directory':item['lingbot_example'],
        'image_sha256':sha(action/'image.jpg'),'input_receipt':receipt,
        'comparison_scope':'within this fixed image/action-conditioned model pair'})
    results=[];reference=None
    others=list(ARMS[1:]);shift=plan['prompts'].index(item)%3
    order=['draft_only']+others[shift:]+others[:shift]
    for arm in order:
        ownership(None)
        result=generate_arm(pipes=pipes,torch=t,item=item,action=action,output=output,
            arm=arm,reference=reference if arm!='target_only' else None,ownership=lambda:ownership(None))
        results.append(result)
        if arm=='draft_only':reference=output/'arms/draft_only'
    import json
    native=json.loads((output/'arms/draft_only/trace.json').read_text())
    for arm in ARMS[1:]:
        observed=json.loads((output/'arms'/arm/'trace.json').read_text())
        require(observed['initial_chunk_noise_sha256']==native['initial_chunk_noise_sha256'],'producer initial noise differs')
        require(observed['renoise']==native['renoise'],'producer renoise differs')
        require([r['conditioning_sha256'] for r in observed['chunks']]==[r['conditioning_sha256'] for r in native['chunks']], 'native shared T5/image/camera conditioning differs')
    return {'arms':results,'execution_order':order,'input_example':item['lingbot_example'],'initialization_seconds':initialization_seconds,'warmup_seconds':warmup_seconds,'all_actual_noise_checks_passed':True}
