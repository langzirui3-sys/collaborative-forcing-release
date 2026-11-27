"""Merge sealed decoder reproduction and matched-history causal interventions."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .cross_model_delay import require,write_new
from .lingbot_sink_probe import SOURCE_RUN,SOURCE_INDEX,VARIANTS,kept_frames

SINK_RUN='lingbot-sink-probe-p03-ntu-v1-20261028T130518Z-ff607e9e-495bd55b-2d999304'
SINK_INDEX='5afe22e0c1aed23454ffa3ee5f810b3e7100967b8ec9cf39d80730f2aafc9c7c'
DECODE_RUN='lingbot-reset-probe-p03-ntu-v1-20261028T124108Z-c531f829-285cca9e-af05dc60'
DECODE_INDEX='84d8a54f99676b629ef5e5c6af8c3d52f842b91a7d60a25ab81aaf3688fd6a4a'


def rmse(a,b,np):
    require(a.shape==b.shape and bool(np.isfinite(a).all()) and bool(np.isfinite(b).all()),'unaligned/nonfinite comparison')
    return float(np.mean((a.astype('float64')-b.astype('float64'))**2)**.5)


def run(run_root):
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from .main_table_baseline_campaign import _run_dir
    from .main_table_merge import _load_frozen_index
    from .lingbot_reset_probe import load_trace_bound_latent,cache_layout
    from .stage_timing import _unsealed_directory
    output=_run_dir();_unsealed_directory(output);torch.set_num_threads(4)
    readers={};bases={};seals={}
    for name,run_id,digest in [('original',SOURCE_RUN,SOURCE_INDEX),('sink',SINK_RUN,SINK_INDEX),('decode',DECODE_RUN,DECODE_INDEX)]:
        source=run_root/run_id;frozen,seal=_load_frozen_index(source)
        require(seal['artifact_index_sha256']==digest,'source index drift')
        readers[name]=(source,frozen);bases[name]=source/'attempts'/seal['attempt_id'];seals[name]=seal
    def bound(name,relative):
        source,frozen=readers[name]
        return frozen.file((bases[name]/relative).relative_to(source).as_posix(),label='reset report source')
    old=json.loads(bound('original','arms/draft_only/trace.json').read_text())
    decode=json.loads(bound('decode','decode-summary.json').read_text())
    require(len(decode['arms'])==4,'decoder arms missing')
    decode_stats=[]
    for arm in decode['arms']:
        a=arm['original_png_comparison'];b=arm['fresh_context_comparison']
        require(len(a)==60 and all(x['exact'] and x['max_abs']==0 for x in a),'decoder failed exact reproduction')
        require([x['second'] for x in b]==list(range(45,57)),'cropped decoder coverage drift')
        decode_stats.append({'arm':arm['arm'],'original_exact_frames':60,'crop_max_abs':max(x['max_abs'] for x in b),
            'crop_max_mae':max(x['mae'] for x in b)})
    image_original=json.loads(bound('original','input-binding.json').read_text())['image_sha256']
    require(json.loads(bound('sink','input-binding.json').read_text())['image_sha256']==image_original,'image changed')
    images={};rgb_rows=[];latent_rows=[];checks=[]
    for variant in VARIANTS:
        prefix=f'{variant}/arms/draft_only';trace=json.loads(bound('sink',prefix+'/trace.json').read_text())
        policy=json.loads(bound('sink',variant+'/attention-policy.json').read_text())
        cutoff=60 if variant=='control' else 50
        require(len(trace['chunks'])==60 and [x['latent_sha256'] for x in trace['chunks'][:cutoff]]==[x['latent_sha256'] for x in old['chunks'][:cutoff]],'source prefix drift')
        require(trace['renoise']==old['renoise'] and trace['initial_chunk_noise_sha256']==old['initial_chunk_noise_sha256'],'noise changed')
        require([x['conditioning_sha256'] for x in trace['chunks']]==[x['conditioning_sha256'] for x in old['chunks']],'conditioning changed')
        require(len(policy['calls'])==1500,'intervention coverage drift')
        for call in policy['calls']:
            require(tuple(call['retained_local_frames'])==kept_frames(variant,call['chunk']),'attention policy drift')
            require(call['key_tokens']*4==len(call['retained_local_frames'])*call['query_tokens'],'attention token count drift')
        frames=[np.asarray(Image.open(bound('sink',prefix+f'/second-{i:02d}.png')).convert('RGB')) for i in range(60)]
        images[variant]=frames
        if variant=='control':
            for i,x in enumerate(frames):require(np.array_equal(x,np.asarray(Image.open(bound('original',f'arms/draft_only/second-{i:02d}.png')).convert('RGB'))),'control RGB drift')
        for i,x in enumerate(frames):
            rgb_rows.append({'variant':variant,'second':i,'rmse_to_first':rmse(x,frames[0],np),
                'rmse_to_previous':rmse(x,frames[max(0,i-1)],np)})
        # Helper takes the path below arms; use this variant's attempt subdirectory.
        chunks=[load_trace_bound_latent(bases['sink']/variant,'draft_only',i,trace['chunks'][i],torch).numpy() for i in range(60)]
        latent=np.concatenate(chunks,axis=1)
        for i in range(240):latent_rows.append({'variant':variant,'latent_frame':i,
            'rmse_to_first':rmse(latent[:,i],latent[:,0],np),
            'rmse_to_previous':rmse(latent[:,i],latent[:,max(0,i-1)],np)})
        checks.append({'variant':variant,'exact_source_prefix_chunks':cutoff,'attention_calls_checked':1500})
    for filename,rows in [('rgb-distances.csv',rgb_rows),('latent-distances.csv',latent_rows)]:
        with (output/filename).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    seconds=(0,48,49,50,51,52,55,59)
    panel=Image.new('RGB',(416*len(seconds),264*3),'white');draw=ImageDraw.Draw(panel)
    for row,variant in enumerate(VARIANTS):
        for col,second in enumerate(seconds):
            panel.paste(Image.fromarray(images[variant][second]),(416*col,264*row+24))
            draw.text((416*col+4,264*row+4),f'{variant} | {second}s',fill='black')
    panel.save(output/'reset-causal-comparison.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for variant in VARIANTS:
        for row,data,coordinate in [(0,rgb_rows,'second'),(1,latent_rows,'latent_frame')]:
            selected=[x for x in data if x['variant']==variant]
            for col,key in enumerate(('rmse_to_first','rmse_to_previous')):
                axes[row,col].plot([x[coordinate]/(4 if row else 1) for x in selected],[x[key] for x in selected],label=variant)
    for row in range(2):
        for col in range(2):
            ax=axes[row,col];ax.set_xlim(45,59);ax.axvline(50,color='grey',linestyle='--');ax.set_xlabel('latent frame endpoint (s)' if row else 'video time (s)');ax.set_ylabel(('RGB' if row==0 else 'latent')+' RMSE');ax.set_title(('distance to initial frame','distance to previous frame')[col]);ax.legend(fontsize=7)
    fig.savefig(output/'reset-causal-distances.png',dpi=160);plt.close(fig)
    write_new(output/'summary.json',{'schema':'lingbot_reset_causal_report_v1','source_seals':seals,
        'decoder_reproduction':decode_stats,'causal_checks':checks,
        'cache_layout_at_chunk_50':cache_layout()[50],
        'location':'generated latent trajectory; reproduced independently of long decoder history',
        'causal_scope':'Effect of retaining initial six-frame attention history, holding the prefix, conditioning and noise fixed; compare equal-size middle-history deletion.',
        'limitations':['one input/seed; not a universal failure mechanism',
            'distance to first frame is not perceptual quality',
            'cache layout is source-derived, original KV tensor contents were not saved',
            'does not establish numerical KV error accumulation or why this seed first switches at latent frame 201',
            'visual comparison pending; no quality benefit claimed'],
        'rgb_window':[x for x in rgb_rows if x['second'] in [49,50,51,52]],
        'latent_201':[x for x in latent_rows if x['latent_frame']==201]})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();run(a.run_root)

if __name__=='__main__':main()
