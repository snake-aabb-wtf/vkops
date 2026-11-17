"""Explicit cross-variant DFlash1 experiment; no download or remote code loading.

Current public drafter targets original Qwen3-4B. The local target is 2507 /
abliterated. Acceptance and speed must be measured, not inferred from its name.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from tokenizers import Tokenizer

import chat
import ops
from bench_ngram import PROMPTS
from dflash_draft import DFlashDraftEngine, load_dflash_weights, generate_dflash
from speculative import generate_greedy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=r'C:\vkops\qwen3-4b')
    p.add_argument('--draft-model',required=True)
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--workloads',default='explanation,code,chinese')
    p.add_argument('--blocks',default='16:8,8:8,4:4',help='draft width:verify width')
    p.add_argument('--cpu-reference',action='store_true')
    p.add_argument('--output',default='dflash_results.json')
    args=p.parse_args()
    choices=[tuple(map(int,x.split(':'))) for x in args.blocks.split(',')]
    cfg=json.loads(Path(args.draft_model,'config.json').read_text(encoding='utf-8'))
    target_cfg=json.loads(Path(args.model,'config.json').read_text(encoding='utf-8'))
    for field in ('hidden_size','num_attention_heads','num_key_value_heads','head_dim','vocab_size'):
        if cfg[field]!=target_cfg[field]:raise ValueError('incompatible '+field)
    if cfg['num_target_layers']!=target_cfg['num_hidden_layers']:
        raise ValueError('incompatible target layer count')
    gpu=ops.GPU();print('[device]',gpu.info(),flush=True)
    weights=chat.stream_upload(gpu,args.model,36)
    engine=chat.Qwen3Engine(gpu,weights,36,2560,9728,max_ctx=512)
    t=time.perf_counter()
    dw=load_dflash_weights(gpu,Path(args.draft_model,'model.safetensors'),cfg)
    draft=DFlashDraftEngine(gpu,dw,engine.embed16,cfg,max_ctx=512)
    print('[draft_load_s]',time.perf_counter()-t,flush=True)
    tok=Tokenizer.from_file(os.path.join(args.model,'tokenizer.json'))
    encode=lambda text:tok.encode('<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n').ids
    # Test true target features, cache crop and an independent CPU GEMM backend.
    ids=encode(PROMPTS['explanation'])
    engine.enable_feature_capture(cfg['dflash_config']['target_layer_ids'])
    logits=engine.forward(ids,0);features=engine.last_features.copy();seed=int(logits.argmax())
    rows=draft.draft(seed,features,len(ids),4)
    cached=draft.draft(seed,np.empty((0,features.shape[1]),np.float32),len(ids),4)
    np.testing.assert_array_equal(rows,cached)
    draft.clear_cache()
    split=len(ids)//2
    draft.draft(ids[split],features[:split],split,4)
    incremental=draft.draft(seed,features[split:],len(ids),4)
    rel=float(np.max(abs(rows-incremental))/(np.max(abs(rows))+1e-9))
    assert rel<.002, ('draft cache equivalence',rel)
    result={'device':gpu.info(),'target_rope_theta':target_cfg['rope_theta'],
            'draft_rope_theta':cfg['rope_theta'],'cross_variant_experiment':True,
            'cache_relative_error':rel,'workloads':{}}
    print('[cache_equivalence]',rel,flush=True)
    if args.cpu_reference:
        from test_dflash_draft import CPUGPU
        cpu=DFlashDraftEngine(CPUGPU(),dw,engine.embed16,cfg,max_ctx=512)
        t=time.perf_counter();ref=cpu.draft(seed,features,len(ids),4)
        rel=float(np.max(abs(rows-ref))/(np.max(abs(ref))+1e-9))
        result['cpu_reference']={'seconds':time.perf_counter()-t,'relative_error':rel,
                                 'argmax_matches':int(np.sum(rows.argmax(-1)==ref.argmax(-1))),'rows':len(rows)}
        print('[cpu_reference]',json.dumps(result['cpu_reference']),flush=True)
        assert rel<.01,('draft CPU/GPU numerical mismatch',rel)
    engine.enable_feature_capture([])
    for name in args.workloads.split(','):
        ids=encode(PROMPTS[name]);runs=[];baseline=None
        # Warm actual generation paths, including feature hooks and all kernels.
        generate_greedy(engine,ids,max_new=2,block_size=1)
        for block,verify in choices:
            generate_dflash(engine,draft,ids,max_new=verify,block_size=block,verify_size=verify)
        for rep in range(args.repeats):
            settings=[(0,1)]+choices if rep%2==0 else list(reversed(choices))+[(0,1)]
            for block,verify in settings:
                stats={}
                if block==0:
                    out=generate_greedy(engine,ids,max_new=args.steps,block_size=1,stats=stats)
                    if baseline is None:baseline=out
                else:
                    out=generate_dflash(engine,draft,ids,max_new=args.steps,block_size=block,
                                        verify_size=verify,stats=stats)
                entry={'draft_block':block,'verify_block':verify,'rep':rep,'ids':out,
                       'decode_tok_s':len(out)/stats['decode_s'] if out else 0,'stats':stats}
                runs.append(entry)
                print('[run]',name,block,verify,json.dumps({k:v for k,v in entry.items() if k!='ids'}),flush=True)
                result['workloads'][name]={'runs':runs,'baseline_ids':baseline,
                                           'outputs_identical':all(x['ids']==baseline for x in runs)}
                Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
        same=all(x['ids']==baseline for x in runs)
        print('[greedy_match]',name,same,flush=True)
    assert all(x['outputs_identical'] for x in result['workloads'].values()),'greedy output divergence'
    print('DFLASH EXPERIMENT COMPLETE',flush=True)


if __name__=='__main__':
    main()
