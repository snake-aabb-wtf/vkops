"""Real greedy generation A/B. Includes drafter time; reports prompt+decode too."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
from tokenizers import Tokenizer

import chat
import ops
from speculative import generate_greedy

PROMPTS = {
    'explanation': 'Explain why the sky is blue in simple terms. Please be concise.',
    'code': 'Write a Python function that returns the first n Fibonacci numbers. Include a short example.',
    'chinese': '\u7528\u4e2d\u6587\u89e3\u91ca\u4e3a\u4ec0\u4e48\u6c34\u4f1a\u7ed3\u51b0\uff0c\u5e76\u7ed9\u51fa\u4e24\u4e2a\u751f\u6d3b\u4e2d\u7684\u4f8b\u5b50\u3002',
    'repetition': 'Repeat exactly the following sentence ten times, with no extra text: The quick brown fox jumps over the lazy dog.',
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=r'C:\vkops\qwen3-4b')
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--output',default='ngram_results.json')
    args=p.parse_args()
    gpu=ops.GPU();print('[device]',gpu.info(),flush=True)
    weights=chat.stream_upload(gpu,args.model,36)
    engine=chat.Qwen3Engine(gpu,weights,36,2560,9728,max_ctx=512)
    tok=Tokenizer.from_file(os.path.join(args.model,'tokenizer.json'))
    results={}
    for name,prompt in PROMPTS.items():
        ids=tok.encode('<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n').ids
        runs={1:[],4:[]};outputs={1:[],4:[]}
        # Warm every kernel specialization outside timing, including M1 and M2.
        for width in (1,2,4):
            engine.clear_cache();engine.forward(ids,0)
            engine.forward([1]*width,len(ids))
        for rep in range(args.repeats):
            for width in ((1,4) if rep%2==0 else (4,1)):
                stats={};t=time.perf_counter()
                out=generate_greedy(engine,ids,max_new=args.steps,block_size=width,stats=stats)
                stats['total_s']=time.perf_counter()-t
                stats['decode_tok_s']=len(out)/stats['decode_s'] if out else 0
                runs[width].append(stats);outputs[width].append(out)
                print('[run]',name,width,json.dumps(stats),flush=True)
        same=all(x==outputs[1][0] for v in outputs.values() for x in v)
        summary={'prompt_tokens':len(ids),'outputs_identical':same,'runs':runs,
                 'baseline_decode_tok_s':statistics.median(x['decode_tok_s'] for x in runs[1]),
                 'ngram_decode_tok_s':statistics.median(x['decode_tok_s'] for x in runs[4]),
                 'baseline_total_s':statistics.median(x['total_s'] for x in runs[1]),
                 'ngram_total_s':statistics.median(x['total_s'] for x in runs[4]),
                 'baseline_ids':outputs[1][0],'ngram_ids':outputs[4][0],
                 'text':tok.decode(outputs[4][0])}
        results[name]=summary
        Path(args.output).write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
        print('[summary]',name,'identical',same,'baseline',summary['baseline_decode_tok_s'],
              'ngram',summary['ngram_decode_tok_s'],flush=True)
    assert all(x['outputs_identical'] for x in results.values()), 'greedy output divergence'
    print('NGRAM GENERATION BENCHMARK PASSED',flush=True)


if __name__=='__main__':
    main()
