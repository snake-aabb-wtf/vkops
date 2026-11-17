"""Full-model cached-block verification ceiling, NOT a speculative speed claim."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=r'C:\vkops\qwen3-4b')
    p.add_argument('--output', default='speculative_ceiling.json')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--steps', type=int, default=24)
    p.add_argument('--prompt-tokens', type=int, default=0,
                   help='repeat the tokenized prompt to this length for long-context tests')
    p.add_argument('--max-ctx', type=int, default=512)
    args = p.parse_args()
    gpu = ops.GPU()
    print('[device]', gpu.info(), flush=True)
    weights = chat.stream_upload(gpu, args.model, 36)
    e = chat.Qwen3Engine(gpu, weights, 36, 2560, 9728, max_ctx=args.max_ctx)
    tok = Tokenizer.from_file(os.path.join(args.model, 'tokenizer.json'))
    prompt = tok.encode('<|im_start|>user\nExplain why the sky is blue in simple terms. Please be concise.<|im_end|>\n<|im_start|>assistant\n').ids
    if args.prompt_tokens:
        prompt = (prompt*((args.prompt_tokens+len(prompt)-1)//len(prompt)))[:args.prompt_tokens]
    fixed = tok.encode('The sky appears blue because molecules in the atmosphere scatter sunlight. Shorter blue wavelengths scatter more strongly than red light.').ids
    fixed = (fixed*((args.steps+len(fixed)-1)//len(fixed)))[:args.steps]
    rows = {}
    result = {'device': gpu.info(), 'prompt_tokens': len(prompt), 'steps': args.steps,
              'label': 'teacher-forced known-token verifier ceiling; excludes drafter cost', 'blocks': {}}
    for block in (1,2,4,8):
        e.clear_cache()
        e.forward(prompt,0)
        e.forward(fixed[:block],len(prompt),all_logits=True)
        times = []
        for rep in range(args.repeats):
            e.clear_cache()
            e.forward(prompt,0)
            seq = []
            t = time.perf_counter()
            for i in range(0,len(fixed),block):
                seq.append(e.forward(fixed[i:i+block],len(prompt)+i,all_logits=True))
            times.append(time.perf_counter()-t)
            rows[block] = np.concatenate(seq)
        got, ref = rows[block], rows[1]
        rel = np.max(abs(got-ref),axis=1)/(np.max(abs(ref),axis=1)+1e-9)
        entry = {'seconds':times, 'median_s':statistics.median(times),
                 'tokens_per_s':len(fixed)/statistics.median(times),
                 'max_relative_logit_error':float(rel.max()),
                 'argmax_matches':int(np.sum(got.argmax(-1)==ref.argmax(-1))),
                 'positions':len(fixed)}
        result['blocks'][str(block)] = entry
        print('[block]',block,json.dumps(entry),flush=True)
        assert rel.max() < .01, (block,rel.max())
    # The first row must not depend on future tokens in the SAME block.
    e.clear_cache(); e.forward(prompt,0)
    first = e.forward(fixed[:4],len(prompt),all_logits=True)
    e.truncate_cache(len(prompt))
    changed = e.forward([fixed[0],1,2,3],len(prompt),all_logits=True)
    np.testing.assert_array_equal(first[0],changed[0])
    e.truncate_cache(len(prompt))
    restored = e.forward(fixed[:4],len(prompt),all_logits=True)
    np.testing.assert_array_equal(first,restored)
    result['future_token_independence'] = True
    result['rollback_replay_identical'] = True
    # Reference per-op path also observes the same causal/cache contract.
    e.truncate_cache(len(prompt))
    perop = e.forward_perop(fixed[:2],len(prompt),all_logits=True)
    rel = float(np.max(abs(perop-first[:2]))/(np.max(abs(first[:2]))+1e-9))
    assert rel < .01, rel
    result['perop_relative_error'] = rel
    # Selected target features must be COMPLETE decoder layer outputs, in order.
    taps = [1,9,17,25,33]
    e.enable_feature_capture(taps)
    e.forward(fixed[:4],0,collect_states=True)
    expected = np.concatenate([e.layer_states[2*(li+1)] for li in taps],axis=-1)
    np.testing.assert_array_equal(e.last_features,expected)
    e.enable_feature_capture([])
    result['feature_taps_match_full_layer_states'] = True
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('VERIFIER BENCHMARK PASSED',flush=True)


if __name__ == '__main__':
    main()
