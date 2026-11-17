"""Optional real-model context-boundary regression (requires local model + GPU)."""
import argparse
import os
import numpy as np
from tokenizers import Tokenizer
import chat
import ops


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=r'C:\vkops\qwen3-4b')
    p.add_argument('--context',type=int,default=1024)
    args=p.parse_args()
    gpu=ops.GPU();weights=chat.stream_upload(gpu,args.model,36)
    e=chat.Qwen3Engine(gpu,weights,36,2560,9728,max_ctx=args.context)
    tok=Tokenizer.from_file(os.path.join(args.model,'tokenizer.json'))
    base=tok.encode('The sky is blue because air scatters sunlight. ').ids
    ids=(base*((args.context+len(base)-1)//len(base)))[:args.context]
    full=e.forward(ids,0)
    e.clear_cache();e.forward(ids[:-4],0)
    block=e.forward(ids[-4:],args.context-4,all_logits=True)
    rel=float(np.max(abs(block[-1]-full))/(np.max(abs(full))+1e-9))
    print('boundary relative_error',rel,'argmax_match',int(full.argmax()==block[-1].argmax()),flush=True)
    assert rel<.01 and full.argmax()==block[-1].argmax()
    e.truncate_cache(args.context-4)
    changed=e.forward([ids[-4],1,2,3],args.context-4,all_logits=True)
    np.testing.assert_array_equal(block[0],changed[0])
    e.truncate_cache(args.context-4)
    replay=e.forward(ids[-4:],args.context-4,all_logits=True)
    np.testing.assert_array_equal(block,replay)
    try:e.forward([1],args.context)
    except ValueError:pass
    else:raise AssertionError('overflow was not rejected')
    assert e.cache_length==args.context
    taps=[1,9,17,25,33]
    e.enable_feature_capture(taps)
    e.forward(ids[:4],0,collect_states=True)
    ref=np.concatenate([e.layer_states[2*(i+1)] for i in taps],axis=-1)
    np.testing.assert_array_equal(e.last_features,ref)
    print('FULL-MODEL CONTEXT, ROLLBACK AND FEATURE TESTS PASSED',flush=True)


if __name__=='__main__':main()
