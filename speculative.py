"""Optional greedy n-gram speculation, without another model or training.

The target checks every proposed token. This preserves greedy decisions for the
target's block logits, not bitwise equivalence between different floating-point
GEMM shapes. Sampling is deliberately NOT handled here.
"""
import time
import numpy as np


def ngram_draft(history, max_new, max_ngram=4):
    """Continue the longest suffix seen earlier, preferring its latest occurrence."""
    for n in range(min(max_ngram, len(history)-1),0,-1):
        suffix=history[-n:]
        for start in range(len(history)-n-1,-1,-1):
            if history[start:start+n] == suffix:
                return list(history[start+n:start+n+max_new])
    return []


def generate_greedy(engine, prompt_ids, max_new=128, eos=151645, callback=None,
                    block_size=4, max_ngram=4, stats=None):
    """Generate greedily, accepting only a contiguous target-verified draft prefix.

    The first token of each block is already known from previous target logits.
    Row i predicts candidate i+1. On rejection, discard every cache row after the
    accepted prefix and use its final logit row for the next correction token.
    Two wholly rejected drafts trigger an eight-cycle backoff.
    """
    for name,value,lo,hi in [('block_size',block_size,1,8),('max_ngram',max_ngram,1,32),
                             ('max_new',max_new,0,2**31-1)]:
        if not isinstance(value,(int,np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in [{lo},{hi}]')
    if max_new == 0:
        return []
    history=list(prompt_ids)
    engine.clear_cache()
    t=time.perf_counter()
    logits=engine.forward(history,0)
    meter={'prefill_s':time.perf_counter()-t,'decode_s':0.,'cycles':0,
           'drafted':0,'accepted_draft':0,'tokens':0,'backoff_cycles':0}
    budget=min(max_new,engine.max_ctx-len(history))
    generated=[]
    rejected=0
    backoff=0
    t=time.perf_counter()
    while len(generated)<budget:
        first=int(np.argmax(logits))
        if first==eos:
            break
        width=min(block_size,budget-len(generated))
        if backoff:
            draft=[]
            backoff-=1
            meter['backoff_cycles']+=1
        else:
            draft=ngram_draft(history+[first],width-1,max_ngram) if width>1 else []
        candidate=[first]+draft
        pos=len(history)
        rows=engine.forward(candidate,pos,all_logits=True)
        accepted=1
        for i,token in enumerate(draft):
            if token!=int(np.argmax(rows[i])):
                break
            accepted+=1
        meter['cycles']+=1
        meter['drafted']+=len(draft)
        meter['accepted_draft']+=accepted-1
        chosen=candidate[:accepted]
        stopped=eos in chosen
        if stopped:
            chosen=chosen[:chosen.index(eos)]
        engine.truncate_cache(pos+len(chosen))
        history.extend(chosen)
        generated.extend(chosen)
        for token in chosen:
            if callback:
                callback(token)
        if stopped:
            break
        logits=rows[accepted-1]
        if draft:
            rejected=rejected+1 if accepted==1 else 0
            if rejected>=2:
                backoff=8
                rejected=0
    meter['decode_s']=time.perf_counter()-t
    meter['tokens']=len(generated)
    if stats is not None:
        stats.update(meter)
    return generated
