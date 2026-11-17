"""CPU-only attention candidates and numeric/latency checks; no Vulkan imports."""
import ast
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np

HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128

def attention_grouped(*args):
    return production_attention(*args)


def reference_attention(q, k, v, M, T, causal):
    """q [M, Hq*d] roped/normalized; k, v full caches [T, Hkv*d]; returns [M, Hq*d]."""
    qh = q.reshape(M, HEADS, HEAD_DIM)
    out = np.empty((M, HEADS * HEAD_DIM), np.float32)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for hq in range(HEADS):
        kvh = hq // (HEADS // KV_HEADS)
        s = qh[:, hq, :] @ k[:T, kvh * HEAD_DIM:(kvh + 1) * HEAD_DIM].T * scale
        if causal:
            s += np.triu(np.full((M, T), -1e30, np.float32), T - M + 1)
        s -= s.max(-1, keepdims=True)
        wgt = np.exp(s)
        wgt /= wgt.sum(-1, keepdims=True)
        out[:, hq * HEAD_DIM:(hq + 1) * HEAD_DIM] = \
            wgt @ v[:T, kvh * HEAD_DIM:(kvh + 1) * HEAD_DIM]
    return out.reshape(M, HEADS * HEAD_DIM)

def load_production():
    source = Path(__file__).with_name('chat.py').read_text(encoding='utf-8')
    node = next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name == 'attention_np')
    ns = dict(np=np, math=math, HEADS=HEADS, KV_HEADS=KV_HEADS, HEAD_DIM=HEAD_DIM)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'chat.py','exec'),ns)
    return ns['attention_np']
production_attention = load_production()

def reference_from_source():
    return reference_attention

def diagnose_roundoff(q, k, v, M, T, causal, reference):
    """Compare BOTH FP32 paths to independent FP64 einsum, stage by stage.

    gamma_n = n*u/(1-n*u) bounds sequential dot-product roundoff.
    For logit perturbation <= delta, softmax distribution L1 difference
    is <= exp(2*delta)-1, so output perturbation is bounded by this times
    max(abs(V)). This isolates matrix reduction, softmax, and output GEMM.
    The post-logit budget below is a conservative regression tolerance,
    not a formal guarantee for every NumPy exp implementation.
    """
    u = np.finfo(np.float32).eps / 2
    gamma = lambda n: n*u/(1-n*u)
    group = HEADS // KV_HEADS
    q64 = q.astype(np.float64).reshape(M, HEADS, HEAD_DIM).transpose(1,0,2)
    k64 = np.repeat(k[:T].astype(np.float64).reshape(T, KV_HEADS, HEAD_DIM),group,axis=1).transpose(1,0,2)
    v64 = np.repeat(v[:T].astype(np.float64).reshape(T, KV_HEADS, HEAD_DIM),group,axis=1).transpose(1,0,2)
    scale = 1 / math.sqrt(HEAD_DIM)
    exact_scores = np.einsum('hmd,htd->hmt',q64,k64,optimize=False)*scale
    abs_dots = np.einsum('hmd,htd->hmt',abs(q64),abs(k64),optimize=False)*scale
    dot_bound = gamma(HEAD_DIM+2)*abs_dots
    mask = np.triu(np.full((M,T),-1e30,np.float64),T-M+1) if causal else np.zeros((M,T))
    def softmax64(scores):
        scores=scores.astype(np.float64)+mask
        probs=np.exp(scores-scores.max(-1,keepdims=True))
        return probs/probs.sum(-1,keepdims=True)
    exact_probs=softmax64(exact_scores)
    truth=np.einsum('hmt,htd->hmd',exact_probs,v64,optimize=False)
    old_scores=np.stack([q.reshape(M,HEADS,HEAD_DIM)[:,h,:] @ k[:T,(h//group)*HEAD_DIM:(h//group+1)*HEAD_DIM].T * scale for h in range(HEADS)])
    qg=q.reshape(M,KV_HEADS,group,HEAD_DIM).transpose(1,0,2,3).reshape(KV_HEADS,M*group,HEAD_DIM)
    kg=k[:T].reshape(T,KV_HEADS,HEAD_DIM).transpose(1,2,0)
    new_scores=((qg@kg)*scale).reshape(KV_HEADS,M,group,T).transpose(0,2,1,3).reshape(HEADS,M,T)
    rows={}
    for label,scores,fn in [('original',old_scores,reference),('grouped',new_scores,attention_grouped)]:
        errors=abs(scores.astype(np.float64)-exact_scores)
        assert np.all(errors <= dot_bound+1e-12), (label,'dot bound failed')
        # Masked positions cannot affect softmax; exclude them from delta.
        delta=np.where(mask[None,:,:]<0,0,errors).max(-1,keepdims=True)
        probs=softmax64(scores)
        score_only=np.einsum('hmt,htd->hmd',probs,v64,optimize=False)
        got=fn(q,k,v,M,T,causal).reshape(M,HEADS,HEAD_DIM).transpose(1,0,2).astype(np.float64)
        vmax=abs(v64).max(1)[:,None,:]
        softmax_dot_bound=np.expm1(2*delta)*vmax
        # Conservative post-logit regression tolerance for FP32 softmax and
        # output GEMM. This assumes ordinary NumPy exp accuracy; unlike
        # the dot and softmax-perturbation checks, it is not a proof.
        arithmetic_bound=2*gamma(T+8)*vmax+np.finfo(np.float32).tiny
        assert np.all(abs(score_only-truth)<=softmax_dot_bound+1e-12)
        assert np.all(abs(got-score_only)<=arithmetic_bound+1e-12), (label,'post-logit arithmetic bound failed')
        assert np.all(abs(got-truth)<=softmax_dot_bound+arithmetic_bound+1e-12)
        rows[label]=dict(max_logit_abs_error=float(errors.max()), max_output_abs_error=float(abs(got-truth).max()),rms_output_error=float(np.sqrt(np.mean((got-truth)**2))),max_post_logit_error=float(abs(got-score_only).max()),max_output_error_bound=float((softmax_dot_bound+arithmetic_bound).max()))
    rows['pair_max_abs']=float(np.max(abs(reference(q,k,v,M,T,causal)-attention_grouped(q,k,v,M,T,causal))))
    print('ROUNDOFF',json.dumps(rows),flush=True)
    return rows

def test_candidates():
    ref = reference_from_source()
    rng = np.random.default_rng(724)
    cases = [(1,t,False) for t in (1,16,64,128,512,1024)] + [(m,m,True) for m in (2,16,64,128,512,1024)] + [(4,19,True),(7,31,False)]
    for M,T,causal in cases:
        q = rng.standard_normal((M, HEADS*HEAD_DIM), dtype=np.float32)
        k = rng.standard_normal((T+3, KV_HEADS*HEAD_DIM), dtype=np.float32)
        v = rng.standard_normal(k.shape, dtype=np.float32)
        originals = [a.copy() for a in (q,k,v)]
        expect = ref(q,k,v,M,T,causal)
        got = attention_grouped(q,k,v,M,T,causal)
        assert got.shape == expect.shape and got.dtype == np.float32
        np.testing.assert_allclose(got,expect,rtol=3e-5,atol=2e-6)
        for a,b in zip((q,k,v),originals): np.testing.assert_array_equal(a,b)
        print('PASS',M,T,causal,'max_abs',float(np.max(np.abs(got-expect))),flush=True)
    # Sliced/non-contiguous inputs and poisoned unused cache suffix.
    for dtype, gain in [(np.float32, 1.0), (np.float32, 20.0), (np.float64, 1.0)]:
        M,T = 3,17
        q = (rng.standard_normal((M, HEADS*HEAD_DIM*2)).astype(dtype) * gain)[:, ::2]
        k = (rng.standard_normal((T+3, KV_HEADS*HEAD_DIM*2)).astype(dtype) * gain)[:, ::2]
        v = rng.standard_normal((T+3, KV_HEADS*HEAD_DIM*2)).astype(dtype)[:, ::2]
        assert all(not x.flags.c_contiguous for x in (q,k,v))
        k[T:] = np.nan
        v[T:] = np.nan
        expected = ref(q,k,v,M,T,True)
        got = attention_grouped(q,k,v,M,T,True)
        assert got.dtype == np.float32 and np.isfinite(got).all()
        if gain == 20.0:
            diagnose_roundoff(q,k,v,M,T,True,ref)
        else:
            np.testing.assert_allclose(got, expected, rtol=3e-5, atol=2e-6)
        print('PASS strided/cache suffix',dtype.__name__,gain,flush=True)
    return ref

def bench(fn,args,repeats=15):
    for _ in range(3): fn(*args)
    times=[]
    for _ in range(repeats):
        t=time.perf_counter();fn(*args);times.append(time.perf_counter()-t)
    return statistics.median(times)*1000

if __name__ == '__main__':
    ref = test_candidates()
    rng=np.random.default_rng(42)
    results=[]
    for M,T,causal in [(1,t,False) for t in (16,64,128,512)] + [(m,m,True) for m in (16,64,128)]:
        args=(rng.standard_normal((M,HEADS*HEAD_DIM),dtype=np.float32),rng.standard_normal((T,KV_HEADS*HEAD_DIM),dtype=np.float32),rng.standard_normal((T,KV_HEADS*HEAD_DIM),dtype=np.float32),M,T,causal)
        old=bench(ref,args);new=bench(attention_grouped,args)
        row=dict(M=M,T=T,causal=causal,reference_ms=old,grouped_ms=new,speedup=old/new)
        results.append(row);print(json.dumps(row),flush=True)
    Path(__file__).with_suffix('.results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
