"""CPU-only regression of the actual engine with a tiny NumPy GPU adapter."""
import ast
import math
import os
from pathlib import Path
import unittest

import numpy as np


def engine_namespace():
    tree = ast.parse(Path(__file__).with_name('chat.py').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    ns = dict(np=np, math=math, os=os, HEADS=2, KV_HEADS=1, HEAD_DIM=32,
              ROPE_THETA=5e6, EPS=1e-6, VOCAB=16, IM_END=15)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'chat.py', 'exec'), ns)
    return ns


class Buffer:
    def __init__(self, size):
        self.size = size
        self.data = np.zeros(size, np.uint8)

    def upload(self, x):
        data = np.ascontiguousarray(x).view(np.uint8).reshape(-1)
        assert data.size <= self.size
        self.data[:data.size] = data

    def np(self, dtype, count):
        return self.data.view(dtype)[:count]


class NumpyGPU:
    def __init__(self):
        self.dev = self

    def empty(self, n):
        return Buffer(n)

    def tensor(self, x):
        b = Buffer(x.nbytes)
        b.upload(x)
        return b

    tensor_f16 = tensor

    def submit_jobs(self, jobs):
        for job in jobs:
            job()

    def job_gemm_fp16(self, a, b, m, n, k, bT=False, out=None):
        def run():
            aa = a.np(np.float16, m*k).astype(np.float32).reshape(m, k)
            bb = b.np(np.float16, n*k).astype(np.float32).reshape(n, k)
            # Fixed reduction order avoids BLAS choosing GEMV vs GEMM routines.
            out.upload(np.einsum('mk,nk->mn', aa, bb, optimize=False))
        return run, out

    def gemm_fp16(self, *args, **kw):
        job, out = self.job_gemm_fp16(*args, **kw)
        job()
        return out

    def rmsnorm(self, x, w, rows, n, out=None):
        xx = x.np(np.float32, rows*n).reshape(rows, n)
        out.upload(xx / np.sqrt((xx*xx).mean(-1, keepdims=True)+1e-6)
                   * w.np(np.float32, n))
        return out

    def to_f16(self, x, n, dst=None):
        dst.upload(x.np(np.float32, n).astype(np.float16))
        return dst

    def job_rmsnorm_f16(self, x, w, rows, n, out=None):
        def run():
            xx = x.np(np.float32, rows*n).reshape(rows, n)
            out.upload((xx / np.sqrt((xx*xx).mean(-1, keepdims=True)+1e-6)
                        * w.np(np.float32, n)).astype(np.float16))
        return run, out

    def add(self, a, b, n, out=None):
        out.upload(a.np(np.float32, n) + b.np(np.float32, n))
        return out

    def job_add(self, *args, **kw):
        return lambda: self.add(*args, **kw), kw['out']

    def silu_mul(self, a, b, n, out=None):
        aa = a.np(np.float32, n)
        out.upload((aa / (1+np.exp(-aa))) * b.np(np.float32, n))
        return out

    def job_silu_mul_f16(self, a, b, n, out=None):
        def run():
            aa = a.np(np.float32, n)
            out.upload(((aa/(1+np.exp(-aa)))*b.np(np.float32, n)).astype(np.float16))
        return run, out


def make_engine():
    ns = engine_namespace()
    gpu = NumpyGPU()
    rng = np.random.default_rng(123)
    weights = {'model.embed_tokens.weight': gpu.tensor(rng.normal(0, .3, (16,32)).astype(np.float16)),
               'model.norm.weight': gpu.tensor(np.ones(32, np.float32))}
    shapes = {'input_layernorm.weight': (32,), 'post_attention_layernorm.weight': (32,),
              'self_attn.q_norm.weight': (32,), 'self_attn.k_norm.weight': (32,),
              'self_attn.q_proj.weight': (64,32), 'self_attn.k_proj.weight': (32,32),
              'self_attn.v_proj.weight': (32,32), 'self_attn.o_proj.weight': (32,64),
              'mlp.gate_proj.weight': (64,32), 'mlp.up_proj.weight': (64,32),
              'mlp.down_proj.weight': (32,64)}
    for li in range(2):
        for name, shape in shapes.items():
            arr = np.ones(shape, np.float32) if len(shape)==1 else rng.normal(0, .12, shape).astype(np.float16)
            weights[f'model.layers.{li}.{name}'] = gpu.tensor(arr)
    return ns['Qwen3Engine'](gpu, weights, 2, 32, 64, max_ctx=16)


class VerifierTests(unittest.TestCase):
    def test_selected_full_layer_features(self):
        e = make_engine()
        e.enable_feature_capture([1,0])
        e.forward([1,2,3],0,collect_states=True)
        expected = np.concatenate([e.layer_states[4],e.layer_states[2]],axis=-1)
        np.testing.assert_array_equal(e.last_features,expected)
        self.assertEqual(e.last_features.shape,(3,64))
        _, states = e.forward_perop([1,2,3],0,collect_states=True)
        np.testing.assert_array_equal(e.last_features,np.concatenate([states[4],states[2]],axis=-1))
        e.forward([4],3)
        self.assertEqual(e.last_features.shape,(1,64))
        e.enable_feature_capture([])
        e.forward([1],0)
        self.assertIsNone(e.last_features)
        for layers in ([2],[-1],[0,0],[.5]):
            with self.assertRaises(ValueError):
                e.enable_feature_capture(layers)

    def test_failed_forward_invalidates_only_overwritten_suffix(self):
        e = make_engine()
        for method in (e.forward,e.forward_perop):
            e.clear_cache(); method([1,2,3],0)
            original = e.gpu.job_gemm_fp16
            calls = [0]
            def broken(*args,**kw):
                calls[0] += 1
                if calls[0] == 5:
                    raise RuntimeError('injected GPU operation failure')
                return original(*args,**kw)
            e.gpu.job_gemm_fp16 = broken
            with self.assertRaises(RuntimeError):
                method([4,5],1)
            e.gpu.job_gemm_fp16 = original
            self.assertEqual(e.cache_length,1)
            got = method([4,5],1)
            e.clear_cache()
            expected = method([1,4,5],0)
            np.testing.assert_allclose(got,expected,rtol=2e-4,atol=5e-4)

    def test_greedy_generation_opt_in(self):
        e = make_engine()
        normal = e.generate([1,2,3], max_new=6, temperature=0, eos=-1)
        got = e.generate([1,2,3], max_new=6, temperature=0, eos=-1,
                         speculative_ngram=True)
        self.assertEqual(normal,got)
        with self.assertRaises(ValueError):
            e.generate([1,2,3], temperature=.7, speculative_ngram=True)

    def test_incremental_causality(self):
        e = make_engine()
        for method in (e.forward, e.forward_perop):
            e.clear_cache()
            method([1,2], 0)
            method([3,4,5], 2)
            block_k = [x[:5].copy() for x in e.kc]
            e.clear_cache()
            method([1,2], 0)
            for pos, token in enumerate([3,4,5], 2):
                method([token], pos)
            for got, want in zip(block_k, e.kc):
                np.testing.assert_allclose(got, want[:5], rtol=2e-4, atol=5e-4)

    def test_all_logits_and_default_last(self):
        e = make_engine()
        for method in (e.forward, e.forward_perop):
            e.clear_cache()
            method([1,2], 0)
            rows = method([3,4,5], 2, all_logits=True)
            self.assertEqual(rows.shape, (3,16))
            e.clear_cache()
            method([1,2], 0)
            sequential = np.stack([method([token], pos) for pos, token in enumerate([3,4,5], 2)])
            np.testing.assert_allclose(rows, sequential, rtol=2e-4, atol=5e-4)
            last = method([3,4,5], 2)
            np.testing.assert_array_equal(last, rows[-1])

    def test_rollback_and_overwrite(self):
        e = make_engine()
        e.forward([1,2,3,4], 0)
        self.assertEqual(e.cache_length, 4)
        e.truncate_cache(2)
        self.assertEqual(e.cache_length, 2)
        e.kc[0][2:] = 999
        e.vc[0][2:] = -999
        got = e.forward([6,7], 2, all_logits=True)
        self.assertEqual(e.cache_length, 4)
        e.clear_cache()
        expected = e.forward([1,2,6,7], 0, all_logits=True)[2:]
        np.testing.assert_allclose(got, expected, rtol=2e-4, atol=5e-4)
        e.forward([8], 1)
        self.assertEqual(e.cache_length, 2)
        e.clear_cache()
        self.assertEqual(e.cache_length, 0)

    def test_invalid_inputs_do_not_touch_cache(self):
        e = make_engine()
        for method in (e.forward, e.forward_perop):
            e.clear_cache()
            method([1,2], 0)
            for ids, pos in [([],2), ([1],-1), ([1],3), ([1]*15,2), ([-1],2), ([16],2), ([1.5],2), ([1],1.5)]:
                with self.assertRaises(ValueError, msg=str((ids,pos))):
                    method(ids,pos)
                self.assertEqual(e.cache_length,2)
            for length in (-1,3,1.5):
                with self.assertRaises(ValueError):
                    e.truncate_cache(length)
            method([1]*14,2)
            self.assertEqual(e.cache_length,16)
            with self.assertRaises(ValueError):
                method([1],16)


if __name__ == '__main__':
    unittest.main()
