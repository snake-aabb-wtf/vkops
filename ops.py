# -*- coding: utf-8 -*-
"""ops.py — tensor-level Vulkan compute operators + safetensors weight loading.

Operators (FP32, compute-shader backed):
  _gemm / linear   -- tiled GEMM, optional bias + fused activation
  silu_mul(a, b)   -- SwiGLU-style fused activation
  add(a, b)
  rmsnorm(x, w)    -- one workgroup per row
  ffn_block(...)   -- whole llama-style FFN recorded into ONE submit

Weights come straight from .safetensors files via st.load_file().
"""
import os
import struct

import numpy as np

import st
from vk import VulkanDevice

ACT_NONE, ACT_RELU, ACT_GELU, ACT_SILU = 0, 1, 2, 3

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")

SHADER_FILES = {
    "gemm": "gemm.comp",
    "gemm_v2": "gemm_v2.comp",
    "gemm_fp16": "gemm_fp16.comp",
    "cvt_f32_f16": "cvt_f32_f16.comp",
    "silu_mul": "silu_mul.comp",
    "add": "add.comp",
    "rmsnorm": "rmsnorm.comp",
}


def _load_shader(name):
    with open(os.path.join(SHADER_DIR, SHADER_FILES[name]), "r", encoding="utf-8") as f:
        return f.read()


def _ceil_div(a, b):
    return (a + b - 1) // b


def _push(v0=0, v1=0, v2=0, v3=0, f0=0.0, f1=0.0, f2=0.0, f3=0.0):
    return struct.pack("<4I4f", v0, v1, v2, v3, f0, f1, f2, f3)


class GPU:
    """High-level operator interface over one Vulkan compute device."""

    def __init__(self):
        self.dev = VulkanDevice()
        self.pipelines = {}
        self._dummy = self.dev.alloc(64)  # stand-in for unused bias binding

    def info(self):
        return self.dev.info()

    # -- buffer helpers -----------------------------------------------------
    def tensor(self, arr):
        """Upload a numpy array to a GPU buffer (float32)."""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        b = self.dev.alloc(max(arr.nbytes, 16))
        b.upload(arr)
        return b

    def tensor_f16(self, arr):
        """Upload a numpy array as float16 (for FP16 weights)."""
        arr = np.ascontiguousarray(arr, dtype=np.float16)
        b = self.dev.alloc(max(arr.nbytes, 16))
        b.upload(arr)
        return b

    def empty(self, nbytes):
        return self.dev.alloc(nbytes)

    @staticmethod
    def to_np(buf, shape):
        return buf.np(np.float32, count=int(np.prod(shape))).reshape(shape).copy()

    # -- pipelines ----------------------------------------------------------
    FP16_PIPELINES = ("gemm_fp16", "cvt_f32_f16")

    def _pipeline(self, name, nbind):
        if name in self.FP16_PIPELINES and not self.dev.fp16_enabled:
            raise RuntimeError(
                f"pipeline '{name}' requires shaderFloat16 + storageBuffer16BitAccess "
                f"(unsupported on {self.dev.device_name})")
        p = self.pipelines.get(name)
        if p is None:
            p = self.dev.make_pipeline(_load_shader(name), nbind)
            self.pipelines[name] = p
        return p

    # -- operators ----------------------------------------------------------
    def gemm(self, A, B, M, N, K, C=None, act=ACT_NONE, bT=False, out=None):
        """D[M,N] = act(A[M,K] @ B + C).  B is [K,N] flat (bT=0) or [N,K] (bT=1)."""
        if out is None:
            out = self.dev.alloc(M * N * 4)
        push = _push(M, N, K, 1 if C is not None else 0,
                     float(act), 1.0 if bT else 0.0)
        pl = self._pipeline("gemm", 4)
        self.dev.submit_jobs([{
            "pipeline": pl, "buffers": [A, B, C if C is not None else self._dummy, out],
            "push": push, "gx": _ceil_div(N, 16), "gy": _ceil_div(M, 16), "gz": 1,
        }])
        return out

    def linear(self, x, W, bias=None, act=ACT_NONE, x_rows=None, w_layout="auto", out=None):
        """nn.Linear style: D = act(x @ W^T + bias)  (x: M rows of width x_rows).
        W layouts: 'out_in' [out,in] (torch convention) or 'in_out' [in,out].
        'auto' detects unless square."""
        K = x_rows
        M = x.size // K
        d0, d1 = W.shape
        if w_layout == "auto":
            if d0 == d1:
                raise ValueError("square weight matrix: pass w_layout explicitly")
            bT = (d1 == K)
        elif w_layout == "out_in":
            bT = True
        elif w_layout == "in_out":
            bT = False
        else:
            raise ValueError(f"bad w_layout: {w_layout}")
        N = d0 if bT else d1
        return self.gemm(x, W, M, N, K, C=bias, act=act, bT=bT, out=out)

    def silu_mul(self, a, b, n=None, out=None):
        n = n if n is not None else a.size
        if out is None:
            out = self.dev.alloc(n * 4)
        pl = self._pipeline("silu_mul", 3)
        self.dev.submit_jobs([{"pipeline": pl, "buffers": [a, b, out],
                               "push": _push(n), "gx": _ceil_div(n, 256)}])
        return out

    def add(self, a, b, n=None, out=None):
        n = n if n is not None else a.size
        if out is None:
            out = self.dev.alloc(n * 4)
        pl = self._pipeline("add", 3)
        self.dev.submit_jobs([{"pipeline": pl, "buffers": [a, b, out],
                               "push": _push(n), "gx": _ceil_div(n, 256)}])
        return out

    def rmsnorm(self, x, w, rows, n, eps=1e-6, out=None):
        if out is None:
            out = self.dev.alloc(rows * n * 4)
        pl = self._pipeline("rmsnorm", 3)
        self.dev.submit_jobs([{"pipeline": pl, "buffers": [x, w, out],
                               "push": _push(rows, n, f0=float(eps)), "gx": rows}])
        return out

    # -- FP16 path -----------------------------------------------------------
    def to_f16(self, src, n, dst=None):
        """GPU-side f32 -> f16 buffer conversion."""
        if dst is None:
            dst = self.dev.alloc(max(n * 2, 16))
        pl = self._pipeline("cvt_f32_f16", 2)
        self.dev.submit_jobs([{"pipeline": pl, "buffers": [src, dst],
                               "push": _push(n), "gx": _ceil_div(n, 256)}])
        return dst

    def gemm_fp16(self, A16, B16, M, N, K, bT=False, out=None):
        """Packed-FP16 GEMM: D[M,N] = A16[M,K] @ B16, f32 output.
        Requires K % 32 == 0; additionally N % 4 == 0 when bT=False."""
        if K % 32:
            raise ValueError(f"gemm_fp16 requires K % 32 == 0 (got K={K})")
        if not bT and N % 4:
            raise ValueError(f"gemm_fp16 requires N % 4 == 0 when bT=False (got N={N})")
        if out is None:
            out = self.dev.alloc(M * N * 4)
        pl = self._pipeline("gemm_fp16", 3)
        self.dev.submit_jobs([{
            "pipeline": pl, "buffers": [A16, B16, out],
            "push": _push(M, N, K, 0, 0.0, 1.0 if bT else 0.0),
            "gx": _ceil_div(N, 64), "gy": _ceil_div(M, 64), "gz": 1,
        }])
        return out

    def ffn_block_fp16(self, x, w_norm, w_gate16, w_up16, w_down16,
                       M, d_model, hidden, eps=1e-6, out=None):
        """llama-style FFN with FP16 weights/GEMMs and f32 norm/activation.
        All 7 dispatches (rmsnorm, cvt, gemm, gemm, silu_mul, cvt, gemm)
        recorded into a single command buffer + submit."""
        h32 = self.empty(M * d_model * 4)
        h16 = self.empty(M * d_model * 2)
        g = self.empty(M * hidden * 4)
        u = self.empty(M * hidden * 4)
        a = self.empty(M * hidden * 4)
        a16 = self.empty(M * hidden * 2)
        if out is None:
            out = self.empty(M * d_model * 4)
        pl_rms = self._pipeline("rmsnorm", 3)
        pl_cvt = self._pipeline("cvt_f32_f16", 2)
        pl_g16 = self._pipeline("gemm_fp16", 3)
        pl_sm = self._pipeline("silu_mul", 3)
        jobs = [
            {"pipeline": pl_rms, "buffers": [x, w_norm, h32],
             "push": _push(M, d_model, f0=float(eps)), "gx": M},
            {"pipeline": pl_cvt, "buffers": [h32, h16],
             "push": _push(M * d_model), "gx": _ceil_div(M * d_model, 256)},
            {"pipeline": pl_g16, "buffers": [h16, w_gate16, g],
             "push": _push(M, hidden, d_model, 0, 0.0, 1.0),
             "gx": _ceil_div(hidden, 64), "gy": _ceil_div(M, 64)},
            {"pipeline": pl_g16, "buffers": [h16, w_up16, u],
             "push": _push(M, hidden, d_model, 0, 0.0, 1.0),
             "gx": _ceil_div(hidden, 64), "gy": _ceil_div(M, 64)},
            {"pipeline": pl_sm, "buffers": [g, u, a],
             "push": _push(M * hidden), "gx": _ceil_div(M * hidden, 256)},
            {"pipeline": pl_cvt, "buffers": [a, a16],
             "push": _push(M * hidden), "gx": _ceil_div(M * hidden, 256)},
            {"pipeline": pl_g16, "buffers": [a16, w_down16, out],
             "push": _push(M, d_model, hidden, 0, 0.0, 1.0),
             "gx": _ceil_div(d_model, 64), "gy": _ceil_div(M, 64)},
        ]
        self.dev.submit_jobs(jobs)
        return out

    # -- fused block: one submit for the whole FFN ---------------------------
    def ffn_block(self, x, w_norm, w_gate, w_up, w_down, M, d_model, hidden,
                  eps=1e-6, out=None):
        """llama-style FFN: rmsnorm -> gate/up projections -> silu_mul -> down proj.
        All five dispatches are recorded into a single command buffer + submit."""
        h = self.empty(M * d_model * 4)
        g = self.empty(M * hidden * 4)
        u = self.empty(M * hidden * 4)
        a = self.empty(M * hidden * 4)
        if out is None:
            out = self.empty(M * d_model * 4)
        pl_rms = self._pipeline("rmsnorm", 3)
        pl_gemm = self._pipeline("gemm_v2", 3)  # register-blocked; FFN has no bias
        pl_sm = self._pipeline("silu_mul", 3)
        jobs = [
            {"pipeline": pl_rms, "buffers": [x, w_norm, h],
             "push": _push(M, d_model, f0=float(eps)), "gx": M},
            {"pipeline": pl_gemm, "buffers": [h, w_gate, g],
             "push": _push(M, hidden, d_model, 0, 0.0, 1.0), "gx": _ceil_div(hidden, 64), "gy": _ceil_div(M, 64)},
            {"pipeline": pl_gemm, "buffers": [h, w_up, u],
             "push": _push(M, hidden, d_model, 0, 0.0, 1.0), "gx": _ceil_div(hidden, 64), "gy": _ceil_div(M, 64)},
            {"pipeline": pl_sm, "buffers": [g, u, a],
             "push": _push(M * hidden), "gx": _ceil_div(M * hidden, 256)},
            {"pipeline": pl_gemm, "buffers": [a, w_down, out],
             "push": _push(M, d_model, hidden, 0, 0.0, 1.0), "gx": _ceil_div(d_model, 64), "gy": _ceil_div(M, 64)},
        ]
        self.dev.submit_jobs(jobs)
        return out


# -- weight generation (for the demo safetensors file) -----------------------
def make_ffn_weights(d_model, hidden, seed=0, path=None):
    """Create llama-style FFN weights (torch layout) and optionally save as safetensors."""
    rng = np.random.default_rng(seed)
    w = {
        "norm.weight": (rng.standard_normal(d_model) * 0.2 + 1.0).astype(np.float32),
        "ffn.gate_proj.weight": (rng.standard_normal((hidden, d_model)) * (1.0 / np.sqrt(d_model))).astype(np.float32),
        "ffn.up_proj.weight": (rng.standard_normal((hidden, d_model)) * (1.0 / np.sqrt(d_model))).astype(np.float32),
        "ffn.down_proj.weight": (rng.standard_normal((d_model, hidden)) * (1.0 / np.sqrt(hidden))).astype(np.float32),
    }
    if path:
        st.save_file(w, path)
    return w


def ffn_numpy(x, w, eps=1e-6):
    """Numpy reference implementation of the same FFN block."""
    h = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w["norm.weight"]
    g = h @ w["ffn.gate_proj.weight"].T
    u = h @ w["ffn.up_proj.weight"].T
    a = (g / (1.0 + np.exp(-g))) * u
    return a @ w["ffn.down_proj.weight"].T
