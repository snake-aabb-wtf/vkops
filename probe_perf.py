# -*- coding: utf-8 -*-
"""probe_perf.py — measure device rooflines: FP16 support, memory bandwidth, FMA peak."""
import ctypes
import statistics
import struct
import sys
import time

import numpy as np

import vk
from vk import (H, VkDescriptorSetLayoutBinding, VkDescriptorSetLayoutCreateInfo,
                ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, DESCRIPTOR_TYPE_STORAGE_BUFFER,
                SHADER_STAGE_COMPUTE)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---- 1. features ------------------------------------------------------------
class VkPhysicalDeviceShaderFloat16Int8Features(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H),
                ("shaderFloat16", ctypes.c_uint32), ("shaderInt8", ctypes.c_uint32)]


class VkPhysicalDevice16BitStorageFeatures(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H),
                ("storageBuffer16BitAccess", ctypes.c_uint32),
                ("uniformAndStorageBuffer16BitAccess", ctypes.c_uint32),
                ("storagePushConstant16", ctypes.c_uint32),
                ("storageInputOutput16", ctypes.c_uint32)]


class VkPhysicalDeviceFeatures2(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("features", ctypes.c_uint32 * 64)]


dev = vk.VulkanDevice()
print(f"device: {dev.info()}\n")

f16 = VkPhysicalDeviceShaderFloat16Int8Features()
f16.sType = 1000082000
b16 = VkPhysicalDevice16BitStorageFeatures()
b16.sType = 1000083000
f2 = VkPhysicalDeviceFeatures2()
f2.sType = 1000059000
f2.pNext = ctypes.addressof(f16)
f16.pNext = ctypes.addressof(b16)
dev.call("vkGetPhysicalDeviceFeatures2", dev.physical_device, ctypes.byref(f2))
print(f"shaderFloat16            = {bool(f16.shaderFloat16)}   (packed FP16 math -> ~2x compute)")
print(f"shaderInt8               = {bool(f16.shaderInt8)}")
print(f"storageBuffer16BitAccess = {bool(b16.storageBuffer16BitAccess)}   (FP16 buffers)\n")


# ---- helpers ----------------------------------------------------------------
BANDWIDTH_GLSL = """
#version 450
layout(local_size_x = 256) in;
layout(std430, set = 0, binding = 0) readonly buffer A { vec4 a[]; };
layout(std430, set = 0, binding = 1) writeonly buffer O { float o[]; };
layout(push_constant) uniform PC { uvec4 dims; } pc;  // y = total vec4 count
shared float s[256];
void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;
    uint stride = gl_NumWorkGroups.x * 256u;
    float acc = 0.0;
    for (uint i = gid; i < pc.dims.y; i += stride) {
        acc += dot(a[i], vec4(1.0));
    }
    s[tid] = acc;
    barrier();
    for (uint st = 128u; st >= 1u; st >>= 1u) {
        if (tid < st) s[tid] += s[tid + st];
        barrier();
    }
    if (tid == 0u) o[gl_WorkGroupID.x] = s[0];
}
"""

FMA_GLSL = """
#version 450
layout(local_size_x = 256) in;
layout(std430, set = 0, binding = 0) readonly buffer A { float a[]; };
layout(std430, set = 0, binding = 1) writeonly buffer O { float o[]; };
layout(push_constant) uniform PC { uvec4 dims; } pc;  // x = iterations
void main() {
    uint gid = gl_GlobalInvocationID.x;
    float x = a[gid];
    float a0 = x, a1 = x, a2 = x, a3 = x, a4 = x, a5 = x, a6 = x, a7 = x;
    for (uint i = 0u; i < pc.dims.x; ++i) {
        a0 = fma(a0, 0.999, 0.001); a1 = fma(a1, 0.999, 0.001);
        a2 = fma(a2, 0.999, 0.001); a3 = fma(a3, 0.999, 0.001);
        a4 = fma(a4, 0.999, 0.001); a5 = fma(a5, 0.999, 0.001);
        a6 = fma(a6, 0.999, 0.001); a7 = fma(a7, 0.999, 0.001);
    }
    o[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
"""


def bench(glsl, buffers, push, gx, flops_per_thread_iter, iters, reps=7):
    pl = dev.make_pipeline(glsl, len(buffers))
    bufs = [dev.alloc(b) for b in buffers]
    out_buf = bufs[-1]
    jobs = [{"pipeline": pl, "buffers": bufs, "push": push, "gx": gx}]
    dev.submit_jobs(jobs)  # warmup / compile
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        dev.submit_jobs(jobs)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return med, bufs


# ---- 2. memory bandwidth ------------------------------------------------------
N_VEC4 = 16 * 1024 * 1024          # 256 MB of vec4 reads
WGS = 8192
t, bufs = bench(BANDWIDTH_GLSL, [N_VEC4 * 16, WGS * 4],
                struct.pack("<4I", 0, N_VEC4, 0, 0), WGS, 1, N_VEC4)
bw = N_VEC4 * 16 / t / 1e9
print(f"memory bandwidth (read+reduce, 256MB): {bw:.1f} GB/s  ({t*1000:.2f} ms)")

# ---- 3. FMA peak ---------------------------------------------------------------
THREADS = 8192 * 256
ITERS = 20000
t, bufs = bench(FMA_GLSL, [THREADS * 4, THREADS * 4],
                struct.pack("<4I", ITERS, 0, 0, 0), 8192, 1, ITERS)
flops = THREADS * ITERS * 8 * 2 / t / 1e12
print(f"FMA peak (8 independent chains): {flops:.2f} TFLOPS  ({t*1000:.2f} ms)")
