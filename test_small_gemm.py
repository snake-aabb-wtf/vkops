"""Small-row verifier GEMM: FP64 reference, repeated buffers and output canaries."""
import os
import time
import statistics
import numpy as np
import ops


def main():
    gpu = ops.GPU()
    rng = np.random.default_rng(1708)
    keep = []
    print('[device]', gpu.info(), flush=True)
    for m,n,k in [(2,1,32),(3,65,96),(4,1024,2560),(5,17,96),(7,65,256),(8,2560,9728)]:
        a = rng.normal(0,.2,(m,k)).astype(np.float16)
        b = rng.normal(0,.2,(n,k)).astype(np.float16)
        aa,bb = gpu.tensor_f16(a),gpu.tensor_f16(b)
        out = gpu.empty(m*n*4+64)
        keep.extend((aa,bb,out))
        out.np(np.float32)[:] = -9876
        job,_ = gpu.job_gemm_fp16(aa,bb,m,n,k,bT=True,out=out)
        batch = 2 if m<=2 else 4 if m<=4 else 8
        name = f'gemm_small{batch}_fp16'
        assert name in gpu.pipelines and job['pipeline'] is gpu.pipelines[name], 'small-row kernel is not selected'
        for rep in range(3):
            if rep:
                a = rng.normal(0,.2,(m,k)).astype(np.float16)
                aa.upload(a)
            gpu.dev.submit_jobs([job])
            got = out.np(np.float32,count=m*n).reshape(m,n).copy()
            ref = a.astype(np.float64)@b.astype(np.float64).T
            u = np.finfo(np.float32).eps/2
            gamma = (k+16)*u/(1-(k+16)*u)
            bound = gamma*(abs(a.astype(np.float64))@abs(b.astype(np.float64)).T)
            assert np.all(abs(got-ref)<=bound+1e-12), (m,n,k,'FP32 error bound')
            assert np.all(out.np(np.float32)[m*n:]==-9876), 'output overflow'
        print('PASS',m,n,k,float(np.max(abs(got-ref))),flush=True)
        single_a = gpu.empty(k*2)
        single_out = gpu.empty(n*4)
        keep.extend((single_a,single_out))
        for row in range(m):
            single_a.upload(a[row])
            gpu.gemm_fp16(single_a,bb,1,n,k,bT=True,out=single_out)
            np.testing.assert_array_equal(got[row],single_out.np(np.float32,count=n))
        print('PASS identical GEMV reduction',m,n,k,flush=True)
    # Production-scale bandwidth/latency, same allocation and warmed pipelines.
    n,k = 9728,2560
    b = rng.normal(0,.1,(n,k)).astype(np.float16)
    bb = gpu.tensor_f16(b);keep.append(bb)
    for m in (1,2,4,8):
        aa = gpu.tensor_f16(rng.normal(0,.1,(m,k)).astype(np.float16))
        out = gpu.empty(m*n*4);keep.extend((aa,out))
        for disable in ('1','0'):
            os.environ['VKOPS_DISABLE_SMALL_GEMM'] = disable
            job,_ = gpu.job_gemm_fp16(aa,bb,m,n,k,bT=True,out=out)
            gpu.dev.submit_jobs([job])
            samples=[]
            for _ in range(7):
                t=time.perf_counter();gpu.dev.submit_jobs([job]);samples.append(time.perf_counter()-t)
            print('BENCH',m,'disabled='+disable,'ms',statistics.median(samples)*1e3,flush=True)
    os.environ.pop('VKOPS_DISABLE_SMALL_GEMM',None)
    print('ALL SMALL GEMM TESTS PASSED',flush=True)


if __name__=='__main__':
    main()
