"""Dedicated decode-kernel regressions, comparing FP16-quantized inputs."""
import numpy as np
import ops

def main():
 gpu=ops.GPU();rng=np.random.default_rng(1927);keep=[]
 print('[vk]',gpu.info(),flush=True)
 for n,k in [(1,32),(3,96),(65,256),(1024,2560),(2560,9728)]:
  a=(rng.standard_normal((1,k))*.2).astype(np.float16);b=(rng.standard_normal((n,k))*.2).astype(np.float16)
  aa=gpu.tensor_f16(a);bb=gpu.tensor_f16(b);out=gpu.empty(n*4+16);keep.extend([aa,bb,out])
  out.np(np.float32)[:]=-9876
  job,result=gpu.job_gemm_fp16(aa,bb,1,n,k,bT=True,out=out)
  assert 'gemv_fp16' in gpu.pipelines and job['pipeline'] is gpu.pipelines['gemv_fp16'], 'M=1 transposed weights must use the dedicated GEMV kernel'
  for rep in range(3):
   if rep:
    a=(rng.standard_normal((1,k))*.2).astype(np.float16)
    aa.upload(a)  # Reuse warm descriptors with changing input values.
   gpu.dev.submit_jobs([job])
   got=out.np(np.float32,count=n).copy()
   ref=(a.astype(np.float64)@b.astype(np.float64).T).ravel()
   err=float(np.max(np.abs(got-ref))/(np.max(np.abs(ref))+1e-9));assert err<2e-5,(n,k,rep,err)
   u=np.finfo(np.float32).eps/2
   gamma=(k+16)*u/(1-(k+16)*u)
   bound=gamma*(np.abs(a.astype(np.float64))@np.abs(b.astype(np.float64)).T).ravel()
   assert np.all(np.abs(got-ref)<=bound+1e-12),'elementwise FP32 accumulation error'
   assert np.all(out.np(np.float32)[n:]==-9876),'write beyond output'
  gpu.gemm_fp16(aa,bb,1,n,k,bT=True,out=out)
  np.testing.assert_allclose(out.np(np.float32,count=n),got,rtol=0,atol=0)
  print('PASS GEMV',n,k,'rel_err',err,flush=True)
 # Both public entry points must retain shape validation.
 for fn in (gpu.job_gemm_fp16,gpu.gemm_fp16):
  for M,N,K,bT in [(1,4,33,True),(1,3,32,False)]:
   try:fn(aa,bb,M,N,K,bT=bT,out=out)
   except ValueError:pass
   else:raise AssertionError('missing FP16 shape validation')
 # Original matrix kernel remains the fallback for prefill and non-transposed B.
 for m,n,k,bt in [(2,65,96,True),(1,68,96,False),(7,64,32,False)]:
  a=rng.standard_normal((m,k)).astype(np.float16)*.1
  b=rng.standard_normal((n,k) if bt else (k,n)).astype(np.float16)*.1
  aa=gpu.tensor_f16(a);bb=gpu.tensor_f16(b);out=gpu.empty(m*n*4);keep.extend((aa,bb,out))
  job,_=gpu.job_gemm_fp16(aa,bb,m,n,k,bT=bt,out=out)
  assert job['pipeline'] is gpu.pipelines['gemm_fp16'],'prefill/non-transposed fallback changed'
  gpu.dev.submit_jobs([job]);ref=a.astype(np.float32)@(b.astype(np.float32).T if bt else b.astype(np.float32))
  got=out.np(np.float32,count=m*n).reshape(m,n)
  err=float(np.max(np.abs(got-ref))/(np.max(np.abs(ref))+1e-9));assert err<.01,(m,n,k,bt,err)
  print('PASS fallback',m,n,k,bt,err,flush=True)
 print('ALL DECODE TESTS PASSED',flush=True)
if __name__=='__main__':main()
