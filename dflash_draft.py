"""Experimental DFlash1 drafter using NumPy and vkops' FP16-input GEMMs.

Architecture reference (MIT): https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16
This is a local, explicit architecture implementation, not remote-code loading.
The public checkpoint targets original Qwen3-4B, not Instruct-2507/abliterated.
No quality or speed claim is made for a different target model.

Linear operations use <=8-row chunks so the new GEMV/small-GEMM kernels keep
FP32 accumulation. CPU normalization/attention stays FP32. Scratch GPU buffers
are persistent: never allocate/free activation buffers inside the draft loop.
This deliberately simple experiment does not yet batch GPU submissions.
"""
import json
import math
import os
import struct
import time

import numpy as np


def _dimensions(config):
    names = ('hidden_size', 'intermediate_size', 'num_hidden_layers',
             'num_attention_heads', 'num_key_value_heads', 'head_dim', 'vocab_size')
    values = []
    for name in names:
        x = config.get(name)
        if isinstance(x, bool) or not isinstance(x, int) or x <= 0:
            raise ValueError(f'invalid {name}')
        values.append(x)
    h, inter, layers, heads, kv_heads, dim, vocab = values
    selected = config.get('dflash_config', {}).get('target_layer_ids')
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError('target_layer_ids must be a nonempty unique list')
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in selected):
        raise ValueError('invalid target layer index')
    if heads % kv_heads or dim % 2 or h % 32 or inter % 32 or (heads*dim) % 32:
        raise ValueError('incompatible GQA/head/GEMM dimensions')
    if max(h, inter, heads*dim, vocab) > 8*65535:
        raise ValueError('output too wide for FP32 small-GEMM kernels')
    if config.get('attention_bias', False) or config.get('hidden_act', 'silu') != 'silu':
        raise ValueError('only bias-free SiLU DFlash1 is supported')
    if config.get('rope_scaling') or config.get('use_sliding_window', False):
        raise ValueError('scaled RoPE/sliding window is unsupported')
    for name, default in [('rms_norm_eps',1e-6), ('rope_theta',1e6)]:
        if not np.isfinite(config.get(name,default)) or config.get(name,default) <= 0:
            raise ValueError(f'invalid {name}')
    mask = config['dflash_config'].get('mask_token_id')
    if isinstance(mask,bool) or not isinstance(mask,int) or not 0 <= mask < vocab:
        raise ValueError('invalid mask_token_id')
    return (*values, len(selected))


def expected_weight_shapes(config):
    h, inter, layers, heads, kv_heads, dim, _, taps = _dimensions(config)
    result = {'fc.weight': (h,taps*h), 'hidden_norm.weight': (h,), 'norm.weight': (h,)}
    for i in range(layers):
        p=f'layers.{i}.'
        result.update({p+key: shape for key,shape in {
            'input_layernorm.weight': (h,), 'post_attention_layernorm.weight': (h,),
            'self_attn.q_proj.weight': (heads*dim,h),
            'self_attn.k_proj.weight': (kv_heads*dim,h),
            'self_attn.v_proj.weight': (kv_heads*dim,h),
            'self_attn.o_proj.weight': (h,heads*dim),
            'self_attn.q_norm.weight': (dim,), 'self_attn.k_norm.weight': (dim,),
            'mlp.gate_proj.weight': (inter,h), 'mlp.up_proj.weight': (inter,h),
            'mlp.down_proj.weight': (h,inter)}.items()})
    return result


def _half(x, label):
    x=np.asarray(x)
    if not np.isfinite(x).all() or np.any(np.abs(x) > np.finfo(np.float16).max):
        raise ValueError(f'{label}: non-finite or FP16 overflow')
    return np.ascontiguousarray(x,dtype=np.float16)


def _prepare_weight(gpu, key, arr, shape):
    arr=np.asarray(arr,dtype=np.float32)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f'{key}: wrong shape or non-finite weight')
    if len(shape)==1:
        return arr.copy()
    return gpu.tensor_f16(_half(arr,key))


def _checked_half_buffer(buffer, count, label):
    # Vulkan buffers do not carry tensor dtype/shape metadata. Prepared buffers
    # must contain packed FP16 in the documented order; check byte capacity and
    # the mapped-buffer interface before allowing any dispatch.
    size=getattr(buffer,'size',None)
    if not isinstance(size,(int,np.integer)) or size<count*2 or not callable(getattr(buffer,'np',None)):
        raise ValueError(f'{label}: expected a sufficiently large packed-FP16 GPU buffer')
    return buffer


def load_dflash_weights(gpu, path, config):
    """Stream a local single safetensors file; do not load/execute Python files.

    Returns FP16 GPU matrices and FP32 CPU norms. Caller owns returned buffers.
    Entire header is validated before any tensor upload. F16/F32 are also
    accepted for local numeric fixtures; the official checkpoint is BF16.
    """
    shapes=expected_weight_shapes(config)
    itemsize={'BF16':2,'F16':2,'F32':4}
    with open(path,'rb') as f:
        size=os.fstat(f.fileno()).st_size
        raw=f.read(8)
        if len(raw)!=8:raise ValueError('truncated safetensors header')
        length=struct.unpack('<Q',raw)[0]
        if length > min(16*1024*1024,size-8):raise ValueError('invalid header length')
        header=json.loads(f.read(length))
        header.pop('__metadata__',None)
        if set(header)!=set(shapes):raise ValueError('unexpected/missing DFlash tensors')
        ranges=[]
        for key,shape in shapes.items():
            entry=header[key];dtype=entry.get('dtype');offsets=entry.get('data_offsets')
            if entry.get('shape')!=list(shape) or dtype not in itemsize:
                raise ValueError(f'{key}: invalid shape/dtype')
            if not isinstance(offsets,list) or len(offsets)!=2 or any(type(x)!=int for x in offsets):
                raise ValueError(f'{key}: invalid offsets')
            start,end=offsets
            if start<0 or end-start!=math.prod(shape)*itemsize[dtype] or end>size-8-length:
                raise ValueError(f'{key}: invalid tensor extent')
            ranges.append((start,end))
        cursor=0
        for start,end in sorted(ranges):
            if start!=cursor:raise ValueError('overlapping/gapped tensor extents')
            cursor=end
        if cursor!=size-8-length:raise ValueError('unclaimed tensor bytes')
        result={}
        for key,shape in shapes.items():
            entry=header[key];start,end=entry['data_offsets']
            f.seek(8+length+start);raw=f.read(end-start)
            if len(raw)!=end-start:raise ValueError('truncated tensor data')
            if entry['dtype']=='BF16':
                arr=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
            else:
                arr=np.frombuffer(raw,dtype={'F16':'<f2','F32':'<f4'}[entry['dtype']]).astype(np.float32)
            result[key]=_prepare_weight(gpu,key,arr.reshape(shape),shape)
    return result


class DFlashDraftEngine:
    """draft(seed, new_target_features, position, B, output_rows=None).

    Returns logits[output_rows,V], default B-1. All B noise positions still
    participate in every noncausal layer; only final LM-head rows are trimmed.
    new_target_features must cover exactly [cache_length, position).
    After success the cache contains target-context KV for [0, position),
    never noise-token KV. Clear this cache when starting a new conversation.
    Weights may be local NumPy arrays or load_dflash_weights' prepared mapping.
    GPU buffers and their owning GPU must outlive this engine.
    """
    def __init__(self,gpu,weights,embed16,config,max_ctx=512):
        dims=_dimensions(config)
        self.H,self.inter,self.layers,self.heads,self.kv_heads,self.dim,self.V,self.taps=dims
        if type(max_ctx)!=int or max_ctx<2:raise ValueError('invalid max_ctx')
        self.gpu,self.max_ctx=gpu,max_ctx
        self.embed16=_checked_half_buffer(embed16,self.V*self.H,'embedding')
        self.eps=config.get('rms_norm_eps',1e-6)
        self.theta=config.get('rope_theta',1e6)
        self.mask=config['dflash_config']['mask_token_id']
        self.target_layer_ids=tuple(config['dflash_config']['target_layer_ids'])
        self.shapes=expected_weight_shapes(config)
        if set(weights)!=set(self.shapes):raise ValueError('unexpected/missing DFlash weights')
        self.w={}
        for key,shape in self.shapes.items():
            value=weights[key]
            if isinstance(value,np.ndarray):value=_prepare_weight(gpu,key,value,shape)
            elif len(shape)==1:raise ValueError('norm weights must be CPU arrays')
            else:value=_checked_half_buffer(value,math.prod(shape),key)
            self.w[key]=value
        self.embed=embed16.np(np.float16,count=self.V*self.H).reshape(self.V,self.H)
        max_in=max(self.H*self.taps,self.inter,self.heads*self.dim)
        max_out=max(self.H,self.inter,self.heads*self.dim,self.V)
        self.a16=gpu.empty(8*max_in*2)
        self.out=gpu.empty(8*max_out*4)
        self.kc=[np.zeros((max_ctx,self.kv_heads,self.dim),np.float32) for _ in range(self.layers)]
        self.vc=[np.zeros_like(x) for x in self.kc]
        self.cache_length=0

    def clear_cache(self):
        self.cache_length=0
        for x in self.kc+self.vc:x.fill(0)

    def _linear(self,x,key):
        weight=self.embed16 if key is None else self.w[key]
        n=self.V if key is None else self.shapes[key][0]
        x=_half(x,'linear input');m,k=x.shape
        result=np.empty((m,n),np.float32)
        for start in range(0,m,8):
            chunk=x[start:start+8];self.a16.upload(chunk)
            self.gpu.gemm_fp16(self.a16,weight,len(chunk),n,k,bT=True,out=self.out)
            result[start:start+len(chunk)]=self.out.np(np.float32,count=len(chunk)*n).reshape(len(chunk),n)
        if not np.isfinite(result).all():raise ValueError('non-finite linear output')
        return result

    def _norm(self,x,key):
        return x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+self.eps)*self.w[key]

    def _rope(self,x,positions):
        z=x.astype(np.float64);half=self.dim//2
        angle=positions[:,None]/self.theta**(np.arange(half,dtype=np.float64)/half)[None,:]
        cs,sn=np.cos(angle)[:,None,:],np.sin(angle)[:,None,:]
        return np.concatenate((z[...,:half]*cs-z[...,half:]*sn,
                               z[...,half:]*cs+z[...,:half]*sn),axis=-1).astype(np.float32)

    def draft(self,seed,target_features,position,block_size=16,output_rows=None):
        if os.environ.get('VKOPS_DISABLE_SMALL_GEMM')=='1' or os.environ.get('VKOPS_DISABLE_GEMV')=='1':
            raise ValueError('DFlash strict FP32 path requires GEMV and small GEMM enabled')
        if not isinstance(seed,(int,np.integer)) or isinstance(seed,(bool,np.bool_)) or not 0<=seed<self.V:
            raise ValueError('seed outside vocabulary')
        if type(position)!=int or type(block_size)!=int or block_size<2 or position<self.cache_length or position+block_size>self.max_ctx:
            raise ValueError('invalid position/block_size/context capacity')
        if output_rows is None:output_rows=block_size-1
        if (isinstance(output_rows,(bool,np.bool_)) or not isinstance(output_rows,(int,np.integer))
                or not 1<=output_rows<block_size):
            raise ValueError('output_rows must be an integer in [1, block_size-1]')
        output_rows=int(output_rows)
        f=np.asarray(target_features,dtype=np.float32)
        if f.shape!=(position-self.cache_length,self.H*self.taps) or not np.isfinite(f).all():
            raise ValueError('features must cover exactly uncached target positions')
        context=self._norm(self._linear(f,'fc.weight'),'hidden_norm.weight')
        h=self.embed[[int(seed)]+[self.mask]*(block_size-1)].astype(np.float32)
        new_keys=[];new_values=[]
        for i in range(self.layers):
            p=f'layers.{i}.'
            x=self._norm(h,p+'input_layernorm.weight')
            q=self._linear(x,p+'self_attn.q_proj.weight').reshape(block_size,self.heads,self.dim)
            combined=np.concatenate((context,x),axis=0)
            k=self._linear(combined,p+'self_attn.k_proj.weight').reshape(-1,self.kv_heads,self.dim)
            v=self._linear(combined,p+'self_attn.v_proj.weight').reshape(-1,self.kv_heads,self.dim)
            q=self._rope(self._norm(q,p+'self_attn.q_norm.weight'),np.arange(position,position+block_size))
            k=self._rope(self._norm(k,p+'self_attn.k_norm.weight'),np.arange(self.cache_length,position+block_size))
            # Stage only real context; commit after every layer/logit succeeds.
            new_keys.append(k[:len(context)].copy());new_values.append(v[:len(context)].copy())
            allk=np.concatenate((self.kc[i][:self.cache_length],k),axis=0)
            allv=np.concatenate((self.vc[i][:self.cache_length],v),axis=0)
            att=np.empty_like(q)
            for head in range(self.heads):
                kv=head//(self.heads//self.kv_heads)
                score=(q[:,head] @ allk[:,kv].T)/math.sqrt(self.dim)
                score-=score.max(-1,keepdims=True)
                prob=np.exp(score);prob/=prob.sum(-1,keepdims=True)
                att[:,head]=prob@allv[:,kv]
            h=h+self._linear(att.reshape(block_size,-1),p+'self_attn.o_proj.weight')
            x=self._norm(h,p+'post_attention_layernorm.weight')
            gate=self._linear(x,p+'mlp.gate_proj.weight')
            up=self._linear(x,p+'mlp.up_proj.weight')
            z=np.exp(-np.abs(gate))
            sigmoid=np.where(gate>=0,1/(1+z),z/(1+z))
            h=h+self._linear(gate*sigmoid*up,p+'mlp.down_proj.weight')
        logits=self._linear(self._norm(h,'norm.weight')[1:1+output_rows],None)
        for i in range(self.layers):
            self.kc[i][self.cache_length:position]=new_keys[i]
            self.vc[i][self.cache_length:position]=new_values[i]
        self.cache_length=position
        return logits


def generate_dflash(target,drafter,prompt_ids,max_new=128,eos=151645,
                    block_size=16,verify_size=8,stats=None,callback=None):
    """Opt-in greedy-only DFlash experiment; never changes target weights.

    B16 is the checkpoint's training shape. B2..15 are experimental shapes.
    The full B-token noise backbone runs, but the LM head computes only the
    at-most verify_size-1 proposals needed. Near context capacity B shrinks;
    below two rows use ordinary decode.
    Accepted prefixes follow target block argmax, not a promise of bit-identical
    logits across different GEMM shapes. EOS is not returned or callbacked.
    Target feature capture is restored on exit, including exceptional exits.
    Timings exclude model loading but include every generation-stage cost in
    total_s (and callbacks in decode_s). draft_s includes first context setup.
    decode_s already contains draft_s/verify_s; do not add them again. drafted
    counts proposals actually sent to the target, computed_draft_rows counts
    actual LM-head output rows, and noise_rows counts full B-row backbone
    evaluations (including the seed), even when fewer logits are requested. tokens
    counts accepted output, not successful callback deliveries on exceptions.
    """
    for name,value,lo,hi in [('max_new',max_new,0,2**31-1),
                             ('block_size',block_size,2,16),('verify_size',verify_size,1,8)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo<=value<=hi:
            raise ValueError(f'{name} must be an integer in [{lo},{hi}]')
    prompt=list(prompt_ids)
    if not prompt or len(prompt)>target.max_ctx:
        raise ValueError('prompt must be nonempty and fit target context')
    meter=dict(prefill_s=0.,draft_s=0.,verify_s=0.,decode_s=0.,total_s=0.,
               cycles=0,draft_calls=0,drafted=0,computed_draft_rows=0,noise_rows=0,accepted_draft=0,tokens=0,
               block_size=int(block_size),verify_size=int(verify_size))
    budget=min(int(max_new),target.max_ctx-len(prompt))
    if budget==0:
        if stats is not None:stats.update(meter)
        return []
    start=time.perf_counter();decode_start=None;generated=[]
    old_capture=getattr(target,'_feature_layers',())
    try:
        target.enable_feature_capture(drafter.target_layer_ids)
        target.clear_cache();drafter.clear_cache()
        tick=time.perf_counter();logits=target.forward(prompt,0)
        meter['prefill_s']=time.perf_counter()-tick
        features=target.last_features
        position=len(prompt);decode_start=time.perf_counter()
        while len(generated)<budget:
            seed=int(np.argmax(logits))
            if seed==eos:break
            remaining=budget-len(generated)
            width=min(int(verify_size),remaining,target.max_ctx-position)
            draft_width=min(int(block_size),target.max_ctx-position,drafter.max_ctx-position)
            proposals=[]
            if width>=2 and draft_width>=2:
                output_rows=min(width-1,draft_width-1)
                tick=time.perf_counter()
                draft_logits=drafter.draft(seed,features,position,draft_width,output_rows=output_rows)
                meter['draft_s']+=time.perf_counter()-tick
                meter['computed_draft_rows']+=output_rows
                meter['noise_rows']+=draft_width
                if draft_logits.shape!=(output_rows,len(logits)) or not np.isfinite(draft_logits).all():
                    raise ValueError('drafter returned invalid logits')
                proposals=np.argmax(draft_logits[:width-1],axis=-1).tolist()
                meter['draft_calls']+=1
            candidate=[seed]+proposals
            tick=time.perf_counter()
            rows=target.forward(candidate,position,all_logits=True)
            meter['verify_s']+=time.perf_counter()-tick
            accepted=1
            for index,token in enumerate(proposals):
                if token!=int(np.argmax(rows[index])):break
                accepted+=1
            chosen=candidate[:accepted]
            stopped=eos in chosen
            if stopped:chosen=chosen[:chosen.index(eos)]
            # Only accepted target rows become the next draft context delta.
            features=target.last_features[:len(chosen)].copy()
            target.truncate_cache(position+len(chosen))
            position+=len(chosen);generated.extend(chosen)
            meter['cycles']+=1;meter['drafted']+=len(proposals)
            meter['accepted_draft']+=max(0,len(chosen)-1)
            for token in chosen:
                if callback is not None:callback(token)
            if stopped:break
            logits=rows[accepted-1]
    finally:
        if decode_start is not None:meter['decode_s']=time.perf_counter()-decode_start
        target.enable_feature_capture(old_capture)
        meter['tokens']=len(generated);meter['total_s']=time.perf_counter()-start
        if stats is not None:stats.update(meter)
    return generated
