"""CPU-only tests for the experimental Vulkan DFlash adapter."""
import importlib.util
import inspect
import json
import pathlib
import struct
import tempfile
import unittest

import numpy as np


class CPUBuffer:
    def __init__(self, size):
        self.size = size
        self.data = np.zeros(size, np.uint8)

    def upload(self, arr):
        raw = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        self.data[:len(raw)] = raw

    def np(self, dtype, count):
        return self.data.view(dtype)[:count]


class CPUGPU:
    """Real NumPy implementation of the narrow GPU buffer/GEMM interface."""
    def __init__(self):
        self.calls = []
        self.allocations = 0

    def empty(self, nbytes):
        self.allocations += 1
        return CPUBuffer(nbytes)

    def tensor_f16(self, arr):
        out = self.empty(arr.size * 2)
        out.upload(np.asarray(arr, np.float16))
        return out

    def gemm_fp16(self, a, b, m, n, k, bT=False, out=None):
        assert bT and m <= 8
        self.calls.append((m, n, k, id(a), id(out)))
        x = a.np(np.float16, m*k).reshape(m, k).astype(np.float32)
        w = b.np(np.float16, n*k).reshape(n, k).astype(np.float32)
        out.upload(x @ w.T)
        return out


def config():
    return dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                vocab_size=19, rms_norm_eps=1e-6, rope_theta=1e6,
                dflash_config=dict(target_layer_ids=[0, 2], mask_token_id=18))


def weights(c, seed=9):
    rng = np.random.default_rng(seed)
    h, inter, d = c['hidden_size'], c['intermediate_size'], c['head_dim']
    shapes = {'fc.weight': (h, 2*h), 'hidden_norm.weight': (h,), 'norm.weight': (h,)}
    for i in range(c['num_hidden_layers']):
        prefix = f'layers.{i}.'
        shapes.update({prefix + key: value for key, value in {
            'input_layernorm.weight': (h,), 'post_attention_layernorm.weight': (h,),
            'self_attn.q_proj.weight': (4*d, h), 'self_attn.k_proj.weight': (2*d, h),
            'self_attn.v_proj.weight': (2*d, h), 'self_attn.o_proj.weight': (h, 4*d),
            'self_attn.q_norm.weight': (d,), 'self_attn.k_norm.weight': (d,),
            'mlp.gate_proj.weight': (inter,h), 'mlp.up_proj.weight': (inter,h),
            'mlp.down_proj.weight': (h,inter)}.items()})
    return {k: (rng.normal(1,.1,s) if len(s)==1 else rng.normal(0,.12,s)).astype(np.float32)
            for k,s in shapes.items()}


def reference(w, embed, c, features, seed, pos, block):
    """No draft cache; independent per-head float64 attention reference.

    Matches half-quantized linear inputs/weights, not BF16 Transformers math.
    """
    def linear(x, key):
        return (x.astype(np.float16).astype(np.float32) @
                w[key].astype(np.float16).astype(np.float32).T)
    def norm(x, key):
        return x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+c['rms_norm_eps']) * w[key]
    def rotate(x, positions, heads):
        z=x.reshape(len(positions),heads,c['head_dim']).astype(np.float64)
        d=c['head_dim']//2
        angles=positions[:,None]/c['rope_theta']**(np.arange(d)/d)[None,:]
        cs,sn=np.cos(angles)[:,None,:],np.sin(angles)[:,None,:]
        return np.concatenate((z[...,:d]*cs-z[...,d:]*sn,z[...,d:]*cs+z[...,:d]*sn),-1).astype(np.float32)
    context=norm(linear(features,'fc.weight'),'hidden_norm.weight')
    ids=[seed]+[c['dflash_config']['mask_token_id']]*(block-1)
    h=embed[ids].astype(np.float32)
    for i in range(c['num_hidden_layers']):
        p=f'layers.{i}.';x=norm(h,p+'input_layernorm.weight')
        allx=np.concatenate((context,x))
        q=linear(x,p+'self_attn.q_proj.weight').reshape(block,4,8)
        k=linear(allx,p+'self_attn.k_proj.weight').reshape(pos+block,2,8)
        v=linear(allx,p+'self_attn.v_proj.weight').reshape(pos+block,2,8)
        q=rotate(norm(q,p+'self_attn.q_norm.weight'),np.arange(pos,pos+block),4)
        k=rotate(norm(k,p+'self_attn.k_norm.weight'),np.arange(pos+block),2)
        att=np.empty_like(q)
        for head in range(4):
            score=q[:,head].astype(np.float64) @ k[:,head//2].astype(np.float64).T / np.sqrt(8)
            prob=np.exp(score-score.max(-1,keepdims=True));prob/=prob.sum(-1,keepdims=True)
            att[:,head]=prob@v[:,head//2].astype(np.float64)
        h=h+linear(att.reshape(block,32),p+'self_attn.o_proj.weight')
        x=norm(h,p+'post_attention_layernorm.weight')
        g=linear(x,p+'mlp.gate_proj.weight');u=linear(x,p+'mlp.up_proj.weight')
        h=h+linear((g/(1+np.exp(-g)))*u,p+'mlp.down_proj.weight')
    return norm(h,'norm.weight')[1:].astype(np.float16).astype(np.float32) @ embed.astype(np.float32).T


class DFlashTests(unittest.TestCase):
    def setUp(self):
        path=pathlib.Path(__file__).with_name('dflash_draft.py')
        self.assertTrue(path.exists(), 'DFlashDraftEngine implementation is missing')
        spec=importlib.util.spec_from_file_location('draft_under_test',path)
        self.mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.mod)
        self.c=config();self.w=weights(self.c);self.gpu=CPUGPU()
        self.embed=np.random.default_rng(31).normal(0,.2,(19,32)).astype(np.float16)
        self.features=np.random.default_rng(41).normal(size=(13,64)).astype(np.float32)
        self.engine=self.mod.DFlashDraftEngine(self.gpu,self.w,self.gpu.tensor_f16(self.embed),self.c,max_ctx=24)

    def test_logits_match_independent_uncached_reference(self):
        for block in (2,4,8,16):
            self.engine.clear_cache()
            got=self.engine.draft(3,self.features[:5],5,block)
            ref=reference(self.w,self.embed,self.c,self.features[:5],3,5,block)
            np.testing.assert_allclose(got,ref,rtol=.003,atol=.001)
            np.testing.assert_array_equal(got.argmax(-1),ref.argmax(-1))
            self.assertEqual(got.shape,(block-1,19))
            self.assertEqual(self.engine.cache_length,5)

    def test_requested_logit_prefix_preserves_full_block_and_cache(self):
        self.assertIn('output_rows',inspect.signature(self.engine.draft).parameters)
        full=self.engine.draft(3,self.features[:5],5,16)
        old=[a.copy() for a in self.engine.kc+self.engine.vc]
        for count in (1,3,7,8,15):
            self.engine.clear_cache();self.gpu.calls.clear()
            got=self.engine.draft(3,self.features[:5],5,16,output_rows=count)
            np.testing.assert_allclose(got,full[:count],rtol=1e-6,atol=1e-6)
            self.assertEqual(self.engine.cache_length,5)
            for a,b in zip(old,self.engine.kc+self.engine.vc):np.testing.assert_array_equal(a,b)
            head_rows=sum(call[0] for call in self.gpu.calls if call[1]==self.c['vocab_size'])
            self.assertEqual(head_rows,count)

    def test_invalid_output_rows_preserve_cache_and_issue_no_dispatch(self):
        self.assertIn('output_rows',inspect.signature(self.engine.draft).parameters)
        self.engine.draft(3,self.features[:5],5,4)
        old=[a.copy() for a in self.engine.kc+self.engine.vc]
        self.gpu.calls.clear()
        for count in (0,-1,16,1.5,True,'2'):
            with self.assertRaises(ValueError):
                self.engine.draft(7,self.features[5:8],8,16,output_rows=count)
            self.assertEqual(self.engine.cache_length,5)
            for a,b in zip(old,self.engine.kc+self.engine.vc):np.testing.assert_array_equal(a,b)
        self.assertEqual(self.gpu.calls,[])

    def test_incremental_cache_keeps_only_context_and_reuses_buffers(self):
        allocations=self.gpu.allocations
        self.engine.draft(3,self.features[:5],5,16)
        old=[x.copy() for x in self.engine.kc]
        got=self.engine.draft(7,self.features[5:8],8,4)
        ref=reference(self.w,self.embed,self.c,self.features[:8],7,8,4)
        np.testing.assert_allclose(got,ref,rtol=.003,atol=.001)
        for a,b in zip(old,self.engine.kc):np.testing.assert_array_equal(a[:5],b[:5])
        self.assertEqual(self.engine.cache_length,8)
        self.assertEqual(len({row[3] for row in self.gpu.calls}),1)
        self.assertEqual(len({row[4] for row in self.gpu.calls}),1)
        self.assertEqual(self.gpu.allocations,allocations)

    def test_late_nonfinite_compute_failure_leaves_all_cache_state_unchanged(self):
        self.engine.draft(3,self.features[:5],5,4)
        before=[a.copy() for a in self.engine.kc+self.engine.vc]
        # Corrupt an already-uploaded late-layer matrix to trigger a real
        # non-finite GEMM result after earlier layers staged their context KV.
        matrix=self.engine.w['layers.1.mlp.down_proj.weight']
        matrix.np(np.float16,32*64)[:]=np.inf
        with np.errstate(invalid='ignore',over='ignore'):
            with self.assertRaises(ValueError):self.engine.draft(7,self.features[5:8],8,4)
        self.assertEqual(self.engine.cache_length,5)
        for old,new in zip(before,self.engine.kc+self.engine.vc):np.testing.assert_array_equal(old,new)

    def test_multi_chunk_prefill_empty_delta_and_poisoned_unused_cache(self):
        self.engine.draft(3,self.features,13,4)
        for cache in self.engine.kc+self.engine.vc:cache[13:]=np.nan
        got=self.engine.draft(7,np.empty((0,64),np.float32),13,8)
        ref=reference(self.w,self.embed,self.c,self.features,7,13,8)
        np.testing.assert_allclose(got,ref,rtol=.003,atol=.001)
        self.assertEqual(self.engine.cache_length,13)
        self.engine.clear_cache()
        self.assertEqual(self.engine.cache_length,0)
        for cache in self.engine.kc+self.engine.vc:self.assertTrue(np.all(cache==0))

    def test_bad_requests_do_not_mutate_cache(self):
        self.engine.draft(3,self.features[:5],5,4)
        old=[x.copy() for x in self.engine.kc]
        for seed,features,pos,block in [(19,self.features[5:6],6,4),(3,self.features[:2],6,4),
                                        (3,self.features[5:6],6,1),(3,self.features[5:6],6,25),
                                        (3,self.features[:1]*np.nan,6,4)]:
            with self.assertRaises(ValueError):self.engine.draft(seed,features,pos,block)
            self.assertEqual(self.engine.cache_length,5)
            for a,b in zip(old,self.engine.kc):np.testing.assert_array_equal(a,b)

    def test_prepared_matrix_buffer_capacity_is_checked_before_dispatch(self):
        prepared={key:(value.copy() if value.ndim==1 else self.gpu.tensor_f16(value))
                  for key,value in self.w.items()}
        for invalid in (self.gpu.empty(16),object()):
            with self.assertRaises(ValueError):
                self.mod.DFlashDraftEngine(self.gpu,dict(prepared,**{'fc.weight':invalid}),
                                          self.gpu.tensor_f16(self.embed),self.c,max_ctx=24)
        self.assertEqual(self.gpu.calls,[])

    def test_loader_streams_bf16_and_validates_shapes_and_overflow(self):
        with tempfile.TemporaryDirectory() as td:
            path=pathlib.Path(td)/'tiny.safetensors'
            self.write_bf16(path,self.w)
            loaded=self.mod.load_dflash_weights(self.gpu,path,self.c)
            self.assertEqual(set(loaded),set(self.w))
            self.assertEqual(loaded['norm.weight'].dtype,np.float32)
            rounded={key:((value.view(np.uint32)>>16)<<16).view(np.float32)
                     for key,value in self.w.items()}
            np.testing.assert_array_equal(loaded['norm.weight'],rounded['norm.weight'])
            loaded_engine=self.mod.DFlashDraftEngine(self.gpu,loaded,self.gpu.tensor_f16(self.embed),self.c,max_ctx=24)
            got=loaded_engine.draft(3,self.features[:5],5,4)
            ref=reference(rounded,self.embed,self.c,self.features[:5],3,5,4)
            np.testing.assert_allclose(got,ref,rtol=.003,atol=.001)
            bad=dict(self.w);bad['fc.weight']=np.full((32,64),1e10,np.float32)
            self.write_bf16(path,bad)
            with self.assertRaises(ValueError):self.mod.load_dflash_weights(self.gpu,path,self.c)

    def test_loader_rejects_invalid_headers_before_upload(self):
        with tempfile.TemporaryDirectory() as td:
            path=pathlib.Path(td)/'tiny.safetensors'
            self.write_bf16(path,self.w)
            raw=path.read_bytes();n=struct.unpack('<Q',raw[:8])[0]
            original=json.loads(raw[8:8+n]);payload=raw[8+n:]
            for mutation in ('missing','extra','dtype','overlap','gap','range'):
                header=json.loads(json.dumps(original))
                if mutation=='missing':header.pop('norm.weight')
                if mutation=='extra':header['extra']=header['norm.weight']
                if mutation=='dtype':header['fc.weight']['dtype']='F64'
                if mutation=='overlap':header['hidden_norm.weight']['data_offsets']=[0,64]
                if mutation=='gap':header['hidden_norm.weight']['data_offsets']=[4098,4162]
                if mutation=='range':header['fc.weight']['data_offsets']=[len(payload),len(payload)+4096]
                hb=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(hb))+hb+payload)
                count=self.gpu.allocations
                with self.subTest(mutation=mutation):
                    with self.assertRaises(ValueError):self.mod.load_dflash_weights(self.gpu,path,self.c)
                    self.assertEqual(self.gpu.allocations,count)
            bad=dict(self.w);bad['norm.weight']=np.full(32,np.nan,np.float32)
            self.write_bf16(path,bad)
            with self.assertRaises(ValueError):self.mod.load_dflash_weights(self.gpu,path,self.c)
            path.write_bytes(b'\xff'*8)
            with self.assertRaises(ValueError):self.mod.load_dflash_weights(self.gpu,path,self.c)
            bad=dict(self.w);bad['norm.weight']=np.ones(31,np.float32)
            self.write_bf16(path,bad)
            with self.assertRaises(ValueError):self.mod.load_dflash_weights(self.gpu,path,self.c)

    @staticmethod
    def write_bf16(path,weights):
        blobs=[];header={};offset=0
        for key,arr in weights.items():
            raw=(arr.view(np.uint32)>>16).astype('<u2').tobytes()
            header[key]=dict(dtype='BF16',shape=list(arr.shape),data_offsets=[offset,offset+len(raw)])
            offset+=len(raw);blobs.append(raw)
        h=json.dumps(header).encode()
        path.write_bytes(struct.pack('<Q',len(h))+h+b''.join(blobs))


class ToyTarget:
    max_ctx=40
    def __init__(self, next_token):
        self.next_token=next_token;self._feature_layers=();self.calls=[]
        self.clear_cache()
    def enable_feature_capture(self,layers):
        self._feature_layers=tuple(layers);self.last_features=None
    def clear_cache(self):
        self.cache_length=0;self.prefix=[];self.last_features=None
    def truncate_cache(self,length):
        assert 0<=length<=self.cache_length
        self.prefix=self.prefix[:length];self.cache_length=length
    def forward(self,ids,pos_start=0,all_logits=False):
        assert pos_start==self.cache_length and pos_start+len(ids)<=self.max_ctx
        self.calls.append(list(ids));rows=[]
        for token in ids:
            self.prefix.append(token);self.cache_length+=1
            row=np.full(16,-10.,np.float32);row[self.next_token(self.prefix)]=10
            rows.append(row)
        self.last_features=np.arange(pos_start,pos_start+len(ids),dtype=np.float32)[:,None]
        return np.asarray(rows) if all_logits else rows[-1]


class ToyDraft:
    max_ctx=40
    target_layer_ids=(0,)
    def __init__(self,propose):
        self.propose=propose;self.calls=[];self.cache_length=0
    def clear_cache(self):self.cache_length=0
    def draft(self,seed,target_features,position,block_size=16,output_rows=None):
        assert position+block_size<=self.max_ctx
        np.testing.assert_array_equal(target_features[:,0],np.arange(self.cache_length,position))
        output_rows=block_size-1 if output_rows is None else output_rows
        self.calls.append((position,block_size,target_features.copy(),output_rows))
        self.cache_length=position
        rows=np.full((output_rows,16),-10.,np.float32)
        for i in range(output_rows):rows[i,self.propose(seed,i)]=10
        return rows


class GenerationTests(unittest.TestCase):
    def setUp(self):
        import dflash_draft
        self.assertTrue(hasattr(dflash_draft,'generate_dflash'),'greedy DFlash loop is missing')
        self.generate=dflash_draft.generate_dflash

    def test_full_acceptance_callbacks_budget_and_feature_deltas(self):
        target=ToyTarget(lambda p:(p[-1]+1)%10)
        draft=ToyDraft(lambda seed,i:(seed+i+1)%10)
        seen=[];stats={}
        got=self.generate(target,draft,[0],max_new=11,eos=15,verify_size=4,
                          callback=seen.append,stats=stats)
        self.assertEqual(got,[1,2,3,4,5,6,7,8,9,0,1]);self.assertEqual(seen,got)
        self.assertEqual(target.cache_length,12)
        self.assertEqual(stats['accepted_draft'],8)
        self.assertTrue(all(call[1]==16 for call in draft.calls))
        self.assertEqual(target._feature_layers,())
        self.assertGreaterEqual(stats['total_s'],stats['prefill_s']+stats['draft_s']+stats['verify_s'])

    def test_partial_acceptance_and_total_rejection_keep_features_aligned(self):
        for proposal in (lambda seed,i:(seed+1)%10 if i==0 else 14,
                         lambda seed,i:14):
            target=ToyTarget(lambda p:(p[-1]+1)%10);draft=ToyDraft(proposal)
            got=self.generate(target,draft,[0],max_new=9,eos=15,verify_size=4)
            self.assertEqual(got,list(range(1,10)))
            self.assertEqual(target.prefix,[0]+got)

    def test_draft_and_verify_widths_and_draft_capacity_can_differ(self):
        for options,capacity in [({},40),({'verify_size':1},40),
                                 ({'block_size':2,'verify_size':8},40),({},4)]:
            target=ToyTarget(lambda p:(p[-1]+1)%10)
            draft=ToyDraft(lambda seed,i:(seed+i+1)%10);draft.max_ctx=capacity
            stats={}
            got=self.generate(target,draft,[0],max_new=12,eos=15,stats=stats,**options)
            self.assertEqual(got,[1,2,3,4,5,6,7,8,9,0,1,2])
            self.assertEqual(stats.get('computed_draft_rows'),sum(c[3] for c in draft.calls))
            self.assertEqual(stats.get('noise_rows'),sum(c[1] for c in draft.calls))
            self.assertEqual(stats['drafted'],sum(len(c)-1 for c in target.calls[1:]))
            self.assertEqual(stats['computed_draft_rows'],stats['drafted'])

    def test_eos_first_and_accepted_eos_stop_without_callback(self):
        for stop in (1,3):
            target=ToyTarget(lambda p:(p[-1]+1)%16);draft=ToyDraft(lambda seed,i:(seed+i+1)%16)
            seen=[]
            got=self.generate(target,draft,[0],max_new=8,eos=stop,callback=seen.append)
            self.assertEqual(got,list(range(1,stop)));self.assertEqual(seen,got)
            self.assertEqual(target.cache_length,stop)

    def test_capacity_shrinks_block_and_last_single_token(self):
        target=ToyTarget(lambda p:(p[-1]+1)%10);target.max_ctx=8
        draft=ToyDraft(lambda seed,i:(seed+i+1)%10);draft.max_ctx=8
        got=self.generate(target,draft,[0,1,2],max_new=20,eos=15,verify_size=4)
        self.assertEqual(got,[3,4,5,6,7]);self.assertEqual(target.cache_length,8)
        self.assertEqual([x[1] for x in draft.calls],[5])
        self.assertEqual(target.calls[-1],[7])

    def test_zero_budget_invalid_options_and_capture_restoration(self):
        target=ToyTarget(lambda p:1);draft=ToyDraft(lambda seed,i:2)
        self.assertEqual(self.generate(target,draft,[0],max_new=0),[])
        self.assertEqual(target.calls,[])
        for opts in ({'block_size':1},{'verify_size':0},{'max_new':-1}):
            with self.assertRaises(ValueError):self.generate(target,draft,[0],**opts)
        target.enable_feature_capture([2])
        def fail(token):raise RuntimeError('callback failed')
        with self.assertRaises(RuntimeError):self.generate(target,draft,[0],max_new=2,callback=fail)
        self.assertEqual(target._feature_layers,(2,))


if __name__=='__main__':unittest.main(verbosity=2)
