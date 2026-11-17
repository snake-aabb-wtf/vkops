"""CPU-only request routing tests; never starts the server or loads weights."""
import ast
import io
import json
from pathlib import Path
import threading
import tempfile
import os
import time
import unittest
import uuid
import types
from unittest.mock import patch
import numpy as np


class ServerOptionTests(unittest.TestCase):
    def test_optional_loader_validates_target_before_weight_upload(self):
        source=Path(__file__).with_name('server.py').read_text(encoding='utf-8')
        nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='load_optional_draft']
        self.assertEqual(len(nodes),1,'optional draft loader is missing')
        calls=[]
        target=types.SimpleNamespace(H=32,V=16,num_layers=2,max_ctx=64,embed16=object())
        def load(*args):calls.append(('weights',args));return {'weight':'fixture'}
        def engine(*args,**kw):calls.append(('engine',args,kw));return 'draft'
        ns=dict(os=os,json=json,load_dflash_weights=load,DFlashDraftEngine=engine)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'server.py','exec'),ns)
        fn=ns['load_optional_draft']
        self.assertIsNone(fn(None,target,None));self.assertEqual(calls,[])
        good=dict(architectures=['DFlashDraftModel'],hidden_size=32,vocab_size=16,
                  num_target_layers=2,dflash_config={'target_layer_ids':[0,1]})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory,'config.json')
            for change in ({'architectures':['Other']},{'hidden_size':64},{'vocab_size':17},
                           {'num_target_layers':3},{'dflash_config':{'target_layer_ids':[2]}}):
                path.write_text(json.dumps(dict(good,**change)),encoding='utf-8')
                with self.assertRaises(ValueError):fn(None,target,directory)
                self.assertEqual(calls,[])
            path.write_text(json.dumps(good),encoding='utf-8')
            for flag in ('VKOPS_DISABLE_GEMV','VKOPS_DISABLE_SMALL_GEMM'):
                with patch.dict(os.environ,{flag:'1'}):
                    with self.assertRaises(ValueError):fn(None,target,directory)
                self.assertEqual(calls,[])
            self.assertEqual(fn(None,target,directory),'draft')
            self.assertEqual(calls[1][2]['max_ctx'],64)

    def request(self, body, draft_available=False):
        source=Path(__file__).with_name('server.py').read_text(encoding='utf-8')
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Handler')
        calls=[];responses=[]
        def generate(*args,**kw):
            calls.append(kw)
            if kw.get('callback'):kw['callback'](3)
            return [3]
        def generate_dflash(*args,**kw):
            kw['speculative_dflash']=True
            return generate(*args,**kw)
        ns=dict(BaseHTTPRequestHandler=object,np=np,json=json,time=time,uuid=uuid,
                TOKEN='unit-test-token',MAX_CTX=1024,MODEL_ID='test',GPU_LOCK=threading.Lock(),
                build_prompt_ids=lambda messages:[1,2],eng=types.SimpleNamespace(generate=generate),
                tok=types.SimpleNamespace(decode=lambda ids:'text'),chat=types.SimpleNamespace(IM_END=15),
                draft_eng=object() if draft_available else None,generate_dflash=generate_dflash)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'server.py','exec'),ns)
        h=ns['Handler']();data=json.dumps(body).encode()
        h.headers={'Content-Length':str(len(data)),'Authorization':'Bearer unit-test-token'}
        h.rfile=io.BytesIO(data);h.wfile=io.BytesIO();h.path='/v1/chat/completions'
        h._send_json=lambda code,obj:responses.append((code,obj))
        h.send_response=lambda code:None;h.send_header=lambda *args:None;h.end_headers=lambda:None
        h.do_POST()
        return calls,responses

    def test_opt_in_reaches_both_generation_paths(self):
        for stream in (False,True):
            calls,_=self.request(dict(messages=[{'role':'user','content':'hi'}],temperature=0,
                                      speculative_ngram=True,stream=stream))
            self.assertTrue(calls[0]['speculative_ngram'])

    def test_sampling_rejects_greedy_only_option(self):
        for extra in ({'temperature':.7,'speculative_ngram':True},
                      {'temperature':0,'speculative_ngram':'true'}):
            calls,responses=self.request(dict(messages=[{'role':'user','content':'hi'}],**extra))
            self.assertEqual(calls,[])
            self.assertEqual(responses[0][0],400)

    def test_default_sampling_remains_disabled(self):
        calls,_=self.request(dict(messages=[{'role':'user','content':'hi'}]))
        self.assertFalse(calls[0].get('speculative_ngram',False))
        self.assertEqual(calls[0]['temperature'],.7)

    def test_optional_dflash_routes_without_changing_defaults(self):
        for stream in (False,True):
            calls,_=self.request(dict(messages=[{'role':'user','content':'hi'}],temperature=0,
                                      speculative_dflash=True,stream=stream),draft_available=True)
            self.assertTrue(calls[0]['speculative_dflash'])
            self.assertEqual(calls[0]['block_size'],4)
            self.assertEqual(calls[0]['verify_size'],4)

    def test_dflash_requires_config_and_unambiguous_greedy_request(self):
        for extra,available in [({'temperature':0,'speculative_dflash':True},False),
                                ({'temperature':.7,'speculative_dflash':True},True),
                                ({'temperature':0,'speculative_dflash':'true'},True),
                                ({'temperature':0,'speculative_dflash':True,'speculative_ngram':True},True)]:
            calls,responses=self.request(dict(messages=[{'role':'user','content':'hi'}],**extra),available)
            self.assertEqual(calls,[])
            self.assertEqual(responses[0][0],400)


if __name__=='__main__':unittest.main()
