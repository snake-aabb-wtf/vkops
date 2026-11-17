"""CPU-only request routing tests; never starts the server or loads weights."""
import ast
import io
import json
from pathlib import Path
import threading
import time
import unittest
import uuid
import types
import numpy as np


class ServerOptionTests(unittest.TestCase):
    def request(self, body):
        source=Path(__file__).with_name('server.py').read_text(encoding='utf-8')
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Handler')
        calls=[];responses=[]
        def generate(*args,**kw):
            calls.append(kw)
            if kw.get('callback'):kw['callback'](3)
            return [3]
        ns=dict(BaseHTTPRequestHandler=object,np=np,json=json,time=time,uuid=uuid,
                TOKEN='unit-test-token',MAX_CTX=1024,MODEL_ID='test',GPU_LOCK=threading.Lock(),
                build_prompt_ids=lambda messages:[1,2],eng=types.SimpleNamespace(generate=generate),
                tok=types.SimpleNamespace(decode=lambda ids:'text'),chat=types.SimpleNamespace(IM_END=15))
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


if __name__=='__main__':unittest.main()
