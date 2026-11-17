"""Reproducible full-36-layer Qwen3-4B benchmark. No model download required."""
import argparse,json,os,time,statistics,sys
import numpy as np
import chat,ops
from tokenizers import Tokenizer
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
p=argparse.ArgumentParser();p.add_argument('--label',default='run');p.add_argument('--model',default=r'C:\vkops\qwen3-4b');p.add_argument('--steps',type=int,default=24);a=p.parse_args()
gpu=ops.GPU();print('[device]',gpu.info(),flush=True)
t=time.perf_counter();bufs=chat.stream_upload(gpu,a.model,36);eng=chat.Qwen3Engine(gpu,bufs,36,2560,9728,max_ctx=512);print('[load_s]',time.perf_counter()-t,flush=True)
tok=Tokenizer.from_file(os.path.join(a.model,'tokenizer.json'))
prompt=tok.encode('<|im_start|>user\nExplain why the sky is blue in simple terms. Please be concise.<|im_end|>\n<|im_start|>assistant\n').ids
fixed=tok.encode('The sky appears blue because molecules in the atmosphere scatter sunlight. Shorter blue wavelengths scatter more strongly than red light.').ids
meter={'submit_s':0.,'submits':0,'attention_s':0.,'rope_s':0.}
orig_submit=gpu.dev.submit_jobs
orig_attention=chat.attention_np
orig_rope=chat.rope
def submit(jobs):
 t=time.perf_counter();r=orig_submit(jobs);meter['submit_s']+=time.perf_counter()-t;meter['submits']+=1;return r
def attention(*args,**kw):
 t=time.perf_counter();r=orig_attention(*args,**kw);meter['attention_s']+=time.perf_counter()-t;return r
def rope(*args,**kw):
 t=time.perf_counter();r=orig_rope(*args,**kw);meter['rope_s']+=time.perf_counter()-t;return r
gpu.dev.submit_jobs=submit;chat.attention_np=attention;chat.rope=rope
eng.forward(prompt,0);eng.forward([fixed[0]],len(prompt));print('[warmup] complete',flush=True)
results={'label':a.label,'device':gpu.info(),'prompt_tokens':len(prompt),'decode_steps':a.steps};logits={}
for key in meter:meter[key]=0
times=[]
for rep in range(3):
 eng.clear_cache();t=time.perf_counter();r=eng.forward(prompt,0);times.append(time.perf_counter()-t)
logits['prefill']=r;results['prefill_s']=times;results['prefill_median_s']=statistics.median(times)
for key in meter:meter[key]=0
dt=[]
for i in range(a.steps):
 t=time.perf_counter();r=eng.forward([fixed[i%len(fixed)]],len(prompt)+i);dt.append(time.perf_counter()-t)
 if i in (0,7,a.steps-1):logits['decode_'+str(i)]=r
results['decode_s']=dt;results['decode_median_s']=statistics.median(dt);results['decode_tok_s']=len(dt)/sum(dt);results['profile']=meter.copy()
print('[benchmark]',json.dumps(results),flush=True)
eng.clear_cache();r=eng.forward(prompt,0);ids=[]
for i in range(12):
 n=int(np.argmax(r));ids.append(n);r=eng.forward([n],len(prompt)+i)
results['greedy_ids']=ids;results['greedy_text']=tok.decode(ids)
print('[greedy]',results['greedy_text'],flush=True)
np.savez(a.label+'_logits.npz',**logits)
with open(a.label+'.json','w',encoding='utf-8') as f:json.dump(results,f,indent=2,ensure_ascii=False)
print('BENCHMARK COMPLETE',a.label,flush=True)
