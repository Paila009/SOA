"""Qwen Transformers inference with actual token signals and block activations."""
import os
import threading
from pathlib import Path
from jobs import Cancelled

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT/'models-local/qwen-research'
SPEC = {'name':'Qwen 1.5B · experimental internal probe', 'parameters':'1.54B',
        'license':'Apache-2.0', 'quantization':'BF16', 'port':0}


class ResearchRuntime:
    def __init__(self):
        self.model = self.tokenizer = None
        self.lock = threading.Lock()
        self.loading = False
        self.error = None
        self.spec = SPEC

    def status(self):
        return {'installed':(WEIGHTS/'model.safetensors').exists(), 'ready':self.model is not None,
                'loading':self.loading, 'error':self.error, 'engine':'Transformers + activation hooks',
                'device':'CPU','parameters':'1.54B','quantization':'BF16','endpoint':'http://127.0.0.1:8766',
                'model':SPEC['name']}

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.loading = True
        try:
            torch.set_num_threads(min(4,os.cpu_count() or 2))
            self.tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS),local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(str(WEIGHTS),local_files_only=True,
                torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval()
        except Exception as exc:
            self.error = str(exc)
            raise
        finally:
            self.loading = False

    def chat(self, question, context, max_tokens=240, comparison=False, history=None,
             on_token=None, job=None, temperature=.15):
        from probe_adapter import ProbeAdapter
        aligned = ROOT/'outputs/probes-aligned'
        adapter = ProbeAdapter(aligned if (aligned/'logistic_probe.joblib').exists() else ROOT/'outputs/probes')
        messages=[{'role':'system','content':'Answer only from the numbered evidence. Cite every factual sentence. If evidence is missing, say so. Use at most three concise sentences.'}]
        messages.extend({'role':row['role'],'content':str(row.get('content',''))[:1000]}
                        for row in (history or [])[-4:] if row.get('role') in ('user','assistant'))
        messages.append({'role':'user','content':f'Sources:\n{context}\n\nQuestion: {question}'})
        return self.generate(messages,max_tokens,temperature,on_token,job,adapter)

    def chat_general(self,question,history=None,max_tokens=120,on_token=None,job=None):
        return self.generate([{'role':'user','content':question}],max_tokens,.15,on_token,job,None)

    def generate(self,messages,max_tokens,temperature,on_token,job,adapter):
        import torch
        while not self.lock.acquire(timeout=.1):
            if job:
                job.check()
        hooks=[]
        try:
            if job:
                job.check()
                job.publish('progress',stage='loading-research-model')
            self.load()
            prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            ids=self.tokenizer(prompt,return_tensors='pt')['input_ids']
            if ids.shape[1] > 3500:
                raise ValueError('Research context exceeds 3500 tokens; use shorter evidence.')
            hidden={}
            def capture(index):
                def hook(_module,_inputs,output):
                    tensor=output[0] if isinstance(output,tuple) else output
                    hidden[index]=tensor[0,-1].detach().float().cpu().numpy().copy()
                return hook
            for index in (0,7,14,21,27):
                hooks.append(self.model.model.layers[index].register_forward_hook(capture(index)))
            generated,entropy,probabilities=[],[],[]
            cache=None
            displayed=''
            rng=torch.Generator().manual_seed(42)
            eos=self.model.generation_config.eos_token_id
            eos=set(eos if isinstance(eos,list) else [eos])
            with torch.inference_mode():
                for _ in range(max_tokens):
                    if job:
                        job.check()
                    output=self.model(input_ids=ids,past_key_values=cache,use_cache=True)
                    logits=output.logits[0,-1].float()
                    probs=logits.softmax(-1)
                    entropy.append(float(-(probs*probs.clamp_min(1e-9).log()).sum()))
                    if temperature <= 0:
                        token=logits.argmax().item()
                    else:
                        distribution=(logits/temperature).softmax(-1)
                        sorted_probs,indices=distribution.sort(descending=True)
                        remove=sorted_probs.cumsum(-1)-sorted_probs > .9
                        sorted_probs[remove]=0
                        token=indices[torch.multinomial(sorted_probs,1,generator=rng)].item()
                    probabilities.append(float(probs[token]))
                    generated.append(token)
                    cache=output.past_key_values
                    ids=torch.tensor([[token]])
                    text=self.tokenizer.decode(generated,skip_special_tokens=True)
                    delta=text[len(displayed):]
                    displayed=text
                    if delta and on_token and on_token(delta) is False:
                        break
                    if token in eos:
                        break
                # Process the final selected token so its block activations align
                # with teacher-forced last-answer-token training features.
                if generated:
                    self.model(input_ids=ids,past_key_values=cache,use_cache=False)
            usage={'completion_tokens':len(generated),'total_tokens':len(generated)+len(self.tokenizer.encode(prompt))}
            if adapter and entropy:
                record={'logit_entropy':entropy,'token_probs':probabilities,'hidden_states':hidden,
                        'metadata':{'model':'qwen2.5-1.5b','extraction':'generation-last-block-output-v1'}}
                usage['research_probe']=adapter.predict(record)
            return displayed.strip(),usage
        finally:
            for hook in hooks:
                hook.remove()
            self.lock.release()


RESEARCH_RUNTIME=ResearchRuntime()
