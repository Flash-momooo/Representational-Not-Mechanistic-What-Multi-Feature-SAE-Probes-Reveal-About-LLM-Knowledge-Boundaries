"""NN31: independent-Qwen judge audit of exact-match labels (100 stratified rows)."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT=Path(__file__).resolve().parents[1]
def question(prompt):
    m=re.search(r'Question:\s*(.*?)\nAnswer:',prompt,re.S)
    return m.group(1).strip() if m else ''
def main():
    rows=[json.loads(x) for x in (ROOT/'data/nn28b_gemma_it_fresh_2wiki_trajectory.jsonl').read_text(encoding='utf8').splitlines()]
    correct=[r for r in rows if r['model_correct']]; near=[r for r in rows if not r['model_correct'] and r.get('token_f1',0)>0]; wrong=[r for r in rows if not r['model_correct'] and r.get('token_f1',0)==0]
    rng=np.random.default_rng(20260805); sample=[]
    for group,n in ((correct,50),(near,25),(wrong,25)):
        sample += [group[i] for i in rng.choice(len(group),size=n,replace=False)]
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen2.5-7B-Instruct'),local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(str(ROOT/'models/Qwen2.5-7B-Instruct'),local_files_only=True,device_map='cuda',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    judged=[]
    for i,r in enumerate(sample):
        text=("Judge answer correctness. Treat equivalent names, date formats, and unambiguous aliases as correct. "
              "Answer exactly Yes or No.\nQuestion: "+question(r['prompt'])+"\nReference answer: "+r['gold_answer']+"\nCandidate answer: "+r['model_answer']+"\nCorrect?")
        prompt=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
        x=tokenizer(prompt,return_tensors='pt').to(model.device)
        with torch.no_grad(): y=model.generate(**x,do_sample=False,max_new_tokens=3,pad_token_id=tokenizer.eos_token_id)
        ans=tokenizer.decode(y[0,x['input_ids'].shape[1]:],skip_special_tokens=True).strip().lower()
        judge=ans.startswith('yes'); judged.append({'question_id':r['question_id'],'exact_label':bool(r['model_correct']),'token_f1':r['token_f1'],'stratum':'correct' if r['model_correct'] else ('near_error' if r['token_f1']>0 else 'wrong_error'),'judge_correct':judge,'judge_text':ans,'candidate':r['model_answer'],'gold':r['gold_answer']})
        if (i+1)%20==0: print('judged',i+1,flush=True)
    agree=sum(x['exact_label']==x['judge_correct'] for x in judged)/len(judged)
    summary={'n':len(judged),'agreement':agree,'exact_false_errors':sum((not x['exact_label']) and x['judge_correct'] for x in judged),'exact_false_correct':sum(x['exact_label'] and (not x['judge_correct']) for x in judged)}
    out=ROOT/'outputs/poc_nn31_qwen_label_audit.json'; out.write_text(json.dumps({'experiment':'NN31 independent Qwen label audit','sampling':'50 exact-correct, 25 nonexact positive-F1, 25 zero-F1','summary':summary,'rows':judged},indent=2,ensure_ascii=False),encoding='utf8'); print('saved',out,summary)
if __name__=='__main__': main()
