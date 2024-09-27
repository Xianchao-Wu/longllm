import json
import sys
import os

from transformers import AutoTokenizer

afn="test.in.jsonl"

kkk = 1024 # TODO 1024 or 1000

def classify(toknum):
    if toknum >= 10 * kkk and toknum <= 100 * kkk:
        return 1 # set 1
    elif toknum > 100 * kkk and toknum <= 200 * kkk:
        return 2 # set 2
    elif toknum > 200 * kkk and toknum <= 500 * kkk:
        return 3 # set 3
    elif toknum > 500 * kkk and toknum <= 1000 * kkk:
        return 4 # set 4
    elif toknum > 1000 * kkk:
        return 5 # set 5
    else:
        return 0 # out of scope

def load_tokenizer():
    #tok_name = 'meta-llama/Llama-2-7b-hf'
    #tok_name = 'Undi95/Meta-Llama-3-8B-hf'
    tok_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(tok_name, cache_dir=".")
    #print(tokenizer)
    return tokenizer

tok = load_tokenizer()

#with open(afn) as br:
idx = 0
for aline in sys.stdin: #br.readlines(): 
    aline = aline.strip()
    objs = json.loads(aline)
    #print(objs.keys())

    if 'text' in objs:
        atext = objs['text']
        toklen = len(tok.encode(atext))
        setid = classify(toklen)
        print(idx, toklen, setid)
    else:
        print(idx, '0', 0)
    idx += 1
