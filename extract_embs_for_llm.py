import os
os.environ['HF_HOME'] = './cache/huggingface'
os.environ['TORCH_HOME'] = './cache/torch'

import json
import tqdm
import torch
import pickle
import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="albert-xxlarge-v2")
parser.add_argument("--tag", type=str, default="albert-xxlarge-v2")
parser.add_argument("--file", type=str, default="./open/test.json")
args = parser.parse_args()

emb_dict = {}
total_data = json.load(open(args.file, 'r', encoding='utf-8'))

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model)

device = torch.device('cuda:0')
model = model.to(device)
model.eval()

data_with_embs = []

for data in tqdm.tqdm(total_data):
    facts = data['facts'].replace('\n', ' ')
    embeddings = []

    for input_data in [facts]:
        encoded_input = tokenizer([input_data], padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            if 'albert' in args.tag:
                model_output = model.albert(**encoded_input)[1]
                embedding = model_output[0].cpu().detach().numpy()
            else:
                raise ValueError(f'Check {args.tag}')
    
        embeddings.append(embedding)
    
    data['embedding'] = embeddings[0]
    data_with_embs.append(data)

pickle.dump(data_with_embs, open(args.file.replace('.json', f'_llm_{args.tag}.pkl'), 'wb'))