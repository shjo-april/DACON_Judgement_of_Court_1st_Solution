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
parser.add_argument("--model", type=str, default="bert-large-uncased")
parser.add_argument("--tag", type=str, default="bert-large-uncased")
parser.add_argument("--file", type=str, default="./open/trainval.json")
args = parser.parse_args()

emb_dict = {}
data_dict = json.load(open(args.file, 'r', encoding='utf-8'))

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model)

device = torch.device('cuda:0')
model = model.to(device)
model.eval()

for domain in ['train', 'validation']:
    emb_dict[domain] = []

    for data in tqdm.tqdm(data_dict[domain]):
        first_party = data['The first party']
        second_party = data['The second party']
        facts = data['facts'].replace('\n', ' ')
        
        embeddings = []

        for input_data in [first_party, second_party, facts]:
            encoded_input = tokenizer([input_data], padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                if 'deberta' in args.tag:
                    model_output = model.deberta(**encoded_input)
                    embedding = cls_pooling(model_output, encoded_input['attention_mask'])[0].cpu().detach().numpy()

                elif 'bigbird' in args.tag:
                    outputs = model.model(**encoded_input)
                    hidden_states = outputs[0]
                    eos_mask = encoded_input['input_ids'].eq(model.config.eos_token_id).to(hidden_states.device)
                    
                    if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                        raise ValueError("All examples must have the same number of <eos> tokens.")

                    embedding = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[0, -1, :]
                    embedding = embedding.cpu().detach().numpy()

                elif 'albert' in args.tag:
                    model_output = model.albert(**encoded_input)[1]
                    embedding = model_output[0].cpu().detach().numpy()

                elif 'rembert' in args.tag:
                    model_output = model.rembert(**encoded_input)[1]
                    embedding = model_output[0].cpu().detach().numpy()

                else:
                    raise ValueError(f'Check {args.tag}')
        
            embeddings.append(embedding)

        emb_dict[domain].append(
            {
                'first_party': embeddings[0],
                'first_party_name': first_party,

                'second_party': embeddings[1],
                'second_party_name': second_party,

                'facts': embeddings[2],
                'output': data['output']
            }
        )
    
    pickle.dump(emb_dict, open(args.file.replace('.json', f'_{args.tag}.pkl'), 'wb'))