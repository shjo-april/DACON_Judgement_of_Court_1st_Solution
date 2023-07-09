import json
import torch
import pickle

import numpy as np

from torch.nn import functional as F

K = 10
model_name = 'albert-xxlarge-v2'

victory_embs = []
victory_list = []

defeat_embs = []
defeat_list = []

for data in pickle.load(open(f'./open/train_llm_{model_name}.pkl', 'rb')):
    if data['output'] == 'Victory':
        victory_embs.append(data['embedding'])
        victory_list.append(data)
    else:
        defeat_embs.append(data['embedding'])
        defeat_list.append(data)

victory_embs = torch.from_numpy(np.asarray(victory_embs, dtype=np.float32))
defeat_embs = torch.from_numpy(np.asarray(defeat_embs, dtype=np.float32))

victory_embs = F.normalize(victory_embs, dim=1)
defeat_embs = F.normalize(defeat_embs, dim=1)

qa_list = []

for data in pickle.load(open(f'./open/test_llm_{model_name}.pkl', 'rb')):
    emb = torch.from_numpy(np.asarray(data['embedding'], dtype=np.float32))[None, :]
    emb = F.normalize(emb, dim=1)

    victory_sims = F.cosine_similarity(victory_embs.unsqueeze(1), emb.unsqueeze(0), dim=2)[:, 0]
    defeat_sims = F.cosine_similarity(defeat_embs.unsqueeze(1), emb.unsqueeze(0), dim=2)[:, 0]

    victory_indices = np.argsort(victory_sims.numpy())[::-1][:K//2]
    defeat_indices = np.argsort(defeat_sims.numpy())[::-1][:K//2]

    data['victory_examples'] = []
    data['defeat_examples'] = []

    for i in victory_indices:
        victory_data = victory_list[i]
        try:
            del victory_data['embedding']
        except KeyError:
            pass
        data['victory_examples'].append(victory_data)

    for i in defeat_indices:
        defeat_data = defeat_list[i]
        try:
            del defeat_data['embedding']
        except KeyError:
            pass
        data['defeat_examples'].append(defeat_data)

    del data['embedding']

    qa_list.append(data)

json.dump(qa_list, open(f'./llm/dacon_submissions/test_qa_{model_name}.json', 'w', encoding='utf-8'), indent='\t', ensure_ascii=False)