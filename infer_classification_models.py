import tqdm
import glob
import torch
import pickle
import argparse
import numpy as np

from core import networks

class Dataset:
    def __init__(self, path):
        target_data = pickle.load(open(path, 'rb'))

        self.dataset = []

        for data in target_data:
            test_id = data['test_id']
            first_emb = data['first_party']
            second_emb = data['second_party']
            fact_emb = data['facts']

            if len(first_emb.shape) > 1:
                first_emb = first_emb[0]
                second_emb = second_emb[0]
                fact_emb = fact_emb[0]
            
            self.dataset.append([test_id, first_emb, second_emb, fact_emb])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        test_id, first, second, fact = self.dataset[index]
        return test_id, first.astype(np.float32), second.astype(np.float32), fact.astype(np.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", type=str, default='rembert,albert-xxlarge-v2,deberta-v2-xxlarge,bigbird-pegasus-large-bigpatent')
    parser.add_argument("--mlp", type=str, default='residual')
    args = parser.parse_args()

    feature_dict = {
        'rembert': 1152,
        'albert-xxlarge-v2': 4096,
        'deberta-v2-xxlarge': 1536,
        'bigbird-pegasus-large-bigpatent': 1024,
    }
    
    f = open(f'./submissions/230709_classification_models.csv', 'w')
    f.write('ID,first_party_winner\n')
    
    models = []

    for i, model_name in enumerate(args.model_names.split(',')):
        model = networks.CosClassifier(feature_dict[model_name], args.mlp)
        
        model.cuda()
        model.eval()

        model.load_state_dict(torch.load(glob.glob(f'./weights/{model_name}_val_*')[0]))

        models.append(model)
    
    test_datasets = []
    for model_name in args.model_names.split(','):
        test_dataset = Dataset(f'./open/test_{model_name}.pkl')
        test_datasets.append(test_dataset)
    
    for i in tqdm.tqdm(range(len(test_datasets[0]))):
        test_ids = []
        preds = []
        probs = []
        uncertainties = []
        
        for test_dataset, model, name in zip(test_datasets, models, args.model_names.split(',')):
            test_id, first, second, fact = test_dataset[i]

            first = torch.from_numpy(first).cuda().unsqueeze(0)
            second = torch.from_numpy(second).cuda().unsqueeze(0)
            fact = torch.from_numpy(fact).cuda().unsqueeze(0)

            with torch.no_grad():
                prob = torch.softmax(model(first, second, fact)[0], dim=0)
                prob = prob.cpu().detach().numpy()

            uncertainty = np.sum([p * (1. - p) for p in prob])

            test_ids.append(test_id)
            preds.append(np.argmax(prob))
            probs.append(prob)
            uncertainties.append(uncertainty)
        
        prob = np.mean([uncertainties[i]*probs[i] for i in [0, 1, 2, 3]], axis=0)
        pred_index = np.argmax(prob)

        if pred_index == 0:
            first_party_winner = 1
        else:
            first_party_winner = 0

        f.write(f'{test_id},{first_party_winner}\n')

    f.close()
