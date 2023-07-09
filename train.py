import os
import tqdm
import json
import torch
import argparse
import pickle
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from core import networks

class Dataset:
    def __init__(self, path, domain):
        target_data = pickle.load(open(path, 'rb'))[domain]

        self.pos_data = []
        self.neg_data = []
        
        for data in target_data:
            first_emb = data['first_party']
            second_emb = data['second_party']
            fact_emb = data['facts']

            if len(first_emb.shape) > 1:
                first_emb = first_emb[0]
                second_emb = second_emb[0]
                fact_emb = fact_emb[0]
            
            if data['output'] == 'Victory':
                self.pos_data.append([first_emb, second_emb, fact_emb, 0])
                self.neg_data.append([second_emb, first_emb, fact_emb, 1])
            else:
                self.pos_data.append([second_emb, first_emb, fact_emb, 0])
                self.neg_data.append([first_emb, second_emb, fact_emb, 1])

        self.dataset = self.pos_data + self.neg_data

    def balance_sampling(self):
        np.random.shuffle(self.pos_data)
        np.random.shuffle(self.neg_data)

        min_length = min(len(self.pos_data), len(self.neg_data))
        self.dataset = self.pos_data[:min_length] + self.neg_data[:min_length]

        return self.pos_data[min_length:], self.neg_data[min_length:]
    
    def add_data(self, pos_data, neg_data):
        print('[Before] {}, {}'.format(len(self.pos_data), len(self.neg_data)))

        self.pos_data += pos_data
        self.neg_data += neg_data

        print('[After] {}, {}'.format(len(self.pos_data), len(self.neg_data)))
        print()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        first, second, fact, label = self.dataset[index]
        return first.astype(np.float32), second.astype(np.float32), fact.astype(np.float32), label

class Evaluator_For_Single_Label_Classification:
    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        
        self.clear()

    def add(self, data):
        pred, gt_index = data

        pred_index = np.argmax(pred)

        correct = int(pred_index == gt_index)
        self.meter_dict['correct'][gt_index] += correct
        self.meter_dict['count'][gt_index] += 1

    def get(self, detail=False, clear=True):
        accuracies = self.meter_dict['correct'] / self.meter_dict['count'] * 100
        accuracy = np.mean(accuracies)

        if clear:
            self.clear()
        
        if detail:
            return accuracy, accuracies
        else:
            return accuracy
    
    def clear(self):
        self.meter_dict = {
            'correct': np.zeros(self.num_classes, dtype=np.float32),
            'count': np.zeros(self.num_classes, dtype=np.float32)
        }

class _Scheduler:
    def __init__(self, optimizer, max_iterations):
        self.optimizer = optimizer

        self.iteration = 1
        self.max_iterations = max_iterations

    def state_dict(self) -> dict:
        return {
            'iteration': self.iteration
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.iteration = state_dict['iteration']

    def get_learning_rate(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def set_learning_rate(self, lrs):
        if isinstance(lrs, float):
            lrs = [lrs for _ in range(len(self.optimizer.param_groups))]
        
        for lr, group in zip(lrs, self.optimizer.param_groups):
            group['lr'] = lr

class PolyLR(_Scheduler):
    def __init__(self, optimizer, max_iterations, power=0.9):
        super().__init__(optimizer, max_iterations)
        
        self.power = power
        self.init_lrs = self.get_learning_rate()
    
    def step(self):
        if self.iteration < self.max_iterations:
            lr_mult = (1 - self.iteration / self.max_iterations) ** self.power

            lrs = [lr * lr_mult for lr in self.init_lrs]
            self.set_learning_rate(lrs)

            self.iteration += 1

class SimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cos_dists, targets):
        pos_indices = torch.sum(targets, dim=1) > 0
        neg_indices = torch.sum(targets, dim=1) == 0

        if torch.sum(pos_indices).item() > 0:
            pos_dists = cos_dists[pos_indices]
            pos_loss = -torch.log(pos_dists + 1e-5).mean()
        else:
            pos_loss = torch.zeros(1, dtype=torch.float32).cuda()

        if torch.sum(neg_indices).item() > 0:
            neg_dists = cos_dists[neg_indices]
            neg_loss = -torch.log(1. - neg_dists + 1e-5).mean()
        else:
            neg_loss = torch.zeros(1, dtype=torch.float32).cuda()

        return pos_loss, neg_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="albert-xxlarge-v2")
    parser.add_argument("--mlp", type=str, default="residual")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)

    args = parser.parse_args()

    model_name = args.model
    mlp_type = args.mlp

    np.random.seed(0)

    train_dataset = Dataset(f'./dataset/trainval_{model_name}.pkl', 'train')
    valid_dataset = Dataset(f'./dataset/trainval_{model_name}.pkl', 'validation')
    
    feature_dict = {
        'albert-xxlarge-v2': 4096,
        'deberta-v2-xxlarge': 1536,
        'bigbird-pegasus-large-bigpatent': 1024,
        'rembert': 1152,
    }

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=16, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False)
    
    model = networks.CosClassifier(feature_dict[model_name], mlp_type)
    model.cuda()
    model.train()

    iteration = 0
    log_iterations = 10
    val_iterations = 50
    max_iterations = 2000

    max_epochs = max_iterations // len(train_loader)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = PolyLR(optimizer, max_iterations)
    
    loss_fn = nn.CrossEntropyLoss().cuda()

    train_losses = []
    pbar = tqdm.tqdm(total=max_iterations)

    best_val_th, best_val_acc, best_val_name = 0, 0, None
    evaluator = Evaluator_For_Single_Label_Classification(['first', 'second'])

    pth_dir = f'./weights/{model_name}/'
    if not os.path.isdir(pth_dir): os.makedirs(pth_dir)

    for epoch in range(1, max_epochs+1):
        train_dataset.balance_sampling()

        for first_embs, second_embs, fact_embs, labels in train_loader:
            first_embs = first_embs.cuda()
            second_embs = second_embs.cuda()
            fact_embs = fact_embs.cuda()
            labels = labels.cuda()
            
            logits = model(first_embs, second_embs, fact_embs)

            loss = loss_fn(logits, labels)

            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            iteration += 1

            if iteration % log_iterations == 0:
                train_loss = np.mean(train_losses)
                print(f'Iter: {iteration:05d}, Loss: {train_loss:.04f}')
                train_losses = []

            if iteration % val_iterations == 0:
                model.eval()

                with torch.no_grad():
                    for first_embs, second_embs, fact_embs, labels in valid_loader:
                        first_embs = first_embs.cuda()
                        second_embs = second_embs.cuda()
                        fact_embs = fact_embs.cuda()
                        labels = labels.cuda()

                        probs = model(first_embs, second_embs, fact_embs)
                        probs = probs.cpu().detach().numpy()
                        
                        for prob, label in zip(probs, labels):
                            evaluator.add([prob, label.item()])

                    val_acc = evaluator.get(clear=True)
                    
                if best_val_acc < val_acc:
                    best_val_th = 0.
                    best_val_acc = val_acc

                    if best_val_name is not None:
                        os.remove(pth_dir + best_val_name)

                    best_val_name = f'{model_name}_val_{args.mlp}_I={iteration:05d}_T={best_val_th:.02f}_Acc={best_val_acc:.02f}.pth'
                    torch.save(model.state_dict(), pth_dir + best_val_name)

                    print(f'Iter: {iteration:05d}, [Validation, Best] T: {best_val_th:.02f}, Acc: {best_val_acc:.02f}, Length: {len(valid_dataset)}')
                
                print(f'Iter: {iteration:05d}, [Validation] T: {0.:.02f}, Acc: {val_acc:.02f}, Length: {len(valid_dataset)}')
                
                model.train()
    
    pbar.close()