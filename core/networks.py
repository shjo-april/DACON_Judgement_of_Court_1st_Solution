import torch
from torch import nn
from torch.nn import functional as F

class Residual_MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()        

        self.fc1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.BatchNorm1d(out_features)
        self.act1 = nn.ReLU()
        self.do1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.BatchNorm1d(out_features)
        self.act2 = nn.ReLU()
        self.do2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(out_features, in_features)
        self.norm3 = nn.BatchNorm1d(in_features)
        self.act3 = nn.ReLU()
        self.do3 = nn.Dropout(0.5)

    def forward(self, x):
        x_ = self.fc1(x)
        x_ = self.norm1(x_)
        x_ = self.act1(x_)
        x_ = self.do1(x_)

        x_ = self.fc2(x_)
        x_ = self.norm2(x_)
        x_ = self.act2(x_)
        x_ = self.do2(x_)

        x_ = self.fc3(x_)
        x_ = self.norm3(x_)
        x_ = self.act3(x + x_)
        return self.do3(x_)

class CosClassifier(nn.Module):
    def __init__(self, num_features, mlp_type):
        super().__init__()

        if mlp_type == 'residual':
            self.mlp_for_person = nn.Sequential(
                Residual_MLP(num_features, num_features // 4),
                nn.Linear(num_features, num_features // 4)
            )
            self.mlp_for_facts = nn.Sequential(
                Residual_MLP(num_features, num_features // 4),
                nn.Linear(num_features, num_features // 4)
            )

        self.scale = nn.Parameter(torch.Tensor(1)) 
    
    def forward(self, first_embs, second_embs, fact_embs):
        first_embs = self.mlp_for_person(first_embs)
        second_embs = self.mlp_for_person(second_embs)
        fact_embs = self.mlp_for_facts(fact_embs)

        first_logits = self.scale * F.cosine_similarity(
            F.normalize(first_embs, dim=1),
            F.normalize(fact_embs, dim=1),
            dim=1
        )
        second_logits = self.scale * F.cosine_similarity(
            F.normalize(second_embs, dim=1),
            F.normalize(fact_embs, dim=1),
            dim=1
        )

        logits = torch.cat([first_logits.unsqueeze(1), second_logits.unsqueeze(1)], dim=1)
        
        return logits