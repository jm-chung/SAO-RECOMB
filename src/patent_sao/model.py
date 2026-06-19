from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class SAOTripleDataset(Dataset):
    def __init__(self, X, y, cluster2label, entity2idx, relation2idx):
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self.X, self.y = X, y
        self.cluster2label = cluster2label
        self.entity2idx, self.relation2idx = entity2idx, relation2idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        head, relation, tail = self.X[index]
        return {
            "head_text": self.cluster2label.get(head, head),
            "rel_text": self.cluster2label.get(relation, relation),
            "tail_text": self.cluster2label.get(tail, tail),
            "head_ids": torch.tensor(self.entity2idx[head], dtype=torch.long),
            "rel_ids": torch.tensor(self.relation2idx[relation], dtype=torch.long),
            "tail_ids": torch.tensor(self.entity2idx[tail], dtype=torch.long),
            "label": torch.tensor(self.y[index], dtype=torch.float32),
        }


class SAO_BC(nn.Module):
    def __init__(self, encoder, tokenizer, num_entities: int, num_relations: int, hidden_size: int,
                 max_length: int, dropout: float, alpha_initial: float, embedding_dim: int,
                 conv_out_channels: int, conv_kernel_size: int, conv_height: int, fc_hidden: int,
                 input_dropout: float, feature_dropout: float, hidden_dropout: float):
        super().__init__()
        if (embedding_dim * 2) % conv_height != 0:
            raise ValueError("2 * embedding_dim must be divisible by conv_height")
        self.encoder, self.tokenizer, self.max_length = encoder, tokenizer, max_length
        self.semantic_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.conv_height = conv_height
        conv_width = (embedding_dim * 2) // conv_height
        self.conv = nn.Conv2d(1, conv_out_channels, conv_kernel_size, padding=conv_kernel_size // 2)
        self.batch_norm = nn.BatchNorm2d(conv_out_channels)
        self.fc = nn.Linear(conv_out_channels * conv_height * conv_width, fc_hidden)
        self.fc_out = nn.Linear(fc_hidden, embedding_dim)
        self.input_dropout = nn.Dropout(input_dropout)
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.alpha_logit = nn.Parameter(torch.tensor([alpha_initial], dtype=torch.float32))

    def freeze_semantic_scorer(self):
        for module in (self.encoder, self.classifier):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def unfreeze_semantic_scorer(self):
        for module in (self.encoder, self.classifier):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def forward_semantic(self, heads, relations, tails):
        texts = [f"[S] {h} [A] {r} [O] {t}" for h, r, t in zip(heads, relations, tails)]
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs = {key: value.to(next(self.encoder.parameters()).device) for key, value in inputs.items()}
        pooled = self.encoder(**inputs).last_hidden_state[:, 0, :]
        return self.classifier(self.semantic_dropout(pooled)).squeeze(-1)

    def forward_structural(self, head_ids, relation_ids, tail_ids):
        head = self.entity_embedding(head_ids)
        relation = self.relation_embedding(relation_ids)
        tail = self.entity_embedding(tail_ids)
        vector = self.input_dropout(torch.cat([head, relation], dim=-1))
        vector = vector.view(-1, 1, self.conv_height, vector.size(-1) // self.conv_height)
        vector = self.feature_dropout(F.relu(self.batch_norm(self.conv(vector))))
        vector = vector.flatten(start_dim=1)
        vector = self.hidden_dropout(F.relu(self.fc(vector)))
        projected = self.fc_out(vector)
        return F.cosine_similarity(projected, tail, dim=-1)

    def forward(self, heads, relations, tails, head_ids, relation_ids, tail_ids):
        semantic_logit = self.forward_semantic(heads, relations, tails)
        structural_logit = self.forward_structural(head_ids, relation_ids, tail_ids)
        alpha = torch.sigmoid(self.alpha_logit)
        return (1 - alpha) * semantic_logit + alpha * structural_logit, alpha


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.exp(-bce)
        return (self.alpha * (1 - probabilities).pow(self.gamma) * bce).mean()
