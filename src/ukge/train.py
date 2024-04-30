import os
import os.path as osp
import numpy as np

import argparse

import torch
from torch.utils.data import DataLoader

from ukge.datasets import KGTripleDataset
from ukge.models import TransE, DistMult
model_map = {
    'transe': TransE,
    'distmult': DistMult,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['transe', 'distmult'], type=str.lower, required=True)
    parser.add_argument('--dataset', choices=['cn15k', 'nl27k', 'ppi5k'], type=str.lower, required=True)
    parser.add_argument('--num_neg_per_positive', default=10, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    
    parser.add_argument('--lr', default=0.01, type=float)
    args = parser.parse_args()
    
    train_dataset = KGTripleDataset(dataset=args.dataset, split='train', num_neg_per_positive=args.num_neg_per_positive)
    val_dataset = KGTripleDataset(dataset=args.dataset, split='val', num_neg_per_positive=args.num_neg_per_positive)
    test_dataset = KGTripleDataset(dataset=args.dataset, split='test', num_neg_per_positive=args.num_neg_per_positive)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_map[args.model](num_nodes=train_dataset.num_cons(), num_relations=train_dataset.num_rels(), hidden_channels=args.hidden_dim).to(device)
