import os
import os.path as osp
import numpy as np

import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from ukge.datasets import KGTripleDataset
from ukge.models import DistMult

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

model_map = {
    'distmult': DistMult,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str.lower, default='distmult', choices=['distmult'])
    parser.add_argument('--model_type', type=str.lower, default='logi', choices=['logi', 'rect'])
    parser.add_argument('--loss_type', type=str.lower, default='logi', choices=['', 'rect'])
    parser.add_argument('--dataset', type=str.lower, default='cn15k', choices=['cn15k', 'nl27k', 'ppi5k'])
    parser.add_argument('--num_neg_per_positive', default=10, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    args = parser.parse_args()
    
    train_dataset = KGTripleDataset(dataset=args.dataset, split='train', num_neg_per_positive=args.num_neg_per_positive)
    val_dataset = KGTripleDataset(dataset=args.dataset, split='val')
    test_dataset = KGTripleDataset(dataset=args.dataset, split='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_map[args.model](num_nodes=train_dataset.num_cons(), num_relations=train_dataset.num_rels(), hidden_channels=args.hidden_dim, model_type=args.model_type).to(device)
    
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = next(iter(train_dataloader))
    
    pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = pos_hrt.long(), pos_score.float(), neg_hn_rt.long(), neg_hr_tn.long()
    pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = pos_hrt.to(device), pos_score.to(device), neg_hn_rt.to(device), neg_hr_tn.to(device)
    pred_pos_score = model(pos_hrt[:,0], pos_hrt[:,1], pos_hrt[:,2])
    pred_hneg_score = model(neg_hn_rt[:,:,0], neg_hn_rt[:,:,1], neg_hn_rt[:,:,2])
    # pred_tneg_score = model(neg_hr_tn[:,0], neg_hr_tn[:,1], neg_hr_tn[:,2])

    print('pred_pos_score', pred_pos_score)
    print('pred_hneg_score', pred_hneg_score)
    # print('pred_tneg_score', pred_tneg_score)

    

if __name__ == '__main__':
    main()