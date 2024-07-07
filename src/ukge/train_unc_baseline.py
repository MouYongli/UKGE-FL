import os
import os.path as osp
import numpy as np
import pandas as pd
import argparse

from ukge.datasets import KGTripleDataset
from ukge.models import TransE, DistMult, ComplEx, RotatE
from ukge.losses import compute_det_transe_distmult_loss, compute_det_complex_loss, compute_det_rotate_loss
from ukge.metrics import Evaluator

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

model_map = {
    'transe': TransE,
    'distmult': DistMult,
    'complex': ComplEx,
    'rotate': RotatE
}

loss_map = {
    'transe': compute_det_transe_distmult_loss,
    'distmult': compute_det_transe_distmult_loss,
    'complex': compute_det_complex_loss,
    'rotate': compute_det_rotate_loss
}

here = osp.dirname(osp.abspath(__file__))
work_dir = osp.join(here, '../../', 'results')
if not osp.exists(work_dir):
    os.makedirs(work_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str.lower, default='distmult', choices=['transe','distmult','complex','rotate'])
    parser.add_argument('--dataset', type=str.lower, default='cn15k', choices=['cn15k', 'nl27k', 'ppi5k'])
    parser.add_argument('--num_neg_per_positive', default=10, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0.005, type=float)
    args = parser.parse_args()

    exp_dir = osp.join(work_dir, f'det_{args.dataset}_{args.model}', f'lr_{args.lr}_hidden_dim_{args.hidden_dim}_threshold_{args.threshold}')
    if not osp.exists(exp_dir):
        os.makedirs(exp_dir)

    # Create a CSV file to save losses and metrics
    train_log_file = os.path.join(exp_dir, 'train_metrics.csv')
    if os.path.exists(train_log_file):
        os.remove(train_log_file)
    with open(train_log_file, 'w') as file:
        file.write(','.join(['Epoch', 'Step', 'Loss']) + '\n')

    val_log_file = os.path.join(exp_dir, 'val_metrics.csv')
    if os.path.exists(val_log_file):
        os.remove(val_log_file)
    with open(val_log_file, 'w') as file:
        file.write(','.join(['Epoch', 'nDCG_lin', 'nDCG_exp']) + '\n')

    test_log_file = os.path.join(exp_dir, 'test_metrics.csv')
    if os.path.exists(test_log_file):
        os.remove(test_log_file)
    with open(test_log_file, 'w') as file:
        file.write(','.join(['Epoch', 'nDCG_lin', 'nDCG_exp']) + '\n')
    
    train_dataset = KGTripleDataset(dataset=args.dataset, split='train', num_neg_per_positive=args.num_neg_per_positive, deterministic=True, threshold=args.threshold)
    val_dataset = KGTripleDataset(dataset=args.dataset, split='val')
    test_dataset = KGTripleDataset(dataset=args.dataset, split='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_map[args.model](num_nodes=train_dataset.num_cons(), num_relations=train_dataset.num_rels(), hidden_channels=args.hidden_dim).to(device)
    criterion = loss_map[args.model]

    val_evaluator = Evaluator(val_dataloader, model, batch_size=args.batch_size, device=device)
    test_evaluator = Evaluator(test_dataloader, model, batch_size=args.batch_size, device=device)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=args.weight_decay)

    best_ndcg = 0.0
    best_ndcg_epoch = 0

    for epoch in range(args.num_epochs):
        model.train()
        loss_total = 0.0
        for idx, (pos_hrt, pos_score, neg_hn_rt, neg_hr_tn) in enumerate(train_dataloader):
            pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = pos_hrt.long(), pos_score.float(), neg_hn_rt.long(), neg_hr_tn.long()
            pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = pos_hrt.to(device), pos_score.to(device), neg_hn_rt.to(device), neg_hr_tn.to(device) 
            neg_hn_rt = neg_hn_rt.view(-1, neg_hn_rt.size(-1))
            neg_hr_tn = neg_hr_tn.view(-1, neg_hr_tn.size(-1))
            optimizer.zero_grad()
            verbose = True if  idx == 100 else False
            pred_pos_score = model(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
            pred_neg_h_score = model(neg_hn_rt[:, 0], neg_hn_rt[:, 1], neg_hn_rt[:, 2])
            pred_neg_t_score = model(neg_hr_tn[:, 0], neg_hr_tn[:, 1], neg_hr_tn[:, 2])
            if args.model == 'transe' or args.model == 'distmult':
                loss_hn = criterion(pred_pos_score, pred_neg_h_score, args.num_neg_per_positive, verbose=verbose)
                loss_tn = criterion(pred_pos_score, pred_neg_t_score, args.num_neg_per_positive, verbose=verbose)
            elif args.model == 'complex' or args.model == 'rotate':
                loss_hn = criterion(pred_pos_score, pred_neg_h_score, verbose=verbose)
                loss_tn = criterion(pred_pos_score, pred_neg_t_score, verbose=verbose)
            else:
                raise ValueError(f"Model {args.model} not supported")
            loss = loss_hn + loss_tn
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            with open(train_log_file, 'a') as file:
                file.write(f"{epoch + 1},{idx + 1},{loss.item():.4f}\n")
            if idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{args.num_epochs}], Step [{idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}    ")
        with open(train_log_file, 'a') as file:
            file.write(f"{epoch + 1}, -1, {loss_total/len(train_dataloader):.4f}\n")
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {loss_total/len(train_dataloader):.4f}")
        
        model.eval()
        val_evaluator.update_hr_scores_map()
        mean_ndcg = val_evaluator.get_mean_ndcg()

        pos_hrt, score = next(iter(val_dataloader))
        pos_hrt = pos_hrt.long()
        pos_hrt = pos_hrt.to(device)
        pred_score = model(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
        print(f"Pred Score: {pred_score[0:10]}", f"True Score: {score[0:10]}")
        with open(val_log_file, 'a') as file:
            file.write(f"{epoch + 1},{mean_ndcg[0]:.4f},{mean_ndcg[1]:.4f}\n")
        print(f"Mean nDCG: {mean_ndcg[0]:.4f}, Exponential Mean nDCG: {mean_ndcg[1]:.4f}")

        if mean_ndcg[0] > best_ndcg:
            best_ndcg = mean_ndcg[0]
            best_ndcg_epoch = epoch + 1
            torch.save({
                'state_dict': model.state_dict(),
                'best_ndcg': best_ndcg,
                'best_ndcg_epoch': best_ndcg_epoch
                }, osp.join(exp_dir, 'best_model.pth'))
        
        model.eval()
        test_evaluator.update_hr_scores_map()
        mean_ndcg = test_evaluator.get_mean_ndcg()
        with open(test_log_file, 'a') as file:
            file.write(f"{epoch + 1},{mean_ndcg[0]:.4f},{mean_ndcg[1]:.4f}\n")
        print(f"Mean nDCG: {mean_ndcg[0]:.4f}, Exponential Mean nDCG: {mean_ndcg[1]:.4f}")
        
if __name__ == "__main__":
    main()