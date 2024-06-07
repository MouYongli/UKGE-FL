import os
import os.path as osp
import numpy as np
import pandas as pd
import argparse

from ukge.datasets import KGTripleDataset
from ukge.models import DistMult

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

model_map = {
    'distmult': DistMult,
}

def get_t_ranks(model, h, r, ts, device):
    """
    Given some t index, return the ranks for each t
    :param model: the knowledge graph embedding model
    :param h: head entity index
    :param r: relation index
    :param ts: list of tail entity indices
    :param device: torch device (e.g., 'cpu' or 'cuda')
    :return: ranks for each tail entity
    """
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        # Create a tensor for head and relation with the same shape as ts
        head_tensor = torch.tensor([h] * len(ts), dtype=torch.long, device=device)
        rel_tensor = torch.tensor([r] * len(ts), dtype=torch.long, device=device)
        tail_tensor = torch.tensor(ts, dtype=torch.long, device=device)

        # Predict scores for each (h, r, t) triplet
        scores = model(head_tensor, rel_tensor, tail_tensor)
        ranks = torch.ones(len(ts), dtype=torch.int, device=device)  # Initialize rank as all 1

        # Calculate scores for all possible tail entities
        all_tails = torch.arange(model.num_nodes, dtype=torch.long, device=device)
        all_scores = model(head_tensor, rel_tensor, all_tails)
        sorted_indices = torch.argsort(all_scores, descending=True)  # Sort scores in descending order

        # Determine the rank for each given tail entity
        for i, t in enumerate(ts):
            ranks[i] = (sorted_indices == t).nonzero(as_tuple=True)[0].item() + 1  # Find the rank of the tail entity

    return ranks

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

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Create a CSV file to save losses and metrics
    train_log_file = os.path.join(script_dir, 'train_metrics.csv')
    with open(train_log_file, 'w') as file:
        file.write(','.join(['Epoch', 'Step', 'Loss', 'Loss pos', 'Loss neg']) + '\n')

    val_log_file = os.path.join(script_dir, 'val_metrics.csv')
    with open(val_log_file, 'w') as file:
        file.write(','.join(['Epoch', 'Loss', 'Loss pos', 'Loss neg']) + '\n')
    
    train_dataset = KGTripleDataset(dataset=args.dataset, split='train', num_neg_per_positive=args.num_neg_per_positive)
    val_dataset = KGTripleDataset(dataset=args.dataset, split='val')
    test_dataset = KGTripleDataset(dataset=args.dataset, split='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_map[args.model](num_nodes=train_dataset.num_cons(), num_relations=train_dataset.num_rels(), hidden_channels=args.hidden_dim, model_type=args.model_type).to(device)
    
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        model.train()
        loss_total = 0.0
        loss_pos_total = 0.0
        loss_neg_total = 0.0
        for idx, (pos_hrt, pos_score, neg_hn_rt, neg_hr_tn) in enumerate(train_dataloader):
            pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = pos_hrt.long(), pos_score.float(), neg_hn_rt.long(), neg_hr_tn.long()
            pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = pos_hrt.to(device), pos_score.to(device), neg_hn_rt.to(device), neg_hr_tn.to(device) 
            pred_pos_score = model(pos_hrt[:,0], pos_hrt[:,1], pos_hrt[:,2])
            pred_hneg_score = model(neg_hn_rt[:,:,0], neg_hn_rt[:,:,1], neg_hn_rt[:,:,2])
            pred_tneg_score = model(neg_hr_tn[:,:,0], neg_hr_tn[:,:,1], neg_hr_tn[:,:,2])
            loss_pos = criterion_mse(pred_pos_score, pos_score)
            loss_neg = (torch.pow(pred_hneg_score, 2).mean() + torch.pow(pred_tneg_score, 2).mean())/2
            loss = loss_pos + loss_neg
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            loss_pos_total += loss_pos.item()
            loss_neg_total += loss_neg.item()
            
            with open(train_log_file, 'a') as file:
                file.write(f"{epoch + 1},{idx + 1},{loss.item():.4f},{loss_pos.item():.4f},{loss_neg.item():.4f}\n")
            if idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{args.num_epochs}], Step [{idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Loss pos: {loss_pos.item():.4f}, Loss neg: {loss_neg.item():.4f}")
        with open(train_log_file, 'a') as file:
            file.write(f"{epoch + 1},{loss_total/len(train_dataloader):.4f},{loss_pos_total/len(train_dataloader):.4f},{loss_neg_total/len(train_dataloader):.4f}\n")
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {loss_total/len(train_dataloader):.4f}, Loss pos: {loss_pos_total/len(train_dataloader):.4f}, Loss neg: {loss_neg_total/len(train_dataloader):.4f}")
        
        # Validation
        model.eval()
        val_loss_mse_total = 0.0
        val_loss_mae_total = 0.0
        htrw_map = {}
        with torch.no_grad():
            for val_idx, (val_hrt, val_score) in enumerate(val_dataloader):
                val_hrt, val_score = val_hrt.long(), val_score.float()
                val_hrt, val_score = val_hrt.to(device), val_score.to(device)
                val_pred_score = model(val_hrt[:,0], val_hrt[:,1], val_hrt[:,2])
                val_loss_mse = criterion_mse(val_pred_score, val_score)
                val_loss_mse_total+= val_loss_mse.item()
                val_loss_mae = criterion_mae(val_pred_score, val_score)
                val_loss_mae_total+= val_loss_mae.item()

            #     for i in range(len(val_hrt)):
            #         h, r, t = val_hrt[i, 0].item(), val_hrt[i, 1].item(), val_hrt[i, 2].item()
            #         tw_truth = [{'index': t, 'score': val_score[i].item()}]  # 此处应为实际的 ground truth 数据
            #         ranks = get_t_ranks(model, h, r, [tw['index'] for tw in tw_truth], device)
            #         gains = torch.tensor([tw['score'] for tw in tw_truth], device=device)
            #         discounts = torch.log2(ranks + 1)
            #         discounted_gains = gains / discounts
            #         dcg = torch.sum(discounted_gains)  # discounted cumulative gain
            #         max_possible_dcg = torch.sum(gains / torch.log2(torch.arange(len(gains), device=device) + 2))  # when ranks = [1, 2, ...len(truth)]
            #         ndcg = dcg / max_possible_dcg  # normalized discounted cumulative gain
            #         exp_gains = torch.tensor([2 ** tw['score'] - 1 for tw in tw_truth], device=device)
            #         exp_discounted_gains = exp_gains / discounts
            #         exp_dcg = torch.sum(exp_discounted_gains)
            #         exp_max_possible_dcg = torch.sum(exp_gains / torch.log2(torch.arange(len(exp_gains), device=device) + 2))  # when ranks = [1, 2, ...len(truth)]
            #         exp_ndcg = exp_dcg / exp_max_possible_dcg  # normalized discounted cumulative gain
            #         ndcg_total += ndcg.item()
            #         exp_ndcg_total += exp_ndcg.item()

            # avg_ndcg = ndcg_total / len(dataloader)
            # avg_exp_ndcg = exp_ndcg_total / len(dataloader)
            with open(val_log_file, 'a') as file:
                file.write(f"{epoch + 1},{val_loss_mse/len(val_dataloader):.4f},{val_loss_mae/len(val_dataloader):.4f}\n")
            print(f"Validation Loss MSE: {val_loss_mse/len(val_dataloader):.4f}, Loss MAE: {val_loss_mae/len(val_dataloader):.4f}")
        
if __name__ == "__main__":
    main()