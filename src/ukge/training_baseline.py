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

    # Create a CSV file to save losses and metrics
    train_log_file = '/home/mou/Projects/KGE/UKGE-FL/src/ukge/train_metrics.csv'
    with open(train_log_file, 'w') as file:
        file.write(','.join(['Epoch', 'Step', 'Loss', 'Loss pos', 'Loss neg']) + '\n')

    val_log_file = '/home/mou/Projects/KGE/UKGE-FL/src/ukge/val_metrics.csv'
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
            file.write(f"{epoch + 1},,{loss_total/len(train_dataloader):.4f},{loss_pos_total/len(train_dataloader):.4f},{loss_neg_total/len(train_dataloader):.4f}\n")
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {loss_total/len(train_dataloader):.4f}, Loss pos: {loss_pos_total/len(train_dataloader):.4f}, Loss neg: {loss_neg_total/len(train_dataloader):.4f}")
        
        # Validation
        model.eval()
        val_loss_mse_total = 0.0
        val_loss_mae_total = 0.0
        with torch.no_grad():
            for val_idx, (val_hrt, val_score) in enumerate(val_dataloader):
                val_hrt, val_score = val_hrt.long(), val_score.float()
                val_hrt, val_score = val_hrt.to(device), val_score.to(device)
                val_pred_score = model(val_hrt[:,0], val_hrt[:,1], val_hrt[:,2])
                val_loss_mse = criterion_mse(val_pred_score, val_score)
                val_loss_mse_total+= val_loss_mse.item()
                val_loss_mae = criterion_mae(val_pred_score, val_score)
                val_loss_mae_total+= val_loss_mae.item()
            with open(val_log_file, 'a') as file:
                file.write(f"{epoch + 1},{val_loss_mse/len(val_dataloader):.4f},{val_loss_mae/len(val_dataloader):.4f}\n")
            print(f"Validation Loss MSE: {val_loss_mse/len(val_dataloader):.4f}, Loss MAE: {val_loss_mae/len(val_dataloader):.4f}")
        
if __name__ == "__main__":
    main()