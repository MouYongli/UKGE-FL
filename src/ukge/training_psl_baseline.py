import os
import os.path as osp
import numpy as np
import pandas as pd
import argparse

from ukge.datasets import KGTripleDataset, KGPSLTripleDataset
from ukge.models import DistMult
from ukge.metrics import Evaluator
from ukge.losses import compute_psl_loss

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

model_map = {
    'distmult': DistMult,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str.lower, default='distmult', choices=['distmult'])
    parser.add_argument('--model_type', type=str.lower, default='rect', choices=['logi', 'rect'])
    parser.add_argument('--dataset', type=str.lower, default='cn15k', choices=['cn15k', 'nl27k', 'ppi5k'])
    parser.add_argument('--num_neg_per_positive', default=10, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    args = parser.parse_args()

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Create a CSV file to save losses and metrics
    train_log_file = os.path.join(script_dir, f'{args.dataset}_train_psl_metrics.csv')
    if os.path.exists(train_log_file):
        os.remove(train_log_file)
    with open(train_log_file, 'w') as file:
        file.write(','.join(['Epoch', 'Step', 'Loss', 'Loss pos', 'Loss neg']) + '\n')
    
    val_log_file = os.path.join(script_dir, f'{args.dataset}_val_psl_metrics.csv')
    if os.path.exists(val_log_file):
        os.remove(val_log_file)
    with open(val_log_file, 'w') as file:
        file.write(','.join(['Epoch', 'Loss', 'Loss pos', 'Loss neg']) + '\n')

    # 先定义psl_batch_size
    train_dataset = KGTripleDataset(dataset=args.dataset, split='train', num_neg_per_positive=args.num_neg_per_positive)
    val_dataset = KGTripleDataset(dataset=args.dataset, split='val')
    test_dataset = KGTripleDataset(dataset=args.dataset, split='test')
    psl_dataset = KGPSLTripleDataset(dataset=args.dataset)
    # 先定义psl_batch_size
    psl_batch_size = int(len(psl_dataset) / len(train_dataset) * args.batch_size)
    print('psl batch size:', psl_batch_size)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    psl_dataloader = DataLoader(psl_dataset, batch_size=psl_batch_size, shuffle=True)





    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_map[args.model](num_nodes=train_dataset.num_cons(), num_relations=train_dataset.num_rels(), hidden_channels=args.hidden_dim, model_type=args.model_type).to(device)

    evaluator = Evaluator(val_dataloader, model, batch_size=args.batch_size, device=device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        pos_loss_total = 0.0
        neg_loss_total = 0.0
        psl_loss_total = 0.0

        for idx, (batch_psl, batch) in enumerate(zip(psl_dataloader, train_dataloader)):
            psl_hrt, psl_score = batch_psl
            pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = batch

            # Move tensors to the configured device
            psl_hrt, psl_score = psl_hrt.to(device).long(), psl_score.to(device).float()
            pos_hrt, pos_score = pos_hrt.to(device).long(), pos_score.to(device).float()
            neg_hn_rt, neg_hr_tn = neg_hn_rt.to(device).long(), neg_hr_tn.to(device).long()

            # Forward pass
            psl_prob = model(psl_hrt[:, 0], psl_hrt[:, 1], psl_hrt[:, 2])
            pred_pos_score = model(pos_hrt[:,0], pos_hrt[:,1], pos_hrt[:,2])
            pred_neg_hn_score = model(neg_hn_rt[:,:,0], neg_hn_rt[:,:,1], neg_hn_rt[:,:,2])
            pred_neg_tn_score = model(neg_hr_tn[:, :, 0], neg_hr_tn[:, :, 1], neg_hr_tn[:, :, 2])

            # Compute targets
            psl_target = psl_score
            pos_target = pos_score
            neg_target = torch.zeros_like(pred_neg_hn_score)
            
            # # Compute losses
            psl_loss = compute_psl_loss(psl_prob, psl_target)
            pos_loss = criterion(pred_pos_score, pos_target)
            neg_loss = (criterion(pred_neg_hn_score, neg_target) + criterion(pred_neg_tn_score, neg_target)) / 2

            # print(f"Epoch [{epoch + 1}], Step [{idx + 1}]")
            # print(f"PSL Prob: {psl_prob}")
            # print(f"PSL Score: {psl_score}")
            # print(f"PSL Loss: {psl_loss.item()}")

            pos_loss_total += pos_loss.item()
            neg_loss_total += neg_loss.item()
            psl_loss_total += psl_loss.item()

            # 总损失
            loss = pos_loss + neg_loss + psl_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log to file
            with open(train_log_file, 'a') as file:
                file.write(f"{epoch + 1},{idx + 1},{loss.item():.4f},{pos_loss.item():.4f},{neg_loss.item():.4f},{psl_loss.item():.4f}\n")

            # Print every 10 steps
            if idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{args.num_epochs}], Step [{idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Pos Loss: {pos_loss.item():.4f}, Neg Loss: {neg_loss.item():.4f}, PSL Loss: {psl_loss.item():.4f}")

        # Log and print at the end of each epoch
        with open(train_log_file, 'a') as file:
            file.write(f"{epoch + 1},{total_loss/len(train_dataloader):.4f},{pos_loss_total/len(train_dataloader):.4f},{neg_loss_total/len(train_dataloader):.4f},{psl_loss_total/len(train_dataloader):.4f}\n")
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Total Loss: {total_loss/len(train_dataloader):.4f}, Pos Loss: {pos_loss_total/len(train_dataloader):.4f}, Neg Loss: {neg_loss_total/len(train_dataloader):.4f}, PSL Loss: {psl_loss_total/len(train_dataloader):.4f}")

        # 每 10 个 epoch 进行一次评估
        if (epoch + 1) % 10 == 0:
            model.eval()
            evaluator.update_hr_scores_map()
            mse = evaluator.get_mse()
            mae = evaluator.get_mae()
            mean_ndcg = evaluator.get_mean_ndcg()
            print(f"Evaluation at Epoch [{epoch + 1}/{args.num_epochs}]")
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"Mean nDCG: {mean_ndcg[0]:.4f}, Exponential Mean nDCG: {mean_ndcg[1]:.4f}")
        
            # 记录评估结果到 val_log_file
            with open(val_log_file, 'a') as file:
                file.write(f"{epoch + 1},{mse:.4f},{mae:.4f},{mean_ndcg[0]:.4f},{mean_ndcg[1]:.4f}\n")

        
if __name__ == "__main__":
    main()