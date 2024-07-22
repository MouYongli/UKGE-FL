import os
import os.path as osp
import shutil
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


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str.lower, default='distmult', choices=['transe','distmult','complex','rotate'])
    parser.add_argument('--dataset', type=str.lower, default='cn15k', choices=['cn15k', 'nl27k', 'ppi5k'])
    parser.add_argument('--num_neg_per_positive', default=10, type=int)
    parser.add_argument('--confidence_score_function', default='logi', type=str, choices=['logi', 'rect'])
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--topk', action='store_true', help='Set topk to True')
    parser.add_argument('--k', default=200, type=int)
    parser.add_argument('--fc_layers', default='none', type=str, choices=['l1', 'l3', 'none'])
    parser.add_argument('--bias', action='store_true', help='Set bias to True')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    exp_dir = osp.join(work_dir, f'unc_{args.dataset}_{args.model}_confi_{args.confidence_score_function}_fc_{args.fc_layers}_bias_{args.bias}_dim_{args.hidden_dim}', f'lr_{args.lr}_wd_{args.weight_decay}_topk_{args.topk}')
    if osp.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    # Create a CSV file to save losses and metrics
    train_log_file = os.path.join(exp_dir, 'train_metrics.csv')
    with open(train_log_file, 'w') as file:
        file.write(','.join(['Epoch', 'Step', 'Loss', 'Pos_Loss', 'Neg_Loss']) + '\n')

    val_log_file = os.path.join(exp_dir, 'val_metrics.csv')
    with open(val_log_file, 'w') as file:
        file.write(','.join(['Epoch', 'MSE', 'MAE', 'nDCG_lin', 'nDCG_exp']) + '\n')
    val_cls_npy_file = os.path.join(exp_dir, 'val_cls.npy')

    test_log_file = os.path.join(exp_dir, 'test_metrics.csv')
    with open(test_log_file, 'w') as file:
        file.write(','.join(['Epoch', 'MSE', 'MAE', 'MSE_with_neg', 'MAE_with_neg', 'nDCG_lin', 'nDCG_exp']) + '\n')
    test_cls_npy_file = os.path.join(exp_dir, 'test_cls.npy')
    test_cls_with_neg_npy_file = os.path.join(exp_dir, 'test_cls_with_neg.npy')

    train_dataset = KGTripleDataset(dataset=args.dataset, split='train', num_neg_per_positive=args.num_neg_per_positive)
    val_dataset = KGTripleDataset(dataset=args.dataset, split='val', topk=args.topk, k=args.k)
    test_dataset = KGTripleDataset(dataset=args.dataset, split='test', topk=args.topk, k=args.k)
    test_with_neg_dataset = KGTripleDataset(dataset=args.dataset, split='test', topk=args.topk, k=args.k, test_with_neg=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_with_neg_dataloader = DataLoader(test_with_neg_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_map[args.model](num_nodes=train_dataset.num_cons(), num_relations=train_dataset.num_rels(), hidden_channels=args.hidden_dim, confidence_score_function=args.confidence_score_function, fc_layers=args.fc_layers, bias=args.bias).to(device)
    criterion = nn.MSELoss()

    val_evaluator = Evaluator(val_dataloader, model, batch_size=args.batch_size, device=device, topk=args.topk)
    test_evaluator = Evaluator(test_dataloader, model, batch_size=args.batch_size, device=device, topk=args.topk)
    test_with_neg_evaluator = Evaluator(test_with_neg_dataloader, model, batch_size=args.batch_size, device=device, topk=args.topk)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=args.weight_decay)

    best_ndcg_lin = 0.0
    best_ndcg_lin_epoch = 0
    best_ndcg_exp = 0.0
    best_ndcg_exp_epoch = 0
    best_mae = float('inf')
    best_mae_epoch = 0
    best_mse = float('inf')
    best_mse_epoch = 0
    val_f1_history, test_f1_history, test_with_neg_f1_history= [], [], []

    for epoch in range(args.num_epochs):
        model.train()
        loss_total = 0.0
        pos_loss_total = 0.0
        neg_loss_total = 0.0

        for idx, (pos_hrt,pos_target, neg_hn_rt, neg_hr_tn) in enumerate(train_dataloader):
            pos_hrt, pos_target, neg_hn_rt, neg_hr_tn = pos_hrt.long(), pos_target.float(), neg_hn_rt.long(), neg_hr_tn.long()
            pos_hrt, pos_target, neg_hn_rt, neg_hr_tn = pos_hrt.to(device), pos_target.to(device), neg_hn_rt.to(device), neg_hr_tn.to(device) 
            neg_hn_rt = neg_hn_rt.view(-1, neg_hn_rt.size(-1))
            neg_hr_tn = neg_hr_tn.view(-1, neg_hr_tn.size(-1))

            optimizer.zero_grad()

            pred_pos_score = model.get_confidence_score(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
            pred_neg_hn_score = model.get_confidence_score(neg_hn_rt[:, 0], neg_hn_rt[:, 1], neg_hn_rt[:, 2])
            pred_neg_tn_score = model.get_confidence_score(neg_hr_tn[:, 0], neg_hr_tn[:, 1], neg_hr_tn[:, 2])
            
            neg_target = torch.zeros_like(pred_neg_hn_score)           

            pos_loss = criterion(pred_pos_score, pos_target)
            neg_loss = (criterion(pred_neg_hn_score, neg_target) + criterion(pred_neg_tn_score, neg_target)) / 2
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

            pos_loss_total += pos_loss.item()
            neg_loss_total += neg_loss.item()
            loss_total += loss.item()

            with open(train_log_file, 'a') as file:
                file.write(f"{epoch + 1},{idx + 1},{loss.item():.4f},{pos_loss.item():.4f},{neg_loss.item()}\n")
            if idx % 200 == 0:
                print(f"Epoch [{epoch + 1}/{args.num_epochs}], Step [{idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Positive Loss: {pos_loss.item():.4f}, Negative Loss: {neg_loss.item():.4f}")
        
        with open(train_log_file, 'a') as file:
            file.write(f"{epoch + 1}, -1, {loss_total/len(train_dataloader):.4f}, {pos_loss_total/len(train_dataloader):.4f}, {neg_loss_total/len(train_dataloader):.4f}\n")
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {loss_total/len(train_dataloader):.4f}, Positive Loss: {pos_loss_total/len(train_dataloader):.4f}, Negative Loss: {neg_loss_total/len(train_dataloader):.4f}")
        
        model.eval()
        val_evaluator.update()
        val_mean_ndcg = val_evaluator.get_mean_ndcg()
        val_mse = val_evaluator.get_mse()
        val_mae = val_evaluator.get_mae()
        ps, rs, f1s = val_evaluator.get_f1()
        val_f1_history.append([ps, rs, f1s])

        with open(val_log_file, 'a') as file:
            file.write(f"{epoch + 1}, {val_mse:.4f}, {val_mae:.4f}, {val_mean_ndcg[0]:.4f},{val_mean_ndcg[1]:.4f}\n")
        print(f"Validation\nMSE: {val_mse:.4f}, MAE: {val_mae:.4f}, Mean nDCG: {val_mean_ndcg[0]:.4f}, Exponential Mean nDCG: {val_mean_ndcg[1]:.4f}")

        if val_mean_ndcg[0] > best_ndcg_lin:
            best_ndcg_lin = val_mean_ndcg[0]
            best_ndcg_lin_epoch = epoch + 1
            torch.save({
                'state_dict': model.state_dict(),
                'best_ndcg_lin': best_ndcg_lin,
                'best_ndcg_lin_epoch': best_ndcg_lin_epoch
                }, osp.join(exp_dir, 'best_model_ndcg_lin.pth'))
        
        if val_mean_ndcg[1] > best_ndcg_exp:
            best_ndcg_exp = val_mean_ndcg[1]
            best_ndcg_exp_epoch = epoch + 1
            torch.save({
                'state_dict': model.state_dict(),
                'best_ndcg_exp': best_ndcg_exp,
                'best_ndcg_exp_epoch': best_ndcg_exp_epoch
                }, osp.join(exp_dir, 'best_model_ndcg_exp.pth'))
        
        if val_mae < best_mae:
            best_mae = val_mae
            best_mae_epoch = epoch + 1
            torch.save({
                'state_dict': model.state_dict(),
                'best_mae': best_mae,
                'best_mae_epoch': best_mae_epoch
                }, osp.join(exp_dir, 'best_model_mae.pth'))
        
        if val_mse < best_mse:
            best_mse = val_mse
            best_mse_epoch = epoch + 1
            torch.save({
                'state_dict': model.state_dict(),
                'best_mse': best_mse,
                'best_mse_epoch': best_mse_epoch
                }, osp.join(exp_dir, 'best_model_mse.pth'))

        model.eval()
        test_evaluator.update()
        test_with_neg_evaluator.update_hr_tp_map()

        test_mean_ndcg = test_evaluator.get_mean_ndcg()

        test_mse = test_evaluator.get_mse()
        test_mae = test_evaluator.get_mae()

        test_mse_with_neg = test_with_neg_evaluator.get_mse()
        test_mae_with_neg = test_with_neg_evaluator.get_mae()

        ps, rs, f1s = test_evaluator.get_f1()
        ps_with_neg, rs_with_neg, f1s_with_neg = test_with_neg_evaluator.get_f1()

        test_f1_history.append([ps, rs, f1s])
        test_with_neg_f1_history.append([ps_with_neg, rs_with_neg, f1s_with_neg])

        with open(test_log_file, 'a') as file:
            file.write(f"{epoch + 1}, {test_mse:.4f}, {test_mae:.4f}, {test_mse_with_neg:.4f}, {test_mae_with_neg:.4f}, {test_mean_ndcg[0]:.4f},{test_mean_ndcg[1]:.4f}\n")
        print(f"Test\nMSE: {test_mse:.4f}, MAE: {test_mae:.4f}, MSE (with neg): {test_mse_with_neg:.4f}, MAE (with neg): {test_mae_with_neg:.4f}, Mean nDCG: {test_mean_ndcg[0]:.4f}, Exponential Mean nDCG: {test_mean_ndcg[1]:.4f}")

    np.save(val_cls_npy_file, np.array(val_f1_history))
    np.save(test_cls_npy_file, np.array(test_f1_history))
    np.save(test_cls_with_neg_npy_file, np.array(test_with_neg_f1_history))


if __name__ == "__main__":
    main()