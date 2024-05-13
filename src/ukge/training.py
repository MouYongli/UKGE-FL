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

model_map = {
    'distmult': DistMult,
}

from ukge.losses import PSL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str.lower, default='distmult', choices=['distmult'])
    parser.add_argument('--model_type', type=str.lower, default='logi', choices=['logi', 'rect'])
    parser.add_argument('--loss_type', type=str.lower, default='logi', choices=['', 'rect'])
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
    #需要一个psl_dataloader
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_map[args.model](num_nodes=train_dataset.num_cons(), num_relations=train_dataset.num_rels(), hidden_channels=args.hidden_dim, model_type=args.model_type).to(device)



    #以下是训练部分
    
    #准备数据集（dataloader加载
    #创建model（就是model.py里定义的
    #设置参数(args里定义
    
    #损失函数
    #优化器
    #训练-传入预测的fl和target的sl，计算loss，backward，opt


        
    #数据集（dataloader加载(前边定义了

    #模型(前边定义了


    #loss和optimizer
    criterion = nn.MSELoss()
    def compute_psl_loss(self): 
        self.prior_psl0 = torch.tensor(self._prior_psl, dtype=torch.float32)
        self.psl_error_each = torch.square(torch.max(self.soft_w + self.prior_psl0 - self.psl_prob, torch.tensor(0)))
        self.psl_mse = torch.mean(self.psl_error_each)
        self.psl_loss = self.psl_mse * self._p_psl

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #训练
    for epoch in range(args.num_epochs):
        model.train()  
        total_loss = 0

        for batch_psl, batch in zip(psl_dataloader, train_dataloader): #需要创建一个psl_dataloader 
            psl_hrt, psl_score = batch_psl
            pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = batch
            
            # PSL的fl
            psl_prob = model(psl_hrt[:,0], psl_hrt[:,1], psl_hrt[:,2])
            # 正样本的fl
            pred_pos_score = model(pos_hrt[:,0], pos_hrt[:,1], pos_hrt[:,2])
            # 负样本hn和tn的fl
            pred_neg_hn_score = model(neg_hn_rt[:,:,0], neg_hn_rt[:,:,1], neg_hn_rt[:,:,2])
            pred_neg_tn_score = model(neg_hr_tn[:, :, 0], neg_hr_tn[:, :, 1], neg_hr_tn[:, :, 2])
            
            # PSL的target
            psl_target = psl_score
            # Dataset的target
            pos_target = pos_score
            neg_target = torch.zeros_like(pred_neg_hn_score)


            # PSLloss
            psl_loss = compute_psl_loss()
            
            # loss
            pos_loss = criterion(pred_pos_score, pos_target)
            neg_loss = (criterion(pred_neg_hn_score, neg_target) + criterion(pred_neg_tn_score, neg_target)) / 2
            loss = pos_loss + neg_loss + psl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}")


if __name__ == "__main__":
    main()


