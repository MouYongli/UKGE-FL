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
from torch.utils.tensorboard import SummaryWriter

model_map = {
    'distmult': DistMult,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str.lower, default='distmult', choices=['distmult'])
    parser.add_argument('--model_type', type=str.lower, default='logi', choices=['logi', 'rect'])
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

    for epoch in range(args.num_epochs):
        model.train()  
        total_loss = 0

        for idx, batch in enumerate(train_dataloader): 
    
            pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = batch
            pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = pos_hrt.to(device), pos_score.to(device), neg_hn_rt.to(device), neg_hr_tn.to(device)
            pred_pos_score = model(pos_hrt[:,0].long(), pos_hrt[:,1].long(), pos_hrt[:,2].long())
            # 负样本hn和tn的fl
            pred_neg_hn_score = model(neg_hn_rt[:,:,0].long(), neg_hn_rt[:,:,1].long(), neg_hn_rt[:,:,2].long())
            pred_neg_tn_score = model(neg_hr_tn[:, :, 0].long(), neg_hr_tn[:, :, 1].long(), neg_hr_tn[:, :, 2].long())
            
            # PSL的target
            psl_target = psl_score
            # Dataset的target
            pos_target = pos_score
            neg_target = torch.zeros_like(pred_neg_hn_score)


            # PSLloss
            psl_loss = compute_psl_loss(psl_prob, psl_target)
            
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












class Trainer(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.dim = args.hidden_dim
        self.num_neg_per_positive = args.num_neg_per_positive
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = args.model_type
        self.model = None
        self.validator = None
        self.l2_lambda = args.l2_lambda  # L2正则化系数

    def build(self, train_dataset, val_dataset, psl_dataset, model_name, model_psl):
        # 先定义psl_batch_size
        self.psl_batch_size = int(len(psl_dataset) / len(train_dataset) * self.batch_size)

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        self.psl_dataloader = DataLoader(psl_dataset, batch_size=self.psl_batch_size, shuffle=True)
        self.model = model_map[model_name](num_nodes=train_dataset.num_cons(), num_relations=train_dataset.num_rels(), hidden_channels=self.dim, model_type=self.model_type).to(self.device)
        self.model_psl = model_psl.to(self.device)

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        train_losses = []
        val_losses = []
        log_dir = f'runs/bs_{self.batch_size}_lr_{self.lr}_l2_{self.l2_lambda}'
        writer = SummaryWriter(log_dir=log_dir)



        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for batch_psl, batch in zip(self.psl_dataloader, self.train_dataloader):
                psl_hrt, psl_score = batch_psl
                pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = batch

                # Move tensors to the configured device
                psl_hrt, psl_score = psl_hrt.to(self.device), psl_score.to(self.device)
                pos_hrt, pos_score = pos_hrt.to(self.device), pos_score.to(self.device)
                neg_hn_rt, neg_hr_tn = neg_hn_rt.to(self.device), neg_hr_tn.to(self.device)

                # Forward pass
                psl_prob = self.model_psl(psl_hrt[:, 0].long(), psl_hrt[:, 1].long(), psl_hrt[:, 2].long())
                pred_pos_score = self.model(pos_hrt[:,0].long(), pos_hrt[:,1].long(), pos_hrt[:,2].long())
                pred_neg_hn_score = self.model(neg_hn_rt[:,:,0].long(), neg_hn_rt[:,:,1].long(), neg_hn_rt[:,:,2].long())
                pred_neg_tn_score = self.model(neg_hr_tn[:, :, 0].long(), neg_hr_tn[:, :, 1].long(), neg_hr_tn[:, :, 2].long())

                # Compute targets
                psl_target = psl_score
                pos_target = pos_score
                neg_target = torch.zeros_like(pred_neg_hn_score)

                # Compute losses
                psl_loss = compute_psl_loss(psl_prob, psl_target)
                pos_loss = criterion(pred_pos_score, pos_target)
                neg_loss = (criterion(pred_neg_hn_score, neg_target) + criterion(pred_neg_tn_score, neg_target)) / 2

                l2_reg = torch.tensor(0., device=self.device)  # 将l2_reg初始化到设备上
                for param in self.model.parameters():
                    l2_reg = l2_reg + torch.norm(param, p=2)**2  # 累加L2损失(平方？

                # 总损失
                loss = pos_loss + neg_loss + psl_loss + self.l2_lambda * l2_reg


                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_dataloader)
            train_losses.append(avg_train_loss)
            val_loss = self.validate()
            val_losses.append(val_loss)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_train_loss:.4f}")
            writer.add_scalar('Loss/training loss', avg_train_loss, epoch)
            writer.add_scalar('Loss/validation loss', val_loss[0], epoch)

            if (epoch + 1) % 10 == 0:
                # val_loss = self.validate()
                # val_losses.append(val_loss)
                print(f"Validation Loss after epoch [{epoch + 1}/{self.num_epochs}]: {val_loss[0]:.4f}") 
                # writer.add_scalar('Loss/training loss', avg_train_loss, epoch)
                # writer.add_scalar('Loss/validation loss', val_loss[0], epoch)
        torch.save(self.model.state_dict(), 'model_final.pth')
        pd.DataFrame(train_losses, columns=['train_loss']).to_csv('training_loss.csv', index=False)
        pd.DataFrame(val_losses, columns=['val_loss', 'mse loss', 'mae loss']).to_csv('validation_loss.csv', index=False)

    def validate(self):
        self.model.eval()
        mse = 0.0
        mae = 0.0
        num_samples = 0

        criterion = nn.MSELoss()
        with torch.no_grad():
            for batch in self.val_dataloader:
                pos_hrt, pos_score = batch

                pos_hrt, pos_score = pos_hrt.to(self.device), pos_score.to(self.device)

                pred_pos_score = self.model(pos_hrt[:, 0].long(), pos_hrt[:, 1].long(), pos_hrt[:, 2].long())

                pos_loss = criterion(pred_pos_score, pos_score)
                loss = pos_loss

                mse += torch.nn.functional.mse_loss(pred_pos_score, pos_score, reduction='sum').item()
                mae += torch.nn.functional.l1_loss(pred_pos_score, pos_score, reduction='sum').item()
                num_samples += pos_score.size(0)

        mse /= num_samples
        mae /= num_samples

        return loss, mse, mae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str.lower, default='distmult', choices=['distmult'])
    parser.add_argument('--model_type', type=str.lower, default='rect', choices=['logi', 'rect'])
    parser.add_argument('--dataset', type=str.lower, default='cn15k', choices=['cn15k', 'nl27k', 'ppi5k'])
    parser.add_argument('--num_neg_per_positive', default=10, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2_lambda', default=0.0005, type=float)  # L2正则化系数
    args = parser.parse_args()

    train_dataset = KGTripleDataset(dataset=args.dataset, split='train', num_neg_per_positive=args.num_neg_per_positive)
    val_dataset = KGValTripleDataset(dataset=args.dataset)
    psl_dataset = KGPSLTripleDataset(dataset=args.dataset)

    # 因为要用zip尽量保持batch数量一致，这里计算psl的batch_size
    # psl_batch_size = int(len(psl_dataset) / len(train_dataset) * args.batch_size)
    # psl_dataloader = DataLoader(psl_dataset, batch_size=psl_batch_size, shuffle=True)

    trainer = Trainer(args)
    model_psl = model_map[args.model](num_nodes=psl_dataset.num_cons(), num_relations=psl_dataset.num_rels(), hidden_channels=args.hidden_dim, model_type=args.model_type).to(trainer.device)
    # trainer.psl_dataloader = psl_dataloader
    trainer.build(train_dataset, val_dataset, psl_dataset, args.model, model_psl)
    trainer.train()




if __name__ == "__main__":
    main()
