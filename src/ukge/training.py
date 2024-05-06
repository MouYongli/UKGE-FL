import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from ukge.datasets import KGTripleDataset
from ukge.models import DistMult

#准备数据集（dataloader加载
#创建model（就是model.py里定义的
#设置参数
#损失函数
#优化器
#训练-传入预测的fl和target的sl，计算loss，backward，opt

#超参数
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_CHANNELS = 128

#数据集（dataloader加载
train_dataset = KGTripleDataset(dataset='cn15k', split='train', num_neg_per_positive=10)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

#模型
model = DistMult(
    num_nodes=train_dataset.num_cons(),
    num_relations=train_dataset.num_rels(),
    hidden_channels=HIDDEN_CHANNELS,
    model_type='logi',  # 选择"logi"或"rect"
)

#loss和optimizer
criterion = nn.CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#训练
for epoch in range(EPOCHS):
    model.train()  
    total_loss = 0

    for batch in train_loader:
        pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = batch
        
        #正样本fl
        pred_pos_score = model(pos_hrt[:,0], pos_hrt[:,1], pos_hrt[:,2])
        #负样本hn和tn的fl
        pred_neg_hn_score = model(neg_hn_rt[:,:,0], neg_hn_rt[:,:,1], neg_hn_rt[:,:,2])
        pred_neg_tn_score = model(neg_hr_tn[:, :, 0], neg_hr_tn[:, :, 1], neg_hr_tn[:, :, 2])
        
        #target（负样本score？
        pos_target = pos_score
        # neg_target = torch.

        #计算损失
        pos_loss = criterion(pred_pos_score, pos_target)
        neg_loss = (criterion(pred_neg_hn_score, neg_target) + criterion(pred_neg_tn_score, neg_target)) / 2  #loss取hn和tn的均值（？
        loss = pos_loss + neg_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")
