import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import KGTripleDataset
from model import DistMult

# 超参数
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_CHANNELS = 128

# 数据集
train_dataset = KGTripleDataset(dataset='cn15k', split='train', num_neg_per_positive=10)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 模型初始化
model = DistMult(
    num_nodes=train_dataset.num_cons(),
    num_relations=train_dataset.num_rels(),
    hidden_channels=HIDDEN_CHANNELS,
    model_type='logi',  # 选择"logi"或"rect"
)

# 损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练循环
for epoch in range(EPOCHS):
    model.train()  # 切换到训练模式
    total_loss = 0

    for batch in train_loader:
        pos_hrt, pos_score, neg_hn_rt, neg_hr_tn = batch
        
        optimizer.zero_grad()  # 重置梯度
        
        # 正样本的预测评分
        pos_pred = model(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
        
        # 负样本的预测评分
        neg_hn_pred = model(neg_hn_rt[:, :, 0], neg_hn_rt[:, :, 1], neg_hn_rt[:, :, 2])
        neg_hr_pred = model(neg_hr_tn[:, :, 0], neg_hr_tn[:, :, 1], neg_hr_tn[:, :, 2])
        
        # 创建正样本和负样本的目标向量
        pos_target = torch.ones_like(pos_pred)  # 正样本的目标是1
        neg_target = torch.zeros_like(neg_hn_pred)  # 负样本的目标是0
        
        # 计算损失
        pos_loss = criterion(pos_pred, pos_target)  # 正样本的损失
        neg_loss = (criterion(neg_hn_pred, neg_target) + criterion(neg_hr_pred, neg_target)) / 2  # 负样本的损失
        loss = pos_loss + neg_loss
        
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        
        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

print("Training complete!")
