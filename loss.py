import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, img_proj, txt_proj):
        # 特征归一化
        img_proj = F.normalize(img_proj, dim=1)
        txt_proj = F.normalize(txt_proj, dim=1)

        # 相似度矩阵：图像对文本
        logits = torch.matmul(img_proj, txt_proj.T) / self.temperature

        # 标签是对角线（i-i 对齐）
        targets = torch.arange(logits.size(0)).to(logits.device)

        # 计算双向 InfoNCE 损失（图像->文本 + 文本->图像）
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)
        loss = (loss_i2t + loss_t2i) / 2

        return loss, logits
