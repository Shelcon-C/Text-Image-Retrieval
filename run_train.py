import torch
from torch.utils.data import DataLoader
from model import CrossModalProjection
from train import train_epoch, evaluate
from data_loader import CrossModalDataset
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from smart_sampler import get_balanced_loader
import torch
from config import DEVICE
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("使用GPU:", torch.cuda.get_device_name(0))
else:
    print("没有检测到 GPU，当前使用 CPU")
# 初始化日志目录
writer = SummaryWriter(log_dir="runs/flickr30k_experiment111")

# 获取平衡采样的 DataLoader
train_loader = get_balanced_loader(
    orig_img_dir="./data/image_embeddings",
    orig_txt_dir="./data/text_embeddings",
    aug_txt_file="./data/flickr30k_keywords_augmented.txt",
    img_emb_dir="./data/image_embeddings",
    aug_txt_emb_dir="./data/aug_text_embeddings",
    device=DEVICE,
    batch_size=64,
    ratio_orig_to_aug=4.0, #原始数据采样频率是增强数据的 4 倍
    total_samples=80000 #总采样数
)
'''
# 只用原始数据
orig_dataset = CrossModalDataset(
    image_dir="./data/image_embeddings",
    text_dir="./data/text_embeddings"
)
train_loader = DataLoader(orig_dataset, batch_size=64, shuffle=True)
'''

# 初始化模型和优化器
model = CrossModalProjection().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# 创建保存目录
os.makedirs("checkpoints", exist_ok=True)
best_top1 = 0.0
best_model_path = "checkpoints/best_model.pt"

for epoch in range(400):
    loss = train_epoch(model, train_loader, optimizer, DEVICE, writer=writer, epoch=epoch)
    top1, top5 = evaluate(model, train_loader, DEVICE, writer=writer, epoch=epoch)
    scheduler.step()

    print(f"\nEpoch {epoch+1} | Loss: {loss:.4f} | Top-1: {top1:.4f} | Top-5: {top5:.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'top1': top1,
            'top5': top5
        }, f"checkpoints/epoch_{epoch+1}.pt")

    if top1 > best_top1:
        best_top1 = top1
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'top1': top1,
            'top5': top5
        }, best_model_path)
        print(f"Best model updated at Epoch {epoch+1} with Top-1: {top1:.4f}")

