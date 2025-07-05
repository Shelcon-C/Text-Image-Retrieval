import torch
from tqdm import tqdm
import torch.nn.functional as F
from loss import NTXentLoss  

loss_fn = NTXentLoss() 

def train_epoch(model, dataloader, optimizer, device, writer=None, epoch=0):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    for step, (img_feat, txt_feat) in enumerate(pbar):
        img_feat, txt_feat = img_feat.to(device), txt_feat.to(device)
        img_proj, txt_proj = model(img_feat, txt_feat)

        loss, _ = loss_fn(img_proj, txt_proj) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    if writer:
        writer.add_scalar("train/loss", avg_loss, epoch)
    return avg_loss


def evaluate(model, data_loader, device, writer=None, epoch=0, chunk_size=512):
    model.eval()
    img_feats_all = []
    txt_feats_all = []

    with torch.no_grad():
        for img_feat, txt_feat in tqdm(data_loader, desc="Evaluating"):
            img_feat = img_feat.to(device)
            txt_feat = txt_feat.to(device)

            img_proj, txt_proj = model(img_feat, txt_feat)
            img_feats_all.append(img_proj)
            txt_feats_all.append(txt_proj)

        img_all = torch.cat(img_feats_all, dim=0)  # [N, D]
        txt_all = torch.cat(txt_feats_all, dim=0)  # [N, D]

        img_all = F.normalize(img_all, dim=1)
        txt_all = F.normalize(txt_all, dim=1)

        top1, top5 = 0.0, 0.0
        N = img_all.size(0)

        for i in range(0, N, chunk_size):
            img_chunk = img_all[i:i+chunk_size]  # [B, D]
            sims = torch.matmul(img_chunk, txt_all.T)  # [B, N]
            targets = torch.arange(i, min(i+chunk_size, N), device=sims.device)

            top1 += (sims.argmax(dim=1) == targets).sum().item()
            top5 += sum([1 if targets[j] in sims[j].topk(5).indices else 0 for j in range(len(targets))])

        acc1 = top1 / N
        acc5 = top5 / N

        if writer:
            writer.add_scalar("eval/top1", acc1, epoch)
            writer.add_scalar("eval/top5", acc5, epoch)

    return acc1, acc5

