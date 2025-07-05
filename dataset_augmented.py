import os
import torch
import numpy as np
from torch.utils.data import Dataset

class AugmentedKeywordDataset(Dataset):
    def __init__(self, pair_file, img_emb_dir, txt_emb_dir, max_len=32, device='cuda'):
        """
        参数：
            pair_file: 包含图像名和文本描述的 .txt 文件（格式：img.jpg\ttext）
            img_emb_dir: 图像特征 .npy 文件目录
            txt_emb_dir: 已预提取好的文本特征 .npy 文件目录（增强后的文本特征）
            max_len: 保留参数，备用
            device: 加载位置
        """
        self.pairs = []
        with open(pair_file, 'r', encoding='utf-8') as f:
            for line in f:
                img_name, _ = line.strip().split('\t')
                self.pairs.append(img_name)

        self.img_emb_dir = img_emb_dir
        self.txt_emb_dir = txt_emb_dir
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name = self.pairs[idx]

        # 加载图像特征
        img_path = os.path.join(self.img_emb_dir, img_name.replace('.jpg', '.npy'))
        img_feat = np.load(img_path)
        img_feat = torch.tensor(img_feat, dtype=torch.float32)

        # 加载文本特征
        txt_path = os.path.join(self.txt_emb_dir, img_name.replace('.jpg', '.npy'))
        txt_feat = np.load(txt_path)
        txt_feat = torch.tensor(txt_feat, dtype=torch.float32)

        return img_feat, txt_feat
