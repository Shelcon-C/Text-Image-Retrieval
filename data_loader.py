import os
import numpy as np
from torch.utils.data import Dataset
import torch
class CrossModalDataset(Dataset):
    def __init__(self, image_dir, text_dir):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.keys = [f[:-4] for f in os.listdir(image_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        image_feat = np.load(os.path.join(self.image_dir, f"{key}.npy"))
        text_feat = np.load(os.path.join(self.text_dir, f"{key}.npy"))
        return torch.tensor(image_feat, dtype=torch.float32), torch.tensor(text_feat, dtype=torch.float32)
