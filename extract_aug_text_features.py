# extract_aug_text_features.py
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from config import BERT_PATH, DEVICE
# 配置参数
TEXT_FILE = "./data/flickr30k_keywords_augmented.txt"
SAVE_DIR = "./data/aug_text_embeddings"


# 加载模型
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
model = BertModel.from_pretrained(BERT_PATH).to(DEVICE).eval()

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 处理文件
with open(TEXT_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in tqdm(lines, desc="Extracting BERT features"):
    try:
        img_name, sentence = line.strip().split('\t')
        tokens = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=32)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        with torch.no_grad():
            feat = model(**tokens).pooler_output.squeeze(0).cpu().numpy()

        # 保存为 .npy 文件，文件名同图像文件
        npy_name = img_name.replace(".jpg", ".npy")
        save_path = os.path.join(SAVE_DIR, npy_name)
        np.save(save_path, feat)
    except Exception as e:
        print(f"Error in line: {line.strip()}\n{e}")
