import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from config import TOKEN_FILE, BERT_PATH, DEVICE

SAVE_DIR = "./data/text_embeddings"

# 加载模型
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
model = BertModel.from_pretrained(BERT_PATH).to(DEVICE).eval()

# 读取文件，组织成字典：image_id → [句子1, 句子2, ...]
caption_dict = defaultdict(list)
with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        img_sent, caption = line.strip().split('\t')
        image_id = img_sent.split('#')[0]
        caption_dict[image_id].append(caption)

# 开始提取并保存
os.makedirs(SAVE_DIR, exist_ok=True)

for image_id, sentences in tqdm(caption_dict.items(), desc="Extracting text embeddings"):
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=32)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs).pooler_output.squeeze(0).cpu().numpy()
        embeddings.append(output)
    
    # 将 5 个句子向量平均
    mean_embedding = np.mean(embeddings, axis=0)

    image_id_without_ext = os.path.splitext(image_id)[0]  # 去掉文件扩展名
    np.save(os.path.join(SAVE_DIR, f"{image_id_without_ext}.npy"), mean_embedding)
