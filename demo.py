import os
import torch
import numpy as np
import gradio as gr
from transformers import BertTokenizer, BertModel
from model import CrossModalProjection
import torch.nn.functional as F
from PIL import Image
from config import IMG_DIR, IMG_EMB_DIR, BERT_PATH, DEVICE

# 模型路径
MODEL_CKPT = "./checkpoints/best_model.pt"

# 加载 BERT
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
bert = BertModel.from_pretrained(BERT_PATH).to(DEVICE).eval()

# 加载投影模型
proj_model = CrossModalProjection().to(DEVICE)
proj_model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE)['model_state_dict'])
proj_model.eval()

# 预加载所有图像投影向量
img_proj_dict = {}
dummy_txt = torch.zeros(1, 768).to(DEVICE)  # 占位用，模型 forward 需要两个输入
for fname in os.listdir(IMG_EMB_DIR):
    if not fname.endswith(".npy"):
        continue
    key = fname[:-4]
    img_feat = torch.tensor(np.load(os.path.join(IMG_EMB_DIR, fname)), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_proj, _ = proj_model(img_feat, dummy_txt)  # 使用 forward
        img_proj = F.normalize(img_proj, dim=1)
    img_proj_dict[key] = img_proj

# 合并成 tensor 矩阵
img_keys = list(img_proj_dict.keys())
img_matrix = torch.cat([img_proj_dict[k] for k in img_keys], dim=0)

# 检索函数
def retrieve_image_from_text(text_input):
    with torch.no_grad():
        inputs = tokenizer(text_input, return_tensors='pt', truncation=True, padding=True, max_length=32)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        txt_feat = bert(**inputs).pooler_output
        dummy_img = torch.zeros(1, 512).to(DEVICE)  # 图像占位
        _, txt_proj = proj_model(dummy_img, txt_feat)  # 使用 forward
        txt_proj = F.normalize(txt_proj, dim=1)

        sims = torch.matmul(txt_proj, img_matrix.T)[0]
        topk_indices = torch.topk(sims, k=5).indices.tolist()
        topk_imgs = [os.path.join(IMG_DIR, f"{img_keys[i]}.jpg") for i in topk_indices]
        return [Image.open(p) for p in topk_imgs]
    
# Gradio UI
demo = gr.Interface(
    fn=retrieve_image_from_text,
    inputs=gr.Textbox(label="输入英文描述（如：Three men are cooking a meal .）或者单个英文单词(如:dog)"),
    outputs=[gr.Image(label=f"Top-{i+1}") for i in range(5)],
    title="文本查图跨模态检索",
    description="输入一句英文描述，我会返回最相关的图片 Top-5"
)

if __name__ == "__main__":
    demo.launch()
