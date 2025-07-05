import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from config import IMG_DIR 


SAVE_DIR = "./data/image_embeddings"   
os.makedirs(SAVE_DIR, exist_ok=True)

# 加载 ResNet18 并移除最后一层（只取特征）
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 去掉fc层
model.eval().cuda()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet标准化
        std=[0.229, 0.224, 0.225]
    )
])

# 批量提取
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
for img_file in tqdm(image_files, desc="Extracting image features"):
    img_path = os.path.join(IMG_DIR, img_file)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).cuda()

    with torch.no_grad():
        feat = model(input_tensor).squeeze().cpu().numpy()  # 输出为512维

    # 保存为 .npy
    image_id = os.path.splitext(img_file)[0]
    np.save(os.path.join(SAVE_DIR, f"{image_id}.npy"), feat)
