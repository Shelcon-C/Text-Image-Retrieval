# 🔍 Flickr30K 跨模态图文检索系统  
*基于 PyTorch 的轻量级图文匹配与检索项目*

本项目实现了一个完整的跨模态检索系统，通过训练图像与文本在同一语义空间中的对齐，实现**文本查图**功能。使用 PyTorch 框架，基于公开数据集 Flickr30K 构建，支持 Gradio Web 前端交互演示。

---

## 项目特点

- 图文匹配：对比学习训练图文在同一语义空间
- 图像特征：ResNet18 提取 512 维表示
- 文本特征：本地 BERT 模型提取 768 维表示（无联网依赖）
- 跨模态对齐：MLP 投影到统一 256 维空间
- Gradio Web UI：支持 Top-5 文本查图
- 支持缓存 & 加载特征，避免重复计算

---



## 项目结构说明

Image_Retrive/
├── data/
│ ├── flickr30k-images/ # 原始图像数据集
│ ├── image_embeddings/ # 提取后的图像特征 (.npy)
│ ├── text_embeddings/ # 提取后的文本特征 (.npy)
│ ├── aug_text_embeddings/ # 文本增强后的特征 (.npy)
│ ├── images/ # demo图像（可选）
│ ├── flickr30k_keywords_augmented.txt #文本增强关键词提取内容
│ └── flickr30k.token # 每张图像对应的5条文本描述
│
├── models/
│ └── bert-base-uncased/ # 本地BERT模型
│ ├── config.json
│ ├── model.safetensors
│ ├── tokenizer_config.json
│ ├── tokenizer.json
│ └── vocab.txt
│
├── app.py # Gradio Web UI，支持文本查图
├── batch_image_feature_extract.py # 批量提取图像特征
├── batch_text_feature_extract.py # 批量提取文本特征
├── config.py # 配置项（文件路径等）
├── data_loader.py # 自定义 PyTorch Dataset & DataLoader
├── debug.py # 辅助调试脚本（测试小bug）
├── image_extractor.py # 使用 ResNet18 提取图像特征
├── loss.py # InfoNCE / 对比损失函数
├── main.py # 入口脚本（可能为 demo 测试）
├── model.py # 投影模型（两层MLP+GELU激活+残差连接）
├── run_train.py # 模型训练脚本
├── token_loader.py # 加载 .token 文件的解析器
└── README.md # 当前项目说明文档



---

## 环境依赖

```bash
conda create -n  yourname  python=3.10
conda activate yourname

pip install torch torchvision transformers gradio tqdm numpy pillow safetensors

 快速开始

 提取图文特征
python batch_image_feature_extract.py   # → 保存到 data/image_embeddings/
python batch_text_feature_extract.py    # → 保存到 data/text_embeddings/
 模型训练
python run_train.py
训练后模型保存在 checkpoints/（可手动添加该文件夹）

 启动 Gradio Web Demo（文本查图）
python app.py
运行后在浏览器中打开本地界面，输入英文描述，即可返回 Top-5 匹配图像。

 Flickr30K 数据说明
共 31,000 张图像，每张对应 5 条英文描述

.token 文件格式为：

1000092795.jpg#0\tTwo young guys with shaggy hair...
1000092795.jpg#1\tTwo young , White males are outside...
图像路径：data/flickr30k-images/
描述文件：data/flickr30k.token

模型结构图示

[ ResNet18 (512维) ] ──> MLP投影 → 256维共享空间 ← MLP投影 ── [ BERT (768维) ]
训练使用对比损失（如 InfoNCE 或 Triplet Loss）

将匹配的图像-文本拉近，非配对样本推远

后续拓展方向
 支持图查文（Image → Text）

 中文支持（改用 bert-base-chinese）

 FAISS/Annoy 加速大规模向量检索

 CLIP集成 + rerank优化结果

 Recall@1/5/10 指标评估模块

作者
本项目由 [Shelcon] 开发，旨在研究和实现跨模态图文语义检索系统。
联系方式：ShelconChan@gmail.com

许可证
本项目基于 MIT License 开源发布。

