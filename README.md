# 基于Pytorch的跨模态检索系统

本项目实现了一个简单完整的跨模态检索系统，通过训练图像与文本在同一语义空间中的对齐，实现**文本查图**功能。使用 PyTorch 框架，基于公开数据集 Flickr30K 构建，支持前端交互演示。

![image](https://github.com/user-attachments/assets/14136f4e-dbc1-494c-b4c4-8c79478e34f9)

![image](https://github.com/user-attachments/assets/8e71e9af-0cd1-4a55-af86-2664bdf718c7)

![image](https://github.com/user-attachments/assets/8160cde2-23ea-40b0-89d7-0caff8e0f981)

模型结构:
[ ResNet18 (512维) ] ──> MLP投影 → 256维共享空间 ← MLP投影 ── [ BERT (768维) ]

## 数据集
 Flickr30K：https://shannon.cs.illinois.edu/DenotationGraph/
共 31,000 张图像，每张对应 5 条英文描述
.token 文件格式为：
1000092795.jpg#0\tTwo young guys with shaggy hair...    
1000092795.jpg#1\tTwo young , White males are outside...  
图像路径：data/flickr30k-images/
描述文件：data/flickr30k.token

# 环境
conda env create -f environment.yml

## 特征提取
采用ResNet18作为图像编码器，用于从图像中提取特征，并将这些特征保存至 `data/image_embeddings/` 目录下。对于文本特征的提取，则使用了BERT预训练模型，以获取文本的深层语义表示，并将提取到的特征存储在 `data/text_embeddings/` 目录。
python batch_image_feature_extract.py  
python batch_text_feature_extract.py  

在初步实验中，我仅对五条描述进行了文本特征提取，发现模型只能记住描述本身而未能理解其语义含义，尤其当输入为单个词汇时，模型无法建立该词与对应图片之间的有效联系。
为了克服上述问题，我利用了NLTK库中的分词工具，对文本描述进行关键词切分和提取，以此来增强原始文本数据。具体步骤如下：
1. **文本增强**：运行 `sentence_augment.py` 脚本对文本描述进行关键词提取。
2. **增强文本特征提取**：执行 `extract_aug_text_features.py` 脚本来提取经过关键词处理后的文本特征。 
## 训练
python run_train.py
训练后模型保存在 checkpoints/
## 测试
python app.py
运行后在浏览器中打开本地界面，输入英文描述，即可返回 Top-5 匹配图像。
## 项目结构说明
- **data/**
  - `flickr30k-images/`              # 原始图像数据集
  - `image_embeddings/`             # 提取后的图像特征 (.npy)
  - `text_embeddings/`              # 提取后的文本特征 (.npy)
  - `aug_text_embeddings/`          # 文本增强后的特征 (.npy)
  - `flickr30k_keywords_augmented.txt` # 文本增强关键词提取内容
  - `flickr30k.token`               # 每张图像对应的5条文本描述
- **models/**
  - `bert-base-uncased/`            # 本地BERT模型
    - `config.json`
    - `model.safetensors`
    - `tokenizer_config.json`
    - `tokenizer.json`
    - `vocab.txt`
- `README.md`
- `config.py`         # 配置项（文件路径等）
- `batch_image_feature_extractor.py`    # 批量提取图像特征
- `batch_text_feature_extractor.py`    # 批量提取文本特征
- `dataset_augmented.py`   # 数据增强（混合）函数
- `demo.py`    # Gradio Web UI，支持文本查图
- `extract_aug_text_features.py`
- `loss.py`     # InfoNCE / 对比损失函数
- `model.py`    # 模型文件
- `run_train.py` # 模型训练脚本
- `sentence_augment.py`  #关键词提取
- `smart_sampler.py`  # 自定义采样函数
- `data_loader.py`      # 自定义 PyTorch Dataset & DataLoader
- `train.py`          #训练函数
