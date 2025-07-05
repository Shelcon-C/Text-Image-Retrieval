import torch
LOCAL_BERT_PATH = "./models/bert-base-uncased"
TOKEN_FILE = "./data/flickr30k.token" 
BERT_PATH = "./models/bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = "./data/flickr30k-images"       
IMG_EMB_DIR = "./data/image_embeddings/"