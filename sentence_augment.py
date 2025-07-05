from nltk.tokenize import word_tokenize
from nltk import pos_tag

def extract_keywords(sentence):
    """
    提取名词/动词关键词 + 一些短语组合
    """
    tokens = word_tokenize(sentence.lower())
    tagged = pos_tag(tokens)

    # 名词 + 动词作为关键词
    keywords = [word for word, tag in tagged if tag.startswith('NN') or tag.startswith('VB')]

    # 构造短语组合
    phrases = []
    for i in range(len(keywords)):
        for j in range(i+1, min(i+3, len(keywords))):
            phrases.append(f"{keywords[i]} {keywords[j]}")

    return list(set(keywords + phrases))

def process_token_file(token_path, save_path):
    """
    处理 Flickr30k 的 .token 文件，输出增强结果
    """
    with open(token_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    img2sent = {}
    for line in lines:
        img_id, caption = line.strip().split('\t')
        img_key = img_id.split('#')[0]
        if img_key not in img2sent:
            img2sent[img_key] = []
        img2sent[img_key].append(caption)

    # 写入增强结果
    with open(save_path, 'w', encoding='utf-8') as f:
        for img, captions in img2sent.items():
            for cap in captions:
                keywords = extract_keywords(cap)
                for kw in keywords:
                    f.write(f"{img}\t{kw}\n")

    print(f"[✓] 已保存增强数据到 {save_path}")


if __name__ == "__main__":
    TOKEN_PATH = "./data/flickr30k.token"
    SAVE_PATH = "./data/flickr30k_keywords_augmented.txt"

    process_token_file(TOKEN_PATH, SAVE_PATH)
