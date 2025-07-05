from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from data_loader import CrossModalDataset
from dataset_augmented import AugmentedKeywordDataset

def get_balanced_loader(
    orig_img_dir,
    orig_txt_dir,
    aug_txt_file,
    img_emb_dir,
    aug_txt_emb_dir,      
    device="cuda",
    batch_size=64,
    ratio_orig_to_aug=1.0,
    total_samples=None
):
    """
    构建 DataLoader，混合原始数据和增强数据，按指定比率采样。

    参数:
        orig_img_dir: 原始图像特征目录（.npy）
        orig_txt_dir: 原始文本特征目录（.npy）
        aug_txt_file: 增强文本对文件，格式：img.jpg \t keyword
        img_emb_dir: 图像特征路径（原始和增强共用）
        aug_txt_emb_dir: 增强文本特征的 .npy 文件路径
        device: 设备（用于张量加载）
        batch_size: 每批样本数
        ratio_orig_to_aug: 原始数据 : 增强数据 的采样频率比

    返回:
        PyTorch DataLoader
    """
    assert ratio_orig_to_aug > 0, "ratio_orig_to_aug 必须大于 0"

    # 加载原始和增强 Dataset
    orig_dataset = CrossModalDataset(orig_img_dir, orig_txt_dir)
    aug_dataset = AugmentedKeywordDataset(
        pair_file=aug_txt_file,
        img_emb_dir=img_emb_dir,
        txt_emb_dir=aug_txt_emb_dir,
        device=device
    )

    len_orig = len(orig_dataset)
    len_aug = len(aug_dataset)
    
    combined_dataset = ConcatDataset([orig_dataset, aug_dataset])
    
    # 采样权重设置
    weight_orig = 1.0
    weight_aug = (len_orig / len_aug) / ratio_orig_to_aug
    weights = [weight_orig] * len_orig + [weight_aug] * len_aug

    num_samples = total_samples if total_samples is not None else len(weights)
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


    loader = DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler)

    return loader
