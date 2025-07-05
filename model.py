import torch.nn as nn

class CrossModalProjection(nn.Module):
    def __init__(self, img_input_dim=512, txt_input_dim=768, projection_dim=256, dropout=0.1):
        super().__init__()

        # 图像分支
        self.image_norm = nn.LayerNorm(img_input_dim)
        self.image_fc1 = nn.Linear(img_input_dim, 1024)
        self.image_fc2 = nn.Linear(1024, projection_dim)
        self.image_proj_shortcut = nn.Linear(img_input_dim, projection_dim)  # 残差映射

        # 文本分支
        self.text_norm = nn.LayerNorm(txt_input_dim)
        self.text_fc1 = nn.Linear(txt_input_dim, 1024)
        self.text_fc2 = nn.Linear(1024, projection_dim)
        self.text_proj_shortcut = nn.Linear(txt_input_dim, projection_dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.output_norm = nn.LayerNorm(projection_dim)

    def forward(self, image_feat, text_feat):
        # 图像路径
        x_img = self.image_norm(image_feat)
        x_img = self.activation(self.image_fc1(x_img))
        x_img = self.dropout(x_img)
        x_img = self.image_fc2(x_img)
        shortcut_img = self.image_proj_shortcut(image_feat)
        img_out = self.output_norm(x_img + shortcut_img)  # 残差连接 + 归一化

        # 文本路径
        x_txt = self.text_norm(text_feat)
        x_txt = self.activation(self.text_fc1(x_txt))
        x_txt = self.dropout(x_txt)
        x_txt = self.text_fc2(x_txt)
        shortcut_txt = self.text_proj_shortcut(text_feat)
        txt_out = self.output_norm(x_txt + shortcut_txt)

        return img_out, txt_out
