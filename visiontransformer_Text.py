import torch
import torch.nn as nn
import torch.nn.functional as F
from vision_transformer import VisionTransformer
from timm.models.layers import trunc_normal_

class VisionTransformerWithText(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, text_encoder=None):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                         qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, hybrid_backbone=hybrid_backbone, norm_layer=norm_layer)
        
        self.text_encoder = text_encoder  # 用于文本特征的编码器

        # 解码器，用于将文本特征映射到与视觉特征相同的维度
        self.text_decoder = nn.Linear(self.embed_dim, self.embed_dim)

        # 初始化文本特征为空
        self.text_features = None

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        print(f"Embed dimension: {self.embed_dim}")
        print(f"Text decoder initialized: {self.text_decoder}")  # 确认文本投影层是否初始化成功

    def set_text_features(self, text_features):
        """
        设置文本特征，确保它们的维度正确
        """
        if text_features.dim() != 2 or text_features.size(1) != self.embed_dim:
            raise ValueError(f"Expected text features of shape [batch_size, {self.embed_dim}], but got {text_features.shape}")
        self.text_features = text_features

    def forward_features(self, x, text_features=None):
        B = x.shape[0]
        x = self.patch_embed(x)  # 将图像分块并映射到特征空间

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # 将cls_token附加到图像特征上
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 通过Transformer块进行处理
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)  # 归一化

        # 如果有文本特征，将其解码并与视觉特征进行融合
        if text_features is not None:
            text_features_expanded = text_features.unsqueeze(1).expand(-1, x.size(1), -1)  # 扩展文本特征
            text_features_decoded = self.text_decoder(text_features_expanded)  # 解码文本特征
            # 计算图像和文本特征的加权表示
            cos_sim = F.cosine_similarity(x, text_features_decoded, dim=-1)  # 计算余弦相似度
            cos_sim = torch.clamp(cos_sim, 0, 1)  # 限制在[0, 1]范围内
            x = cos_sim.unsqueeze(-1) * x  # 加权视觉特征

        return x[:, 0]  # 返回CLS token的输出

    def forward(self, x, text_features=None):
        x = self.forward_features(x, text_features)  # 获取视觉特征
        x = self.head(x)  # 通过分类头进行分类
        return x
