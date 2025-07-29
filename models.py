# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from vision_transformer import VisionTransformer
from visiontransformer_Text import VisionTransformerWithText
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F


__all__ = [
    'tikd_tiny_patch16_224', 'tikd_small_patch16_224', 'tikd_base_patch16_224',
    'tikd_tiny_distilled_patch16_224', 'tikd_small_distilled_patch16_224',
    'tikd_base_distilled_patch4_32', 'tikd_base_patch16_384',
    'tikd_base_distilled_patch16_384',
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, text_encoder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.text_features = None

        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.text_decoder = nn.Linear(self.embed_dim, self.embed_dim)

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def set_text_features(self, text_features):
        if text_features.dim() != 2 or text_features.size(1) != self.embed_dim:
            raise ValueError(
                f"Expected text features of shape [batch_size, {self.embed_dim}], but got {text_features.shape}")
        self.text_features = text_features
        print("text_features shape:", text_features.shape)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        alignment_loss = 0.0
        if self.text_features is not None:
            text_features_expanded = self.text_features[:B]
            text_features_expanded = text_features_expanded.unsqueeze(1).expand(-1, x.size(1), -1)
            cos_sim = F.cosine_similarity(x, text_features_expanded, dim=-1)
            cos_sim = torch.clamp(cos_sim, 0, 1)
            x_d = cos_sim.unsqueeze(-1) * x

            xdecode = self.text_decoder(x_d)

            for blk in self.blocks:
                x = blk(x, text_features=xdecode)

            alignment_loss = torch.mean(1 - cos_sim)

        return x[:, 0], x[:, 1], alignment_loss

    def forward(self, x):
        x, x_dist, alignment_loss = self.forward_features(x)

        x = self.head(x)
        x_dist = self.head_dist(x_dist)

        if self.training:
            return x, x_dist, alignment_loss
        else:
            return (x + x_dist) / 2, alignment_loss



@register_model
def tikd_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def tikd_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def tikd_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def tikd_base_patch4_32(pretrained=False, **kwargs):
    model = VisionTransformerWithText(
        patch_size=4, embed_dim=512, depth=5, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def tikd_base_patch4_32(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=4, embed_dim=512, depth=5, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model
def tikd_small_patch4_32(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=4, embed_dim=256, depth=5, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def tikd_tiny_patch4_32(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=4, embed_dim=128, depth=5, num_heads=2, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def tikd_base_distilled_patch4_32(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=4, embed_dim=512, depth=5, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def tikd_small_distilled_patch4_32(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=4, embed_dim=256, depth=5, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def tikd_tiny_distilled_patch4_32(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=4, embed_dim=128, depth=5, num_heads=2, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def tikd_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def tikd_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
