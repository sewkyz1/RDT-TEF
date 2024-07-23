import torch
import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial

# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg
# import math
# v1

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_ = nn.Linear(384, 384, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(384, 384)
        self.proj_drop = nn.Dropout(proj_drop)

        self.act = nn.GELU()

 
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x, x_cat, pos, pos_cat):
        B, N, C = x.shape
        q = x + pos
        q = self.q_(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = x_cat

        k = kv + pos_cat
        k = self.k(k).reshape(B, -1, int(self.num_heads), C// self.num_heads).permute(0, 2, 1, 3)
        v = self.v(kv).reshape(B, -1, int(self.num_heads), C// self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
