import torch
import torch.nn as nn
from timm.models.layers import DropPath 

from .self_attention import Attention
# change in 1102 # v2
# cat in patch_dim 1*800*384

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): (B, L, C), input tensor
            Returns:
                torch.Tensor: (B, L, C), output tensor
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(AttentionBlock, self).__init__()
        self.norm1 = norm_layer(dim)         
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)

        self.drop_path = drop_path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_rgb, x_depth, pos):
        # B, C, H, W = x.shape
        B, C, H, W = x_rgb.shape
        # x = x.view(B, C, H*W).permute(0, 2, 1)
        x_rgb = x_rgb.view(B, C, H*W).permute(0, 2, 1)
        x_depth = x_depth.view(B, C, H*W).permute(0, 2, 1)

        x = torch.cat((x_rgb, x_depth),dim = 1)  # x: x_cat 1*800*384
        pos = torch.cat((pos, pos),dim = 1)

        temp1 = self.norm1(x)  #  linear 768 -> 384
        temp2 = self.attn(temp1,pos)
        temp = self.drop_path(temp2)
        x = x + temp
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x_rgb = x[:, 0:H*W]
        x_depth = x[:, H*W:]

        # x = x.permute(0,2,1)
        # x = x.view(B, C, H, W)
        x_rgb = x_rgb.permute(0, 2, 1)
        x_rgb = x_rgb.view(B, C, H, W)
        x_depth = x_depth.permute(0, 2, 1)
        x_depth = x_depth.view(B, C, H, W)

        return x_rgb, x_depth
    
