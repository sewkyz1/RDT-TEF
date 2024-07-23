import torch
import torch.nn as nn
from timm.models.layers import DropPath 

from .self_attention import Attention
# yuan baseline # v1
# cat in token_dim 1*400*768

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

    def forward(self, x, pos):
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).permute(0, 2, 1)
        temp1 = self.norm1(x)
        temp2 = self.attn(temp1,pos)
        temp = self.drop_path(temp2)
        x = x + temp
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0,2,1)
        x = x.view(B, C, H, W)
        return x
    
