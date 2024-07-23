import torch
import torch.nn as nn
from .cross_attention import CrossAttention
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

class CrossAttentionBlock(nn.Module):
    # def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
    #              drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), wh=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(CrossAttentionBlock, self).__init__()
        self.norm1_q = norm_layer(384)
        # self.norm1_z = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        # self.attn = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.attn = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, wh)

        self.drop_path = drop_path
        self.norm2 = norm_layer(384)
        mlp_hidden_dim = int(384 * mlp_ratio)
        self.mlp = Mlp(in_features=384, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_cat, pos):
        '''x, z, x_pos, concatenated_pos_enc, concatenated_pos_enc_1, attn_pos_enc, attn_pos_enc_1
            Args:
                q (torch.Tensor): (B, L_q, C)
                kv (torch.Tensor): (B, L_kv, C)
                q_ape (torch.Tensor | None): (1 or B, L_q, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L_kv, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L_q, L_kv), untied positional encoding
            Returns:
                torch.Tensor: (B, L_q, C)
        '''
        pos_cat = torch.cat((pos, pos), dim=2)

        B, C, H, W = x_cat.shape
        x_cat = x_cat.view(B, C, H*W).permute(0, 2, 1)
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1_q(x), self.norm1_kv(x_cat), pos, pos_cat))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # q = q + self.drop_path(self.mlp(self.norm2(q))), attn_pos_enc, attn_pos_enc_1
        x = x.permute(0,2,1)
        x = x.view(B, C, H, W)
        return x
