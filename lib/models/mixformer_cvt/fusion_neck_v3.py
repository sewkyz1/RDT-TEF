import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .fusion.attention_block import AttentionBlock
from .fusion.cross_attention_block import CrossAttentionBlock

# change in 1110 # v3
# cat in token_dim 1*400*768  decoder use x.transpose(0,2,1)

class FUSION(nn.Module):
    # def __init__(self,  encoder, decoder):
    def __init__(self,  encoder, decoder_template, decoder_search):
        
        super(FUSION, self).__init__()
        self.layers1 = nn.ModuleList(encoder)
        # self.layers2 = nn.ModuleList(decoder)
        self.layers2 = nn.ModuleList(decoder_template)
        self.layers3 = nn.ModuleList(decoder_search)

    def forward(self, x_rgb, x_depth, xpos_cat, xpos):
        '''
            Args:
                z (torch.Tensor): (B, L_z, C), template image feature tokens
                x (torch.Tensor): (B, L_x, C), search image feature tokens
                z_pos (torch.Tensor | None): (1 or B, L_z, C), optional positional encoding for z
                x_pos (torch.Tensor | None): (1 or B, L_x, C), optional positional encoding for x
            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    (B, L_z, C): template image feature tokens
                    (B, L_x, C): search image feature tokens
        '''                
        x_cat = torch.cat((x_rgb, x_depth),dim = 1)  # patch_len: 384+384
        for attention in self.layers1:
            x_cat = attention(x_cat, xpos_cat)
        # for attention in self.layers1:
        #     x_depth = attention(x_depth, xpos)
        
        x = 0.5*x_rgb + 0.5*x_depth  # gai ke xue xi
        
        # for attention in self.layers2:
        #     x = attention(x, x_cat, xpos)
        if x.shape[2]==8:
            for attention in self.layers2:
                x = attention(x, x_cat, xpos)
        elif x.shape[2]==20:
            for attention in self.layers3:
                x = attention(x, x_cat, xpos)
        # else:
            
            
        return x
            
def get_rgbd_fusion(config, **kwargs):
    fusion_spec = config.MODEL.FUSION
    drop_path_allocator1 = DropPath(0.1)
    drop_path_allocator2 = nn.Identity()
    num_encoders = fusion_spec.num_encoders
    num_decoders = fusion_spec.num_decoders  
    dim = fusion_spec.dim*2
    num_heads = fusion_spec.num_heads
    mlp_ratio = fusion_spec.mlp_ratio
    qkv_bias = fusion_spec.qkv_bias
    drop_rate = fusion_spec.drop_rate
    attn_drop_rate = fusion_spec.attn_drop_rate
    
    # for test
    print(fusion_spec)



    encoder = []
    # decoder = []
    decoder_template = []
    decoder_search = []
    
    for index_of_encoder in range(num_encoders):       
        encoder.append(AttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator1))
        # drop_path_allocator.increase_depth()

    # for index_of_encoder in range(num_decoders):
    #     decoder.append(CrossAttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator2))
    #     # drop_path_allocator.increase_depth()
    
    for index_of_encoder in range(num_decoders):  # index_of_encoder ->decoder
        decoder_template.append(CrossAttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator2, wh=64))
        # drop_path_allocator.increase_depth()

    for index_of_encoder in range(num_decoders):  # index_of_encoder ->decoder
        decoder_search.append(CrossAttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator2, wh=400))
    
    # Fusion_netork = FUSION(encoder, decoder)
    Fusion_netork = FUSION(encoder, decoder_template, decoder_search)

    return Fusion_netork