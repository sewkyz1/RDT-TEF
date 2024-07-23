import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .fusion.attention_block import AttentionBlock
from .fusion.cross_attention_block import CrossAttentionBlock
# change in 0206 # fusion_neck_v5plus3 only encoder
# cat in patch_dim 1*800*384   use template in search's encoder and decoder
import numpy as np
import cv2
from torchvision.transforms import Resize 

class FUSION(nn.Module):
    def __init__(self,  encoder, decoder):
        
        super(FUSION, self).__init__()
        self.layers1 = nn.ModuleList(encoder)
        # self.layers2 = nn.ModuleList(decoder)
        # self.f_l = nn.Conv2d(768, 384, kernel_size=1, bias=False)
        # self.f_l = nn.Conv2d(768, 384, kernel_size=1)

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
        # x_cat = torch.cat((x_rgb, x_depth),dim = 1)  # patch_len: 384+384
        
        # def draw_hot_map(feature_input):
        #     aaaa = feature_input.mean(dim=1)
        #      #hot_map = F.softmax(aaaa.cpu().reshape([400]),dim=0).reshape([1,1,20,20])
        #     hot_map = aaaa.cpu().reshape([1,1,20,20])
        #     hot_map = hot_map-hot_map.min()
        #     trans = Resize([320,320])
        #     # hot_map=hot_map/hot_map.max()
        #     hot_map = trans(hot_map).reshape([320,320])
        #     hot_map=hot_map/(hot_map.max())

        #     hot_map = 1-hot_map

        #     hot = np.uint8(255 * hot_map)
        #     hot_output = cv2.applyColorMap(hot, cv2.COLORMAP_JET)
        #     # cv2.imwrite('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/hot_tl.jpg', hot_tl)
        #     return hot_output

        x_rgb_ori = x_rgb
        x_depth_ori = x_depth

        # for attention in self.layers1:  
        #     x_cat = attention(x_cat, xpos_cat)
        attention_map = []
        # if x_rgb_ori.shape[3]==20:
        #     attention_map.append(draw_hot_map(x_rgb))

        for attention in self.layers1:  
            x_rgb, x_depth = attention(x_rgb, x_depth, xpos)
            # if x_rgb_ori.shape[3]==20:
            #     attention_map.append(draw_hot_map(x_rgb))

        # aaaa = x_rgb.mean(dim=1)
        # # hot_map = F.softmax(aaaa.cpu().reshape([400]),dim=0).reshape([1,1,20,20])
        # hot_map = aaaa.cpu().reshape([1,1,20,20])
        # hot_map = hot_map-hot_map.min()
        # trans = Resize([320,320])
        # # hot_map=hot_map/hot_map.max()
        # hot_map = trans(hot_map).reshape([320,320])
        # hot_map=hot_map/(hot_map.max())
        # hot = np.uint8(255 * hot_map)
        # hot_tl = cv2.applyColorMap(hot, cv2.COLORMAP_JET)
        # cv2.imwrite('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/hot_map_e2.jpg', hot_tl)






        # for attention in self.layers1:  # ???????????
        #     x_depth = attention(x_depth, xpos)
        
        # x = 0.5*x_rgb + 0.5*x_depth  # gai ke xue xi
        # x = 0.5*x_rgb_ori + 0.5*x_depth_ori  # gai ke xue xi

        x = 0.5*x_rgb_ori + 0.5*x_depth_ori  # yuan
        # x = 0.5*x_rgb_ori + 0.5*x_depth_ori + self.f_l(torch.cat((x_rgb_ori, x_depth_ori), dim=1))
        # x = 0.8*x_rgb_ori + 0.2*x_depth_ori
        # x = x_rgb_ori
        # if x_rgb_ori.shape[3]==20:    
        #     attention_map.append(draw_hot_map(0.2*x_rgb_ori + 0.8*x_depth_ori))


        # x = self.f_l(torch.cat((x_rgb_ori, x_depth_ori), dim=1))
        # x = x + self.f_l(torch.cat((x_rgb_ori, x_depth_ori), dim=1))

        # for attention in self.layers2:
        #     x = attention(x, x_cat, xpos)
        # for attention in self.layers2:
        #     x = attention(x, x_rgb, x_depth, xpos)
            # if x_rgb_ori.shape[3]==20:    
            #     attention_map.append(draw_hot_map(x))

        # encoder_output = x_rgb   
        # return x, attention_map
        return x  # yuan
            
def get_rgbd_fusion(config, **kwargs):
    fusion_spec = config.MODEL.FUSION
    drop_path_allocator1 = DropPath(0.1)
    drop_path_allocator2 = nn.Identity()
    num_encoders = fusion_spec.num_encoders
    num_decoders = fusion_spec.num_decoders
      
    # dim = fusion_spec.dim*2
    dim = fusion_spec.dim

    num_heads = fusion_spec.num_heads
    mlp_ratio = fusion_spec.mlp_ratio
    qkv_bias = fusion_spec.qkv_bias
    drop_rate = fusion_spec.drop_rate
    attn_drop_rate = fusion_spec.attn_drop_rate
    
    # for test
    print(fusion_spec)



    encoder = []
    decoder = []
    
    for index_of_encoder in range(num_encoders):       
        encoder.append(AttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator1))
        # drop_path_allocator.increase_depth()
    
    # for index_of_encoder in range(num_decoders):
    #     decoder.append(CrossAttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator2))
    #     # drop_path_allocator.increase_depth()
    
    Fusion_netork = FUSION(encoder, decoder)

    return Fusion_netork



