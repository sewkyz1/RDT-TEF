# from torchsummary import summary
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import numpy as np
from thop import profile, clever_format

# v1
from fusion_v1_yuan.attention_block import AttentionBlock  
from fusion_v1_yuan.cross_attention_block import CrossAttentionBlock
dim = 768  # v1

# # v2
# from fusion_v2.attention_block import AttentionBlock  
# from fusion_v2.cross_attention_block import CrossAttentionBlock
# dim = 384  # v2

# fusion_spec = {'num_encoders': 4, 'num_decoders': 2, 'dim': 384, 'num_heads': 8, 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.1, 'attn_drop_rate': 0}
drop_path_allocator1 = DropPath(0.1)
drop_path_allocator2 = nn.Identity()
num_encoders = 4
num_decoders = 2


num_heads = 8
mlp_ratio = 4
qkv_bias = True
drop_rate = 0.1
attn_drop_rate = 0

encoder = []
decoder = []

# for index_of_encoder in range(num_encoders):       
#     encoder.append(AttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator1))
#     # drop_path_allocator.increase_depth()

# for index_of_encoder in range(num_decoders):
#     decoder.append(CrossAttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator2))
#     # drop_path_allocator.increase_depth()

# net = FUSION(encoder, decoder)
net = AttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator1).eval()

# v1
# all
input_tensor = torch.randn(1,768,20,20)
pos = torch.randn(1,400,768)
# output = net.forward(input_tensor,pos)
flops, params = profile(net, inputs=(input_tensor,pos))  # 2.834G 7.088M
# attn
input_tensor = torch.randn(1,400,768)
pos = torch.randn(1,400,768)
flops, params = profile(net.attn, inputs=(input_tensor,pos))  # 0.943718G 2.362M
# mlp
input_tensor = torch.randn(1,400,768)
flops, params = profile(net.mlp, inputs=(input_tensor))  # 1.887G 4.722M

# # v2
# # all
# input_tensor_1 = torch.randn(1,384,20,20)
# input_tensor_2 = torch.randn(1,384,20,20)
# pos = torch.randn(1,400,384)
# output = net.forward(input_tensor_1,input_tensor_2,pos)
# flops, params = profile(net, inputs=(input_tensor_1,input_tensor_2,pos))  # 1.418G 1.774M
# # attn
# input_tensor = torch.randn(1,800,384)
# pos = torch.randn(1,800,384)
# flops, params = profile(net.attn, inputs=(input_tensor,pos))  # 0.471859G 0.591360M
# # mlp
# input_tensor = torch.randn(1,800,384)
# flops, params = profile(net.mlp, inputs=(input_tensor))  # 0.943718G 1.182M


flops, params = clever_format([flops, params], "%.3f")

print("FLOPs: %s" %(flops))
print("params: %s" %(params))