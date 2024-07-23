import torch

# -----------------------------------------------------
# 
# trans weight
#
# -----------------------------------------------------
# a = torch.load('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/save/models/mixformer_online_22k.pth.tar', map_location='cpu')
a = torch.load('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/save/models/baseline_choose_in_1027/MixFormerOnlineScore_ep0005.pth.tar', map_location='cpu')
b = a['net']
# b = a
c = torch.load('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/save/models/MixFormerOnlineScore_ep0020.pth.tar', map_location='cpu')
c = c['net']
# trans
# for key_net, value_net in list(b.items()):
#     name_split = key_net.split('.')
#     if 'backbone' in name_split:
#         name_join = '.'.join(name_split[1:])
#         print(name_join)
#         if c[name_join].shape == value_net.shape:
#             c[name_join] = value_net
#         else:
#             print('error')
#             print(c[name_join].shape, value_net.shape)
#             break
# torch.save(c, '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/backbone_d_cvt_1027_trans.pth')
# # test
# for key_net, value_net in list(c.items()):
#     print(key_net)

# test
i = 0
for key_net, value_net in list(b.items()):
    name_split = key_net.split('.')
    # if 'fusion' in name_split:
    #     i = i + 1
    if 'score_branch' in name_split:
        print(key_net)
        name_join = '.'.join(name_split[1:])
        name_d = '.'.join(['backbone_d', name_join])
        if b[key_net].equal(c[key_net]):
            # if c[key_net].equal(c[name_d]):
            #     dadad = 1
            # else:
            #     print('error1',key_net)
            #     break
            dadad = 1
        else:
            print('error2',key_net)
            break

dfadfa = 1
    

# -----------------------------------------------------
# 
# test weight
#
# -----------------------------------------------------

# a = torch.load('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/networktest_cvt.pth', map_location='cpu')
# for key_net, value_net in list(a.items()):
#     name_split = key_net.split('.')
#     if 'backbone' in name_split:
#         name_split[0] = 'backbone_d'
#         name_join = '.'.join(name_split)
#         print(name_join)
#         if a[name_join].shape == value_net.shape:
#             # if a[name_join] == value_net:
#             if a[name_join].equal(value_net):
#                 d = 1
#             else:
#                 print('error')
#                 print(key_net)
#                 print(a[name_join])
#                 break
#         else:
#             print('error')
#             print(a[name_join].shape, value_net.shape)
#             break

# -----------------------------------------------------
# 
# test weight2
#
# -----------------------------------------------------
# a = torch.load('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/save/models/MixFormerOnlineScore_ep0190.pth.tar', map_location='cpu')
# A = a['net']
# c = torch.load('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/save/models/MixFormerOnlineScore_ep0030.pth.tar', map_location='cpu')
# C = c['net']


# for key_net, value_net in list(A.items()):
#     print(key_net)
#     # if C[key_net].equal(value_net)==False:
#     #     print(key_net)


# g = 1