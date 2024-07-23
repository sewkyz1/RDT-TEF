import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.models.mixformer_cvt.utils import FrozenBatchNorm2d
import numpy as np
import cv2
from torchvision.transforms import Resize 
import os

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            

            # -------------------------------draw
            # # import numpy as np
            # # import cv2
            # # from torchvision.transforms import Resize 
            # hot_map = F.softmax(score_map_tl.cpu().reshape([400]),dim=0).reshape([1,1,20,20])
            # # hot_map = score_map_tl.cpu().reshape([1,1,20,20])
            # # hot_map = hot_map-hot_map.min()
            # trans = Resize([320,320])
            # # hot_map=hot_map/hot_map.max()
            # hot_map_tl = trans(hot_map).reshape([320,320])
            # hot_map=hot_map_tl/hot_map_tl.max()
            # hot = np.uint8(255 * hot_map)
            # hot_tl = cv2.applyColorMap(hot, cv2.COLORMAP_JET)
            # # cv2.imwrite('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/hot_tl.jpg', hot_tl)

            # hot_map = F.softmax(score_map_br.cpu().reshape([400]),dim=0).reshape([1,1,20,20])
            # # trans = Resize([320,320])
            # hot_map_br = trans(hot_map).reshape([320,320])
            # hot_map=hot_map_br/hot_map_br.max()
            # hot = np.uint8(255 * hot_map)
            # hot_br = cv2.applyColorMap(hot, cv2.COLORMAP_JET)
            # # cv2.imwrite('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/hot_br.jpg', hot_br)


            # hot_map_tl = hot_map_tl*hot_map_br.max()/hot_map_tl.max()
            # hot_both = hot_map_tl + hot_map_br
            # hot_both=hot_both/hot_both.max()
            # hot_both = np.uint8(255 * hot_both)
            # hot_both = cv2.applyColorMap(hot_both, cv2.COLORMAP_JET)
            
            # numfile = len(os.listdir('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/cdtb_hotmap/map_both/robot_human_corridor_noocc_2'))
            # cv2.imwrite(f'/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/cdtb_hotmap/map_both/robot_human_corridor_noocc_2/{numfile+2}.jpg', hot_both)
            
                

            # draw
            # import numpy as np
            # import cv2
            # from torchvision.transforms import Resize 

            # class_score_map_x, class_score_map_y = score_map_tl, score_map_tl
            
            # x = class_score_map_x.mean(dim=2,keepdim=True)
            # y = class_score_map_y.mean(dim=3,keepdim=True)

            # x = F.softmax(x,dim=3)
            # # x = (x - x.min())/(x.max()-x.min())
            # y = F.softmax(y,dim=2)
            # # y = (y - y.min())/(y.max()-y.min())

            # trans_x = Resize([1,320])  # 320*320
            # trans_y = Resize([320,1])
            # class_score_map = trans_y(y)@trans_x(x)
            # # x = x.reshape([1,1,1,20])  # 20*20
            # # y = y.reshape([1,1,20,1])
            # # class_score_map = y@x

            # class_score_map = class_score_map*(1/class_score_map.max() - 0.0001)

            # N, C, H, W = class_score_map.shape
            # assert C == 1
            # score = class_score_map.view(N, H * W)
            # score = score.permute(1,0)
            # score = score.data[:,0].cpu().numpy()
            # pscore = score

            # best_idx = np.argmax(pscore)
            # hot = pscore.reshape(H, W)
            # hot = np.uint8(255 * hot)
            # hot_tl = cv2.applyColorMap(hot, cv2.COLORMAP_JET)
            # # cv2.imwrite('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/cdtb_hotmap/hot_tl.jpg', hot_tl)


            # class_score_map_x, class_score_map_y = score_map_br, score_map_br
            
            # x = class_score_map_x.mean(dim=2,keepdim=True)
            # y = class_score_map_y.mean(dim=3,keepdim=True)

            # x = F.softmax(x,dim=3)
            # # x = (x - x.min())/(x.max()-x.min())
            # y = F.softmax(y,dim=2)
            # # y = (y - y.min())/(y.max()-y.min())

            # trans_x = Resize([1,320])  # 320*320
            # trans_y = Resize([320,1])
            # class_score_map = trans_y(y)@trans_x(x)
            # # x = x.reshape([1,1,1,20])  # 20*20
            # # y = y.reshape([1,1,20,1])
            # # class_score_map = y@x

            # class_score_map = class_score_map*(1/class_score_map.max() - 0.0001)

            # N, C, H, W = class_score_map.shape
            # assert C == 1
            # score = class_score_map.view(N, H * W)
            # score = score.permute(1,0)
            # score = score.data[:,0].cpu().numpy()
            # pscore = score

            # best_idx = np.argmax(pscore)
            # hot = pscore.reshape(H, W)
            # hot = np.uint8(255 * hot)
            # hot_br = cv2.applyColorMap(hot, cv2.COLORMAP_JET)
            # # cv2.imwrite('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/cdtb_hotmap/hot_br.jpg', hot_br)

            # -----------------------------------

            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz  # yuan
            # return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, hot_tl, hot_br 

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y



class Pyramid_Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Pyramid_Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        self.adjust1_tl = conv(inplanes, channel // 2, freeze_bn=freeze_bn)
        self.adjust2_tl = conv(inplanes, channel // 4, freeze_bn=freeze_bn)

        self.adjust3_tl = nn.Sequential(conv(channel // 2, channel // 4, freeze_bn=freeze_bn),
                                        conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        conv(channel // 8, 1, freeze_bn=freeze_bn))
        self.adjust4_tl = nn.Sequential(conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        conv(channel // 8, 1, freeze_bn=freeze_bn))

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        self.adjust1_br = conv(inplanes, channel // 2, freeze_bn=freeze_bn)
        self.adjust2_br = conv(inplanes, channel // 4, freeze_bn=freeze_bn)

        self.adjust3_br = nn.Sequential(conv(channel // 2, channel // 4, freeze_bn=freeze_bn),
                                        conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        conv(channel // 8, 1, freeze_bn=freeze_bn))
        self.adjust4_br = nn.Sequential(conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        conv(channel // 8, 1, freeze_bn=freeze_bn))

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        x_init = x
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)

        #up-1
        x_init_up1 = F.interpolate(self.adjust1_tl(x_init), scale_factor=2)
        x_up1 = F.interpolate(x_tl2, scale_factor=2)
        x_up1 = x_init_up1 + x_up1

        x_tl3 = self.conv3_tl(x_up1)

        #up-2
        x_init_up2 = F.interpolate(self.adjust2_tl(x_init), scale_factor=4)
        x_up2 = F.interpolate(x_tl3, scale_factor=2)
        x_up2 = x_init_up2 + x_up2

        x_tl4 = self.conv4_tl(x_up2)
        score_map_tl = self.conv5_tl(x_tl4) + F.interpolate(self.adjust3_tl(x_tl2), scale_factor=4) + F.interpolate(self.adjust4_tl(x_tl3), scale_factor=2)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)

        # up-1
        x_init_up1 = F.interpolate(self.adjust1_br(x_init), scale_factor=2)
        x_up1 = F.interpolate(x_br2, scale_factor=2)
        x_up1 = x_init_up1 + x_up1

        x_br3 = self.conv3_br(x_up1)

        # up-2
        x_init_up2 = F.interpolate(self.adjust2_br(x_init), scale_factor=4)
        x_up2 = F.interpolate(x_br3, scale_factor=2)
        x_up2 = x_init_up2 + x_up2

        x_br4 = self.conv4_br(x_up2)
        score_map_br = self.conv5_br(x_br4) + F.interpolate(self.adjust3_br(x_br2), scale_factor=4) + F.interpolate(self.adjust4_br(x_br3), scale_factor=2)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_box_head(cfg):
    if cfg.MODEL.HEAD_TYPE == "MLP":
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        return mlp_head
    elif "CORNER" in cfg.MODEL.HEAD_TYPE:
        channel = getattr(cfg.MODEL, "HEAD_DIM", 384)
        freeze_bn = getattr(cfg.MODEL, "HEAD_FREEZE_BN", False)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD_TYPE == "CORNER":
            stride = 16
            feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride, freeze_bn=freeze_bn)
        elif cfg.MODEL.HEAD_TYPE == "CORNER_UP":
            stride = 4
            feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
            corner_head = Pyramid_Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                                   feat_sz=feat_sz, stride=stride, freeze_bn=freeze_bn)
        else:
            raise ValueError()
        return corner_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)
