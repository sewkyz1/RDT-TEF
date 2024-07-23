from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
import vot
import sys
import time
import os
import numpy as np

# # for test
# prj_path = os.path.join('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz')
# if prj_path not in sys.path:
#     sys.path.append(prj_path)


# prj_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/AR'
# if prj_path not in sys.path:
#     sys.path.append(prj_path)


# from lib.test.tracker.mixformer_vit_online import MixFormerOnline
from lib.test.tracker.mixformer_cvt_online import MixFormerOnline

from pytracking.ARcm_seg import ARcm_seg
from pytracking.vot20_utils import *

# import lib.test.parameter.mixformer_vit_online as vot_params
import lib.test.parameter.mixformer_cvt_online as vot_params


# from vot import Rectangle
# from pytracking.VOT2020_super_only_mask_384_HP.vot import Rectangle
from vot import Rectangle

# 
# rectangle
# 0922 rgbd
class MIXFORMER_ALPHA_SEG(object):
    def __init__(self, tracker):
        
        self.tracker = tracker
        # '''create tracker'''
        # '''Alpha-Refine'''
        # project_path = os.path.join(os.path.dirname(__file__), '..', '..')
        # refine_root = os.path.join(project_path, 'ltr/checkpoints/ltr/ARcm_seg/')
        # refine_path = os.path.join(refine_root, refine_model_name)
        # '''2020.4.25 input size: 384x384'''
        # self.alpha = ARcm_seg(refine_path, input_sz=384)

    def initialize(self, image_rgb, image_d, region):
        
        # 111region = rect_from_mask(mask)

        # init_info = {'init_bbox': region}
        # self.tracker.initialize(image, init_info)

        self.H, self.W, _ = image_rgb.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}
        self.tracker.initialize(image_rgb, image_d, init_info)
        '''initilize refinement module for specific video'''
        # print('init alpha')
        # self.alpha.initialize(image_rgb, np.array(gt_bbox_np))
        # print('alpha over')

    def track(self, img_RGB, img_D):
        '''TRACK'''
        '''base tracker'''
        outputs = self.tracker.track(img_RGB, img_D)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['target_score']
        # '''Step2: Mask report'''
        # pred_mask, search, search_mask = self.alpha.get_mask(img_RGB, np.array(pred_bbox), vis=True)
        # final_mask = (pred_mask > self.THRES).astype(np.uint8)
        # return final_mask, 1



        # test confidence
        # print(pred_bbox)
        # print(type(pred_bbox[0]))
        # print(type(pred_bbox))
        # print(pred_score)
        # print(afaufg)
        return pred_bbox, pred_score
        # return pred_bbox, 1
    





# def make_full_size(x, output_sz):
#     '''
#     zero-pad input x (right and down) to match output_sz
#     x: numpy array e.g., binary mask
#     output_sz: size of the output [width, height]
#     '''
#     if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
#         return x
#     pad_x = output_sz[0] - x.shape[1]
#     if pad_x < 0:
#         x = x[:, :x.shape[1] + pad_x]
#         # padding has to be set to zero, otherwise pad function fails
#         pad_x = 0
#     pad_y = output_sz[1] - x.shape[0]
#     if pad_y < 0:
#         x = x[:x.shape[0] + pad_y, :]
#         # padding has to be set to zero, otherwise pad function fails
#         pad_y = 0
#     return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)


def get_image_d(imagefile_d):
    dp = cv2.imread(imagefile_d, -1)
    max_depth = min(np.median(dp)*3,10000)
    dp[dp>max_depth] = max_depth
    dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dp = np.asarray(dp, dtype=np.uint8)
    colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)
    return colormap




# refine_model_name = 'ARcm_coco_seg_only_mask_384'
# params = vot_params.parameters("baseline", model="mixformer_online_22k.pth.tar")  # yuan
# params = vot_params.parameters("baseline", model="networktest_cvt.pth")  # cvt_base
# params = vot_params.parameters("baseline", model="MixFormerOnlineScore_ep0028.pth.tar")
# params = vot_params.parameters("baseline", model="MixFormerOnlineScore_ep0190.pth.tar")
params = vot_params.parameters("baseline", model="MixFormerOnlineScore_ep0036.pth.tar")
# params = vot_params.parameters("baseline_large", model="mixformerL_online_22k.pth.tar")  
# params = vot_params.parameters("baseline", model="mixformer_vit_base_online.pth.tar")
# params = vot_params.parameters("baseline_large", model="mixformer_vit_large_online.pth.tar")
# params = vot_params.parameters("baseline")



print(params)
# mixformer = MixFormerOnline(params, "VOT20")
# mixformer = MixFormerOnline(params, "VOT20")  # short 
# mixformer = MixFormerOnline(params, "VOT20RGBD")  # long 
mixformer = MixFormerOnline(params, "CDTB")
# tracker = MIXFORMER_ALPHA_SEG(tracker=mixformer, refine_model_name=refine_model_name)
tracker = MIXFORMER_ALPHA_SEG(tracker=mixformer)
# handle = vot.VOT("mask","rgbd")
handle = vot.VOT("rectangle","rgbd")
selection = handle.region()
# imagefile = handle.frame()
imagefile_rgb, imagefile_d = handle.frame()

init_box = [selection.x, selection.y, selection.width, selection.height]

# print(imagefile)

if not imagefile_rgb:
    sys.exit(0)

if not imagefile_d:
    sys.exit(0)


# if not imagefile:
#     sys.exit(0)

# image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # 
image_rgb = cv2.cvtColor(cv2.imread(imagefile_rgb), cv2.COLOR_BGR2RGB)  # Right
# image_d = cv2.cvtColor(cv2.imread(imagefile_d), cv2.COLOR_BGR2RGB)  # Right
image_d = get_image_d(imagefile_d)



# mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
# mask = make_full_size(selection, (image.shape[1], image.shape[0]))
# mask = make_full_size(selection, (image_rgb.shape[1], image_rgb.shape[0]))

tracker.H = image_rgb.shape[0]
tracker.W = image_rgb.shape[1]

# tracker.H = image.shape[0]
# tracker.W = image.shape[1]
print('no initialize')
# tracker.initialize(image_rgb, image_d, mask)
tracker.initialize(image_rgb, image_d, init_box)
print('initialize over')

while True:
    # imagefile = handle.frame()
    # if not imagefile:
    #     break
    # image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    imagefile_rgb, imagefile_d = handle.frame()
    if not imagefile_rgb:
        break
    if not imagefile_d:
        break
    image_rgb = cv2.cvtColor(cv2.imread(imagefile_rgb), cv2.COLOR_BGR2RGB)  # Right
    # image_d = cv2.cvtColor(cv2.imread(imagefile_d), cv2.COLOR_BGR2RGB)  # Right
    image_d = get_image_d(imagefile_d)
    print('no track')
    region, confidence = tracker.track(image_rgb, image_d)
    # print(type(region))
    # print(region[0], region[1], region[2], region[3])
    selection = Rectangle(region[0], region[1], region[2], region[3])
    print(type(selection))
    # print(region[0], region[1], region[2], region[3])
    print('track over')
    # print(affasf)
    handle.report(selection, confidence)
