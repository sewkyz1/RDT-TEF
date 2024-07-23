from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
# import vot
import sys
import time
import os
import numpy as np
# from lib.test.tracker.mixformer_vit_online import MixFormerOnline
from lib.test.tracker.mixformer_cvt_online import MixFormerOnline


prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)


prj_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/AR'
if prj_path not in sys.path:
    sys.path.append(prj_path)


from external.AR.pytracking.ARcm_seg import ARcm_seg
from external.AR.pytracking.vot20_utils import *

# import lib.test.parameter.mixformer_vit_online as vot_params
import lib.test.parameter.mixformer_cvt_online as vot_params

# new
import external.AR.pytracking.VOT2020_super_only_mask_384_HP.vot as vot


class MIXFORMER_ALPHA_SEG(object):
    def __init__(self, tracker,
                 refine_model_name='ARcm_coco_seg', threshold=0.6):
        self.THRES = threshold
        self.tracker = tracker
        '''create tracker'''
        '''Alpha-Refine'''
        # project_path = os.path.join(os.path.dirname(__file__), '..', '..')
        project_path = os.path.join('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/AR/pytracking/VOT2020_super_only_mask_384_HP', '..', '..')
        refine_root = os.path.join(project_path, 'ltr/checkpoints/ltr/ARcm_seg/')
        refine_path = os.path.join(refine_root, refine_model_name)
        '''2020.4.25 input size: 384x384'''
        self.alpha = ARcm_seg(refine_path, input_sz=384)

    def initialize(self, image, mask):
        region = rect_from_mask(mask)
        # init_info = {'init_bbox': region}
        # self.tracker.initialize(image, init_info)

        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}
        self.tracker.initialize(image, init_info)
        '''initilize refinement module for specific video'''
        self.alpha.initialize(image, np.array(gt_bbox_np))

    def track(self, img_RGB):
        '''TRACK'''
        '''base tracker'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        '''Step2: Mask report'''
        pred_mask, search, search_mask = self.alpha.get_mask(img_RGB, np.array(pred_bbox), vis=True)
        final_mask = (pred_mask > self.THRES).astype(np.uint8)
        return final_mask, 1


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)



refine_model_name = 'ARcm_coco_seg_only_mask_384'
# params = vot_params.parameters("baseline", model="mixformer_online_22k.pth.tar")  # yuan
params = vot_params.parameters("baseline", model="mixformer_vit_base_online.pth.tar")
print(params)
# params = vot_params.parameters("baseline")
mixformer = MixFormerOnline(params, "VOT20")
tracker = MIXFORMER_ALPHA_SEG(tracker=mixformer, refine_model_name=refine_model_name)
# handle = vot.VOT("mask")  # TIAOSHI KA ZAI ZHE
handle = vot.VOT("mask","rgbd")  # TIAOSHI KA ZAI ZHE
selection = handle.region()
imagefile_rgb, imagefile_d = handle.frame()

# print(imagefile)

if not imagefile_rgb:
    sys.exit(0)

if not imagefile_d:
    sys.exit(0)

image_rgb = cv2.cvtColor(cv2.imread(imagefile_rgb), cv2.COLOR_BGR2RGB)  # Right
image_d = cv2.cvtColor(cv2.imread(imagefile_d), cv2.COLOR_BGR2RGB)  # Right

# mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
mask = make_full_size(selection, (image_rgb.shape[1], image_rgb.shape[0]))

tracker.H = image_rgb.shape[0]
tracker.W = image_rgb.shape[1]

tracker.initialize(image_rgb, mask)

while True:
    imagefile_rgb, imagefile_d = handle.frame()
    if not imagefile_rgb:
        break
    image = cv2.cvtColor(cv2.imread(imagefile_rgb), cv2.COLOR_BGR2RGB)  # Right
    region, confidence = tracker.track(image)
    handle.report(region, confidence)
