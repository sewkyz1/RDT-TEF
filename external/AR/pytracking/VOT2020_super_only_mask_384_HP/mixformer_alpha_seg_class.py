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

# from pytracking.ARcm_seg import ARcm_seg
from pytracking.refine_modules.refine_module import RefineModule
from pytracking.vot20_utils import *

# import lib.test.parameter.mixformer_vit_online as vot_params
import lib.test.parameter.mixformer_cvt_online as vot_params

from vot import Rectangle

# 1120 rgbd 
# ar rectangle
# mixformer_alpha_seg_class_arrectangle


class MIXFORMER_ALPHA_SEG(object):
    def __init__(self, tracker,
                 refine_model_name='ARcm_coco_seg', threshold=0.6):  # yuan: threshold=0.6
        self.THRES = threshold
        self.tracker = tracker
        '''create tracker'''
        '''Alpha-Refine'''
        project_path = os.path.join(os.path.dirname(__file__), '..', '..')
        # refine_root = os.path.join(project_path, 'ltr/checkpoints/ltr/ARcm_seg/')  # yuan
        refine_root = os.path.join(project_path, 'ltr/checkpoints/ltr/SEx_beta/')
        refine_path = os.path.join(refine_root, refine_model_name)
        print(refine_path)
        '''2020.4.25 input size: 384x384'''
        '''2023.11.20 input size: 256x256'''
        # self.alpha = ARcm_seg(refine_path, input_sz=384)  # yuan
        # self.alpha = RefineModule(refine_path, input_sz=384)  
        # self.alpha = get_ar(img, init_box, ar_path)
        self.alpha = get_ar(refine_path)

    # def initialize(self, image_rgb, image_d, mask):
    def initialize(self, image_rgb, image_d, region):
        # # test
        # print('mask')
        # print(mask)
        # print('mask.shape')
        # print(mask.shape)
        # cv2.imwrite('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/test_ar/inputmask.jpg',mask*255)
        # print(afuau)


        # # region = rect_from_mask(mask)  # mask to rectangle

        # init_info = {'init_bbox': region}
        # self.tracker.initialize(image, init_info)



        self.H, self.W, _ = image_rgb.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}
        self.tracker.initialize(image_rgb, image_d, init_info)
        '''initilize refinement module for specific video'''
        print('init alpha')
        # self.alpha.initialize(image_rgb, np.array(gt_bbox_np))  # yuan
        self.alpha.initialize(image_rgb, np.array(gt_bbox_np))
        print('alpha over')

    def track(self, img_RGB, img_D):
        '''TRACK'''
        '''base tracker'''
        outputs = self.tracker.track(img_RGB, img_D)
        pred_bbox = outputs['target_bbox']
        '''Step2: Mask report'''
        # pred_mask, search, search_mask = self.alpha.get_mask(img_RGB, np.array(pred_bbox), vis=True)  # yuan

        # final_mask = (pred_mask > self.THRES).astype(np.uint8)  # threshold

        # test
        pred_score = outputs['target_score']

        # # test
        # print('mask')
        # print(final_mask)
        # print('mask.shape')
        # print(final_mask.shape)
        # final_mask045 = (pred_mask > 0.55).astype(np.uint8)
        # final_mask040 = (pred_mask > 0.5).astype(np.uint8)
        # img_test = img_RGB
        # img_test[:,:,1] = img_RGB[:,:,1]/2 + final_mask040*255/2
        # img_test[:,:,0] = img_RGB[:,:,0]/2 + final_mask045*255/2
        # img_test[:,:,2] = img_RGB[:,:,2]/2 + final_mask*255/2
        # gt = [278.33,98.0,167.66,216.67]
        # cv2.rectangle(img_test,(int(gt[0]),int(gt[1])),
        #               (int(gt[0]+gt[2]),int(gt[1]+gt[3])),(0,0,255,1))
        # img_test = np.array(img_test, dtype=np.uint8)
        # cv2.imwrite('/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/test_ar/outputmaskfusioncolor.jpg',img_test)
        # print(afuau)

        
        # return final_mask, 1
        # return final_mask, pred_score


        # AR
        # try:
        #     # pred_bbox = rect_from_mask(final_mask)
        #     # print(pred_bbox)
        #     # print(np.array(pred_bbox))
            
        #     pred = self.alpha.refine(img_RGB, np.array(pred_bbox))
        #     pred_bbox = pred['corner']    
        #     # print('11111111111111')        
        # except:
        #     pred_bbox = outputs['target_bbox']


        
        # print('pred_bbox')
        # print(pred_bbox)
        # print('pred_score')
        # print(pred_score)
        # print(iafha)
        return pred_bbox, pred_score


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


def get_image_d(imagefile_d):
    dp = cv2.imread(imagefile_d, -1)
    max_depth = min(np.median(dp)*3,10000)
    dp[dp>max_depth] = max_depth
    dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dp = np.asarray(dp, dtype=np.uint8)
    colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)
    return colormap

# def get_ar(img, init_box, ar_path):  # yuan
def get_ar(ar_path):
    """ set up Alpha-Refine """
    selector_path = 0
    sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
    RF_module = RefineModule(ar_path, selector_path, search_factor=sr, input_sz=input_sz)
    # RF_module.initialize(img, np.array(init_box))
    return RF_module



# start

# refine_model_name = 'ARcm_coco_seg_only_mask_384'  # yuan
refine_model_name = 'SEcmnet_ep0040-c.pth.tar'
# params = vot_params.parameters("baseline", model="networktest_cvt.pth")  # cvt_base
# params = vot_params.parameters("baseline", model="mixformer_online_22k.pth.tar")  # yuan
# params = vot_params.parameters("baseline_large", model="mixformerL_online_22k.pth.tar")  
# params = vot_params.parameters("baseline", model="mixformer_vit_base_online.pth.tar")
# params = vot_params.parameters("baseline_large", model="mixformer_vit_large_online.pth.tar")
# params = vot_params.parameters("baseline")
# params = vot_params.parameters("baseline", model="MixFormerOnlineScore_ep0025_744_v5.pth.tar")  # cvt_base
params = vot_params.parameters("baseline", model="MixFormerOnlineScore_ep0023_v5plus3_e4d2_0757_ar_0764.pth.tar")

params.AR = False
print(params)
# mixformer = MixFormerOnline(params, "VOT20")
# mixformer = MixFormerOnline(params, "VOT20RGBD")







mixformer = MixFormerOnline(params, "CDTB")
tracker = MIXFORMER_ALPHA_SEG(tracker=mixformer, refine_model_name=refine_model_name)
# handle = vot.VOT("mask","rgbd")
handle = vot.VOT("rectangle","rgbd")
selection = handle.region()
init_box = [selection.x, selection.y, selection.width, selection.height]
# imagefile = handle.frame()
imagefile_rgb, imagefile_d = handle.frame()

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
    # print('no track')
    region, confidence = tracker.track(image_rgb, image_d)
    selection = Rectangle(region[0], region[1], region[2], region[3])
    # print('track over')
    # print(affasf)
    # handle.report(region, confidence)
    handle.report(selection, confidence)
