from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.models.mixformer_cvt import build_mixformer_cvt_online_score
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box
from lib.test.tracker.tracker_utils import vis_attn_maps
# change in 1206 add ar
# mixformer_cvt_online_ar    only use ar to correct box
import sys
prj_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/AR'
if prj_path not in sys.path:
    sys.path.append(prj_path)

from external.AR.pytracking.refine_modules.refine_module import RefineModule
import numpy as np

class MixFormerOnline(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MixFormerOnline, self).__init__(params)
        network = build_mixformer_cvt_online_score(params.cfg,  train=False)


        # network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        # network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        # missing_keys, unexpected_keys = network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)  
        # missing_keys, unexpected_keys = network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)



        missing_keys, unexpected_keys = network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        print(self.params.checkpoint)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        # print(adadad)


        # test
        # torch.save(network.state_dict(), '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/networktestcvt.pth')
        # print(afafawfa)

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.attn_weights = []

        self.preprocessor = Preprocessor_wo_mask()
        self.state = None



        if params.AR:
            self.alpha = get_ar(params.AR_PATH)


        # for debug
        # self.debug = params.debug
        self.debug = False

        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # self.z_dict1 = {}

        # Set the update interval
        DATASET_NAME = dataset_name.upper()  # mei you dui ying de CDTB canshu
        print(f"DATASET_NAME:{DATASET_NAME}")

        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
            self.online_sizes = self.cfg.TEST.ONLINE_SIZES[DATASET_NAME]
            self.online_size = self.online_sizes[0]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
            self.online_size = 3
        self.update_interval = self.update_intervals[0]
        if hasattr(params, 'online_sizes'):
            self.online_size = params.online_sizes
        print("Online size is: ", self.online_size)
        if hasattr(params, 'update_interval'):
            self.update_interval = params.update_interval
        print("Update interval is: ", self.update_interval)
        if hasattr(params, 'max_score_decay'):  # change this parameter in test.py
            self.max_score_decay = params.max_score_decay
        else:
            # self.max_score_decay = 1.0  # yuan
            self.max_score_decay = 0.98
        if not hasattr(params, 'vis_attn'):
            self.params.vis_attn = 0
        print("change max score decay in tracking/test.py or lib/test/tracker/mixformer_cvt_online.py")
        print("max score decay = {}".format(self.max_score_decay))

        # test
        # print(jhagsvfhjugasfvuhas)


    # def initialize(self, image, info: dict):
    def initialize(self, image_rgb, image_d, info: dict):
        # forward the template once
        z_patch_arr_rgb, _, z_amask_arr = sample_target(image_rgb, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        if self.params.vis_attn==1:
            self.z_patch_rgb = z_patch_arr_rgb
            self.oz_patch_rgb = z_patch_arr_rgb
        template_rgb = self.preprocessor.process(z_patch_arr_rgb)
        self.template_rgb = template_rgb
        self.online_template_rgb = template_rgb

        z_patch_arr_d, _, z_amask_arr = sample_target(image_d, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        if self.params.vis_attn==1:
            self.z_patch_d = z_patch_arr_d
            self.oz_patch_d = z_patch_arr_d
        template_d = self.preprocessor.process(z_patch_arr_d)
        self.template_d = template_d
        self.online_template_d = template_d

    
        if self.online_size > 1:
            with torch.no_grad():
                self.network.set_online_rgb(self.template_rgb, self.online_template_rgb)

        if self.online_size > 1:
            with torch.no_grad():
                self.network.set_online_d(self.template_d, self.online_template_d)

        self.online_state = info['init_bbox']
        
        # self.online_image = image  # ???????? mei yong dao self.online_image
        self.online_image_rgb = image_rgb
        self.online_image_d = image_d


        self.max_pred_score = -1.0
        self.online_max_template_rgb = template_rgb
        self.online_max_template_d = template_d
        self.online_forget_id = 0
        
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}
        
        # init AR
        if self.params.AR:
            self.alpha.initialize(image_rgb, np.array(info['init_bbox']))

    def track(self, image_rgb, image_d, info: dict = None):
        H, W, _ = image_rgb.shape

        self.frame_id += 1
        x_patch_arr_rgb, resize_factor, x_amask_arr = sample_target(image_rgb, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search_rgb = self.preprocessor.process(x_patch_arr_rgb)

        x_patch_arr_d, resize_factor, x_amask_arr = sample_target(image_d, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search_d = self.preprocessor.process(x_patch_arr_d)

        with torch.no_grad():
            if self.online_size==1:  # online_size==1 haven't been revised
                # for visualize attention maps
                print('online_size==1 haven*t been revised')
                print(wrfqwwfwqfq)
                # if self.params.vis_attn==1 and self.frame_id % 200 == 0:
                #     attn_weights = []
                #     hooks = []
                #     for i in range(len(self.network.backbone.stage2.blocks)):
                #         hooks.append(self.network.backbone.stage2.blocks[i].attn.attn_drop.register_forward_hook(
                #             lambda self, input, output: attn_weights.append(output)))
                # # out_dict, _ = self.network(self.template, self.online_template, search, run_score_head=True)
                # out_dict, _ = self.network(self.template_rgb, self.online_template_rgb, search_rgb, self.template_d, self.online_template_d, search_d, run_score_head=True)
                # if self.params.vis_attn==1 and self.frame_id % 200 == 0:
                #     for hook in hooks:
                #         hook.remove()
                #     # attn0(t_ot) / 1(t_ot) / 2(t_ot_s)
                #     # shape: torch.Size([1, 6, 64, 32]), torch.Size([1, 6, 64, 32]), torch.Size([1, 6, 400, 132])
                #     # vis attn weights: online_template-to-template
                #     vis_attn_maps(attn_weights[::3], q_w=8, k_w=4, skip_len=16, x1=self.oz_patch, x2=self.z_patch,
                #                   x1_title='Online Template', x2_title='Template',
                #                   save_path= 'vis_attn_weights/t2ot_vis/%04d' % self.frame_id)
                #     # vis attn weights: template-to-online_template
                #     vis_attn_maps(attn_weights[1::3], q_w=8, k_w=4, skip_len=0, x1=self.z_patch, x2=self.oz_patch,
                #                   x1_title='Template', x2_title='Online Template',
                #                   save_path='vis_attn_weights/ot2t_vis/%04d' % self.frame_id)
                #     # vis attn weights: template-to-search
                #     vis_attn_maps(attn_weights[2::3], q_w=20, k_w=4, skip_len=0, x1=self.z_patch, x2=x_patch_arr,
                #                   x1_title='Template', x2_title='Search',
                #                   save_path='vis_attn_weights/s2t_vis/%04d' % self.frame_id)
                #     # vis attn weights: online_template-to-search
                #     vis_attn_maps(attn_weights[2::3], q_w=20, k_w=4, skip_len=16, x1=self.oz_patch, x2=x_patch_arr,
                #                   x1_title='Online Template', x2_title='Search',
                #                   save_path='vis_attn_weights/s2ot_vis/%04d' % self.frame_id)
                #     # vis attn weights: search-to-search
                #     vis_attn_maps(attn_weights[2::3], q_w=20, k_w=10, skip_len=32, x1=x_patch_arr, x2=x_patch_arr,
                #                   x1_title='Search1', x2_title='Search2', idxs=[(160, 160)],
                #                   save_path='vis_attn_weights/s2s_vis/%04d' % self.frame_id)
                #     print("save vis_attn of frame-{} done.".format(self.frame_id))
            else:
                out_dict, _ = self.network.forward_test(search_rgb, search_d, run_score_head=True)  # yuan
                # out_dict, outputs_coord_new, hot_tl, hot_br, attention_map = self.network.forward_test(search_rgb, search_d, run_score_head=True)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        pred_score = out_dict['pred_scores'].view(1).sigmoid().item()
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        self.max_pred_score = self.max_pred_score * self.max_score_decay
        # update template
        if pred_score > 0.5 and pred_score > self.max_pred_score:
            z_patch_arr_rgb, _, z_amask_arr = sample_target(image_rgb, self.state,
                                                        self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            self.online_max_template_rgb = self.preprocessor.process(z_patch_arr_rgb)
            z_patch_arr_d, _, z_amask_arr = sample_target(image_d, self.state,
                                                        self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            self.online_max_template_d = self.preprocessor.process(z_patch_arr_d)
            if self.params.vis_attn == 1:
                self.oz_patch_max_rgb = z_patch_arr_rgb
                self.oz_patch_max_d = z_patch_arr_d
            self.max_pred_score = pred_score
        if self.frame_id % self.update_interval == 0:            
            # print(self.update_interval)
            # print(jhajvhaqwhgvfasvhg)
            if self.online_size == 1:
                self.online_template_rgb = self.online_max_template_rgb
                self.online_template_d = self.online_max_template_d
                if self.params.vis_attn == 1:
                    self.oz_patch_rgb = self.oz_patch_max_rgb
                    self.oz_patch_d = self.oz_patch_max_d
            elif self.online_template_rgb.shape[0] < self.online_size:
                self.online_template_rgb = torch.cat([self.online_template_rgb, self.online_max_template_rgb])
                self.online_template_d = torch.cat([self.online_template_d, self.online_max_template_d])
            else:
                # self.online_template[self.online_forget_id:self.online_forget_id+1] = self.online_max_template
                # self.online_forget_id = (self.online_forget_id + 1) % self.online_size
                self.online_template_rgb[self.online_forget_id:self.online_forget_id+1] = self.online_max_template_rgb
                self.online_template_d[self.online_forget_id:self.online_forget_id+1] = self.online_max_template_d
                self.online_forget_id = (self.online_forget_id + 1) % self.online_size

            if self.online_size > 1:
                with torch.no_grad():
                    self.network.set_online_rgb(self.template_rgb, self.online_template_rgb)
                    self.network.set_online_d(self.template_d, self.online_template_d)

            self.max_pred_score = -1
            self.online_max_template_rgb = self.template_rgb
            self.online_max_template_d = self.template_d

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            # image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image)

        # draw
        # draw = {"rgb_img": x_patch_arr_rgb,
        #         "d_img": x_patch_arr_d,
        #         "map_tl": hot_tl,
        #         "map_br": hot_br,
        #         "pred_box": (pred_boxes.mean(dim=0) * self.params.search_size).tolist(),
        #         }

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        # else:  # yuan
        #     return {"target_bbox": self.state,
        #             "target_score": pred_score}

        else:  # add ar
            if self.params.AR:
                # print(AFBA)  # TO STOP VOT     DELETE
                pred = self.alpha.refine(image_rgb, np.array(self.state))
                pred_bbox = pred['corner'] 
                return {"target_bbox": pred_bbox,
                        "target_score": pred_score}  # , draw, attention_map
            else:
                return {"target_bbox": self.state,
                        "target_score": pred_score}  # , draw, attention_map

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return MixFormerOnline

# AR
def get_ar(ar_path):
    """ set up Alpha-Refine """
    selector_path = 0
    sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
    RF_module = RefineModule(ar_path, selector_path, search_factor=sr, input_sz=input_sz)
    # RF_module.initialize(img, np.array(init_box))
    return RF_module