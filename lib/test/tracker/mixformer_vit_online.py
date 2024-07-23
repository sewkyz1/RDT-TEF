from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.models.mixformer_vit import build_mixformer_vit_online_score
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box
# rgbd only input

class MixFormerOnline(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MixFormerOnline, self).__init__(params)
        network = build_mixformer_vit_online_score(params.cfg,  train=False)
        # network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        print(f"Load checkpoint {self.params.checkpoint} successfully!")
        print('Warning:strict=False')
        # print(network)
        # print(affafaf)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.attn_weights = []

        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
            self.online_sizes = self.cfg.TEST.ONLINE_SIZES[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
            self.online_size = 3
        self.update_interval = self.update_intervals[0]
        self.online_size = self.online_sizes[0]
        if hasattr(params, 'online_sizes'):
            self.online_size = params.online_sizes
        if hasattr(params, 'update_interval'):
            self.update_interval = params.update_interval
        if hasattr(params, 'max_score_decay'):
            self.max_score_decay = params.max_score_decay
        else:
            self.max_score_decay = 1.0
        if not hasattr(params, 'vis_attn'):
            self.params.vis_attn = 0
        print("Search scale is: ", self.params.search_factor)
        print("Online size is: ", self.online_size)
        print("Update interval is: ", self.update_interval)
        print("Max score decay is ", self.max_score_decay)


    def initialize(self, image_rgb, image_d, info: dict):
        # forward the template once
        print('sample_target')
        z_patch_arr_rgb, _, z_amask_arr_rgb = sample_target(image_rgb, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        print('process')
        template_rgb = self.preprocessor.process(z_patch_arr_rgb)
        # d 
        z_patch_arr_d, _, z_amask_arr_d = sample_target(image_d, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template_d = self.preprocessor.process(z_patch_arr_d)   


        self.template_rgb = template_rgb
        self.online_template_rgb = template_rgb
        # d
        self.template_d = template_d
        self.online_template_d = template_d     

        print('no set')
        if self.online_size > 1:
            with torch.no_grad():
                self.network.set_online_rgb(self.template_rgb, self.online_template_rgb)
        print('set rgb over')
        # d
        if self.online_size > 1:
            with torch.no_grad():
                self.network.set_online_d(self.template_d, self.online_template_d)
        print('set d over')
        self.online_state = info['init_bbox']
        
        
        self.online_image_rgb = image_rgb  # ????????????????????????????????????  # zhu shi kan shi fou you bug  # zhe ge bian liang mei yong shang
        self.online_image_d = image_d

        
        self.max_pred_score = -1.0
        self.online_max_template_rgb = template_rgb
        # d
        self.online_max_template_d = template_d
        self.online_forget_id = 0
        
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image_rgb, image_d, info: dict = None):
        H, W, _ = image_rgb.shape
        self.frame_id += 1
        x_patch_arr_rgb, resize_factor, x_amask_arr_rgb = sample_target(image_rgb, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search_rgb = self.preprocessor.process(x_patch_arr_rgb)
        # d
        x_patch_arr_d, _, x_amask_arr_d = sample_target(image_d, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search_d = self.preprocessor.process(x_patch_arr_d)

        with torch.no_grad():
            if self.online_size==1:
                out_dict, _ = self.network(self.template_rgb, self.template_d, self.online_template_rgb, self.online_template_d, search_rgb, search_d, run_score_head=True)
                print('error: size==1 no change')
                print(qwafwqafa)
            else:
                # out_dict, _ = self.network.forward_test(search_rgb, search_d, run_score_head=True)
                # out_dict, _ = self.network.forward_test(search_d, run_score_head=True)
                # out_dict, _ = self.network.forward_test(search_rgb, run_score_head=True)
                out_dict, _ = self.network.forward_test(search_rgb, search_d, run_score_head=True)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        pred_score = out_dict['pred_scores'].view(1).sigmoid().item()
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        self.max_pred_score = self.max_pred_score * self.max_score_decay
        # update template
        if pred_score > 0.5 and pred_score > self.max_pred_score:
            z_patch_arr_rgb, _, z_amask_arr_rgb = sample_target(image_rgb, self.state,
                                                        self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            self.online_max_template_rgb = self.preprocessor.process(z_patch_arr_rgb)
            # d
            z_patch_arr_d, _, z_amask_arr_d = sample_target(image_d, self.state,
                                                        self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            self.online_max_template_d = self.preprocessor.process(z_patch_arr_d)

            self.max_pred_score = pred_score
        if self.frame_id % self.update_interval == 0:
            if self.online_size == 1:
                self.online_template_rgb = self.online_max_template_rgb
                self.online_template_d = self.online_max_template_d
            elif self.online_template_rgb.shape[0] < self.online_size:
                self.online_template_rgb = torch.cat([self.online_template_rgb, self.online_max_template_rgb])
                self.online_template_d = torch.cat([self.online_template_d, self.online_max_template_d])
            else:
                self.online_template_rgb[self.online_forget_id:self.online_forget_id+1] = self.online_max_template_rgb
                self.online_template_d[self.online_forget_id:self.online_forget_id+1] = self.online_max_template_d
                self.online_forget_id = (self.online_forget_id + 1) % self.online_size

            if self.online_size > 1:
                with torch.no_grad():
                    self.network.set_online_rgb(self.template_rgb, self.online_template_rgb)
                    self.network.set_online_d(self.template_d, self.online_template_d)

            self.max_pred_score = -1
            self.online_max_template_rgb = self.template_rgb  # chong zhi zui gao fen template wei chushi template
            self.online_max_template_d = self.template_d

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

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
