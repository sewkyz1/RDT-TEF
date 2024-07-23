import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
# tracker for vot

def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False, tracker_params=None):
        # assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        self.params = self.get_parameters(tracker_params)

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        # params = self.get_parameters()
        params = self.params

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        # image = self._read_image(seq.frames[0])
        image_rgb = self._read_image(seq.frames_rgb[0])
        image_depth = self._read_image_d(seq.frames_depth[0])

        start_time = time.time()
        # out = tracker.initialize(image, init_info)
        out = tracker.initialize(image_rgb, image_depth, init_info)        
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        output_pred_score = []
        output_pred_score.append(-9)


        # # read groundtruth for draw
        # groundtruth = []
        # with open(f"/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/vot20/workspacergbd/sequences/{seq.name}/groundtruth.txt") as f:
        #     for line in f:
        #         groundtruth.append(line)

        # save_long_path = f"/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/raw_data_cdtb/raw_data/{seq.name}"
        # if not os.path.exists(save_long_path):
        #     os.mkdir(save_long_path)

        # # -------------for draw
        # # cdtb
        # our_pred = []
        # with open(f"/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/raw_data_cdtb/RAW_DATA_CDTB/our/MixFormerPython/rgbd-unsupervised/{seq.name}/{seq.name}_001.txt") as f:
        #     for line in f:
        #         # line.split()
        #         our_pred.append(line.split())
        # ATCAIS_pred = []
        # with open(f"/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/raw_data_cdtb/RAW_DATA_CDTB/ATCAIS/rgbd-unsupervised/{seq.name}/{seq.name}_001.txt") as f:
        #     for line in f:
        #         # line.split()
        #         ATCAIS_pred.append(line.split())
        # SiamDW_D_pred = []
        # with open(f"/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/raw_data_cdtb/RAW_DATA_CDTB/SiamDW_D/rgbd-unsupervised/{seq.name}/{seq.name}_001.txt") as f:
        #     for line in f:
        #         # line.split()
        #         SiamDW_D_pred.append(line.split())
        # DeT_DiMP50_Mean_pred = []
        # with open(f"/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/raw_data_cdtb/RAW_DATA_CDTB/DeT_DiMP50_Mean/rgbd-unsupervised/{seq.name}/{seq.name}_001.txt") as f:
        #     for line in f:
        #         # line.split()
        #         DeT_DiMP50_Mean_pred.append(line.split())
        # DiMP50_pred = []
        # with open(f"/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/raw_data_cdtb/RAW_DATA_CDTB/DiMP50/rgbd-unsupervised/{seq.name}/{seq.name}_001.txt") as f:
        #     for line in f:
        #         # line.split()
        #         DiMP50_pred.append(line.split())  
        # gt_pred = []
        # # external/vot20/workspacergbd/sequences/backpack_blue/groundtruth.txt
        # with open(f"/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/vot20/workspacergbd/sequences/{seq.name}/groundtruth.txt") as f:
        #     for line in f:
        #         # line.split()
        #         gt_pred.append(line.split())  
            
        # rgbd1k  
        # corner_pred = []
        # with open(f"/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/raw_data_rgbd1k/resultscorner560/MixFormerPython/rgbd-unsupervised/{seq.name}/{seq.name}_001.txt") as f:
        #     for line in f:
        #         # line.split()
        #         corner_pred.append(line.split())      
        # center_pred = []
        # with open(f"/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/raw_data_rgbd1k/resultscenter547/MixFormerPython/rgbd-unsupervised/{seq.name}/{seq.name}_001.txt") as f:
        #     for line in f:
        #         # line.split()
        #         center_pred.append(line.split())
        # gt_pred = []
        # with open(f"/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/vot20/workspacergbd1k/sequences/{seq.name}/groundtruth.txt") as f:
        #     for line in f:
        #         # line.split()
        #         gt_pred.append(line.split())  
        # #-----------


        time_new = []
        # for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
        for frame_num, frame_path in enumerate(seq.frames_rgb[1:], start=1):
            # image = self._read_image(frame_path)
            image_rgb = self._read_image(frame_path)
            image_depth = self._read_image_d(seq.frames_depth[frame_num])

            # ------------for track
            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            # out = tracker.track(image_rgb, image_depth, info)  # yuan
            
            out, time_onlynetwprk = tracker.track(image_rgb, image_depth, info)  # yuan
            time_new.append(time_onlynetwprk)
            # out, draw, attention_map = tracker.track(image_rgb, image_depth, info)  # for draw
            # ---------------

            # # -------------for draw raw data
            # image_rgb = cv.cvtColor(image_rgb,cv.COLOR_RGB2BGR)


            # # cdtb
            # # blue
            # ATCAIS_box = ATCAIS_pred[frame_num][0].split(',')
            # ATCAIS_box_int = [float(x) for x in ATCAIS_box]
            # cv.rectangle(image_rgb, (int(ATCAIS_box_int[0]),int(ATCAIS_box_int[1])),(int(ATCAIS_box_int[0]+ATCAIS_box_int[2]),int(ATCAIS_box_int[1]+ATCAIS_box_int[3])), (255,0,0), 2) 
            # SiamDW_D_box = SiamDW_D_pred[frame_num][0].split(',')
            # # grean
            # SiamDW_D_int = [float(x) for x in SiamDW_D_box]
            # cv.rectangle(image_rgb, (int(SiamDW_D_int[0]),int(SiamDW_D_int[1])),(int(SiamDW_D_int[0]+SiamDW_D_int[2]),int(SiamDW_D_int[1]+SiamDW_D_int[3])), (0,255,0), 2)             
            # # purple
            # DeT_DiMP50_Mean_box = DeT_DiMP50_Mean_pred[frame_num][0].split(',')
            # DeT_DiMP50_Mean_int = [float(x) for x in DeT_DiMP50_Mean_box]
            # cv.rectangle(image_rgb, (int(DeT_DiMP50_Mean_int[0]),int(DeT_DiMP50_Mean_int[1])),(int(DeT_DiMP50_Mean_int[0]+DeT_DiMP50_Mean_int[2]),int(DeT_DiMP50_Mean_int[1]+DeT_DiMP50_Mean_int[3])), (178,7,83), 2) 
            # # yellow
            # DiMP50_box = DiMP50_pred[frame_num][0].split(',')
            # DiMP50_int = [float(x) for x in DiMP50_box]
            # cv.rectangle(image_rgb, (int(DiMP50_int[0]),int(DiMP50_int[1])),(int(DiMP50_int[0]+DiMP50_int[2]),int(DiMP50_int[1]+DiMP50_int[3])), (118,213,246), 2) 
            # # our red
            # our_box = our_pred[frame_num][0].split(',')
            # our_box_int = [float(x) for x in our_box]
            # cv.rectangle(image_rgb, (int(our_box_int[0]),int(our_box_int[1])),(int(our_box_int[0]+our_box_int[2]),int(our_box_int[1]+our_box_int[3])), (0,0,255), 2) 
            # # depth gt and our
            # gt_box = gt_pred[frame_num][0].split(',')
            # gt_box_int = [float(x) for x in gt_box] 
            # try:
            #     cv.rectangle(image_depth, (int(gt_box_int[0]),int(gt_box_int[1])),(int(gt_box_int[0]+gt_box_int[2]),int(gt_box_int[1]+gt_box_int[3])), (0,0,0), 2)            
            # except:
            #     aa  = 1
            # cv.rectangle(image_depth, (int(our_box_int[0]),int(our_box_int[1])),(int(our_box_int[0]+our_box_int[2]),int(our_box_int[1]+our_box_int[3])), (0,0,255), 2)                     
            # picture = np.concatenate((image_rgb, image_depth),axis=0)
            # # cv.imwrite(f'/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/{frame_num+1}.jpg', picture)
            # cv.imwrite(f'/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/raw_data_cdtb/raw_data/{seq.name}/{frame_num+1}.jpg', picture)

            # rgbd1k
            # # blue corner
            # ATCAIS_box = corner_pred[frame_num][0].split(',')
            # ATCAIS_box_int = [float(x) for x in ATCAIS_box]
            # cv.rectangle(image_rgb, (int(ATCAIS_box_int[0]),int(ATCAIS_box_int[1])),(int(ATCAIS_box_int[0]+ATCAIS_box_int[2]),int(ATCAIS_box_int[1]+ATCAIS_box_int[3])), (255,0,0), 2)
            # # red center
            # our_box = center_pred[frame_num][0].split(',')
            # our_box_int = [float(x) for x in our_box]
            # cv.rectangle(image_rgb, (int(our_box_int[0]),int(our_box_int[1])),(int(our_box_int[0]+our_box_int[2]),int(our_box_int[1]+our_box_int[3])), (0,0,255), 2)  
            # # black gt
            # gt_box = gt_pred[frame_num][0].split(',')
            # gt_box_int = [float(x) for x in gt_box] 
            # try:
            #     cv.rectangle(image_rgb, (int(gt_box_int[0]),int(gt_box_int[1])),(int(gt_box_int[0]+gt_box_int[2]),int(gt_box_int[1]+gt_box_int[3])), (0,0,0), 2)
            # except:
            #     aa = 1
            # cv.imwrite(f'/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/raw_data_rgbd1k/raw_data/{seq.name}/{frame_num+1}.jpg', image_rgb)
            # # ---------------


            # # draw 1
            # image_rgb_gt = cv.cvtColor(draw['rgb_img'], cv.COLOR_RGB2BGR)
            # # image_draw = image_rgb_gt
            # xx,yy,ww,hh = draw['pred_box']
            # try:
            #     cv.rectangle(image_rgb_gt, (int(xx-ww/2),int(yy-hh/2)), (int(xx+ww/2),int(yy+hh/2)), (0,0,255), 2) 
            # except:
            #     image_rgb_gt = image_rgb_gt

            # picture = np.concatenate((image_rgb_gt,cv.cvtColor(draw['rgb_img'], cv.COLOR_RGB2BGR),draw['d_img'],draw['map_tl'],draw['map_br']),axis=1)
            # # tl = ((draw['map_tl']+image_draw)/2).astype(np.uint8)
            # # br = ((draw['map_br']+image_draw)/2).astype(np.uint8)
            # # picture = np.concatenate((image_rgb_gt,image_draw,draw['d_img'],tl,br),axis=1)

            # # cv.imwrite(f'/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/cdtb_hotmap/corner/{seq.name}/long{frame_num+1}.jpg', picture)

            # # draw 2
            # image_rgb_gt = cv.cvtColor(draw['rgb_img'], cv.COLOR_RGB2BGR)
            # image_draw = image_rgb_gt
            # xx,yy,ww,hh = draw['pred_box']
            # try:
            #     cv.rectangle(image_rgb_gt, (int(xx-ww/2),int(yy-hh/2)), (int(xx+ww/2),int(yy+hh/2)), (0,0,255), 2) 
            # except:
            #     image_rgb_gt = image_rgb_gt
            # tl = ((draw['map_tl']+image_draw)/2).astype(np.uint8)
            # br = ((draw['map_br']+image_draw)/2).astype(np.uint8)
            # encoder_over = ((attention_map[4]+image_draw)/2).astype(np.uint8)
            # decoder_over = ((attention_map[6]+image_draw)/2).astype(np.uint8)
            # picture1 = np.concatenate((image_rgb_gt,attention_map[0],attention_map[1],attention_map[2],attention_map[3],attention_map[4]),axis=1)
            # picture2 = np.concatenate((attention_map[5],attention_map[6],encoder_over,decoder_over,tl,br),axis=1)
            # picture = np.concatenate((picture1,picture2),axis=0)
            # cv.imwrite(f'/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/cdtb_hotmap/corner/{seq.name}/long{frame_num+1}.jpg', picture)

            # draw 3
            # image_rgb_gt = cv.cvtColor(draw['rgb_img'], cv.COLOR_RGB2BGR)
            # image_draw = image_rgb_gt
            # xx,yy,ww,hh = draw['pred_box']
            # try:
            #     cv.rectangle(image_rgb_gt, (int(xx-ww/2),int(yy-hh/2)), (int(xx+ww/2),int(yy+hh/2)), (0,0,255), 2) 
            # except:
            #     image_rgb_gt = image_rgb_gt
            # tl = ((draw['map_tl']+image_draw)/2).astype(np.uint8)
            # br = ((draw['map_br']+image_draw)/2).astype(np.uint8)
            # encoder_over = ((attention_map[4]+image_draw)/2).astype(np.uint8)
            # decoder_over = ((attention_map[6]+image_draw)/2).astype(np.uint8)
            # picture = np.concatenate((image_rgb_gt,draw['d_img'],attention_map[5],attention_map[4],attention_map[7],draw['map_tl'],draw['map_br']),axis=1)
            # cv.imwrite(f'/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/cdtb_hotmap/corner/{seq.name}/long{frame_num+1}.jpg', picture)








            # save
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

            output_pred_score.append(out['target_score'])









            # # for test
            # if frame_num==6:
            #     break

        # save vot results
        name = seq.name
        filename = f"{name}"
        filename = f'/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/results/MixFormerPython/rgbd-unsupervised/{filename}'
        # if dir exist?
        folder = os.path.exists(filename)
        if not folder:
            os.makedirs(filename)

        filename_bbox = f'/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/results/MixFormerPython/rgbd-unsupervised/{name}/{name}_001.txt'
        filename_confidence = f'/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/results/MixFormerPython/rgbd-unsupervised/{name}/{name}_001_confidence.txt'
        filename_time = f'/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/results/MixFormerPython/rgbd-unsupervised/{name}/{name}_001_time.txt'

        with open(filename_bbox, "a") as f:
            for f_num, bbox in enumerate(output['target_bbox']):
                if f_num == 0:
                    f.write("1\n")
                else:
                    f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")

        with open(filename_confidence, "a") as f:
            for f_num, pred_score in enumerate(output_pred_score):
                if f_num == 0:
                    f.write("\n")
                else:
                    f.write(f"{pred_score}\n")                

        with open(filename_time, "a") as f:
            for f_time in output['time']:
                f.write(f"{f_time}\n")

        os.rename(filename_confidence,
                  f'/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/results/MixFormerPython/rgbd-unsupervised/{name}/{name}_001_confidence.value')       
        os.rename(filename_time,
                  f'/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/results/MixFormerPython/rgbd-unsupervised/{name}/{name}_001_time.value')

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        # params = self.get_parameters()
        params = self.params

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []
        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        # cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            raise NotImplementedError("We haven't support cv_show now.")
            # while True:
            #     # cv.waitKey()
            #     frame_disp = frame.copy()
            #
            #     cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
            #                1.5, (0, 0, 0), 1)
            #
            #     x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            #     init_state = [x, y, w, h]
            #     tracker.initialize(frame, _build_init_info(init_state))
            #     output_boxes.append(init_state)
            #     break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            # cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
            #              (0, 255, 0), 5)
            #
            # font_color = (0, 0, 0)
            # cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #            font_color, 1)
            # cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #            font_color, 1)
            # cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #            font_color, 1)

            # Display the resulting frame
            # cv.imshow(display_name, frame_disp)
            # key = cv.waitKey(1)
            # if key == ord('q'):
            #     break
            # elif key == ord('r'):
            #     ret, frame = cap.read()
            #     frame_disp = frame.copy()
            #
            #     cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
            #                (0, 0, 0), 1)
            #
            #     # cv.imshow(display_name, frame_disp)
            #     x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            #     init_state = [x, y, w, h]
            #     tracker.initialize(frame, _build_init_info(init_state))
            #     output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self, tracker_params=None):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        search_area_scale = None
        if tracker_params is not None and 'search_area_scale' in tracker_params:
            search_area_scale = tracker_params['search_area_scale']
        model = ''
        if tracker_params is not None and 'model' in tracker_params:
            model = tracker_params['model']
        params = param_module.parameters(self.parameter_name, model, search_area_scale)
        if tracker_params is not None:
            for param_k, v in tracker_params.items():
                setattr(params, param_k, v)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")
        
    def _read_image_d(self, imagefile_d: str):
        if isinstance(imagefile_d, str):
            dp = cv.imread(imagefile_d, -1)
            max_depth = min(np.median(dp)*3,10000)
            dp[dp>max_depth] = max_depth
            dp = cv.normalize(dp, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            colormap = cv.applyColorMap(dp, cv.COLORMAP_JET)
            # im = cv.imread(image_file)
            # return cv.cvtColor(im, cv.COLOR_BGR2RGB)
            return colormap
        # elif isinstance(image_file, list) and len(image_file) == 2:
        #     return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



