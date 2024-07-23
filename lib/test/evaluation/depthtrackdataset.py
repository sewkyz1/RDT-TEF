import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class RGBD_1kDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.depthtrack_path
        self.sequence_list = self._get_sequence_list('val')
        # self.clean_list = self.clean_seq_list()

    # def clean_seq_list(self):
    #     clean_lst = []
    #     for i in range(len(self.sequence_list)):
    #         cls, _ = self.sequence_list[i].split('-')
    #         clean_lst.append(cls)
    #     return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        # occlusion_label_path = '{}/{}/full_occlusion.txt'.format(self.base_path, sequence_name)

        # # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        # full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        # out_of_view_label_path = '{}/{}/out_of_view.txt'.format(self.base_path, sequence_name)
        # out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        # target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path_rgb = '{}/{}/color'.format(self.base_path, sequence_name)
        frames_path_depth = '{}/{}/depth'.format(self.base_path, sequence_name)

        frames_list_rgb = ['{}/{:08d}.jpg'.format(frames_path_rgb, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]
        frames_list_depth = ['{}/{:08d}.png'.format(frames_path_depth, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]
        
        return Sequence(sequence_name, frames_list_rgb, frames_list_depth, 'rgbd1k', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()
        return sequence_list
