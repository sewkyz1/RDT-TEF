class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/pretrained_networks'
        self.lasot_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/LaSOT/LaSOTBenchmark'
        self.got10k_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/got-10k/train'
        self.lasot_lmdb_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/got10k_lmdb'
        self.trackingnet_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/TrackingNet'
        self.trackingnet_lmdb_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/trackingnet_lmdb'
        self.coco_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/coco/coco2017'
        self.coco_lmdb_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-mai_n/data/vid'
        self.imagenet_lmdb_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.rgbd1k_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/RGBD1K_dataset/RGBD1K_train_labelled'
        # self.rgbd1k_val_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/RGBD1K_dataset/sequences_RGBD1K'
        self.rgbd1k_val_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/vot20/workspacergbd1k/sequences'

