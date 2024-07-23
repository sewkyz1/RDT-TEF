from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/got10k_lmdb'
    settings.got10k_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/lasot_lmdb'
    settings.lasot_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/LaSOT/LaSOTBenchmark'
    # settings.network_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-1/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/nfs'
    settings.otb_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/OTB2015'
    settings.prj_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz'
    # settings.result_plot_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-1/test/result_plots'
    # settings.results_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-1/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/save'
    # settings.segmentation_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-1/test/segmentation_results'
    settings.tc128_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/TrackingNet'
    settings.uav_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/UAV123'
    settings.vot_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/SwinTrack-main/data/VOT2019'
    # settings.rgbd_1k_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/RGBD1K_dataset/sequences_RGBD1K'
    settings.rgbd_1k_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/vot20/workspacergbd1k/sequences'
    settings.depthtrack_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/vot20/workspacedepthtrack/sequences'
    settings.youtubevos_dir = ''
    settings.cdtb_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/vot20/workspacergbd/sequences'
    # settings.rgbd_test_path = '/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/vot20/workspacergbd/sequences'

    return settings

