import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
# from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, TNL2k
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, TNL2k, RGBD1K
# from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb, RGBD1K_lmdb
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        # assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "COCO17", "VID", "TRACKINGNET", "TNL2k"]
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "COCO17", "VID", "TRACKINGNET", "TNL2k", "RGBD1K", "RGBD1K_val"]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "TNL2k":
            datasets.append(TNL2k(settings.env.tnl2k_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
        # new
        if name == "RGBD1K":
            if settings.use_lmdb:
                print("Building RGBD1K from lmdb")
                datasets.append(RGBD1K_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(RGBD1K(settings.env.rgbd1k_dir, split='train',image_loader=image_loader))
        if name == "RGBD1K_val":
            if settings.use_lmdb:
                print("Building RGBD1K from lmdb")
                datasets.append(RGBD1K_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(RGBD1K(settings.env.rgbd1k_dir, split='val',image_loader=image_loader))
                datasets.append(RGBD1K(settings.env.rgbd1k_val_dir, split='val',image_loader=image_loader))  # change in 1007

                # error
                # datasets.append(RGBD1K(settings.env.rgbd1k_dir, split='train',image_loader=image_loader)) 
                # print('theres error in lib/train/base_functions.py line 97, because of lack of rgbd1k_val_dir')
        

    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)


    train_score = True


    print("sampler_mode", sampler_mode)

    data_processing_train = processing.MixformerProcessing(search_area_factor=search_area_factor,
                                                           output_sz=output_sz,
                                                           center_jitter_factor=settings.center_jitter_factor,
                                                           scale_jitter_factor=settings.scale_jitter_factor,
                                                           mode='sequence',
                                                           transform=transform_train,
                                                           joint_transform=transform_joint,
                                                           settings=settings,
                                                           train_score=train_score)

    data_processing_val = processing.MixformerProcessing(search_area_factor=search_area_factor,
                                                         output_sz=output_sz,
                                                         center_jitter_factor=settings.center_jitter_factor,
                                                         scale_jitter_factor=settings.scale_jitter_factor,
                                                         mode='sequence',
                                                         transform=transform_val,
                                                         joint_transform=transform_joint,
                                                         settings=settings,
                                                         train_score=train_score)


    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)
    freeze_stage0 = getattr(cfg.TRAIN, "FREEZE_STAGE0", False)
    freeze_first_6layers = getattr(cfg.TRAIN, "FREEZE_FIRST_6LAYERS", False)
    only_backbone_d = getattr(cfg.TRAIN, "ONLY_BACKBONE_D", False)
    only_head = getattr(cfg.TRAIN, "ONLY_HEAD", False)
    only_fusion = getattr(cfg.TRAIN, "ONLY_FUSION", False)
    only_fusion_weight = getattr(cfg.TRAIN, "ONLY_FUSION_WEIGHT", False)
    fusion_and_head = getattr(cfg.TRAIN, "FUSION_AND_HEAD", False)
    fusion_and_head_and_backbone = getattr(cfg.TRAIN, "FUSION_AND_HEAD_AND_BACKBONE", False)
    center_head = getattr(cfg.TRAIN, "CENTER_HEAD", False)

    if train_score:
        print("Only training score_branch. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "score" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters(): 
            if "score" not in n:  # freeze 
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
    elif freeze_stage0: # only for CVT-large backbone
        assert "cvt_24" == cfg.MODEL.VIT_TYPE
        print("Freeze Stage0 of MixFormer cvt backbone. Learnable parameters are shown below. # only for CVT-large backbone")
        for n, p in net.named_parameters():
            if "stage2" not in n and "box_head" not in n and "stage1" not in n:
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if (("stage2" in n or "stage1" in n) and p.requires_grad)],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
    elif freeze_first_6layers:  # only for ViT-Large backbone
        assert "large_patch16" == cfg.MODEL.VIT_TYPE
        print("Freeze the first 6 layers of MixFormer vit backbone. Learnable parameters are shown below.")
        for n, p in net.named_parameters():
            if 'blocks.0.' in n or 'blocks.1.' in n or 'blocks.2.' in n or 'blocks.3.' in n or 'blocks.4.' in n or 'blocks.5.' in n \
                or 'patch_embed' in n:
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]

    # elif only_backbone_d:  # train all backbone_d layer  # old
    #     for n, p in net.named_parameters(): 
    #         if "backbone_d" not in n and 'box_head' not in n:  # freeze 
    #             p.requires_grad = False
    #         else:
    #             if is_main_process():
    #                 print(n)
    #     param_dicts = [
    #         {
    #             "params": [p for n, p in net.named_parameters() if (("backbone_d" in n) and p.requires_grad)],
    #             "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,  # ???
    #         },
    #     ]
    elif only_backbone_d:  # train all backbone_d layer  # new 1004 add fusion
        for n, p in net.named_parameters(): 
            if "backbone_d" not in n and 'box_head' not in n and 'fusion' not in n:  # freeze 
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
        param_dicts = [
            {
                "params": [p for n, p in net.named_parameters() if (("backbone_d" in n) and p.requires_grad)],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,  # ???
            },
        ]
    elif only_head:  # train head
        for n, p in net.named_parameters(): 
            if 'box_head' not in n:  # freeze 
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
        param_dicts = [
            {
                "params": [p for n, p in net.named_parameters() if (("box_head" in n) and p.requires_grad)],
                # "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,  # ???
                "lr": cfg.TRAIN.LR,
            },
        ]
    elif only_fusion:  # train fusion
        for n, p in net.named_parameters(): 
            if 'box_head' not in n and 'fusion' not in n:  # freeze 
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
        param_dicts = [
            {
                "params": [p for n, p in net.named_parameters() if (("fusion" in n) and p.requires_grad)],
                # "lr": cfg.TRAIN.LR * cfg.TRAIN.FUSION_MULTIPLIER,  # ???  # cfg.TRAIN.FUSION_MULTIPLIER == 0.01
                "lr": cfg.TRAIN.LR,
            },
        ]
    elif fusion_and_head:  # train fusion
        for n, p in net.named_parameters(): 
            if 'box_head' not in n and 'fusion' not in n:  # freeze 
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
        param_dicts = [
            {
                # "params": [p for n, p in net.named_parameters() if (("fusion" in n) and p.requires_grad)],
                "params": [p for n, p in net.named_parameters() if p.requires_grad],
                # "lr": cfg.TRAIN.LR * cfg.TRAIN.FUSION_MULTIPLIER,  # ???  # cfg.TRAIN.FUSION_MULTIPLIER == 0.01
                "lr": cfg.TRAIN.LR,
            },
        ]
    elif fusion_and_head_and_backbone:  # train fusion
        for n, p in net.named_parameters(): 
            if 'box_head' not in n and 'fusion' not in n and 'backbone_d' not in n:  # freeze 
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
        param_dicts = [
            {
                # "params": [p for n, p in net.named_parameters() if (("fusion" in n) and p.requires_grad)],
                "params": [p for n, p in net.named_parameters() if p.requires_grad],
                # "lr": cfg.TRAIN.LR * cfg.TRAIN.FUSION_MULTIPLIER,  # ???  # cfg.TRAIN.FUSION_MULTIPLIER == 0.01
                "lr": cfg.TRAIN.LR,
            },
        ]
    elif only_fusion_weight:  # train fusion
        for n, p in net.named_parameters(): 
            if 'box_head' not in n and 'fusion' not in n:  # freeze 
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
        param_dicts = [
            {
                "params": [p for n, p in net.named_parameters() if (("learnable_weight" in n) and p.requires_grad)],
                # "lr": cfg.TRAIN.LR * cfg.TRAIN.FUSION_MULTIPLIER,  # ???  # cfg.TRAIN.FUSION_MULTIPLIER == 0.01
                "lr": cfg.TRAIN.LR,
            },
        ]
    elif center_head:  # train fusion
        for n, p in net.named_parameters(): 
            if 'box_center_head' not in n:  # freeze 
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
        param_dicts = [
            {
                # "params": [p for n, p in net.named_parameters() if (("learnable_weight" in n) and p.requires_grad)],  # yuan
                "params": [p for n, p in net.named_parameters() if p.requires_grad],
                # "lr": cfg.TRAIN.LR * cfg.TRAIN.FUSION_MULTIPLIER,  # ???  # cfg.TRAIN.FUSION_MULTIPLIER == 0.01
                "lr": cfg.TRAIN.LR,
            },
        ]



    # elif only_backbone_d:  # train all backbone_d layer  freeze stage 0???

    else: # train network except for score prediction module
        for n, p in net.named_parameters():
            if "score" in n:
                p.requires_grad = False
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.LR_DROP_EPOCH,
                                                            gamma=cfg.TRAIN.SCHEDULER.DECAY_RATE)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
