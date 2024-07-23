import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8, tracker_params=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id, tracker_params=tracker_params)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    # parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    # parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--tracker_name', type=str, default='mixformer_cvt_online', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='baseline', help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    # parser.add_argument('--dataset_name', type=str, default='depthtrack', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    # parser.add_argument('--dataset_name', type=str, default='rgbd_1k', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--dataset_name', type=str, default='cdtb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=1)
    # parser.add_argument('--threads', type=int, default=8, help='Number of threads.')
    # parser.add_argument('--num_gpus', type=int, default=2)

    # parser.add_argument('--params__model', type=str, default='', help="Tracking model path.")
    # parser.add_argument('--params__model', type=str, default='networktest_cvt.pth', help="Tracking model path.")
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0023_boxhead.pth.tar', help="Tracking model path.")
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0040_547_centerhead.pth.tar', help="Tracking model path.")
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0030_baseline.pth.tar', help="Tracking model path.")  
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0023_maybe_baseline.pth.tar', help="Tracking model path.")
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0023_boxhead_baseline_false_train_backbone_d.pth.tar', help="Tracking model path.")
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0025_744_v5.pth.tar', help="Tracking model path.")
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0029_v5plus3_ar0760.pth.tar', help="Tracking model path.")
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0020_v2.pth.tar', help="Tracking model path.")  # no train  
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0005_bseline_0723.pth.tar', help="Tracking model path.")
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0015_ft15_741_v1.pth.tar', help="Tracking model path.")
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0005_742_v2.pth.tar', help="Tracking model path.").
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0036_741_v3.pth.tar', help="Tracking model path.")
    parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0011.pth.tar', help="Tracking model path.")
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0001_rgbd1k_0560.pth.tar', help="Tracking model path.")
    # parser.add_argument('--params__model', type=str, default='MixFormerOnlineScore_ep0023_v5plus3_e4d2_0757_ar_0764.pth.tar', help="Tracking model path.")

    parser.add_argument('--params__update_interval', type=int, default=None, help="Update interval of online tracking.")
    parser.add_argument('--params__online_sizes', type=int, default=None)
    parser.add_argument('--params__search_area_scale', type=float, default=None)  # cai jian qu yu bianchang fang da bei shu  # crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    parser.add_argument('--params__max_score_decay', type=float, default=1.0)  # yuan
    # parser.add_argument('--params__max_score_decay', type=float, default=0.99)
    parser.add_argument('--params__vis_attn', type=int, choices=[0, 1], default=0, help="Whether visualize the attention maps.")

    parser.add_argument('--params__AR', type=bool, default=False, help="Whether USE AR.")  
    # /media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/AR/ltr/checkpoints/ltr/SEx_beta
    parser.add_argument('--params__AR_PATH', type=str, default='/media/dell/f62e06ef-019a-4728-bd1f-906bcb35e057/MixFormer-main-kyz/external/AR/ltr/checkpoints/ltr/SEx_beta/SEcmnet_ep0040-c.pth.tar', help="Whether USE AR.")

    args = parser.parse_args()

    tracker_params = {}
    for param in list(filter(lambda s: s.split('__')[0] == 'params' and getattr(args, s) != None, args.__dir__())):
        tracker_params[param.split('__')[1]] = getattr(args, param)
    print(tracker_params)

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus, tracker_params=tracker_params)


if __name__ == '__main__':
    main()
