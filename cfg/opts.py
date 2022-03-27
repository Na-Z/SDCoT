""" Argument configurations for incremental few-shot 3D object detection.

Author: Zhao Na
Date: September, 2020
"""
import os
import argparse
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def parse_args():
    parser = argparse.ArgumentParser()

    # basic experimental setting
    parser.add_argument('--phase', type=str, default='train', choices=['train','test'])
    parser.add_argument('--method', type=str, default='basetrain', choices=['basetrain', 'finetune', 'SDCoT'])
    parser.add_argument('--dataset', type=str, default='sunrgbd', help='Dataset name: scannet|sunrgbd')

    # model config
    parser.add_argument('--model_name', default=None, help='set to [teacher] if uses teacher model to evaluate')
    parser.add_argument('--model_checkpoint_path', default=None,
                        help='Detection model checkpoint path [default: None]')
    parser.add_argument('--num_target', type=int, default=128, help='Proposal number [default: 128]')
    parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
    parser.add_argument('--cluster_sampling', default='vote_fps',
                        help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')

    # input and dataset
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number')
    parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--pc_augm', action='store_true', help='Data augmentation')

    # basic learning settings
    parser.add_argument('--n_epochs', type=int, default=150, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during base training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_steps', default='80,120', help='When to decay the learning rate (in epochs)')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Decay rates for lr decay')
    parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs)')
    parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay')

    # finetune
    parser.add_argument('--ft_layers', default='all',
                            help='The module(s) to finetune/train with novel class data, Options: [last, all]')

    # incremental learning
    parser.add_argument('--pseudo_obj_conf_thresh', type=float, default=0.95,
                        help='Confidence score threshold w.r.t. objectness prediction for hard selection of psuedo bboxes')
    parser.add_argument('--pseudo_cls_conf_thresh', type=float, default=0.9,
                        help='Confidence score threshold w.r.t. class prediction for hard selection of psuedo bboxes')

    parser.add_argument('--ema-decay', type=float, default=0.999, help='ema variable decay rate')
    parser.add_argument('--consistency_weight', type=float, default=10.0,
                                                help='use consistency loss with given weight')
    parser.add_argument('--consistency_ramp_len', type=int, default=30, help='length of the consistency loss ramp-up')

    parser.add_argument('--distillation_weight', type=float, default=1.0,
                                                help='use distillation loss with given weight')
    parser.add_argument('--distillation_ramp_len', type=int, default=30, help='length of the stabilization loss ramp-up')

    # test
    parser.add_argument('--n_novel_class', type=int, default=5, help='Number of novel classes to incrementally learn')

    # evaluation config
    parser.add_argument('--ap_iou_threshold', default='0.25', help='AP IoU thresholds')
    parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
    parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
    parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
    parser.add_argument('--per_class_proposal', action='store_true',
                        help='Duplicate each proposal num_class times.')
    parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold.')
    parser.add_argument('--conf_thresh', type=float, default=0.05, help='Filter out predictions with obj prob less than it.')
    parser.add_argument('--faster_eval', action='store_true',
                        help='Faster evaluation by skippling empty bounding box removal.')

    args = parser.parse_args()

    args.num_input_channel = int(args.use_color) * 3 + int(not args.no_height) * 1
    args.lr_decay_steps = [int(x) for x in args.lr_decay_steps.split(',')]

    LOG_DIR = os.path.join(ROOT_DIR, 'log_%s' %args.dataset)
    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

    if args.phase == 'train':
        args.log_dir = os.path.join(LOG_DIR, 'log_%s_%s' % (args.method, datetime.now().strftime("%Y%m%d-%H:%M")))
    elif args.phase == 'test':
        args.log_dir = os.path.join(ROOT_DIR, args.model_checkpoint_path)
    else:
        print('Unknown phase %s. Exiting...' % (args.phase))
        exit(-1)

    return args
