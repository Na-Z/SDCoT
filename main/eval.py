""" Evaluation phase for incremental 3D object detection.

Author: Zhao Na
Date: Oct, 2020
"""

import os
import sys
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'cfg'))
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'trainers'))
from opts import parse_args
from logger import init_logger
from model import create_detection_model, load_detection_model
from loss_helper import get_supervised_loss
from ap_helper import APCalculator, parse_predictions, parse_groundtruths
from train_fs import my_worker_init_fn


def evaluate(args, model, dataloader, logger, device, dataset_config):
    logger.cprint(str(datetime.now()))
    # Used for AP calculation
    CONFIG_DICT = {'remove_empty_box': (not args.faster_eval), 'use_3d_nms': args.use_3d_nms,
                   'nms_iou': args.nms_iou, 'use_old_type_nms': args.use_old_type_nms,
                   'cls_nms': args.use_cls_nms, 'per_class_proposal': args.per_class_proposal,
                   'conf_thresh': args.conf_thresh, 'dataset_config': dataset_config}

    ap_calculator = APCalculator(args.ap_iou_threshold, dataset_config.class2type)

    stat_dict = {}
    model.eval()  # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(dataloader):

        if batch_idx % 50 == 0:
            print('Eval batch: %d' % (batch_idx))

        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = model(batch_data_label['point_clouds'])

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = get_supervised_loss(end_points, dataset_config)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # log statstics
    for key in sorted(stat_dict.keys()):
        logger.cprint('eval mean %s: %f' % (key, stat_dict[key] / float(batch_idx + 1)))

    # Evaluate average precision
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        logger.cprint('eval %s: %f' % (key, metrics_dict[key]))


def main(args):
    logger = init_logger(args)

    # ======== Init Dataset =========
    if args.method == 'basetrain':
        EVAL_ALL_CLASS = False
    else:
        EVAL_ALL_CLASS = True

    if args.dataset == 'sunrgbd':
        from sunrgbd_val import SunrgbdValDataset
        test_dataset = SunrgbdValDataset(all_classes=EVAL_ALL_CLASS,
                                         num_novel_class=args.n_novel_class,
                                         num_points=args.num_point,
                                         use_color=args.use_color,
                                         use_height=(not args.no_height),
                                         augment=False)

    elif args.dataset == 'scannet':
        from scannet_val import ScannetValDataset
        test_dataset = ScannetValDataset(all_classes=EVAL_ALL_CLASS,
                                         num_novel_class=args.n_novel_class,
                                         num_points=args.num_point,
                                         use_color=args.use_color,
                                         use_height=(not args.no_height),
                                         augment=False)
    else:
        print('Unknown dataset %s. Exiting...' % (args.dataset))
        exit(-1)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.batch_size//2, worker_init_fn=my_worker_init_fn)
    DATASET_CONFIG = test_dataset.dataset_config

    model = create_detection_model(args, DATASET_CONFIG)
    if args.model_checkpoint_path is not None:
        if args.method == 'finetune':
            model_checkpoint = torch.load(os.path.join(ROOT_DIR, args.model_checkpoint_path, 'checkpoint.tar'))
            base_classifier_weights = np.load(os.path.join(ROOT_DIR, args.model_checkpoint_path,
                                                                 'base_classifier_weights.npy'))
            base_classifier_weights = torch.from_numpy(base_classifier_weights).cuda()
            loaded_model_state_dict = model_checkpoint['model_state_dict']
            model_state_dict = model.state_dict()

            for k in loaded_model_state_dict:
                if k in model_state_dict:
                    if loaded_model_state_dict[k].shape != model_state_dict[k].shape:
                        print('For %s, concatenate base and novel classifier weights...' %k)
                        assert loaded_model_state_dict[k].shape[0] + base_classifier_weights.shape[0] == \
                               model_state_dict[k].shape[0]
                        loaded_model_state_dict[k] = torch.cat((loaded_model_state_dict[k], base_classifier_weights),
                                                                dim=0)
                else:
                    print('Drop parameter {}.'.format(k))

            for k in model_state_dict:
                if not (k in loaded_model_state_dict):
                    print('No param {}.'.format(k))
                    loaded_model_state_dict[k] = model_state_dict[k]

            model.load_state_dict(loaded_model_state_dict, strict=True)
        else:
            model = load_detection_model(model, args.model_checkpoint_path, model_name=args.model_name)
    else:
        raise ValueError('Detection model checkpoint path must be given!')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    evaluate(args, model, test_dataloader, logger, device, DATASET_CONFIG)


if __name__ == '__main__':
    args = parse_args()
    main(args)