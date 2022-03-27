""" Static-Dynamic Co-teaching method (SDCoT) for Incremental 3D object detection.

Author: Zhao Na
Date: Oct, 2020
"""
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'cfg'))
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
sys.path.append(os.path.join(ROOT_DIR, 'trainers'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from opts import parse_args
from sdcot_trainer import SDCoTTrainer
from logger import init_logger
import ramps
from model import save_model
from ap_helper import APCalculator, parse_predictions, parse_groundtruths
from train_bt import evaluate_one_epoch, my_worker_init_fn


def get_current_weight(epoch, weight, ramp_len):
    # ramp-up from https://arxiv.org/abs/1610.02242
    return weight * ramps.sigmoid_rampup(epoch, ramp_len)


def train_one_epoch(trainer, dataloader, logger, writer, epoch_idx, batch_size, global_step, consistency_weight,
                    consistency_ramp_len, distillation_weight, distillation_ramp_len, batch_interval=20):
    stat_dict = {}  # collect statistics
    cur_consistency_weight = get_current_weight(epoch_idx, consistency_weight, consistency_ramp_len)
    cur_distillation_weight = get_current_weight(epoch_idx, distillation_weight, distillation_ramp_len)
    logger.cprint('Current consistency weight: %f' % cur_consistency_weight)
    logger.cprint('Current distillation weight: %f' % cur_distillation_weight)
    trainer.set_train_mode()

    for batch_idx, batch_data_label in enumerate(dataloader):
        end_points, global_step = trainer.train_batch(batch_data_label, global_step, cur_consistency_weight,
                                                      cur_distillation_weight)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % batch_interval == 0:
            logger.cprint(' ---- batch: %03d ----' % (batch_idx + 1))

            for key in sorted(stat_dict.keys()):
                stat =  stat_dict[key]
                step = (epoch_idx * len(dataloader) + batch_idx) * batch_size
                logger.cprint('train mean %s: %f' % (key, stat))
                writer.add_scalar('Train/'+key, stat, step)
                stat_dict[key] = 0

    trainer.lr_scheduler.step()

    return global_step


def main(args, eval_interval=10):
    logger = init_logger(args)

    # Init Dataset and Dataloader
    if args.dataset == 'sunrgbd':
        from sunrgbd import SunrgbdBaseDatasetConfig
        from sunrgbd_inc import SunrgbdIncDataset
        from sunrgbd_val import SunrgbdValDataset
        base_model_config = SunrgbdBaseDatasetConfig()
        train_dataset = SunrgbdIncDataset(args, num_novel_class=args.n_novel_class,
                                          num_points=args.num_point,
                                          use_color=args.use_color,
                                          use_height=(not args.no_height),
                                          augment=args.pc_augm)
        valid_dataset = SunrgbdValDataset(all_classes=True,
                                          num_novel_class=args.n_novel_class,
                                          num_points=args.num_point,
                                          use_color=args.use_color,
                                          use_height=(not args.no_height),
                                          augment=False)
    elif args.dataset == 'scannet':
        from scannet import ScannetBaseDatasetConfig
        from scannet_inc import ScannetIncDataset
        from scannet_val import ScannetValDataset
        base_model_config = ScannetBaseDatasetConfig()
        train_dataset = ScannetIncDataset(args, num_novel_class=args.n_novel_class,
                                          num_points=args.num_point,
                                          use_color=args.use_color,
                                          use_height=(not args.no_height),
                                          augment=args.pc_augm)
        valid_dataset = ScannetValDataset(all_classes=True,
                                          num_novel_class=args.n_novel_class,
                                          num_points=args.num_point,
                                          use_color=args.use_color,
                                          use_height=(not args.no_height),
                                          augment=False)
    else:
        print('Unknown dataset %s. Exiting...' % (args.dataset))
        exit(-1)

    # use CUDA in multiprocessing
    # REF: https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    torch.multiprocessing.set_start_method('spawn')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=3, worker_init_fn=my_worker_init_fn, timeout=20)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=3, worker_init_fn=my_worker_init_fn)

    trainer = SDCoTTrainer(args, train_dataset.dataset_config, base_model_config)

    writer = SummaryWriter(log_dir=args.log_dir)

    VAL_DATA_CONFIG = valid_dataset.dataset_config
    # Used for AP calculation
    config_dict = {'remove_empty_box': False, 'use_3d_nms': True,
                   'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
                   'per_class_proposal': True, 'conf_thresh': 0.05,
                   'dataset_config': VAL_DATA_CONFIG}

    num_train_samples = len(train_dataloader) * args.batch_size
    global_step = 0
    for epoch in range(args.n_epochs):
        logger.cprint('\n**** EPOCH %03d, STEP %d ****' % (epoch, global_step))
        logger.cprint(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()

        global_step = train_one_epoch(trainer, train_dataloader, logger, writer, epoch, args.batch_size, global_step,
                                      args.consistency_weight, args.consistency_ramp_len,
                                      args.distillation_weight, args.distillation_ramp_len)

        if (epoch + 1) % eval_interval == 0:
            ap_calculator = APCalculator(ap_iou_thresh=0.25, class2type_map=VAL_DATA_CONFIG.class2type)

            evaluate_one_epoch(trainer, valid_dataloader, ap_calculator, logger, config_dict,
                               writer, epoch, num_train_samples)

        # Save checkpoint
        save_model(args.log_dir, epoch, trainer.model, optimizer=trainer.optimizer)
        save_model(args.log_dir, epoch, trainer.ema_model, model_name='teacher')

    logger.close()
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
