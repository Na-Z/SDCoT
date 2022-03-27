""" Baseline method [Fine-tuning] for incremental few-shot 3D object detection.

Author: Zhao Na
Date: Oct, 2020
"""
import os
import sys
import numpy as np
from datetime import datetime
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
from fine_tuner import FineTuner
from logger import init_logger
from model import save_model
from ap_helper import APCalculator, parse_predictions, parse_groundtruths
from train_bt import evaluate_one_epoch, my_worker_init_fn


def train_one_epoch(trainer, dataloader, logger, writer, epoch_idx, batch_size, batch_interval=20):
    stat_dict = {}  # collect statistics

    trainer.set_train_mode()

    for batch_idx, batch_data_label in enumerate(dataloader):
        end_points = trainer.train_batch(batch_data_label)

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


def main(args, eval_interval=10):
    logger = init_logger(args)

    os.system('cp trainers/fine_tuner.py %s' % (args.log_dir))

    # Init Dataset and Dataloader
    if args.dataset == 'sunrgbd':
        from sunrgbd import SunrgbdBaseDatasetConfig
        from sunrgbd_novel import SunrgbdNovelDataset
        from sunrgbd_val import SunrgbdValDataset
        train_dataset = SunrgbdNovelDataset(num_novel_class=args.n_novel_class,
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
        BASE_DATA_CONFIG = SunrgbdBaseDatasetConfig()
    elif args.dataset == 'scannet':
        from scannet import ScannetBaseDatasetConfig
        from scannet_novel import ScannetNovelDataset
        from scannet_val import ScannetValDataset
        train_dataset = ScannetNovelDataset(num_novel_class=args.n_novel_class,
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
        BASE_DATA_CONFIG = ScannetBaseDatasetConfig()
    else:
        print('Unknown dataset %s. Exiting...' % (args.dataset))
        exit(-1)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.batch_size//2, worker_init_fn=my_worker_init_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.batch_size//2, worker_init_fn=my_worker_init_fn)

    NOVEL_DATA_CONFIG = train_dataset.dataset_config
    VAL_DATA_CONFIG = valid_dataset.dataset_config
    trainer = FineTuner(args, BASE_DATA_CONFIG, NOVEL_DATA_CONFIG, VAL_DATA_CONFIG)

    writer = SummaryWriter(log_dir=args.log_dir)

    # Used for AP calculation
    config_dict = {'remove_empty_box': False, 'use_3d_nms': True,
                   'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
                   'per_class_proposal': True, 'conf_thresh': 0.05,
                   'dataset_config': VAL_DATA_CONFIG}

    num_train_samples = len(train_dataloader) * args.batch_size
    for epoch in range(args.n_epochs):
        logger.cprint('\n**** EPOCH %03d ****' % (epoch))
        logger.cprint(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()

        train_one_epoch(trainer, train_dataloader, logger, writer, epoch, args.batch_size)

        if (epoch + 1) % eval_interval == 0:
            ap_calculator = APCalculator(ap_iou_thresh=0.25, class2type_map=VAL_DATA_CONFIG.class2type)
            evaluate_one_epoch(trainer, valid_dataloader, ap_calculator, logger, config_dict,
                                   writer, epoch, num_train_samples)

        # Save checkpoint
        save_model(args.log_dir, epoch, trainer.model, optimizer=trainer.optimizer)
        np.save(os.path.join(args.log_dir, 'base_classifier_weights'),
                trainer.classifier_weights_base.unsqueeze(-1).detach().cpu().numpy())

    logger.close()
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)