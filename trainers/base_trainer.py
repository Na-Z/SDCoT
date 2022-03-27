""" Base Learner for first-stage training in incremental few-shot 3D object detection

Author: Zhao Na
Date: September, 2020
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from model import create_detection_model, load_detection_model
from loss_helper import get_supervised_loss
from pytorch_utils import BNMomentumScheduler


class BaseTrainer(object):
    def __init__(self, args, dataset_config):

        self.dataset_config = dataset_config

        # build model
        self.model = create_detection_model(args, dataset_config)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use %d GPUs!" % (torch.cuda.device_count()))
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        # init optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # set learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_decay_steps,
                                                           gamma=args.lr_decay_rate)

        it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler
        self.start_epoch = 0

        # load checkpoint if there is any
        if args.model_checkpoint_path is not None:
            self.model, self.optimizer, self.start_epoch = load_detection_model(self.model,
                                                                                args.model_checkpoint_path,
                                                                                optimizer=self.optimizer,
                                                                                lr=args.lr,
                                                                                lr_step=args.lr_decay_steps,
                                                                                lr_rate=args.lr_decay_rate)

        # Decay Batchnorm momentum from 0.5 to 0.999
        # note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
        BN_MOMENTUM_INIT = 0.5
        BN_MOMENTUM_MAX = 0.001
        bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * args.bn_decay_rate ** (int(it / args.bn_decay_step)), BN_MOMENTUM_MAX)
        self.bnm_scheduler = BNMomentumScheduler(self.model, bn_lambda=bn_lbmd, last_epoch=self.start_epoch - 1)

    def train_batch(self, batch_data_label):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        end_points = self.model(batch_data_label['point_clouds'])

        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = get_supervised_loss(end_points, self.dataset_config)
        loss.backward()
        self.optimizer.step()

        return end_points

    def eval_batch(self, batch_data_label):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(self.device)

        # Forward pass
        with torch.no_grad():
            end_points = self.model(batch_data_label['point_clouds'])

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = get_supervised_loss(end_points, self.dataset_config)
        return end_points