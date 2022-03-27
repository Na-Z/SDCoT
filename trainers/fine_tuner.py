""" Fine-tuner for fine-tuning with novel class data in incremental few-shot 3D object detection

Author: Zhao Na
Date: Oct, 2020
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


class FineTuner(object):
    def __init__(self, args, base_data_config, train_data_config, valid_data_config):

        self.train_data_config = train_data_config
        self.valid_data_config = valid_data_config
        self.ft_layers = args.ft_layers

        # build model
        self.model = create_detection_model(args, base_data_config)

        # initialize query detection model with checkpoint from first-stage training
        if args.model_checkpoint_path is not None:
            self.model = load_detection_model(self.model, args.model_checkpoint_path)
        else:
            raise ValueError('Detection model checkpoint path must be given!')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.classifier_weights_base = torch.empty_like(self.model.prediction_header.classifier_weights).copy_(
                                            self.model.prediction_header.classifier_weights.detach())
        self.classifier_weights_base = self.classifier_weights_base.squeeze(-1)

        if args.ft_layers == 'last':
            self._freeze_detection_model(frozen_params=['backbone_net', 'vgen', 'pgen',
                                                        'prediction_header.regressor'])
        elif args.ft_layers == 'all':
            pass
        else:
            print('Unknown input for funetune_layers argument %s. Exiting...' % (args.ft_layers))
            exit(1)

        print('------------ VoteNet parameters -------------')
        for name, param in self.model.named_parameters():
            print('{0} | trainable: {1}'.format(name, param.requires_grad))
        print('---------------------------------------------\n')

        # random init last layer for class prediction
        self.init_classifier_weights_random(train_data_config.num_class)

        # init optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)

        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_decay_steps,
                                                           gamma=args.lr_decay_rate)

    def _freeze_detection_model(self, frozen_params=None):
        """
        Freeze the modules according to the frozen_params
        """
        for name, param in self.model.named_parameters():
            if sum([name.startswith(frozen_key) for frozen_key in frozen_params]):
                param.requires_grad = False

    def init_classifier_weights_random(self, num_class):
        proposal_feature_dim = self.model.prediction_header.classifier_weights.shape[1]
        self.model.prediction_header.classifier_weights = nn.Parameter(torch.empty(
                                num_class, proposal_feature_dim, 1).cuda(), requires_grad=True)
        nn.init.kaiming_normal_(self.model.prediction_header.classifier_weights)

    def set_train_mode(self):
        if self.ft_layers == 'all':
            self.model.train()  # set model to training mode
        if self.ft_layers == 'last':
            self.model.eval()

    def train_batch(self, batch_data_label):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        end_points = self.model(batch_data_label['point_clouds'], mean_size_arr=self.train_data_config.mean_size_arr)

        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = get_supervised_loss(end_points, self.train_data_config)
        loss.backward()
        self.optimizer.step()

        return end_points

    def eval_batch(self, batch_data_label):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(self.device)

        # Forward pass
        with torch.no_grad():
            classifier_weights = self.model.prediction_header.enroll_weights(self.classifier_weights_base)
            end_points = self.model(batch_data_label['point_clouds'], classifier_weights=classifier_weights,
                                    mean_size_arr=self.valid_data_config.mean_size_arr)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = get_supervised_loss(end_points, self.valid_data_config)
        return end_points