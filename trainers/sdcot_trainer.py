""" Trainer for incremental learning with novel class data (second stage training) in incremental 3D object detection

Author: Zhao Na
Date: Oct, 2020
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from model import create_detection_model, load_detection_model
from loss_helper import get_supervised_loss, get_consistency_loss, get_distillation_loss


class SDCoTTrainer(object):
    def __init__(self, args, data_config, base_data_config):

        self.data_config = data_config
        self.ema_decay = args.ema_decay

        # build student model
        print('+++ Init student detector +++')
        self.model = create_detection_model(args, data_config)

        # initialize student detection model with checkpoint from first-stage training
        if args.model_checkpoint_path is not None:
            self.model = load_detection_model(self.model, args.model_checkpoint_path)
        else:
            raise ValueError('Detection model checkpoint path must be given!')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.classifier_weights_base = torch.empty_like(self.model.prediction_header.classifier_weights).copy_(
                                            self.model.prediction_header.classifier_weights.detach())
        self.classifier_weights_base = self.classifier_weights_base.squeeze(-1)

        # init last layer for class prediction
        self.init_classifier_weights(data_config.num_class_final)

        # build dynamic teacher model
        print('+++ Init dynamic teacher detector +++')
        self.ema_model = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.detach_()

        # build static teacher models
        print('+++ Init static teacher detector +++')
        self.base_model = create_detection_model(args, base_data_config)
        self.base_model = load_detection_model(self.base_model, args.model_checkpoint_path)
        self.base_model.to(self.device)
        self.base_model.eval()

        # init optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # set learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_decay_steps,
                                                           gamma=args.lr_decay_rate)


    def init_classifier_weights(self, num_class_final):
        num_base_class, proposal_feature_dim = self.classifier_weights_base.size()
        classifier_weights_novel = torch.empty(num_class_final-num_base_class, proposal_feature_dim).cuda()
        nn.init.kaiming_normal_(classifier_weights_novel)
        classifier_weights = torch.cat((self.classifier_weights_base, classifier_weights_novel), dim=0)
        self.model.prediction_header.classifier_weights = nn.Parameter(classifier_weights.unsqueeze(-1),
                                                                       requires_grad=True)

    def set_train_mode(self):
        self.model.train()  # set model to training mode
        self.ema_model.train()

    def train_batch(self, batch_data_label, global_step, consistency_weight, distillation_weight):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        end_points = self.model(batch_data_label['point_clouds'])
        ema_end_points = self.ema_model(batch_data_label['ema_point_clouds'])

        with torch.no_grad():
            vote_fps_inds = end_points['vote_fps_inds'].clone()
            reference_end_points = self.base_model.backbone_net(batch_data_label['point_clouds'],
                                                                sa1_inds=end_points['sa1_inds'],
                                                                sa2_inds=end_points['sa2_inds'],
                                                                sa3_inds=end_points['sa3_inds'],
                                                                sa4_inds=end_points['sa4_inds'])
            xyz, features = self.base_model.vgen(reference_end_points['fp2_xyz'], reference_end_points['fp2_features'])
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            reference_end_points['vote_xyz'] = xyz
            reference_end_points['vote_features'] = features
            xyz, features = self.base_model.pgen(reference_end_points, vote_fps_inds)
            reference_end_points['aggregated_vote_xyz'] = xyz  # (batch_size, num_proposal, 3)
            reference_end_points = self.base_model.prediction_header(reference_end_points, features)

        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]

        supervised_loss, end_points = get_supervised_loss(end_points, self.data_config)
        consistency_loss, end_points = get_consistency_loss(end_points, ema_end_points, self.data_config)
        distillation_loss, end_points = get_distillation_loss(end_points, reference_end_points)
        loss = supervised_loss + consistency_weight*consistency_loss + distillation_weight*distillation_loss
        loss.backward()
        self.optimizer.step()

        global_step += 1
        self.update_ema_variables(self.ema_decay, global_step)

        return end_points, global_step

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
        loss, end_points = get_supervised_loss(end_points, self.data_config)
        return end_points

    def update_ema_variables(self, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
