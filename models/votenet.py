# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany

Modified by Zhao Na
Date: September, 2020
"""

import torch
import torch.nn as nn
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'components'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_generator import ProposalGenerator
from prediction_header import PredictionHeader


class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        mean_size_arr: numpy array, shape (num_class_final, 3)
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, mean_size_arr, input_feature_dim=0,
                       num_proposal=128, vote_factor=1, sampling='vote_fps', seed_feature_dim=256,
                       proposal_feature_dim=128):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.mean_size_arr = mean_size_arr
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=input_feature_dim,
                                              seed_feature_dim=seed_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, seed_feature_dim)

        self.pgen = ProposalGenerator(num_proposal, sampling, seed_feat_dim=seed_feature_dim,
                                                    proposal_feat_dim=proposal_feature_dim)

        self.prediction_header = PredictionHeader(num_class, num_heading_bin, mean_size_arr, proposal_feature_dim)

    def forward(self, point_clouds, classifier_weights=None, mean_size_arr=None):
        """ Forward pass of the network

        Args:
            point_clouds: Variable(torch.cuda.FloatTensor) (B, N, 3 + input_channels) tensor
                          Point cloud to run predicts on
                          Each point in the point-cloud MUST be formated as (x, y, z, features...)
            classifier_weights: (num_class_final, proposal_feature_dim, 1) tensor
                          updated weights of classifier, which combine the weights of base and novel classes
            mean_size_arr: numpy array, shape (num_class_final, 3)
        Returns:
            end_points: dict
                        Use to store intermediate features and output predictions
        """


        end_points = self.backbone_net(point_clouds)

        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz

        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        xyz, features = self.pgen(end_points)
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)

        end_points = self.prediction_header(end_points, features, classifier_weights, mean_size_arr)

        return end_points