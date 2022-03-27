""" Object proposal generator module, including vote clustering and proposal feature extraction.

Author: Zhao Na
Date: September, 2020
"""
import numpy as np
import torch
import torch.nn as nn
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from pointnet2_modules import PointnetSAModuleVotes
import pointnet2_utils


class ProposalGenerator(nn.Module):
    """ Proposal generator that generates proposal features from vote clusters

            Args:
            ----------
            num_proposal: int (default: 128)
                Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
            sampling: str (default: 'vote_fps')
                The sampling strategy to get a subset of votes. Options: ['vote_fps', 'seed_fps', 'random']
            seed_feat_dim: int (default=256)
                The number of channels of seed point/vote features
    """

    def __init__(self, num_proposal=128, sampling='vote_fps', seed_feat_dim=256, proposal_feat_dim=128):
        super().__init__()

        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        self.radius = 0.3
        self.sigma = self.radius/2
        self.mlp_spec = [seed_feat_dim, 128, 128, proposal_feat_dim]
        self.pooling = 'max'

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
                npoint=self.num_proposal,
                radius=self.radius,
                nsample=16,
                mlp=self.mlp_spec,
                use_xyz=True,
                pooling=self.pooling,
                sigma=self.sigma,
                normalize_xyz=True
            )

        #add two more mlp layers to generate the final proposal features
        self.mlp_module = nn.Sequential(nn.Conv1d(proposal_feat_dim, 128, 1),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Conv1d(128,proposal_feat_dim,1),
                                        nn.BatchNorm1d(proposal_feat_dim),
                                        nn.ReLU())

    def forward(self, end_points, vote_fps_inds=None):
        """
            Args:
                end_points: dict
                    {seed_inds, seed_xyz, vote_xyz, vote_features}
            Returns:
                end_points
        """
        xyz = end_points['vote_xyz']
        features = end_points['vote_features']

        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features, vote_fps_inds)
            end_points['vote_fps_inds'] = fps_inds
            # sample_inds = fps_inds
        elif self.sampling == 'seed_fps':
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            # Note: do not use this sampling manner when vote_factor>1 or num_class>1
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            batch_size = xyz.shape[0]
            num_seed = xyz.shape[1]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            print('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        features = self.mlp_module(features)

        return xyz, features