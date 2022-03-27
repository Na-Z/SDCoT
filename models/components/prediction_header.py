""" Prediction header to predict the 3D object parameters (objectness, center, size, orientation, and semantic class)

Author: Zhao Na
Date: Oct, 2020
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictionHeader(nn.Module):
    def __init__(self, num_class, num_heading_bin, mean_size_arr, proposal_feature_dim):
        """ Prediction headers that takes the proposal features and outputs two branches of
            predictions, i.e., box parameters via regressor, and box category via classifier.
            Args:
                propoal_feature_dim:
                    number of channels of input features for predicting module
        """
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.mean_size_arr = mean_size_arr
        self.proposal_feature_dim = proposal_feature_dim

        self.regressor = nn.Conv1d(proposal_feature_dim, 2+3+3+2*num_heading_bin, 1)

        self.classifier_weights = nn.Parameter(torch.ones(num_class, proposal_feature_dim, 1))
        torch.nn.init.kaiming_normal_(self.classifier_weights)

    def forward(self, end_points, proposal_features, classifier_weights=None, mean_size_arr=None):
        """
        Args:
            end_points: dict
                {aggregated_vote_xyz}
            proposal_features: (batch_size, proposal_feature_dim, num_proposal)
            classifier_weights: (num_class_final, proposal_feature_dim)
        Returns:
             end_points
        """

        out_regressor = self.regressor(proposal_features) #(batch_size, 2+3+3+2*num_heading_bin, num_proposal)

        # decode_scores from regressor
        out_regressor = out_regressor.transpose(1,2)
        objectness_scores = out_regressor[:, :, 0:2]
        end_points['objectness_scores'] = objectness_scores

        base_xyz = end_points['aggregated_vote_xyz']  # (batch_size, num_proposal, 3)
        center = base_xyz + out_regressor[:, :, 2:5]  # (batch_size, num_proposal, 3)
        end_points['center'] = center

        size_residuals_normalized = out_regressor[:, :, 5:8]  # Bxnum_proposalx3
        end_points['size_residuals_normalized'] = size_residuals_normalized

        heading_scores = out_regressor[:, :, 8:8+self.num_heading_bin]
        heading_residuals_normalized = out_regressor[:, :, 8+self.num_heading_bin:8+self.num_heading_bin*2]
        end_points['heading_scores'] = heading_scores  # Bxnum_proposalxnum_heading_bin
        end_points['heading_residuals_normalized'] = heading_residuals_normalized  # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
        end_points['heading_residuals'] = heading_residuals_normalized * (np.pi / self.num_heading_bin)  # Bxnum_proposalxnum_heading_bin

        if mean_size_arr is None:
            mean_size_arr = self.mean_size_arr
        if classifier_weights is None:
            classifier_weights = self.classifier_weights

        out_classifier = F.conv1d(proposal_features, classifier_weights, bias=None) #(batch_size, num_class_final, num_proposal)

        # decode_scores from classifier
        sem_cls_scores = out_classifier.transpose(1, 2)  # Bxnum_proposalxnum_class_final
        end_points['sem_cls_scores'] = sem_cls_scores
        sem_cls = torch.argmax(sem_cls_scores, dim=2)  # Bxnum_proposals
        end_points['sem_cls'] = sem_cls
        mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda()
        selected_mean_size_arr = mean_size_arr[sem_cls.view(-1)]
        selected_mean_size_arr = selected_mean_size_arr.view(sem_cls.shape[0], sem_cls.shape[1], 3)  # Bxnum_proposalx3
        end_points['size_residuals'] = size_residuals_normalized * selected_mean_size_arr

        return end_points

    def enroll_weights(self, base_weights):
        """ Enroll base/novel weights for the classifier
        Args:
            base_weights: (num_base_classes, proposal_feature_dim) tensor
        Returns:
            classifier_weights: (num_class_final, proposal_feature_dim, 1) tensor
                updated weights of classifier, which combine the weights of base and novel classes
        """
        classifier_weights = self.classifier_weights.detach().squeeze(-1)
        classifier_weights = torch.cat((base_weights, classifier_weights), dim=0)

        return classifier_weights.unsqueeze(-1)