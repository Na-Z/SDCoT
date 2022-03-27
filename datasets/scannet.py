# coding: utf-8

"""ScanNet Dataset Basic Class for training and testing data organization
Acknowledge: https://github.com/facebookresearch/votenet/blob/master/scannet/scannet_detection_dataset.py

Author: Zhao Na
Date: Oct, 2020
"""

import os
import sys
import numpy as np
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'cfg'))
sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
import pc_util
import box_util
from scannet_cfg import cfg, get_class2scans
import scannet_utils


class ScannetBaseDatasetConfig(object):
    def __init__(self):
        self.dataset = 'scannet'
        self.num_class = cfg.NUM_BASE_CLASSES #the number of_classes in the original classifier of predicting module
        self.num_class_final = cfg.NUM_BASE_CLASSES # the number of classes in the final classifier of predicting module
        self.num_heading_bin = cfg.NUM_HEADING_BIN

        self.nyu40ids = cfg.BASE_NYUIDS
        self.nyu40id2class = {nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))}
        self.types = cfg.BASE_TYPES
        self.type2class = {t:c for (c, t) in enumerate(self.types)}
        self.class2type = {c:t for (c, t) in enumerate(self.types)}

        self.type_mean_size = cfg.TYPE_MEAN_SIZE
        self.mean_size_arr = np.zeros((self.num_class_final, 3))
        for i in range(self.num_class_final):
            self.mean_size_arr[i, :] = self.type_mean_size[self.class2type[i]]


class ScannetNovelDatasetConfig(object):
    def __init__(self, num_novel_classes):
        self.dataset = 'scannet'
        self.num_class = num_novel_classes
        self.num_class_final = self.num_class
        self.num_heading_bin = cfg.NUM_HEADING_BIN

        self.nyu40ids = cfg.NOVEL_NYUIDS[:num_novel_classes]
        self.nyu40id2class = {nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))}
        self.types = cfg.NOVEL_TYPES[:num_novel_classes]
        self.type2class = {t:c for (c, t) in enumerate(self.types)}
        self.class2type = {c:t for (c, t) in enumerate(self.types)}

        self.type_mean_size = cfg.TYPE_MEAN_SIZE
        self.mean_size_arr = np.zeros((self.num_class_final, 3))
        for i in range(self.num_class_final):
            self.mean_size_arr[i, :] = self.type_mean_size[self.class2type[i]]


class ScannetAllDatasetConfig(object):
    def __init__(self, num_novel_classes, incremental=False):
        self.dataset = 'scannet'
        if incremental:
            self.num_class = cfg.NUM_BASE_CLASSES
            self.labeled_nyu40ids = cfg.NOVEL_NYUIDS[:num_novel_classes]
            self.labeled_types = cfg.NOVEL_TYPES[:num_novel_classes]
        else:
            self.num_class = cfg.NUM_BASE_CLASSES + num_novel_classes
        self.num_class_final = cfg.NUM_BASE_CLASSES + num_novel_classes
        self.num_heading_bin = cfg.NUM_HEADING_BIN

        self.nyu40ids = np.concatenate((cfg.BASE_NYUIDS, cfg.NOVEL_NYUIDS[:num_novel_classes]))
        self.nyu40id2class = {nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))}
        self.types = cfg.BASE_TYPES + cfg.NOVEL_TYPES[:num_novel_classes]
        self.type2class = {t:c for (c, t) in enumerate(self.types)}
        self.class2type = {c:t for (c, t) in enumerate(self.types)}

        self.type_mean_size = cfg.TYPE_MEAN_SIZE
        self.mean_size_arr = np.zeros((self.num_class_final, 3))
        for i in range(self.num_class_final):
            self.mean_size_arr[i, :] = self.type_mean_size[self.class2type[i]]


class ScannetDataset(Dataset):
    def __init__(self, num_points, use_color, use_height, augment):
        super(ScannetDataset, self).__init__()

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.pc_attri_dim = 3 + 3 * self.use_color + 1 * self.use_height
        self.augment = augment

        self.data_path = os.path.join(ROOT_DIR, 'scannet')
        self.train_data_path = os.path.join(self.data_path, 'scannet_train_detection_data')
        self.val_data_path = os.path.join(self.data_path, 'scannet_val_detection_data')

    ##================================= Process One Scene =================================##
    def _process_pointcloud(self, mesh_vertices):
        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - cfg.MEAN_COLOR_RGB) / 256.0

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        return point_cloud

    def _sample_pointcloud(self, point_cloud):
        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        return point_cloud, choices

    def _augment(self, point_cloud, target_bboxes):
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            point_cloud[:, 0] = -1 * point_cloud[:, 0]
            target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

        if np.random.random() > 0.5:
            # Flipping along the XZ plane
            point_cloud[:, 1] = -1 * point_cloud[:, 1]
            target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
        rot_mat = pc_util.rotz(rot_angle)
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
        target_bboxes[:,0:6] = scannet_utils.rotate_aligned_boxes(target_bboxes[:,0:6], rot_mat)

        return point_cloud, target_bboxes

    def _generate_votes(self, point_cloud, instance_labels, semantic_labels, valid_nyu40ids):
        '''generate vote for each point according to instance labeling
           Note: since there's no map between bbox instance labels and pc instance_labels (it had been filtered in the
           data preparation step) we'll compute the instance bbox from the points sharing the same instance label.
        '''
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        for i_instance in np.unique(instance_labels):
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label
            if semantic_labels[ind[0]] in valid_nyu40ids:
                x = point_cloud[ind, :3]
                center = 0.5 * (x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3))  # make 3 votes identical

        return point_votes_mask, point_votes

    def _generate_votes_with_bboxes(self, point_cloud, target_bboxes, scan_name):
        '''generate vote for each point according to the bounding boxes (same way to generate votes as in SUN RGB-D)
           Note: one point may belong to several objects if there are overlaps between these objects

           Returns:
                point_votes_mask: shape (num_points,)
                point_votes: shape (num_points, 9)
        '''
        point_votes = np.zeros((self.num_points, 10))  # 3 votes and 1 vote mask
        point_vote_idx = np.zeros((self.num_points)).astype(np.int32)  # in the range of [0,2]
        indices = np.arange(self.num_points)
        for target_box in target_bboxes:
            try:
                # Find all points in this object's OBB
                box3d_pts_3d = scannet_utils.my_compute_box_3d(target_box[0:3], target_box[3:6])
                #TODO: visualize the computed 3D box
                pc_in_box3d, inds = box_util.extract_pc_in_box3d(point_cloud, box3d_pts_3d)
                # Assign first dimension to indicate it is in an object box
                point_votes[inds, 0] = 1
                # Add the votes (all 0 if the point is not in any object's OBB)
                votes = np.expand_dims(target_box[0:3], 0) - pc_in_box3d[:, 0:3]
                sparse_inds = indices[inds]  # turn dense True,False inds to sparse number-wise inds
                for i in range(len(sparse_inds)):
                    j = sparse_inds[i]
                    point_votes[j, int(point_vote_idx[j] * 3 + 1):int((point_vote_idx[j] + 1) * 3 + 1)] = votes[i, :]
                    # Populate votes with the fisrt vote
                    if point_vote_idx[j] == 0:
                        point_votes[j, 4:7] = votes[i, :]
                        point_votes[j, 7:10] = votes[i, :]
                point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds] + 1)
            except:
                print('ERROR ----', scan_name, target_box[6])

        return point_votes[:,0], point_votes[:,1:]

    def _generate_data_label(self, point_cloud, instance_bboxes, dataset_config, point_votes, point_votes_mask):
        '''Generate a dict storing the processed point cloud and annotations, with following keys:
                point_clouds: (N,3+C)
                center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
                sem_cls_label: (MAX_NUM_OBJ,) semantic class index
                heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
                heading_residual_label: (MAX_NUM_OBJ,)
                size_residual_label: (MAX_NUM_OBJ,3)
                box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
                vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                    if there is only one vote than X1==X2==X3 etc.
                vote_label_mask: (N,) with 0/1 with 1 indicating the point
                    is in one of the object's OBB.
        '''
        target_bboxes = np.zeros((cfg.MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((cfg.MAX_NUM_OBJ))
        angle_classes = np.zeros((cfg.MAX_NUM_OBJ,))
        angle_residuals = np.zeros((cfg.MAX_NUM_OBJ,))
        size_residuals = np.zeros((cfg.MAX_NUM_OBJ, 3))
        target_bboxes_semcls = np.zeros((cfg.MAX_NUM_OBJ))

        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        class_ind = [dataset_config.nyu40id2class[x] for x in instance_bboxes[:, -1]]
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = class_ind
        # NOTE: set size class as semantic class. Consider use size2class.
        # size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - dataset_config.mean_size_arr[class_ind, :]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        return ret_dict

    def _process_one_scene(self, scan_name, scan_data_path, dataset_config):
        '''Process one scene/scan and return the corresponding point cloud and annotations
        '''
        mesh_vertices = np.load(os.path.join(scan_data_path, scan_name) + '_vert.npy')
        instance_labels = np.load(os.path.join(scan_data_path, scan_name) + '_ins_label.npy')
        semantic_labels = np.load(os.path.join(scan_data_path, scan_name) + '_sem_label.npy')
        instance_bboxes = np.load(os.path.join(scan_data_path, scan_name) + '_bbox.npy')

        bbox_mask = np.in1d(instance_bboxes[:, -1], dataset_config.nyu40ids)
        instance_bboxes = instance_bboxes[bbox_mask, :]

        point_cloud = self._process_pointcloud(mesh_vertices)

        point_cloud, choices = self._sample_pointcloud(point_cloud)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]

        # data augmentation
        if self.augment:
            point_cloud, instance_bboxes = self._augment(point_cloud, instance_bboxes)

        # compute votes *AFTER* augmentation
        point_votes_mask, point_votes = self._generate_votes(point_cloud, instance_labels, semantic_labels,
                                                             dataset_config.nyu40ids)

        return self._generate_data_label(point_cloud, instance_bboxes, dataset_config, point_votes, point_votes_mask)
