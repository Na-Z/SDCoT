# coding: utf-8

"""SUN RGBD Dataset Basic Class for training and testing data organization
Acknowledge: https://github.com/facebookresearch/votenet/blob/master/sunrgbd/sunrgbd_detection_dataset.py

Author: Zhao Na
Date: September, 2020
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
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
import pc_util
import box_util
import model_util
from sunrgbd_cfg import cfg, get_class2scans
import sunrgbd_utils


class SunrgbdBaseDatasetConfig(object):
    def __init__(self):
        self.dataset = 'sunrgbd'
        self.num_class = cfg.NUM_BASE_CLASSES #the number of_classes in the original classifier of predicting module
        self.num_class_final = cfg.NUM_BASE_CLASSES  # the number of classes in the final classifier of predicting module
        self.num_heading_bin = cfg.NUM_HEADING_BIN

        self.class_inds = cfg.BASE_CLASSES
        self.types = cfg.BASE_TYPES
        self.type2class = {t:c for (c, t) in enumerate(self.types)}
        self.class2type = {c:t for (c, t) in enumerate(self.types)}

        self.type_mean_size = cfg.TYPE_MEAN_SIZE
        self.mean_size_arr = np.zeros((self.num_class_final, 3))
        for i in range(self.num_class_final):
            self.mean_size_arr[i, :] = self.type_mean_size[self.class2type[i]]


class SunrgbdNovelDatasetConfig(object):
    def __init__(self, num_novel_classes):
        self.dataset = 'sunrgbd'
        self.num_class = num_novel_classes
        self.num_class_final = self.num_class
        self.num_heading_bin = cfg.NUM_HEADING_BIN

        self.class_inds = cfg.NOVEL_CLASSES[:num_novel_classes]
        self.types = cfg.NOVEL_TYPES[:num_novel_classes]
        self.type2class = {t:c for (c, t) in enumerate(self.types)}
        self.class2type = {c:t for (c, t) in enumerate(self.types)}

        self.type_mean_size = cfg.TYPE_MEAN_SIZE
        self.mean_size_arr = np.zeros((self.num_class_final, 3))
        for i in range(self.num_class_final):
            self.mean_size_arr[i, :] = self.type_mean_size[self.class2type[i]]


class SunrgbdAllDatasetConfig(object):
    def __init__(self, num_novel_classes, incremental=False):
        self.dataset = 'sunrgbd'
        if incremental:
            self.num_class = cfg.NUM_BASE_CLASSES
            self.labeled_class_inds = cfg.NOVEL_CLASSES[:num_novel_classes]
            self.labeled_types = cfg.NOVEL_TYPES[:num_novel_classes]
        else:
            self.num_class = cfg.NUM_BASE_CLASSES + num_novel_classes
        self.num_class_final = cfg.NUM_BASE_CLASSES + num_novel_classes
        self.num_heading_bin = cfg.NUM_HEADING_BIN

        self.class_inds = cfg.BASE_CLASSES + cfg.NOVEL_CLASSES[:num_novel_classes]
        self.types = cfg.BASE_TYPES + cfg.NOVEL_TYPES[:num_novel_classes]
        self.type2class = {t:c for (c, t) in enumerate(self.types)}
        self.class2type = {c:t for (c, t) in enumerate(self.types)}

        self.type_mean_size = cfg.TYPE_MEAN_SIZE
        self.mean_size_arr = np.zeros((self.num_class_final, 3))
        for i in range(self.num_class_final):
            self.mean_size_arr[i, :] = self.type_mean_size[self.class2type[i]]


class SunrgbdDataset(Dataset):
    def __init__(self, num_points, use_color, use_height, augment):
        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.pc_attri_dim = 3 + 3 * self.use_color + 1 * self.use_height
        self.augment = augment

        self.data_path = os.path.join(ROOT_DIR, 'sunrgbd')
        self.train_data_path = os.path.join(self.data_path, 'sunrgbd_%s_pc_bbox_50k_train' %'v1' if cfg.USE_V1 else 'v2')
        self.val_data_path = os.path.join(self.data_path, 'sunrgbd_%s_pc_bbox_50k_val' %'v1' if cfg.USE_V1 else 'v2')

    def _process_pointcloud(self, mesh_vertices):
        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3]
        else:
            point_cloud = mesh_vertices[:,0:6]
            point_cloud[:,3:] = (point_cloud[:,3:]-cfg.MEAN_COLOR_RGB)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)

        return point_cloud

    def _sample_pointcloud(self, point_cloud):
        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        return point_cloud, choices

    def _augment(self, point_cloud, target_bboxes):
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            point_cloud[:, 0] = -1 * point_cloud[:, 0]
            target_bboxes[:, 0] = -1 * target_bboxes[:, 0]
            target_bboxes[:, 6] = np.pi - target_bboxes[:, 6]

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 9) - np.pi / 18  # -10 ~ +10 degree
        rot_mat = sunrgbd_utils.rotz(rot_angle)
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
        target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3], np.transpose(rot_mat))
        target_bboxes[:, 6] -= rot_angle

        # Augment RGB color
        if self.use_color:
            rgb_color = point_cloud[:, 3:6] + cfg.MEAN_COLOR_RGB
            rgb_color *= (1 + 0.4 * np.random.random(3) - 0.2)  # brightness change for each channel
            rgb_color += (0.1 * np.random.random(3) - 0.05)  # color shift for each channel
            rgb_color += np.expand_dims((0.05 * np.random.random(point_cloud.shape[0]) - 0.025),
                                        -1)  # jittering on each pixel
            rgb_color = np.clip(rgb_color, 0, 1)
            # randomly drop out 30% of the points' colors
            rgb_color *= np.expand_dims(np.random.random(point_cloud.shape[0]) > 0.3, -1)
            point_cloud[:, 3:6] = rgb_color - cfg.MEAN_COLOR_RGB

        # Augment point cloud scale: 0.85x-1.15x
        scale_ratio = np.random.random() * 0.3 + 0.85
        scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
        point_cloud[:, 0:3] *= scale_ratio
        target_bboxes[:, 0:3] *= scale_ratio
        target_bboxes[:, 3:6] *= scale_ratio
        if self.use_height:
            point_cloud[:, -1] *= scale_ratio[0, 0]

        return point_cloud, target_bboxes

    def _generate_votes(self, point_cloud, target_bboxes, scan_name):
        '''generate vote for each point
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
                box3d_pts_3d = sunrgbd_utils.my_compute_box_3d(target_box[0:3], target_box[3:6], target_box[6])
                pc_in_box3d, inds = box_util.extract_pc_in_box3d(point_cloud, box3d_pts_3d)
                # Assign first dimension to indicate it is in an object box
                point_votes[inds, 0] = 1
                # Add the votes (all 0 if the point is not in any object's OBB)
                votes = np.expand_dims(target_box[0:3], 0) - pc_in_box3d[:, 0:3]
                sparse_inds = indices[inds]  # turn dense True,False inds to sparse number-wise inds
                for i in range(len(sparse_inds)):
                    j = sparse_inds[i]
                    point_votes[j, int(point_vote_idx[j] * 3 + 1):int((point_vote_idx[j] + 1) * 3 + 1)] = votes[i, :]
                    # Populate votes with the first vote
                    if point_vote_idx[j] == 0:
                        point_votes[j, 4:7] = votes[i, :]
                        point_votes[j, 7:10] = votes[i, :]
                point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds] + 1)
            except:
                print('ERROR ----', scan_name, target_box[7])

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
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = [dataset_config.class_inds.index(x)
                                                            for x in instance_bboxes[:, -1]]

        for i in range(instance_bboxes.shape[0]):
            bbox = instance_bboxes[i]
            angle_class, angle_residual = model_util.angle2class(bbox[6], dataset_config)
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here
            box3d_size = bbox[3:6]*2
            # size_class, size_residual = model_util.size2class(box3d_size, cfg.TYPE_WHITELIST[semantic_class], dataset_config)
            class_ind = target_bboxes_semcls[i]
            size_residual = box3d_size - dataset_config.mean_size_arr[int(class_ind)]
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_residuals[i] = size_residual

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
        point_cloud = np.load(os.path.join(scan_data_path, scan_name)+'_pc.npz')['pc'] # Nx6
        instance_bboxes = np.load(os.path.join(scan_data_path, scan_name)+'_bbox.npy') # K,8

        bbox_mask = np.in1d(instance_bboxes[:, -1], dataset_config.class_inds)
        instance_bboxes = instance_bboxes[bbox_mask, :]

        point_cloud = self._process_pointcloud(point_cloud)
        point_cloud, _ = self._sample_pointcloud(point_cloud)

        # data augmentation
        if self.augment:
            point_cloud, instance_bboxes = self._augment(point_cloud, instance_bboxes)

        # compute votes *AFTER* augmentation
        point_votes_mask, point_votes = self._generate_votes(point_cloud, instance_bboxes, scan_name)

        return self._generate_data_label(point_cloud, instance_bboxes, dataset_config, point_votes, point_votes_mask)