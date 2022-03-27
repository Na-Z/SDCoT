# coding: utf-8

""" ScanNet Dataset for first stage training (train with massive base class data).
Acknowledge: https://github.com/facebookresearch/votenet/blob/master/scannet/scannet_detection_dataset.py

Author: Zhao Na
Date: Oct, 2020
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'cfg'))
import pc_util
from scannet_cfg import get_class2scans
from scannet import ScannetBaseDatasetConfig, ScannetDataset


class ScannetBaseDataset(ScannetDataset):

    def __init__(self, num_points=40000, use_color=False, use_height=False, augment=False):
        super(ScannetBaseDataset, self).__init__(num_points, use_color, use_height, augment)

        self.dataset_config = ScannetBaseDatasetConfig()

        class2scans = get_class2scans(self.data_path)
        all_scan_names = [scan_name for class_name in self.dataset_config.types for scan_name in class2scans[class_name]]
        self.scan_names = list(set(all_scan_names))
        print('Training classes: {0} | number of scenes: {1}'.format(self.dataset_config.types, len(self.scan_names)))

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        return self._process_one_scene(scan_name, self.train_data_path, self.dataset_config)


##################################### Visualizaion ################################

def viz_votes(pc, point_votes, point_votes_mask, name=''):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask == 1)
    pc_obj = pc[inds, 0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds, 0:3]
    pc_util.write_ply(pc_obj, 'pc_obj{}.ply'.format(name))
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1{}.ply'.format(name))


def viz_obb(label, mask, mean_size_arr, size_classes, size_residuals, name=''):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i, 0:3]
        heading_angle = 0  # hard code to 0
        box_size = mean_size_arr[size_classes[i], :] + size_residuals[i, :]
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs{}.ply'.format(name))
    pc_util.write_ply(label[mask == 1, :], 'gt_centroids{}.ply'.format(name))


if __name__=='__main__':
    dset = ScannetBaseDataset(num_points=20000, use_color=True, use_height=True, augment=True)
    DC = dset.dataset_config()
    for i_example in range(4):
        example = dset.__getitem__(1)
        pc_util.write_ply(example['point_clouds'], 'pc_{}.ply'.format(i_example))
        viz_votes(example['point_clouds'], example['vote_label'],
            example['vote_label_mask'],name=str(i_example))
        viz_obb(label=example['center_label'], mask=example['box_label_mask'], mean_size_arr=DC.mean_size_arr,
            size_classes=example['sem_cls_label'], size_residuals=example['size_residual_label'], name=str(i_example))