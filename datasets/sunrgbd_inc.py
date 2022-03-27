# coding: utf-8

""" SUN RGB-D Dataset for incremental learning with novel class data.

Author: Zhao Na
Date: Oct, 2020
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'cfg'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
from sunrgbd_cfg import cfg, get_class2scans
from sunrgbd import SunrgbdAllDatasetConfig, SunrgbdDataset
from model import init_detection_model, generate_pseudo_bboxes
import pc_util


class SunrgbdIncDataset(SunrgbdDataset):

    def __init__(self, args, num_novel_class=3, num_points=20000, use_color=False, use_height=False, augment=False):
        super(SunrgbdIncDataset, self).__init__(num_points, use_color, use_height, augment)

        self.dataset_config = SunrgbdAllDatasetConfig(num_novel_class, incremental=True)

        class2scans = get_class2scans(self.data_path)
        all_scan_names = [scan_name for class_name in self.dataset_config.labeled_types
                          for scan_name in class2scans[class_name]]
        self.scan_names = list(set(all_scan_names))
        print('Training classes: {0} | number of scenes: {1}'.format(self.dataset_config.labeled_types,
                                                                     len(self.scan_names)))

        print('+++ Init base detector for generating pseudo bounding boxes +++')
        self.base_detector, self.device = init_detection_model(args, self.dataset_config)
        self.base_detector.share_memory()
        self.base_detector.eval()

        self.pseudo_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
                                   'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
                                   'obj_conf_thresh': args.pseudo_obj_conf_thresh,
                                   'cls_conf_thresh': args.pseudo_cls_conf_thresh,
                                   'dataset_config': self.dataset_config}

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]

        raw_point_cloud = np.load(os.path.join(self.train_data_path, scan_name)+'_pc.npz')['pc'] # Nx6
        instance_bboxes = np.load(os.path.join(self.train_data_path, scan_name)+'_bbox.npy') # K,8

        bbox_mask = np.in1d(instance_bboxes[:, -1], self.dataset_config.labeled_class_inds)
        instance_bboxes = instance_bboxes[bbox_mask, :]

        raw_point_cloud = self._process_pointcloud(raw_point_cloud)

        point_cloud, _ = self._sample_pointcloud(raw_point_cloud)
        ema_point_cloud, _  = self._sample_pointcloud(raw_point_cloud)

        pseudo_bboxes = generate_pseudo_bboxes(self.base_detector, self.device, self.pseudo_config_dict, point_cloud)

        if pseudo_bboxes is not None:
            pseudo_bboxes[:, 3:6] = pseudo_bboxes[:, 3:6]/2
            instance_bboxes = np.concatenate((instance_bboxes, pseudo_bboxes))
            if instance_bboxes.shape[0] > cfg.MAX_NUM_OBJ:
                print('The number of bbox [%d] exceed MAX_NUM_OBJ [%d], removing a few...' % (instance_bboxes.shape[0],
                                                                                              cfg.MAX_NUM_OBJ))
                instance_bboxes  = instance_bboxes[0:cfg.MAX_NUM_OBJ, :]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        flip_x_axis = 0
        flip_y_axis = 0
        rot_mat = np.identity(3)
        scale_ratio = np.ones((1, 3))
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                flip_x_axis = 1
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                instance_bboxes[:, 0] = -1 * instance_bboxes[:, 0]
                instance_bboxes[:, 6] = np.pi - instance_bboxes[:, 6]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 9) - np.pi / 18  # -10 ~ +10 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            instance_bboxes[:, 0:3] = np.dot(instance_bboxes[:, 0:3], np.transpose(rot_mat))
            instance_bboxes[:, 6] -= rot_angle

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio
            instance_bboxes[:, 0:3] *= scale_ratio
            instance_bboxes[:, 3:6] *= scale_ratio
            if self.use_height:
                point_cloud[:, -1] *= scale_ratio[0, 0]

        point_votes_mask, point_votes = self._generate_votes(point_cloud, instance_bboxes, scan_name)

        ret_dict =  self._generate_data_label(point_cloud, instance_bboxes, self.dataset_config,
                                              point_votes, point_votes_mask)

        ret_dict['ema_point_clouds'] = ema_point_cloud.astype(np.float32)
        ret_dict['flip_x_axis'] = np.array(flip_x_axis).astype(np.int64)
        ret_dict['flip_y_axis'] =  np.array(flip_y_axis).astype(np.int64)
        ret_dict['rot_mat'] =  rot_mat.astype(np.float32)
        ret_dict['scale'] = np.array(scale_ratio).astype(np.float32)

        return ret_dict


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint_path', default=None,
                        help='Detection model checkpoint path [default: None]')
    parser.add_argument('--num_target', type=int, default=128, help='Proposal number [default: 128]')
    parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
    parser.add_argument('--cluster_sampling', default='vote_fps',
                        help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')

    parser.add_argument('--pseudo_obj_conf_thresh', type=float, default=0.95,
                        help='Confidence score threshold w.r.t. objectness prediction for hard selection of psuedo bboxes')
    parser.add_argument('--pseudo_cls_conf_thresh', type=float, default=0.9,
                        help='Confidence score threshold w.r.t. class prediction for hard selection of psuedo bboxes')
    args = parser.parse_args()

    args.num_input_channel = int(args.use_color) * 3 + int(not args.no_height) * 1

    dset = SunrgbdIncDataset(args, num_novel_class=3, num_points=args.num_point,  use_color=args.use_color,
                             use_height=(not args.no_height), augment=False)
    for idx in range(10):
        example = dset.__getitem__(idx)