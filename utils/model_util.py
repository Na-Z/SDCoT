""" Util function for bbox re-parameterization
Modified from https://github.com/facebookresearch/votenet/blob/master/scannet/model_util_scannet.py

Author: Zhao Na
Date: September, 2020
"""

import numpy as np


def size2class(size, type_name, data_config):
    ''' Convert 3D box size (l,w,h) to size class and size residual '''
    size_class = data_config.type2class[type_name]
    size_residual = size - data_config.type_mean_size[type_name]
    return size_class, size_residual


def class2size(pred_cls, residual, data_config):
    ''' Inverse function to size2class '''
    mean_size = data_config.type_mean_size[data_config.class2type[pred_cls]]
    return mean_size + residual


def angle2class(angle, data_config):
    ''' Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        return is class of int32 of 0,1,...,N-1 and a number such that
            class*(2pi/N) + number = angle
    '''
    if data_config.dataset == 'scannet':
        assert (False)
    elif data_config.dataset == 'sunrgbd':
        num_class = data_config.num_heading_bin
        angle = angle % ( 2 * np.pi)
        assert (angle >= 0 and angle <= 2 * np.pi)
        angle_per_class = 2* np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle
    else:
        raise ValueError('Wrong dataset name %s' %data_config.dataset)


def class2angle(pred_cls, residual, data_config, to_label_format=True):
    ''' Inverse function to angle2class '''
    if data_config.dataset == 'scannet':
        #As ScanNet only has axis-alined boxes so angles are always 0. '''
        return 0
    elif data_config.dataset == 'sunrgbd':
        num_class = data_config.num_heading_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle
    else:
        raise ValueError('Wrong dataset name %s' %data_config.dataset)


def param2obb(center, heading_class, heading_residual, size_class, size_residual, data_config):
    heading_angle = class2angle(heading_class, heading_residual, data_config)
    box_size = class2size(int(size_class), size_residual, data_config)
    obb = np.zeros((7,))
    obb[0:3] = center
    obb[3:6] = box_size
    obb[6] = heading_angle * -1
    return obb