""" Util for model initialization

Author: Zhao Na
Date: September, 2020
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from ap_helper import parse_prediction_to_pseudo_bboxes


def init_detection_model(args, dataset_config):
    model = create_detection_model(args, dataset_config)
    if args.model_checkpoint_path is not None:
        model = load_detection_model(model, args.model_checkpoint_path)
    else:
        raise ValueError('Detection model checkpoint path must be given!')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device


def generate_pseudo_bboxes(base_detector, device, pseudo_config_dict, point_cloud):
    ''' generate pseudo bounding boxes w.r.t. base classes
    Args:
        base_detector: nn.Module, model
        pseudo_config_dict: dict
        point_cloud: numpy array, shape (num_point, pc_attri_dim)
    Returns:
        pseudo_bboxes: numpy array, shape (num_valid_detections, 8)
    '''
    point_cloud_tensor = torch.from_numpy(point_cloud.astype(np.float32)).to(device).unsqueeze(0)
    with torch.no_grad():
        end_points = base_detector(point_cloud_tensor)
    pseudo_bboxes = parse_prediction_to_pseudo_bboxes(end_points, pseudo_config_dict, point_cloud)
    return pseudo_bboxes


def create_detection_model(args, dataset_config):
    from votenet import VoteNet
    model = VoteNet(dataset_config.num_class,
                    dataset_config.num_heading_bin,
                    dataset_config.mean_size_arr,
                    input_feature_dim=args.num_input_channel,
                    num_proposal=args.num_target,
                    vote_factor=args.vote_factor,
                    sampling=args.cluster_sampling)
    return model


def check_state_dict_consistency(loaded_state_dict, model_state_dict):
    """check consistency between loaded parameters and created model parameters
    """
    valid_state_dict = {}
    for k in loaded_state_dict:
        if k in model_state_dict:
            if loaded_state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}'.format(
                    k, model_state_dict[k].shape, loaded_state_dict[k].shape))
                valid_state_dict[k] = model_state_dict[k]
            else:
                valid_state_dict[k] = loaded_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))

    for k in model_state_dict:
        if not (k in loaded_state_dict):
            print('No param {}.'.format(k))
            valid_state_dict[k] = model_state_dict[k]

    return valid_state_dict


def load_detection_model(model, model_path, model_name=None, optimizer=None, lr=None, lr_step=None, lr_rate=None):
    start_epoch = 0
    if model_name is None:
        checkpoint_filename = os.path.join(ROOT_DIR, model_path, 'checkpoint.tar')
        checkpoint = torch.load(checkpoint_filename) #Load all tensors onto the CPU
    else:
        checkpoint_filename = os.path.join(ROOT_DIR, model_path, model_name+'_checkpoint.tar')
        checkpoint = torch.load(checkpoint_filename)
    print('loaded {}, epoch {}'.format(checkpoint_filename, checkpoint['epoch']))
    state_dict_ = checkpoint['model_state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    state_dict = check_state_dict_consistency(state_dict, model_state_dict)
    model.load_state_dict(state_dict, strict=True)

    # resume optimizer parameters
    if optimizer is not None:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= lr_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
        return model, optimizer, start_epoch
    else:
        return model


def save_model(log_dir, epoch, model, model_name=None, optimizer=None):
    # Save checkpoint
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict() # with nn.DataParallel() the net is added as a submodule of DataParallel
    else:
        model_state_dict = model.state_dict()

    save_dict = {'model_state_dict': model_state_dict}

    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()

    if model_name is None:
        # if model_name is not specified, the saved model is the detection model
        save_dict['epoch'] = epoch + 1  # after training one epoch, the start_epoch should be epoch+1
        torch.save(save_dict, os.path.join(log_dir, 'checkpoint.tar'))
    else:
        # otherwise the saved model is the meta-weight generator
        save_dict['epoch'] = epoch + 1
        torch.save(save_dict, os.path.join(log_dir, model_name + '_checkpoint.tar'))