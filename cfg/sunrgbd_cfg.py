''' Global configurations of SUN RGB-D dataset.

Author: Zhao Na
Data: September, 2020
'''

import os
import numpy as np
import pickle
from easydict import EasyDict


__C = EasyDict()
cfg = __C

__C.NUM_CLASSES = 10
__C.TYPE_WHITELIST = ['bathtub', 'bed', 'bookshelf', 'chair', 'desk', 'dresser', 'night_stand', 'sofa', 'table', 'toilet']
__C.CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
__C.MAX_NUM_POINT = 50000
__C.USE_V1 = True
__C.SKIP_EMPTY_SCENE = True

__C.TYPE_MEAN_SIZE = {'bathtub': np.array([0.765840,1.398258,0.472728]),
                      'bed': np.array([2.114256,1.620300,0.927272]),
                      'bookshelf': np.array([0.404671,1.071108,1.688889]),
                      'chair': np.array([0.591958,0.552978,0.827272]),
                      'desk': np.array([0.695190,1.346299,0.736364]),
                      'dresser': np.array([0.528526,1.002642,1.172878]),
                      'night_stand': np.array([0.500618,0.632163,0.683424]),
                      'sofa': np.array([0.923508,1.867419,0.845495]),
                      'table': np.array([0.791118,1.279516,0.718182]),
                      'toilet': np.array([0.699104,0.454178,0.756250])
                     }

__C.NUM_HEADING_BIN = 12

__C.MAX_NUM_OBJ = 64 # maximum number of objects allowed per scene
__C.MEAN_COLOR_RGB = np.array([0.5,0.5,0.5]) # sunrgbd color is in 0~1

__C.NUM_BASE_CLASSES = 5
__C.BASE_TYPES = ['bathtub', 'bed', 'bookshelf', 'chair', 'desk']
__C.BASE_CLASSES = [0, 1, 2, 3, 4]

__C.NUM_NOVEL_CLASSES = 5
__C.NOVEL_TYPES = ['dresser', 'night_stand', 'sofa', 'table', 'toilet']
__C.NOVEL_CLASSES = [5, 6, 7, 8, 9]


def get_class2scans(data_path, split='train'):
    '''Generate a mapping dictionary whose key is the class name and the values are the corresponding scan names
       containing objects of this class
    '''
    index_data_path = os.path.join(data_path, 'index_data')
    class2scans_file = os.path.join(index_data_path, '%s_class2scans.pkl' %split)
    if not os.path.exists(index_data_path): os.mkdir(index_data_path)

    if os.path.exists(class2scans_file):
        with open(class2scans_file, 'rb') as f:
            class2scans = pickle.load(f)
    else:
        class2scans = {c: [] for c in __C.TYPE_WHITELIST}
        scan_data_path = os.path.join(data_path, 'sunrgbd_%s_pc_bbox_50k_%s' %('v1' if __C.USE_V1 else 'v2', split))
        print(scan_data_path)
        all_scan_names = list(set([os.path.basename(x)[0:6] for x in os.listdir(scan_data_path)]))
        for scan_name in all_scan_names:
            bboxes = np.load(os.path.join(scan_data_path, scan_name)+'_bbox.npy')
            label_ids = bboxes[:,-1]
            unique_label_ids = np.unique(label_ids)
            for label_id in unique_label_ids:
                class_name = __C.TYPE_WHITELIST[int(label_id)]
                class2scans[class_name].append(scan_name)

        print('==== Split: %s | class to scans mapping is done ====' %split)
        for class_name in __C.TYPE_WHITELIST:
            print('\t class_name: {0} | num of scans: {1}'.format(class_name, len(class2scans[class_name])))

        #save class2scans to file...
        with open(class2scans_file, 'wb') as f:
            pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
    return class2scans


if __name__ == '__main__':
    get_class2scans('../sunrgbd/', split='val')
