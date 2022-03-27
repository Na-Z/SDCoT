# coding: utf-8

""" SUN RGBD Dataset for testing (evaluate with validation set).

Author: Zhao Na
Date: Oct, 2020
"""
import os
from sunrgbd import SunrgbdBaseDatasetConfig, SunrgbdAllDatasetConfig, SunrgbdDataset


class SunrgbdValDataset(SunrgbdDataset):
    def __init__(self,  all_classes=True, num_novel_class=3, num_points=20000,
                        use_color=False, use_height=False, augment=False):
        super(SunrgbdValDataset, self).__init__(num_points, use_color, use_height, augment)

        self.scan_names = list(set([os.path.basename(x)[0:6] for x in os.listdir(self.val_data_path)]))

        if all_classes:
            # construct validation dataset for evaluating the performance on both base and novel classes (incremental learning)
            self.dataset_config = SunrgbdAllDatasetConfig(num_novel_class)
        else:
            # construct validation dataset for evaluating the performance on base classes
            self.dataset_config = SunrgbdBaseDatasetConfig()

        print('Testing classes: {0} | number of scenes: {1}'.format(self.dataset_config.types, len(self.scan_names)))

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        return self._process_one_scene(scan_name, self.val_data_path, self.dataset_config)


if __name__=='__main__':
    dset = SunrgbdValDataset(num_novel_class=3, num_points=20000, use_color=False, use_height=True, augment=False)
    for i_example in range(4):
        example = dset.__getitem__(1)
