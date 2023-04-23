# -*- coding:utf-8 -*-
# @Time   : 2022/4/30 15:17
# @Author : quzhuo
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from Utils.utils_dataset import convert_from_color


class MyDataset(data.Dataset):
    def __init__(self, dataset_path, dataset_name, num_classes, dataset_trans=False, transforms=None, txt_name: str = 'train.txt', test_model: bool = False, palette=None):
        super(MyDataset, self).__init__()
        # --------------------------------------------------#
        #   dataset_path  '../../Dataset'
        #   dataset_name  数据集名称
        # --------------------------------------------------#

        root = os.path.join(dataset_path, dataset_name)
        assert os.path.exists(root), "path '{}' does not exists.".format(root)
        image_dir = os.path.join(root, 'train')
        mask_dir = os.path.join(root, 'labels')

        txt_path = os.path.join(dataset_path, 'Data', txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exists.".format(txt_path)

        with open(os.path.join(txt_path), 'r') as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        if test_model:
            file_names = file_names[:20]

        # i_list = os.listdir(image_dir)
        # m_list = os.listdir(mask_dir)
        #
        # image_end = i_list[0].split('.')[-1]
        # mask_end = m_list[0].split('.')[-1]
        #
        # self.images = [os.path.join(image_dir, x + '.' + image_end) for x in file_names]
        # self.masks = [os.path.join(mask_dir, x + '.' + mask_end) for x in file_names]

        self.images = [os.path.join(image_dir, x + '.png') for x in file_names]
        self.masks = [os.path.join(mask_dir, x + '.png') for x in file_names]
        self.num_classes = num_classes
        assert (len(self.images)) == len(self.masks)
        self.dataset_trans = dataset_trans
        self.transforms = transforms
        self.palette = palette

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index]).convert('RGB')

        if self.dataset_trans:
            img, target = self.transforms(img, target)   # 图像增强

        img = np.copy(img)
        target = np.copy(target)

        img = np.asarray(img, np.float32).transpose((2, 0, 1)) / 255.0   # 图像归一化
        target = np.asarray(convert_from_color(target, palette=self.palette), dtype='int64')   # 标签转换

        img, target = torch.from_numpy(img), torch.from_numpy(target)    # numpy-->tensor

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=250)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
