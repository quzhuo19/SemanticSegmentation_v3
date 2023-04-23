# -*- coding:utf-8 -*-
# @Time   : 2022/4/30 15:22
# @Author : quzhuo
import os
import random
import numpy as np
import transforms as T

# ---------------------------------------------------------#
#   将不同像素值映射成对应数字
#   3D-->2D
# ---------------------------------------------------------#
# palette_org = {
#     0: (0,   0,   255),  # Buildings (blue)
#     1: (0,   255, 0),    # Trees (green)
#     2: (0,   255, 255),  # Low vegetation (cyan)
#     3: (255, 0,   0),    # Clutter (red)
#     4: (255, 255, 255),  # Impervious surfaces (white)
#     5: (255, 255, 0),    # Cars (yellow)
#     6: (0,   0,   0)     # Undefined (black)
# }


def convert_from_color(arr_3d, palette=None):
    palette = palette
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    invert_palette = {v: k for k, v in palette.items()}

    for c, i in invert_palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


# ---------------------------------------------------------#
#   构造数据集索引
# ---------------------------------------------------------#
def build_dataset(dataset_path, dataset_name, train_percent=0.8, test_percent=0.2):
    data_divide_pth = os.path.join(dataset_path, 'Data')
    dataset_path = os.path.join(dataset_path, dataset_name, 'train')
    # dataset_path = os.path.join(dataset_path, dataset_name, 'images')

    # random.seed(0)

    temp_seg = os.listdir(dataset_path)
    total_seg = []
    for seg in temp_seg:
        # if seg.endswith('.jpg'):
        total_seg.append(seg[:-4])

    num = len(total_seg)
    list_item = range(num)

    train_test_num = int(num * (train_percent + test_percent))
    test_num = int(num * test_percent)

    train_test_random = random.sample(list_item, train_test_num)
    test_random = random.sample(train_test_random, test_num)

    f_data = open(os.path.join(data_divide_pth, 'data.txt'), 'w')
    f_train = open(os.path.join(data_divide_pth, 'train.txt'), 'w')
    f_val = open(os.path.join(data_divide_pth, 'val.txt'), 'w')
    f_test = open(os.path.join(data_divide_pth, 'test.txt'), 'w')

    for i in list_item:
        name_item = total_seg[i] + '\n'
        f_data.write(name_item)
        if i in train_test_random:
            if i in test_random:
                f_val.write(name_item)
            else:
                f_train.write(name_item)
        else:
            f_test.write(name_item)

    f_data.close()
    f_train.close()
    f_test.close()
    f_val.close()


# ---------------------------------------------------------#
#   数据增强（随机水平、竖直翻转）
# ---------------------------------------------------------#
class DatasetTrans:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5):
        self.transforms = T.Compose([T.RandomHorizontalFlip(hflip_prob),
                                     T.RandomVerticalFlip(vflip_prob)
                                     ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform():
    return DatasetTrans()
