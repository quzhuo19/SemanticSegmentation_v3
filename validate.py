# -*- coding:utf-8 -*-
# @Time   : 2022/4/30 15:17
# @Author : quzhuo
import argparse
import math
import os.path
import random
import time

import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn
from dataset import MyDataset as MyDataset
from Utils.utils import fit_one_epoch, check_mkdir, fit_one_test
from Utils.utils_dataset import build_dataset, get_transform
from Models.Lu.SMKNet import SMKNet as create_model
from Models.bisenet.bisenet import BiSeNetV2 as bise

# ---------------------------------------------------------#
#   标签色板索引
# ---------------------------------------------------------#
palette = {'Potsdam': {0: (0,   0,   255),   # Buildings (blue)
                       1: (0,   255, 0),     # Trees (green)
                       2: (0,   255, 255),   # Low vegetation (cyan)
                       3: (255, 0,   0),     # Clutter (red)
                       4: (255, 255, 255),   # Surface (white)
                       5: (255, 255, 0),     # Cars (yellow)
                       6: (0,   0,   0)      # Undefined (black)
                       },
           'DeepGoogle_LandCover': {0: (0,   255, 255),  # Buildings (light blue)
                                    1: (255, 255, 0),    # Agricultural (yellow)
                                    2: (255, 0,   255),  # Meadow (purple)
                                    3: (0,   255, 0),    # Forest (green)
                                    4: (0,   0,   255),  # Water (blue)
                                    5: (255, 255, 255),  # Wasteland (white)
                                    6: (0,   0,   0)     # Undefined (black)
                                    },
           'GID': {0: (255, 0,   0),    # Built-up
                   1: (0,   255, 0),    # Farmland
                   2: (0,   255, 255),  # Forest
                   3: (255, 255, 0),    # Meadow
                   4: (0,   0,   255)   # Water
                   },
           'GID_15': {0:  (200, 0,   0),     # Industrial Land
                      1:  (250, 0,   150),   # Urban Residential
                      2:  (200, 150, 150),   # Rural Residential
                      3:  (250, 150, 150),   # Traffic Land
                      4:  (0,   200, 0),     # Paddy Field
                      5:  (150, 250, 0),     # Irrigated Land
                      6:  (150, 200, 150),   # Dry Cropland
                      7:  (200, 0,   200),   # Garden Plot
                      8:  (150, 0,   250),   # Arbor Woodland
                      9:  (150, 150, 250),   # Shrub Land
                      10: (250, 200, 0),     # Natural Grassland
                      11: (200, 200, 0),     # Artificial Grassland
                      12: (0,   0,   200),   # River
                      13: (0,   150, 200),   # Lake
                      14: (0,   200, 250)    # Pond
                      },
           'UDD6': {0:  (0, 0, 0),     # _background_
                    1:  (102, 102, 156),   # Facade
                    2:  (128, 64, 128),   # Road
                    3:  (107, 142, 35),   # Vegetation
                    4:  (0, 0, 142),     # Vehicle
                    5:  (70, 70, 70)    # Roof
                    },
           'drone': {0:  (0, 0,   0),     # unlabeled
                     1:  (128, 64, 128),   # paved-area
                     2:  (130, 76, 0),   # dirt
                     3:  (0, 102, 0),   # grass
                     4:  (112, 103, 87),     # gravel
                     5:  (28, 42, 168),     # water
                     6:  (48, 41, 30),   # rocks
                     7:  (0, 50, 89),   # pool
                     8:  (107, 142, 35),   # vegetation
                     9:  (70, 70, 70),   # roof
                     10: (102, 102, 156),     # wall
                     11: (254, 228, 12),     # window
                     12: (254, 148, 12),   # door
                     13: (190, 153, 153),   # fence
                     14: (153, 153, 153),   # fence-pole
                     15: (255, 22, 96),   # person
                     16: (102, 51, 0),   # dog
                     17: (9, 143, 150),   # car
                     18: (119, 11, 32),   # bicycle
                     19: (51, 51, 0),   # tree
                     20: (190, 250, 190),   # bald-tree
                     21: (112, 150, 146),   # ar-marker
                     22: (2, 135, 115)   # obstacle
                     #23: (255, 0, 0)   # conflicting
                     },
           'UAVID': {0:  (0, 0, 0),     # background
                     1:  (128, 0, 0),   # building
                     2:  (128, 64, 128),   # road
                     3:  (0, 128, 0),   # tree
                     4:  (128, 128, 0),     # low vegetation
                     5:  (64, 0, 128),     # moving car
                     6:  (192, 0, 192),   # static car
                     7:  (64, 64, 0),  # human
                     }

           }


# ---------------------------------------------------------#
#   分类类别
# ---------------------------------------------------------#
features = {'Potsdam': ['Buildings', 'Trees', 'Low vegetation', 'Clutter', 'Surface', 'Cars'],
            'DeepGoogle_LandCover': ['Buildings', 'Agricultural', 'Meadow', 'Forest', 'Water', 'Wasteland'],
            'GID': ['Built-up', 'Farmland', 'Forest', 'Meadow', 'Water'],
            'GID_15': ['Industrial Land', 'Urban Residential', 'Rural Residential', 'Traffic Land', 'Paddy Field', 'Irrigated Land', 'Dry Cropland', 'Garden Plot', 'Arbor Woodland', 'Shrub Land', 'Natural Grassland', 'Artificial Grassland', 'River', 'Lake', 'Pond'],
            'UDD6': ["_background_", "Facade", "Road", "Vegetation", "Vehicle", "Roof"],
            'drone': ['unlabeled', 'paved-area', 'dirt', 'grass', 'gravel', 'water', 'rocks', 'pool', 'vegetation', 'roof', 'wall', 'window', 'door', 'fence', 'fence-pole', 'person', 'dog', 'car', 'bicycle', 'tree', 'bald-tree', 'ar-marker', 'obstacle'],
            'UAVID': ['background', 'building', 'road', 'tree', 'low vegetation', 'moving car', 'static car', 'human']
            }


# ---------------------------------------------------------#
#   随机可复现
# ---------------------------------------------------------#
def random_seed(seed):
    np.random.random(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    random_seed(args.random_seed)

    print('-----Load Dataset-----')
    start_time = time.time()

    # write logs
    # check_mkdir(args.save_path)
    # with open(os.path.join(args.save_path, 'logs.txt'), 'w') as f:
    #     f.write('--------------------Configs--------------------\n')
    #     f.write('[dataset_name]: {}\n'.format(args.dataset_name))
    #     f.write('[save_path]: {}\n'.format(args.save_path))
    #     f.write('[batch_size]: {}\n'.format(args.batch_size))
    #     f.write('[epochs]: {}\n'.format(args.epochs))
    #     f.write('[lr]: {}\n'.format(args.lr))
    #     f.write('[random_seed]: {}\n'.format(args.random_seed))
    #     f.write('[pretrained]: {}\n'.format(args.pretrained))
    #     f.write('[model_path]: {}\n'.format(args.model_path))
    #     f.write('[dataset_trans]: {}\n'.format(args.dataset_trans))
    #     f.write('[update_mode]: {}\n'.format(args.update_mode))
    #     f.write('[resume]: {}\n'.format(args.resume))
    #     f.write('[checkpoint]: {}\n'.format(args.checkpoint))
    #     f.write('------------------Start Train------------------\n')
    #     f.write('[Start Time]: {}-{}-{} {}:{}:{}\n'.format(time.localtime(start_time).tm_year,
    #                                                        time.localtime(start_time).tm_mon,
    #                                                        time.localtime(start_time).tm_mday,
    #                                                        time.localtime(start_time).tm_hour,
    #                                                        time.localtime(start_time).tm_min,
    #                                                        time.localtime(start_time).tm_sec))
    #     f.close()

    # ---------------------------------------------------------#
    #   数据索引
    # ---------------------------------------------------------#
    with open(os.path.join(args.dataset_path, 'Data/test.txt'), "r") as f:
        test_lines = f.readlines()

    # ---------------------------------------------------------#
    #   实例化数据集
    # ---------------------------------------------------------#
    test_dataset = MyDataset(dataset_path=args.dataset_path,
                             dataset_name=args.dataset_name,
                             num_classes=args.num_classes,
                             dataset_trans=args.dataset_trans,
                             transforms=get_transform(),
                             txt_name='test.txt',
                             test_model=args.test_model,
                             palette=args.palette)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=args.num_workers,
                                              drop_last=True)

    # ---------------------------------------------------------#
    #   实验结果存储路径
    #   根据需要调整
    # ---------------------------------------------------------#
    best_record = {'epoch': 0, 'val_loss': 0, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0, 'f1': 0}

    # ---------------------------------------------------------#
    #   模型初始化
    #   载入预训练模型
    # ---------------------------------------------------------#
    bil = bise(num_classes=args.num_classes).to(device)
    model = bise(num_classes=args.num_classes).to(device)
    # model = create_model(num_classes=args.num_classes, aux_loss=None).to(device)

    if os.path.exists(args.model_path):
        weights_dict = torch.load(args.model_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
        print(model.load_state_dict(load_weights_dict, strict=False))
    else:
        raise FileNotFoundError("not found weights file: {}".format(args.model_path))

    # ---------------------------------------------------------#
    #   optimizer: 参数优化
    #   scheduler: 学习率调整
    #   criterion: loss函数
    #   可根据需求调整
    # ---------------------------------------------------------#
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=False)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0, last_epoch=-1)         # 余弦退火
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

    # ---------------------------------------------------------#
    # ......
    # ---------------------------------------------------------#
    fit_one_test(model=model,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 criterion=criterion,
                 test_loader=test_loader,
                 test_lines=test_lines,
                 args=args,
                 best_record=best_record)

    end_time = time.time()
    with open(os.path.join(args.save_path, 'logs.txt'), 'a') as f:
        f.write('[End Time]: {}-{}-{} {}:{}:{}\n'.format(time.localtime(end_time).tm_year,
                                                         time.localtime(end_time).tm_mon,
                                                         time.localtime(end_time).tm_mday,
                                                         time.localtime(end_time).tm_hour,
                                                         time.localtime(end_time).tm_min,
                                                         time.localtime(end_time).tm_sec))
        f.write('[Total Time]: %.3f\n' % (end_time - start_time))
        f.write('----------------------End----------------------\n')
        f.close()

    print('-----[time {}]-----'.format(end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path',     default='./Dataset', type=str, help='数据集根目录')
    parser.add_argument('--dataset_name',     default='UAVID_1024',   type=str, help='数据集名称，对应文件夹下放置数据')
    parser.add_argument('--save_path',        default='./Catch/bise/uavid_test',   type=str, help='模型参数存放路径')
    parser.add_argument('--palette',          default=palette['UAVID'], help='数据集对应的色板')
    parser.add_argument('--features',         default=features['UAVID'], help='分类类别')

    parser.add_argument('--epochs',        default=50,  type=int, help='')
    parser.add_argument('--batch_size',    default=12,    type=int, help='')
    parser.add_argument('--num_classes',   default=8,    type=int, help='')
    parser.add_argument('--lr',            default=0.01, type=float, help='')
    parser.add_argument('--lrf',           default=0.01, type=float, help='自定义学习率更新公式所需参数')
    parser.add_argument('--num_workers',   default=4,    type=int, help='')
    parser.add_argument('--print_freq',    default=25,   type=int, help='模型训练进度打印频率')
    parser.add_argument('--random_seed',   default=0,    type=int, help='全局随机种子')
    parser.add_argument('--save_rate',     default=1, type=float, help='模型预测结果保存几率')
    parser.add_argument('--update_mode',   default=0, type=int, help='[0, 1, 2] 1: miou; 2: mf1; 3: miou and mf1')

    parser.add_argument('--model_path',    default='./Catch/bise/uavid/epoch_47_loss_0.27972_acc_0.89715_acc-cls_0.83864_mean-iu_0.71418_fwavacc_0.81801_f1_0.81985_lr_0.0002555134.pth',    type=str, help='预训练参数存放地址')
    parser.add_argument('--dataset_trans', default=False,  type=bool, help='')
    parser.add_argument('--device',        default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--test_model', default=False, type=bool, help='')

    opt = parser.parse_args()

    main(opt)
