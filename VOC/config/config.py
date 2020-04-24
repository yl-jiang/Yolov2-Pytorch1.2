#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/21 下午4:52
# @Author  : jyl
# @File    : config.py
import os
import torch
from pathlib import Path
import random
import numpy as np
from pprint import pprint


class Config:
    # set random seed
    # torch.manual_seed(7)  # cpu
    # torch.cuda.manual_seed(7)  # gpu
    # np.random.seed(7)  # numpy
    # random.seed(7)  # random and transforms
    # torch.backends.cudnn.deterministic = True  # cudnn
    base_path = Path('./').absolute()
    base_path = Path('/home/dk/ML/V2/VOC')

    VOC_BBOX_LABEL_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'bg']

    # training
    reg_scale = 1.
    noobj_scale = 1.
    obj_scale = 5.
    cls_scale = 1.
    batch_size = 32
    epoch_num = 400
    lr = 1e-4
    pth_lr = lr
    yolo_lr = lr
    optim_momentum = 0.9
    optim_weight_decay = 0.0005
    pos_iou_thresh = 0.5
    use_focal_loss = True
    use_smooth_labels = True
    display_every = 15
    eval_every = 50
    save_every = 100
    summary_dir = base_path / 'summary' / 'yolov2_loss'
    # 'SGD | 'Adam
    optimizer_type = 'SGD'
    # 对那些存在目标的cell下的且与gt匹配的预测框决定是否使用pred_gt_iou作为conf的值
    rescore = False
    model_save_dir = base_path / 'ckpt'

    # path
    model_best = base_path / 'ckpt' / 'model_best.pth'
    model_every = base_path / 'ckpt' / 'model_yolov3_loss_every.pth'
    github_model = base_path / 'ckpt' / 'only_params_trained_yolo_voc'

    # torch
    num_workers = 8

    # data
    anchors_path = base_path / 'data' / 'anchors.txt'
    anchor_num = 5
    kmean_converge = 1e-6
    B = 5
    S = 13
    img_size = 416
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    aug_thresh = 0.15
    voc_class_num = 20

    # testing
    nms_score_thresh = 0.3
    nms_iou_thresh = 0.45
    nms_max_box_num = 20
    nms_min_box_area = 64
    result_img_dir = base_path / 'result' / 'yolov2_loss'

    if torch.cuda.is_available():
        device = 'cuda'
        pin_memory = True
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        pin_memory = False

    def __init__(self, is_train):
        if is_train:
            self.data_dir = '/home/dk/Share/JYL/TX2/Dataset/VOC2007/VOCtrainval_06-Nov-2007/VOC2007'
            self.obj_path = self.base_path / 'data/obj_dict_train.pkl'
        else:
            self.data_dir = '/home/dk/Share/JYL/TX2/Dataset/VOC2007/VOCtest_06-Nov-2007/VOC2007'
            self.obj_path = self.base_path / 'data/obj_dict_test.pkl'

    def _state_dict(self):
        state_dict = {k: getattr(self, k) for k, _ in Config.__dict__.items() if not k.startswith('_')}
        return state_dict

    def _update(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError(f'keyword "{k}" is not in config')
            setattr(self, k, v)
        print(f'+++++++++++++++++++++++++ use config +++++++++++++++++++++++++')
        pprint(self._state_dict())
        print(f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

