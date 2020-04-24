#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/30 下午3:06
# @Author  : jyl
# @File    : config.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/21 下午4:52
# @Author  : jyl
# @File    : config.py
import os
import torch
import logging.config
import random
import numpy as np


class Config:
    # set random seed
    # torch.manual_seed(7)  # cpu
    # torch.cuda.manual_seed(7)  # gpu
    # np.random.seed(7)  # numpy
    # random.seed(7)  # random and transforms
    # torch.backends.cudnn.deterministic = True  # cudnn
    base_path = os.path.abspath(os.path.dirname('__file__'))

    VOC_BBOX_LABEL_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    # training
    coord_scale = 1.
    noobj_scale = 1.
    obj_scale = 5.
    class_scale = 1.
    batch_size = 32
    epochs = 300
    lr = 1e-4
    pth_lr = lr
    yolo_lr = lr
    optimizer_momentum = 0.9
    optimizer_weight_decay = 0.0005
    match_iou_threshold = 0.6
    use_focal_loss = True
    use_smooth_labels = True
    display_step = 50
    eval_step = 200
    save_step = 100
    save_every = 5
    # 'SGD | 'Adam
    optimizer_type = 'Adam'
    model_save_dir = os.path.join(base_path, 'ckpt')

    # path
    base_path = os.path.split(base_path)[0]
    testdata_txtpath = os.path.join(base_path, 'data', 'testing.txt')
    traindata_txtpath = os.path.join(base_path, 'data', 'training.txt')
    testdata_resize_path = os.path.join(base_path, 'data', 'testing_resize.txt')
    traindata_resize_path = os.path.join(base_path, 'data', 'training_resize.txt')
    vocdata_dir = '/home/dk/jyl/Data/VOC'
    saved_model_path = os.path.join(base_path, 'model', 'ckpt', f'model_best_{optimizer_type}.pkl')
    log_config_path = os.path.join(base_path, 'log', f'logging.conf')
    log_file_path = os.path.join(base_path, 'log', f'log_{optimizer_type}.log')

    # logger
    logging.config.fileConfig(log_config_path)
    logger = logging.getLogger('Yolov2Logger')
    summary_writer_path = os.path.join(base_path, 'log', f'summary_{optimizer_type}')

    # torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type(torch.FloatTensor)
    num_workers = 0

    # anchor
    anchor_num = 5
    kmean_converge = 1e-6
    anchors_path = os.path.join(base_path, 'data', 'anchors.txt')

    # grid
    B = 5
    S = 13

    # img
    img_h = 416
    img_w = 416
    img_size = 416
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    aug_threshold = 0.25
    
    # data
    voc_class_num = 20
    coco_class_num = 80

    # evaluation
    result_img_dir = os.path.join(base_path, 'data', 'result_imgs')

    # testing
    score_threshold = 0.3
    iou_threshold = 0.3
    max_boxes_num = 200


opt = Config()


