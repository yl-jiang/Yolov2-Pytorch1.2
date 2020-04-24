#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/16 下午5:04
# @Author  : jyl
# @File    : plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
from config import opt
import random
import os
import cv2
from PIL import Image

COCO_NAMES = opt.COCO_NAMES + ['BG']


def make_color_table(color_num):
    random.seed(7)  # random and transforms
    color_table = {}
    # '+2' means add background class and ground_truth
    for i in range(color_num + 2):
        color_table[i] = [random.randint(a=0, b=255) for _ in range(3)]
    return color_table


def plot_one(plot_dict):
    pred_box = plot_dict['pred_box']  # [xmin, ymin, xmax, ymax]
    pred_score = plot_dict['pred_score']
    pred_label = plot_dict['pred_label']
    img = plot_dict['img']
    ratio = plot_dict['ratio']
    dh = plot_dict['dh']
    dw = plot_dict['dw']
    gt_box = plot_dict['gt_box']
    gt_label = plot_dict['gt_label']
    img_name = plot_dict['img_name']
    save_path = plot_dict['save_path']

    pred_box[:, [1, 3]] = pred_box[:, [1, 3]] - dh
    pred_box[:, [0, 2]] = pred_box[:, [0, 2]] - dw
    pred_box = pred_box / ratio

    assert isinstance(img, np.ndarray) and img.dtype == np.uint8
    color_table = make_color_table(opt.coco_class_num)
    img = draw_pred(img, pred_box, pred_score, pred_label, color_table)
    # img = draw_gt(img, gt_box, gt_label, color_table)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(save_path + f'/{img_name}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def draw_pred(img, boxes, confs, labels, color_table):
    """
    :param img:
    :param boxes: [xmin, ymin, xmax, ymax]
    :param confs:
    :param labels:
    :param color_table:
    :return:
    """
    assert (boxes.shape[-1] == 4) and (len(confs) == len(boxes))
    for j in range(len(boxes)):
        if len(boxes[j]) == 0:
            cv2.putText(img=img, text=COCO_NAMES[-1], org=(0, 0), fontFace=cv2.FONT_ITALIC, fontScale=1,
                        color=(255, 255, 0), thickness=1)
            continue

        label = COCO_NAMES[labels[j]]
        score = confs[j]
        caption = [label, '%.2f' % score]

        img = cv2.rectangle(img=img,
                            pt1=(boxes[j][0], boxes[j][1]),
                            pt2=(boxes[j][2], boxes[j][3]),
                            color=color_table[labels[j]],
                            thickness=2)
        img = cv2.putText(img=img,
                          text=':'.join(caption),
                          org=(boxes[j][0], boxes[j][1]),
                          fontFace=cv2.FONT_ITALIC,
                          fontScale=0.6,
                          color=color_table[labels[j]])

    return img


def draw_gt(img, gt_boxes, gt_labels, color_table):
    """
    :param img:
    :param gt_boxes: [xmin, ymin, xmax, ymax]
    :param gt_labels:
    :param color_table:
    :return:
    """
    assert (gt_boxes.shape[-1] == 4 and len(gt_labels) == len(gt_boxes))

    for j in range(len(gt_boxes)):
        label = gt_labels[j]
        img = cv2.rectangle(img=img,
                            pt1=(gt_boxes[j][0], gt_boxes[j][1]),
                            pt2=(gt_boxes[j][2], gt_boxes[j][3]),
                            color=color_table[opt.coco_class_num + 1],
                            thickness=1)
        img = cv2.putText(img=img,
                          text=f'GT-{label}',
                          org=(gt_boxes[j][0], gt_boxes[j][1]),
                          fontFace=cv2.FONT_ITALIC,
                          fontScale=1,
                          color=color_table[opt.coco_class_num + 1])
    return img


def xywh2xyxy(bbox_xywh):
    """
    :param bbox_xywh:
        element in the last dimension's format is: [[center_x, center_y, w, h], ...]
    :return:
        [[xmin, ymin, xmax, ymax], ...]
    """
    ymax = bbox_xywh[..., [1]] + bbox_xywh[..., [3]] / 2
    xmax = bbox_xywh[..., [0]] + bbox_xywh[..., [2]] / 2
    ymin = bbox_xywh[..., [1]] - bbox_xywh[..., [3]] / 2
    xmin = bbox_xywh[..., [0]] - bbox_xywh[..., [2]] / 2

    yxyx = np.concatenate([xmin, ymin, xmax, ymax], axis=-1)
    return yxyx


if __name__ == '__main__':
    pass