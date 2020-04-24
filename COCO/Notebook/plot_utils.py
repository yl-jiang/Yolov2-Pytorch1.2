#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/30 下午4:18
# @Author  : jyl
# @File    : plot_utils.py
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


VOC_BBOX_LABEL_NAMES = opt.VOC_BBOX_LABEL_NAMES + ['BG']


def make_color_table(color_num):
    random.seed(7)  # random and transforms
    color_table = {}
    # '+2' means add background class and ground_truth
    for i in range(color_num + 2):
        color_table[i] = [random.random() for _ in range(3)]
    return color_table


def image_ax(img, ax=None):
    assert img.shape[-1] == 3
    if ax is None:
        fig = plt.figure(figsize=[16, 9])
        ax = fig.add_subplot(111)
    ax.imshow(img.astype(np.uint8))
    return ax


def plot_one(plot_dict):
    pred_box = plot_dict['pred_box']  # [xmin, ymin, xmax, ymax]
    pred_score = plot_dict['pred_score']
    pred_label = plot_dict['pred_label']
    img = plot_dict['img']
    gt_box = plot_dict['gt_box']
    gt_label = plot_dict['gt_label']

    color_table = make_color_table(opt.class_num)
    ax = image_ax(img)
    ax = draw_pred(ax, pred_box, pred_score, pred_label, color_table)
    ax = draw_gt(ax, gt_box, gt_label, color_table)
    ax.set_axis_off()
    plt.show()


def draw_pred(ax, boxes, confs, labels, color_table):
    """
    :param ax:
    :param boxes: [xmin, ymin, xmax, ymax]
    :param confs:
    :param labels:
    :param color_table:
    :return:
    """
    for j in range(len(boxes)):
        if len(boxes[j]) == 0:
            ax.text(0, 0, VOC_BBOX_LABEL_NAMES[-1], style='italic',
                    bbox={'facecolor': color_table[-2], 'alpha': 0.5, 'pad': 1})
            continue

        # [xmin, ymin, xmax, ymax]
        # xy:左上角坐标；width：框的宽度；height：框的高度
        xy = (boxes[j][0], boxes[j][1])
        width = boxes[j][2] - boxes[j][0]
        heigth = boxes[j][3] - boxes[j][1]
        label = VOC_BBOX_LABEL_NAMES[labels[j]]
        score = confs[j]
        ax.add_patch(plt.Rectangle(xy, width, heigth, fill=False, edgecolor=color_table[labels[j]], linewidth=1.5))
        caption = [label, '%.2f' % score]
        # 左上角
        ax.text(xy[0], xy[1], s=':'.join(caption), style='italic', bbox={'facecolor': color_table[labels[j]], 'alpha': 0.5, 'pad': 1})
        # logger.info(f'{img_id} [ymin, xmin, ymax, xmax] = [{xy[1]}, {xy[0]}, {xy[1]+heigth}, {xy[0]+width}]')
    return ax


def draw_gt(ax, gt_boxes, gt_labels, color_table):
    """
    :param ax:
    :param gt_boxes: [ymax, xmax, ymin, xmin]
    :param gt_labels:
    :param color_table:
    :return:
    """
    for j in range(len(gt_boxes)):
        # xy:左上角坐标；width：框的宽度；height：框的高度
        xy = (gt_boxes[j][3], gt_boxes[j][2])
        width = gt_boxes[j][1] - gt_boxes[j][3]
        heigth = gt_boxes[j][0] - gt_boxes[j][2]
        label = VOC_BBOX_LABEL_NAMES[gt_labels[j]]
        ax.add_patch(plt.Rectangle(xy, width, heigth, fill=False, edgecolor=color_table[opt.class_num+1], linewidth=1.5))
        # 左上角
        ax.text(xy[0], xy[1], f'GT-{label}', style='italic', bbox={'facecolor': color_table[opt.class_num+1], 'alpha': 0.5, 'pad': 1})
    return ax


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


