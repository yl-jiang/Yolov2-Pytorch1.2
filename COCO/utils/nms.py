#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/16 下午1:07
# @Author  : jyl
# @File    : nms_utils.py
import numpy as np
import torch


def cpu_nms(boxes, scores, iou_threshold):
    """
    :param boxes:
        [N, 4] / 'N' means not sure
    :param scores:
        [N, 1]
    :param iou_threshold:
        a scalar
    :return:
        keep_index
    """
    # boxes format : [xmin, ymin, xmax, ymax]
    assert isinstance(boxes, np.ndarray) and isinstance(scores, np.ndarray)
    assert boxes.shape[0] == scores.shape[0]
    box_copy = boxes.copy()
    score_copy = scores.copy()
    keep_index = []
    while np.sum(score_copy) > 0.:
        # mark reserved box
        max_score_index = np.argmax(score_copy)
        box1 = box_copy[[max_score_index]]
        keep_index.append(max_score_index)
        score_copy[max_score_index] = 0.
        ious = cpu_iou(box1, box_copy)
        # mark unuseful box
        # keep_mask shape [N,] / 'N' means uncertain
        del_index = np.greater(ious, iou_threshold)
        score_copy[del_index] = 0.

    return keep_index


def cpu_iou(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...]
    :param bbox2: [[xmin, ymin, xmax, ymax], ...]
    :return:
    """
    assert bbox1.shape[-1] == bbox2.shape[-1] == 4

    bbox1_area = np.prod(bbox1[:, [2, 3]] - bbox1[:, [0, 1]] + 1, axis=-1)
    bbox2_area = np.prod(bbox2[:, [2, 3]] - bbox2[:, [0, 1]] + 1, axis=-1)

    intersection_ymax = np.minimum(bbox1[:, 3], bbox2[:, 3])
    intersection_xmax = np.minimum(bbox1[:, 2], bbox2[:, 2])
    intersection_ymin = np.maximum(bbox1[:, 1], bbox2[:, 1])
    intersection_xmin = np.maximum(bbox1[:, 0], bbox2[:, 0])

    intersection_w = np.maximum(0., intersection_xmax - intersection_xmin + 1)
    intersection_h = np.maximum(0., intersection_ymax - intersection_ymin + 1)
    intersection_area = intersection_w * intersection_h
    iou_out = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return iou_out


def gpu_nms(boxes, scores, iou_threshold):
    """
    :param boxes: [M, 4]
    :param scores: [M, 1]
    :param iou_threshold:
    :return:
    """
    assert isinstance(boxes, torch.Tensor) and isinstance(scores, torch.Tensor)
    assert boxes.shape[0] == scores.shape[0]

    box_copy = boxes.detach().clone()
    score_copy = scores.detach().clone()
    keep_index = []
    while torch.sum(score_copy) > 0.:
        # mark reserved box
        max_score_index = torch.argmax(score_copy).item()
        box1 = box_copy[[max_score_index]]
        keep_index.append(max_score_index)
        score_copy[max_score_index] = 0.
        ious = gpu_iou(box1, box_copy)
        del_index = ious.gt(iou_threshold)
        score_copy[del_index] = 0.

    return keep_index


def gpu_iou(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor
    :param bbox2: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor
    :return:
    """
    assert bbox1.shape[-1] == bbox2.shape[-1] == 4

    bbox1_area = torch.prod(bbox1[:, [2, 3]] - bbox1[:, [0, 1]] + 1, dim=-1)
    bbox2_area = torch.prod(bbox2[:, [2, 3]] - bbox2[:, [0, 1]] + 1, dim=-1)

    intersection_ymax = torch.min(bbox1[:, 3], bbox2[:, 3])
    intersection_xmax = torch.min(bbox1[:, 2], bbox2[:, 2])
    intersection_ymin = torch.max(bbox1[:, 1], bbox2[:, 1])
    intersection_xmin = torch.max(bbox1[:, 0], bbox2[:, 0])

    intersection_w = torch.max(torch.tensor(0., dtype=torch.float32, device='cuda'), intersection_xmax - intersection_xmin + 1)
    intersection_h = torch.max(torch.tensor(0., dtype=torch.float32, device='cuda'), intersection_ymax - intersection_ymin + 1)
    intersection_area = intersection_w * intersection_h
    iou_out = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return iou_out
