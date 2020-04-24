#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/16 下午1:07
# @Author  : jyl
# @File    : nms_utils.py
import numpy as np
import torch


def cpu_nms(boxes, scores, score_threshold, iou_threshold, max_num=None):
    """
    :param boxes:[N, 4] / 'N' means not sure
    :param scores:[N, 1]
    :param score_threshold: float
    :param iou_threshold:a scalar
    :param max_num:
    :return:keep_index
    """
    # boxes format : [xmin, ymin, xmax, ymax]
    assert isinstance(boxes, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert boxes.ndim == 2
    assert scores.ndim == 2
    assert boxes.shape[-1] == 4
    assert (boxes[2, 3] >= boxes[:, [0, 1]]).all(), 'boxes format must be [xmin, ymin, xmax, ymax]'
    assert len(boxes.shape) == len(scores)

    box_copy = boxes.copy()
    score_copy = scores.copy()

    ignore_mask = np.where(score_copy < score_threshold)[0]
    score_copy[ignore_mask] = 0.

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

    if max_num is not None and len(keep_index) > max_num:
        keep_index = keep_index[: max_num]

    return keep_index


def cpu_iou(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...]
    :param bbox2: [[xmin, ymin, xmax, ymax], ...]
    :return:
    """
    assert isinstance(bbox1, np.ndarray)
    assert isinstance(bbox2, np.ndarray)
    assert bbox1.ndim == 2
    assert bbox2.ndim == 2
    assert bbox1.shape[-1] == bbox2.shape[-1] == 4
    assert (bbox1[:, [2, 3]] >= bbox1[:, [0, 1]]).all(), 'format of bbox must be [xmin, ymin, xmax, ymax]'
    assert (bbox2[:, [2, 3]] >= bbox2[:, [0, 1]]).all(), 'format of bbox must be [xmin, ymin, xmax, ymax]'

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


def gpu_nms(boxes, scores, score_threshold, iou_threshold, max_num=None):
    """
    :param boxes: [M, 4]
    :param scores: [M, 1]
    :param score_threshold: float
    :param iou_threshold: float
    :param max_num:
    :return:
    """
    assert isinstance(boxes, torch.Tensor)
    assert isinstance(scores, torch.Tensor)
    assert boxes.dim() == 2
    assert scores.dim() == 2
    assert boxes.size(0) == scores.size(0)

    all_boxes = boxes.detach().clone()
    score_copy = scores.detach().clone()

    # ignore boxes whom score lower than score_threshold
    ignore_mask = scores.le(score_threshold)
    score_copy[ignore_mask] = 0.

    keep_index = []
    while torch.sum(score_copy) > 0.:
        # mark reserved box
        selected_index = score_copy.argmax().item()
        box1 = all_boxes[[selected_index]]
        keep_index.append(selected_index)
        score_copy[selected_index] = 0.
        ious = gpu_iou(box1, all_boxes)
        ignore_index = ious.ge(iou_threshold)
        score_copy[ignore_index] = 0.
    if max_num is not None and len(keep_index) > max_num:
        keep_index = keep_index[: max_num]
    return keep_index


def gpu_iou(bbox1, bbox2):
    """
    the shape of bbox1 and bbox2 is the same or at leat there is one box's first dimension is 1
    :param bbox1: shape: [M, 4] / [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor
    :param bbox2: shape: [N, 4] / [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor
    :return:
    """
    assert isinstance(bbox1, torch.Tensor)
    assert isinstance(bbox2, torch.Tensor)
    assert bbox1.dim() == 2
    assert bbox2.dim() == 2
    assert bbox1.shape[-1] == 4
    assert bbox2.shape[-1] == 4
    assert (bbox1[:, [2, 3]] >= bbox1[:, [0, 1]]).bool().all(), 'the format of boxes must be [xmin, ymin, xmax, ymax]'
    assert (bbox2[:, [2, 3]] >= bbox2[:, [0, 1]]).bool().all(), 'the format of boxes must be [xmin, ymin, xmax, ymax]'

    bbox1_area = torch.prod(bbox1[:, [2, 3]] - bbox1[:, [0, 1]] + 1, dim=-1)
    bbox2_area = torch.prod(bbox2[:, [2, 3]] - bbox2[:, [0, 1]] + 1, dim=-1)

    intersection_ymax = torch.min(bbox1[:, 3], bbox2[:, 3])
    intersection_xmax = torch.min(bbox1[:, 2], bbox2[:, 2])
    intersection_ymin = torch.max(bbox1[:, 1], bbox2[:, 1])
    intersection_xmin = torch.max(bbox1[:, 0], bbox2[:, 0])

    intersection_w = torch.max(torch.tensor(0.).float().cuda(), intersection_xmax - intersection_xmin + 1)
    intersection_h = torch.max(torch.tensor(0.).float().cuda(), intersection_ymax - intersection_ymin + 1)
    intersection_area = intersection_w * intersection_h
    ious = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return ious


def gpu_nms_mutil_class(boxes, scores, score_threshold, iou_threshold, max_num=None, min_size=None):
    """
    Do nms in each class.
    :param boxes: [M, 4] / [xmin, ymin, xmax, yamx]
    :param scores: [M, C]
    :param score_threshold: float
    :param iou_threshold: float
    :param max_num:
    :param min_size:
    :return:
    """
    assert isinstance(boxes, torch.Tensor)
    assert isinstance(scores, torch.Tensor)
    assert boxes.dim() == 2
    assert scores.dim() == 2
    assert boxes.size(-1) == 4
    assert (boxes[:, [2, 3]] >= boxes[:, [0, 1]]).all()
    assert boxes.size(0) == scores.size(0)

    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]
    areas = (xmax - xmin + 1) * (ymax - ymin + 1)

    keep_index_list = []
    class_num = scores.size(-1)
    for i in range(class_num):
        score = scores[:, [i]]
        if min_size is not None:
            areas_mask = areas.le(min_size)
            score[areas_mask] = 0.
        inds = gpu_nms(boxes, score, score_threshold, iou_threshold, max_num)
        keep_index_list.append(inds)
    return keep_index_list

