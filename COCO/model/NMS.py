#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 下午10:00
# @Author  : jyl
# @File    : NMS.py
import torch
from utils import gpu_nms
def nms_each_class(boxes, scores, score_threshold, iou_threshold, max_box_num):
    """
    :param boxes: [13*13*5, 4]
    :param scores: [13*13*5, 20]
    :param score_threshold:
    :param iou_threshold:
    :param max_box_num:
    :return:
     boxes_output shape: [X, 4]
     scores_output shape: [X,]
     labels_output shape: [X,]
    """
    assert boxes.dim() == 2 and scores.dim() == 2

    boxes_output = []
    scores_output = []
    labels_output = []
    # [13*13*5, 20]
    score_mask = scores.ge(score_threshold)

    a = torch.sum(score_mask)
    # do nms for each class
    for k in range(80):
        valid_mask = score_mask[:, k]  # [M, 20]
        if valid_mask.sum() == 0:
            continue
        else:
            aa = torch.sum(valid_mask)
            valid_boxes = boxes[valid_mask]  # [M, 4]
            valid_scores = scores[:, k][valid_mask]  # [M, 1]
            keep_index = gpu_nms(valid_boxes, valid_scores, iou_threshold)
            for keep_box in valid_boxes[keep_index]:
                boxes_output.append(keep_box)
            scores_output.extend(valid_scores[keep_index])
            labels_output.extend([k for _ in range(len(keep_index))])

    num_out = len(labels_output)
    if num_out == 0:
        return torch.tensor([], device='cuda'), torch.tensor([], device='cuda'), torch.tensor([],
                                                                                                      device='cuda')
    else:
        boxes_output = torch.stack(boxes_output, dim=0)
        scores_output = torch.tensor(scores_output)
        labels_output = torch.tensor(labels_output)
        if num_out > max_box_num:
            descend_order_index = torch.argsort(scores_output)[::-1]
            output_index = descend_order_index[:max_box_num]
        else:
            output_index = torch.arange(num_out)
        return boxes_output[output_index], scores_output[output_index], labels_output[output_index]

def nms_all_class(boxes, scores, labels, score_threshold, iou_threshold, max_box_num):
    """
    :param boxes: [300, 4]
    :param scores: [300,]
    :param score_threshold: 0.3
    :param iou_threshold: 0.45
    :param max_box_num:
    :return:
     boxes_output shape: [X, 4]
     scores_output shape: [X,]
     labels_output shape: [X,]
    """
    # assert boxes.dim() == 2 and scores.dim() == 2

    boxes_output = []
    scores_output = []
    labels_output = []

    max_scores = scores
    # [13*13*5, 1]
    valid_mask = max_scores.ge(score_threshold)
    # do nms for all class
    if valid_mask.sum() != 0:
        valid_boxes = boxes[valid_mask]  # [M, 4]
        valid_scores = max_scores[valid_mask]  # [M, 1]
        valid_labels = labels[valid_mask]
        keep_index = gpu_nms(valid_boxes, valid_scores, iou_threshold)
        boxes_output = valid_boxes[keep_index]
        scores_output.extend(valid_scores[keep_index])
        labels_output.extend(valid_labels[keep_index])

    num_out = len(boxes_output)
    if num_out == 0:
        return torch.tensor([], device='cuda'), torch.tensor([], device='cuda'), torch.tensor([], device='cuda')
    else:
        boxes_output = torch.tensor(boxes_output)
        scores_output = torch.tensor(scores_output)
        labels_output = torch.tensor(labels_output)
        if num_out > max_box_num:
            descend_order_index = torch.argsort(scores_output)[::-1]
            output_index = descend_order_index[:max_box_num]
        else:
            output_index = torch.arange(num_out)
        return boxes_output[output_index], scores_output[output_index], labels_output[output_index]