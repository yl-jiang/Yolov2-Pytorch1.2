#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/30 下午3:44
# @Author  : jyl
# @File    : nms_utils.py
import torch


def nms(boxes, scores, score_threshold, iou_threshold, max_box_num, device, class_num, img_size):
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
    boxes = boxes.clamp(0., img_size)
    boxes_output = []
    scores_output = []
    labels_output = []
    # [13*13*5, 20]
    score_mask = scores.ge(score_threshold)
    # do nms for each class
    for k in range(class_num):
        valid_mask = score_mask[:, k]  # [M, 20]
        if valid_mask.sum() == 0:
            continue
        else:
            valid_boxes = boxes[valid_mask]  # [M, 4]
            valid_scores = scores[:, k][valid_mask]  # [M, 1]
            keep_index = gpu_nms(valid_boxes, valid_scores, iou_threshold)
            boxes_output.append(valid_boxes[keep_index])
            scores_output.extend(valid_scores[keep_index])
            labels_output.extend([k for _ in range(len(keep_index))])

    num_out = len(boxes_output)
    if num_out == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor([],
                                                                                                      device=device)
    else:
        boxes_output = torch.cat(boxes_output, dim=0)
        scores_output = torch.tensor(scores_output)
        labels_output = torch.tensor(labels_output)
        if num_out > max_box_num:
            descend_order_index = torch.argsort(scores_output)[::-1]
            output_index = descend_order_index[:max_box_num]
        else:
            output_index = torch.arange(num_out)
        return boxes_output[output_index], scores_output[output_index], labels_output[output_index]

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
