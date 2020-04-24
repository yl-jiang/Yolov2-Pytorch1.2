#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/22 下午5:03
# @Author  : jyl
# @File    : bbox_utils.py
import numpy as np
import torch


def yolov1_bbox_iou(gt_bbox, pre_bbox):
    """
    :param gt_bbox:
         [[ymax,xmax,ymin,xmin], ...]
    :param pre_bbox:
         [[ymax,xmax,ymin,xmin], ...]
    :return:
         [a, b, c,...]
    """
    assert gt_bbox.shape == pre_bbox.shape, "target_bbox and predic_bbox's shape must be same!"

    if isinstance(gt_bbox, torch.Tensor):
        gt_bbox = gt_bbox.cpu().detach().numpy()
    if isinstance(pre_bbox, torch.Tensor):
        pre_bbox = pre_bbox.cpu().detach().numpy()
    tl = np.maximum(gt_bbox[:, 2:], pre_bbox[:, 2:])  # (len(gt_bbox),2)
    br = np.minimum(gt_bbox[:, :2], pre_bbox[:, :2])  # (len(gt_bbox),2)
    area_i = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)  # (len(gt_bbox),)
    area_gt = np.prod(gt_bbox[:, :2] - gt_bbox[:, 2:], axis=1)  # (len(gt_bbox),)
    # 确保gt_bbox的前后两个bbox值相同
    assert (area_gt[::2] == area_gt[1::2]).all()
    area_pre = np.prod(pre_bbox[:, :2] - pre_bbox[:, 2:], axis=1)  # (len(gt_bbox),)
    area_pre_mask = area_pre > 0
    area_pre = area_pre * area_pre_mask.astype(np.uint8)
    iou = area_i / (area_pre + area_gt - area_i)  # (len(gt_bbox),)

    odd_iou = torch.tensor(iou[0::2])
    even_iou = torch.tensor(iou[1::2])
    ones = torch.ones_like(odd_iou)
    zeros = torch.zeros_like(odd_iou)
    odd_iou_mask = torch.where(odd_iou >= even_iou, ones, zeros)
    even_iou_mask = torch.where(even_iou > odd_iou, ones, zeros)
    iou_mask = torch.empty_like(torch.tensor(iou))
    # 确保odd_iou_mask和even_iou_mask对应元素值的和为1
    assert (odd_iou_mask + even_iou_mask == ones).byte().all()

    iou_mask[0::2] = odd_iou_mask
    iou_mask[1::2] = even_iou_mask
    return iou_mask


def iou_normal(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...]
    :param bbox2: [[xmin, ymin, xmax, ymax], ...]
    :return:
    """
    assert isinstance(bbox1, np.ndarray)
    assert isinstance(bbox2, np.ndarray)
    assert bbox1.ndim == 2
    assert bbox2.ndim == 2
    assert bbox1.shape[-1] == 4
    assert bbox2.shape[-1] == 4

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


def iou_general(bbox1, bbox2):
    """
    :param bbox1: format: [xmin, ymin, xmax, ymax]
    :param bbox2: format: [xmin, ymin, xmax, ymax]
    :return:
    """
    assert isinstance(bbox1, np.ndarray)
    assert isinstance(bbox2, np.ndarray)
    assert bbox1.shape[-1] == 4
    assert bbox2.shape[-1] == 4

    M = len(bbox1)
    N = len(bbox2)
    if M != N:
        assert bbox2.ndim != bbox1.ndim

    bbox1_area = np.prod(bbox1[..., [2, 3]] - bbox1[..., [0, 1]] + 1, axis=-1)
    bbox2_area = np.prod(bbox2[..., [2, 3]] - bbox2[..., [0, 1]] + 1, axis=-1)

    intersection_ymax = np.minimum(bbox1[..., 3], bbox2[..., 3])
    intersection_xmax = np.minimum(bbox1[..., 2], bbox2[..., 2])
    intersection_ymin = np.maximum(bbox1[..., 1], bbox2[..., 1])
    intersection_xmin = np.maximum(bbox1[..., 0], bbox2[..., 0])

    intersection_w = np.maximum(0., intersection_xmax - intersection_xmin + 1)
    intersection_h = np.maximum(0., intersection_ymax - intersection_ymin + 1)
    intersection_area = intersection_w * intersection_h
    ious = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    if M != N:
        assert ious.shape[-2:] == (M, N) or ious.shape[-2:] == (N, M)
    else:
        assert len(ious) == M
    return ious


def yolov2_bbox_iou(bbox1, bbox2):
    """
    :param bbox1:[center_x, center_y, w, h] / shape: [13, 13, 5, 4]
    :param bbox2:[center_x, center_y, w, h] / shape: [M, 4]
    :return:[13, 13, 5, M]
    """
    device = bbox1.device
    assert bbox1.shape[-1] == bbox2.shape[-1] == 4

    bbox1_xyxy = torch.empty_like(bbox1)
    bbox1_xyxy[..., 0] = bbox1[..., 0] - bbox1[..., 2] / 2
    bbox1_xyxy[..., 1] = bbox1[..., 1] - bbox1[..., 3] / 2
    bbox1_xyxy[..., 2] = bbox1[..., 0] + bbox1[..., 2] / 2
    bbox1_xyxy[..., 3] = bbox1[..., 1] + bbox1[..., 3] / 2

    bbox2_xyxy = torch.empty_like(bbox2)
    bbox2_xyxy[..., 0] = bbox2[..., 0] - bbox2[..., 2] / 2
    bbox2_xyxy[..., 1] = bbox2[..., 1] - bbox2[..., 3] / 2
    bbox2_xyxy[..., 2] = bbox2[..., 0] + bbox2[..., 2] / 2
    bbox2_xyxy[..., 3] = bbox2[..., 1] + bbox2[..., 3] / 2

    # expand dimension for broadcast :[13, 13, 5, 1, 4]
    bbox1_xyxy = torch.unsqueeze(bbox1_xyxy, -2)
    # [13, 13, 5, 1, 2] & [13, 13, 5, 1, 2] -> [13, 13, 5, 1]
    bbox1_area = torch.prod(bbox1_xyxy[..., [0, 1]] - bbox1_xyxy[..., [2, 3]] + 1, dim=-1)
    # [M, 2] & [M, 2] -> [M,]
    bbox2_area = torch.prod(bbox2_xyxy[..., [0, 1]] - bbox2_xyxy[..., [2, 3]] + 1, dim=-1)

    # [13, 13, 5, 1] & [M,] -> [13, 13, 5, M]
    intersection_xmin = torch.max(bbox1_xyxy[..., 0], bbox2_xyxy[..., 0])
    intersection_ymin = torch.max(bbox1_xyxy[..., 1], bbox2_xyxy[..., 1])
    intersection_xmax = torch.min(bbox1_xyxy[..., 2], bbox2_xyxy[..., 2])
    intersection_ymax = torch.min(bbox1_xyxy[..., 3], bbox2_xyxy[..., 3])
    # [13, 13, 5, M] & [13, 13, 5, M] -> [13, 13, 5, M]
    intersection_w = torch.max(intersection_xmax - intersection_xmin + 1, torch.tensor(0.).float().to(device))
    intersection_h = torch.max(intersection_ymax - intersection_ymin + 1, torch.tensor(0.).float().to(device))
    intersection_area = intersection_w * intersection_h
    # [13, 13, 5, M] & ([13, 13, 5, 1] & [M,] & [13, 13, 5, M]) -> [13, 13, 5, M]
    ious = intersection_area / (bbox1_area + bbox2_area - intersection_area + 1e-10)
    # ious shape: [13, 13, 5, M]
    return ious


def xyxy2xywh(boxes):
    """
    :param boxes:
        [[xmin, ymin, xmax, ymax], ...]
    :return:
        [[center_x, center_y, w, h], ...]
    """
    assert isinstance(boxes, np.ndarray)
    assert boxes.ndim == 2
    assert boxes.shape[-1] == 4
    assert (boxes[:, 2] >= boxes[:, 0]).all()
    assert (boxes[:, 3] >= boxes[:, 1]).all()

    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    ctr_x = boxes[:, 0] + ws / 2
    ctr_y = boxes[:, 1] + hs / 2

    xywh = np.stack([ctr_x, ctr_y, ws, hs], axis=1)
    return xywh


def xywh2xyxy(boxes):
    """
    :param boxes:
        element in the last dimension's format is: [[center_x, center_y, w, h], ...]
    :return:
        [[xmin, ymin, xmax, ymax], ...]
    """
    assert isinstance(boxes, np.ndarray)
    assert boxes.ndim == 2
    assert boxes.shape[-1] == 4

    xmin = boxes[:, 0] - boxes[:, 2] / 2
    ymin = boxes[:, 1] - boxes[:, 3] / 2
    xmax = boxes[:, 0] + boxes[:, 2] / 2
    ymax = boxes[:, 1] + boxes[:, 3] / 2

    xyxy = np.stack([xmin, ymin, xmax, ymax], axis=1)
    return xyxy


def resize_bbox(bboxes, org_shape, dst_shape):
    """
    :param bboxes: [[xmin, ymin, xmax, ymax], ...]
    :param org_shape: [width, heigth]
    :param dst_shape: [width, heigth]
    :return: [[xmin, ymin, xmax, ymax], ...]
    """
    assert isinstance(bboxes, np.ndarray)
    assert bboxes.ndim == 2, 'the dimension of bboxes must be 2.'

    w_scale = dst_shape[0] / org_shape[0]
    h_scale = dst_shape[1] / org_shape[1]
    resized_bboxes = np.empty_like(bboxes)
    resized_bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]] * w_scale, 0., dst_shape[0])
    resized_bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]] * h_scale, 0., dst_shape[1])

    return resized_bboxes


def plot_boxes(boxes, ax, c, alpha=0.5):
    assert isinstance(boxes, np.ndarray)
    assert boxes.shape[-1] == 4
    assert boxes.ndim == 2
    for box in boxes:
        ax.plot((box[0], box[2]), (box[1], box[1]), c=c, alpha=alpha)
        ax.plot((box[0], box[0]), (box[1], box[3]), c=c, alpha=alpha)
        ax.plot((box[0], box[2]), (box[3], box[3]), c=c, alpha=alpha)
        ax.plot((box[2], box[2]), (box[3], box[1]), c=c, alpha=alpha)


def inverse_letter_resize(boxes, scale, dh, dw):
    """
    :param boxes: [xmin, ymin, xmax, ymax] / ndarray / number dimension = 2
    :param scale: float
    :param dh: float
    :param dw: float
    :return:
    """
    assert isinstance(boxes, np.ndarray)
    assert boxes.ndim == 2
    assert boxes.shape[-1] == 4
    assert (boxes[:, [2, 3]] >= boxes[:, [0, 1]]).all()

    boxes_out = np.empty_like(boxes)
    boxes_out[:, [0, 2]] = (boxes[:, [0, 2]] / scale) - dw // 2
    boxes_out[:, [1, 3]] = (boxes[:, [1, 3]] / scale) - dh // 2
    return boxes_out


if __name__ == '__main__':
    gt_bbox = np.array([[2,3,0,1]])
    pre_bbox = np.array([[3,4,1,2]])


