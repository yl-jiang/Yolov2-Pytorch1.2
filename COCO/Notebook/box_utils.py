#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/30 下午3:08
# @Author  : jyl
# @File    : box_utils.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/22 下午5:03
# @Author  : jyl
# @File    : bbox_utils.py
import numpy as np
import torch
from config import opt


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


def iou(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...]
    :param bbox2: [[xmin, ymin, xmax, ymax], ...]
    :return:
    """
    assert bbox1.shape == bbox2.shape, 'for computing IOU, the shape of input two bboxes must be the same.'

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


def yolov2_bbox_iou(bbox1, bbox2, device):
    """
    :param bbox1:
        [center_x, center_y, w, h] / shape: [13, 13, 5, 4]
    :param bbox2:
        [center_x, center_y, w, h] / shape: [M, 4]
    :return:
        [13, 13, 5, M]
    """
    assert bbox1.shape[-1] == bbox2.shape[-1] == 4

    bbox2 = xywh2xyxy(bbox2)
    bbox1 = xywh2xyxy(bbox1)
    # expand dimension for broadcast :[13, 13, 5, 1, 4]
    bbox1 = torch.unsqueeze(bbox1, -2)
    # [13, 13, 5, 1, 2] & [13, 13, 5, 1, 2] -> [13, 13, 5, 1]
    bbox1_area = torch.prod(bbox1[..., [0, 1]] - bbox1[..., [2, 3]], dim=-1)
    # [M, 2] & [M, 2] -> [M,]
    bbox2_area = torch.prod(bbox2[..., [0, 1]] - bbox2[..., [2, 3]], dim=-1)

    # [13, 13, 5, 1] & [M,] -> [13, 13, 5, M]
    intersection_xmin = torch.max(bbox1[..., 0], bbox2[..., 0])
    intersection_ymin = torch.max(bbox1[..., 1], bbox2[..., 1])
    intersection_xmax = torch.min(bbox1[..., 2], bbox2[..., 2])
    intersection_ymax = torch.min(bbox1[..., 3], bbox2[..., 3])
    # [13, 13, 5, M] & [13, 13, 5, M] -> [13, 13, 5, M]
    intersection_w = torch.max(intersection_xmax - intersection_xmin, torch.tensor(0., device=device))
    intersection_h = torch.max(intersection_ymax - intersection_ymin, torch.tensor(0., device=device))
    intersection_area = intersection_w * intersection_h
    # [13, 13, 5, M] & ([13, 13, 5, 1] & [M,] & [13, 13, 5, M]) -> [13, 13, 5, M]
    ious = intersection_area / (bbox1_area + bbox2_area - intersection_area + 1e-10)
    # ious shape: [13, 13, 5, M]
    return ious


def xyxy2xywh(bbox_xyxy):
    """
    :param bbox_yxyx:
        [[xmin, ymin, xmax, ymax], ...]
    :return:
        [[center_x, center_y, w, h], ...]
    """
    assert isinstance(bbox_xyxy, torch.Tensor)

    x = (bbox_xyxy[..., [2]] + bbox_xyxy[..., [0]]) / 2
    y = (bbox_xyxy[..., [3]] + bbox_xyxy[..., [1]]) / 2
    w = bbox_xyxy[..., [2]] - bbox_xyxy[..., [0]]
    h = bbox_xyxy[..., [3]] - bbox_xyxy[..., [1]]

    xywh = torch.cat([x, y, w, h], dim=1)
    return xywh


def xywh2xyxy(bbox_xywh):
    """
    :param bbox_xywh:
        element in the last dimension's format is: [[center_x, center_y, w, h], ...]
    :return:
        [[xmin, ymin, xmax, ymax], ...]
    """
    assert isinstance(bbox_xywh, torch.Tensor)

    ymax = bbox_xywh[..., [1]] + bbox_xywh[..., [3]] / 2
    xmax = bbox_xywh[..., [0]] + bbox_xywh[..., [2]] / 2
    ymin = bbox_xywh[..., [1]] - bbox_xywh[..., [3]] / 2
    xmin = bbox_xywh[..., [0]] - bbox_xywh[..., [2]] / 2

    yxyx = torch.cat([xmin, ymin, xmax, ymax], dim=-1)
    return yxyx


def resize_bbox(bboxes, ori_img_shape, dest_img_shape):
    """
    :param bboxes:
        [[ymax, xmax, ymin, xmin], ...]
    :param ori_img_shape:
        [width, heigth]
    :param dest_img_shape:
        [width, heigth]
    :return:
    """
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.array(bboxes)

    assert len(bboxes.shape) == 2, 'the dimension of bboxes must be 2.'

    w_scale = dest_img_shape[0] / ori_img_shape[0]
    h_scale = dest_img_shape[1] / ori_img_shape[1]
    resized_bboxes = np.empty_like(bboxes)
    resized_bboxes[:, [1, 3]] = np.ceil(bboxes[:, [1, 3]] * w_scale)
    resized_bboxes[:, [0, 2]] = np.ceil(bboxes[:, [0, 2]] * h_scale)

    return resized_bboxes


if __name__ == '__main__':
    gt_bbox = np.array([[2,3,0,1]])
    pre_bbox = np.array([[3,4,1,2]])

    print(iou(gt_bbox, pre_bbox))



