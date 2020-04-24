#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/31 16:41
# @Author  : jyl
# @File    : data.py
import numpy as np
import cv2
from utils import CVTransform, xywh2xyxy, xyxy2xywh, letterbox_resize, iou_general, cv2plot, plot_boxes
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils import parse_anchors
from pathlib import Path
from utils import traverse_voc
import pickle
from skimage import io


class VOC2007:
    def __init__(self, opt):
        self.label_names = opt.VOC_BBOX_LABEL_NAMES
        self.label_names_dict = {name: index for index, name in enumerate(self.label_names)}
        self.ann_dir = Path(opt.data_dir) / 'Annotations'
        self.img_dir = Path(opt.data_dir) / 'JPEGImages'
        self.obj_dict_path = opt.obj_path
        if not Path(self.obj_dict_path).exists():
            traverse_voc(self.ann_dir, self.obj_dict_path)
        self.obj_dicts = pickle.load(open(self.obj_dict_path, 'rb'))
        self.filenames = [_ for _ in self.obj_dicts.keys()]

    def __len__(self):
        return len(self.filenames)

    def get_example(self, idx):
        filename = self.filenames[idx]
        obj_dict = self.obj_dicts[filename]
        obj_boxes = obj_dict['boxes']
        obj_names = obj_dict['names']
        obj_labels = [self.label_names_dict[name] for name in obj_names]
        img_path = self.img_dir / f'{filename}'
        img = io.imread(img_path)
        return img, np.array(obj_labels), np.asarray(obj_boxes)


def pytorch_normailze(img, mean, std):
    torch_normailze = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    img = torch_normailze(img)
    return img


class VOC2007Dataset(Dataset):
    """
    :return
    training:
        1.img:(batch_size,3,448,448)/tensor
        2.gt_bbox:(batch_size,-1,4)/tensor
        3.gt_label:(batch_size,-1)/ndarray
        4.scale:(batch_size,1,2)/ndarray
        5.y_true['target']:(13,13,5,25)/tensor
    """

    def __init__(self, opt):
        self.opt = opt
        self.database = VOC2007(opt)
        # anchor's scale is 416 / shape: [5, 2] / [w, h]
        self.anchor_base = parse_anchors(opt)
        self.image_aug = CVTransform(opt.aug_thresh)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, index):
        img_org, labels_org, boxes_org = self.database.get_example(index)
        img_aug, labels_aug, boxes_aug = self.image_aug(img_org, boxes_org, labels_org, 'RGB')
        resized_img, resized_boxes, _, _, _ = letterbox_resize(img_aug, boxes_aug, [self.opt.img_size, self.opt.img_size])
        target = self.make_target(resized_boxes, labels_aug, self.opt.img_size)
        img_norm = pytorch_normailze(resized_img, self.opt.mean, self.opt.std)
        return img_norm, target

    def get_example_for_testing(self, index):
        img_org, labels_org, boxes_org = self.database.get_example(index)
        img_aug, labels_aug, boxes_aug = self.image_aug(img_org, boxes_org, labels_org, 'RGB')
        resized_img, resized_boxes, _, _, _ = letterbox_resize(img_aug, boxes_aug, [self.opt.img_size, self.opt.img_size])
        target = self.make_target(resized_boxes, labels_aug, self.opt.img_size)
        img_norm = pytorch_normailze(resized_img, self.opt.mean, self.opt.std)
        return img_norm, target, img_org, labels_org, boxes_org

    def for_test(self, index):
        img, gt_labels, gt_boxes = self.database.get_example(index)
        img, gt_labels, gt_boxes = self.image_aug(img, gt_boxes, gt_labels, 'RGB')
        img, gt_boxes, _, _, _ = letterbox_resize(img, gt_boxes, [self.opt.img_size, self.opt.img_size])
        target = self.make_target(gt_boxes, gt_labels, self.opt.img_size)
        cv2plot(img, gt_boxes, gt_labels)
        plt.show()

    def make_target(self, gt_boxes, gt_labels, img_size):
        """
        :param gt_boxes: [M, 4] / [xmin, ymin, xmax, ymax] / ndarray
        :param gt_labels: [M,] / ndarray
        :param img_size: 416
        :return:
            [[x, y, w, h, label], ...] / shape: [13, 13, 5, 25]
            x, y, w, h的scale均为416
        """
        assert len(gt_boxes) == len(gt_labels)
        assert isinstance(gt_boxes, np.ndarray)
        assert isinstance(gt_labels, np.ndarray)

        grid_size = img_size / self.opt.S
        target = np.zeros((self.opt.S, self.opt.S, self.opt.B, 4 + 1 + self.opt.voc_class_num), dtype=np.float32)
        # [xmin, ymin, xmax, ymax] -> [center_x, center_y, w, h]
        gt_boxes_xywh = xyxy2xywh(gt_boxes)
        # [M, 2]
        box_coors = np.floor(gt_boxes_xywh[:, :2] / grid_size).astype(np.int32)
        keep_index = self._fliter_duplicate(box_coors)
        keep_box_coors = box_coors[keep_index]
        keep_labels = gt_labels[keep_index]
        keep_boxes = gt_boxes_xywh[keep_index]
        # 存在目标的cell的5个预测框中，只有gt_box与5个anchor的iou值最大的那个预测框的ground_truth才有值
        gt_anchor_ious = self._iou(keep_boxes, self.anchor_base)
        best_match = np.argmax(gt_anchor_ious, axis=-1)
        max_iou = np.max(gt_anchor_ious, axis=-1)

        for grid, k, iou, xywh, label in zip(keep_box_coors, best_match, max_iou, keep_boxes, keep_labels):
            target[grid[1], grid[0], k, :2] = xywh[:2]  # x,y
            target[grid[1], grid[0], k, 2:4] = xywh[2:]  # w,h
            target[grid[1], grid[0], k, 4] = 1.  # confidence
            # target[grid[1], grid[0], k, 4] = iou  # confidence
            target[grid[1], grid[0], k, 5 + label] = 1.  # label

        return target

    @staticmethod
    def resize_img_bbox(img_rgb, bbox):
        resized_img = cv2.resize(img_rgb, (opt.img_size, opt.img_size))
        w_scale = opt.img_size / img_rgb.shape[1]
        h_scale = opt.img_size / img_rgb.shape[0]
        resized_bbox = np.ceil(bbox * [h_scale, w_scale, h_scale, w_scale])
        return resized_img, resized_bbox

    @staticmethod
    def _fliter_duplicate(coor):
        """
        若同一个cell包含多个不同的目标，则只保留一个
        """
        keep = {str(v): idx for idx, v in enumerate(coor)}
        keep_index = list(keep.values())
        return keep_index

    @staticmethod
    def _iou(gt_boxes, base_anchors):
        """
        :param gt_boxes: [M, 4] / [ctr_x, ctr_y, w, h]
        :param base_anchors: [N, 2] / [w, h]
        :return: [M,]
        """
        dummy_anchors = np.zeros(shape=[len(base_anchors), 4])
        dummy_gt_boxes = np.zeros(shape=[len(gt_boxes), 4])
        dummy_anchors[:, 2:] = base_anchors
        dummy_gt_boxes[:, 2:] = gt_boxes[:, 2:]
        dummy_anchors = xywh2xyxy(dummy_anchors)
        dummy_gt_boxes = xywh2xyxy(dummy_gt_boxes)
        ious = iou_general(dummy_gt_boxes[:, None, :], dummy_anchors)
        # fig, ax = plt.subplots(1)
        # plot_boxes(dummy_anchors, ax, 'r')
        # plot_boxes(dummy_gt_boxes, ax, 'b')
        # plot_boxes(dummy_anchors[ious.argmax(axis=-1)], ax, 'g')
        # plt.show()
        return ious


def make_grid():
    grid_x = np.arange(13, dtype=np.float32)
    grid_y = np.arange(13, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    xy_offset = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return xy_offset.reshape(13, 13, 2)


class TestDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.database = VOC2007(opt)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, index):
        img_org, labels_org, boxes_org = self.database.get_example(index)
        resized_img, resized_boxes, ratio, dh, dw = letterbox_resize(img_org,
                                                                     boxes_org,
                                                                     [self.opt.img_size, self.opt.img_size])
        img_norm = pytorch_normailze(resized_img, self.opt.mean, self.opt.std)
        return img_org, img_norm, labels_org, boxes_org, ratio, dh, dw


if __name__ == '__main__':
    # vd = VocDataset(show_img=True, is_train=True)
    # img, target = vd[6]
    # print(vd.grid_idx[0])
    # print(vd.grid_labels[0])
    # print('resized_bbox:', vd.resized_bboxes[0])
    # print('xywh', vd.xywh[0] / 32)
    # for xywh in vd.xywh[0:1]:
    #     xmin = xywh[0] - xywh[2] / 2
    #     ymin = xywh[1] - xywh[3] / 2
    #     xmax = xywh[0] + xywh[2] / 2
    #     ymax = xywh[1] + xywh[3] / 2
    #     print([ymax, xmax, ymin, xmin])
    #
    # for id in vd.grid_idx[0:1]:
    #     print(target[id[0], id[1]])
    #
    #
    # a = np.arange(13)
    # b = np.arange(13)
    # x, y = np.meshgrid(a, b)
    # xy_offset = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=-1).reshape(13, 13, 2)
    # print(xy_offset[vd.grid_idx[0][0], vd.grid_idx[0][1]])
    from config import Config
    from pathlib import Path
    opt = Config(False)
    vd = VOC2007Dataset(opt)
    for i in range(1):
        random_id = np.random.randint(0, len(vd))
        vd.for_test(random_id)
