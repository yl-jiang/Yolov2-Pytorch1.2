#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/5 下午3:04
# @Author  : jyl
# @File    : coco_dataset.py
import numpy as np
import cv2
import os
from config import opt
from pycocotools.coco import COCO
from utils import CVTransform
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils import parse_anchors


class COCODataset(Dataset):
    """
    :return
    training:
        1.img:(batch_size,3,448,448)/tensor
        2.gt_bbox:(batch_size,-1,4)/tensor
        3.gt_label:(batch_size,-1)/ndarray
        4.scale:(batch_size,1,2)/ndarray
        5.y_true['target']:(13,13,5,25)/tensor
    """

    def __init__(self, coco, img_dir, is_train=True, show_img=False):
        self.is_train = is_train
        self.show_img = show_img
        # if self.is_train:
        #     self.anns_path = os.path.join(opt.COCO2017_dir, 'train_val_ann', 'annotations', 'instances_train2017.json')
        #     self.img_dir = os.path.join(opt.COCO2017_dir, 'train2017')
        # else:
        #     self.anns_path = os.path.join(opt.COCO2017_dir, 'train_val_ann', 'annotations', 'instances_val2017.json')
        #     self.img_dir = os.path.join(opt.COCO2017_dir, 'val2017')
        # anchor's scale is 416 / shape: [5, 2]
        self.img_dir = img_dir
        self.anchors = parse_anchors(opt.anchors_path)
        self.coco = coco
        self.images = self.coco.dataset['images']
        self.anns = self.coco.dataset['annotations']
        self.id2cat, self.cat2id = self.get_id_cat_dicts()
        self.mark = 0

    def __len__(self):
        return len(self.images)

    def get_id_cat_dicts(self):
        catIds = self.coco.getCatIds()
        cats = self.coco.loadCats(catIds)
        catNames = [cat['name'] for cat in cats]
        id2cat_dict = dict(zip(catIds, catNames))
        cat2id_dict = dict(zip(catNames, range(len(cats))))
        return id2cat_dict, cat2id_dict

    def __getitem__(self, index):
        img_id = self.images[index]['id']
        img_filename = self.images[index]['file_name']
        img_path = os.path.join(self.img_dir, f'{img_filename}')
        img_bgr = cv2.imread(img_path)
        bboxes = []
        labels_id = []
        labels_name = []
        for ann in self.anns:
            if img_id == ann['image_id']:
                box = ann['bbox']  # [xmin, ymin, w, h]
                xmin = box[0]
                ymin = box[1]
                xmax = box[0] + box[2]
                ymax = box[1] + box[3]
                bboxes.append([ymax, xmax, ymin, xmin])  # to suite for CVTransfrom
                catId = ann['category_id']
                cat = self.id2cat[catId]
                labels_name.append(cat)
                labels_id.append(self.cat2id[cat])
        if len(bboxes) == 0:
            random_index = np.random.randint(low=0, high=len(self.images))
            self.mark += 1
            print(self.mark)
            return self.__getitem__(random_index)

        bboxes = np.asarray(bboxes)
        labels_id = np.asarray(labels_id)
        if self.is_train:
            img_aug = CVTransform(img_bgr, bboxes, labels_id)
            img_bgr, bboxes, labels = img_aug.img, img_aug.bboxes, img_aug.labels
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # resize_bboxes: [ymax, xmax, ymin, xmin]
        resized_img, self.resized_bboxes = self.letterbox_resize(img_rgb, bboxes, [opt.img_h, opt.img_w])
        self.grid_idx, self.grid_labels, self.xywh, target = self.make_target(self.resized_bboxes, labels_id, opt.img_size)

        if self.show_img:
            font = cv2.FONT_ITALIC
            for i, bbox in enumerate(self.resized_bboxes):
                bbox = bbox.astype(np.uint16)
                cv2.rectangle(resized_img, (bbox[3], bbox[2]), (bbox[1], bbox[0]), (55, 255, 155), 1)
                cv2.putText(resized_img, labels_name[i], (bbox[3], bbox[2]), font, 0.3, (255, 0, 0), 1)
            fig = plt.figure(figsize=(28, 14))
            ax1 = fig.add_subplot(111)
            ax1.xaxis.set_major_locator(plt.MultipleLocator(32))
            ax1.yaxis.set_major_locator(plt.MultipleLocator(32))
            ax1.grid(which='major', axis='x', linewidth=1, linestyle='-', color='0.01')
            ax1.grid(which='major', axis='y', linewidth=1, linestyle='-', color='0.01')
            ax1.imshow(resized_img)
            # ax1.axis('off')
            plt.show()

        img = self.normailze(resized_img, opt.mean, opt.std)
        if self.is_train:
            # target:[[x, y, w, h, label], ...] / shape: [13, 13, 5, 25]
            return img, target
        else:
            return resized_img, self.resized_bboxes, self.grid_labels

    def make_target(self, resized_bboxes, labels, img_size):
        """
        :param resized_bboxes: [M, 4]
        :param labels: [M,]
        :param img_size: 416
        :return:
            [[x, y, w, h, label], ...] / shape: [13, 13, 5, 25]
            x, y, w, h的scale均为416
        """
        grid_h, grid_w = img_size / opt.S, img_size / opt.S
        target = np.zeros((opt.S, opt.S, opt.B, 4 + 1 + opt.coco_class_num), dtype=np.float32)
        xywh = self.yxyx2xywh(resized_bboxes)  # [center_x, center_y, w, h]
        # [M, 2] / [row_id, col_id]
        grid_idx = np.floor(xywh[:, [1, 0]] / [grid_h, grid_w]).astype(np.int16)
        # grid_idx, grid_labels, xywh = self.remove_duplicate(grid_idx, labels, xywh)
        # [M,] / 存在目标的cell的5个预测框中，只有gt_box与5个anchor的iou值最大的那个预测框的ground_truth才有值
        anchor_mask, max_iou = self.anchor_mask(xywh[:, 2:], self.anchors)

        for idx, k, iou, xy, wh, label in zip(grid_idx, anchor_mask, max_iou, xywh[:, [0, 1]], xywh[:, [2, 3]], labels):
            target[idx[0], idx[1], k, [0, 1]] = [xy[0], xy[1]]  # x,y
            target[idx[0], idx[1], k, [2, 3]] = [wh[0], wh[1]]  # w,h
            target[idx[0], idx[1], k, 4] = 1.  # confidence
            target[idx[0], idx[1], k, 5 + label] = 1.  # label

        return grid_idx, labels, xywh, target

    @staticmethod
    def yxyx2xywh(bboxes):
        bboxes = np.asarray(bboxes) if not isinstance(bboxes, np.ndarray) else bboxes
        new_bbox = np.zeros_like(bboxes)
        hw = bboxes[:, [0, 1]] - bboxes[:, [2, 3]]
        yx = (bboxes[:, [2, 3]] + bboxes[:, [0, 1]]) / 2  # [center_y, center_x]
        new_bbox[:, [1, 0]] = yx
        new_bbox[:, [3, 2]] = hw
        # [x, y, w, h]
        return new_bbox

    @staticmethod
    def resize_img_bbox(img_rgb, bbox):
        resized_img = cv2.resize(img_rgb, (opt.img_size, opt.img_size))
        w_scale = opt.img_size / img_rgb.shape[1]
        h_scale = opt.img_size / img_rgb.shape[0]
        resized_bbox = np.ceil(bbox * [h_scale, w_scale, h_scale, w_scale])
        return resized_img, resized_bbox

    @staticmethod
    def letterbox_resize(img_rgb, bboxes, target_img_size):
        """
        :param img_rgb:
        :param bbox: format [ymax, xmax, ymin, xmin]
        :param target_img_size: [416, 416]
        :return:
            letterbox_img
            resized_bbox: [ymax, xmax, ymin, xmin]
        """
        bboxes = np.asarray(bboxes) if not isinstance(bboxes, np.ndarray) else bboxes
        letterbox_img = np.full(shape=[target_img_size[0], target_img_size[1], 3], fill_value=128, dtype=np.float32)
        org_img_shape = [img_rgb.shape[0], img_rgb.shape[1]]
        ratio = np.min([target_img_size[0] / org_img_shape[0], target_img_size[1] / org_img_shape[1]])
        # resized_shape format : [h, w]
        resized_shape = tuple([int(org_img_shape[0] * ratio), int(org_img_shape[1] * ratio)])
        resized_img = cv2.resize(img_rgb, resized_shape[::-1])
        dh = target_img_size[0] - resized_shape[0]
        dw = target_img_size[1] - resized_shape[1]
        letterbox_img[(dh//2):(dh//2+resized_shape[0]), (dw//2):(dw//2+resized_shape[1]), :] = resized_img
        resized_bbox = bboxes * ratio
        resized_bbox[:, [0, 2]] += dh // 2
        resized_bbox[:, [1, 3]] += dw // 2
        letterbox_img = letterbox_img.astype(np.uint8)
        return letterbox_img, resized_bbox

    @staticmethod
    def normailze(img, mean, std):
        torch_normailze = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        img = torch_normailze(img)
        return img

    @staticmethod
    def remove_duplicate(bboxes, labels, center_wh):
        """
        若同一个cell包含多个不同的目标，则只保留一个
        """
        container = {}
        assert bboxes.shape[0] == len(labels)
        mark = 0
        index = 0
        remove_ids = []
        for key, value in zip(bboxes, labels):
            container.setdefault(tuple(key), value)
            if len(container.keys()) == mark:
                remove_ids.append(index)
            mark = len(container.keys())
            index += 1
        center_wh_clear = np.delete(center_wh, remove_ids, axis=0)
        return np.array([list(k) for k in container.keys()]), list(container.values()), center_wh_clear

    @staticmethod
    def anchor_mask(bbox1, bbox2):
        """
        :param bbox1: [M, 2]
        :param bbox2: [N, 2]
        :return: [M,]
        """
        # [M, 1, 2]
        bbox1 = np.expand_dims(bbox1, axis=1)
        # [M, 1]
        bbox1_area = np.prod(bbox1, axis=-1)
        # [N,]
        bbox2_area = np.prod(bbox2, axis=-1)
        # [M, N]
        union_area = bbox1_area + bbox2_area

        # [M, 1, 2] & [N, 2] ->  [M, N, 2]
        intersection_min = np.maximum(-bbox1 / 2, -bbox2 / 2)
        intersection_max = np.minimum(bbox1 / 2, bbox2 / 2)
        # [M, N, 2]
        intersection_wh = intersection_max - intersection_min
        # [M, N]
        intersection_area = np.prod(intersection_wh, axis=-1)
        # [M, N]
        iou = intersection_area / (union_area - intersection_area)
        anchor_mask = np.argmax(iou, axis=-1)
        max_iou = np.max(iou, axis=-1)
        return anchor_mask, max_iou


if __name__ == '__main__':
    is_train = True
    if is_train:
        anns_path = os.path.join(opt.COCO2017_dir, 'train_val_ann', 'annotations', 'instances_train2017.json')
        img_dir = os.path.join(opt.COCO2017_dir, 'train2017')
    else:
        anns_path = os.path.join(opt.COCO2017_dir, 'train_val_ann', 'annotations', 'instances_val2017.json')
        img_dir = os.path.join(opt.COCO2017_dir, 'val2017')
    coco = COCO(anns_path)
    vd = COCODataset(coco, img_dir, show_img=True, is_train=True)
    img, target = vd[6]
    print(vd.grid_idx[0])
    print(vd.grid_labels[0])
    print('resized_bbox:', vd.resized_bboxes[0])
    print('xywh', vd.xywh[0] / 32)
    for xywh in vd.xywh[0:1]:
        xmin = xywh[0] - xywh[2] / 2
        ymin = xywh[1] - xywh[3] / 2
        xmax = xywh[0] + xywh[2] / 2
        ymax = xywh[1] + xywh[3] / 2
        print([ymax, xmax, ymin, xmin])

    for id in vd.grid_idx[0:1]:
        print(target[id[0], id[1]])


    a = np.arange(13)
    b = np.arange(13)
    x, y = np.meshgrid(a, b)
    xy_offset = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=-1).reshape(13, 13, 2)[:, :, None, :]
    target = target[..., :2] / 32 - xy_offset
    print(target[vd.grid_idx[0][0], vd.grid_idx[0][1]])
    #
    #
    # vd  = COCODataset(show_img=True, is_train=False)
    # resized_img, resized_bboxes, grid_labels = vd[2560]

