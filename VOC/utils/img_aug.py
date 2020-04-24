#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/9 17:05
# @Author  : jyl
# @File    : img_aug.py
import random
import cv2
import numpy as np


class CVTransform:
    def __init__(self, aug_thresh=0.25):
        self.aug_thresh = aug_thresh

    def __call__(self, img, bboxes, labels, img_mode='RGB'):
        """
        :param img:
        :param bboxes: [xmin, ymin, xmax, ymax]
        :param labels:
        :param img_mode:
        :return:
        """
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[-1] == 3
        if img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = img
        img, bboxes = self.randomFlip(img, bboxes)
        # self.img, self.bboxes = self.randomScale(self.img, self.bboxes)
        img = self.randomBlur(img)
        img = self.RandomBrightness(img)
        img = self.RandomHue(img)
        img = self.RandomSaturation(img)
        img, bboxes, labels = self.randomShift(img, bboxes, labels)
        # print('shift:', self.bboxes)
        img, bboxes, labels = self.randomCrop(img, bboxes, labels)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, labels, bboxes

    @classmethod
    def _check_input(cls, cv_img, bboxes, labels):
        if not isinstance(cv_img, np.ndarray):
            raise ValueError("Image's type must be ndarray")
        if len(cv_img.shape) < 3:
            raise ValueError("Image must be colorful")
        if not isinstance(bboxes, np.ndarray):
            raise ValueError("bboxes's type must be ndarray")
        if not isinstance(labels, np.ndarray):
            raise ValueError("labels's type must be ndarray")
        return cls(cv_img, bboxes, labels)

    def randomFlip(self, img, bboxes):
        """
        :param img:
        :param bboxes: [xmin, ymin, xmax, ymax]
        :return:
        """
        # 垂直翻转/y坐标不变，x坐标变化
        if random.random() < self.aug_thresh:
            img = np.fliplr(img).copy()
            h, w, _ = img.shape
            xmax = w - bboxes[:, 0]
            xmin = w - bboxes[:, 2]
            bboxes[:, 0] = xmin
            bboxes[:, 2] = xmax
            return img, bboxes
        return img, bboxes

    def randomScale(self, img, bboxes):
        """
        :param img:
        :param bboxes: [xmin, ymin, xmax, ymax]
        :return:
        """
        # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < self.aug_thresh:
            scale = random.uniform(0.8, 1.2)
            h, w, _ = img.shape
            # cv2.resize(img, shape)/其中shape->[宽，高]
            img = cv2.resize(img, (int(w * scale), h))
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale
            return img, bboxes
        return img, bboxes

    def randomBlur(self, img):
        # 均值滤波平滑图像
        if random.random() < self.aug_thresh:
            img = cv2.blur(img, (5, 5))
            return img
        return img

    def RandomHue(self, img_bgr):
        # 图片色调
        if random.random() < self.aug_thresh:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return img_bgr
        return img_bgr

    def RandomSaturation(self, img_bgr):
        # 图片饱和度
        if random.random() < self.aug_thresh:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return img_bgr
        return img_bgr

    def RandomBrightness(self, img_bgr):
        # 图片亮度
        if random.random() < self.aug_thresh:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            # hsv分别表示：色调（H），饱和度（S），明度（V）
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return img_bgr
        return img_bgr

    def randomShift(self, img, bboxes, labels):
        """
        :param img:
        :param bboxes: [xmin, ymin, xmax, ymax]
        :param labels:
        :return:
        """
        # 随机平移
        hs = bboxes[:, 3] - bboxes[:, 1] + 1
        ws = bboxes[:, 2] - bboxes[:, 0] + 1
        ctr_x = bboxes[:, 0] + ws / 2
        ctr_y = bboxes[:, 1] + hs / 2
        if random.random() < self.aug_thresh:
            h, w, c = img.shape
            after_shfit_image = np.zeros((h, w, c), dtype=img.dtype)
            # after_shfit_image每行元素都设为[104,117,123]
            after_shfit_image[:, :, :] = (128, 128, 128)  # bgr
            shift_x = int(random.uniform(-w * 0.2, w * 0.2))
            shift_y = int(random.uniform(-h * 0.2, h * 0.2))
            # 图像平移
            if shift_x >= 0 and shift_y >= 0:  # 向下向右平移
                after_shfit_image[shift_y:, shift_x:, :] = img[:h - shift_y, :w - shift_x, :]
                min_x, min_y, max_x, max_y = shift_x, shift_y, w, h
            elif shift_x >= 0 and shift_y < 0:  # 向上向右平移
                after_shfit_image[:h + shift_y, shift_x:, :] = img[-shift_y:, :w - shift_x, :]
                min_x, min_y, max_x, max_y = shift_x, 0, w, h - shift_y
            elif shift_x <= 0 and shift_y >= 0:  # 向下向左平移
                after_shfit_image[shift_y:, :w + shift_x, :] = img[:h - shift_y, -shift_x:, :]
                min_x, min_y, max_x, max_y = 0, shift_y, w, h
            else:  # 向上向左平移
                after_shfit_image[:h + shift_y, :w + shift_x, :] = img[-shift_y:, -shift_x:, :]
                min_x, min_y, max_x, max_y = 0, 0, w - shift_x, h - shift_y

            ctr_shift_y = ctr_y + shift_y
            ctr_shift_x = ctr_x + shift_x
            mask1 = (ctr_shift_x > 0) & (ctr_shift_x < w)
            mask2 = (ctr_shift_y > 0) & (ctr_shift_y < h)
            mask = np.logical_and(mask1, mask2)
            boxes_in = bboxes[mask]
            # 如果做完平移后bbox的中心点被移到了图像外，就撤销平移操作
            if len(boxes_in) == 0:
                return img, bboxes, labels
            else:
                # bbox平移
                boxes_in[:, [1, 3]] = np.clip(boxes_in[:, [1, 3]] + shift_y, a_min=min_y, a_max=max_y-1)
                boxes_in[:, [0, 2]] = np.clip(boxes_in[:, [0, 2]] + shift_x, a_min=min_x, a_max=max_x-1)
                labels_in = labels[mask]
                return after_shfit_image, boxes_in, labels_in
        return img, bboxes, labels

    def randomCrop(self, img, bboxes, labels):
        """
        :param img:
        :param bboxes: [xmin, ymin, xmax, ymax]
        :param labels:
        :return:
        """
        # 随机裁剪
        if random.random() < self.aug_thresh:
            box_hs = bboxes[:, 3] - bboxes[:, 1] + 1
            box_ws = bboxes[:, 2] - bboxes[:, 0] + 1
            ctr_x = bboxes[:, 0] + box_ws / 2
            ctr_y = bboxes[:, 1] + box_hs / 2

            height, width, c = img.shape
            # x,y代表裁剪后的图像的中心坐标，h,w表示裁剪后的图像的高，宽
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(width / 4, 3 * width / 4)
            y = random.uniform(height / 4, 3 * height / 4)
            x, y, h, w = int(x), int(y), int(h), int(w)

            new_img_lt_x = np.clip(x - (w / 2), a_min=0, a_max=width).astype(np.int32)
            new_img_lt_y = np.clip(y - (h / 2), a_min=0, a_max=height).astype(np.int32)
            new_img_rb_x = np.clip(x + (w / 2), a_min=0, a_max=width).astype(np.int32)
            new_img_rb_y = np.clip(y + (h / 2), a_min=0, a_max=height).astype(np.int32)

            mask1 = (ctr_y < new_img_rb_y) & (ctr_x < new_img_rb_x)
            mask2 = (ctr_y > new_img_lt_y) & (ctr_x > new_img_lt_x)
            mask = np.logical_and(mask1, mask2)
            bbox_in = bboxes[mask]
            if len(bbox_in) == 0:
                return img, bboxes, labels
            else:
                new_width = new_img_rb_x - new_img_lt_x
                new_height = new_img_rb_y - new_img_lt_y
                bbox_in[:, [1, 3]] = np.clip(bbox_in[:, [1, 3]] - new_img_lt_y, a_min=0, a_max=new_height-1)
                bbox_in[:, [0, 2]] = np.clip(bbox_in[:, [0, 2]] - new_img_lt_x, a_min=0, a_max=new_width-1)
                labels_in = labels[mask]
                new_img = img[new_img_lt_y:new_img_rb_y, new_img_lt_x:new_img_rb_x, :]
                return new_img, bbox_in, labels_in
        else:
            return img, bboxes, labels


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    img_path = r'/home/dk/jyl/Object_Detection/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_000423.jpg'
    bbox_head = np.array([[96, 374, 143, 413]])
    cv_img = cv2.imread(img_path)
    s = cv_img.shape
    print(cv_img.shape)
    labels = np.array([1])
    trans = CVTransform(cv_img, bbox_head, labels)
    # img, bbox = trans.randomFlip(cv_img, bbox_head)
    print(trans.img.shape)
    # print(trans.bboxes)
    cv2.rectangle(trans.img, (trans.bboxes[:, 1], trans.bboxes[:, 0]), (trans.bboxes[:, 3], trans.bboxes[:, 2]),
                  (55, 255, 155), 5)

    cv2.imshow('image', trans.img)
    cv2.waitKey(50000)
