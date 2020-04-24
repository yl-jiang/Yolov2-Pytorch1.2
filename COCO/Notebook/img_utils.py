#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/30 下午3:04
# @Author  : jyl
# @File    : img_utils.py
import numpy as np
import cv2


def images_db(file_path):
    f = open(file_path, 'r')
    file_names = list()
    bboxes = list()
    labels = list()
    for line in f.readlines():
        splits = line.strip().split()
        file_names.append(splits[0])
        num_obj = int(len(splits[1:]) / 5)
        bbox = []
        label = []
        for i in range(num_obj):
            bbox.append([int(splits[5*i+1]), int(splits[5*i+2]), int(splits[5*i+3]), int(splits[5*i+4])])
            label.append(int(splits[5*i+5]))
        bboxes.append(bbox)
        labels.append(label)
    return file_names, np.array(bboxes), np.array(labels)


def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

