#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/22 下午3:53
# @Author  : jyl
# @File    : prior_anchors.py
from config import opt
import numpy as np
import matplotlib.pyplot as plt
import random
from utils import alias_sample
from tqdm import tqdm
import os
from bs4 import BeautifulSoup
import pickle


VOC_BBOX_LABEL_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def parse_voc2012_xml(xml_file):
    bboxes = []
    labels = []
    obj_names = []
    bs = BeautifulSoup(open(xml_file), features='lxml')
    img_file_name = bs.find('filename').string

    size_obj = bs.find('size')
    width = int(float(size_obj.find('width').string))
    height = int(float(size_obj.find('height').string))

    for obj in bs.find_all('object'):
        diffcult = int(obj.find('difficult').string)
        if diffcult == 1:
            continue
        name = obj.find('name').string
        obj_names.append(name)
        if name in VOC_BBOX_LABEL_NAMES:
            label = VOC_BBOX_LABEL_NAMES.index(name)
            bndbox_obj = obj.find('bndbox', recursive=False)
            y1 = int(float(bndbox_obj.find('ymax').string))
            x1 = int(float(bndbox_obj.find('xmax').string))
            y2 = int(float(bndbox_obj.find('ymin').string))
            x2 = int(float(bndbox_obj.find('xmin').string))
            bboxes.append([y1, x1, y2, x2])
            labels.append(label)

    return img_file_name, bboxes, labels, obj_names, width, height


def parse_voc(train_path):
    AnnotationsPath = train_path
    xml_fils = os.listdir(AnnotationsPath)
    data_list = []
    for f in tqdm(xml_fils):
        tmp_dict = {}
        xml_path = os.path.join(AnnotationsPath, f)
        img_file_name, bboxes, labels, obj_names, width, height = parse_voc2012_xml(xml_path)
        if len(labels) == 0:
            # print(img_file_name)
            continue
        tmp_dict['file_name'] = img_file_name
        tmp_dict['obj'] = {'bbox': bboxes, 'label': labels, 'name': obj_names}
        tmp_dict['width'] = width
        tmp_dict['height'] = height

        data_list.append(tmp_dict)
    return data_list


def iou(center_box, other_boxes):
    intersection_box = np.where(center_box < other_boxes, center_box, other_boxes)
    intersection_area = np.prod(intersection_box, axis=1)
    center_box_area = np.prod(center_box)
    otherbox_areas = np.prod(other_boxes, axis=1)
    ious = intersection_area / (center_box_area + otherbox_areas - intersection_area)
    return ious


def classification(k, bboxes, use_alias=True):
    """
    :param k: 簇个数
    :param bboxes: 聚类输入数据
    :param use_alias: 为True表示使用alias method进行聚类中心的选择，为False表示使用numpy的choice方法选择中心点
    :return:
    """
    length = len(bboxes)
    center_index = get_centers(k, bboxes, use_alias)
    center_coord = bboxes[center_index]
    center_tmp = np.zeros_like(center_coord)
    ori_dis = np.full(shape=length, fill_value=np.inf)
    class_list = np.zeros(shape=length) - 1

    times = 1
    while np.sum(np.square(center_coord - center_tmp)) > 1e-7:
        times += 1
        center_tmp = center_coord.copy()
        for i in range(k):
            new_dis = 1 - iou(center_coord[i], bboxes)
            class_list = np.where(ori_dis < new_dis, class_list, i)
            ori_dis = np.where(ori_dis < new_dis, ori_dis, new_dis)
        # update center
        for i in range(k):
            center_coord[i] = np.mean(bboxes[class_list == i], axis=0)

    return class_list, center_coord


def show_result(raw_data, center_coordinate, class_list, mean_iou):
    print('Showing... ...')
    colors = [
        '#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#228B22',
        '#0000FF', '#FF1493', '#EE82EE', '#000000', '#FFA500',
        '#00FF00', '#006400', '#00FFFF', '#0000FF', '#FFFACD',
    ]

    use_color = []
    for node in class_list:
        use_color.append(colors[int(node)])

    plt.figure(num=1, figsize=(16, 9))
    plt.scatter(x=raw_data[:, 0], y=raw_data[:, 1], c=use_color, s=50, marker='o', alpha=0.3)
    plt.scatter(x=center_coordinate[:, 0], y=center_coordinate[:, 1], c='b', s=200, marker='+', alpha=0.8)
    plt.title('Mean IOU: %.3f' % mean_iou)
    plt.show()


def get_centers(k, bboxes, use_alias):
    if use_alias:
        centers = [random.randint(a=0, b=len(bboxes))]
        tmp_dis = np.full(shape=len(bboxes), fill_value=np.inf)
        while len(centers) < k:
            for i, center in enumerate(centers):
                dis = 1 - iou(center, bboxes)
                dis = np.where(dis < tmp_dis, dis, tmp_dis)
            probs = dis / np.sum(dis)
            # centers.append(np.random.choice(a=len(bboxes), size=1, p=probs)[0])
            centers.append(alias_sample(probs, 1)[0])
        return centers
    else:
        return np.random.choice(a=np.arange(len(bboxes)), size=k)


def normalize(data_list):
    cluster_x = []
    cluster_y = []

    for img in data_list:
        img_width = img['width']
        img_height = img['height']
        # box: [ymax, xmax, ymin, xmin]
        for box in img['obj']['bbox']:
            box_width = box[1] - box[-1]
            box_height = box[0] - box[2]
            cluster_x.append(box_width / img_width)
            cluster_y.append(box_height / img_height)
    cluster_x = np.array(cluster_x).reshape(-1, 1)
    cluster_y = np.array(cluster_y).reshape(-1, 1)
    bboxes = np.hstack([cluster_x, cluster_y])
    return bboxes


def kmeans(raw_data, k, use_alias):
    class_list, center_coordinate = classification(k, raw_data, use_alias)
    return class_list, center_coordinate


def mean_iou(bboxes, class_list, center_coordinate):
    ious = []
    for label, center in enumerate(center_coordinate):
        ious.append(iou(center, bboxes[class_list == label]))
    every_class_mean_iou = []
    for u in ious:
        every_class_mean_iou.append(np.mean(u))
    return np.mean(every_class_mean_iou)


def cluster_anchors(voc_path):
    if not os.path.exists('./bboxes.pkl'):
        voc_train_path = os.path.join(voc_path, 'VOC2012train', 'VOCdevkit', 'VOC2012', 'Annotations')
        data_list = parse_voc(voc_train_path)
        bboxes = normalize(data_list)
        pickle.dump(bboxes, open('./bboxes.pkl', 'wb'))
    else:
        bboxes = pickle.load(open('./bboxes.pkl', 'rb'))
    class_list, center_coordinate = kmeans(bboxes, opt.anchor_num, True)
    avg_iou = mean_iou(bboxes, class_list, center_coordinate)
    show_result(bboxes, center_coordinate, class_list, avg_iou)
    with open('./anchors.txt', 'a') as f:
        for wh in center_coordinate[:]:
            f.write(str(wh[0] * opt.img_w) + ',' + str(wh[1] * opt.img_h) + '\n')
    return center_coordinate, class_list


if __name__ == '__main__':
    voc_dir = '/media/dk/MyFiles/Data/VOC/VOC2012train/VOCdevkit/VOC2012/Annotations/'
    cluster_anchors(voc_dir)
