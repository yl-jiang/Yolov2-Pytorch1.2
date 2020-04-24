#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/16 下午5:04
# @Author  : jyl
# @File    : plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
import cv2
from PIL import Image
np.random.seed(123)

VOC_BBOX_LABEL_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'bg']
VOC_BBOX_LABEL_DICT = {index: name for index, name in enumerate(VOC_BBOX_LABEL_NAMES)}


def matplot(img, bboxes, labels, scores):
    """
    :param img: rgb
    :param bboxes: [xmin, ymin, xmax, ymax] / ndarray
    :param labels:
    :param scores:
    :return:
    """
    assert isinstance(img, np.ndarray)
    assert isinstance(bboxes, np.ndarray)
    assert len(bboxes) == len(labels) and len(bboxes) == len(scores)

    colors = random_color(len(VOC_BBOX_LABEL_NAMES))
    fig, ax = plt.subplots(figsize=[16, 16])
    ax.imshow(img)
    ax.xaxis.set_ticks_position('top')
    for box, label, score in zip(bboxes, labels, scores):
        width = box[2] - box[0]
        height = box[3] - box[1]
        xmin = box[0]
        ymax = box[3]
        color = colors[label]
        style = "dashed"
        alpha = 1
        p = patches.Rectangle((xmin, ymax), width, height, linewidth=2,
                              alpha=alpha, linestyle=style,
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)
        caption = f'{VOC_BBOX_LABEL_NAMES[label]}:{score:.3f}'
        ax.text(xmin, ymax, caption, size=15, verticalalignment='top',
                color='w', backgroundcolor="none",
                bbox={'facecolor': color, 'alpha': 0.5, 'pad': 2, 'edgecolor': 'none'})

    plt.show()
    return fig


def random_color(n):
    colors = {}
    for i in range(n):
        colors[i] = [np.random.rand() for _ in range(3)]
    return colors


def cv2plot(*args):
    if len(args) == 5:
        img, boxes, resized_img, resized_boxes, labels = args
    else:
        img, boxes, labels = args

    assert isinstance(boxes, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert boxes.ndim == 2
    assert (boxes[:, [2, 3]] >= boxes[:, [0, 1]]).all(), 'format of box should be [xmin, ymin, xmax, ymax]'
    assert len(boxes) == len(labels)

    boxes = np.floor(boxes).astype(np.int32)
    names = list()
    for label in labels:
        names.append(VOC_BBOX_LABEL_DICT[label])
    for name, bbox in zip(names, boxes):
        img = cv2.rectangle(img=img,
                            pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]),
                            color=(0, 255, 0), thickness=1)
        img = cv2.putText(img=img,
                          text=f'{name}',
                          org=(bbox[0], bbox[1]),
                          fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                          fontScale=0.3, color=(0, 125, 255))
    if len(args) == 5:
        for name, bbox in zip(names, resized_boxes.astype(np.int32)):
            resized_img = cv2.rectangle(img=resized_img,
                                        pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]),
                                        color=(0, 255, 0), thickness=2)
            resized_img = cv2.putText(img=resized_img,
                                      text=f'{name}',
                                      org=(bbox[0], bbox[1]),
                                      fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                      fontScale=0.6, color=(0, 125, 255))

        plt.figure(figsize=[16, 10])
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(resized_img)
        plt.show()
    else:
        plt.figure(figsize=[16, 10])
        plt.imshow(img)
        plt.show()


def cv2_savefig(img, boxes, labels, scores, filename):
    """
    :param img: RGB / ndarray
    :param boxes: [xmin, ymin, xmax, ymax] / ndarray / dimension number is 2
    :param labels: list / ndarray
    :param filename: string
    :param scores:
    :return:
    """
    if len(boxes) != 0:
        assert isinstance(img, np.ndarray)
        assert isinstance(boxes, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert img.ndim == 3
        assert scores.ndim == 1
        assert labels.ndim == 1
        assert boxes.ndim == 2
        assert (boxes[:, [2, 3]] >= boxes[:, [0, 1]]).all(), 'format of box should be [xmin, ymin, xmax, ymax]'
        assert len(boxes) == len(labels)
        assert len(labels) == len(scores)

        img_h, img_w, img_c = img.shape
        # 当xmin/ymin值为0, 或xmax为img_w, 或ymax为img_h时，在图上显示不出box的边界
        # boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w).astype(np.int32)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 1, img_w - 1).astype(np.int32)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 1, img_h - 1)
        names = [VOC_BBOX_LABEL_DICT[label] for label in labels]
        for name, bbox, score in zip(names, boxes, scores):
            img = cv2.rectangle(img=img,
                                pt1=(bbox[0], bbox[1]),
                                pt2=(bbox[2], bbox[3]),
                                color=(0, 255, 0),
                                thickness=1)

            caption = f'{name}:{score*100:.2f}%'
            img = cv2.putText(img=img,
                              text=caption,
                              org=(bbox[0], bbox[1]),
                              fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                              fontScale=0.4,
                              color=(218, 165, 32))

    pil_img = Image.fromarray(img)
    pil_img.save(filename)
    # # plt.figure(figsize=[16, 16])
    # plt.imshow(img)
    # plt.axis('off')
    # plt.savefig(figname)
    # plt.close('all')


if __name__ == '__main__':
    random_color(21)

