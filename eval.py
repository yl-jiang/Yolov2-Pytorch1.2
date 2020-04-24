#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/16 下午2:19
# @Author  : jyl
# @File    : eval.py
import sys
from pathlib import Path
pwd = Path('./').absolute()
print(f'add directory: [{pwd}] in python path')
sys.path.append(pwd)
sys.path.append('/home/dk/anaconda3/envs/fun/lib/python3.7/site-packages')
import torch
from config import Config
from data import TestDataset
from trainer import YOLOV2VOCTrainer
import numpy as np
from tqdm import tqdm
from utils import gpu_nms, gpu_nms_mutil_class
from metric import mAP
from utils import xyxy2xywh
from utils import inverse_letter_resize
from utils import cv2_savefig
import pickle


class YOLOV2:

    def __init__(self):
        self.opt_test = Config(is_train=False)
        self.testDataset = TestDataset(self.opt_test)
        self.trainer = YOLOV2VOCTrainer(self.opt_test)
        _ = self.trainer.load(model_dir=self.opt_test.model_every,
                              load_optimizer=False,
                              lr_scheduler=None)

    def eval(self):
        with tqdm(total=len(self.testDataset), ncols=80) as t:
            for img_id in range(len(self.testDataset)):
                img_org, img_norm, labels_org, boxes_org, ratio, dh, dw = self.testDataset[img_id]
                # [3, 416, 416] -> [1, 3, 416, 416]
                img_norm = img_norm.unsqueeze(0)
                img_in = img_norm.to(self.opt_test.device)
                boxes, confs, probs = self.trainer.predict(img_in)

                boxes = boxes.squeeze(0)  # [13*13*5, 4] / Tensor
                confs = confs.squeeze(0)  # [13*13*5, 1] / Tensor
                probs = probs.squeeze(0)  # [13*13*5, 20] / Tensor
                cls_scores = (confs * probs).squeeze(0)  # [13*13*5, 20] / Tensor
                cls_scores, labels = cls_scores.max(dim=-1, keepdim=True)  # [13*13*5, 1] / Tensor
                labels = labels.squeeze()

                # box_out: [xmin, ymin, xmax, ymax]
                score_thresh = self.opt_test.nms_score_thresh
                iou_thresh = self.opt_test.nms_iou_thresh
                max_box_num = self.opt_test.nms_max_box_num
                keep_index = gpu_nms(boxes, cls_scores, score_thresh, iou_thresh, max_box_num)
                cls_scores = cls_scores.squeeze()
                keep_boxes = boxes[keep_index].detach().cpu().numpy()
                keep_scores = cls_scores[keep_index].detach().cpu().numpy()
                keep_labels = labels[keep_index].detach().cpu().numpy()
                keep_boxes = inverse_letter_resize(keep_boxes, ratio, dh, dw)

                keep_index = self.filter_small_box(keep_boxes, 16)
                keep_boxes = keep_boxes[keep_index]
                keep_labels = keep_labels[keep_index]
                keep_scores = keep_scores[keep_index]

                imgname = Path('/home/dk/Desktop/Yolov2/v3loss/nms_all_class') / f'{img_id}.png'
                cv2_savefig(img_org, keep_boxes, keep_labels, keep_scores, imgname)

                t.update(1)

    @staticmethod
    def filter_small_box(boxes, min_size):
        boxes_xywh = xyxy2xywh(boxes)
        boxes_ws = boxes_xywh[:, 2]
        boxes_hs = boxes_xywh[:, 3]
        boxes_areas = boxes_ws * boxes_hs
        keep_index = np.where(boxes_areas > min_size)[0]
        return keep_index

    def eval_mutil_class_nms(self):
        with tqdm(total=len(self.testDataset), ncols=80) as t:
            for img_id in range(len(self.testDataset)):
                img_org, img_norm, labels_org, boxes_org, ratio, dh, dw = self.testDataset[img_id]
                img_norm = img_norm.unsqueeze(0)
                img = img_norm.to(self.opt_test.device)
                boxes, confs, probs = self.trainer.predict(img)

                boxes = boxes.squeeze(0)  # [13*13*5, 4] / Tensor
                confs = confs.squeeze(0)  # [13*13*5, 1] / Tensor
                probs = probs.squeeze(0)  # [13*13*5, 20] / Tensor
                cls_scores = (confs * probs).squeeze(0)  # [13*13*5, 20] / Tensor

                # box_out: [xmin, ymin, xmax, ymax]
                score_thresh = self.opt_test.nms_score_thresh
                iou_thresh = self.opt_test.nms_iou_thresh
                max_box_num = self.opt_test.nms_max_box_num
                min_box_area = self.opt_test.nms_min_box_area

                keep_index_list = gpu_nms_mutil_class(boxes, cls_scores, score_thresh, iou_thresh, max_box_num, min_box_area)
                assert len(keep_index_list) == self.opt_test.voc_class_num
                keep_boxes, keep_labels, keep_scores = [], [], []
                for k in range(self.opt_test.voc_class_num):
                    if len(keep_index_list[k]) == 0:
                        continue
                    for ind in keep_index_list[k]:
                        keep_boxes.append(boxes[ind].detach().cpu().numpy())
                        keep_labels.append(k)
                        keep_scores.append(cls_scores[ind, k].detach().cpu().item())
                if len(keep_boxes) != 0:
                    keep_boxes = inverse_letter_resize(np.asarray(keep_boxes), ratio, dh, dw)

                imgname = Path('/home/dk/Desktop/Yolov2/v3loss/nms_each_class') / f'{img_id}.png'
                cv2_savefig(img_org, keep_boxes, np.asarray(keep_labels), np.asarray(keep_scores), imgname)

                t.update(1)

    def compute_mAP_nms_all_class(self):
        gt_boxes = []
        pred_boxes = []
        with tqdm(total=len(self.testDataset), ncols=80) as t:
            for img_id in range(len(self.testDataset)):
                img_org, img_norm, labels_org, boxes_org, ratio, dh, dw = self.testDataset[img_id]
                # [3, 416, 416] -> [1, 3, 416, 416]
                img_norm = img_norm.unsqueeze(0)
                img_in = img_norm.to(self.opt_test.device)
                boxes, confs, probs = self.trainer.predict(img_in)

                boxes = boxes.squeeze(0)  # [13*13*5, 4] / Tensor
                confs = confs.squeeze(0)  # [13*13*5, 1] / Tensor
                probs = probs.squeeze(0)  # [13*13*5, 20] / Tensor
                cls_scores = (confs * probs).squeeze(0)  # [13*13*5, 20] / Tensor
                cls_scores, labels = cls_scores.max(dim=-1, keepdim=True)  # [13*13*5, 1] / Tensor
                labels = labels.squeeze()

                # box_out: [xmin, ymin, xmax, ymax]
                score_thresh = self.opt_test.nms_score_thresh
                iou_thresh = self.opt_test.nms_iou_thresh
                max_box_num = self.opt_test.nms_max_box_num
                keep_index = gpu_nms(boxes, cls_scores, score_thresh, iou_thresh, max_box_num)
                cls_scores = cls_scores.squeeze()
                keep_boxes = boxes[keep_index].detach().cpu().numpy()
                keep_scores = cls_scores[keep_index].detach().cpu().numpy()
                keep_labels = labels[keep_index].detach().cpu().numpy()
                keep_boxes = inverse_letter_resize(keep_boxes, ratio, dh, dw)

                keep_index = self.filter_small_box(keep_boxes, 16)
                keep_boxes = keep_boxes[keep_index]
                keep_labels = keep_labels[keep_index]
                keep_scores = keep_scores[keep_index]

                if len(keep_boxes) > 0:
                    pred_map_boxe = np.hstack([keep_boxes, np.expand_dims(keep_scores, axis=-1)])
                else:
                    pred_map_boxe = np.zeros(shape=[1, 5])
                gt_boxes.append(boxes_org)
                pred_boxes.append(pred_map_boxe)
                t.update(1)
        pickle.dump({'gt_boxes': gt_boxes, 'pred_boxes': pred_boxes},
                    open('/home/dk/ML/V2/VOC/result/yolov3_loss_nms_all_class_mAP.pkl', 'wb'))
        map = mAP(predict=pred_boxes, ground_truth=gt_boxes, iou_threshold=0.5)
        print(map.elevenPointAP * 100)

    def compute_mAP_nms_each_class(self):
        gt_boxes = []
        pred_boxes = []
        with tqdm(total=len(self.testDataset), ncols=80) as t:
            for img_id in range(len(self.testDataset)):
                img_org, img_norm, labels_org, boxes_org, ratio, dh, dw = self.testDataset[img_id]
                img_norm = img_norm.unsqueeze(0)
                img = img_norm.to(self.opt_test.device)
                boxes, confs, probs = self.trainer.predict(img)

                boxes = boxes.squeeze(0)  # [13*13*5, 4] / Tensor
                confs = confs.squeeze(0)  # [13*13*5, 1] / Tensor
                probs = probs.squeeze(0)  # [13*13*5, 20] / Tensor
                cls_scores = (confs * probs).squeeze(0)  # [13*13*5, 20] / Tensor

                # box_out: [xmin, ymin, xmax, ymax]
                score_thresh = self.opt_test.nms_score_thresh
                iou_thresh = self.opt_test.nms_iou_thresh
                max_box_num = self.opt_test.nms_max_box_num
                min_box_area = self.opt_test.nms_min_box_area

                keep_index_list = gpu_nms_mutil_class(boxes, cls_scores, score_thresh, iou_thresh, max_box_num, min_box_area)
                assert len(keep_index_list) == self.opt_test.voc_class_num
                keep_boxes, keep_labels, keep_scores = [], [], []
                for k in range(self.opt_test.voc_class_num):
                    if len(keep_index_list[k]) == 0:
                        continue
                    for ind in keep_index_list[k]:
                        keep_boxes.append(boxes[ind].detach().cpu().numpy())
                        keep_labels.append(k)
                        keep_scores.append(cls_scores[ind, k].detach().cpu().item())

                if len(keep_boxes) != 0:
                    keep_boxes = inverse_letter_resize(np.asarray(keep_boxes), ratio, dh, dw)

                if len(keep_boxes) > 0:
                    pred_map_boxe = np.hstack([keep_boxes, np.expand_dims(keep_scores, axis=-1)])
                else:
                    pred_map_boxe = np.zeros(shape=[1, 5])
                gt_boxes.append(boxes_org)
                pred_boxes.append(pred_map_boxe)
                t.update(1)

        pickle.dump({'gt_boxes': gt_boxes, 'pred_boxes': pred_boxes},
                    open('/home/dk/ML/V2/VOC/result/yolov3_loss_nms_each_class_mAP.pkl', 'wb'))
        map = mAP(predict=pred_boxes, ground_truth=gt_boxes, iou_threshold=0.5)
        print(map.elevenPointAP * 100)


if __name__ == '__main__':
    model = YOLOV2()
    # model.eval_mutil_class_nms()
    model.compute_mAP_nms_each_class()


