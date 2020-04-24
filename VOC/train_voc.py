#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/25 下午3:33
# @Author  : jyl
# @File    : yolov2.py
import sys
from pathlib import Path
pwd = Path('./').absolute()
sys.path.append(pwd)
import torch
from config import Config
from data import VOC2007Dataset
from trainer import YOLOV2VOCTrainer
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import gpu_nms, gpu_nms_mutil_class
from metric import mAP
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from utils import xyxy2xywh
from utils import letterbox_resize, inverse_letter_resize
from utils import cv2_savefig


class YOLOV2:

    def __init__(self):
        self.opt_train = Config(is_train=True)
        self.opt_test = Config(is_train=False)
        self.testDataset = VOC2007Dataset(self.opt_test)
        self.trainDataset = VOC2007Dataset(self.opt_train)
        self.VOCTrainDataLoader = DataLoader(self.trainDataset, batch_size=self.opt_train.batch_size,
                                             shuffle=True, num_workers=self.opt_test.num_workers, drop_last=True,
                                             pin_memory=self.opt_train.pin_memory)
        self.VOCTestDataLoader = DataLoader(self.testDataset, batch_size=self.opt_test.batch_size,
                                            shuffle=False, num_workers=self.opt_test.num_workers, drop_last=True,
                                            pin_memory=self.opt_train.pin_memory)
        self.data_length = self.opt_train.batch_size * (len(self.trainDataset) // self.opt_train.batch_size)
        self.trainer = YOLOV2VOCTrainer(self.opt_train)

        self.normailze = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(self.opt_train.mean, self.opt_train.std)])
        self.lr_scheduler = MultiStepLR(optimizer=self.trainer.optimizer, milestones=[80, 200, 280], gamma=0.1)

    def train(self):
        last_loss, last_epoch, last_steps = self.trainer.load(self.opt_train.model_every)
        print(f'last info: loss={last_loss} steps={last_steps} epoch={last_epoch}')
        for epoch in range(last_epoch, self.opt_train.epoch_num):
            with tqdm(total=self.data_length, ncols=120) as t:
                t.set_description(f'epoch {epoch}/{self.opt_train.epoch_num}')
                for i, (imgs, targets) in enumerate(self.VOCTrainDataLoader):
                    imgs = imgs.to(self.opt_train.device)
                    tragets = targets.to(self.opt_train.device)
                    last_steps += 1
                    self.trainer.train_step(imgs, tragets, last_steps)
                    loss = self.trainer.get_loss_meters()
                    if last_steps % self.opt_train.display_every == 0:
                        t.set_postfix_str(f'loss= {loss["total_loss"]:.5f} lr= {self.lr_scheduler.get_lr()[0]:.2e}')
                    if last_steps % self.opt_train.eval_every == 0:
                        self.eval(epoch, i + 1)
                    if last_loss > loss['total_loss']:
                        save_path = self.opt_train.model_save_dir / 'model_yolov2_loss_best.pth'
                        self.trainer.save(loss['total_loss'], self.lr_scheduler, epoch, last_steps, save_path)
                    if last_steps % self.opt_train.save_every == 0:
                        save_path = self.opt_train.model_save_dir / 'model_yolov2_loss_every.pth'
                        self.trainer.save(loss['total_loss'], self.lr_scheduler, epoch, last_steps, save_path)
                    t.update(imgs.size(0))
            self.lr_scheduler.step(epoch=epoch)

    def eval(self, epoch, step):
        test_imgs = np.random.randint(low=0, high=len(self.testDataset), size=7)
        for img_id in test_imgs:
            _, _, img_org, labels, boxes = self.testDataset.get_example_for_testing(img_id)
            img_resized, _, ratio, dh, dw = letterbox_resize(img_org, boxes, [416, 416])
            img = self.normailze(img_resized)
            # [3, 416, 416] -> [1, 3, 416, 416]
            img = torch.unsqueeze(img, dim=0)
            img = img.to(self.opt_train.device)
            boxes, confs, probs = self.trainer.predict(img)

            boxes = boxes.squeeze(0)  # [13*13*5, 4] / Tensor
            confs = confs.squeeze(0)  # [13*13*5, 1] / Tensor
            probs = probs.squeeze(0)  # [13*13*5, 20] / Tensor
            cls_scores = (confs * probs).squeeze(0)  # [13*13*5, 20] / Tensor
            cls_scores, labels = cls_scores.max(dim=-1, keepdim=True)  # [13*13*5, 1] / Tensor
            labels = labels.squeeze()

            # box_out: [xmin, ymin, xmax, ymax]
            score_thresh = self.opt_train.nms_score_thresh
            iou_thresh = self.opt_train.nms_iou_thresh
            max_box_num = self.opt_train.nms_max_box_num
            keep_index = gpu_nms(boxes, cls_scores, score_thresh, iou_thresh, max_box_num)
            cls_scores = cls_scores.squeeze()
            keep_boxes = boxes[keep_index].detach().cpu().numpy()
            keep_scores = cls_scores[keep_index].detach().cpu().numpy()
            keep_labels = labels[keep_index].detach().cpu().numpy()
            keep_boxes = inverse_letter_resize(keep_boxes, ratio, dh, dw)

            keep_index = self.fliter_small_box(keep_boxes, 16)
            keep_boxes = keep_boxes[keep_index]
            keep_labels = keep_labels[keep_index]
            keep_scores = keep_scores[keep_index]

            imgname = self.opt_train.result_img_dir / f'E{epoch}_S{step}_img{img_id}.png'
            cv2_savefig(img_org, keep_boxes, keep_labels, keep_scores, imgname)

    @staticmethod
    def fliter_small_box(boxes, min_size):
        boxes_xywh = xyxy2xywh(boxes)
        boxes_ws = boxes_xywh[:, 2]
        boxes_hs = boxes_xywh[:, 3]
        boxes_areas = boxes_ws * boxes_hs
        keep_index = np.where(boxes_areas > min_size)[0]
        return keep_index

    def eval_mutil_class_nms(self, epoch, step):
        # last_loss, last_epoch, last_steps = self.trainer.load(self.opt_train.model_every, lr_scheduler=self.lr_scheduler)
        # print(f'last info: loss={last_loss} steps={last_steps} epoch={last_epoch}')
        random_img_ids = np.random.randint(0, len(self.VOCTestDataLoader), 10)
        for img_id in random_img_ids:
            _, _, img_org, labels, boxes = self.testDataset.get_example_for_testing(img_id)
            img_resized, _, ratio, dh, dw = letterbox_resize(img_org, boxes, [416, 416])
            img = self.normailze(img_resized)
            # [3, 416, 416] -> [1, 3, 416, 416]
            img = torch.unsqueeze(img, dim=0)
            img = img.to(self.opt_train.device)
            boxes, confs, probs = self.trainer.predict(img)

            boxes = boxes.squeeze(0)  # [13*13*5, 4] / Tensor
            confs = confs.squeeze(0)  # [13*13*5, 1] / Tensor
            probs = probs.squeeze(0)  # [13*13*5, 20] / Tensor
            cls_scores = (confs * probs).squeeze(0)  # [13*13*5, 20] / Tensor

            # box_out: [xmin, ymin, xmax, ymax]
            score_thresh = self.opt_train.nms_score_thresh
            iou_thresh = self.opt_train.nms_iou_thresh
            max_box_num = self.opt_train.nms_max_box_num
            min_box_area = self.opt_train.nms_min_box_area
            keep_index_list = gpu_nms_mutil_class(boxes, cls_scores, score_thresh, iou_thresh, max_box_num, min_box_area)
            assert len(keep_index_list) == self.opt_train.voc_class_num
            keep_boxes, keep_labels, keep_scores = [], [], []
            for k in range(self.opt_train.voc_class_num):
                if len(keep_index_list[k]) == 0:
                    continue
                for ind in keep_index_list[k]:
                    keep_boxes.append(boxes[ind].detach().cpu().numpy())
                    keep_labels.append(k)
                    keep_scores.append(cls_scores[ind, k].detach().cpu().item())
            if len(keep_boxes) != 0:
                keep_boxes = inverse_letter_resize(np.asarray(keep_boxes), ratio, dh, dw)

            imgname = self.opt_train.result_img_dir / f'E{epoch}_S{step}_img{img_id}.png'
            cv2_savefig(img_org, keep_boxes, np.asarray(keep_labels), np.asarray(keep_scores), imgname)


if __name__ == '__main__':
    model = YOLOV2()
    model.train()
