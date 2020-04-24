#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/6 上午10:13
# @Author  : jyl
# @File    : train_coco_v2.py
import torch
from config import opt
from data import COCODataset
from model import YOLOV2COCOTrainer
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import gpu_nms
from metric import mAP
from utils import plot_one
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model import load_weights
import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR


class Yolov2:

    def __init__(self):
        self.VocTrainDataLoader = DataLoader(COCODataset(is_train=True), batch_size=opt.batch_size,
                                             shuffle=True, num_workers=opt.num_workers, drop_last=True, pin_memory=True)
        self.VocTestDataLoader = DataLoader(COCODataset(is_train=False), batch_size=opt.batch_size,
                                            shuffle=False, num_workers=opt.num_workers, drop_last=True, pin_memory=True)

        self.train_data_length = len(COCODataset(is_train=True))
        self.val_data_length = len(COCODataset(is_train=False))
        self.trainer = YOLOV2COCOTrainer()
        self.testDataset = COCODataset(is_train=False)
        # self.test_imgs = np.random.randint(low=0, high=len(self.testDataset), size=35)
        self.normailze = transforms.Compose([transforms.ToTensor(), transforms.Normalize(opt.mean, opt.std)])
        self.logger = opt.logger
        self.writer = SummaryWriter(log_dir=os.path.join(opt.base_path, 'log', f'summary_{opt.optimizer_type}'))
        self.scheduler = CosineAnnealingLR(optimizer=self.trainer.optimizer, T_max=5)

    def train(self):
        # self.writer.add_graph(self.trainer.yolov2.cpu(),
        # input_to_model=torch.rand(opt.batch_size, 3, opt.img_h, opt.img_w, device='cpu'))
        self.trainer.yolov2.train()
        loss_tmp = float('inf')
        if os.path.exists(opt.saved_model_path):
            self.logger.info(f'Use pretrained model: {opt.saved_model_path}')
            self.trainer.use_pretrain(opt.saved_model_path)
            loss_tmp = self.trainer.last_loss
            start_epoch = self.trainer.epoch_num
            total_steps = self.trainer.total_steps
        else:
            start_epoch = 0
            total_steps = 0
            self.logger.info('Train from stratch ...')
            pretrained_ckpt_path = '/home/dk/jyl/V2/COCO/model/ckpt/only_params_trained_yolo_coco'
            load_weights(self.trainer.yolov2, pretrained_ckpt_path, reinit_last=False)

        for epoch in range(start_epoch, opt.epochs):
            for i, (imgs, targets) in tqdm(enumerate(self.VocTrainDataLoader)):
                imgs, tragets = imgs.to(opt.device), targets.to(opt.device)
                self.trainer.train_step(imgs, tragets, epoch + 1)
                total_steps += 1
                mean_loss = self.trainer.loss_meter.mean
                loss_dict = self.trainer.loss_dict
                self.add_train_summary(loss_dict, total_steps)
                if total_steps % opt.display_step == 0:
                    message = f'Epoch[{epoch: 03}] Step[{(epoch * self.train_data_length + i + 1): 06}]] \n' \
                              f'mean_loss : {mean_loss:.3f} \n' \
                              f'xy_loss   : {loss_dict["dxdy_loss"]:.3f} \n' \
                              f'wh_loss   : {loss_dict["twth_loss"]:.3f} \n' \
                              f'conf_loss : {loss_dict["conf_loss"]:.3f} \n' \
                              f'class_loss: {loss_dict["class_loss"]:.3f} \n' \
                              f'obj_loss  : {loss_dict["obj_loss"]:.3f} \n' \
                              f'noobj_loss: {loss_dict["noobj_loss"]:.3f} \n' \
                              f'total_loss: {loss_dict["total_loss"]:.3f} \n' \
                              f'current learning rate: {self.scheduler.get_lr()}'
                    self.logger.info(message)
                if total_steps % opt.eval_step == 0:
                    self.eval(epoch, i + 1)
                if mean_loss < loss_tmp:
                    loss_tmp = mean_loss
                    self.trainer.save(epoch, total_steps, mean_loss, opt.model_save_dir + f'/model_best_{opt.optimizer_type}_2.pkl')
                if total_steps % opt.save_step == 0:
                    self.trainer.save(epoch, total_steps, mean_loss, opt.model_save_dir + f'/model_every_2.pkl')
            self.scheduler.step()

    def eval(self, epoch, step):
        self.trainer.yolov2.eval()
        conf_list = []
        cls_list = []
        score_list = []
        steps = 0
        self.test_imgs = np.random.randint(0, self.val_data_length, 10)
        with torch.no_grad():
            for img_id in self.test_imgs:
                steps += 1
                ori_img, resized_img, true_box, true_label = self.testDataset[img_id]
                _, ratio, dh, dw = self.letter_resize(ori_img, [416, 416])
                img = self.normailze(resized_img)
                # [3, 416, 416] -> [1, 3, 416, 416]
                img = torch.unsqueeze(img, dim=0)
                img = img.to(opt.device)
                preds = self.trainer.yolov2(img)
                boxes, confs, probs = self.predict(preds)

                boxes = boxes.squeeze(0)  # [13*13*5, 4]
                confs = confs.squeeze(0)  # [13*13*5, 1]
                probs = probs.squeeze(0)  # [13*13*5, 80]
                scores = confs * probs  # [13*13*5, 80]
                conf_list.extend(confs.detach().cpu().numpy())
                cls_list.extend(probs.detach().cpu().numpy())
                score_list.extend(scores.detach().cpu().numpy())
                pred_dict = {'conf': conf_list[-1].flatten(), 'cls': cls_list[-1].flatten(), 'score': score_list[-1].flatten()}
                self.add_test_summary(pred_dict, steps)
                # box_out: [xmin, ymin, xmax, ymax]
                box_out, score_out, label_out = self.nms_each_class(boxes, scores, opt.score_threshold, opt.iou_threshold, opt.max_boxes_num)

                if len(box_out) != 0:
                    plot_dict = {'img': ori_img,
                                 'ratio': ratio, 
                                 'dh': dh, 
                                 'dw': dw, 
                                 'pred_box': box_out.cpu().numpy(),
                                 'pred_score': score_out.cpu().numpy(),
                                 'pred_label': label_out.cpu().numpy(),
                                 'gt_box': None,
                                 'gt_label': None,
                                 'img_name': f'epoch_{epoch}_step{step}_{img_id}.jpg',
                                 'save_path': os.path.join(opt.base_path, 'data', 'result')}

                    plot_one(plot_dict)

        msg = f"\n" \
              f"Score Mean: {np.mean(score_list):.5f} \n" \
              f"Score Max: {np.max(score_list):.5f} \n" \
              f"Score Min: {np.min(score_list):.5f} \n" \
              f"Max conf: {np.max(conf_list)} \n" \
              f"Max cls: {np.max(cls_list)}"
        self.logger.info(msg)

        self.trainer.yolov2.train()

    def test(self):
        self.trainer.yolov2.eval()
        mAP_predicts = []
        mAP_ground_truths = []
        torch_normailze = transforms.Compose([transforms.ToTensor(), transforms.Normalize(opt.mean, opt.std)])
        with torch.no_grad():
            for i, (imgs, true_boxes, true_labels) in tqdm(enumerate(self.VocTestDataLoader)):
                imgs = torch_normailze(imgs)
                imgs = imgs.to(opt.device)
                # pred shape: [N, 13, 13, 125]
                preds = self.trainer.yolov2(imgs)
                # boxes shape: [batch_size, 13*13*5, 4]
                # confs shape: [batch_size, 13*13*5, 1]
                # probs shape: [batch_size, 13*13*5, 20]
                boxes, confs, probs = self.predict(preds)
                # process one img by one img
                for box, conf, prob in zip(boxes, confs, probs):
                    # score shape: [13*13*5, 20]
                    score = conf * prob
                    box_output, score_output, _ = self.nms_each_class(box, score, opt.score_threshold, opt.iou_threshold,
                                                                     opt.max_boxes_num)
                    score_output = score_output.reshape(-1, 1)
                    if box_output.size(0) != 0:
                        # [X, 5]
                        mAP_predict_in = torch.cat([box_output, score_output], dim=-1)
                    else:
                        mAP_predict_in = torch.tensor([])
                    # shape: [N, 4]; format: [xmin, ymin, xmax, ymax]
                    mAP_ground_truth_in = true_boxes.reshape(-1, 4)[:, ::-1]
                    mAP_predicts.append(mAP_predict_in.detach().cpu().numpy())
                    mAP_ground_truths.append(mAP_ground_truth_in.detach().cpu().numpy())

        MAP = mAP(mAP_predicts, mAP_ground_truths, 0.5)
        self.logger.info('AP: %.2f %%' % (MAP.elevenPointAP * 100))
        self.logger.info('mAP: %.2f %%' % (MAP.everyPointAP * 100))

    def predict(self, preds):
        xy_offset, bboxes, confs_logit, classes_logit = self.trainer.reorg_layer(preds)
        grid_size = [xy_offset.shape[0], xy_offset.shape[1]]

        boxes = bboxes.reshape(-1, grid_size[0] * grid_size[1] * opt.B, 4)
        confs = confs_logit.reshape(-1, grid_size[0] * grid_size[1] * opt.B, 1)
        probs = classes_logit.reshape(-1, grid_size[0] * grid_size[1] * opt.B, opt.coco_class_num)

        confs = confs.sigmoid()  # confs: [N, 13*13*5, 1]
        probs = torch.softmax(probs, dim=-1)  # probs: [N, 13*13*5, 20]

        xmin = boxes[..., [0]] - boxes[..., [2]] / 2
        ymin = boxes[..., [1]] - boxes[..., [3]] / 2
        xmax = boxes[..., [0]] + boxes[..., [2]] / 2
        ymax = boxes[..., [1]] + boxes[..., [3]] / 2
        # [N, 13*13*5, 4] / [xmin, ymin, xmax, ymax]
        boxes = torch.cat([xmin, ymin, xmax, ymax], dim=-1)
        return boxes, confs, probs

    @staticmethod
    def nms_each_class(boxes, scores, score_threshold, iou_threshold, max_box_num):
        """
        :param boxes: [13*13*5, 4]
        :param scores: [13*13*5, 20]
        :param score_threshold:
        :param iou_threshold:
        :param max_box_num:
        :return:
         boxes_output shape: [X, 4]
         scores_output shape: [X,]
         labels_output shape: [X,]
        """
        assert boxes.dim() == 2 and scores.dim() == 2
        boxes = boxes.clamp(0., opt.img_size)
        boxes_output = []
        scores_output = []
        labels_output = []
        # [13*13*5, 20]
        score_mask = scores.ge(score_threshold)
        # do nms for each class
        for k in range(opt.coco_class_num):
            valid_mask = score_mask[:, k]  # [M, 20]
            if valid_mask.sum() == 0:
                continue
            else:
                valid_boxes = boxes[valid_mask]  # [M, 4]
                valid_scores = scores[:, k][valid_mask]  # [M, 1]
                keep_index = gpu_nms(valid_boxes, valid_scores, iou_threshold)
                for keep_box in valid_boxes[keep_index]:
                    boxes_output.append(keep_box)
                scores_output.extend(valid_scores[keep_index])
                labels_output.extend([k for _ in range(len(keep_index))])

        assert len(boxes_output) == len(scores_output) == len(labels_output)
        num_out = len(labels_output)
        if num_out == 0:
            return torch.tensor([], device=opt.device), torch.tensor([], device=opt.device), torch.tensor([], device=opt.device)
        else:
            boxes_output = torch.stack(boxes_output, dim=0)
            scores_output = torch.tensor(scores_output)
            labels_output = torch.tensor(labels_output)
            assert boxes_output.dim() == 2
            assert labels_output.dim() == scores_output.dim() == 1
            assert boxes_output.size(0) == scores_output.numel() == labels_output.numel()
            if num_out > max_box_num:
                descend_order_index = torch.argsort(scores_output)[::-1]
                output_index = descend_order_index[:max_box_num]
            else:
                output_index = torch.arange(num_out)
            return boxes_output[output_index], scores_output[output_index], labels_output[output_index]

    @staticmethod
    def nms_all_class(boxes, scores, score_threshold, iou_threshold, max_box_num):
        """
        :param boxes: [13*13*5, 4]
        :param scores: [13*13*5, 20]
        :param score_threshold: 0.3
        :param iou_threshold: 0.45
        :param max_box_num:
        :return:
         boxes_output shape: [X, 4]
         scores_output shape: [X,]
         labels_output shape: [X,]
        """
        assert boxes.dim() == 2 and scores.dim() == 2
        boxes = boxes.clamp(0., opt.img_size)
        boxes_output = []
        scores_output = []
        labels_output = []
        # [13*13*5, 20] -> [13*13*5, 1]
        scores_mask = scores.max(dim=-1)
        labels = scores_mask[1]
        max_scores = scores_mask[0]
        # [13*13*5, 1]
        valid_mask = max_scores.ge(score_threshold)
        # do nms for all class
        if valid_mask.sum() != 0:
            valid_boxes = boxes[valid_mask]  # [M, 4]
            valid_scores = max_scores[valid_mask]  # [M, 1]
            valid_labels = labels[valid_mask]
            keep_index = gpu_nms(valid_boxes, valid_scores, iou_threshold)
            for keep_box in valid_boxes[keep_index]:
                boxes_output.append(keep_box)
            scores_output.extend(valid_scores[keep_index])
            labels_output.extend(valid_labels[keep_index])

        assert len(boxes_output) == len(scores_output) == len(labels_output)
        num_out = len(labels_output)
        if num_out == 0:
            return torch.tensor([], device=opt.device), torch.tensor([], device=opt.device), torch.tensor([], device=opt.device)
        else:
            boxes_output = torch.stack(boxes_output, dim=0)
            scores_output = torch.tensor(scores_output)
            labels_output = torch.tensor(labels_output)
            assert boxes_output.dim() == 2
            assert labels_output.dim() == scores_output.dim() == 1
            assert boxes_output.size(0) == scores_output.numel() == labels_output.numel()
            if num_out > max_box_num:
                descend_order_index = torch.argsort(scores_output)[::-1]
                output_index = descend_order_index[:max_box_num]
            else:
                output_index = torch.arange(num_out)
            return boxes_output[output_index], scores_output[output_index], labels_output[output_index]

    @staticmethod
    def letter_resize(img, target_img_size):
        """
        :param img:
        :param bboxes: format [ymax, xmax, ymin, xmin]
        :param target_img_size: [416, 416]
        :return:
            letterbox_img
            resized_bbox: [ymax, xmax, ymin, xmin]
        """
        if isinstance(img, str) and os.path.exists(img):
            img = cv2.imread(img)
        else:
            assert isinstance(img, np.ndarray)
        letterbox_img = np.full(shape=[target_img_size[0], target_img_size[1], 3], fill_value=128, dtype=np.float32)
        org_img_shape = [img.shape[0], img.shape[1]]  # [height, width]
        ratio = np.min([target_img_size[0] / org_img_shape[0], target_img_size[1] / org_img_shape[1]])
        # resized_shape format : [height, width]
        resized_shape = tuple([int(org_img_shape[0] * ratio), int(org_img_shape[1] * ratio)])
        resized_img = cv2.resize(img, resized_shape[::-1])
        dh = target_img_size[0] - resized_shape[0]
        dw = target_img_size[1] - resized_shape[1]
        letterbox_img[(dh // 2):(dh // 2 + resized_shape[0]), (dw // 2):(dw // 2 + resized_shape[1]), :] = resized_img
        letterbox_img = letterbox_img.astype(np.uint8)
        return letterbox_img, ratio, dh // 2, dw // 2

    def add_train_summary(self, loss_dict, step):
        self.writer.add_scalar(tag='Train/total_loss', scalar_value=loss_dict['total_loss'], global_step=step)
        self.writer.add_scalar('Train/xy_loss', loss_dict['dxdy_loss'], step)
        self.writer.add_scalar('Train/wh_loss', loss_dict['twth_loss'], step)
        self.writer.add_scalar('Train/conf_loss', loss_dict['conf_loss'], step)
        self.writer.add_scalar('Train/class_loss', loss_dict['class_loss'], step)
        self.writer.add_scalar('Train/obj_loss', loss_dict['obj_loss'], step)
        self.writer.add_scalar('Train/noobj_loss', loss_dict['noobj_loss'], step)

    def add_test_summary(self, pred_dict, step):
        self.writer.add_histogram('Test/score', pred_dict['score'], global_step=step)
        self.writer.add_histogram('Test/conf', pred_dict['conf'], step)
        self.writer.add_histogram('Test/cls', pred_dict['cls'], step)


if __name__ == '__main__':
    model = Yolov2()
    model.train()




