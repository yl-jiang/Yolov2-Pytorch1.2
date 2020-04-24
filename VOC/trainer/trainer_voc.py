#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/30 下午1:33
# @Author  : jyl
# @File    : trainer.py
import torch
from model import BackboneVOC
import os
from torchnet.meter import AverageValueMeter
from utils import parse_anchors, yolov2_bbox_iou, xywh2xyxy
import numpy as np
from collections import namedtuple
from torch.utils.tensorboard.writer import SummaryWriter
from model import load_weights

LossTuple = namedtuple('LOSSTUPLE',
                       field_names=['xy_loss', 'wh_loss', 'conf_loss', 'cls_loss', 'total_loss'])


class YOLOV2VOCTrainer:

    def __init__(self, opt):
        self.opt = opt
        self.anchor_base = torch.from_numpy(parse_anchors(opt)).to(opt.device)
        self.yolov2 = BackboneVOC(opt.voc_class_num, self.anchor_base.size(0)).to(opt.device)
        self.optimizer = self.init_optimizer()
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.mse = torch.nn.MSELoss(reduction='sum')
        self.ce = torch.nn.CrossEntropyLoss(reduction='sum')
        self.avgmeters = {k: AverageValueMeter() for k in LossTuple._fields}
        self.summary = SummaryWriter(log_dir=self.opt.summary_dir)

    def load(self, model_dir, load_optimizer=True, lr_scheduler=None):
        if not os.path.exists(model_dir):
            print(f'file: {model_dir} not found')
            load_weights(self.yolov2, self.opt.github_model)
            # self.init_weights(self.yolov2)
            return float('inf'), 0, 0
        else:
            state_dict = torch.load(model_dir)
            if 'loss' in state_dict:
                loss = state_dict['loss']
            else:
                loss = float('inf')
            if 'epoch' in state_dict:
                epoch = state_dict['epoch']
            else:
                epoch = 0
            if 'total_steps' in state_dict:
                steps = state_dict['total_steps']
            else:
                steps = 0
            if 'model' in state_dict:
                print(f'loading pretrained model: {model_dir}')
                self.yolov2.load_state_dict(state_dict['model'])
            if load_optimizer and 'optimizer' in state_dict:
                self.optimizer.load_state_dict(state_dict['optimizer'])
            if lr_scheduler is not None and 'lr_scheduler' in state_dict:
                lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        return loss, epoch, steps

    def train_step(self, imgs, labels, global_step):
        """
        :param imgs: [N, 3, 416, 416]
        :param labels: [N, 13, 13, 5, 25]
        :param global_step:
        :return:
        """
        # img_size's format is [h, w]
        self.yolov2.train()
        self.yolov2.zero_grad()
        img_size = torch.tensor([imgs.shape[2], imgs.shape[3]]).float().to(self.opt.device)
        # preds: [N, 13, 13, 125]
        preds = self.yolov2(imgs)
        total_loss = self.loss_layer(preds, labels, self.anchor_base, img_size[0])
        # total_loss = self.compute_loss_v3(preds, labels, self.anchor_base, img_size[0])
        total_loss.backward()
        self.optimizer.step()
        self.update_loss_summary(global_step)

    def predict(self, img):
        self.yolov2.eval()
        with torch.no_grad():
            preds = self.yolov2(img)
            img_size = img.size(2)
            xy_offset, bboxes, confs_logit, classes_logit = self.reorg_layer(preds, self.anchor_base, img_size)
            grid_size = preds.size(1)

            boxes = bboxes.reshape(-1, grid_size * grid_size * self.opt.B, 4)
            confs = confs_logit.reshape(-1, grid_size * grid_size * self.opt.B, 1)
            probs = classes_logit.reshape(-1, grid_size * grid_size * self.opt.B, self.opt.voc_class_num)
            # confs: [N, 13*13*5, 1]
            confs = confs.sigmoid()
            # probs: [N, 13*13*5, 20]
            probs = torch.softmax(probs, dim=-1)
            # boxes: [ctr_x, ctr_y, w, h]
            xmin = boxes[..., [0]] - boxes[..., [2]] / 2
            ymin = boxes[..., [1]] - boxes[..., [3]] / 2
            xmax = boxes[..., [0]] + boxes[..., [2]] / 2
            ymax = boxes[..., [1]] + boxes[..., [3]] / 2
            # [N, 13*13*5, 4] / [xmin, ymin, xmax, ymax]
            boxes = torch.cat([xmin, ymin, xmax, ymax], dim=-1)
            return boxes, confs, probs

    # rescale predicts to input_image scale / 该函数在训练和预测的时候都要用到
    def reorg_layer(self, preds, anchor_base, img_size):
        """
        :param preds:[N, 13, 13, 125]
        :param anchor_base:[[w, h], ,,,]
        :param img_size: 416
        注意：kmeans得到anchors的w,h是相对于输入图片的scale的(也就是416)，论文中的p_w,p_h是相对于feature map大小的，
        计算loss时需要先将anchor的scale转换到feature map的scale
        :return:
            bboxes: [N, 13, 13, 5, 4]
            confs_logit: [N, 13, 13, 5, 1]
            classes_logit: [N, 13, 13, 5, 20]
        """
        # grid_size format is [h,w]
        grid_size = [preds.size(1), preds.size(2)]

        # ratio: feature map与输入图片的缩放比
        # ratio format is [h,w]
        ratio = img_size / torch.tensor(grid_size).float().to(self.opt.device)
        # rescaled_anchors format is [w,h] / make anchors's scale same as predicts
        rescaled_anchors = (anchor_base / ratio.flip(0)).to(self.opt.device)
        # resahpe preds to [N, 13, 13, 5, 25]
        preds = preds.reshape(-1, grid_size[0], grid_size[1], self.opt.B, 5 + self.opt.voc_class_num)

        # box_xy: [N, 13, 13, 5, 2] / format [x, y]
        # box_wh: [N, 13, 13, 5, 2] / format [w, h]
        # confs: [N, 13, 13, 5, 1]
        # classes: [N, 13, 13, 5, 20]
        pred_box_xy, pred_box_wh, confs_logit, classes_logit = preds.split([2, 2, 1, self.opt.voc_class_num], dim=-1)
        pred_box_xy = pred_box_xy.sigmoid()
        grid_x = np.arange(grid_size[1])
        grid_y = np.arange(grid_size[0])
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        xy_offset = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        # xy_offset: [13, 13, 1, 2] / [[ 0.,  0.],[ 1.,  0.],[ 2.,  0.],...,[11., 12.],[12., 12.]]
        xy_offset = torch.from_numpy(xy_offset).float().to(self.opt.device)
        xy_offset = xy_offset.reshape(grid_size[1], grid_size[0], 1, 2)

        # rescale to input_image scale(416)
        pred_box_xy = (pred_box_xy + xy_offset) * ratio
        # compute in the scale 13
        pred_box_wh = torch.exp(pred_box_wh) * rescaled_anchors
        # rescale to input_image scale(416)
        pred_box_wh = pred_box_wh * ratio

        # reset scaled pred_box to bounding box format [x, y, w, h]
        # bboxes: [N, 13, 13, 5, 4]
        bboxes = torch.cat([pred_box_xy, pred_box_wh], dim=-1)

        return xy_offset, bboxes, confs_logit, classes_logit

    def compute_loss_v3(self, preds, ground_truth, anchor_base, img_size):
        """
        :param preds: [N, 125, 13, 13]
        :param ground_truth: [N, 13, 13, 5, 25]
        :param anchor_base: [5, 2]
        :param img_size: 416
        :return:
        """
        # grid_size format is [h, w]
        N = preds.size(0)
        grid_size = preds.shape[2]
        bce_no_reduce = torch.nn.BCEWithLogitsLoss(reduction='none')
        # ratio's format is [h, w]
        ratio = (img_size / grid_size).float().to(self.opt.device)
        xy_offset, pred_bboxes, pred_confs, pred_classes = self.reorg_layer(preds, anchor_base, img_size)
        # obj_mask记录存在目标的cell
        # obj_mask: [N, 13, 13, 5]
        obj_mask = ground_truth[..., 4].bool()

        # ignore_mask：忽略那些与任一gt_box的iou值大于0.6的pred_box的conf损失
        # ignore_mask: [N, 13, 13, 5]
        ignore_mask = torch.zeros_like(obj_mask).bool()
        for i in range(N):
            # [13, 13, 5, 4] & [13, 13, 5] -> [M/4, 4]
            # valid_bbox: [M, 4] / [ctr_x, ctr_y, w, h]
            valid_bbox = ground_truth[i, ..., :4][obj_mask[i]]
            # valid_bbox = torch.masked_select(ground_truth[i, ..., :4], obj_mask[i, ..., None]).reshape(-1, 4)
            # ious: [13, 13, 5, M]
            # [13, 13, 5, 4] & [M, 4] -> [13, 13, 5, M]
            ious = yolov2_bbox_iou(pred_bboxes[i], valid_bbox)
            # best_iou: [13, 13, 5]
            max_iou, _ = torch.max(ious, dim=-1)
            ignore_mask[i] = max_iou.lt(self.opt.pos_iou_thresh)

        # pred_xy: [N, 13, 13, 5, 2]
        # pred_xy's and label_xy's format is [w, h]
        # 因为pred_bboxes经过reorg_layer处理后rescale到了input_img的scale,在计算loss时需要把pred_xy的scale缩放到grid的scale
        pred_dxdy = (pred_bboxes[..., 0:2] / ratio) - xy_offset
        true_dxdy = (ground_truth[..., 0:2] / ratio) - xy_offset

        # pred_wh: [N, 13, 13, 5, 2]
        # 这里除以anchor是因为reorg_layer函数处理后对predict_bbox_wh乘以了anchor，这里只是还原模型最初输出的预测值
        pred_twth = pred_bboxes[..., 2:4] / anchor_base
        true_twth = ground_truth[..., 2:4] / anchor_base
        # for numercial stability
        # 防止等于0的值在进行对数运算时得到负无穷
        pred_twth[pred_twth == 0.] = 1.
        true_twth[true_twth == 0.] = 1.
        # 这里取对数是因为reorg_layer对pred_wh进行了exponential运算
        pred_twth = torch.clamp_min(pred_twth, min=1e-9).log()
        true_twth = torch.clamp_min(true_twth, min=1e-9).log()

        # box with smaller area has higer weight
        # [N, 13, 13, 5]
        loc_loss_weight = 1.5 - (ground_truth[..., 2] / img_size) * (ground_truth[..., 3] / img_size)
        assert (loc_loss_weight <= 1.5).all()
        # 对存在目标的预测框计算xy和wh损失
        # [N, 13, 13, 5, 2] & [N, 13, 13, 5, 1] & [N, 13, 13, 5, 1] -> [N,13,13,5,1]
        obj_mask = obj_mask[..., None].float()
        dxdy_loss = torch.pow(true_dxdy - pred_dxdy, 2.) * obj_mask * loc_loss_weight[..., None]
        dxdy_loss = self.opt.reg_scale * dxdy_loss.sum() / N
        twth_loss = torch.pow(true_twth - pred_twth, 2.) * obj_mask * loc_loss_weight[..., None]
        twth_loss = self.opt.reg_scale * twth_loss.sum() / N

        # 对存在目标的预测框计算置信度损失
        # [N, 13, 13, 5, 1] & ([N,13,13,5,1] & [N,13,13,5,1] -> [N,13,13,5,1]
        conf_loss_obj = obj_mask * bce_no_reduce(pred_confs, obj_mask)
        # 不存在目标且与任一gt_box之间的iou值小于0.6的预测框计算置信度损失
        # [N,13,13,5,1] & [N,13,13,5,1] & [N,13,13,5,1] -> [N,13,13,5,1]
        # ignore_mask: [N, 13, 13, 5, 1]
        ignore_mask = ignore_mask[..., None].float()
        conf_loss_noobj = (1. - obj_mask) * ignore_mask * bce_no_reduce(pred_confs, obj_mask)

        # total conf loss
        # [batch_size, 13, 13, 5, 1]
        conf_loss = self.opt.obj_scale * conf_loss_obj + self.opt.noobj_scale * conf_loss_noobj
        if self.opt.use_focal_loss:
            focal_mask = self.focal_loss(labels=obj_mask, preds=pred_confs)
            conf_loss = (focal_mask * conf_loss).sum() / N
        else:
            conf_loss = conf_loss.sum() / N

        # 对存在目标的预测框计算分类损失
        if self.opt.use_smooth_labels:
            true_classes = self.smooth_labels(ground_truth[..., 5:], self.opt.voc_class_num)
        else:
            true_classes = ground_truth[..., 5:]
        # [batch_size,13,13,5] & [batch_size,13,13,5,20] & [batch_size,13,13,5,20] -> [batch_size,13,13,5,20]
        class_loss = obj_mask * bce_no_reduce(pred_classes, true_classes)
        class_loss = self.opt.cls_scale * class_loss.sum() / N

        total_loss = dxdy_loss + twth_loss + conf_loss + class_loss
        loss_list = [dxdy_loss, twth_loss, conf_loss, class_loss, total_loss]
        self.update_meters(loss_list)
        return total_loss

    def loss_layer(self, preds, targets, anchor_base, img_size):
        """
        class loss使用cross entropy
        :param preds:[N, 13, 13, 125]
        :param targets:[N, 13, 13, 5, 25]
        :param anchor_base: [5, 2]
        :param img_size: 416
        :return:
        """
        N = preds.size(0)
        # grid_size format is [h, w]
        grid_size = preds.shape[1]
        ratio = img_size / grid_size
        assert grid_size == 13
        assert ratio == 32.

        ratio = ratio.float()
        xy_offset, pred_bboxes, pred_confs, pred_classes = self.reorg_layer(preds, anchor_base, img_size)
        # [N, 13, 13, 5, 4] & [N, 13, 13, 5, 4] -> [N, 13, 13, 5]
        gt_pred_ious = []
        obj_mask = targets[..., 4].bool()
        for i in range(N):
            valid_bbox = targets[i, ..., :4][obj_mask[i]]
            ious = yolov2_bbox_iou(pred_bboxes[i], valid_bbox)
            gt_pred_ious.append(ious)

        # pred_xy: [N, 13, 13, 5, 2] / pred_xy's and label_xy's format is [w, h]
        # 因为pred_bboxes经过reorg_layer处理后rescale到了input_img的scale,
        # 在计算loss时需要把pred_xy的scale缩放到grid的scale
        pred_txty = (pred_bboxes[..., 0:2] / ratio) - xy_offset
        true_txty = (targets[..., 0:2] / ratio) - xy_offset
        # pred_wh: [N, 13, 13, 5, 2]
        # 这里除以anchor是因为reorg_layer函数处理后对predict_bbox_wh乘以了scale为416的anchor，
        # 这里只是还原模型最初输出的预测值
        pred_twth = pred_bboxes[..., 2:4] / anchor_base
        true_twth = targets[..., 2:4] / anchor_base
        # for numercial stability
        # 防止等于0的值在进行对数运算时得到负无穷
        pred_twth[pred_twth == 0] = 1.
        true_twth[true_twth == 0] = 1.
        # 这里取对数是因为reorg_layer对pred_wh进行了exponential运算
        pred_twth = pred_twth.clamp(1e-9, 1e9).log()
        true_twth = true_twth.clamp(1e-9, 1e9).log()

        # [N, 13, 13, 5, 4]
        pred_bbox = torch.cat([pred_txty, pred_twth], dim=-1)
        targets[..., :4] = torch.cat([true_txty, true_twth], dim=-1)

        # conf_mask: [N, 13, 13, 5]
        # tar_confs: [N, 13, 13, 5]
        # tar_txty: [N, 13, 13, 5, 2]
        # tar_twth: [N, 13, 13, 5, 2]
        # tar_calsses: [N, 13, 13, 5, 20]
        noobj_mask, tar_confs, tar_txty, tar_twth, tar_classes = self.build_targets(pred_bbox, targets, gt_pred_ious)
        obj_mask = targets[..., 4] > 0

        # txty_loss: [N, 13, 13, 5, 2]
        txty_loss = self.mse(pred_txty[obj_mask], tar_txty[obj_mask]) / N
        # twth_loss: [N, 13, 13, 5, 2]
        twth_loss = self.mse(pred_twth[obj_mask], tar_twth[obj_mask]) / N

        # [N, 13, 13, 5, 1] -> [N, 13, 13, 5]
        pred_confs = pred_confs.sigmoid().squeeze(dim=-1)
        # pred_box that has no object and iou lower than 0.6 compute confidence loss
        # pred_box that has object compute confidence loss
        conf_mask = ~obj_mask & noobj_mask
        noobj_conf_loss = self.opt.noobj_scale * self.mse(pred_confs[conf_mask],
                                                          torch.zeros_like(pred_confs)[conf_mask]) / N
        obj_conf_loss = self.opt.obj_scale * self.mse(pred_confs[obj_mask], tar_confs[obj_mask]) / N
        conf_loss = noobj_conf_loss + obj_conf_loss

        tar_classes = tar_classes[obj_mask].reshape(-1)  # [N,]
        pred_classes = pred_classes[obj_mask].reshape(-1, self.opt.voc_class_num)  # [N, 20]
        cls_loss = self.ce(pred_classes, tar_classes) / N
        total_loss = self.opt.reg_scale * (txty_loss + twth_loss) + conf_loss + self.opt.cls_scale * cls_loss
        loss_list = [txty_loss, twth_loss, conf_loss, cls_loss, total_loss]
        self.update_meters(loss_list)
        return total_loss

    def build_targets(self, pred_bbox, tragets, gt_pred_ious):
        """
        :param pred_bbox:[N, 13, 13, 5, 4]
        :param tragets:[N, 13, 13, 5, 25] / last dimension: [dx, dy, tw, th, conf, labels]
        :param gt_pred_ious:[N, 13, 13, M]
        :return:
        """
        N = pred_bbox.size(0)
        grid_size = [pred_bbox.shape[1], pred_bbox.shape[2]]
        # [batch_size, 13, 13, 5]
        noobj_mask = torch.ones(N, grid_size[0], grid_size[1], self.opt.B).bool().to(self.opt.device)
        tar_conf = torch.zeros(N, grid_size[0], grid_size[1], self.opt.B).float().to(self.opt.device)
        tar_txty = torch.zeros(N, grid_size[0], grid_size[1], self.opt.B, 2).float().to(self.opt.device)
        tar_twth = torch.zeros(N, grid_size[0], grid_size[1], self.opt.B, 2).float().to(self.opt.device)
        tar_class = torch.zeros(N, grid_size[0], grid_size[1], self.opt.B).long().to(self.opt.device)

        # [batch_size, 13, 13, 5]
        obj_index = tragets[..., 4].bool()

        for b in range(N):
            # 每个cell的所有(5个)预测框与所有gt_bbox计算iou，最大iou值小于阈值的预测框标记为background并计算conf损失，
            # 最大iou大于阈值的cell不计入损失函数的计算
            # [13, 13, 5, M]
            has_obj = obj_index[b]
            iou = gt_pred_ious[b]
            # max_iou: [13, 13, 5]
            max_iou, argmax_iou = torch.max(iou, dim=-1)
            # [13, 13, 5] / bool / 不包含目标的cell的所有预测框中，预测box和所有gt_box的iou小于0.6的计算conf损失
            mask = torch.ge(max_iou, self.opt.pos_iou_thresh)
            noobj_mask[b][mask] = False

            # 存在目标cell
            if self.opt.rescore:
                tar_conf[b][has_obj] = gt_pred_ious[b][has_obj]
            else:
                tar_conf[b][has_obj] = 1.
            tar_txty[b][has_obj] = tragets[b][has_obj][:, 0:2]
            tar_twth[b][has_obj] = tragets[b][has_obj][:, 2:4]
            # obj_class[b][obj_mask[b]] = ground_truth[b][obj_mask[b]][:, 5:]
            tar_class[b][has_obj] = torch.argmax(tragets[b][has_obj][:, 5:], dim=-1)

        return noobj_mask, tar_conf, tar_txty, tar_twth, tar_class

    @staticmethod
    # 增大模型比较没有把握的case的损失，减小那些比较有把握case的损失
    def focal_loss(labels, preds):
        alpha = 2.
        gamma = 1.
        # [batch_size,13,13,5,1] & [batch_size,13,13,5,1] -> [batch_size,13,13,5,1]
        focal_weights = alpha * torch.pow(torch.abs(labels - torch.sigmoid(preds)), gamma)
        return focal_weights

    @staticmethod
    def smooth_labels(labels, class_num):
        delta = 0.01
        smoothness = (1 - delta) * labels + delta * (1. / class_num)
        return smoothness

    def init_optimizer(self):
        params = [{'params': self.yolov2.pth_layer.parameters(), 'lr': self.opt.pth_lr},
                  {'params': self.yolov2.mid_layer.parameters(), 'lr': self.opt.yolo_lr},
                  {'params': self.yolov2.output_layer.parameters(), 'lr': self.opt.yolo_lr}]
        if self.opt.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(params=params, lr=self.opt.lr, momentum=self.opt.optim_momentum,
                                        weight_decay=self.opt.optim_weight_decay)
        else:
            optimizer = torch.optim.Adam(params=params, lr=self.opt.lr, weight_decay=self.opt.optim_weight_decay)
        return optimizer

    def save(self, loss, lr_scheduler, epoch, steps, save_path):
        model_state = self.yolov2.state_dict()
        optimizer_state = self.optimizer.state_dict()
        if lr_scheduler is not None:
            lr_scheduler_state = lr_scheduler.state_dict()
        else:
            lr_scheduler_state = lr_scheduler
        state_dict = {'model': model_state,
                      'optimizer': optimizer_state,
                      'lr_scheduler': lr_scheduler_state,
                      'epoch': epoch,
                      'total_steps': steps,
                      'loss': loss}

        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(obj=state_dict, f=save_path)

    def adjust_lr(self, epoch):

        def set_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if self.opt.optimizer_type == 'SGD':
            if epoch < 15:
                set_lr(self.optimizer, 1e-4)
            elif 15 < epoch < 45:
                set_lr(self.optimizer, 1e-3)
            elif 45 < epoch < 115:
                set_lr(self.optimizer, 1e-5)
            else:
                set_lr(self.optimizer, self.opt.lr * 0.1 ** (epoch // 50))
        else:
            if epoch < 50:
                set_lr(self.optimizer, 1e-3)
            elif 50 <= epoch < 150:
                set_lr(self.optimizer, 1e-4)
            else:
                set_lr(self.optimizer, self.opt.lr * 0.05 ** (epoch // 30))

    def update_meters(self, loss_list):
        tmp = []
        for loss in loss_list:
            if isinstance(loss, torch.Tensor):
                tmp.append(loss.detach().cpu().item())
            else:
                tmp.append(loss)
        _meter = LossTuple(*tmp)
        _dict = {k: v for k, v in _meter._asdict().items()}
        for k, met in self.avgmeters.items():
            met.add(_dict[k])

    def get_loss_meters(self):
        return {k: v.mean for k, v in self.avgmeters.items()}

    def update_loss_summary(self, global_step):
        _lossdict = self.get_loss_meters()
        self.summary.add_scalar('train/total_loss', _lossdict['total_loss'], global_step)
        self.summary.add_scalar('train/conf_loss', _lossdict['conf_loss'], global_step)
        self.summary.add_scalar('train/cls_loss', _lossdict['cls_loss'], global_step)
        self.summary.add_scalar('train/xy_loss', _lossdict['xy_loss'], global_step)
        self.summary.add_scalar('train/wh_loss', _lossdict['wh_loss'], global_step)

    @staticmethod
    def init_weights(model):
        print('Training from stratch ...')
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(tensor=m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(tensor=m.bias, val=0.0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(tensor=m.weight, val=1.0)
                torch.nn.init.constant_(tensor=m.bias, val=0.0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(tensor=m.weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(tensor=m.bias, val=0.0)
