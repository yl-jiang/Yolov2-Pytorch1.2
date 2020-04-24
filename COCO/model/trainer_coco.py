#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/5 下午5:14
# @Author  : jyl
# @File    : trainer_coco.py
import torch
from model import BackboneCOCO
from config import opt
import os
from torchnet.meter import AverageValueMeter
from utils import parse_anchors, yolov2_bbox_iou, xywh2xyxy
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


class YOLOV2COCOTrainer:

    def __init__(self):
        self.yolov2 = BackboneCOCO().to(opt.device)
        self.optimizer = self.init_optimizer()
        # self.anchors dtype: np.ndarray
        self.anchors = torch.from_numpy(parse_anchors(opt.anchors_path)).to(opt.device)
        self.loss_meter = AverageValueMeter()
        self.logger = opt.logger

    def use_pretrain(self, model_dir, load_optimizer=True):
        if not os.path.exists(model_dir):
            raise OSError(2, 'No such file or directory', model_dir)
        else:
            state_dict = torch.load(model_dir)
            if 'loss' in state_dict.keys():
                self.last_loss = state_dict['loss']
            else:
                self.last_loss = float('inf')
            if 'epoch' in state_dict.keys():
                self.epoch_num = state_dict['epoch']
            else:
                self.epoch_num = 0
            if 'total_steps' in state_dict.keys():
                self.total_steps = state_dict['total_steps']
            else:
                self.total_steps = 0
            if 'model' in state_dict.keys():
                self.logger.info(f'load pretrained model from : {model_dir}')
                self.yolov2.load_state_dict(state_dict['model'])
            if load_optimizer and 'optimizer' in state_dict.keys():
                self.optimizer.load_state_dict((state_dict['optimizer']))

    def train_step(self, imgs, labels, epoch):
        """
        :param imgs:
            [batch_size, 3, 416, 416]
        :param labels:
            [batch_size, 13, 13, 5, 85]
        :param epoch:
        :return:
        """
        # img_size's format is [h, w]
        self.img_size = torch.tensor([imgs.shape[2], imgs.shape[3]], dtype=torch.float32, device=opt.device)
        # preds: [N, 13, 13, 425]
        preds = self.yolov2(imgs)
        self.loss_dict = self.compute_loss(preds, labels)
        self.adjust_lr(epoch)
        self.optimizer.zero_grad()
        self.loss_dict['total_loss'].backward()
        self.optimizer.step()
        self.loss_meter.add(self.loss_dict['total_loss'].item())

    # rescale predicts to input_image scale
    # 该函数在训练和预测的时候都要用到
    def reorg_layer(self, preds):
        """
        :param preds:
            [N, 13, 13, 125]
        :param anchors:
            [[w, h], ,,,]
            注意：kmeans得到anchors的w,h是相对于输入图片的scale的(也就是416)，论文中的p_w,p_h是相对于feature map大小的，
            计算loss时需要先将anchor的scale转换到feature map的scale
        :return:
            bboxes: [batch_size, 13, 13, 5, 4]
            confs_logit: [batch_size, 13, 13, 5, 1]
            classes_logit: [batch_size, 13, 13, 5, 20]
        """
        # grid_size format is [h,w]
        grid_size = [preds.size(1), preds.size(2)]

        # ratio: feature map与输入图片的缩放比
        # ratio format is [h,w]
        ratio = self.img_size / torch.tensor(grid_size, dtype=torch.float32, device=opt.device)
        # rescaled_anchors format is [w,h] / make anchors's scale same as predicts
        self.rescaled_anchors = (self.anchors / ratio.flip(0)).to(opt.device)
        # resahpe preds to [N, 13, 13, 5, 25]
        preds = preds.contiguous().view(-1, grid_size[0], grid_size[1], opt.B, 5+opt.coco_class_num)

        # box_xy: [N, 13, 13, 5, 2] / format [x, y]
        # box_wh: [N, 13, 13, 5, 2] / format [w, h]
        # confs: [N, 13, 13, 5, 1]
        # classes: [N, 13, 13, 5, 20]
        box_xy, box_wh, confs_logit, classes_logit = preds.split([2, 2, 1, opt.coco_class_num], dim=-1)
        box_xy = box_xy.sigmoid()
        grid_x = np.arange(grid_size[1], dtype=np.float32)
        grid_y = np.arange(grid_size[0], dtype=np.float32)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        xy_offset = np.concatenate([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], axis=-1)
        # xy_offset: [13, 13, 1, 2]
        xy_offset = torch.from_numpy(xy_offset).to(torch.float32).to(opt.device)
        xy_offset = xy_offset.contiguous().view(grid_size[1], grid_size[0], 1, 2)

        # rescale to input_image scale
        box_xy = (box_xy + xy_offset) * ratio.flip(0)
        # compute in the scale 13
        box_wh = torch.exp(box_wh) * self.rescaled_anchors
        # rescale to input_image scale
        box_wh = box_wh * ratio.flip(0)

        # reset scaled pred_box to bounding box format [x, y, w, h]
        # bboxes: [N, 13, 13, 5, 4]
        bboxes = torch.cat([box_xy, box_wh], dim=-1)

        return xy_offset, bboxes, confs_logit, classes_logit

    def compute_loss_v3(self, preds, ground_truth):
        """
        :param preds:
            [batch_size, 125, 13, 13]
        :param ground_truth:
            [batch_size, 13, 13, 5, 25]
        :return:
        """
        # grid_size format is [h, w]
        grid_size = [preds.shape[2], preds.shape[3]]
        # ratio's format is [h, w]
        ratio = torch.tensor([self.img_size[0] / grid_size[0], self.img_size[1] / grid_size[1]], dtype=torch.float32)
        ratio = ratio.to(opt.device)
        xy_offset, pred_bboxes, pred_confs, pred_classes = self.reorg_layer(preds)
        # obj_mask记录存在目标的cell
        # obj_mask: [batch_size, 13, 13, 5]
        obj_mask = ground_truth[..., 4].to(torch.bool)
        # ignore_mask：记录bbox iou不满足条件的预测框位置
        # ignore_mask: [batch_size, 13, 13, 5]
        ignore_mask = torch.empty_like(obj_mask)

        for i in range(preds.size(0)):
            # [13, 13, 5, 4] & [13, 13, 5] -> [M/4, 4]
            # valid_bbox: [M, 4]
            valid_bbox = ground_truth[i, ..., :4][obj_mask[i]]
            # valid_bbox = torch.masked_select(ground_truth[i, ..., :4], obj_mask[i, ..., None]).reshape(-1, 4)
            # ious: [13, 13, 5, M]
            # [13, 13, 5, 4] & [M, 4] -> [13, 13, 5, M]
            ious = yolov2_bbox_iou(pred_bboxes[i], valid_bbox)
            # best_iou: [13, 13, 5]
            best_iou = torch.max(ious, dim=-1)[0]
            ignore_mask[i] = torch.lt(best_iou, opt.best_iou_threshold)

        # pred_xy: [batch_size, 13, 13, 5, 2]
        # pred_xy's and label_xy's format is [w, h]
        # 因为pred_bboxes经过reorg_layer处理后rescale到了input_img的scale,在计算loss时需要把pred_xy的scale缩放到grid的scale
        pred_xy = pred_bboxes[..., 0:2] / ratio.flip((0, )) - xy_offset
        true_xy = ground_truth[..., 0:2] / ratio.flip((0, )) - xy_offset

        # pred_wh: [batch_size, 13, 13, 5, 2]
        # 这里除以anchor是因为reorg_layer函数处理后对predict_bbox_wh乘以了anchor，这里只是还原模型最初输出的预测值
        pred_twth = pred_bboxes[..., 2:4] / self.anchors
        true_twth = ground_truth[..., 2:4] / self.anchors
        # for numercial stability
        # 防止等于0的值在进行对数运算时得到负无穷
        pred_twth[(pred_twth == 0.).nonzero()] = 1.
        true_twth[(true_twth == 0.).nonzero()] = 1.
        pred_twth = torch.clamp_min(pred_twth, min=1e-9)
        true_twth = torch.clamp_min(true_twth, min=1e-9)
        # 这里取对数是因为reorg_layer对pred_wh进行了exponential运算
        pred_twth = torch.log(pred_twth)
        true_twth = torch.log(true_twth)

        # box with smaller area has higer weight
        # [batch_size, 13, 13, 5]
        box_loss_scale = 2. - (ground_truth[..., 2] / self.img_size[1]) * (ground_truth[..., 3] / self.img_size[0])

        # 对存在目标的预测框计算xy和wh损失
        # [batch_size, 13, 13, 5, 2] & [batch_size, 13, 13, 5, 1] & [batch_size, 13, 13, 5, 1] -> [batch_size,13,13,5,1]
        obj_mask = obj_mask[..., None].to(torch.float32)
        xy_loss = torch.sum(torch.pow(true_xy - pred_xy, 2.) * obj_mask * box_loss_scale[..., None])
        wh_loss = torch.sum(torch.pow(true_twth - pred_twth, 2.) * obj_mask * box_loss_scale[..., None])

        # 对存在目标的预测框计算置信度损失
        # [batch_size, 13, 13, 5, 1] & ([batch_size,13,13,5,1] & [batch_size,13,13,5,1] -> [batch_size,13,13,5,1]
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        conf_loss_obj = obj_mask * bce_loss(pred_confs, obj_mask)

        # 对不存在目标且bbox iou也不符合要求的预测框计算置信度损失
        # [batch_size,13,13,5,1] & [batch_size,13,13,5,1] & [batch_size,13,13,5,1] -> [batch_size,13,13,5,1]
        # ignore_mask: [batch_size, 13, 13, 5, 1]
        ignore_mask = ignore_mask[..., None].to(torch.float32)
        conf_loss_noobj = (1. - obj_mask) * ignore_mask * bce_loss(pred_confs, obj_mask)

        # 对不存在目标但与gt_bbox的iou满足条件的预测框不计入损失

        # total conf loss
        # [batch_size, 13, 13, 5, 1]
        conf_loss = conf_loss_obj + conf_loss_noobj
        if opt.use_focal_loss:
            focal_mask = self.focal_loss(labels=obj_mask, preds=pred_confs)
            conf_loss = torch.sum(focal_mask * conf_loss)
        else:
            conf_loss = torch.sum(conf_loss)

        # 对存在目标的预测框计算分类损失
        if opt.use_smooth_labels:
            true_classes = self.smooth_labels(ground_truth[..., 5:], opt.coco_class_num)
        else:
            true_classes = ground_truth[..., 5:]
        # [batch_size,13,13,5] & [batch_size,13,13,5,20] & [batch_size,13,13,5,20] -> [batch_size,13,13,5,20]
        class_loss = torch.sum(obj_mask * bce_loss(pred_classes, true_classes))

        # get loss of single img
        total_loss = (xy_loss + wh_loss + conf_loss + class_loss) / opt.batch_size
        loss_dict = {'total_loss': total_loss,
                     'xy_loss': xy_loss / opt.batch_size,
                     'wh_loss': wh_loss / opt.batch_size,
                     'conf_loss': conf_loss / opt.batch_size,
                     'class_loss': class_loss / opt.batch_size}

        return loss_dict

    def compute_loss(self, preds, ground_truth):
        """
        class loss使用cross entropy
        :param preds:
            [batch_size, 13, 13, 425]
        :param ground_truth:
            [batch_size, 13, 13, 5, 25]
        :return:
        """
        # grid_size format is [h, w]
        grid_size = preds.shape[1:3]
        assert grid_size == torch.Size([13, 13])
        # ratio's format is [h, w]
        ratio = ([self.img_size[0] / grid_size[0], self.img_size[1] / grid_size[1]])
        assert (ratio[0].item(), ratio[1].item()) == (32., 32.)
        ratio = torch.tensor(ratio, dtype=torch.float32, device=opt.device)

        xy_offset, pred_bboxes, pred_confs, pred_classes = self.reorg_layer(preds)
        # [batch_size, 13, 13, 5, 4] & [batch_size, 13, 13, 5, 4] -> [batch_size, 13, 13, 5]
        iou_pred_gt = self.bbox_iou(pred_bboxes, ground_truth[..., :4])

        # pred_xy: [batch_size, 13, 13, 5, 2]
        # pred_xy's and label_xy's format is [w, h]
        # 因为pred_bboxes经过reorg_layer处理后rescale到了input_img的scale,在计算loss时需要把pred_xy的scale缩放到grid的scale
        pred_txty = pred_bboxes[..., 0:2] / ratio.flip((0, )) - xy_offset
        true_txty = ground_truth[..., 0:2] / ratio.flip((0, )) - xy_offset
        # pred_wh: [batch_size, 13, 13, 5, 2]
        # 这里除以anchor是因为reorg_layer函数处理后对predict_bbox_wh乘以了scale为416的anchor，这里只是还原模型最初输出的预测值
        pred_twth = pred_bboxes[..., 2:4] / self.anchors
        true_twth = ground_truth[..., 2:4] / self.anchors
        # for numercial stability
        # 防止等于0的值在进行对数运算时得到负无穷
        pred_twth[pred_twth == 0.] = 1.
        true_twth[true_twth == 0.] = 1.
        # 这里取对数是因为reorg_layer对pred_wh进行了exponential运算
        pred_twth = pred_twth.clamp(1e-9, 1e9).log()
        true_twth = true_twth.clamp(1e-9, 1e9).log()

        # [batch_size, 13, 13, 5, 4]
        pred_bbox = torch.cat([pred_txty, pred_twth], dim=-1)
        ground_truth[..., :4] = torch.cat([true_txty, true_twth], dim=-1)

        # conf_mask: [batch_size, 13, 13, 5]
        # coord_mask: [batch_size, 13, 13, 5, 1]
        # class_mask: [batch_size, 13, 13, 5]
        # target_confs: [batch_size, 13, 13, 5]
        # target_dxdy: [batch_size, 13, 13, 5, 2]
        # target_twth: [batch_size, 13, 13, 5, 2]
        # target_calsses: [batch_size, 13, 13, 5, 20]
        noobj_mask, coord_mask, class_mask, target_confs, target_txty, target_twth, target_classes = self.build_targets(pred_bbox, ground_truth, iou_pred_gt)
        obj_mask = ground_truth[..., 4] > 0

        mse = torch.nn.MSELoss(reduction='sum')
        ce = torch.nn.CrossEntropyLoss(reduction='sum')
        # dxdy_loss: [batch_size, 13, 13, 5, 2]
        dxdy_loss = mse(pred_txty[coord_mask], target_txty[coord_mask]) / opt.batch_size
        # twth_loss: [batch_size, 13, 13, 5, 2]
        twth_loss = mse(pred_twth[coord_mask], target_twth[coord_mask]) / opt.batch_size

        # [batch_szie, 13, 13, 5, 1] -> [batch_size, 13, 13, 5]
        pred_confs = pred_confs.sigmoid().squeeze(dim=-1)
        noobj_mask = ~obj_mask & noobj_mask
        noobj_loss = opt.noobj_scale * mse(pred_confs[noobj_mask], torch.zeros_like(pred_confs)[noobj_mask]) / opt.batch_size
        obj_loss = opt.obj_scale * mse(pred_confs[obj_mask], target_confs[obj_mask]) / opt.batch_size
        conf_loss = noobj_loss + obj_loss

        target_classes = target_classes[class_mask].contiguous().view(-1)  # [N,]
        pred_classes = pred_classes[class_mask].contiguous().view(-1, opt.coco_class_num)  # [N, 20]
        class_loss = ce(pred_classes, target_classes) / opt.batch_size
        total_loss = opt.coord_scale * (dxdy_loss + twth_loss) + conf_loss + opt.class_scale * class_loss

        loss_dict = {'total_loss': total_loss,
                     'dxdy_loss': dxdy_loss,
                     'twth_loss': twth_loss,
                     'conf_loss': conf_loss,
                     'obj_loss': obj_loss,
                     'noobj_loss': noobj_loss,
                     'class_loss': class_loss}

        return loss_dict

    @staticmethod
    def build_targets(pred_bbox, ground_truth, iou_pred_gt):
        """
        :param pred_bbox:
            [batch_size, 13, 13, 5, 4]
        :param ground_truth:
            [batch_size, 13, 13, 5, 25] / last dimension: [dx, dy, tw, th, conf, labels]
        :param iou_pred_gt:
            [batch_size, 13, 13, 5]
        :return:
        """
        grid_size = [pred_bbox.shape[1], pred_bbox.shape[2]]
        # [batch_size, 13, 13, 5]
        noobj_mask = torch.ones(opt.batch_size, grid_size[0], grid_size[1], opt.B, dtype=torch.bool, requires_grad=False, device=opt.device)
        coord_mask = torch.zeros(opt.batch_size, grid_size[0], grid_size[1], opt.B, dtype=torch.bool, requires_grad=False, device=opt.device)
        class_mask = torch.zeros(opt.batch_size, grid_size[0], grid_size[1], opt.B, dtype=torch.bool, requires_grad=False, device=opt.device)
        obj_conf = torch.zeros(opt.batch_size, grid_size[0], grid_size[1], opt.B, dtype=torch.float32, requires_grad=False, device=opt.device)
        obj_txty = torch.zeros(opt.batch_size, grid_size[0], grid_size[1], opt.B, 2, dtype=torch.float32, requires_grad=False, device=opt.device)
        obj_twth = torch.zeros(opt.batch_size, grid_size[0], grid_size[1], opt.B, 2, dtype=torch.float32, requires_grad=False, device=opt.device)
        obj_class = torch.zeros(opt.batch_size, grid_size[0], grid_size[1], opt.B, dtype=torch.long, requires_grad=False, device=opt.device)

        # [batch_size, 13, 13, 5]
        obj_mask = ground_truth[..., 4] > 0.

        for b in range(ground_truth.size(0)):
            # 每个cell的所有(5个)预测框与所有gt_bbox计算iou，最大iou值小于阈值的预测框标记为background并计算conf损失，最大iou大于阈值的cell不计入损失函数的计算
            # [M, 4]
            valid_bbox = ground_truth[b, ..., :4][obj_mask[b]]
            # [13, 13, 5, 4] & [M, 4] -> [13, 13, 5, M]
            iou_pred_valid = yolov2_bbox_iou(pred_bbox[b], valid_bbox)
            # [13, 13, 5]
            max_iou = torch.max(iou_pred_valid, dim=-1)[0]
            # [13, 13, 5] / bool
            tmp_mask = torch.ge(max_iou, opt.match_iou_threshold)
            # 不包含目标的cell的所有预测框中，预测box和所有gt_box的iou小于0.6的计算conf损失；对于包含目标的cell的预测框的conf下面的code会进行处理
            noobj_mask[b][tmp_mask] = False

            # 存在目标的cell中，最大iou对应的那个预测框负责预测并计算该预测框的bbox损失，conf损失以及分类损失
            coord_mask[b][obj_mask[b]] = True
            class_mask[b][obj_mask[b]] = True
            if opt.rescore:
                obj_conf[b][obj_mask[b]] = iou_pred_gt[b][obj_mask[b]]
            else:
                obj_conf[b][obj_mask[b]] = 1.
            obj_txty[b][obj_mask[b]] = ground_truth[b][obj_mask[b]][:, :2]
            obj_twth[b][obj_mask[b]] = ground_truth[b][obj_mask[b]][:, 2:4]
            # obj_class[b][obj_mask[b]] = ground_truth[b][obj_mask[b]][:, 5:]
            obj_class[b][obj_mask[b]] = torch.argmax(ground_truth[b][obj_mask[b]][:, 5:], dim=-1)

        return noobj_mask, coord_mask, class_mask, obj_conf, obj_txty, obj_twth, obj_class

    @staticmethod
    def bbox_iou(bbox1, bbox2):
        """
        :param bbox1:
            [13, 13, 5, 4] / [x, y, w, h];
        :param bbox2:
            [13, 13, 5, 4] / [x, y, w, h];
        :return:
            [13, 13, 5];
        """
        bbox1_area = bbox1[..., 2] * bbox1[..., 3]
        bbox2_area = bbox2[..., 2] * bbox2[..., 3]
        # assert bbox1.shape == bbox2.shape
        bbox2 = xywh2xyxy(bbox2)
        bbox1 = xywh2xyxy(bbox1)
        # [13, 13, 5] & [13, 13, 5] -> [13, 13, 5]
        intersection_xmin = torch.max(bbox1[..., 0], bbox2[..., 0])
        intersection_ymin = torch.max(bbox1[..., 1], bbox2[..., 1])
        intersection_xmax = torch.min(bbox1[..., 2], bbox2[..., 2])
        intersection_ymax = torch.min(bbox1[..., 3], bbox2[..., 3])
        # [13, 13, 5] & [13, 13, 5] -> [13, 13, 5]
        intersection_w = torch.max(intersection_xmax - intersection_xmin, torch.tensor(0., device=opt.device))
        intersection_h = torch.max(intersection_ymax - intersection_ymin, torch.tensor(0., device=opt.device))
        intersection_area = intersection_w * intersection_h
        # [13, 13, 5] & ([13, 13, 5] & [13, 13, 5] & [13, 13, 5]) -> [13, 13, 5]
        ious = intersection_area / (bbox1_area + bbox2_area - intersection_area + 1e-10)
        # ious shape: [13, 13, 5]
        return ious

    @staticmethod
    # 增大模型比较没有把握的case的损失，减小那些比较有把握case的损失
    def focal_loss(labels, preds):
        alpha = 2.
        gamma = 1.
        # [batch_size,13,13,5,1] & [batch_size,13,13,5,1] -> [batch_size,13,13,5,1]
        focal_mask = alpha * torch.pow(torch.abs(labels - torch.sigmoid(preds)), gamma)
        return focal_mask

    @staticmethod
    def smooth_labels(labels, class_num):
        delta = 0.01
        smoothness = (1 - delta) * labels + delta * (1. / class_num)
        return smoothness

    def init_optimizer(self):
        params = [{'params': self.yolov2.pth_layer.parameters(), 'lr': opt.pth_lr},
                  {'params': self.yolov2.mid_layer.parameters(), 'lr': opt.yolo_lr},
                  {'params': self.yolov2.output_layer.parameters(), 'lr': opt.yolo_lr}]
        if opt.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(params=params, lr=opt.lr,  momentum=opt.optimizer_momentum,
                                        weight_decay=opt.optimizer_weight_decay)
        else:
            optimizer = torch.optim.Adam(params=params, lr=opt.lr, weight_decay=opt.optimizer_weight_decay)
        return optimizer

    def save(self, epoch, steps, loss, save_path):
        model_state = self.yolov2.state_dict()
        optimizer_state = self.optimizer.state_dict()
        state_dict = {
            'model': model_state,
            'optimizer': optimizer_state,
            'epoch': epoch,
            'total_steps': steps,
            'loss': loss}
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(obj=state_dict, f=save_path)
        if 'best' in save_path:
            self.logger.info(f'Best model has been saved: {save_path}')
        else:
            self.logger.info(f'model has been saved: {save_path}')

    def adjust_lr(self, epoch):

        def set_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.optimizer_type == 'SGD':
            if epoch < 5:
                set_lr(self.optimizer, 1e-5)
            elif 5 <= epoch < 45:
                set_lr(self.optimizer, 1e-4)
            elif 45 <= epoch < 155:
                set_lr(self.optimizer, 1e-6)
            else:
                set_lr(self.optimizer, opt.lr * 0.1 ** (epoch // 50))
        else:
            if epoch < 50:
                set_lr(self.optimizer, 1e-3)
            elif 50 <= epoch < 150:
                set_lr(self.optimizer, 1e-4)
            else:
                set_lr(self.optimizer, opt.lr * 0.05 ** (epoch // 30))






