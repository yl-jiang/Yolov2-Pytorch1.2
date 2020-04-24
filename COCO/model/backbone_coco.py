#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/5 下午5:11
# @Author  : jyl
# @File    : backbone_coco.py
import torch
from model import make_layers, init_model_variables
from config import opt

torch.set_default_dtype(torch.float32)


class DarkNet19(torch.nn.Module):

    def __init__(self):
        super(DarkNet19, self).__init__()
        # [32, 3, 1, 1] represent the parameters in torch.nn.Conv2d:
        # out_channels = 32
        # kernel_size = 3
        # stride = 1
        # padding = 1
        cfg = [[32, 3, 1, 1], 'M',  # 208x208x32
               [64, 3, 1, 1], 'M',  # 104x104x64
               [128, 3, 1, 1], [64, 1, 1, 0], [128, 3, 1, 1], 'M',  # 52x52x128
               [256, 3, 1, 1], [128, 1, 1, 0], [256, 3, 1, 1], 'M',  # 26x26x256
               [512, 3, 1, 1], [256, 1, 1, 0], [512, 3, 1, 1], [256, 1, 1, 0], [512, 3, 1, 1],  # residual:26x26x512
               'M', [1024, 3, 1, 1], [512, 1, 1, 0], [1024, 3, 1, 1], [512, 1, 1, 0],   # 13x13x512
               [1024, 3, 1, 1], [1024, 3, 1, 1], [1024, 3, 1, 1]]  # 13x13x1024

        self.front_layer = make_layers(cfg[:17], in_channels=3)
        self.rear_layer = make_layers(cfg[17:], in_channels=512)

    def forward(self, x):
        feature = self.front_layer(x)
        residual = feature
        feature = self.rear_layer(feature)
        return residual, feature


class BackboneCOCO(torch.nn.Module):
    def __init__(self):
        super(BackboneCOCO, self).__init__()
        self.darknet19 = DarkNet19()

        total_bbox_num = (5 + opt.coco_class_num) * opt.anchor_num
        yolov2_cfg = [[64, 1, 1, 0],  # 26x26x64
                      [1024, 3, 1, 1]]  # 13x13x1024
        # 13x13x125
        # passthrough layer
        self.pth_layer = make_layers(yolov2_cfg[:1], in_channels=512)
        # predict layer
        self.mid_layer = make_layers(yolov2_cfg[1:], in_channels=1280)
        # [total_bbox_num, 1, 1, 0]
        self.output_layer = torch.nn.Conv2d(in_channels=1024, out_channels=total_bbox_num, kernel_size=1, stride=1, padding=0, bias=False)
        init_model_variables(self)

    def forward(self, x):
        residual, feature = self.darknet19(x)
        residual = self.pth_layer(residual)  # 26x26x64
        batch_size, num_channel, height, width = residual.size()
        # [batch_size, 64, 26, 26] -> [batch_size, 16, 26, 2, 26, 2]
        residual = residual.reshape(batch_size, num_channel//4, height, 2, width, 2)
        # [batch_size, 16, 26, 2, 26, 2] -> [batch_size, 2, 2, 16, 26, 26]
        residual = residual.permute(0, 3, 5, 1, 2, 4)
        # [batch_size, 2, 2, 16, 26, 26] -> [batch_size, 256, 13, 13]
        residual = residual.reshape(batch_size, -1, height//2, width//2)
        # [batch_size, 1024+256, 13, 13]
        feature = torch.cat([residual, feature], dim=1)
        # [batch_size, 1024, 13, 13]
        feature = self.mid_layer(feature)
        # [batch_size, 125, 13, 13]
        output = self.output_layer(feature)
        # [batch_szie, 13, 13, 425]
        output = output.permute(0, 2, 3, 1)
        return output


if __name__ == '__main__':
    y = BackboneCOCO()
    for m in y.modules():
        print(m)


