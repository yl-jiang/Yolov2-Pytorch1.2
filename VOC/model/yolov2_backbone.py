#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/21 下午3:51
# @Author  : jyl
# @File    : yolov2_backbone.py
import torch
import torch.nn as nn
import math


class DarkNet19(nn.Module):
    """
    backbone network:
        input: raw img
        output: feature map of shape [13, 13, 1024]
    """

    def __init__(self):
        super(DarkNet19, self).__init__()
        self.feature_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),  # 416, 416, 32
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 208, 208, 32

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # 208, 208, 64
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 104, 104, 64

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 104, 104, 128
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),  # 104, 104, 64
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 104, 104, 128
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 52, 52, 128

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 52, 52, 256
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),  # 52, 52, 128
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 52, 52, 256
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 26, 26, 256

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 26, 26, 512
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),  # 26, 26, 256
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 26, 26, 512
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),  # 26, 26, 256
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.route = torch.nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 26, 26, 512
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.feature_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 13, 13, 512
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 13, 13, 1024
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),  # 13, 13, 512
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 13, 13, 1024
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),  # 13, 13, 512
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 13, 13, 1024
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        feature = self.feature_1(x)  # 26, 26, 256
        route = self.route(feature)  # 26, 26, 512
        feature = self.feature_2(route)  # 13, 13, 1024
        return route, feature


class Yolov2(nn.Module):
    """
    yolov2 network:
        input: image　with shape [416, 416]
        output: features for detection/ shape: [13, 13, 125]
    """

    def __init__(self):
        super(Yolov2, self).__init__()
        total_bbox_num = opt.anchor_num * opt.B + opt.class_num
        self.darknet19 = DarkNet19()
        self.passthrough = torch.nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0),   # 26, 26, 64
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1),   # 13, 13, 256
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.yolov2 = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 13, 13, 1024
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=total_bbox_num, kernel_size=1, stride=1, padding=0),  # 13, 13, 125
            nn.BatchNorm2d(num_features=total_bbox_num),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        route, feature = self.darknet19(x)  # 26, 26, 512 / 13, 13, 1024
        route = self.passthrough(route)  # 13, 13, 256
        feature = torch.cat([route, feature], dim=-1)  # 13, 13, 1280
        output = self.yolov2(feature)  # 13, 13, 125
        return output


if __name__ == '__main__':
    yolov2 = Yolov2()

