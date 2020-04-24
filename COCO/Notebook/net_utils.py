#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/30 下午3:14
# @Author  : jyl
# @File    : net_utils.py

import torch


def make_layers(cfg, in_channels, batch_norm=True):
    layers = []
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=v[0],
                                           kernel_size=v[1], stride=v[2], padding=v[3], bias=False)
            acvt_layer = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
            if batch_norm:
                bn_layer = torch.nn.BatchNorm2d(num_features=v[0])
                layers += [conv2d_layer, bn_layer, acvt_layer]
            else:
                layers += [conv2d_layer, acvt_layer]
            in_channels = v[0]

    return torch.nn.Sequential(*layers)


def init_model_variables(model):
    print("Initialize model's weights ...")
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

