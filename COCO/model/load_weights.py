#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/2 下午6:30
# @Author  : jyl
# @File    : load_weights.py
import torch


def load_weights(model, ckpt_path, reinit_last=False):
    trained_yolov2 = torch.load(ckpt_path)
    trained_values = list(trained_yolov2.values())

    idx = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            assert m.weight.shape == trained_values[idx].shape
            m.weight.data = trained_values[idx]
            idx += 1
        if isinstance(m, torch.nn.BatchNorm2d):
            assert m.weight.shape == trained_values[idx].shape
            m.weight.data = trained_values[idx]
            idx += 1
            assert m.bias.shape == trained_values[idx].shape
            m.bias.data = trained_values[idx]
            idx += 1
            assert m.running_mean.shape == trained_values[idx].shape
            m.running_mean.data = trained_values[idx]
            idx += 1
            assert m.running_var.shape == trained_values[idx].shape
            m.running_var.data = trained_values[idx]
            idx += 1
    # random initialize the last layer's weight
    if reinit_last:
        torch.nn.init.normal_(list(model.modules())[-1].weight, mean=0., std=0.01)

    state_dict = {'model': model.state_dict()}
    torch.save(state_dict, '/home/dk/jyl/V2/COCO/model/ckpt/darknet19.pkl')
    print(f'Load weights from pretrained model:{ckpt_path}')


if __name__ == '__main__':
    pass