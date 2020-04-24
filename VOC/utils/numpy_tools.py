#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 下午3:21
# @Author  : jyl
# @File    : numpy_utils.py
import numpy as np


def fill_nan(x, value=0.):
    assert isinstance(x, np.ndarray), 'the dtype of inpute<x> must be np.ndarray.'
    if np.isnan(np.sum(x)):
        output = x.copy()
        nan_index = np.isnan(x)
        output[nan_index] = value
        return output
    else:
        return x


