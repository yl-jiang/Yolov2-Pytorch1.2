#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/22 下午1:42
# @Author  : jyl
# @File    : alias_method.py

# 函数功能说明：
#   输入：由各个概率值组成的list(概率之和必须等于1):[0.1, 0.5, 0.2, ...]; 采样次数Ｎ
#   输出：在Ｎ次采样中０出现的概率(从采样结果上来说是频率)为0.1,１出现的概率为0.5, ...

import numpy as np


def alias_setup(probs):
    """
    probs： 某个概率分布
    返回: Alias数组与Prob数组
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger = []
    for i, prob in enumerate(probs):
        q[i] = K * prob  # 概率
        if q[i] < 1.0:
            smaller.append(i)
        else:
            larger.append(i)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    输入: Prob数组和Alias数组
    输出: 一次采样结果
    '''
    K = len(J)
    # Draw from the overall uniform mixture.
    k = int(np.floor(np.random.rand() * K))  # 随机取一列

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if np.random.rand() < q[k]:
        return k
    else:
        return J[k]


def alias_sample(probs, samples):
    assert isinstance(samples, int), 'Samples must be a integer.'
    sample_result = []
    J, p = alias_setup(probs)
    for i in range(samples):
        sample_result.append(alias_draw(J, p))
    return sample_result
