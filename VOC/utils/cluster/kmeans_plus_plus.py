#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/22 上午10:45
# @Author  : jyl
# @File    : kmeans_plus_plus.py
import numpy as np
from utils import alias_sample
import collections


def kmeans_plus_plus(dataset, k):
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset)
    center_ids = choose_centers(dataset, k)
    centers = dataset[center_ids]
    classes_before = np.arange(len(dataset))
    while True:
        classes_after = do_cluster(dataset, centers)
        if (classes_before == classes_after).all():
            break

        classes_before = classes_after
        for c in range(k):
            data_c = dataset[np.argwhere(classes_after == c)]
            center_c = np.mean(data_c, axis=0)
            centers[c] = center_c

    return centers, classes_after


def choose_centers(dataset, k):
    center_ids = [np.random.choice(len(dataset), size=1)]
    dist_mat = np.empty(shape=[len(dataset), len(dataset)])
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            if i == j:
                dist_mat[i, j] = 0.
            elif i < j:
                dist_mat[i, j] = np.mean(np.square(dataset[i] - dataset[j]))
            else:
                dist_mat[i, j] = dist_mat[j, i]
    while len(center_ids) < k:
        nodes_min_dist = np.min(dist_mat[:, center_ids], axis=1)
        probs = nodes_min_dist / np.sum(nodes_min_dist)
        center_ids.append(alias_sample(probs.reshape(-1), 1))
    center_ids = np.array(center_ids).reshape(-1)
    return center_ids


def do_cluster(dataset, centers):
    dist = []
    for center in centers:
        dist.append(np.mean(np.square(dataset - center), axis=1))
    dist = np.vstack(dist)
    classes = np.argmin(dist, axis=0)
    return classes


def show_result(class_list, raw_data, center_coordinate):
    colors = [
              '#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#228B22',
              '#0000FF', '#FF1493', '#EE82EE', '#000000', '#FFA500',
              '#00FF00', '#006400', '#00FFFF', '#0000FF', '#FFFACD',
              ]

    # 画最终聚类效果图
    use_color = {}
    total_color = list(dict(collections.Counter(class_list)).keys())
    for index, i in enumerate(total_color):
        use_color[i] = index
    plt.figure(num=1, figsize=(16, 9))
    for index, point in enumerate(class_list):
        plt.scatter(x=raw_data[index, 0], y=raw_data[index, 1], c=colors[use_color[point]], s=50, marker='o', alpha=0.9)
    plt.scatter(x=center_coordinate[:, 0], y=center_coordinate[:, 1], c='b', s=200, marker='+', alpha=0.8)
    plt.title('K-means++')
    plt.savefig('./kmeans++_result.jpg')
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_path = '/media/dk/MyFiles/Data/clustering/Aggregation.txt'
    data = np.loadtxt(data_path, delimiter='	', usecols=[0, 1], dtype=np.float32)
    centers, classes = kmeans_plus_plus(data, 7)
    show_result(classes, data, centers)



