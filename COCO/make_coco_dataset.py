#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/6 下午12:38
# @Author  : jyl
# @File    : make_coco_dataset.py
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import copy
import pickle


dataDir = '/home/dk/jyl/Data/COCO2017/train_val_ann'
dataType = 'train2017'
annFile = f'{dataDir}/annotations/instances_{dataType}.json'
# 创建COCO API
coco = COCO(annFile)

# categories数组元素的数量为80（2017年）
print('categories: ', len(coco.dataset['categories']))
# annotations数组元素的数量等同于训练集（或者测试集）中bounding box的数量
print('bboxs num : ', len(coco.dataset['annotations']))
# images数组元素的数量等同于划入训练集（或者测试集）的图片的数量
print('imgs num  : ', len(coco.dataset['images']))

catIds_ = coco.getCatIds()
print('COCO categories id:\n', coco.getCatIds(), '\n')
cats = coco.loadCats(catIds_)
catNames = [cat['name'] for cat in cats]
print(f"COCO categories: \n{';'.join(catNames)}\n")
supercatNames = set([cat['supercategory'] for cat in cats])
print(f"COCO supercategories: \n{';'.join(supercatNames)}\n")

id2cat_dict = dict(zip(catIds_, catNames))
cat2id_dict = dict(zip(catNames, catIds_))
id2cat_dict_resort = dict(zip(range(len(cat2id_dict)), cat2id_dict.keys()))
cat2id_dict_resort = dict(zip(cat2id_dict.keys(), range(len(cat2id_dict))))
print(id2cat_dict)
print('\n')
print(cat2id_dict)
print('\n')
print(id2cat_dict_resort)
print('\n')
print(cat2id_dict_resort)

all_anns = coco.dataset['annotations']
img_ids = []
for ann in all_anns:
    img_ids.append(ann['image_id'])
print(len(img_ids))
print(len(set(img_ids)))
img_ids = set(img_ids)


list_to_store = []
anns_copy = copy.deepcopy(all_anns)
for img_id in tqdm(img_ids):
    tmp_bbox = []
    tmp_catId = []
    tmp_catNam = []
    for i, ann in enumerate(anns_copy):
        if img_id == ann['image_id']:
            tmp_bbox.append(ann['bbox'])
            catNam = id2cat_dict[ann['category_id']]
            tmp_catNam.append(catNam)
            tmp_catId.append(cat2id_dict_resort[catNam])
    tmp_dict = {'image_id': img_id,
                'categories': tmp_catNam,
                'categories_id': np.asarray(tmp_catId),
                'bbox': np.asarray(tmp_bbox)}
#     print(tmp_dict)
    list_to_store.append(tmp_dict)

pickle.dump(list_to_store, open('/home/dk/Desktop/coco2017_train_v2.pkl', 'wb'))
