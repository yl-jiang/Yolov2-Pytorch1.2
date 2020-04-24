#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/30 下午3:07
# @Author  : jyl
# @File    : xml_utils.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/31 16:28
# @Author  : jyl
# @File    : extract_xml.py
import os
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from config import opt
import lxml
from tqdm import tqdm
from box_utils import resize_bbox


VOC_BBOX_LABEL_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def read_image(path, dtype=np.float32, color=True):
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:  # 灰度图片
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:   # 彩色图片
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def parse_voc2012_xml(xml_file, is_training):
    one_file_bboxes = []
    one_file_labels = []
    soup = BeautifulSoup(open(xml_file), 'lxml')
    img_file_name = soup.find('filename').string
    size_soup = soup.find_all('size')
    assert len(size_soup) == 1, 'one xml file must have only one image size record.'
    width = int(float(size_soup[0].find('width').string))
    heigth = int(float(size_soup[0].find('height').string))
    img_shape = (width, heigth)
    for obj in soup.find_all('object'):
        if is_training:
            diffcult = int(obj.find('difficult').string)
            if diffcult == 1:
                continue
        name = obj.find('name').string
        if name in VOC_BBOX_LABEL_NAMES:
            label = VOC_BBOX_LABEL_NAMES.index(name)
            bndbox_obj = obj.find('bndbox', recursive=False)
            y1 = int(float(bndbox_obj.find('ymax').string))
            x1 = int(float(bndbox_obj.find('xmax').string))
            y2 = int(float(bndbox_obj.find('ymin').string))
            x2 = int(float(bndbox_obj.find('xmin').string))
            one_file_bboxes.append([y1, x1, y2, x2])
            one_file_labels.append(label)

    return img_file_name, one_file_bboxes, one_file_labels, img_shape


def xml2txt(voc_data_dir, train_path, test_path, is_resize=False):
    tr_AnnotationsPath = os.path.join(voc_data_dir, 'VOC2012train', 'VOCdevkit', 'VOC2012', 'Annotations')
    te_AnnotationsPath = os.path.join(voc_data_dir, 'VOC2012test', 'VOCdevkit', 'VOC2012', 'Annotations')
    tr_xml_fils = os.listdir(tr_AnnotationsPath)
    te_xml_fils = os.listdir(te_AnnotationsPath)

    tr_writer = open(train_path, 'a')
    te_writer = open(test_path, 'a')

    # training data
    for f in tqdm(tr_xml_fils):
        xml_path = os.path.join(tr_AnnotationsPath, f)
        img_file_name, bboxes, labels, img_shape = parse_voc2012_xml(xml_path, True)
        if len(labels) == 0:
            continue
        if is_resize:
            bboxes = resize_bbox(bboxes, img_shape, [opt.img_size, opt.img_size])
        tr_writer.write(img_file_name)
        for bbox, label in zip(bboxes, labels):
            tr_writer.write(' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(label))
        tr_writer.write('\n')
    tr_writer.close()

    # testing data
    for f in tqdm(te_xml_fils):
        xml_path = os.path.join(te_AnnotationsPath, f)
        img_file_name, bboxes, labels, img_shape = parse_voc2012_xml(xml_path, False)
        if len(labels) == 0:
            continue
        if is_resize:
            bboxes = resize_bbox(bboxes, img_shape, [opt.img_size, opt.img_size])
        te_writer.write(img_file_name)
        for bbox, label in zip(bboxes, labels):
            te_writer.write(' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(label))
        te_writer.write('\n')
    te_writer.close()


if __name__ == '__main__':
    # test all data parser
    # xml2txt(opt.vocdata_path, opt.traindata_path, opt.testdata_path, False)
    # xml2txt(opt.vocdata_path, opt.traindata_resize_path, opt.testdata_resize_path, True)
    parse_voc2012_xml('/media/dk/MyFiles/Data/VOC/VOC2012train/VOCdevkit/VOC2012/Annotations/2008_005145.xml', True)


