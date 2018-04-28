#-*-coding:utf-8-*-
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ['hat', 'sunglass', 'upperclothes', 'skirt', 'pants', 'dress', 'belt', 'shoe', 'bag', 'scarf']

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0 # bbx的x中心点坐标
    y = (box[2] + box[3])/2.0 # bbx的y中心点坐标
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw # 归一化中心点坐标x
    w = w*dw # 归一化bbx宽
    y = y*dh # 归一化中心点坐标y
    h = h*dh # 归一化bbx高
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open('./FashionDataset/Annotations/trainval/%s.xml'%(image_id))
    out_file = open('./FashionDataset/labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls) # 标签序号从0开始
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()

# 先生成fulTrainPath.txt
if not os.path.exists('./FashionDataset/labels/'):
    os.makedirs('./FashionDataset/labels/')
image_ids = open('./FashionDataset/train.txt').read().strip().split()
list_file = open('./FashionDataset/fulTrainPath.txt', 'w')
for image_id in image_ids:
    list_file.write('%s/FashionDataset/JPEGImages/%s.jpg\n'%(wd, image_id))
    convert_annotation(image_id)
# fulTrainTest.txt
image_ids = open('./FashionDataset/test.txt').read().strip().split()
list_file = open('./FashionDataset/fulTrainTest.txt', 'w')
for image_id in image_ids:
    list_file.write('%s/FashionDataset/JPEGImages/%s.jpg\n'%(wd, image_id))
    convert_annotation(image_id)