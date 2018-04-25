# -*- coding: utf-8 -*-
import os
import os.path as osp
import sys
import cv2
from itertools import islice
from xml.dom.minidom import Document
import numpy as np
import random

'''
功能：
    读取语义标签图像，生成包含bounding box标签信息的txt文件
'''

BbxIndexName = {0:'background',1:'hat',2:'sunglass',3:'upperclothes',4:'skirt',\
                5:'pants',6:'dress',7:'belt',8:'shoe',9:'bag',10:'scarf'}

SegIndexName = {0:'null',1:'hat',2:'head',3:'sunglass',4:'upperclothes',\
                5:'skirt',6:'pants',7:'dress',8:'belt',9:'left-shoe',\
                10:'right-shoe',11:'face',12:'left-leg',13:'right-leg',\
                14:'left-arm',15:'right-arm',16:'bag',17:'scarf'}

FashionId = [1,3,4,5,6,7,8,9,10,16,17] 
LabelId =   [1,2,3,4,5,6,7,8,8,9,10]

SegPath = './SegmentationClassAug'
ImgRoot = 'JPEGImages'

SegName = os.listdir(SegPath)
Total = len(SegName)
random.shuffle(SegName)
TrainRatio = 0.6
ValTatio = 0.4
TrainName = SegName[0:int(len(SegName)*TrainRatio)]
ValName = SegName[int(len(SegName)*TrainRatio):]

'''
    File name eg 'JPEGImages/997_1.jpg'
    Numbers W H C eg 5 224 224 3
    Bounding box
        ClassId xmin ymin xmax ymax
'''
CurNum = 0
f_train = open('./FashionDataset_train_bbx_gt.txt', 'w')
f_val = open('./FashionDataset_val_bbx_gt.txt', 'w')


for name in TrainName:
    Seg = cv2.imread(osp.join(SegPath, name))
    H, W, C = Seg.shape
    CurBbx = []
    for ind, i in enumerate(FashionId):
        clss = LabelId[ind]
        temp = [-1,W,H,0,0] # class, x_min, y_min, x_max, y_max
        for j in range(H):
            for k in range(W):
                if Seg[j,k,0] == i:
                    temp[0] = clss
                    if j<temp[2]:
                        temp[2] = j
                    if j>temp[4]:
                        temp[4] = j
                    if k<temp[1]:
                        temp[1] = k
                    if k>temp[3]:
                        temp[3] = k
        if temp[0]>0:
            CurBbx.append(temp)
    # replace suffix of name with .jpg
    CurNum = CurNum+1
    name = name.split('.')[0] + '.jpg'
    print name,"  done: ", str(CurNum), "  total:  ",str(Total)
    f_train.writelines(osp.join(ImgRoot, name)+'\n')
    f_train.writelines(str(len(CurBbx))+' '+str(W)+' '+str(H)+' '+str(C)+'\n')
    for bbx in CurBbx:
        f_train.writelines(str(bbx[0])+' '+str(bbx[1])+' '+str(bbx[2])+' '+str(bbx[3])+' '+str(bbx[4])+'\n')

f_train.close()

for name in ValName:
    Seg = cv2.imread(osp.join(SegPath, name))
    H, W, C = Seg.shape
    CurBbx = []
    for ind, i in enumerate(FashionId):
        clss = LabelId[ind]
        temp = [-1,W,H,0,0] # class, x_min, y_min, x_max, y_max
        for j in range(H):
            for k in range(W):
                if Seg[j,k,0] == i:
                    temp[0] = clss
                    if j<temp[2]:
                        temp[2] = j
                    if j>temp[4]:
                        temp[4] = j
                    if k<temp[1]:
                        temp[1] = k
                    if k>temp[3]:
                        temp[3] = k
        if temp[0]>0:
            CurBbx.append(temp)
    # replace suffix of name with .jpg
    CurNum = CurNum+1
    name = name.split('.')[0] + '.jpg'
    print name, "  done: ", str(CurNum), "  total:  ",str(Total)
    f_val.writelines(osp.join(ImgRoot, name)+'\n')
    f_val.writelines(str(len(CurBbx))+' '+str(W)+' '+str(H)+' '+str(C)+'\n')
    for bbx in CurBbx:
        f_val.writelines(str(bbx[0])+' '+str(bbx[1])+' '+str(bbx[2])+' '+str(bbx[3])+' '+str(bbx[4])+'\n')

f_val.close()



