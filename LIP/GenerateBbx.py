# -*- coding: utf-8 -*-
'''
功能：
    获取语义信息，得到bouding box标签，只处理图片中只存在单个人的情况。（负责FashionDataset数据）
语义标签信息：
    0:'null'
    1:'hat'
    2:'head'
    3:'sunglass'
    4:'upperclothes'
    5:'skirt'
    6:'pants'
    7:'dress'
    8:'belt'
    9:'left-shoe'
    10:'right-shoe'
    11:'face'
    12:'left-leg'
    13:'right-leg'
    14:'left-arm'
    15:'right-arm'
    16:'bag'
    17:'scarf'
bounding box标签信息：
    0 background
    1 hat
    2 sunglass
    3 upperclothes
    4 skirt
    5 pants
    6 dress
    7 belt
    8 shoe
    9 bag
    10 scarf
'''

# BGR Color Space
ColorSpace = [[  0,   0,   0],[193, 182, 255],[ 60,  20, 220],[245, 240, 255],[180, 105, 255],[147,  20, 255],\
[133,  21, 199],[214, 112, 218],[255,   0, 255],[139,   0, 139],[211,   0, 148]]
# [0,255,127],[50,205,50],[85,107,47],[218,165,32],[255,215,0],[0,0,255],[0,255,255]]




import cv2
import os
import os.path as osp
import numpy as np
import math

SegIndexName = {0:'null',1:'hat',2:'head',3:'sunglass',4:'upperclothes',\
                5:'skirt',6:'pants',7:'dress',8:'belt',9:'left-shoe',\
                10:'right-shoe',11:'face',12:'left-leg',13:'right-leg',\
                14:'left-arm',15:'right-arm',16:'bag',17:'scarf'}

BbxIndexName = {0:'background',1:'hat',2:'sunglass',3:'upperclothes',4:'skirt',\
                5:'pants',6:'dress',7:'belt',8:'shoe',9:'bag',10:'scarf'}

FashionId = [1,3,4,5,6,7,8,9,10,16,17] 
LabelId =   [1,2,3,4,5,6,7,8,8,9,10]

ImgBbx = {}

SegPath = './SegmentationClassAug'
ImgPath = './JPEGImages'

if not osp.exists('./BoundingBoxImg'):
    os.mkdir('./BoundingBoxImg')

SavePath = './BoundingBoxImg'

SegName = os.listdir(SegPath)

for name in SegName:
    print name
    Seg = cv2.imread(osp.join(SegPath, name))
    ImgName = name.split('.')[0]+'.jpg'
    Img = cv2.imread(osp.join(ImgPath, ImgName))
    # cv2.imshow("tsq", Img)
    # cv2.waitKey(2000)
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
    ImgBbx[name] = CurBbx
    for bbx in CurBbx:
        print bbx[1],bbx[2],bbx[3],bbx[4]
        cv2.rectangle(Img, (bbx[1],bbx[2]), (bbx[3],bbx[4]), tuple(ColorSpace[bbx[0]]), 2)
        # cv2.putText(Img,str(bbx[0]),(bbx[1],bbx[2]),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)

    cv2.imwrite(osp.join(SavePath, ImgName), Img)

