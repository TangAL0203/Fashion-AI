#-*-coding:utf-8-*-
import cv2
import numpy as np
import os
import os.path as osp

'''
功能：
    读取颜色很暗的(颜色值很小)的语义标签图像，然后加上彩色配色重新显示
'''

# RGB Color Space
ColorSpace = [[0,0,0],[255,182,193],[220,20,60],[255,240,245],[255,105,180],\
[255,20,147],[199,21,133],[218,112,214],[255,0,255],[139,0,139],\
[148,0,211],[75,0,130],[123,104,238],[0,0,255],[0,255,255],\
[0,255,127],[50,205,50],[85,107,47],[218,165,32],[255,215,0]]

ReadRootPath = 'D:/DataSet/LIP/FashionDataset/humanparsing/SegmentationClassAug'
SaveRootPath = 'D:/DataSet/LIP/FashionDataset/humanparsing/ColorSegmentationClassAug'

SegName = os.listdir(ReadRootPath)

for name in SegName:
    Seg = cv2.imread(osp.join(ReadRootPath, name))
    H, W, C = Seg.shape
    for i in range(H):
        for j in range(W):
            Seg[i,j,:] = np.array(ColorSpace[Seg[i,j,0]])[::-1]
    SavePath = osp.join(SaveRootPath, name)
    cv2.imwrite(SavePath, Seg)




