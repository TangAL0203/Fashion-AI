#-*-coding:utf-8-*-
import cv2
import numpy as np
import os.path as osp

'''
功能：
    显示纯彩色图像
'''

# ImgPath = ''
# SegPath = ''

# Img = cv2.imread(ImgPath)
# Seg = cv2.imread(SegPath)

# define color space, total 20 classes
# 2 x 2 x 5 = 20

# ColorSpace = []

# opencv: color order: BGR
# BColor = [0,128]
# GColor = [0,128]
# RColor = [0,63,126,189,252]

# for i in range(2):
#     B = BColor[i]
#     for j in range(2):
#         G = GColor[j]
#         for k in range(5):
#             R = RColor[k]
#             ColorSpace.append([B,G,R])

# refer link: http://www.sioe.cn/yingyong/yanse-rgb-16/
# ColorSpace = [RGB]
ColorSpace = [[0,0,0],[255,182,193],[220,20,60],[255,240,245],\
[255,105,180],[255,20,147],[199,21,133],[218,112,214],\
[255,0,255],[139,0,139],[148,0,211],[75,0,130],[123,104,238],\
[0,0,255],[0,255,255],[0,255,127],[50,205,50],[85,107,47],[218,165,32],[255,215,0]]

# display color img
for i in range(len(ColorSpace)):
    color1 = np.ones((224,224,1))*ColorSpace[i][2]
    color2 = np.ones((224,224,1))*ColorSpace[i][1]
    color3 = np.ones((224,224,1))*ColorSpace[i][0]
    img = np.concatenate((color1, color2, color3), axis=2)
    # cv2.imwrite(osp.join('./',str(i)+'.jpg'), img)
    cv2.imshow("Color Examples", img)
    cv2.waitKey(500)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     break
    # elif cv2.waitKey(0) & 0xFF == ord('c'):
    #     continue