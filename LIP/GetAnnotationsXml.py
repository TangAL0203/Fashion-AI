# -*- coding: utf-8 -*-
'''
功能：
    获取语义标签，按照PASCAL VOC格式转为xml文件

ClassId:
    BbxIndexName = {0:'background',1:'hat',2:'sunglass',3:'upperclothes',4:'skirt',\
                5:'pants',6:'dress',7:'belt',8:'shoe',9:'bag',10:'scarf'}

xml标签格式：
    annotation
        folder
        filename
        path
        source
        size
            width
            height
            depth
        segmented
        object1
            name eg clothes.
            pose eg Unspecified.
            truncated eg 0.
            difficult eg 0.
            bndbox
                xmin eg 176.
                ymin eg 129.
                xmax eg 347.
                ymax eg 383.
        object2
            name
            pose
            truncated
            difficult
            bndbox
                xmin
                ymin
                xmax
                ymax
'''

import os
import os.path as osp
import sys
import cv2
from itertools import islice
from xml.dom.minidom import Document
import numpy as np

BbxIndexName = {0:'background',1:'hat',2:'sunglass',3:'upperclothes',4:'skirt',\
                5:'pants',6:'dress',7:'belt',8:'shoe',9:'bag',10:'scarf'}

SegIndexName = {0:'null',1:'hat',2:'head',3:'sunglass',4:'upperclothes',\
                5:'skirt',6:'pants',7:'dress',8:'belt',9:'left-shoe',\
                10:'right-shoe',11:'face',12:'left-leg',13:'right-leg',\
                14:'left-arm',15:'right-arm',16:'bag',17:'scarf'}

FashionId = [1,3,4,5,6,7,8,9,10,16,17] 
LabelId =   [1,2,3,4,5,6,7,8,8,9,10]

SegPath = './SegmentationClassAug'
AnnotationsPath = './Annotations'


SegName = os.listdir(SegPath)

if not osp.exists('./Annotations'):
    os.mkdir('./Annotations')

def isValidBox(bbx, width, height):
    xmin = bbx[1]
    ymin = bbx[2]
    xmax = bbx[3]
    ymax = bbx[4]

    if xmin>=width or xmax<=0 or ymin>=height or ymax<=0 or width<=0 or height<=0:
        return False
    else:
        return True

'''
#=====Object example:=======
    <object>
        <name>face</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{{}}</xmin>
            <ymin>{{}}</ymin>
            <xmax>{{}}</xmax>
            <ymax>{{}}</ymax>
        </bndbox>
    </object>
'''
def insertObject(doc, bbx):
    obj = doc.createElement('object')
    name = doc.createElement('name')
    name.appendChild(doc.createTextNode(BbxIndexName[bbx[0]]))
    obj.appendChild(name)
    pose = doc.createElement('pose')
    pose.appendChild(doc.createTextNode('Unspecified'))
    obj.appendChild(pose)
    truncated = doc.createElement('truncated')
    truncated.appendChild(doc.createTextNode(str(0)))
    obj.appendChild(truncated)
    difficult = doc.createElement('difficult')
    difficult.appendChild(doc.createTextNode(str(0)))
    obj.appendChild(difficult)
    bndbox = doc.createElement('bndbox')

    left = bbx[1]
    top  = bbx[2]
    right = bbx[3]
    bottom = bbx[4]

    xmin = doc.createElement('xmin')
    xmin.appendChild(doc.createTextNode(str(int(left))))
    bndbox.appendChild(xmin)    
    ymin = doc.createElement('ymin')                
    ymin.appendChild(doc.createTextNode(str(int(top))))
    bndbox.appendChild(ymin)                
    xmax = doc.createElement('xmax')                
    xmax.appendChild(doc.createTextNode(str(int(right))))
    bndbox.appendChild(xmax)                
    ymax = doc.createElement('ymax')    
    ymax.appendChild(doc.createTextNode(str(int(bottom))))
    bndbox.appendChild(ymax)

    obj.appendChild(bndbox)                
    return obj

# xml file name: file.xml eg 000013.xml
def create(xmlRootPath, folder, filename, path, width1, height1, depth1, mat):
    for objIndex, bbx in enumerate(mat):
        if objIndex==0:
            # generate head info of xml file
            folderString = folder
            filenameString = filename

            doc = Document()
            annotation = doc.createElement('annotation')
            doc.appendChild(annotation)
            
            folder = doc.createElement('folder')
            folder.appendChild(doc.createTextNode(folderString))
            annotation.appendChild(folder)
            
            filename = doc.createElement('filename')
            filename.appendChild(doc.createTextNode(filenameString))
            annotation.appendChild(filename)
            
            source = doc.createElement('source')                
            database = doc.createElement('database')
            database.appendChild(doc.createTextNode('LIP FashionDataset'))
            source.appendChild(database)
            source_annotation = doc.createElement('annotation')
            source_annotation.appendChild(doc.createTextNode('LIP FashionDataset'))
            source.appendChild(source_annotation)
            image = doc.createElement('image')
            image.appendChild(doc.createTextNode('flickr'))
            source.appendChild(image)
            flickrid = doc.createElement('flickrid')
            flickrid.appendChild(doc.createTextNode('NULL'))
            source.appendChild(flickrid)
            annotation.appendChild(source)
            
            owner = doc.createElement('owner')
            flickrid = doc.createElement('flickrid')
            flickrid.appendChild(doc.createTextNode('NULL'))
            owner.appendChild(flickrid)
            name = doc.createElement('name')
            name.appendChild(doc.createTextNode('SYSU and CMU'))
            owner.appendChild(name)
            annotation.appendChild(owner)
            
            size = doc.createElement('size')
            width = doc.createElement('width')
            width.appendChild(doc.createTextNode(str(width1)))
            size.appendChild(width)
            height = doc.createElement('height')
            height.appendChild(doc.createTextNode(str(height1)))
            size.appendChild(height)
            depth = doc.createElement('depth')
            depth.appendChild(doc.createTextNode(str(depth1)))
            size.appendChild(depth)
            annotation.appendChild(size)
            
            segmented = doc.createElement('segmented')
            segmented.appendChild(doc.createTextNode(str(0)))
            annotation.appendChild(segmented)
            # generate object info
            if isValidBox(bbx, width1, height1):
                annotation.appendChild(insertObject(doc, bbx))
        else:
            # generate object info
            if isValidBox(bbx, width1, height1):
                annotation.appendChild(insertObject(doc, bbx))
    xmlName = xmlRootPath+filenameString.split('.')[0]+'.xml'
    try:
        f = open(xmlName, "w")
        f.write(doc.toprettyxml(indent = '    '))
        f.close()
    except:
        pass


def getAnnos(fidin, num):
    mat = []
    for i in range(num):
        line = fidin.readline().strip('\n')
        line = map(int, line.split(' '))
        mat.append(list(line))
    return mat, fidin

if __name__ == "__main__":

    # FullPictureRoot = 'D:/DataSet/LIP/FashionDataset/JPEGImages'
    PictureRoot = './JPEGImages'
    '''
        File name eg 'JPEGImages/997_1.jpg'
        Numbers W H C eg 5 224 224 3
        Bounding box
            ClassId xmin ymin xmax ymax
    '''
    Bbx_train_gt_txt = './FashionDataset_train_bbx_gt1.txt'
    Bbx_val_gt_txt = './FashionDataset_val_bbx_gt.txt'

    rootPath = 'D:/DataSet/LIP/FashionDataset/'

    xmlTrainRootPath = './Annotations/train/'
    xmlValRootPath = './Annotations/val/'

    with open(Bbx_train_gt_txt, 'r') as f_train:
        line = f_train.readline().strip('\n')
        while line:
            path = rootPath + line
            folder = line.split('\\')[0] # for windows
            filename = line.split('\\')[-1] # for windows
            print filename
            line = f_train.readline().strip('\n')
            nmu_bbx = int(line.split(' ')[0])
            width = int(line.split(' ')[1])
            height = int(line.split(' ')[2])
            depth = int(line.split(' ')[3])
            # get bbx info
            mat, f_train = getAnnos(f_train, nmu_bbx)
            # create xml file for every img
            create(xmlTrainRootPath, folder, filename, path, width, height, depth, mat)
            line = f_train.readline().strip('\n')
    f_train.close()

    # with open(Bbx_train_gt_txt, 'r') as f_val:

