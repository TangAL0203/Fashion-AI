#-*-coding:utf-8-*-
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""

import cPickle
import logging
import numpy as np
import os
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text) # 0 或者　１
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    11-point interpolated average precision
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end 在尾部添加哨兵值
        #　计算recall 从1/M 到　M/M　之间，不同精度的平均值,每一个recall对应一个最大的precision值,计算M个precision的均值得到AP值
        # Top-N预测中, 正样本个数为M
        # [0,0,0.2,0.4,0.8,1] => [0, 0,0,0.2,0.4,0.8,1, 1]
        mrec = np.concatenate(([0.], rec, [1.])) # 在首尾加上0.0和1.0
        mpre = np.concatenate(([0.], prec, [0.])) # 在首尾加上0.0和0.0

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            # mpre某处的值一直是其后面的值最大的,mpre初始值为整个数组最大值, 比如mpre[5]为mpre[5:end]中最大的值

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        # array([ 0. ,  0.1,  0.1,  0.2,  0.2,  0.2,  0.3,  0.3,  0.3,  0.3])
        # i = np.where(mrec[1:] != mrec[:-1])[0] => i==[0, 2, 5]
        i = np.where(mrec[1:] != mrec[:-1])[0] # i记录recall值发生变化的位置

        # and sum (\Delta recall) * prec
        # (mrec[i + 1] - mrec[i]) = 1/M
        # 对这M个precision值取平均即得到最后的AP值
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]) # 取r'>r处最大的precision，所以i+1
    return ap

# detpath 存放检测结果的路径
# annopath 检测图片的xml标签文件路径
# imagesetfile 存放待检测的图片列表信息的txt文档
# classname 类别名字
# cachedir 暂存annotations的路径
# 计算某一类别的ap
def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    1. detpath: Path to detections 
        detpath.format(classname) should produce the detection results file.
    ２. annopath: Path to annotations 
        annopath.format(imagename) should be the xml annotations file.
    3. imagesetfile: Text file containing the list of images, one image per line.
    4. classname: Category name (duh)
    5. cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    imageset = os.path.splitext(os.path.basename(imagesetfile))[0] # txt文件名
    cachefile = os.path.join(cachedir, imageset + '_annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines] # imagenames为所有测试图片的全路径

    if not os.path.isfile(cachefile):
        # load annots
        recs = {} # recs['图片全路径': bbx信息]
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                logger.info(
                    'Reading annotation for {:d}/{:d}'.format(
                        i + 1, len(imagenames)))
        # save
        logger.info('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)  #　cPickle将字典recs保存为序列化的对象．
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {} # 存放某图片某一类别对应的bbx信息+difficult目标个数+[False*目标个数]
    npos = 0 # 统计总的不属于difficult目标的个数, 用来计算Recall,作为分母
    for imagename in imagenames:
        #　找出recs中，某一类别所有的obj(包括difficult和not difficult)
        R = [obj for obj in recs[imagename] if obj['name'] == classname] # recs[imagename]为字典列表,存放单张图片所有bbx的信息
        bbox = np.array([x['bbox'] for x in R]) # [[xmin,ymin,xmax,ymax],...]
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R) # 记录某个gt bbx是否找与之匹配的预测框了
        npos = npos + sum(~difficult) 
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    # 读取某一类的检测结果文件
    # comp3_det_test_car.txt:
    # 000004 0.702732 89 112 516 466
    # 000006 0.870849 373 168 488 229
    # 000006 0.852346 407 157 500 213
    # 000006 0.914587 2 161 55 221
    # 000008 0.532489 175 184 232 201
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines] # 存放图片的名字
    confidence = np.array([float(x[1]) for x in splitlines]) # 存放检测的Bbx的confidence信息
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]) # 存放Bbx的xmin,ymin,xmax,ymax

    # sort by confidence
    # np.argsort返回数组按照升序排对应元素的索引序号
    sorted_ind = np.argsort(-confidence) #　降序排对应元素的索引序号
    BB = BB[sorted_ind, :] # BB按照confidence降序进行重排序
    image_ids = [image_ids[x] for x in sorted_ind] # 图片名字也按照confidence降序进行重排序

    # go down dets and mark TPs and FPs
    nd = len(image_ids) # 检测结果数目
    tp = np.zeros(nd) # 记录检测结果是否为TP
    fp = np.zeros(nd) # 记录检测结果是否为FP
    for d in range(nd):
        R = class_recs[image_ids[d]] # class_recs存ground truth bbx信息
        bb = BB[d, :].astype(float)
        ovmax = -np.inf # numpy中的inf表示一个无限大的正数,取负则表示无线小的负数
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # 计算Bbx与gt bbx之间的IOU
            # compute overlaps
            # intersection
            #　np.maximum会对bb[0]进行broadcast
            # bbox[:,0] = array([1, 5])
            # np.maximum(bbox[:,0],3) = array([3, 5])
            ixmin = np.maximum(BBGT[:, 0], bb[0]) # 将检测结果与对应图片中的多个目标的ground truth bbx进行比较
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.) # 为啥要加1?
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni # 该预测的BBX与对应的ground truth bbx之间的IOU
            ovmax = np.max(overlaps) # 一个检测结果只对应一个ground truth bbx,那就是IOU最大的ground truth bbx
            jmax = np.argmax(overlaps) # 返回最大IOU对应的该图片的ground truth bbx的序号

        if ovmax > ovthresh:
            # 如果jmax对应的ground truth bbx不属于difficult样本
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.  # 某个检测结果为tp的条件: 1.IOU与某个gt bbx大于阈值 2.对应的gt bbx不是difficult 3.该gt bbx还没有预测框与之匹配
                    R['det'][jmax] = 1 # 已有匹配的预测框,将该标志位置1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp) # tp值一直累加,比如之前tp=[0,0,1,0,1,1] => [0,0,1,1,2,3]
    rec = tp / float(npos) # tp除以总的gt bbx个数等于召回率
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  # np.finfo(np.float64).eps=2.2204460492503131e-16
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


# 计算MAP,多个类AP值的均值
def getMAP(f, detpath, annopath, imagesetfile, classnames, cachedir, ovthresh=0.5, use_07_metric=False):
    for classname in classnames:
        rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, classname, cachedir, ovthresh=0.5, use_07_metric=False)
        f.write(classname+'  ap is: '+str(round(ap,3))+'\n')


if __name__ == "__main__":
    detpath = './result/{}.txt'  # shoe.txt
    annopath = './Annotations/val/{}.xml' # 00001.xml
    imagesetfile = '../LIP/val_id.txt'  # imagename
    classnames = ['hat', 'sunglass', 'upperclothes', 'skirt', 'pants', \
                  'dress', 'belt', 'shoe', 'bag', 'scarf', 'glove', \
                  'coat', 'socks', 'jumpsuits']
    cachedir = '/cache'

    f = open('./detect_map_result.txt', 'w')

    MAP = getMAP(f, detpath, annopath, imagesetfile, classnames, cachedir)

    f.write("MAP is: "+str(round(MAP, 3)+'\n')
    f.close()

    print("the value of MAP is: ", round(MAP, 3)