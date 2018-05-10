#-*-coding:utf-8-*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import functional as F
import utils.dataset as dataset
import utils.models as models
from utils.train_Components import *
import torchvision
import torchvision.transforms as transforms
import shutil
import math
import os
import os.path as osp
import math
import argparse
from collections import OrderedDict


AttrKey_num = { 'skirt_length_labels':6,'coat_length_labels':8,\
            'collar_design_labels':5, 'lapel_design_labels':5,\
            'neck_design_labels':5, 'neckline_design_labels':10,\
            'pant_length_labels':6, 'sleeve_length_labels':9}

AttrValues = {'skirt_length_labels':['Invisible','Short_Length','Knee_Length','Midi_Length','Ankle_Length','Floor_Length'],\
            'coat_length_labels':['Invisible','High_Waist_Length','Regular_Length','Long_Length','Micro_Length','Knee_Length','Midi_Length','Ankle&Floor_Length'],\
            'collar_design_labels':['Invisible','Shirt_Collar','Peter_Pan','Puritan_Collar','Rib_Collar'],\
            'lapel_design_labels':['Invisible','Notched','Collarless','Shawl_Collar','Plus_Size_Shawl'],\
            'neck_design_labels':['Invisible','Turtle_Neck','Ruffle_Semi-High Collar','Low_Turtle_Neck','Draped_Collar'],\
            'neckline_design_labels':['Invisible','Strapless_Neck','Deep_V_Neckline','Straight_Neck','V_Neckline','Square_Neckline','Off_Shoulder','Round_Neckline','Sweat_Heart_Neck','One_Shoulder_Neckline'],\
            'pant_length_labels':['Invisible','Short_Pant','Mid_Length','3/4_Length','Cropped_Pant','Full_Length'],\
            'sleeve_length_labels':['Invisible','Sleeveless','Cup Sleeves','Short Sleeves','Elbow Sleeves','3/4 Sleeves','Wrist Length','Long Sleeves','Extra Long Sleeves']}

def get_args():
    parser = argparse.ArgumentParser(description='Fourth Baidu Competition Experiment')

    parser.add_argument('--arch', metavar='ARCH', default='Resnet50', help='model architecture')
    parser.add_argument('--gpuId', default='0', type=str, help='GPU Id')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('-s', '--input_size', default=224, type=int,
                        metavar='N', help='input size (default: 224)')

    parser.add_argument('--AttrKey', default='coat_length_labels', type=str, help='AttrKey')

    parser.add_argument('--train_path', metavar='DATA_PATH', type=str, default=['./base', './label.csv'],
                        help='root to train Image and csv file name', nargs=2)

    parser.add_argument('--val_path', metavar='DATA_PATH', type=str, default=['./rank', './fashionAI_attributes_answer_a_20180428.csv'],
                        help='root to val Image and csv file name', nargs=2)

    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--savePath', default='./models', type=str, \
                        help='path to save model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--zeroTrain', default=False, action='store_true', help='choose if train from Scratch or not')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')


    args = parser.parse_args()
    return args

args = get_args()

print("arch         is: {}".format(args.arch))
print("gpuId        is: {}".format(args.gpuId))
print("init lr      is: {}".format(args.lr))
print("batch size   is: {}".format(args.batch_size))
print("input_size   is: {}".format(args.input_size))
print("AttrKey      is: {}".format(args.AttrKey))
print("train_path   is: {}".format(args.train_path))
print("val_path     is: {}".format(args.val_path))
print("epochs       is: {}".format(args.epochs))
print("savePath     is: {}".format(args.savePath))
print("resume       is: {}".format(args.resume))
print("momentum     is: {}".format(args.momentum))
print("zeroTrain    is: {}".format(args.zeroTrain))
print("weight_decay is: {}".format(args.weight_decay))
print("print_freq   is: {}".format(args.print_freq))


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuId

use_gpu = torch.cuda.is_available()
num_batches = 0

def train_val(model, train_loader, val_loader, print_freq=50, optimizer=None, epoches=10):
    global args, num_batches
    print("Start training.")
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    StepLr = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    max_train_acc = 0
    max_val_acc = 0
    for i in range(epoches):
        if i<=15:
            StepLr.step(i)
        model.train()
        print("Epoch: ", i, "lr is: {}".format(StepLr.get_lr()))
        num_batches = train_epoch(model, num_batches, train_loader, print_freq=print_freq, optimizer=optimizer)

        cur_train_acc, cur_val_acc = get_train_val_acc(model, train_loader, val_loader)

        if i==0:
            max_train_acc, max_val_acc = cur_train_acc, cur_val_acc
            filename = "{}_{}_{}_{}.pth".format(args.arch, args.AttrKey, str(cur_train_acc), str(cur_val_acc))
            if not os.path.exists(osp.join(args.savePath, args.AttrKey)):
                os.mkdir(osp.join(args.savePath, args.AttrKey))
            torch.save(model.state_dict(), osp.join(args.savePath, args.AttrKey, filename))
        elif max_val_acc<cur_val_acc:
            # delete old state_dict
            old_filename = "{}_{}_{}_{}.pth".format(args.arch, args.AttrKey, str(max_train_acc), str(max_val_acc))
            os.remove(osp.join(args.savePath, args.AttrKey, old_filename))
            max_train_acc, max_val_acc = cur_train_acc, cur_val_acc
            filename = "{}_{}_{}_{}.pth".format(args.arch, args.AttrKey, str(cur_train_acc), str(cur_val_acc))
            torch.save(model.state_dict(), osp.join(args.savePath, args.AttrKey, filename))

    print("Finished training.")

def main():
    global args, num_batches, use_gpu
    if not args.zeroTrain:
        if args.arch == "densenet201":
            num_classs = AttrKey_num[args.AttrKey]
            input_size = args.input_size
            model = models.Modified_densenet201(num_classs, input_size)
        elif args.arch == "Resnet50":
            num_classs = AttrKey_num[args.AttrKey]
            input_size = args.input_size
            model = models.Modified_Resnet50(num_classs, input_size)
        elif args.arch == "Resnet101":
            num_classs = AttrKey_num[args.AttrKey]
            input_size = args.input_size
            model = models.Modified_Resnet101(num_classs, input_size)
        elif args.arch == "Resnet152":
            num_classs = AttrKey_num[args.AttrKey]
            input_size = args.input_size
            model = models.Modified_Resnet152(num_classs, input_size)
        elif args.arch == "nasnetalarge":
            model = models.Modified_nasnetalarge()

        if args.resume:
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint, strict=True)

        if use_gpu:
            model = model.cuda()
            print("Use GPU!")
        else:
            print("Use CPU!")

        if not os.path.exists('./models'):
            os.mkdir('./models')

        train_root, train_csvName = args.train_path
        val_root, val_csvName = args.val_path

        train_loader = dataset.train_loader(train_root, args.AttrKey, train_csvName, batch_size=args.batch_size, num_workers=10, pin_memory=True)
        val_loader = dataset.test_loader(val_root, args.AttrKey, val_csvName, batch_size=1, num_workers=10, pin_memory=True)


    train_val(model, train_loader, val_loader, print_freq=args.print_freq, optimizer=None, epoches=args.epochs)

if __name__ == "__main__":
    main()