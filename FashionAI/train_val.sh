#!/usr/bin/env sh
echo "train ALI dataset"
python train_val.py --arch Resnet50 --batch_size 8 --epochs 50 --gpuId 0 --momentum 0.9 --weight_decay 1e-4 --print_freq 50
