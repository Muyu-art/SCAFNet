#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1,2,3,4,5 nohup python3 -m torch.distributed.launch --nproc_per_node=5 train.py --launcher pytorch > log.txt&


CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --batch_size 8 --epochs 30 --cfg_file cfgs/kitti_models/CaEPNet-V2.yaml
