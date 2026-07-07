#!/bin/bash
set -e
cd /home/najo/NAS/DIP-multirobot/3_pose_models/DINObotPose3/TRAIN
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
/opt/anaconda3/bin/python train_angle.py   --detector-ckpt ./outputs_meca500/detector_20260705_181652/best_heatmap.pth   --train-dir /home/najo/NAS/DIP/datasets/meca500_synth/train   --val-dir /home/najo/NAS/DIP/datasets/meca500_synth/val   --keypoint-names link0,link1,link2,link3,link4,link5,link6   --output-dir ./outputs_meca500/angle_20260706_092838   --crop-to-robot --crop-margin 1.5   --fk-weight 0 --reproj-weight 0   --head-type mlp --epochs 40 --batch-size 32 --lr 1e-3 --num-workers 8
touch ./outputs_meca500/ANGLE_DONE
