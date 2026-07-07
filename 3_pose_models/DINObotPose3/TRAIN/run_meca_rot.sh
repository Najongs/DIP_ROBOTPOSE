#!/bin/bash
set -e
cd /home/najo/NAS/DIP-multirobot/3_pose_models/DINObotPose3/TRAIN
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
/opt/anaconda3/bin/python train_rotation.py   --detector-ckpt ./outputs_meca500/detector_20260705_181652/best_heatmap.pth   --train-dir /home/najo/NAS/DIP/datasets/meca500_synth/train   --val-dir /home/najo/NAS/DIP/datasets/meca500_synth/val   --keypoint-names link0,link1,link2,link3,link4,link5,link6   --fk-robot meca500   --output-dir ./outputs_meca500/rot_20260706_111845   --crop-to-robot --crop-margin 1.5   --epochs 20 --batch-size 32 --lr 1e-3 --t-weight 1.0 --num-workers 8
touch ./outputs_meca500/ROT_DONE
