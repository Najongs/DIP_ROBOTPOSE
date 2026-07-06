#!/bin/bash
set -u
cd /home/najo/NAS/DIP-multirobot/3_pose_models/DINObotPose3/TRAIN
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate dino
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
export WANDB_MODE=offline
CD=/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset
python3 train_heatmap.py   --data-dir "$CD/meca_real_train" --val-dir "$CD/meca_real_val"   --checkpoint ./outputs_meca500/detector_20260705_181652/best_heatmap.pth   --keypoint-names link0,link1,link2,link3,link4,link5,link6   --output-dir "./outputs_meca500/real_detector_20260706_122304"   --image-size 512 --heatmap-size 512 --crop-to-robot --crop-margin 1.5   --unfreeze-blocks 4 --aug-level strong --occlusion-prob 0.0 --fda-prob 0.0   --epochs 30 --batch-size 32 --num-workers 12   --learning-rate 2e-4 --backbone-lr 2e-5 --min-lr 1e-7   --wandb-project meca500-real-detector
touch outputs_meca500/REAL_DET_DONE
