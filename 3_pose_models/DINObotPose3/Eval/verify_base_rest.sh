#!/bin/bash
# Base-only passes for orb + azure (rs/kinect base already done; rs RC confirmed 0.8213).
set -u
cd "$(dirname "$0")"
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate dino
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CROPDET=../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
ST=../TRAIN/outputs_selftrain
DATA=../Dataset/Converted_dataset/DREAM_real
NF=800
base() { # 1=cam 2=crop_angle 3=rot_head 4=val
  python selfbbox_eval.py --stage1-detector $DET --stage1-angle $S1ANG --stage1-rot $S1ROT \
    --crop-detector $CROPDET --crop-angle "$2" --rot-head "$3" \
    --bbox-from-solved --bbox-guard --dark-decode --cov-pnp \
    --frac-range 0.7 1.0 --max-frames $NF --val-dir "$4" --dump-npz rc_dumps/$1.npz \
    > sota_logs/$1_base.log 2>&1
}
base orb $ST/orb_rot_r1/best_selftrain_head.pth $ST/orb_rot_r1/best_selftrain_rot.pth $DATA/panda-orb
base azure ../TRAIN/outputs_angle/angle_crop_20260605_174740/best_angle_head.pth ../TRAIN/outputs_rotation/rot_crop_20260606_022535/best_rot_head.pth $DATA/panda-3cam_azure
echo BASEREST_DONE > sota_logs/DONE_BASE
