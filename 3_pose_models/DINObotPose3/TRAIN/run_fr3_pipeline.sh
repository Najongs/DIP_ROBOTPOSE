#!/bin/bash
# Phase 1 chained: (wait FR3 crop detector) -> FR3 angle head -> FR3 rot head -> FR3 eval.
# Backbone frozen throughout; heads warm-started from Panda crop heads; supervised on FR3 real labels.
set -u
cd "$(dirname "$0")"
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate dino
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
export WANDB_MODE=offline
DATA=/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset
TR=$DATA/fr3_train; VA=$DATA/fr3_val
P_STAGE1=./outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
P_S1ANG=./outputs_angle/angle_20260603_013948/best_angle_head.pth
P_S1ROT=./outputs_rotation/rot_20260604_162336/best_rot_head.pth
P_CROPANG=./outputs_angle/angle_crop_20260605_174740/best_angle_head.pth
P_CROPROT=./outputs_rotation/rot_crop_20260606_022535/best_rot_head.pth

# 1) wait for detector
echo "[pipeline] waiting for FR3 detector..."
while [ ! -f outputs_fr3/DET_DONE ]; do sleep 30; done
DET=$(ls -t outputs_fr3/crop_det_*/best_heatmap.pth 2>/dev/null | head -1)
echo "[pipeline] FR3 detector = $DET"

# 2) angle head
AOUT=./outputs_fr3/angle; mkdir -p "$AOUT"
python3 train_angle.py --detector-ckpt "$DET" --train-dir "$TR" --val-dir "$VA" \
  --output-dir "$AOUT" --crop-to-robot --crop-margin 1.5 --fk-weight 10.0 \
  --head-type mlp --epochs 40 --batch-size 32 --lr 1e-3 --min-lr 1e-6 --num-workers 12 \
  --init-head "$P_CROPANG" > "$AOUT/train.log" 2>&1
FR3ANG=$AOUT/best_angle_head.pth
echo "[pipeline] angle done -> $FR3ANG"

# 3) rot head
ROUT=./outputs_fr3/rot; mkdir -p "$ROUT"
python3 train_rotation.py --detector-ckpt "$DET" --train-dir "$TR" --val-dir "$VA" \
  --output-dir "$ROUT" --crop-to-robot --crop-margin 1.5 --t-weight 50.0 \
  --epochs 25 --batch-size 32 --lr 1e-3 --min-lr 1e-6 --num-workers 12 \
  --init-head "$P_CROPROT" > "$ROUT/train.log" 2>&1
FR3ROT=$ROUT/best_rot_head.pth
echo "[pipeline] rot done -> $FR3ROT"

# 4) eval on FR3 held-out (Panda stage1 for rough bbox; FR3 crop det + FR3 heads for precise pose)
mkdir -p ../Eval/fr3_logs
cd ../Eval
python3 selfbbox_eval.py \
  --stage1-detector "../TRAIN/$P_STAGE1" --stage1-angle "../TRAIN/$P_S1ANG" --stage1-rot "../TRAIN/$P_S1ROT" \
  --crop-detector "../TRAIN/$DET" --crop-angle "../TRAIN/$FR3ANG" --rot-head "../TRAIN/$FR3ROT" \
  --bbox-from-solved --bbox-guard --dark-decode --cov-pnp \
  --val-dir "$VA" --max-frames 1442 --dump-npz rc_dumps/fr3_finetuned.npz \
  > fr3_logs/finetuned_eval.log 2>&1
echo "[pipeline] EVAL: $(grep -oE 'ADD-AUC@100mm: [0-9.]+' fr3_logs/finetuned_eval.log | tail -1)"
echo FR3_PIPELINE_DONE > ../TRAIN/outputs_fr3/PIPELINE_DONE
