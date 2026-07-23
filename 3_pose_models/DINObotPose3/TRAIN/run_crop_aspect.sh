#!/bin/bash
# crop detector 재학습 — 학습/배포 크롭 기하 불일치 교정.
#
# 진단 (Eval/crop_chain_probe.py, ORACLE bbox, 원본 640x480 px, mediocre_band 1000 프레임):
#   chain          exc.med   med.med   med.p90  clean.med  clean.p90
#   A_deploy          1.62      3.35     16.69       1.78       6.98   <- 현재 배포 경로
#   B_train           1.03      2.58     15.03       1.14       6.48   <- 학습 경로(정사각, 왜곡 없음)
#   C_iso2x           1.04      2.59     14.98       1.16       6.50   <- 등방 2회 리샘플
#   D_distort1x       1.59      3.46     16.34       1.72       6.61   <- 왜곡만, 해상도 손실 없음
# D≈A 이고 B≈C 이므로 페널티는 전부 4:3->1:1 '종횡비 왜곡' 탓이며 해상도 손실은 0이다.
# 배포(selfbbox_eval.py)는 원본을 512x512 로 비등방 리사이즈한 뒤 그 공간에서 정사각 roi_align
# 크롭을 뜬다 => 원본 좌표로 보면 4:3 직사각형. 그런데 crop detector 는 원본 공간의 '정사각'
# 크롭으로 학습됐다. 즉 배포에서만 세로로 33% 늘어난 입력을 받는 순수 도메인 불일치다.
#
# 개입: dataset.py 에 --crop-aspect 를 추가(기본 1.0 = 기존 동작 완전 보존)하고 배포 프레임
# 종횡비 640/480=4/3 로 학습. 배포된 crop_20260605_010622 에서 warm-start 하여 기하만 적응.
# RUN2 는 여기에 --crop-res-jitter 를 얹어 작은 로봇(유효해상도 저하) 잔여 격차를 추가로 노린다.
#
# GPU0 전용. GPU1~4 는 장기 학습 중이므로 절대 건드리지 말 것.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=GPU-7ff6997b-14c1-9283-5119-251c9c899b8e
export HF_HUB_OFFLINE=1
cd /home/najo/NAS/DIP/3_pose_models/DINObotPose3/TRAIN

PY=/home/najo/.conda/envs/dino/bin/python
TRAIN_DIR=../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr
VAL_DIR=../Dataset/Converted_dataset/DREAM_real/panda-3cam_realsense
WARM=./outputs_heatmap/crop_20260605_010622/best_heatmap.pth   # 배포 crop detector (읽기 전용)
ASPECT=1.3333333333                                            # 640/480 — 배포 프레임 종횡비

run() {
  TAG=$1; shift
  OUT=./outputs_heatmap/${TAG}; mkdir -p "$OUT"
  echo "=== $TAG start $(date -Is) ===" >> ./outputs_heatmap/${TAG}.log
  $PY train_heatmap.py \
    --data-dir "$TRAIN_DIR" \
    --val-dir "$VAL_DIR" \
    --checkpoint "$WARM" \
    --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --output-dir "$OUT" \
    --image-size 512 --heatmap-size 512 \
    --crop-to-robot --crop-margin 1.5 --crop-aspect $ASPECT \
    --unfreeze-blocks 4 \
    --aug-level strong \
    --occlusion-prob 0.0 --fda-prob 0.0 \
    --epochs 8 \
    --batch-size 32 \
    --num-workers 12 \
    --learning-rate 5e-5 --backbone-lr 5e-6 --min-lr 1e-7 --weight-decay 1e-5 \
    --wandb-project "dinov3-stage1-detector" --wandb-run-name "$TAG" \
    "$@" >> ./outputs_heatmap/${TAG}.log 2>&1
  echo "=== $TAG done rc=$? $(date -Is) ===" >> ./outputs_heatmap/${TAG}.log
}

# RUN 1 (주): 배포 기하 정합만. 예측 — 배포 경로 clean 2D median 1.78 -> ~1.14px
run cropasp_a43

# RUN 2 (부가): + 유효해상도 지터. 작은 로봇 잔여 격차(사분위 최소 1.52px vs 중간 0.82px) 대상
run cropasp_a43_res --crop-res-jitter 0.5 --crop-res-min 140
