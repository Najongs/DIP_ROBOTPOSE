#!/usr/bin/env bash
# §4.10 pose-level backbone comparison: DINOv3 vs SigLIP2 (matched ViT-B/16 ~86M) at ADD-AUC level.
# Controlled apples-to-apples: identical protocol (--oracle-bbox GT box to bypass stage1 and isolate
# the backbone in the crop-detector + crop pose heads), identical CLEAN synthetic crop heads (no
# self-train, no RC), base-only, same 4 real cameras, same held-out 1000. Only the backbone differs.
#   backbone_poselevel.sh <dino|siglip> <gpu_uuid>
set -uo pipefail
cd "$(dirname "$0")"
BB="${1:?dino|siglip}"; export CUDA_VISIBLE_DEVICES="${2:?gpu uuid}"
NF=1000; T=../TRAIN
if [ "$BB" = "dino" ]; then
  MODEL=facebook/dinov3-vitb16-pretrain-lvd1689m
  S1DET=$T/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
  CROPDET=$T/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
  ANG=$T/outputs_angle/angle_crop_20260605_174740/best_angle_head.pth
  ROT=$T/outputs_rotation/rot_crop_20260606_022535/best_rot_head.pth
else
  MODEL=google/siglip2-base-patch16-512
  S1DET=$T/outputs_heatmap/siglip2_unfrozen_20260602_184024/best_heatmap.pth
  CROPDET=$T/outputs_heatmap/siglip_crop_ddp_20260715_014111/best_heatmap.pth
  ANG=$T/outputs_angle/siglip_angle_crop_20260716_010615/best_angle_head.pth   # norm-fixed (mean=std=0.5)
  ROT=$T/outputs_rotation/siglip_rot_crop_20260716_010615/best_rot_head.pth     # norm-fixed (mean=std=0.5)
fi
DR=../Dataset/Converted_dataset/DREAM_real
RES=ablation_logs/backbone_poselevel; mkdir -p "$RES"
OUT="$RES/results_${BB}.tsv"; : > "$OUT"
declare -A CAM=( [azure]=panda-3cam_azure [kinect]=panda-3cam_kinect360 [realsense]=panda-3cam_realsense [orb]=panda-orb )
for cam in azure kinect realsense orb; do
  L="$RES/${BB}_${cam}.log"
  python selfbbox_eval.py --model-name "$MODEL" \
    --stage1-detector "$S1DET" --crop-detector "$CROPDET" --crop-angle "$ANG" --rot-head "$ROT" \
    --oracle-bbox --cov-pnp --dark-decode --conf-gate 0.05 \
    --frac-range 0.7 1.0 --max-frames $NF --val-dir "$DR/${CAM[$cam]}" > "$L" 2>&1
  A=$(grep -a "ADD-AUC@100mm" "$L" | tail -1 | sed -E 's/.*ADD-AUC@100mm: ([0-9.]+).*/\1/')
  echo -e "${BB}\t${cam}\tADD-AUC=${A}" | tee -a "$OUT"
done
echo "BACKBONE ${BB} DONE" > /tmp/claude-1002/-home-najo-NAS-DIP/5aafbd5b-1895-41b2-90ed-8d6e9438b7dd/scratchpad/backbone_${BB}_done.txt
