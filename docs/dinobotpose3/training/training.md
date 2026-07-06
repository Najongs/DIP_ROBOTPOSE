# 학습 (Training)

> 스크립트: `3_pose_models/DINObotPose3/TRAIN/`. env `dino` (torch 2.10+cu128). **backbone은 항상 동결** — head/솔버만 학습.

## 스테이지 구조

```
Stage 1   2D 검출기 (히트맵)         train_heatmap.py / run_train_stage1_unfrozen.sh
   │        DINOv3 + ViTKeypointHead, 강한 aug. crop 변형(run_train_detector_crop.sh)
   ▼
Stage 1.5 관절각 head               train_angle.py / run_train_angle_crop.sh
   │        frozen 검출기 위 AngleHead(MLP). 손실 = sin/cos + FK
   │        crop / jitter / patch / occlude-aug / reproj-consistency 변형
   ▼
회전 head                          train_rotation.py / run_train_rotation.sh
   │        DINO 외관 → R_init (6D). GT 포즈(Kabsch) 지도
   ▼
Stage 3   자가학습 (배포에 실제 사용)    selftrain_pseudo.py / selftrain_pseudo_rot.py / selftrain_detector.py
            솔버 pseudo-label로 head를 real에 적응 (카메라별). 라벨 불필요.
```

## 대표 설정 (crop angle head)

`run_train_angle_crop.sh`: DINOv3 ViT-B/16, image 512, batch 32, LR 1e-3(cos→1e-6), 50ep,
fk-weight 10, `--crop-to-robot --crop-margin 1.5`. warm-start `--init-head`.

## train_angle.py 손실 항 (2026-07-04 기준)

```
loss = sin/cos SmoothL1(θ)                         # 각도 회귀
     + fk_weight · MSE(FK(θ_pred), FK(θ_gt))       # robot-frame FK 일관성
     + occlude-aug (--occlude-aug, --kp-drop)      # 가림 증강 (강건성 config)
     + reproj_weight · Huber(proj(FK(θ)|GT pose), gt2d)   # RoboTAG식 cross-dim 일관성 (실험 중)
```
- `--occlude-aug R`: 학습 시 RoI의 U(0.05,R) 면적을 검은 occluder로 페이스트(p=0.5). 가림 입력 강건화.
- `--reproj-weight W`: GT 카메라 포즈로 FK(pred) 재투영 → GT 2D 정렬. 카메라 프레임 신호(azure 격차 공략).
- 증강 유틸: `Eval/occl_util.py::paste_random_occluders_`.

## 자가학습 (Stage 3) — 배포 head의 근원

- `selftrain_pseudo.py [--crop]`: 솔버가 refine한 관절각을 high-conf real 프레임에서 pseudo-label로 → head fine-tune (+합성 anti-forgetting). 이득 ∝ sim2real 갭 (realsense +0.137).
- `selftrain_pseudo_rot.py`: 각도+**회전** head 동시 적응 (배포 realsense/orb config).
- `selftrain_detector.py`: 솔버 재투영 키포인트를 2D head에 distill (kinect +0.052).

## 배포 체크포인트 (현재)

| 구성요소 | 경로 |
|---|---|
| stage1 검출기 | `outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth` |
| crop 검출기 | `outputs_heatmap/crop_20260605_010622/best_heatmap.pth` |
| crop angle | `outputs_angle/angle_crop_20260605_174740/best_angle_head.pth` |
| 회전(full/crop) | `outputs_rotation/{rot_20260604_162336, rot_crop_20260606_022535}/best_rot_head.pth` |
| self-train 페어 | `outputs_selftrain/{realsense,orb,kinect}_rot_r1/best_selftrain_{head,rot}.pth` |

## GPU 배치 주의 (중요)

- **UUID로 지정** — 이 머신은 정수 `CUDA_VISIBLE_DEVICES`가 물리 GPU와 뒤엉킴(0→물리3, 2→물리1). `CUDA_VISIBLE_DEVICES=GPU-<uuid>` 사용.
- 백그라운드 학습은 `setsid`로 완전 분리 (런처 셸 종료 시 orphan 방지).
- wandb 로깅 관례(`--use-wandb`). torchrun 멀티GPU 지원.

## 관련
- 모델 구조: [../architecture/model.md](../architecture/model.md)
- 학습 시 반증된 것(백본 적응 등): [../00_overview.md](../00_overview.md)
