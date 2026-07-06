# 최종 배포 모델 (Final Deployed Model) — 2026-07-05

> DREAM 4-real-split 배포 mean **0.804** vs RoboPEPP 0.780 / RoboTAG 0.740. 전 4카메라 가림 강건.
> 전체 맥락은 [00_overview.md](00_overview.md), 스택 실험은 [experiments/2026-07-05](experiments/2026-07-05_occaug_selftrain_stack.md).

## 성적표 (🔒 재잠금: held-out 1000/cam; azure full-1000)

| 카메라 | 배포 ADD-AUC | 가림 40% | RoboPEPP | 격차 |
|---|---|---|---|---|
| realsense | **0.815** | 0.396 | 0.805 | +0.010 BEAT |
| kinect360 | **0.828** | 0.393 | 0.785 | +0.043 BEAT |
| azure | **0.795** | 0.429 | 0.753 | +0.042 BEAT |
| orb | **0.778** | 0.399 | 0.775 | +0.003 BEAT |
| **mean** | **0.804** | 전부 >0.351 | 0.780 | **+0.024** |

프로토콜: predicted angles + 완전 자동 bbox(bbox-from-solved) + sim-to-real. rs/kinect/orb는 anti-leak held-out(뒤 30% 영역, 1000프레임 조밀 샘플). 가림 강건성은 별도 벤치(RoboPEPP Fig.6, synth_photo) — light+RC 곡선 0.812/0.765/0.678/0.575/0.429 (0~40%), 전 구간 RoboPEPP 초과.

**2026-07-06 재잠금(800→1000)**: mean 0.8037→**0.8039** (Δ+0.0002, 표본 수에 강건). 개별 ≤0.006 변동. **4/4 카메라 모두 RoboPEPP 초과**(orb −0.002→+0.003 전환). 확정 SOTA.
값(1000): rs 0.8153 / kinect 0.8275 / azure 0.7945 / orb 0.7784.

## 파이프라인 (배포)

```
이미지 512
 → DINOv3 ViT-B/16 (frozen) 검출기
 → self-bbox: 풀-프레임 검출+솔버 → FK 7kp 투영 → crop bbox → roi_align
 → crop 검출기 heatmap → [DARK sub-pixel 디코드] → 2D kp + conf
 → crop angle head(θ) + crop rot head(R_init)   ← 카메라별 (아래 표)
 → 운동학 솔버: PnP init + 재투영 refine, [cov-PnP], conf-gate 0.05
 → [nvdiffrast+SAM render-and-compare @448/512]  ← azure 제외 (근거리)
 → θ + 카메라 포즈 (R,t)
```

## 카메라별 배포 체크포인트

**공용 (전 카메라):**
- 풀-프레임 검출기: `TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth`
- crop 검출기: `TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth`
- 백본: `facebook/dinov3-vitb16-pretrain-lvd1689m` (frozen)

**angle + rot head (카메라별 최적, 전부 가림 강건):**

| 카메라 | angle head | rot head | RC |
|---|---|---|---|
| realsense | `outputs_selftrain/realsense_lightstack_20260705_003546/best_selftrain_head.pth` | 같은 폴더 `best_selftrain_rot.pth` | on @448 |
| kinect360 | `outputs_selftrain/kinect_lightstack_20260705_003552/best_selftrain_head.pth` | 같은 폴더 `best_selftrain_rot.pth` | on @448 |
| orb | `outputs_selftrain/orb_lightstack_20260705_003549/best_selftrain_head.pth` | 같은 폴더 `best_selftrain_rot.pth` | on @512 |
| azure | `outputs_angle/angle_occaug_light_20260704_015400/best_angle_head.pth` (light) | `outputs_rotation/rot_crop_occaug_20260704_002102/best_rot_head.pth` | **off** (근거리) |

**공용 옵션 (전 카메라 켬):** `--cov-pnp --dark-decode --bbox-from-solved --bbox-guard`
**RC:** `Eval/rc_refine_from_dump.py --render-h 448` (orb 512), SAM `weights_sam/sam_vit_b_01ec64.pth`.

## head 계보 (어떻게 만들어졌나)

```
crop angle head (angle_crop, real self-train)
        │
        ▼ 가림 증강 학습 (occlude-aug ratio≤0.3, kp_drop 없음)     ← "light head"
   light angle head (angle_occaug_light) : 가림 강건 + 클린 최고
        │
        ├─ azure: 그대로 사용 (self-train ~0)
        │
        ▼ 카메라별 self-train (+occlude-aug on synth anti-forget)   ← "스택"
   {realsense,kinect,orb}_lightstack : 가림 강건 + real 적응
```
- 가림 강건성은 **light head를 처음부터 증강 학습**해야 배어듦. 이미 적응된 head에 짧은 self-train으로 증강만 얹으면 안 됨(realsense 개선 스택 실증, 40% base 수준).
- rot head는 occaug rot(`rot_crop_occaug`) 공용 warm-start, 스택에서 카메라별 재학습.

## 재현 (배포 평가)

```bash
cd 3_pose_models/DINObotPose3/Eval
# 예: kinect 배포 (dump → RC)
CUDA_VISIBLE_DEVICES=GPU-<uuid> python selfbbox_eval.py \
  --stage1-detector ../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth \
  --stage1-angle ../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth \
  --stage1-rot ../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth \
  --crop-detector ../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth \
  --crop-angle ../TRAIN/outputs_selftrain/kinect_lightstack_20260705_003552/best_selftrain_head.pth \
  --rot-head ../TRAIN/outputs_selftrain/kinect_lightstack_20260705_003552/best_selftrain_rot.pth \
  --bbox-from-solved --bbox-guard --cov-pnp --dark-decode --frac-range 0.7 1.0 \
  --val-dir ../Dataset/Converted_dataset/DREAM_real/panda-3cam_kinect360 \
  --max-frames 1000 --dump-npz rc_dumps/kinect_final.npz
python rc_refine_from_dump.py --dump rc_dumps/kinect_final.npz \
  --val-dir ../Dataset/Converted_dataset/DREAM_real/panda-3cam_kinect360 \
  --sam-checkpoint ../weights_sam/sam_vit_b_01ec64.pth --render-h 448
```
- **GPU는 UUID로 지정** (정수 인덱스 뒤엉킴). 유휴 UUID: 7ff6997b / 70a2a406.
- 결정적 샘플링(selfbbox_eval, EvalDataset) 사용. K 스케일링 자동.

## 대안 config (참고)

- **max-정확도 realsense**: `realsense_robuststack_*`(배포 head + self-train 2라운드) = 0.8254 정확도, 단 가림 강건성 없음(40% base). 강건성 불필요 시.
- **max-강건성 (단일 head 전 카메라)**: light head 직접 = 가림 곡선 0.812/…/0.429, real 정확도는 카메라별 −0.01~0.05.

## 채택 레버 (이 모델을 만든 것)

1. nvdiffrast+SAM render-and-compare (`rc_refine_from_dump.py`) — 07-03 SOTA
2. cov-PnP (`--cov-pnp`) — 무료
3. DARK 디코딩 (`decode_util.py`, `--dark-decode`) — 무료, 전 카메라 +0.005~0.017
4. light 가림-증강 head — 가림 강건성 (약한 증강이 강한 것보다 나음)
5. occ-aug→self-train 스택 (`selftrain_pseudo_rot.py --occlude-aug`) — 강건성+real적응

반증 목록(재실험 금지): [references/next_directions.md](references/next_directions.md) §3, [00_overview.md](00_overview.md).
