# DINObotPose3 전체 테스트셋 평가 기준

최종 갱신: 2026-07-24

문서 authority 경로: `/home/najo/NAS/DIP/docs/dinobotpose3/`

이 문서는 Panda, KUKA iiwa7, Baxter의 전체 테스트 데이터셋 평가 entrypoint와
보고 기준을 고정한다. subset 진단값과 full-test headline 수치를 혼용하지 않는다.

모든 명령은 다음 위치에서 실행한다.

```bash
cd /home/najo/NAS/DIP/3_pose_models/DINObotPose3/Eval
```

## 1. 평가 결과 표기 원칙

1. 기본 지표는 `ADD-AUC@100mm`, mean ADD, median ADD다.
2. `--max-frames 0`을 지원하는 evaluator에서는 0을 사용해 전체셋을 평가한다.
3. stride/subset 결과는 headline full-test 결과로 보고하지 않는다.
4. Panda는 현재 full-frame `bbox-from-2d` end-to-end 평가가 가능하다.
5. KUKA와 Baxter는 full-frame detector가 아직 없으므로 현재 결과를
   **GT-keypoint crop pose/RC 평가**라고 명시한다.
6. RC 결과는 반드시 RC 전 baseline과 함께 보고한다.
7. adoption gate가 full set에서 검증되지 않은 RC 결과는 headline이 아니라
   experimental 결과로 분리한다.
8. synthetic KUKA/Baxter geometry에는 identity fallback `K`를 사용하지 않는다.
   `refine_eval.geometric_K()`로 실제 focal과 crop principal point를 복원한다.

## 2. 전체 데이터셋 목록

| 로봇 | 도메인 | 경로 | JSON 수 |
|---|---|---|---:|
| Panda | synth DR | `datasets/ICRA_multiview/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr` | 5,997 |
| Panda | synth Photo | `datasets/ICRA_multiview/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_photo` | 5,997 |
| Panda | Realsense | `datasets/ICRA_multiview/Converted_dataset/DREAM_to_DREAM/panda-3cam_realsense` | 5,944 |
| Panda | Kinect360 | `datasets/ICRA_multiview/Converted_dataset/DREAM_to_DREAM/panda-3cam_kinect360` | 4,966 |
| Panda | Azure | `datasets/ICRA_multiview/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure` | 6,394 |
| Panda | ORB | `datasets/ICRA_multiview/Converted_dataset/DREAM_to_DREAM/panda-orb` | 32,315 |
| KUKA | synth DR | `datasets/synthetic/kuka_synth_test_dr` | 5,997 |
| KUKA | synth Photo | `datasets/synthetic/kuka_synth_test_photo` | 5,999 |
| Baxter | synth DR | `datasets/synthetic/baxter_synth_test_dr` | 5,982 |

현재 workspace에는 Baxter synth Photo 데이터셋이 없다.

## 3. Panda 전체 평가

### 3.1 사용 파일

```text
Eval/selfbbox_eval.py
Eval/rc_refine_from_dump.py
```

Panda는 모든 카메라에 동일한 canonical synthetic head를 사용한다.

```text
TRAIN/checkpoints/panda/detector_full.pth
TRAIN/checkpoints/panda/detector_crop.pth
TRAIN/checkpoints/panda/angle_final.pth
TRAIN/checkpoints/panda/rotation_final.pth
```

### 3.2 Base: bbox-from-2d + crop + solver

```bash
DATASET=/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr
OUT=/home/najo/NAS/DIP/3_pose_models/DINObotPose3/Eval/full_test_dumps/panda_synth_dr.npz

mkdir -p "$(dirname "$OUT")"

python selfbbox_eval.py \
  --stage1-detector ../TRAIN/checkpoints/panda/detector_full.pth \
  --crop-detector ../TRAIN/checkpoints/panda/detector_crop.pth \
  --crop-angle ../TRAIN/checkpoints/panda/angle_final.pth \
  --rot-head ../TRAIN/checkpoints/panda/rotation_final.pth \
  --val-dir "$DATASET" \
  --max-frames 0 \
  --dark-decode \
  --cov-pnp \
  --dump-npz "$OUT"
```

데이터셋과 출력 이름만 바꿔 다음 6개 전체셋에 반복한다.

```text
panda_synth_test_dr
panda_synth_test_photo
panda-3cam_realsense
panda-3cam_kinect360
panda-3cam_azure
panda-orb
```

현재 확정 방법론에서는 다음 옵션을 사용하지 않는다.

```text
--bbox-from-solved
--frac-range
```

`--bbox-from-solved`는 pose-derived bbox 실험이고, `--frac-range`는 subset 제한이다.

### 3.3 Panda RC

```bash
python rc_refine_from_dump.py \
  --dump "$OUT" \
  --val-dir "$DATASET" \
  --sam-checkpoint ../weights_sam/sam_vit_b_01ec64.pth \
  --render-h 448 \
  --max-frames 0
```

카메라마다 다른 render resolution을 선택하지 않고 우선 `448`로 통일한다.
변경이 필요하면 모든 split에 같은 설정으로 재평가한다.

### 3.4 사용하지 않을 기존 runner

```text
Eval/verify_sota.sh
```

이 스크립트는 다음 이유로 현재 headline 재평가에 사용하지 않는다.

- 카메라별 adapted/self-training head 사용
- `bbox-from-solved` 사용
- `--frac-range 0.7 1.0`로 마지막 30%만 평가
- `--max-frames 800` subset 평가
- 현재 workspace와 다른 과거 dataset 경로 사용

과거 결과 재현용으로만 취급한다.

## 4. KUKA 전체 평가

### 4.1 사용 파일과 체크포인트

```text
Eval/iiwa7_rc_eval.py
TRAIN/checkpoints/kuka/detector.pth
TRAIN/checkpoints/kuka/angle.pth
TRAIN/checkpoints/kuka/rotation.pth
```

`iiwa7_rc_eval.py`는 true-K solver baseline과 RC 결과를 한 번에 출력한다.

### 4.2 DR 전체

```bash
python iiwa7_rc_eval.py \
  --mode rc \
  --detector ../TRAIN/checkpoints/kuka/detector.pth \
  --angle-head ../TRAIN/checkpoints/kuka/angle.pth \
  --rot-head ../TRAIN/checkpoints/kuka/rotation.pth \
  --val-dir /home/najo/NAS/DIP/datasets/synthetic/kuka_synth_test_dr \
  --max-frames 0 \
  --init solver \
  --cov-pnp \
  --refine-rot \
  --refine-angles \
  --repro-w 5 \
  --adopt-iou-abs 0.83 \
  --min-iou 0 \
  --max-uv-shift 0
```

### 4.3 Photo 전체

위 명령의 `--val-dir`만 다음으로 바꾼다.

```text
/home/najo/NAS/DIP/datasets/synthetic/kuka_synth_test_photo
```

### 4.4 보고 방법

- `BEFORE (solver)`: true-K solver baseline
- `AFTER (RC)`: adoption gate가 적용된 RC
- 현재 KUKA detector가 crop 학습 모델이므로 결과에는 `GT-crop`을 명시한다.
- `--init direct`는 과거 translation-head/direct-pose A/B용이며 headline에 사용하지 않는다.

## 5. Baxter 17kp 전체 평가

### 5.1 사용 파일과 체크포인트

```text
Eval/baxter_fullbody_add_eval.py
Eval/baxter_fullbody_render.py
Eval/baxter_fullbody_rc_eval.py
TRAIN/checkpoints/baxter/fullbody_17kp/detector.pth
TRAIN/checkpoints/baxter/fullbody_17kp/angle.pth
TRAIN/checkpoints/baxter/fullbody_17kp/rotation.pth
```

Primary Baxter 모델은 과거 left-arm 7kp가 아니라 whole-body 17kp 모델이다.

### 5.2 Headline baseline: 17kp reprojection solver

```bash
python baxter_fullbody_add_eval.py \
  --detector ../TRAIN/checkpoints/baxter/fullbody_17kp/detector.pth \
  --angle-head ../TRAIN/checkpoints/baxter/fullbody_17kp/angle.pth \
  --rot-head ../TRAIN/checkpoints/baxter/fullbody_17kp/rotation.pth \
  --val-dir /home/najo/NAS/DIP/datasets/synthetic/baxter_synth_test_dr \
  --mode solver \
  --max-frames 0
```

현재 확정 headline은 이 solver 결과다. all-keypoint 17kp ADD-AUC를 사용한다.

### 5.3 Full-body RC

```bash
python baxter_fullbody_rc_eval.py \
  --val-dir /home/najo/NAS/DIP/datasets/synthetic/baxter_synth_test_dr \
  --max-frames 5982 \
  --reproj-iters 150 \
  --rc-iters 60
```

현재 `baxter_fullbody_rc_eval.py`의 `--max-frames`는 0=all semantics가 아니므로
전체 frame 수인 `5982`를 명시한다.

### 5.4 Baxter RC 보고 제한

2026-07-24 기준:

- canonical 17kp FK 정합: 약 `0.011mm RMS`
- torso+양팔+양 gripper GPU renderer smoke test 통과
- 1프레임 end-to-end smoke test 통과
- full-set adoption gate는 아직 선정하지 않음

따라서 full-set RC를 실행하더라도 gate 검증 전에는 experimental 결과로 보고한다.
baseline보다 나빠지는 frame을 복원하는 do-no-harm/adoption 규칙을 full set에서
정한 뒤 headline 승격 여부를 결정한다.

KUKA와 마찬가지로 현재 Baxter 평가도 detector dataset에서
`crop_to_robot=True`를 사용하므로 `GT-crop`임을 명시한다.

## 6. 권장 실행 순서

GPU 시간과 실패 비용을 고려해 다음 순서로 진행한다.

1. Panda 6개 split base 전체 평가
2. KUKA DR/Photo solver+RC 전체 평가
3. Baxter 17kp solver baseline 전체 평가
4. Panda RC 6개 split
5. Baxter full-body RC 전체 평가

각 전체 실행 전에 동일 명령으로 `--max-frames 10~50` smoke test를 수행한다.
단, 최종 표에는 smoke/subset 수치를 기록하지 않는다.

## 7. 최종 결과표에 필요한 필드

| 로봇 | split | bbox protocol | detector/head | base AUC | RC AUC | mean ADD | median ADD | N |
|---|---|---|---|---:|---:|---:|---:|---:|
| Panda | 각 6 split | detected `bbox-from-2d` | canonical |  |  |  |  |  |
| KUKA | DR/Photo | GT-crop | robot-specific |  |  |  |  |  |
| Baxter | DR | GT-crop | full-body 17kp |  | experimental |  |  |  |

결과를 기록할 때 command, Git/worktree 상태, checkpoint symlink target과 실행 날짜를
함께 보존한다.
