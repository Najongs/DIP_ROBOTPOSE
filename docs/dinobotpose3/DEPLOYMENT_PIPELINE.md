# DINObotPose3 공통 추론·RC 파이프라인

최종 갱신: 2026-07-24

문서 authority 경로: `/home/najo/NAS/DIP/docs/dinobotpose3/`

전체 테스트셋별 evaluator와 실행 명령은
[FULL_TEST_EVALUATION.md](FULL_TEST_EVALUATION.md)를 기준으로 한다.

이 문서는 Panda, KUKA iiwa7, Baxter에 동일한 추론 방법론을 적용하기 위한
운영 기준이다. 로봇마다 detector/head/FK/mesh는 다르지만 처리 순서와 최적화
원칙은 동일하게 유지한다.

현재 우선 대상은 **Panda, KUKA, Baxter**다. FR3, FR5, Meca500의 자산과
체크포인트는 보존하지만 이 파이프라인의 구현·평가 범위에서는 보류한다.

## 1. 확정된 추론 순서

1. 원본 RGB와 해당 카메라의 intrinsic `K`를 입력한다.
2. 로봇별 full-frame detector로 거친 2D keypoint와 confidence를 예측한다.
3. confidence gate를 통과한 2D keypoint의 min/max로 bbox를 만든다.
   이것이 `bbox-from-2d`이며 pose로 bbox를 만드는 방식은 기본 경로가 아니다.
4. bbox에 로봇 전체가 포함되도록 margin을 적용한다. 기본 권장값은
   Panda/KUKA `1.3~1.5`, Baxter `1.3~1.5`다.
5. bbox를 crop/ROI-align하고 crop 좌표계에 맞게 `K`의 principal point와
   scale을 갱신한다.
6. 로봇별 crop detector로 정밀 2D keypoint를 다시 예측한다.
7. DARK decode로 heatmap peak를 sub-pixel 좌표로 변환한다.
8. 같은 crop feature에서 angle head가 초기 관절각 `theta0`, rotation head가
   초기 base-to-camera 회전 `R0`를 예측한다.
9. 로봇별 FK(`theta0`)로 robot-frame 3D keypoint를 생성한다.
10. 2D–3D 대응점, confidence, heatmap covariance와 `K`를 사용해
    cov-PnP/pose initialization을 수행한다. rotation head의 `R0`는 회전 basin
    초기값 또는 prior로 사용한다.
11. confidence-weighted reprojection loss로 `theta, R, t`를 정제한다.
12. 로봇별 canonical mesh를 렌더링하고 실제 robot mask와 비교하는 RC로
    `theta, R, t`를 최종 정제한다. reprojection anchor와 do-no-harm gate 없이
    silhouette만 자유 최적화하는 것은 금지한다.

### Translation head 처리

Rotation head 구현에는 translation 출력이 함께 있지만, 공통 배포 설계에서는
translation head를 최종값으로 신뢰하지 않는다. `t`는 PnP와 재투영 최적화에서
계산한다. 기존 일부 evaluator는 과거 호환성과 초기 smoke test를 위해 translation
head를 초기값으로 사용한다. 공통 runner로 통합할 때 이 부분은 PnP 초기값으로
교체해야 한다.

## 2. 현재 구현 상태

| 항목 | Panda | KUKA | Baxter 17kp |
|---|---|---|---|
| full-frame detector | 있음 | 추후 학습 | 추후 학습 |
| crop detector | 있음 | 있음 | 있음 |
| angle head | 있음 | 있음 | 12-angle head 있음 |
| rotation head | 있음 | 있음 | 있음 |
| FK/solver | 있음 | 있음 | 17kp FK/solver 있음 |
| canonical mesh renderer | 있음 | 있음 | 2026-07-24 구현 |
| RC evaluator | 있음 | 있음 | 2026-07-24 구현 |

KUKA와 Baxter는 full-frame detector가 준비되기 전까지 crop detector를 전체
letterbox image에 한 번 적용해 bbox를 얻고, 같은 detector를 crop에 다시 적용하는
임시 경로를 사용할 수 있다. 첫 pass에서 로봇 scale이 학습 분포보다 작아질 수
있으므로 최소 keypoint 수와 confidence gate가 필요하다. 이 임시 경로는
full-frame detector 학습을 대체하는 최종 설계가 아니다.

## 3. 체크포인트

canonical checkpoint index:

```text
TRAIN/checkpoints/README.md
```

### Panda

```text
TRAIN/checkpoints/panda/detector_full.pth
TRAIN/checkpoints/panda/detector_crop.pth
TRAIN/checkpoints/panda/angle_final.pth
TRAIN/checkpoints/panda/rotation_final.pth
```

네 실사 카메라 모두 같은 synthetic angle/rotation head를 사용한다. 카메라별로
달라지는 것은 입력과 intrinsic `K`다. camera-adapted head는 primary 모델이 아니며
다음 위치에 격리되어 있다.

```text
TRAIN/checkpoints_experimental/panda_camera_adapted/
```

### KUKA

```text
TRAIN/checkpoints/kuka/detector.pth
TRAIN/checkpoints/kuka/angle.pth
TRAIN/checkpoints/kuka/rotation.pth
```

### Baxter 17kp

```text
TRAIN/checkpoints/baxter/fullbody_17kp/detector.pth
TRAIN/checkpoints/baxter/fullbody_17kp/angle.pth
TRAIN/checkpoints/baxter/fullbody_17kp/rotation.pth
```

Baxter primary 모델은 과거 left-arm 7kp가 아니라 다음 순서의 whole-body 17kp다.

```text
torso_t0,
left_s0, left_s1, left_e0, left_e1, left_w0, left_w1, left_w2, left_hand,
right_s0, right_s1, right_e0, right_e1, right_w0, right_w1, right_w2, right_hand
```

angle head는 좌우 `s0,s1,e0,e1,w0,w1`의 총 12개 관절을 예측한다. 양쪽 `w2`는
현재 keypoint parameterization에서 관측되지 않으므로 0으로 고정한다.

## 4. FK와 mesh/renderer 경로

### Panda

```text
TRAIN/model_v4.py                         # panda_forward_kinematics
Eval/render_nvdr.py                       # differentiable Panda renderer
Eval/rc_refine_from_dump.py
Eval/rc_refine_wrist.py
ViS/Panda/meshes/
```

### KUKA

```text
TRAIN/model_v4.py                         # iiwa7 fitted keypoint FK
Eval/iiwa7_render.py                      # canonical URDF mesh FK/renderer
Eval/iiwa7_rc_eval.py
RoboPEPP/urdfs/iiwa_description/
```

KUKA의 geometric solver에는 `refine_eval.geometric_K()`로 복원한 실제 focal을
사용해야 한다. synthetic KUKA/Baxter JSON의 identity `K` fallback을 geometry에
사용하면 안 된다.

### Baxter 17kp

```text
TRAIN/model_v4.py                         # baxter_forward_kinematics (17kp position FK)
Eval/baxter_fullbody_add_eval.py          # 17kp reprojection solver/eval
Eval/baxter_fullbody_render.py            # canonical full-body renderer
Eval/baxter_fullbody_rc_eval.py           # reprojection + SAM + full-body RC
RoboPEPP/urdfs/Baxter/baxter_description/ # torso/arm canonical URDF meshes
_assets_src/baxter_common/rethink_ee_description/
                                              # electric gripper meshes
```

`model_v4.baxter_forward_kinematics()`는 keypoint 위치에는 정확하지만 intermediate
frame orientation은 fitting gauge이므로 mesh를 직접 매달면 안 된다.
`baxter_fullbody_render.py`는 공식 URDF frame을 사용하고 좌우 arm을 DREAM 17kp
공통 gauge에 고정 강체변환으로 배치한다.

현재 renderer 구성:

- frame 17개: torso 1 + left 8 + right 8
- mesh: torso + 양팔 7-link + 양 electric gripper
- 468,497 vertices / 403,674 faces
- DREAM test 17kp 정합: 약 `0.011 mm RMS`

## 5. 실행 방법

모든 명령은 다음 디렉터리에서 실행한다.

```bash
cd /home/najo/NAS/DIP/3_pose_models/DINObotPose3/Eval
```

### Baxter FK 및 renderer self-test

CPU FK 검증:

```bash
python baxter_fullbody_render.py --max-frames 200
```

GPU mesh posing/nvdiffrast 초기화까지 확인:

```bash
python baxter_fullbody_render.py --max-frames 10 --render-smoke
```

기대 조건:

- 17kp RMS가 0.05mm 미만
- `render mesh: 468497 vertices, 403674 faces` 출력

### Baxter end-to-end smoke test

```bash
python baxter_fullbody_rc_eval.py \
  --max-frames 1 \
  --reproj-iters 1 \
  --rc-iters 1
```

2026-07-24 확인값은 한 프레임에서 다음과 같다. 이는 성능 보고값이 아니라 wiring
검사용이다.

```text
before ADD mean 69.3mm
after  ADD mean 64.9mm
SAM/render IoU 0.724
```

### Baxter 평가 실행

작은 검증:

```bash
python baxter_fullbody_rc_eval.py \
  --max-frames 100 \
  --reproj-iters 150 \
  --rc-iters 60
```

전체 데이터:

```bash
python baxter_fullbody_rc_eval.py \
  --max-frames 5997 \
  --reproj-iters 150 \
  --rc-iters 60
```

현재 full-body visual mesh는 약 40만 triangle이므로 전체 평가는 오래 걸릴 수 있다.
속도가 문제가 되면 원본을 수정하지 말고 silhouette를 보존하는 RC 전용 decimated
mesh를 별도 생성한 뒤 GT-mask IoU가 유지되는지 다시 검증한다.

### KUKA FK/mesh 검증

```bash
python iiwa7_render.py \
  --data /home/najo/NAS/DIP/datasets/synthetic/kuka_synth_test_dr \
  --n 300
```

KUKA RC 옵션과 배포 gate의 근거는 `Eval/iiwa7_rc_eval.py` 상단 문서를 기준으로
한다. 특히 `geometric_K`, reprojection anchor, adoption gate를 유지한다.

### Panda bbox-from-2d 기존 runner

Panda의 두 단계 crop 실험은 다음 파일에서 확인한다.

```text
Eval/selfbbox_eval.py
```

주요 옵션:

```text
--stage1-detector
--crop-detector
--crop-angle
--rot-head
--bbox-conf
--margin
--cov-pnp
--dark-decode
```

현재 파일에는 과거 `bbox-from-solved` 실험 옵션도 남아 있다. 공통 최종 설계에서는
직접 2D keypoint bbox를 기본으로 사용한다.

## 6. 구현 시 지켜야 할 공통 규칙

1. 로봇마다 별도 head를 사용해도 되지만 카메라별 head는 primary로 두지 않는다.
2. 카메라별 차이는 RGB와 `K`로 처리한다.
3. bbox는 confidence가 유효한 2D keypoint로 만든다.
4. crop 이후 `K`를 반드시 crop 좌표계로 변환한다.
5. DARK/covariance/confidence gate를 모든 로봇에 같은 방식으로 적용한다.
6. rotation head는 유지하되 translation head를 최종 pose로 사용하지 않는다.
7. PnP 결과만으로 끝내지 않고 관절 제약을 포함한 재투영 정제를 수행한다.
8. RC는 모든 우선 대상 로봇에 적용하되 silhouette-only free optimization은 금지한다.
9. position-fit FK와 canonical mesh FK를 혼동하지 않는다.
10. RC 결과는 init보다 나빠지지 않도록 reprojection/IoU 기반 adoption guard를 둔다.

## 7. 남은 작업

- KUKA full-frame detector 학습
- Baxter full-frame 17kp detector 학습
- Panda/KUKA/Baxter를 하나의 robot-config 기반 runner로 통합
- target pipeline에서 translation-head init을 cov-PnP `t`로 완전히 교체
- Baxter RC의 full-set 평가와 adoption gate 선정
- 필요 시 Baxter RC 전용 mesh 경량화 및 GT-mask IoU 재검증

FR3, FR5, Meca500은 위 세 로봇의 공통 runner가 안정화된 뒤 같은 인터페이스로
추가한다.
