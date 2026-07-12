# 멀티로봇 DREAM detector — KUKA iiwa7 / Baxter-left (2026-07-09~10)

> 목적: Panda로 SOTA를 낸 DINObotPose3 **검출기(heatmap keypoint detector)**가 DREAM의 다른 로봇
> (**KUKA iiwa7**, **Baxter 좌완)에도 일반화되는지 — Panda 검출기에서 fine-tune 시 성능이 오르는지 확인.**
> 판정: KUKA ✅ (합성 test 2D AUC 0.735) · Baxter ✅ (합성 test 2D AUC **0.817**). 둘 다 Panda 검출기 transfer로 학습.
> 공통 발견: 높은 평균 L2는 품질 문제가 아니라 **link-identity 혼동 tail**(catastrophic의 90%+가 타 키포인트로 스냅) — 솔버가 복구할 유형.

---

## ⚠️ 지표·데이터 구분 (혼동 금지)

- 여기 AUC는 **2D 키포인트 PCK-AUC (합성 test_dr)** — 검출기 학습 지표. Panda의 헤드라인
  **real ADD-AUC@100mm 0.804**(포즈 지표, real 4-split)와 **다른 지표·다른 데이터**다. 직접 비교 불가.
- **KUKA/Baxter는 local에 real 평가셋이 없다.** DREAM_real은 Panda 4split(azure/kinect360/realsense/orb)뿐.
  → KUKA/Baxter 수치는 전부 **합성 domain-randomized(DR) test**. 이건 "synth→synth 일반화 + fine-tune 이득"
  확인이지, real SOTA 주장이 아니다. (KUKA는 photorealistic test셋 `kuka_synth_test_photo`도 있어 향후 도메인갭 측정 가능.)

## 설정 (KUKA·Baxter 공통)

| 항목 | 값 |
|---|---|
| 스크립트 | `TRAIN/run_train_kuka_detector.sh` / `run_train_baxter_left_detector.sh` |
| 학습 코드 | `train_heatmap.py` (torchrun DDP, 5-GPU: `GPU_IDS=0,1,2,3,4`) |
| 초기 가중치 | Panda 검출기 `outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth` (transfer) |
| 백본 | frozen DINOv3 + **상위 2블록 unfreeze** (`--unfreeze-blocks 2`) |
| 입력/히트맵 | 512 / 512, `--crop-to-robot --crop-margin 1.5` |
| 증강 | `--aug-level strong`, occlusion `prob 0.3 size 0.18` |
| 최적화 | 20 epoch, batch 16×5=80, lr 2e-4 / backbone 2e-5 / min 1e-7 |
| wandb | offline (`dinov3-dream-{kuka,baxter}-detector`) |

**키포인트(각 로봇 7개, DREAM 라벨과 exact-match)**
- KUKA: `iiwa7_link_1..7`
- Baxter 좌완: `left_s0, left_s1, left_e0, left_e1, left_w0, left_w1, left_w2`
- 매칭: `dataset.py`가 exact(priority 3) > substring-visual(2) > substring-collision(1)로 해소 →
  `iiwa7_link_5` vs `iiwa7_link_5 (collision)`, `left_w0` vs `left_w0_fixed/…_to_itb_fixed` 혼동 없음.

**데이터 (`datasets/synthetic/`)**
| 로봇 | train | test |
|---|---|---|
| KUKA | `kuka_synth_train_dr` (104,977) | `kuka_synth_test_dr` (5,997) |
| Baxter | `baxter_synth_train_dr` (104,982) | `baxter_synth_test_dr` (5,982) |

---

## 결과

### KUKA iiwa7 — ✅ 완료 (`kuka_dream_detector_20260709_183119`)

20 epoch, AUC **0.6208 → 0.7354** 단조 상승(19ep에서도 미세 상승 = 완전 수렴 전, 더 돌리면 소폭 여지).

| | Epoch 0 | Epoch 10 | Epoch 19 (best) |
|---|---|---|---|
| Val Loss | 0.0645 | 0.0476 | **0.0450** |
| 2D AUC | 0.6208 | 0.7236 | **0.7354** |
| L2 (px) | 115.6 | 107.1 | **105.5** |
| PCK@2.5 / @5 / @10 | — | — | **0.573 / 0.734 / 0.818** |

- 체크포인트: `outputs_heatmap/kuka_dream_detector_20260709_183119/{best,last}_heatmap.pth`
- 해석: Panda 검출기에서 시작해 KUKA 키포인트로 fine-tune → AUC +0.11. **transfer가 유효**(from-scratch 대비 빠른 수렴, epoch0부터 0.62).

#### L2 진단 — "높은 평균 L2(105px)"는 품질 문제가 아니라 **link-identity 혼동 tail**

test_dr 800프레임 per-keypoint 분해(`scratchpad/kuka_perkp.py`, hard-argmax 디코드):

| keypoint | mean | **median** | p99 | PCK@10 | >50px |
|---|---|---|---|---|---|
| iiwa7_link_1 | 99.1 | 1.6 | 1587 | 78% | 15.1% |
| iiwa7_link_2 | 87.0 | 1.9 | 1433 | 79% | 13.9% |
| iiwa7_link_3 | 69.7 | 2.4 | 1220 | 77% | 14.7% |
| iiwa7_link_4 | 72.0 | 3.0 | 1160 | 78% | 11.4% |
| iiwa7_link_5 | 30.7 | 1.9 | 507 | 85% | 8.6% |
| iiwa7_link_6 | 25.0 | 2.3 | 404 | 84% | 7.5% |
| iiwa7_link_7 | 33.7 | 1.7 | 554 | 85% | 6.5% |
| **ALL** | **60.0** | **2.1** | 1144 | 81% | 11.2% |

- **중앙값 2.1px** — 대부분 키포인트는 정밀하게 잡힌다. 평균을 끌어올리는 건 **catastrophic(>50px) 11.2%**(전체 오차 질량의 **94%**).
- **catastrophic의 90%가 다른 키포인트 GT로 스냅** = link-identity 혼동. 근위 링크(link_1~3)가 특히 심하고(자기가림 잦음), 원위/손목(link_5~7)으로 오검(예: link_1 catastrophic 118건 중 L7:42, L6:23). iiwa7의 닮은 원통형 7마디 + base 자기가림이 원인.
- **함의**: (a) 검출기 자체는 우수(median 2px), (b) 더 학습·백본 교체로 안 풀리는 2D 본질적 모호성, (c) **DINObotPose3 솔버(conf-gate + cov-PnP)가 Panda에서 이미 처리하는 유형** — link이 엉뚱한 마디로 스냅되면 재투영 outlier로 튀어 robust PnP가 기각. 즉 **raw 2D AUC 0.735는 최종 포즈 정확도를 과소평가**하며, 솔버 단계에서 대부분 복구될 것으로 기대. (Baxter는 arm이 크게 벌어져 혼동이 적음 → L2 30px대, 아래 참조.)

### Baxter 좌완 — ✅ 완료 (`baxter_left_dream_detector_20260710_152926`)

20 epoch, AUC **0.7239 → 0.8174** (KUKA보다 훨씬 높고, **epoch 0(0.724)부터 KUKA 최종 수준** — transfer가 Baxter에서 더 잘 먹힘).

| | Epoch 0 | Epoch 10 | Epoch 19 (best) |
|---|---|---|---|
| Val Loss | 0.0551 | 0.0377 | **0.0346** |
| 2D AUC | 0.7239 | 0.8063 | **0.8174** |
| L2 (px) | 30.2 | 22.3 | **21.4** |

- 체크포인트: `outputs_heatmap/baxter_left_dream_detector_20260710_152926/{best,last}_heatmap.pth`

#### L2 진단 — KUKA와 **같은 혼동 메커니즘, 정반대 위치**

test_dr 800프레임 per-keypoint 분해(`scratchpad/baxter_perkp.py`):

| keypoint | mean | **median** | p99 | PCK@10 | >50px | in/tot |
|---|---|---|---|---|---|---|
| left_s0 (어깨) | 4.6 | 1.0 | 102 | 96% | 1.6% | 800/800 |
| left_s1 | 5.1 | 1.3 | 117 | 96% | 1.8% | 800/800 |
| left_e0 | 7.3 | 1.4 | 200 | 94% | 3.1% | 800/800 |
| left_e1 (팔꿈치) | 10.4 | 1.6 | 208 | 91% | 4.6% | 788/800 |
| left_w0 | 13.5 | 1.8 | 314 | 88% | 5.4% | 777/800 |
| left_w1 | 61.7 | 1.8 | 782 | 82% | 13.8% | 683/800 |
| left_w2 (손목끝) | 78.4 | 2.3 | 1120 | 76% | 16.1% | 670/800 |
| **ALL** | **23.9** | **1.5** | 643 | 89% | 6.2% | — |

- 중앙값 **1.5px**, catastrophic(>50px) **6.2%**(전체 오차의 89%). link-혼동 **93%**. → KUKA와 동일한 tail-driven 구조지만 **전체적으로 더 깨끗**(KUKA cat 11.2% / mean 60px 대비).
- **혼동 위치가 KUKA와 반대**: KUKA는 근위(base) 링크가 혼동, Baxter는 **원위(손목 w1·w2)**가 혼동(cat 14~16%, off-frame도 잦아 in-frame 683/670). 어깨(s0/s1)는 크고 고정돼 거의 완벽(cat 1.6%).
- **통합 해석**: 두 로봇 모두 "**팔에서 작고·자기유사하고·자주 가려지거나 off-frame 되는 끝단**"이 혼동된다. KUKA는 base(Allegro hand가 있는 원위가 특징적 → base가 상대적으로 헷갈림), Baxter는 wrist(어깨가 특징적 → wrist가 헷갈림). **어느 쪽이든 솔버의 운동학 체인 + conf-gate + cov-PnP가 재투영 outlier로 걸러 복구할 유형** — Panda에서 검증된 그 메커니즘.

---

## KUKA 포즈 파이프라인 준비 (2026-07-11) — FK + 각도/회전 head 배선

검출기 다음 단계(관절각→FK→솔버)를 KUKA에 대해 **준비 완료**(학습은 미실행).

### iiwa7 FK — DREAM 데이터로 피팅·검증 (`model_v4.iiwa7_forward_kinematics`)
- DREAM kuka는 **표준 iiwa7 URDF와 링크 길이가 다름**(측정 오프셋 [0.15,0.19,0.21,0.19,0.21,0.1995,0.1012]m). 그래서 표준값 대신 **데이터로 fixed joint transform을 피팅**(scipy LM, sim_state 관절각↔키포인트 3D). J1은 물리 base(0,0,0.15)로 고정. 유도·검증 스크립트: `Eval/iiwa7_fk_fit.py`.
- **검증: link_1..7 원점 재현 RMS = train/held-out/test 모두 0.003mm** (합성 라벨 정밀도 한계). torch 버전 test셋 0.005mm, 미분가능.
- 손목 J6/J7이 복합 오프셋(iiwa 특유 0.0607m wrist offset)이라 단순 축-정렬 가정이 실패했고, 피팅이 이를 자동 복원.
- ⚠️ 위치는 정확하나 **중간 프레임 방향은 gauge**(위치만 관측 가능) — 솔버/PnP엔 무해, **mesh render-and-compare 전엔 재검토** 필요.
- joint_7은 link_7 원점을 안 움직여(자기축 회전) Panda처럼 **6각도 예측 + joint7=0**으로 키포인트 위치 정확.

### 배선 (로봇 무관화)
- `dataset.py`: **관절각 로딩 버그 수정** — 기존 `sim_state.joints[:7]`은 KUKA에서 `iiwa7_base_link_iiwa7_joint`(고정 0)를 먼저 잡아 **θ7 누락**. `angle_joint_names` 인자로 이름 기반 선택(None=기존 Panda 동작 유지). 검증: raw JSON joint_1..7과 일치.
- `train_angle.py`·`train_rotation.py`: `--fk-robot {panda,kuka,...}` + `--angle-joint-names`로 FK·GT각도 선택.
- 스모크 테스트: angle 경로(forward→iiwa7 FK loss→backward, grad_norm 81 OK), rotation 경로(Kabsch GT 라벨 재투영 잔차 0.00mm) 통과.
- 스크립트: `run_train_kuka_{angle,rotation}.sh` (로컬, frozen backbone 단일 GPU, UUID 선택).

## 재현

```bash
cd 3_pose_models/DINObotPose3/TRAIN
# 검출기 (5-GPU)
GPU_IDS=0,1,2,3,4 WANDB_MODE=offline bash run_train_kuka_detector.sh
GPU_IDS=0,1,2,3,4 WANDB_MODE=offline bash run_train_baxter_left_detector.sh
# KUKA 각도/회전 head (단일 GPU, UUID)
GPU=GPU-<uuid> bash run_train_kuka_angle.sh
GPU=GPU-<uuid> bash run_train_kuka_rotation.sh
```

### 솔버 연결 (2026-07-12, `Eval/kuka_add_eval.py`)
meca_add_eval.py 방식 — 배포 SOTA 솔버(`solve_pose_kinematic`)를 **수정 없이 monkeypatch**(FK/limits/mean을 iiwa7로 교체). ADD-AUC@100mm 측정. **head 학습 중(Ep~15) 예비 측정으로 파이프라인 검증 + 최적 모드 발견**:

| 검증 | 결과 |
|---|---|
| **구조 정합** (GT각도+GT포즈) | **ADD 0.00mm, AUC 0.9999** — FK·단위·대응 전부 정상 |
| 솔버 각도정제 (oracle 2D+R에서도) | ❌ **발산** (J2 7.5°→25°) — link-혼동이 재투영을 잘못된 basin으로. **Meca 선례 "솔버 각도정제 금지"와 동일** |
| head-direct (t 재투영 재계산) | ❌ ADD 632mm — R 9°오차+깊이 모호성으로 **t가 깊이 발산** |
| **`--direct-pose`** (head각도 + rot-head R,t 직접) | ✅ **ADD-AUC 0.22, median 82mm** (검출 2D, Ep~15 미수렴) |

- **채택 모드 = `--direct-pose`**: 각도정제·t재계산 둘 다 끄고 head 각도 + rot-head R,t를 직접 신뢰. iiwa7엔 재투영 최적화가 해로움(link-혼동 + 깊이 모호성).
- 현 병목은 **미수렴 rot-head**(R 8.9°, t-err 82mm) — GT각도든 head각도든 ADD 동일(97 vs 100mm)이라 **각도는 이미 충분**, rot-head 수렴 시 ADD 상승 예상.

### 최종 ADD (2026-07-12, rot-head 30ep 수렴, full 5997 test)

| 모드 | ADD-AUC@100mm | mean | median |
|---|---|---|---|
| 예비 (Ep15 미수렴) | 0.22 | 99.7mm | 82mm |
| **최종 `--direct-pose`** | **0.34** | **77.2mm** | **63.8mm** |

- rot-head 수렴(t-err 82→**56mm**, geo 8.9→**7.4°**)으로 **0.22 → 0.34**.
- **oracle-2D == detected-2D (완전 동일)**: direct-pose는 2D 키포인트를 안 씀(head 각도+rot-head R,t 직접) → **검출기 link-혼동을 완전히 우회**. 진단이 예측한 "혼동 복구"가 "혼동에 안 걸리는 경로로 우회"로 실현.
- **각도는 병목 아님**(GT각도=head각도 ADD 동일, MAE 6.6°). **병목 = rot-head R,t**(7.4°/56mm)가 ADD를 직접 결정.
- **천장 = rot-head 품질**. 넘으려면 **render-and-compare 깊이 보정**(Panda 0.7→0.80 레버) 필요 — 단 iiwa7 mesh RC엔 FK 방향 gauge 정리 선행.
- ⚠️ 합성 test·RC 없음 → Panda real 0.80과 직접 비교 불가. KUKA 포즈 파이프라인 **end-to-end 작동 확인**이 의의.

## 남은 일

1. ✅ **iiwa7 FK** — 완료·검증(0.003mm). Baxter FK는 미구현(동일 방식으로 피팅 가능).
2. ✅ **관절각/회전 head 배선** — 완료·스모크 통과. head 학습 진행 중(angle 60ep/rot 30ep).
3. ✅ **솔버 연결 + 최종 ADD** — `kuka_add_eval.py`, 구조 검증 0.00mm, `--direct-pose` **최종 ADD-AUC 0.34**(median 64mm, 수렴).
4. **rot-head 품질 개선** — 현 천장(7.4°/56mm)이 ADD 병목. render-and-compare 깊이 보정이 다음 레버.
5. **mesh RC 전 FK 방향 gauge 정리** (iiwa7 URDF 메쉬 정합 시) — 4번의 선행조건.
6. **real 평가셋** — KUKA/Baxter real DREAM 확보 시에만 real SOTA 비교 가능.

→ 검출기 일반화 + KUKA 포즈 파이프라인(FK·head·솔버) **연결·검증·측정 완료** (ADD-AUC 0.34). 병목은 rot-head, 다음 레버는 render-and-compare.

### KUKA 포즈 개선(A) 조사 — mesh render-and-compare **에셋 부재로 차단** (2026-07-13)
- **render-and-compare에 필요한 iiwa7/Allegro 메쉬가 repo·시스템 어디에도 없음** (`ViS/`엔 Panda·Meca만). DREAM kuka 모델(URDF+mesh) 확보 없이는 mesh RC 불가. 이 환경에선 다운로드 불가 → **차단**.
- 메쉬-불필요 대안 **robust cov-PnP**(FK(head각도) vs 검출 2D)도 **link-혼동으로 완전 발산**(median 1074mm). 검출 2D의 confident-wrong를 PnP가 못 걸러냄.
- 결론: **현 도구로 KUKA 포즈 천장 = direct-pose 0.34**. 넘으려면 (a) DREAM kuka 메쉬 확보 → 올바른 URDF FK + mesh RC, 또는 (b) rot-head t 개선(현 56mm, plateau). 둘 다 외부 에셋/추가 연구 필요.

### Baxter 포즈 파이프라인 (2026-07-13, main f974c4e)
- ✅ **baxter-left FK** — 40-start scipy 피팅(single-start는 local minima), left_s0..w2 재현 **0.003mm**(train/held-out/test). `Eval/baxter_fk_fit.py`, `model_v4.baxter_left_forward_kinematics`.
- ✅ 배선(`--fk-robot baxter`) + `run_train_baxter_{angle,rotation}.sh` + `Eval/baxter_add_eval.py`. 스모크 통과.
- 🔄 **angle+rotation head 학습 중**(GPU0/1). 수렴 시 `--direct-pose`로 Baxter ADD 측정 예정.

관련: [multi_robot.md](../data/multi_robot.md)(FR5/FR3/Meca 실촬영, 별개 트랙) · [FINAL_MODEL.md](../FINAL_MODEL.md)(Panda 배포) · [training.md](../training/training.md)
