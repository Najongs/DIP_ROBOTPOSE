# 멀티로봇 DREAM detector — KUKA iiwa7 / Baxter-left (2026-07-09~10)

> 목적: Panda로 SOTA를 낸 DINObotPose3 **검출기(heatmap keypoint detector)**가 DREAM의 다른 로봇
> (**KUKA iiwa7**, **Baxter 좌완)에도 일반화되는지 — Panda 검출기에서 fine-tune 시 성능이 오르는지 확인.**
> 판정: KUKA ✅ 학습 완료(합성 test AUC 0.735), Baxter 🔄 학습 중.

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

### Baxter 좌완 — 🔄 학습 중 (`baxter_left_dream_detector_20260710_152926`)

동일 설정 5-GPU. 로그 `TRAIN/logs/baxter_left_detector_20260710_152926.log`. 완료 시 표 갱신.

---

## 재현

```bash
cd 3_pose_models/DINObotPose3/TRAIN
GPU_IDS=0,1,2,3,4 WANDB_MODE=offline bash run_train_kuka_detector.sh          # KUKA
GPU_IDS=0,1,2,3,4 WANDB_MODE=offline bash run_train_baxter_left_detector.sh   # Baxter 좌완
# STAMP=<...> 로 출력 디렉토리 고정 가능
```

## 남은 일 (검출기 그 다음)

검출기는 **파이프라인 첫 단계**일 뿐. Panda급 포즈 추정(ADD-AUC)까지 가려면 로봇별로 추가 필요:
1. **관절각/회전 head** (`train_angle.py`/`train_rotation.py`) — 로봇별 재학습.
2. **운동학 모델(FK)** — KUKA iiwa7 / Baxter의 DH·URDF FK 함수. 현재 `model_v4.panda_forward_kinematics`는 Panda 전용 → 로봇별 FK 필요(솔버·mesh RC·Kabsch가 전부 FK 의존).
3. **real 평가셋** — KUKA/Baxter real DREAM 데이터 확보 시에만 real SOTA 비교 가능.

→ 지금 단계 결론은 "**검출기는 타 로봇으로 일반화되고 fine-tune 이득 있음**"까지. 전체 포즈 SOTA 확장은 위 3개가 전제.

관련: [multi_robot.md](../data/multi_robot.md)(FR5/FR3/Meca 실촬영, 별개 트랙) · [FINAL_MODEL.md](../FINAL_MODEL.md)(Panda 배포) · [training.md](../training/training.md)
