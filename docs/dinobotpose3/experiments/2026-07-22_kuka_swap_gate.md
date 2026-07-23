# 2026-07-22 — KUKA 격차 공략: recoverability/swap 게이트 → **detector-limited** 확정 + 검출기 재학습 착수

> **문제.** KUKA iiwa7(kuka_synth_test_dr, 7kp link_1..7)는 intrinsics 수정 후 solver+참K ADD-AUC **0.690**(median 13.1mm, R 5.77°)로 **유일하게 SOTA 미만**(RoboPEPP 0.762, gap −0.07). fail 15.9%(>100mm), mean 91.1=median×7, p99 ~1012mm → **격차는 전적으로 ~16% 파국 꼬리**. 선행 가정: 꼬리 = **link-identity 혼동**(iiwa 원통형 링크가 시각적으로 동일).
>
> **핵심 검증 질문(mission).** 꼬리가 (A) **swap이면 신호는 존재·엉뚱한 링크에 있음** → top-M 모드에서 skeleton-consistent decode로 정답 링크 선택 가능(Panda에선 死), 아니면 (B) **정답 모드 부재** → detector 문제.
>
> 재현: `Eval/_debate_tmp/kuka_gate/` (게이트 `kuka_swap_gate.py`, dump `kuka_gate_full.npz`, 꼬리분석 `analyze_tail.py`, A/B `kuka_ab.py`).
> 배포 검출기 `TRAIN/outputs_heatmap/kuka_dream_detector_20260709_183119/best_heatmap.pth`, angle `kuka_angle_20260712_060212`.

---

## TL;DR (결론 먼저)

1. **게이트 판정 = detector-limited (kinematic-decode 死).** 전체 39,173 valid kp 중 **22.95%가 파국**(배포 soft-argmax err >10px). 그중 **85.2%가 true hard-peak swap**(정답 링크 모드가 argmax에 부재). true swap에서:
   - top-3 모드 내 정답 회수 **7.2%** / top-5 **10.9%** (oracle 모드선택 err **median 40px**),
   - 정답 위치 heatmap 응답 `gt_resp = GT창 최대/peak` **median 0.113**(정답 위치가 peak의 ~11%).
   → **선택할 정답 모드가 없다.** Panda의 "정답 모드 부재→decode 死"와 **동일 서명**. mission의 선행가설(swap→skeleton decode)은 **게이트로 반증**.
2. **꼬리의 정체 = 확산(diffuse)·저신뢰 heatmap, "확신에 찬 오답"이 아님.** true-swap peak conf **0.04**(median 0.01) vs good **0.65**(median 0.71); good-p10(0.33) 넘는 swap은 **0.8%**. 즉 검출기는 **자신이 못 맞췄음을 안다(캘리브레이션됨)**. hard-argmax는 거의 평평한 heatmap의 잡음 위치(다른 kp GT의 15px내 착지 29%, 60px내 55%).
3. **꼬리는 소수 프레임에 집중.** 프레임의 44.4%는 파국 kp 0개(→ median 13mm의 우수 프레임). 파국 kp의 **71.4%가 파국 kp ≥3개인 프레임**에 몰림. 이 프레임들은 **성한 키포인트가 너무 적어** solver가 붕괴.
4. **왜 conf-gate가 못 잡나 = solver의 min_kp=6 바닥.** kuka_add_eval solver는 `conf_gate=0.05`지만 `min_kp=6` 바닥이 **항상 top-6을 유지** → 7개 중 최대 1개만 버림. 3–6개가 확산인 프레임에서 5–6개 쓰레기 kp를 강제 사용 → 파국. (배포 Panda의 무료 레버 **cov-PnP는 KUKA eval에서 꺼져 있음**.)
5. **처방 = KUKA 검출기 재학습(착수·조기 성공 신호)** — 확산-heatmap 비율을 직접 낮추는 유일한 "새 정보" 레버(추론 레버 windowed/cov-PnP는 실측 무효/노이즈). warm-start + unfreeze 2→4 + KUKA-정렬 joint-weights. **2D val AUC 배포 0.735 → epoch0 0.748 → epoch1 0.775**(PCK@10 80.2→85.5%, 파국-kp율 19.8→14.5%, 단조 개선, 10ep 중 2ep). GPU3(GPU-05b804ff) PID 3973038, log `TRAIN/outputs_heatmap/kuka_detector_retrain_20260722_214107/`, ETA ~4h.
6. 🔴 **2차 발견: KUKA solver 자체가 불안정.** oracle-2D(완벽 2D)에서도 mean ADD **563m**(일부 프레임 km-규모 발산) → 2D와 직교하는 solver 안정성 잔차. 천장 oracle-2D **0.834**(≫ RoboPEPP 0.762)라 재학습만으로도 격차 폐쇄 헤드룸은 충분하나, 완전·안정 폐쇄엔 solver robust-init(결정론적 t_z 클램프)이 2차 레버.

---

## 1. Recoverability / swap 게이트 (forward-only, 전체 5,982 프레임)

`kuka_swap_gate.py` — 배포 검출기로 프레임당 7kp heatmap을 512 crop 공간에서 분석. hard-argmax(모드), 전역 soft-argmax(배포 solver 소비), windowed soft-argmax(win15), 반복 NMS top-5 모드, GT창(±3px) 응답을 dump.

**디코드 2D 오차(px, mean/median/p90):**

| 디코드 | mean | median | p90 | PCK@10px |
|---|---|---|---|---|
| hard-argmax | 77.4 | 2.04 | 99.4 | 80.2% |
| soft-argmax(배포) | 81.1 | 2.01 | 178.1 | 77.0% |
| windowed-sa(win15) | 77.2 | 2.00 | 99.4 | **80.4%** |

- median ~2px → **좋은 프레임은 서브픽셀 정확**. 격차는 순수하게 꼬리(p90에서 갈림).
- 전역 soft-argmax p90(178px)이 hard/windowed(99px)보다 훨씬 나쁨 = 확산 heatmap의 distractor 질량이 soft-argmax를 멀리 끌어감.

**파국(배포 soft-argmax err >10px) 분해 (8,991 kp):**
- hard-peak도 오답 (true swap): **85.2%**
- hard-peak OK, soft-argmax만 끌림: **14.8%** → **windowed decode가 15.3%를 무료로 복구**
- true swap의 정답 회수: top-2 **4.5%** / top-3 **7.2%** / top-5 **10.9%**; oracle 모드선택 err mean 298 / median 40px
- gt_resp(정답 위치 응답/peak): mean 0.195 / **median 0.113** / >0.3: 24.6% / >0.5: 12.0%

→ **VERDICT: 정답 모드 부재 = detector-limited.** skeleton-consistent 모드선택 decode는 선택 대상이 없어 무효(완벽 선택기도 파국 kp의 ~11%만 <5px, median 잔차 40px).

## 2. 꼬리 특성 (`analyze_tail.py`)

- **신뢰도 이중분포**: true-swap peak 0.04(p50 0.01, p90 0.10) vs good 0.65(p50 0.71). swap의 0.8%만 good-p10 초과 → **저신뢰·캘리브레이션됨**(확신 오답 아님).
- **프레임 집중**: 파국 kp 0/1/2/3+ 프레임 = 44.4/16.8/13.0/25.8%. 파국 kp의 71.4%가 ≥3-파국 프레임.
- **자기가림 proxy(자기 GT의 최근접 타 kp GT 거리)**: true-swap이 40px내 이웃 20.8% vs good 8.0% → 접힘/자기가림 기여는 있으나 지배적 아님(대부분은 원통 대칭에 의한 저신호).
- **per-link 파국률**: L2 **28.9%**(최악) > L1 24.8 > L3 25.3 ≈ L0 24.2 > L5 21.4 > L4 18.3 > L6 **17.5%**(최선). 대체로 균일(원통 전반의 대칭 문제).

## 3. 처방

### (P0, 착수) KUKA 검출기 재학습 — 확산-heatmap 꼬리 공략
게이트가 detector-limited이므로 **새 정보(더 나은 2D)** 만이 꼬리 프레임을 <100mm로 되돌린다(AUC 트랩: 탐지·게이팅만으론 이득 0).

배포 검출기 대비 변경(warm-start 유지):
- **unfreeze 2→4 블록**: 근접-동일 원통 링크의 identity/positional context를 인코딩할 용량.
- **KUKA-정렬 joint-weights** `[1.6,1.65,1.9,1.65,1.2,1.4,1.15]`(≈per-link 파국률): 배포는 legacy Panda U-shape `[2.5,1.5,1.3,1.0,1.3,1.5,2.5]`로 **KUKA의 최선 링크 L6을 up-weight·최악 L2를 under-weight** — 오정렬.
- 나머지 동일(crop 1.5, aug strong, occ 0.3/0.18, 512), 10 epoch cosine, head lr 2e-4 / backbone lr 2e-5, amp.

**A/B 게이트**: KUKA synth ADD-AUC (solver+참K, 전체 5,997f) vs **0.690**; 2D-keypoint val AUC vs **0.735**; 파국-kp 비율 vs 22.95% / fail 프레임 15.9%. do-no-harm: good 프레임(median 13mm) 불변.

### (부차, A/B 실측 — 전부 무효/노이즈 내) 무료/저가 추론 레버
전체 5,997f, solver+참K. 🔴 **핵심: KUKA solver는 RANSAC-seed에 극도로 불안정.** 동일 검출기·설정에서 frame-fail(>100mm)이 **16%(참조 0.690) ↔ 25%(내 재현)** 로 요동 → 단일-run A/B는 ±0.03 노이즈. mean ADD가 **43,259mm**(수 프레임이 km 규모로 결정론적 발산)으로 터짐.

| config | ADD-AUC | mean ADD | median | TAIL>100mm |
|---|---|---|---|---|
| no-cov baseline (내 재현) | 0.628 | 43,259mm | 15.1mm | 25.1% |
| windowed decode (win15) | 0.658 | 6,331mm | 14.3mm | — |
| cov-PnP | 0.630 | 43,259mm | 15.0mm | 25.1% |
| conf-adaptive (cov+gate.15+minkp4+anchor.05) | **0.583** | 43,264mm | 23.0mm | 25.2% |
| **참조(intrinsics_rootcause)** | **0.690** | 91mm | 13.1mm | 15.9% |
| **oracle-2D 천장** | **0.834** | 563,680mm | 3.9mm | — |

- **windowed decode = net HARM(무효)**: 게이트의 "15.3% 무료 복구"는 soft-pull 부분집합만; 85% true-swap(확산 heatmap)에서 **잡음 hard-peak에 확신 커밋** → solver가 실검출로 취급 → 악화. 0.658은 내 no-cov 0.628보다 높으나 참조 0.690보다 낮고 노이즈 밴드 내.
- **cov-PnP = flat**: 0.630 ≈ 0.628, mean **byte-identical(43,259mm)** — 확산 heatmap 발산을 못 잡음. 참조 0.690이 cov-PnP 산물이라는 가설 **기각**.
- **conf-adaptive anchor-to-head = net HARM(실측 반증)**: 0.583 < baseline 0.628, median **15→23mm 악화**(kp 드롭 + head 앵커가 good 프레임까지 손상). `2026-07-22_p0_decoupled_solve.md`의 "solver에서 head θ 고정/앵커 계열 전체" 반증이 **KUKA에도 전이 확정**(head θ와 solver θ 오차 상관 — 나쁜 프레임에서 head도 나쁨).
- ⇒ **모든 추론-decode 레버(windowed/cov-PnP/conf-adaptive)가 net flat-or-harmful** — 게이트의 detector-limited 판정과 완전 일치(선택·재가중할 정답 신호 자체가 없음).
- 🔴 **oracle-2D도 mean 563m로 발산** → **완벽한 2D에서도 solver가 일부 프레임에서 절대깊이 basin 오착지**. = 2D와 직교하는 **solver 안정성 잔차**. 결정론적 t_z 클램프/robust init이 2차 레버(후속).

### A/B 검증 계획 (재학습 완료 시)
solver RANSAC 노이즈(±0.03) 때문에 단일-run 비교 부적절 → 재학습 검출기 A/B는 **RANSAC seed 고정**(`kuka_ab.py`에 `cv2.setRNGSeed(0)` 추가)으로 측정. **seed-0 baseline 앵커 = 0.628**(median 15.1mm, `ab_seed0_baseline.log`) — 재학습 검출기를 **동일 seed-0**로 돌려 clean delta. (AUC는 seed에 robust(~0.628)하나 km-규모 발산 프레임은 seed마다 바뀜 → AUC가 유일 신뢰 지표. 참조 0.690/13.1mm은 재현 실패 = 원 run의 solver-config 차이 추정, 미해결.)
1차 게이트 = **2D val AUC**(seed-무관, robust): 배포 0.735 → 재학습 **ep0 0.748 / ep1 0.775 / ep2 0.790 / ep3 0.801**(단조, PCK@10 80.2→85.5% @ep1).

---

## 부록 — 재현 명령

```bash
cd Eval/_debate_tmp/kuka_gate
# 게이트(forward-only, 전체)
CUDA_VISIBLE_DEVICES=GPU-<uuid> python kuka_swap_gate.py \
  --detector ../../../TRAIN/outputs_heatmap/kuka_dream_detector_20260709_183119/best_heatmap.pth \
  --val-dir ../../../../../datasets/synthetic/kuka_synth_test_dr --max-frames 0 --out kuka_gate_full.npz
python analyze_tail.py
# 재학습(GPU3, 10ep, warm-start)
GPU=GPU-<uuid> bash launch_kuka_retrain.sh
```
