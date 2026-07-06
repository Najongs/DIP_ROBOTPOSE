# Related Work & 정직한 포지셔닝 — render-and-compare 계보에서 우리 위치

> 목적: "우리 render-and-compare가 RoboPose/CtRNet과 같은 방법 아니냐?"에 대한 정직한 답.
> 결론 먼저: **render-and-compare·미분가능 실루엣·자기지도 학습은 우리가 발명한 게 아니다(CtRNet 계보).**
> 우리 기여는 "개념"이 아니라 **predicted-angles + 자동 bbox 체제에서의 특정 형태·조합**이다.
> 검증일 2026-07-06 (arXiv/GitHub 원문 확인).

---

## 0. 가장 중요한 것 — 프로토콜 축이 다르면 수치 비교 무효

DREAM-real AUC를 나란히 놓기 전에 **관절각 known vs predicted**를 반드시 맞춰야 한다.

- **known angles**: 로봇 엔코더에서 관절각을 받고 **카메라↔로봇 base 변환만** 추정 (쉬움). FK 3D 키포인트가 정확히 주어짐.
- **predicted angles**: 관절각도 이미지에서 **예측**해야 함 (어려움). ← **우리, RoboPEPP, RoboTAG, RoboPose, HoRoPose**

**CtRNet(86.4)·CtRNet-X(86.2)는 known-angles + real self-train** → 우리 predicted-angles 80.4와 **직접 비교 불가**(더 쉬운 문제를 푼 수치). predicted-angles 체제의 실제 경쟁자는 RoboPose 73.2 / HoRoPose 77.2 / RoboTAG 74.0 / RoboPEPP 77.9이고, **우리 80.4가 이 체제의 최고**.

> **"CtRNet의 unknown-joint 버전은 없나?"** — 없다. CtRNet·CtRNet-X 둘 다 **known-angles 전용**(엔코더 관절각 → FK 3D 키포인트). 관절각을 이미지에서 **예측**하는(unknown states) 계열은 CtRNet이 아니라 **RoboPose(CVPR'21) · HoRoPose(ECCV'24) · RoboKeyGen(ICRA'24) · RoboPEPP(CVPR'25) · RoboTAG('25) · 우리**다. 즉 render-compare 자기지도(CtRNet)와 unknown-joint(RoboPose 계열)는 **다른 계보**이고, 우리는 후자(어려운 쪽)에 속한다.

---

## 1. Render-and-compare 계보

| 방법 | 연도/베뉴 | R&C 역할 | 비교 신호 | 관절각 | 세그/특징 | 가림·부분가시 처리 |
|---|---|---|---|---|---|---|
| DeepIM | ECCV'18 | 학습된 반복 포즈-업데이트 net | 렌더 RGB↔실제 | (일반물체) | — | — |
| **RoboPose** | CVPR'21 | **학습된** 반복 업데이트 net | 렌더 RGB↔실제 | predicted | 자체 렌더 | init bbox 의존 |
| **CtRNet** | CVPR'23 | **학습-타임 자기지도 손실** | 렌더 **실루엣**↔예측 전경마스크 | **known** | 자체 seg net | (미대응) |
| **CtRNet-X** | ICRA'25 | 학습-타임 자기지도(실루엣, RGB) | 렌더 실루엣↔예측 마스크 | **known** | **CLIP**로 가시 부품 검출 | **CLIP 가시-부품 → 키포인트 선택** |
| **Ours** | 2026-07 | **테스트-타임 포즈 최적화 (학습X)** | 렌더 실루엣↔**SAM 마스크** | **predicted** | **SAM+frozen DINOv3** | **conf-gate + cov-PnP + occ-aug 학습** |

출처: RoboPose [arXiv:2104.09359, github.com/ylabbe/robopose] · CtRNet [CVPR'23, arXiv:2302.14332] · CtRNet-X [ICRA'25, arXiv:2409.10441, sites.google.com/ucsd.edu/ctrnet-x] · HoRoPose(Holistic) [ECCV'24, arXiv:2402.05655, github.com/Oliverbansk/Holistic-Robot-Pose-Estimation] · RoboKeyGen [ICRA'24, arXiv:2403.18259].

---

## 2. "같은 방법인가?" — 아니오, 그러나 같은 **계열**이다 (정직하게)

### 공유하는 것 (= 우리 신규 기여 아님, 선행연구)
- **render-and-compare 개념**: RoboPose(2021)가 로봇에 도입, DeepIM(2018)이 일반물체에 원조.
- **미분가능 실루엣 매칭**: CtRNet(2023)이 이미 실루엣(전경마스크) render-compare 사용.
- **real 자기지도/self-training**: CtRNet의 핵심 — 렌더 마스크와 예측 마스크 일치로 라벨 없이 학습. 우리도 self-train(`selftrain_pseudo_rot.py`) 함 → **목표 겹침**.
- **부분가시/가림에서 키포인트 선택**: CtRNet-X(2024)가 이미 다룸 (CLIP로 보이는 부품만 선택). 우리 "가림 강건성"도 **목표 자체는 신규 아님**.

### 우리가 실제로 다른 것 (기여점)
1. **R&C의 역할이 다르다** — CtRNet/CtRNet-X는 R&C를 **학습-타임 자기지도 손실**로 씀(네트워크를 훈련). 우리는 **테스트-타임 포즈 최적화**로 씀 — 학습된 refiner 없이 미분가능 실루엣 IoU에 경사하강. 이미 좋은 키포인트+운동학 추정 위에 얹는 **깊이/스케일 보정기**(카메라별 on/off).
2. **관절각을 예측한다** — CtRNet-X는 엔코더 관절각을 받음(known). 우리는 이미지에서 예측(predicted) — 더 어려운 프로토콜.
3. **세그가 zero-shot** — CtRNet은 자체 seg net 학습, CtRNet-X는 CLIP 파인튜닝. 우리는 **SAM**(zero-shot) — 로봇 학습 불필요, 외부 가림체를 로봇에서 분리.
4. **백본 frozen DINOv3** — CtRNet은 keypoint net을 학습·적응. 우리는 백본 동결(적응 계열 반증, sub-pixel 정밀도 보존).
5. **가림 처리 메커니즘 상이** — CtRNet-X: CLIP 가시-부품 선택. 우리: 히트맵 conf-gate + cov-PnP 불확실도 가중 + **occ-aug 학습**(head가 처음부터 가림 노출).
6. **완전 자동 bbox** — bbox-from-solved. CtRNet-X의 bbox 프로토콜 미명시, RoboPEPP는 GT-bbox.
7. **무료 레버** — DARK sub-pixel decode + cov-PnP (선행연구에 없음, 학습 0).

### 한 줄 정직한 포지셔닝
> "우리가 render-and-compare를 발명했다"가 아니다.
> **"CtRNet 계보의 학습-타임 실루엣 자기지도를, predicted-angles·자동-bbox 체제에서 frozen-DINOv3 키포인트 프론트엔드 위의 zero-shot SAM 마스크 테스트-타임 깊이 보정기로 재구성하고, occ-aug 가림 강건성과 무료 디코딩 레버를 결합"**한 것이 기여.

---

## 3. predicted-angles 체제 DREAM-real 비교 (프로토콜 통제)

| 방법 | mean AUC | 관절각 | bbox | 깊이 모호성 해법 | 비고 |
|---|---|---|---|---|---|
| RoboPose | 73.2 | predicted | init 의존 | 반복 render&compare(RGB) | auto orb 32.7 |
| RoboKeyGen | — | predicted | — | diffusion 3D 키포인트 생성 후 R&C | synth 학습, unseen config 일반화 |
| RoboTAG | 74.0 | predicted | **자동** | topological align graph | orb 58.8(auto 붕괴) |
| **HoRoPose** | 77.2 | predicted | — | **학습된 root-DepthNet** | 키포인트+DepthNet+기하, real 자기지도 |
| RoboPEPP | 77.9 | predicted | **GT-bbox** | masking-pretrain 인코더 | auto orb≈34; 가림 최강(기존) |
| **Ours (재잠금)** | **80.4** | predicted | **완전 자동** | **테스트-타임 SAM-실루엣 R&C 보정기** | +DARK+cov-PnP; 가림 전 구간 RoboPEPP 초과 |

(참고: known-angles 체제 CtRNet-X 86.2 / CtRNet 86.4 — 별도 리그, 비교 불가.)

### 핵심 통찰 — predicted-angles 단안의 진짜 병목은 "깊이/스케일 모호성"
2D 키포인트만으로는 "멀리 있는 팔 vs 짧게 접힌 팔"을 구분 못 한다(foreshortening). predicted-angles SOTA들은 **바로 이걸 어떻게 푸느냐**로 갈린다:
- **HoRoPose**: 학습된 **root-DepthNet**으로 루트 깊이를 직접 회귀.
- **RoboPEPP**: 물리모델을 masking-pretrain으로 인코더에 주입.
- **우리**: **render-and-compare 실루엣**(SAM 마스크 vs 렌더 메쉬)이 깊이를 기하적으로 보정 — 학습 불필요, 카메라별 on/off.

즉 우리 render-compare의 실제 역할은 "포즈 전체 추정"이 아니라 **HoRoPose의 DepthNet이 하는 일을 학습 없이 기하로 대체**하는 것. 이게 fig5(카메라별 RC 기여 = 원거리 +0.07, 근거리 0)가 보여주는 바 — RC는 깊이 신호가 약한 원거리에서만 크게 이득.

---

## 4. 리뷰어 예상 질문 & 답

- **Q. render-and-compare는 RoboPose/CtRNet에 있는데 신규성은?**
  A. R&C 개념은 선행. 신규는 (a) **테스트-타임 학습-불필요** 형태(CtRNet은 학습-타임 손실), (b) **SAM zero-shot 마스크**(CtRNet은 자체/CLIP seg), (c) **predicted-angles·자동-bbox** 체제 적용, (d) 깊이 보정기로서 **카메라별 선택 적용**.
- **Q. CtRNet-X가 DREAM에서 86.2인데 왜 너희가 SOTA?**
  A. CtRNet-X는 **known joint angles**(엔코더) — 카메라 pose만 푸는 쉬운 문제. 우리는 관절각까지 예측(predicted). 같은 프로토콜(predicted)의 경쟁자 중 우리가 최고(80.4 > RoboPEPP 77.9).
- **Q. 가림 강건성도 CtRNet-X가 이미 하지 않나?**
  A. 목표는 같으나 메커니즘·프로토콜 다름. CtRNet-X는 known-angle + CLIP 부품선택. 우리는 predicted-angle에서 occ-aug 학습으로 head 자체를 강건화, RoboPEPP 가림 곡선 전 구간 초과.

---

관련: [sota_survey.md](sota_survey.md)(전체 계보·프로토콜 3축), [next_directions.md](next_directions.md)(로드맵), [../FINAL_MODEL.md](../FINAL_MODEL.md).
