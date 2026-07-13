# DINObotPose3 — Paper Draft (한국어 본문 · 영어 주석 병기)

> 작성 방식: **한국어를 본문**으로 쓰고, 각 문단 아래 `> EN:` 블록에 **영어 초안**을 병기한다(추후 영어 논문 본문으로 승격).
> 상태: Abstract / Introduction / Related Work / Method(개요·핵심) 초안. Experiments·Ablation·Conclusion은 TODO.
> 근거 문서: [references/related_work.md](references/related_work.md)(정직한 포지셔닝), [FINAL_MODEL.md](FINAL_MODEL.md), [references/sota_survey.md](references/sota_survey.md).
>
> <!-- EN: Korean is the working body; the `> EN:` blockquote under each paragraph is the English draft to be promoted to the paper. Draft covers Abstract/Intro/Related Work/Method; Experiments/Ablation/Conclusion are TODO. -->

---

## 제목 (Title)

**단안 로봇 포즈 추정: 동결 파운데이션 특징과 불확실도-인지 기하 최적화, 제로샷 렌더 비교의 결합**

> EN: **Geometry-Guided Monocular Articulated Robot Pose Estimation with Frozen Foundation Features, Uncertainty-Aware Optimization, and Zero-Shot Render-and-Compare**

---

## Abstract (초록)

관절각까지 미지인 단안 로봇 포즈 추정은, 이미지 공간의 2D 키포인트만으로는 깊이·스케일·자기가림 구성에 대한 제약이 약해 여전히 어렵다. 본 논문은 **관절각을 예측하고(predicted-joint) 바운딩 박스를 완전 자동으로 잡는** 설정에서 단일 이미지 관절형 로봇 포즈를 추정하는 기하-유도(geometry-guided) 파이프라인 **DINObotPose3**를 제안한다. 제안 방법은 동결된 DINOv3 키포인트 프론트엔드에 관절각·회전 헤드를 결합하고, 공분산-인지 PnP(cov-PnP)와 운동학 재투영 정제, DARK 서브픽셀 히트맵 디코딩, 그리고 제로샷 SAM 마스크와 미분가능 렌더링을 이용한 **테스트-타임 렌더-비교(render-and-compare) 깊이 보정** 단계를 더한다. 부분 가시(가림) 상황의 강건성을 위해, 합성 데이터의 망각 방지를 유지한 채 약한 가림 증강과 카메라별 자가학습을 쌓는다. DREAM 실측 벤치마크에서 DINObotPose3는 4개 카메라 스플릿 평균 ADD-AUC **0.804**를 달성하여, **완전 자동 바운딩 박스**를 쓰면서도 RoboPEPP·RoboTAG 등 기존 predicted-joint 기준선을 능가한다. 또한 평가한 모든 가림 수준에서 RoboPEPP를 상회하여, 파운데이션 특징 + 불확실도-인지 기하 최적화 + 제로샷 렌더 비교의 조합이 학습형 깊이 회귀나 종단간(end-to-end) 포즈 회귀에 대한 강력한 대안임을 보인다.

> EN: Monocular robot pose estimation with unknown joint states remains challenging because image-space keypoints alone provide weak constraints for depth, scale, and self-occluded configurations. We present **DINObotPose3**, a geometry-guided pipeline for single-image articulated robot pose estimation under the **predicted-joint and fully automatic bounding-box** setting. The method combines a frozen DINOv3 keypoint front-end with joint-angle and rotation heads, covariance-aware PnP and kinematic reprojection refinement, DARK sub-pixel heatmap decoding, and a zero-shot SAM plus differentiable-rendering **render-and-compare** stage for test-time depth correction. To improve robustness under partial visibility, we stack light occlusion augmentation with camera-specific self-training while preserving synthetic anti-forgetting. On the DREAM real benchmark, DINObotPose3 achieves a mean ADD-AUC of **0.804** across four camera splits, outperforming prior predicted-joint baselines including RoboPEPP and RoboTAG **while using fully automatic bounding boxes**. It also exceeds RoboPEPP across all evaluated occlusion levels, showing that foundation features, uncertainty-aware geometric optimization, and zero-shot render comparison form an effective alternative to learned depth or end-to-end pose regression.

---

## 1. 서론 (Introduction)

로봇 팔을 단안 RGB 한 장에서 추정하는 문제는 카메라-로봇 캘리브레이션, 원격조작, 시각 서보잉의 기반이 된다. 표준 벤치마크 DREAM은 두 가지 난이도 축을 갖는다. (i) 관절각을 엔코더로 아는 **known-joint**인지, 이미지에서 **예측(predicted-joint)**해야 하는지, (ii) 로봇 바운딩 박스를 GT로 주는지 **자동 검출**하는지. 본 연구는 가장 어려운 조합, 즉 **predicted-joint + 완전 자동 bbox**를 목표로 한다.

> EN: Estimating a robot arm from a single RGB image underpins camera-to-robot calibration, teleoperation, and visual servoing. The standard DREAM benchmark has two difficulty axes: (i) whether joint angles are **known** from encoders or must be **predicted** from the image, and (ii) whether the robot bounding box is given as GT or **auto-detected**. We target the hardest combination: **predicted-joint with fully automatic bounding boxes**.

이 설정의 근본 난제는 **깊이·스케일 모호성**이다. 2D 키포인트만으로는 "멀리 있는 팔"과 "가까이서 짧게 접힌 팔"을 구분하기 어렵다(foreshortening). 기존 predicted-joint SOTA들은 바로 이 문제를 서로 다르게 푼다 — HoRoPose는 학습된 root-depth 회귀로, RoboKeyGen은 확산(diffusion) 기반 2D→3D 리프팅으로, RoboPEPP는 마스킹 사전학습으로 물리 사전지식을 인코더에 주입한다.

> EN: The core difficulty of this setting is **depth/scale ambiguity**: from 2D keypoints alone, "a far arm" and "a foreshortened near arm" are hard to disambiguate. Existing predicted-joint methods resolve this differently — HoRoPose via a learned root-depth regressor, RoboKeyGen via diffusion-based 2D→3D lifting, and RoboPEPP by injecting a physics prior through masking pretraining.

우리는 **학습 없이 기하로 깊이를 보정하는** 대안을 제시한다. 동결 DINOv3에서 뽑은 서브픽셀 키포인트를 운동학 솔버에 넣어 포즈를 초기화한 뒤, 제로샷 SAM 마스크와 렌더된 실루엣을 미분가능 렌더링으로 정합해 **테스트-타임에 깊이/스케일만 보정**한다. 여기에 학습이 전혀 필요 없는 두 레버(cov-PnP, DARK 디코딩)를 더해 정확도를 공짜로 끌어올린다.

> EN: We propose a **training-free geometric alternative** for depth correction. Sub-pixel keypoints from a frozen DINOv3 initialize a kinematic solver; a zero-shot SAM mask and a rendered silhouette are then aligned via differentiable rendering to **correct only depth/scale at test time**. Two training-free levers (cov-PnP and DARK decoding) further improve accuracy at no cost.

**기여(Contributions).**
1. **Predicted-joint + 자동 bbox 최고 성능**: DREAM 실측 4-스플릿 평균 ADD-AUC 0.804로 RoboPEPP(0.780)·RoboTAG(0.740)를 능가한다. RoboPEPP의 헤드라인이 GT-bbox인 반면 우리는 완전 자동이다.
2. **테스트-타임·학습불필요 렌더-비교 깊이 보정기**: 렌더-비교를 *발명*한 것이 아니라(RoboPose'21, CtRNet'23 선행), 제로샷 SAM 마스크 + 동결 DINOv3 키포인트 프론트엔드 위에서 predicted-joint·자동-bbox 체제에 맞게 **학습 없이 재구성**하고, 카메라별로 선택 적용한다.
3. **가림 강건성**: 약한 가림 증강 + 카메라별 자가학습 스택으로, 평가한 전(全) 가림 수준(0–40%)에서 RoboPEPP를 상회한다.
4. **일반화 연구(부록/후속)**: 동일 파이프라인을 KUKA iiwa7·Baxter로 확장하여 검출·FK·포즈까지 일반화됨을 보이고, 로봇별 병목(관측성 천장 등)을 분석한다.

> EN: **Contributions.** (1) **State of the art under predicted-joint + auto-bbox**: mean ADD-AUC 0.804 on DREAM-real, surpassing RoboPEPP (0.780) and RoboTAG (0.740), while using fully automatic boxes (RoboPEPP's headline uses GT boxes). (2) A **test-time, training-free render-and-compare depth corrector**: we do not *invent* render-and-compare (RoboPose'21, CtRNet'23 are prior art) but **recast it without training** as a zero-shot-SAM + frozen-DINOv3 depth corrector for the predicted-joint / auto-bbox regime, applied per camera. (3) **Occlusion robustness**: light occlusion augmentation plus camera-specific self-training exceeds RoboPEPP across all evaluated occlusion levels (0–40%). (4) **Generalization study (appendix/follow-up)**: extending the same pipeline to KUKA iiwa7 and Baxter, we show detection/FK/pose all generalize, and analyze per-robot bottlenecks (e.g., a wrist-observability ceiling).

---

## 2. Related Work (관련 연구 — 정직한 포지셔닝)

**관절각 known vs predicted가 비교의 전제.** 수치를 나란히 놓기 전에 이 축을 반드시 맞춰야 한다. CtRNet·CtRNet-X는 엔코더 관절각을 받는 **known-joint**(DREAM 86.x)로, 관절각까지 예측하는 우리 0.804와 **직접 비교 대상이 아니다**(더 쉬운 문제). Predicted-joint 체제의 실제 경쟁자는 RoboPose(73.2)·HoRoPose(77.2)·RoboTAG(74.0)·RoboPEPP(78.0)이며, 이 중 우리가 최고다.

> EN: **Known vs predicted joints is a prerequisite for any comparison.** CtRNet and CtRNet-X consume encoder joint angles (**known-joint**, DREAM ~86) and are therefore **not directly comparable** to our predicted-joint 0.804 — they solve an easier problem. The actual competitors in the predicted-joint regime are RoboPose (73.2), HoRoPose (77.2), RoboTAG (74.0), and RoboPEPP (78.0), among which ours is best.

**렌더-비교는 우리 발명이 아니다.** DeepIM(2018)이 일반 물체에, RoboPose(CVPR'21)가 로봇에 반복적 렌더-비교를 도입했고, CtRNet(CVPR'23)은 미분가능 **실루엣** 정합을 자기지도 학습 손실로 사용했다. 우리 기여는 "개념"이 아니라 **형태·체제·조합**이다: (a) 학습된 refiner 없이 **테스트-타임 포즈 최적화**로 쓰고, (b) 자체 세그가 아닌 **제로샷 SAM** 마스크를 타깃으로 하며, (c) known-joint가 아닌 **predicted-joint·자동-bbox** 체제에 적용하고, (d) 전체 포즈가 아니라 **깊이/스케일 보정기**로 카메라별 선택 적용한다.

> EN: **Render-and-compare is not our invention.** DeepIM (2018) introduced iterative render-and-compare for general objects, RoboPose (CVPR'21) for robots, and CtRNet (CVPR'23) used differentiable **silhouette** matching as a self-supervised training loss. Our contribution is not the *concept* but its **form, regime, and combination**: (a) used as **test-time pose optimization** with no learned refiner, (b) targeting **zero-shot SAM** masks rather than a trained segmenter, (c) applied in the **predicted-joint / auto-bbox** regime rather than known-joint, and (d) acting as a **depth/scale corrector** applied selectively per camera rather than estimating the whole pose.

**대부분의 최신 predicted-joint 기준선은 렌더-비교를 쓰지 않는다.** RoboPEPP(feed-forward+PnP), HoRoPose(DepthNet), RoboKeyGen(diffusion), RoboTAG(end-to-end 회귀)는 모두 렌더-비교가 없다. 따라서 우리 테스트-타임 SAM-실루엣 깊이 보정은 이들 대비 **구조적 차별점**이다. 또한 우리 가림 처리 메커니즘의 최근접 이웃은 (CLIP을 쓰는 CtRNet-X가 아니라) **RoboPEPP**로, 둘 다 마스킹/가림 학습 + 신뢰도 필터를 쓰나 우리는 렌더-비교 깊이 보정과 무료 디코딩 레버로 전 가림 구간에서 앞선다.

> EN: **Most recent predicted-joint baselines do not use render-and-compare.** RoboPEPP (feed-forward + PnP), HoRoPose (DepthNet), RoboKeyGen (diffusion), and RoboTAG (end-to-end regression) all omit it, making our test-time SAM-silhouette depth corrector a **structural differentiator**. The nearest neighbor to our occlusion mechanism is **RoboPEPP** (not CLIP-based CtRNet-X): both use masking/occlusion training plus confidence filtering, but we lead across all occlusion levels via the render-compare depth corrector and free decoding levers.

---

## 3. Method (제안 방법)

### 3.1 개요 (Overview)

파이프라인은 다섯 단계다: **(1)** 동결 DINOv3 ViT-B/16 백본이 히트맵 키포인트를 검출하고, **(2)** DARK 서브픽셀 디코딩으로 격자 양자화 오차를 제거하며, **(3)** 관절각·회전 헤드가 관절 구성과 카메라-로봇 회전을 예측하고, **(4)** 공분산-인지 PnP + 운동학 재투영 정제가 포즈를 푼 뒤, **(5)** 제로샷 SAM 마스크와 렌더 실루엣을 미분가능 렌더링으로 정합해 깊이/스케일을 테스트-타임에 보정한다. 백본은 끝까지 동결한다(적응은 서브픽셀 키포인트 정밀도를 파괴함을 3회 실험으로 확인).

> EN: The pipeline has five stages: **(1)** a frozen DINOv3 ViT-B/16 backbone detects heatmap keypoints, **(2)** DARK sub-pixel decoding removes grid-quantization error, **(3)** joint-angle and rotation heads predict the joint configuration and camera-to-robot rotation, **(4)** covariance-aware PnP with kinematic reprojection refinement solves the pose, and **(5)** a zero-shot SAM mask and a rendered silhouette are aligned by differentiable rendering to correct depth/scale at test time. The backbone stays frozen throughout (we verified across three experiments that adapting it destroys the sub-pixel keypoint precision the solver needs).

### 3.2 동결 특징과 무료 레버 (Frozen features and free levers)

**DARK 서브픽셀 디코딩.** 히트맵은 저해상도 격자라 argmax는 위치를 격자 단위로 양자화한다. DARK는 봉우리 근처 밝기의 1·2차 미분으로 테일러 보정을 적용해 정점을 소수점 위치로 재국소화한다. 학습이 전혀 필요 없으며, 원거리 카메라에서 작은 위치 오차가 증폭되던 것을 완화한다.

> EN: **DARK sub-pixel decoding.** Heatmaps are low-resolution grids, so argmax quantizes locations to the grid. DARK applies a Taylor correction using the first/second derivatives of the intensity around the peak, re-localizing it to a sub-pixel position. It requires no training and mitigates the amplification of small localization errors on far cameras.

**공분산-인지 PnP(cov-PnP).** 히트맵의 국소 2차 모멘트에서 키포인트별 이방성 2×2 공분산을 공짜로 추출해, 재투영 잔차를 마할라노비스(백색화) 거리로 바꾼다. 흐리거나 다봉인(가려진) 키포인트는 방향별로 연속 다운웨이트되어, 스칼라 신뢰도 게이팅을 자연스럽게 일반화한다.

> EN: **Covariance-aware PnP (cov-PnP).** From the local second moments of the heatmap we extract a per-keypoint anisotropic 2×2 covariance for free, turning the reprojection residual into a Mahalanobis (whitened) distance. Diffuse or multimodal (occluded) keypoints are continuously down-weighted per direction, generalizing scalar confidence gating.

### 3.3 테스트-타임 렌더-비교 깊이 보정 (Test-time render-and-compare)

이미 좋은 키포인트+운동학 추정 위에, 제로샷 SAM으로 얻은 로봇 전경 마스크와 nvdiffrast로 렌더한 로봇 메쉬 실루엣의 IoU를 최대화하도록 포즈의 깊이/스케일을 미분가능하게 정제한다. SAM은 외부 가림체를 로봇에서 분리해 주므로 마스크가 깨끗하다. 이 단계는 카메라별로 켜고 끈다 — 근거리 카메라(azure)는 깊이 신호가 이미 강해 끄고, 원거리에서 크게 이득을 본다(카메라별 +0.04~0.07 ADD-AUC).

> EN: **Test-time render-and-compare.** On top of already-good keypoint + kinematic estimates, we differentiably refine the pose's depth/scale to maximize the IoU between a zero-shot SAM robot foreground mask and an nvdiffrast-rendered robot silhouette. SAM cleanly separates external occluders from the robot. This stage is toggled per camera — off for near cameras (azure), whose depth signal is already strong, and most beneficial on far cameras (+0.04–0.07 ADD-AUC per camera).

### 3.4 가림 강건성 (Occlusion robustness)

<!-- TODO(KR): light occ-aug head + 카메라별 self-train(합성 anti-forget 유지) 상세, 가림 곡선 프로토콜(RoboPEPP Fig.6). -->
> EN TODO: detail the light occlusion-augmentation head plus camera-specific self-training with synthetic anti-forgetting, and the occlusion-curve protocol (RoboPEPP Fig. 6).

---

## 4. Experiments (실험) — TODO
<!-- 성적표(재잠금 1000프레임), 가림 곡선, ablation(RC/cov-PnP/DARK/occ-stack), 프로토콜 표. 근거: FINAL_MODEL.md, figures/. -->
> EN TODO: scorecard (1000-frame re-lock), occlusion curve, ablations (RC / cov-PnP / DARK / occ-stack), protocol table. Sources: FINAL_MODEL.md, figures/.

## 5. Multi-Robot Generalization (일반화) — TODO
<!-- KUKA/Baxter 검출·data-fit FK·direct-pose 포즈, 관측성 천장 분석. 근거: experiments/2026-07-10_multirobot. -->
> EN TODO: KUKA/Baxter detection, data-fit FK, direct-pose pose, wrist-observability analysis. Source: experiments/2026-07-10_multirobot.

## 6. Conclusion (결론) — TODO
