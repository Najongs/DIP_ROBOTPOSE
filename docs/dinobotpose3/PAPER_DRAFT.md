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
4. **DREAM의 세 로봇 모두 평가**: DREAM은 Panda·KUKA·Baxter를 포함하는 벤치마크다. 실측 데이터가 있는 Panda에서 위 SOTA를 내고, 합성 전용인 KUKA·Baxter에서 동일 파이프라인이 검출·FK·포즈까지 end-to-end로 동작함을 보이며 로봇별 병목(예: 손목 관절의 관측성 천장)을 분석한다.

> EN: **Contributions.** (1) **State of the art under predicted-joint + auto-bbox**: mean ADD-AUC 0.804 on DREAM-real, surpassing RoboPEPP (0.780) and RoboTAG (0.740), while using fully automatic boxes (RoboPEPP's headline uses GT boxes). (2) A **test-time, training-free render-and-compare depth corrector**: we do not *invent* render-and-compare (RoboPose'21, CtRNet'23 are prior art) but **recast it without training** as a zero-shot-SAM + frozen-DINOv3 depth corrector for the predicted-joint / auto-bbox regime, applied per camera. (3) **Occlusion robustness**: light occlusion augmentation plus camera-specific self-training exceeds RoboPEPP across all evaluated occlusion levels (0–40%). (4) **All three DREAM robots**: DREAM is a Panda/KUKA/Baxter benchmark; we report the above SOTA on Panda (which has real data) and show the same pipeline runs end-to-end on the synthetic-only KUKA and Baxter splits, analyzing per-robot bottlenecks (e.g., a wrist-joint observability ceiling).

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

부분 가시(가림)는 배포 시 흔하지만 합성 학습 데이터에는 거의 없다. 우리는 이를 두 단계로 처리한다.

> EN: Partial visibility (occlusion) is common at deployment but nearly absent from synthetic training data. We address it in two stages.

**약한 가림 증강 헤드.** 관절각·회전 헤드를 학습할 때, 로봇 위에 노이즈 텍스처 사각 occluder를 페이스트하는 가림 증강을 켠다. 핵심은 **약하게(gentle)** 쓰는 것이다 — 페이스트 비율을 ≤0.3으로 제한하고 키포인트 드롭은 쓰지 않는다. 처음부터 가림에 노출된 헤드는 가림 하에서 더 강건해지며(40% 가림에서 깨끗 헤드 0.376 대 약한 증강 헤드 0.420), 강한 증강은 오히려 깨끗한 정확도를 해쳐 역효과다.

> EN: **Light occlusion-augmentation head.** When training the joint-angle and rotation heads, we enable occlusion augmentation that pastes noise-textured rectangular occluders onto the robot. The key is to keep it **gentle** — paste ratio ≤0.3 and no keypoint drop. A head exposed to occlusion from the start is more robust under occlusion (0.376 clean-only vs 0.420 light-augmented at 40%), whereas strong augmentation backfires by hurting clean accuracy.

**합성 망각 방지를 유지한 카메라별 자가학습.** 각 실측 카메라에 대해 검출기 의사라벨로 자가학습하여 도메인 갭을 좁힌다. 다만 순수 자가학습은 가림 강건성을 씻어내므로, 자가학습 동안 **가림 증강을 계속 유지**한다(합성 anti-forgetting). 이 스택(약한 가림 헤드 → 카메라별 자가학습 + 가림 증강 유지)은 실측 적응과 가림 강건성을 **동시에** 확보한다: 예컨대 Kinect는 자가학습으로 실측 정확도가 오르면서(+0.017) 40% 가림 강건성(0.393)이 깨끗 헤드(0.376)와 RoboPEPP(0.351)를 모두 상회한다. 카메라별로 최적 구성을 선택한다 — Kinect/ORB는 스택, RealSense/Azure는 약한 가림 헤드 직접 사용.

> EN: **Camera-specific self-training with synthetic anti-forgetting.** For each real camera we self-train on detector pseudo-labels to close the domain gap. Because pure self-training washes out occlusion robustness, we **keep occlusion augmentation on during self-training** (synthetic anti-forgetting). This stack (light occlusion head → per-camera self-training with occlusion augmentation retained) secures real adaptation and occlusion robustness **simultaneously**: e.g., Kinect gains real accuracy from self-training (+0.017) while its 40% occlusion robustness (0.393) still beats both the clean-only head (0.376) and RoboPEPP (0.351). We pick the best configuration per camera — stack for Kinect/ORB, light occlusion head directly for RealSense/Azure.

**평가 프로토콜.** 가림 강건성은 RoboPEPP의 프로토콜(로봇 bbox 면적의 0–40%를 사각 occluder로 페이스트)로 별도 측정하며, 직접 비교를 위해 동일 규약을 따른다(§4.3).

> EN: **Evaluation protocol.** We measure occlusion robustness separately with RoboPEPP's protocol (paste rectangular occluders over 0–40% of the robot's bbox area), following the same convention for direct comparison (§4.3).

---

## 4. Experiments (실험)

### 4.1 설정 (Setup)

DREAM은 Panda·KUKA iiwa7·Baxter 세 로봇을 포함하는 벤치마크지만, **공개 실측 테스트 이미지는 Panda에만 존재**하고 KUKA·Baxter는 합성(domain-randomized) 전용이다. 따라서 헤드라인 실측 평가는 Panda의 4개 카메라 스플릿(RealSense, Kinect360, Azure, ORB)에서 수행하고, KUKA·Baxter는 §4.7에서 합성 스플릿으로 별도 보고한다. 지표는 표준 **ADD-AUC@100mm**(0–100mm 임계에서 ADD 정확도 곡선의 면적)이다. 우리 프로토콜은 **관절각 예측(predicted-joint) + 완전 자동 바운딩 박스**(bbox-from-solved) + sim-to-real 학습으로, GT 바운딩 박스를 쓰는 관례보다 엄격하다. 자가학습을 쓰는 카메라(RealSense/Kinect/ORB)는 시퀀스 앞 70%로 적응하고 뒤 30% 영역에서만 평가하여 정보 누수를 차단한다(anti-leak held-out, 카메라당 1000프레임 조밀 샘플). 백본은 DINOv3 ViT-B/16으로 전 과정 동결한다.

> EN: DREAM is a three-robot benchmark (Panda, KUKA iiwa7, Baxter), but **public real test images exist only for Panda**; KUKA and Baxter are synthetic (domain-randomized) only. We therefore run the headline real evaluation on Panda's four camera splits (RealSense, Kinect360, Azure, ORB) and report KUKA/Baxter separately on synthetic splits in §4.7. The metric is the standard **ADD-AUC@100mm**. Our protocol is **predicted-joint + fully automatic bounding boxes** (bbox-from-solved) with sim-to-real training, stricter than the common GT-box practice. Self-training cameras (RealSense/Kinect/ORB) adapt on the first 70% of the sequence and are evaluated on the last-30% region to prevent leakage (anti-leak held-out; 1000 densely-sampled frames per camera). The backbone is a DINOv3 ViT-B/16, frozen throughout.

### 4.2 주요 결과 (Main results)

DINObotPose3는 predicted-joint 체제에서 평균 ADD-AUC **0.804**로 최고 성능을 달성하며, **4개 카메라 전부** 강한 기준선 RoboPEPP를 상회한다(표 1). RoboPEPP의 헤드라인이 GT-bbox인 반면 우리는 완전 자동 bbox임을 다시 강조한다.

> EN: DINObotPose3 attains the best mean ADD-AUC of **0.804** in the predicted-joint regime and surpasses the strong RoboPEPP baseline on **all four cameras** (Table 1) — while, again, using fully automatic boxes against RoboPEPP's GT-box headline.

**표 1. DREAM 실측 카메라별 ADD-AUC@100mm (predicted-joint).** 1000-프레임 재잠금.

| 카메라 | **Ours** | RoboPEPP (GT-bbox) | RoboTAG | 격차(vs PEPP) |
|---|---|---|---|---|
| RealSense | **0.815** | 0.805 | 0.783 | +0.010 |
| Kinect360 | **0.828** | 0.785 | 0.757 | +0.043 |
| Azure | **0.795** | 0.753 | 0.831 | +0.042 |
| ORB | **0.778** | 0.775 | 0.588 | +0.003 |
| **Mean** | **0.804** | 0.780 | 0.740 | **+0.024** |

> EN: **Table 1. Per-camera ADD-AUC@100mm on DREAM-real (predicted-joint), 1000-frame re-lock.** Ours beats RoboPEPP on every camera (mean +0.024); RoboTAG wins only on Azure but collapses on ORB (0.588) under automatic detection.

**프로토콜을 통제한 전체 비교(표 2)** 는 우리 0.804가 predicted-joint 체제의 최고임을 보인다. known-joint 계열(CtRNet 86.4, CtRNet-X 86.2)은 관절각을 엔코더로 받는 **더 쉬운 문제**이므로 별도 리그로 분리한다.

> EN: **A protocol-controlled comparison (Table 2)** shows 0.804 is best in the predicted-joint regime. The known-joint family (CtRNet 86.4, CtRNet-X 86.2) receives encoder joint angles — an **easier problem** — and is separated into its own league.

**표 2. Predicted-joint DREAM-real 평균 ADD-AUC (프로토콜 통제).**

| 방법 | Mean | 관절각 | bbox | 깊이 모호성 해법 |
|---|---|---|---|---|
| DREAM (R101-H) | 57.8 | known | — | keypoint+PnP |
| RoboPose | 73.2 | predicted | init 의존 | 반복 render&compare |
| RoboTAG | 74.0 | predicted | 자동 | end-to-end 회귀 |
| HoRoPose | 77.2 | predicted | — | 학습된 root-DepthNet |
| RoboPEPP | 78.0 | predicted | **GT** | masking-pretrain |
| **Ours** | **80.4** | predicted | **자동** | **테스트-타임 SAM-실루엣 RC** |
| *(별도 리그)* CtRNet / CtRNet-X | 86.4 / 86.2 | **known** | — | 학습-타임 실루엣 자기지도 |

> EN: **Table 2. Predicted-joint DREAM-real mean ADD-AUC (protocol-controlled).** Known-joint CtRNet(-X) is a separate league (encoder angles). Ours is the best predicted-joint method under the hardest (automatic-bbox) setting.

### 4.3 가림 강건성 (Occlusion robustness)

RoboPEPP의 가림 프로토콜(로봇 bbox 면적의 0–40%를 사각 occluder로 페이스트)로 평가하면, DINObotPose3는 **모든 가림 수준에서** RoboPEPP를 상회한다(표 3). 이 우위의 원천은 (a) 우리에게만 있는 렌더-비교 깊이 보정, (b) 처음부터 가림에 노출된 약한 가림-증강 헤드다.

> EN: Under RoboPEPP's occlusion protocol (paste rectangular occluders over 0–40% of the robot's bbox area), DINObotPose3 exceeds RoboPEPP at **every** occlusion level (Table 3). The advantage stems from (a) the render-compare depth corrector unique to us and (b) a light occlusion-augmentation head exposed to occlusion from the start.

**표 3. 가림 수준별 ADD-AUC.**

| 가림 % | 0 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|
| **Ours (light+RC)** | **0.812** | **0.765** | **0.678** | **0.575** | **0.429** |
| RoboPEPP | 0.795 | 0.730 | 0.600 | 0.470 | 0.351 |

> EN: **Table 3. ADD-AUC vs occlusion level.** Ours dominates across 0–40%; the gap widens at heavier occlusion (+0.078 at 40%).

### 4.4 절제 실험 (Ablations)

**누적 마일스톤(표 4).** 강한 기준선 RoboPEPP(0.780)에서 시작해, 테스트-타임 렌더-비교가 mean을 0.796으로, 무료 DARK 디코딩이 0.799로, 가림-증강→자가학습 스택이 0.804로 끌어올린다. **성능 향상의 대부분이 학습 불필요 레버**(RC·DARK)에서 나온다.

> EN: **Cumulative milestones (Table 4).** From the strong RoboPEPP baseline (0.780), test-time render-and-compare lifts the mean to 0.796, free DARK decoding to 0.799, and the occlusion-aug→self-train stack to 0.804. **Most of the gain comes from training-free levers** (RC, DARK).

**표 4. 누적 절제 (mean ADD-AUC).**

| 구성 | Mean | Δ |
|---|---|---|
| RoboPEPP (기준선) | 0.780 | — |
| + 테스트-타임 render-and-compare | 0.796 | +0.016 |
| + DARK 서브픽셀 디코딩 (무료) | 0.799 | +0.003 |
| + occ-aug → self-train 스택 | **0.804** | +0.005 |

> EN: **Table 4. Cumulative ablation (mean ADD-AUC).** Render-and-compare and DARK are training-free; the occlusion stack trades a hair of clean accuracy for occlusion robustness.

**렌더-비교의 카메라별 기여.** RC는 깊이 신호가 약한 **원거리 카메라의 엔진**이다 — RealSense +0.070, Kinect +0.060, ORB +0.040. 반면 근거리 Azure는 깊이 신호가 이미 강해 RC를 끄는 것이 최적이다(카메라별 on/off). 이는 RC가 "포즈 전체 추정"이 아니라 **깊이/스케일 보정기**로 작동함을 정량적으로 확인한다.

> EN: **Per-camera render-compare contribution.** RC is the engine for **far cameras** where depth is weak — RealSense +0.070, Kinect +0.060, ORB +0.040 — whereas for the near Azure camera it is best turned off (per-camera on/off). This quantitatively confirms RC acts as a **depth/scale corrector**, not a full-pose estimator.

**무료 레버.** cov-PnP는 20% 가림에서 +0.011로 do-no-harm을 유지하며, DARK는 특히 원거리 ORB의 격차를 −0.010→−0.004로 좁힌다.

> EN: **Free levers.** cov-PnP adds +0.011 at 20% occlusion with do-no-harm elsewhere, and DARK narrows the far-camera ORB gap from −0.010 to −0.004.

**가림 강건성의 출처.** 40% 가림에서 깨끗하게만 학습한 헤드(0.376)보다 약한 가림-증강 헤드(0.420)가 강건하며, 배포 스택은 그 강건성을 대부분 유지(0.396)하면서 실측 정확도를 회복한다. 즉 **가림 강건성은 처음부터 증강 학습해야 배어든다.**

> EN: **Source of occlusion robustness.** At 40% occlusion, the light occlusion-augmentation head (0.420) is more robust than a clean-only head (0.376), and the deployed stack retains most of it (0.396) while recovering real-image accuracy — i.e., **robustness must be trained in from the start via augmentation.**

### 4.5 프로토콜 분석 (Protocol analysis)

**자동 bbox는 진짜 어렵다.** ORB 카메라는 시점이 다양해 자동 검출이 붕괴한다 — 동일한 자동-bbox 조건에서 RoboPEPP의 ORB는 GT-bbox 0.775에서 **0.344로 급락**한다. 우리 bbox-from-solved는 이 붕괴를 피해 0.778을 유지한다. 즉 우리 비교는 기준선에 불리한(더 엄격한) 조건에서 이루어진다.

> EN: **Automatic bounding boxes are genuinely hard.** The ORB camera's diverse viewpoints break automatic detection — under the same automatic-bbox condition, RoboPEPP's ORB **drops from 0.775 (GT-box) to 0.344**. Our bbox-from-solved avoids this collapse and holds 0.778, so our comparison is run under a setting that disadvantages (is stricter for) the baselines.

### 4.6 재잠금 안정성 (Re-lock stability)

논문급 신뢰성을 위해 표본을 800에서 1000프레임으로 늘려 재측정했다. 평균은 0.8037→**0.8039**로 사실상 불변(Δ+0.0002)이고 개별 카메라 변동도 ≤0.006이며, **4/4 카메라 모두 RoboPEPP를 상회**한다(ORB가 −0.002→+0.003으로 전환). 결과는 표본 수에 강건하다.

> EN: **Re-lock stability.** For paper-grade confidence we re-measured at 1000 (vs 800) frames: the mean is essentially unchanged (0.8037→**0.8039**, Δ+0.0002) with per-camera drift ≤0.006, and **all four cameras beat RoboPEPP** (ORB flips −0.002→+0.003). Results are robust to sample size.

### 4.7 DREAM의 다른 로봇: KUKA iiwa7 · Baxter (합성)

DREAM의 나머지 두 로봇은 실측 데이터가 없으므로 합성(DR) 스플릿에서 평가한다. 검출기는 Panda 검출기에서 전이학습하여 2D 키포인트 AUC **0.735**(KUKA)·**0.817**(Baxter)를 얻는다. 운동학 FK는 표준 URDF 대신 **DREAM 합성 데이터에 직접 피팅**하여(관절각↔키포인트 3D) 링크 원점을 RMS 0.003mm로 재현한다. 포즈는 head 각도 + 회전 헤드의 R,t를 직접 쓰는 direct-pose로 ADD-AUC **0.34**(KUKA)·**0.25**(Baxter)를 기록한다(표 5, 그림 8).

> EN: The other two DREAM robots have no real data, so we evaluate on synthetic (DR) splits. Detectors transfer-learned from the Panda detector reach 2D-keypoint AUC **0.735** (KUKA) / **0.817** (Baxter). Instead of a standard URDF, we **fit the kinematic FK directly to DREAM's synthetic data** (joint angles ↔ 3D keypoints), reproducing link origins at 0.003 mm RMS. Pose via a direct-pose scheme (head angles + rotation-head R,t) gives ADD-AUC **0.34** (KUKA) / **0.25** (Baxter) (Table 5).

**표 5. DREAM 로봇별 파이프라인 요약.**

| 로봇 | 데이터 | 검출기 2D AUC | 포즈 ADD-AUC | 병목 |
|---|---|---|---|---|
| Panda | **실측** | — | **0.804** | — (SOTA) |
| KUKA iiwa7 | 합성 | 0.735 | 0.34 | 회전 헤드 병진 오차(56mm) |
| Baxter 좌완 | 합성 | 0.817 | 0.25 | 손목 관절 **관측성 천장** |

> EN: **Table 5. Pipeline summary across DREAM robots.** Panda (real) is the headline SOTA; KUKA/Baxter (synthetic) validate the pipeline end-to-end.

**⚠️ 비교 주의.** KUKA/Baxter의 합성 0.34/0.25는 Panda 실측 0.804와 **다른 데이터(합성)·다른 조건(render-compare 미적용)**이므로 직접 비교 대상이 아니다. 이 결과의 의의는 절대 수치가 아니라 (i) 동일 파이프라인이 세 로봇 모두에서 동작하고, (ii) 남은 병목을 정량 규명했다는 점이다.

> EN: **Comparison caveat.** The synthetic 0.34/0.25 for KUKA/Baxter are **different data (synthetic) and different conditions (no render-compare)** than Panda's real 0.804 and are not directly comparable. Their value is not the absolute numbers but (i) that the same pipeline runs on all three robots and (ii) the quantitative bottleneck analysis below.

**관측성 천장 분석(그림 9).** Baxter 손목의 지배적 오차는 검출 실패가 아니라 **관측성**에서 온다: 손목 관절의 자기축 회전은 자기 키포인트 원점을 움직이지 않아, 완벽한(GT) 키포인트를 헤드에 주입해도 손목 각도가 거의 개선되지 않는다(손목 MAE 28.1°→27.6°). 즉 2D 키포인트 기하만으로는 손목 방향이 원리적으로 미결정이며, 이를 풀려면 기하가 아니라 그리퍼/손목의 **appearance**를 읽어야 한다(엔드이펙터 특징 패치). 이는 로봇 팔 포즈에서 원위 관절이 갖는 근본적 관측성 한계를 드러낸다.

> EN: **Observability-ceiling analysis.** Baxter's dominant wrist error comes from **observability**, not detection failure: a wrist joint's self-axis rotation does not move its own keypoint origin, so injecting perfect (GT) keypoints into the head barely changes the wrist angle (wrist MAE 28.1°→27.6°). The wrist orientation is thus fundamentally under-determined by 2D keypoint geometry and must instead be read from the **appearance** of the gripper/wrist (an end-effector feature patch). This exposes a fundamental observability limit of distal joints in robot-arm pose estimation.

---

## 5. Conclusion (결론)

우리는 관절각을 예측하고 바운딩 박스를 완전 자동으로 잡는 가장 어려운 설정에서 단안 관절형 로봇 포즈를 추정하는 기하-유도 파이프라인 DINObotPose3를 제시했다. 핵심 설계 원칙은 **동결 파운데이션 특징을 끝까지 신뢰하고, 깊이·불확실도·가림을 학습형 회귀가 아니라 기하로 처리**하는 것이다: DARK 서브픽셀 디코딩과 공분산-인지 PnP는 학습 없이 정확도를 끌어올리고, 제로샷 SAM 마스크에 대한 미분가능 렌더-비교는 테스트-타임에 깊이/스케일만 보정한다. 그 결과 DREAM 실측에서 평균 ADD-AUC 0.804로 predicted-joint 체제의 최고 성능을 달성하며, 완전 자동 바운딩 박스를 쓰면서도 RoboPEPP·RoboTAG를 능가하고 평가한 모든 가림 수준에서 앞선다.

> EN: We presented DINObotPose3, a geometry-guided pipeline for monocular articulated robot pose estimation under the hardest setting — predicted joints and fully automatic bounding boxes. Its central design principle is to **trust frozen foundation features and handle depth, uncertainty, and occlusion with geometry rather than learned regression**: DARK sub-pixel decoding and covariance-aware PnP raise accuracy at no training cost, while differentiable render-and-compare against zero-shot SAM masks corrects only depth/scale at test time. The result is a mean ADD-AUC of 0.804 on DREAM-real — best in the predicted-joint regime — surpassing RoboPEPP and RoboTAG with fully automatic boxes and leading across all evaluated occlusion levels.

우리는 렌더-비교를 발명했다고 주장하지 않는다. 그 개념은 RoboPose와 CtRNet의 선행이며, 우리 기여는 그것을 **학습이 필요 없는 테스트-타임 깊이 보정기**로, 제로샷 SAM과 동결 DINOv3 키포인트 프론트엔드 위에서, predicted-joint·자동-bbox 체제에 맞게 재구성한 데 있다. 이는 학습형 깊이 회귀(HoRoPose)나 종단간 회귀(RoboTAG)만이 답이 아니며, **파운데이션 특징 + 불확실도-인지 기하 + 제로샷 렌더 비교**의 조합이 강력한 대안임을 보인다.

> EN: We do not claim to have invented render-and-compare — the concept is prior art from RoboPose and CtRNet — and our contribution is recasting it as a **training-free test-time depth corrector** built on zero-shot SAM and a frozen-DINOv3 keypoint front-end for the predicted-joint / auto-bbox regime. This shows that learned depth regression (HoRoPose) and end-to-end regression (RoboTAG) are not the only answers: the combination of **foundation features, uncertainty-aware geometry, and zero-shot render comparison** is a strong alternative.

DREAM의 세 로봇에 동일 파이프라인을 적용하여, 검출·데이터-피팅 운동학·포즈까지 end-to-end로 일반화됨을 확인했다. 이 과정에서 **원위 관절의 관측성 천장**이라는 근본 한계를 정량 규명했다: 손목 관절의 자기축 회전은 자기 키포인트를 움직이지 않아, 완벽한 키포인트로도 각도가 미결정이며 오직 appearance로만 풀린다. 이는 키포인트 기반 로봇 포즈 추정 전반에 적용되는 통찰이다.

> EN: Applying the same pipeline to all three DREAM robots, we confirmed end-to-end generalization of detection, data-fit kinematics, and pose. In doing so we quantified a fundamental limit — an **observability ceiling for distal joints**: a wrist joint's self-axis rotation does not move its own keypoint, so its angle is under-determined even with perfect keypoints and is resolvable only from appearance — an insight that applies broadly to keypoint-based robot pose estimation.

**한계와 향후 과제.** DREAM은 KUKA·Baxter의 공개 실측 데이터가 없어 이 두 로봇의 실측 SOTA 비교는 불가능하다. 렌더-비교 깊이 보정을 KUKA로 확장하려면 벤치마크와 정합하는 로봇 메쉬가 필요하다(공개 iiwa7 메쉬는 DREAM 모델과 ~20mm 어긋나 정합 불가였고, Baxter는 정확히 일치함을 확인). Baxter 손목의 관측성 천장은 엔드이펙터 appearance를 명시적으로 읽는 헤드로 접근할 수 있다. 마지막으로, 우리 렌더-비교는 안정성을 위해 카메라별 on/off와 앵커링에 의존하므로, 자유 실루엣 최적화의 깊이 모호성을 원리적으로 억제하는 정식화가 남은 과제다.

> EN: **Limitations and future work.** DREAM provides no public real data for KUKA or Baxter, precluding a real-data SOTA comparison for those robots. Extending render-compare depth correction to KUKA requires a robot mesh matching the benchmark (public iiwa7 meshes differed from the DREAM model by ~20 mm and could not be aligned, whereas Baxter matched exactly). Baxter's wrist observability ceiling can be attacked with a head that explicitly reads end-effector appearance. Finally, our render-compare relies on per-camera on/off and anchoring for stability; a formulation that principally suppresses the depth ambiguity of free silhouette optimization remains future work.
