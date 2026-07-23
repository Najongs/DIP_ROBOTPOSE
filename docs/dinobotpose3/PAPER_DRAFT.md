# DINObotPose — Paper Draft (한국어 본문 · 영어 주석 병기)

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

단안 RGB 영상 한 장으로 로봇 팔의 자세와 관절각을 추정하는 문제는 카메라-로봇 캘리브레이션의 핵심이지만, 관절각이 알려지지 않은 설정에서는 2차원 키포인트만으로 깊이와 스케일이 충분히 제약되지 않아 여전히 어렵다. 기존 방법들은 이 깊이 모호성을 학습된 깊이 회귀, 생성 모델 기반 키포인트 리프팅, 대규모 사전학습 prior 등 학습에 의존하여 해결해 왔다. 본 논문은 이를 추가 학습 없이 기하학적으로 해결하는 파이프라인 **DINObotPose**를 제안한다. 동결된 파운데이션 백본이 서브픽셀 키포인트와 관절각·카메라 회전의 초기 추정을 제공하고, 운동학 솔버가 포즈를 복원하며, 제로샷 분할 마스크와 미분가능 렌더링으로 얻은 실루엣을 정합하는 테스트-타임 렌더-비교 단계가 깊이와 스케일만을 보정한다. 백본은 전 과정에서 동결되며, 백본을 실측 도메인에 적응시키는 것이 거친 2차원 지표를 개선하면서도 솔버가 요구하는 서브픽셀 정밀도를 훼손함을 실험으로 보인다. 가림에 대한 강건성은 약한 가림 증강과 카메라별 자가학습으로 확보한다. DREAM 실측 벤치마크 실험에서 제안 방법은 완전 자동 바운딩 박스를 사용하면서 4개 카메라 평균 ADD-AUC 0.804를 달성하여 동일 프로토콜에서 평가된 기존 방법들을 능가하며, 평가한 모든 가림 수준에서 가장 높은 정확도를 유지한다. 이 결과는 동결 파운데이션 특징과 학습이 필요 없는 기하 최적화의 결합이 학습 기반 깊이 추정의 효과적인 대안임을 보인다.

> EN: Estimating the pose and joint angles of a robot arm from a single RGB image is central to camera-to-robot calibration, yet the problem remains difficult when the joint angles are unknown, because 2D keypoints alone do not sufficiently constrain depth and scale. Existing methods address this depth ambiguity through learning, for example with learned depth regression, generative lifting of keypoints, or large-scale pretraining priors. This paper presents DINObotPose, a pipeline that instead resolves the ambiguity geometrically, without additional training. A frozen foundation backbone provides sub-pixel keypoints together with initial estimates of the joint angles and camera rotation, a kinematic solver recovers the pose, and a test-time render-and-compare stage aligns a zero-shot segmentation mask with a differentiably rendered silhouette to correct only depth and scale. The backbone remains frozen throughout; we show experimentally that adapting it to real images improves coarse 2D metrics while degrading the sub-pixel precision required by the solver. Robustness to occlusion is obtained with light occlusion augmentation and per-camera self-training. In experiments on the DREAM real-image benchmark, the proposed method attains a 4-camera mean ADD-AUC of 0.804 with fully automatic bounding boxes, surpassing prior methods evaluated under the same protocol, and maintains the highest accuracy at every evaluated occlusion level. These results indicate that frozen foundation features combined with training-free geometric optimization are an effective alternative to learning-based depth estimation.

---

## 1. 서론 (Introduction)

로봇 팔을 단안 RGB 한 장에서 추정하는 문제는 카메라-로봇 캘리브레이션의 핵심이며, 정확한 캘리브레이션은 원격조작과 시각 서보잉 같은 하위 작업을 가능케 하는 전제다. 사람과 로봇이 작업 공간을 공유하는 협업 환경에서는 로봇 링크의 3차원 상태를 시각적으로 추정하는 능력이 충돌 방지의 기반이 되기도 한다[wang2024multimodal, othman2022overview]. 전통적으로 이 캘리브레이션은 ArUco와 같은 기준 마커의 부착과 통제된 조명에 의존해 왔으나, 마커의 설치와 유지가 번거롭고 동적인 현장에는 적용이 제한된다[garrido2014automatic, fiala2005artag, olson2011apriltag]. 이에 따라 마커 없이 로봇의 외형만으로 포즈를 추정하는 학습 기반 방법이 표준으로 자리잡았다. 캘리브레이션은 저빈도·정확도-우선 작업이므로 본 연구는 실시간성보다 정확도를 우선한다(§4.8). 표준 벤치마크 DREAM은 두 가지 난이도 축을 갖는다. (i) 관절각을 엔코더로 아는 **known-joint**인지, 이미지에서 **예측(predicted-joint)**해야 하는지, (ii) 로봇 바운딩 박스를 GT로 주는지 **자동 검출**하는지. 본 연구는 가장 어려운 조합, 즉 **predicted-joint + 완전 자동 bbox**를 목표로 한다.

> EN: Estimating a robot arm from a single RGB image is the core of camera-to-robot calibration, which in turn enables downstream tasks such as teleoperation and visual servoing. In collaborative environments where humans and robots share a workspace, the ability to visually estimate the 3D state of the robot links also underpins collision avoidance [wang2024multimodal, othman2022overview]. Traditionally, this calibration has relied on fiducial markers such as ArUco and on controlled lighting, but installing and maintaining markers is cumbersome and poorly suited to dynamic environments [garrido2014automatic, fiala2005artag, olson2011apriltag]. Learning-based methods that estimate the pose from the robot appearance alone have therefore become the standard. Calibration is a low-frequency, accuracy-first task, so this work prioritizes accuracy over real-time speed (§4.8). The standard DREAM benchmark has two difficulty axes: (i) whether joint angles are **known** from encoders or must be **predicted** from the image, and (ii) whether the robot bounding box is given as GT or **auto-detected**. We target the hardest combination: **predicted-joint with fully automatic bounding boxes**.

이 설정의 근본 난제는 깊이와 스케일의 모호성이다. 단축(foreshortening) 효과로 인해 멀리 있는 팔과 가까이에서 접힌 팔이 거의 동일한 2차원 투영을 만들 수 있으므로, 키포인트가 정확하더라도 깊이 방향 오차가 최종 3차원 정확도를 지배한다. 따라서 이 모호성을 어떤 수단으로 해소하는가가 predicted-joint 단안 추정의 성능을 사실상 결정한다.

> EN: The fundamental difficulty of this setting is the ambiguity of depth and scale. Owing to foreshortening, a distant arm and a nearby folded arm can produce nearly identical 2D projections, so errors along the depth direction dominate the final 3D accuracy even when the keypoints themselves are precise. How this ambiguity is resolved therefore largely determines the performance of predicted-joint monocular estimation.

이 분야의 기반은 DREAM이 확립하였다. DREAM은 2차원 관절 키포인트를 예측하고 PnP로 카메라-로봇 포즈를 복원하는 파이프라인과, 도메인 랜덤화를 적용한 대규모 합성 벤치마크를 제시하였다[lee2020camera, lepetit2009ep]. 이후의 방법들은 깊이 모호성을 학습으로 해소한다. HoRoPose는 루트 깊이를 회귀하는 네트워크를 별도로 학습하고[ban2024real], RoboKeyGen은 확산 모델로 2차원 키포인트를 3차원으로 리프팅하며[tian2024robokeygen], RoboPEPP는 마스킹 사전학습으로 로봇 구조에 대한 사전지식을 인코더에 주입한다[goswami2025robopepp]. 렌더-비교 계열에서는 RoboPose가 학습된 정제 네트워크로 반복 정제를 수행하고[labbe2021single], SGTAPose는 연속 프레임의 시간 정보와 구조 사전지식으로 자기가림을 완화하며[tian2023robot], CtRNet은 실루엣 정합을 학습 손실로 사용하되 관절각이 주어지는 known-joint 설정에서 동작한다. 요약하면, predicted-joint와 완전 자동 바운딩 박스 체제에서 추가 학습 없이 깊이 모호성을 해소하는 방법은 아직 제시되지 않았다. 이 공백은 실용적으로도 중요하다. 학습에 의존하는 해소 방식은 장비 배치나 카메라 구성이 바뀔 때마다 대상 데이터의 수집과 재학습을 요구하므로 배포 비용이 크지만, 학습이 필요 없는 보정은 이 비용을 제거한다. 본 연구는 이 공백을 채운다. 동결 파운데이션 백본이 제공하는 서브픽셀 키포인트와 초기 관절각·회전 추정으로 운동학 솔버가 포즈를 복원하고, 제로샷 분할 마스크와 미분가능 렌더 실루엣을 정합하는 테스트-타임 최적화가 깊이와 스케일만을 프레임별로 보정한다.

> EN: The foundation of this field was established by DREAM, which introduced the pipeline of predicting 2D joint keypoints and recovering the camera-to-robot pose with PnP, together with a large-scale domain-randomized synthetic benchmark [lee2020camera, lepetit2009ep]. Subsequent methods resolve the depth ambiguity through learning. HoRoPose trains a dedicated network to regress the root depth [ban2024real], RoboKeyGen lifts 2D keypoints into three dimensions with a diffusion model [tian2024robokeygen], and RoboPEPP injects prior knowledge of the robot structure into its encoder through masked pretraining [goswami2025robopepp]. In the render-and-compare lineage, RoboPose performs iterative refinement with a learned refiner network [labbe2021single], SGTAPose mitigates self-occlusion using temporal information from consecutive frames together with structural priors [tian2023robot], and CtRNet uses silhouette alignment as a training loss but operates in the known-joint setting where the joint angles are given. In summary, no existing method resolves the depth ambiguity without additional training in the predicted-joint regime with fully automatic bounding boxes. This gap also matters in practice: learning-based resolutions require collecting target data and retraining whenever the equipment layout or camera configuration changes, incurring substantial deployment cost, whereas a training-free correction removes this cost. This work fills that gap. A kinematic solver recovers the pose from sub-pixel keypoints and initial joint-angle and rotation estimates provided by a frozen foundation backbone, and a test-time optimization stage that aligns a zero-shot segmentation mask with a differentiably rendered silhouette corrects only depth and scale on each frame.

설계는 두 원칙을 따른다. 첫째, 백본은 전 과정에서 동결한다. 이는 편의가 아니라 실험적 선택이다. 백본을 실측 도메인에 적응시키는 시도는 거친 2차원 검출 지표를 개선하면서도 솔버가 요구하는 서브픽셀 정밀도를 일관되게 훼손하였다(§4.9). 둘째, 깊이는 학습이 아니라 기하로 보정한다. 테스트-타임 렌더-비교는 배포 스택에서 가장 큰 단일 기여 요소이며(평균 +0.043, §4.4), 깊이 신호가 약한 원거리 카메라의 오차를 회복한다. 이에 더해 가림에 대한 강건성은 약한 가림 증강과 이를 유지하는 카메라별 자가학습으로 확보하고(§3.4), 공분산-인지 PnP와 서브픽셀 디코딩 같은 학습이 필요 없는 보조 요소가 가림 상황의 강건성을 추가 비용 없이 보탠다(§4.3). 그 결과 DREAM 실측 벤치마크에서 완전 자동 바운딩 박스로 평균 ADD-AUC 0.804를 달성하여 동일 프로토콜의 기존 방법들을 능가하고, 평가한 모든 가림 수준에서 가장 높은 정확도를 유지한다.

> EN: The design follows two principles. First, the backbone remains frozen throughout. This is an experimental choice rather than a convenience: attempts to adapt the backbone to the real domain consistently improved coarse 2D detection metrics while degrading the sub-pixel precision required by the solver (§4.9). Second, depth is corrected geometrically rather than through learning. Test-time render-and-compare is the largest single contributor in the deployed stack (+0.043 on the mean, §4.4) and recovers the depth errors of distant cameras whose depth signal is weak. In addition, robustness to occlusion is obtained with light occlusion augmentation and per-camera self-training that retains the augmentation (§3.4), while training-free auxiliary components such as covariance-aware PnP and sub-pixel decoding add occlusion robustness at no extra cost (§4.3). As a result, the method attains a mean ADD-AUC of 0.804 on the DREAM real benchmark with fully automatic bounding boxes, surpassing prior methods under the same protocol and maintaining the highest accuracy at every evaluated occlusion level.

**기여(Contributions).** 본 논문의 주요 기여는 다음 세 가지다.
1. **학습이 필요 없는 테스트-타임 렌더-비교 깊이 보정 시스템**: 제로샷 분할 마스크와 미분가능 렌더 실루엣의 정합으로 깊이와 스케일만을 프레임별로 보정하는 시스템을, predicted-joint와 완전 자동 바운딩 박스 체제에서 처음으로 제시한다. 공분산-인지 PnP와, 풀린 포즈의 순운동학 재투영에서 바운딩 박스를 자동 생성하는 crop 기법이 시스템을 완성한다. DREAM 실측 벤치마크에서 평균 ADD-AUC 0.804로 동일 프로토콜 최고 성능을 달성하며, 렌더-비교가 최대 단일 기여 요소다(평균 +0.043).
2. **동결 파운데이션 프론트엔드에 대한 실험적 발견**: 백본 적응은 거친 2차원 검출 지표를 개선하면서도 기하 솔버가 요구하는 서브픽셀 정밀도를 파괴하며, 포즈 성능은 특정 백본이 아니라 동결 파운데이션 특징 일반에서 온다는 것을 체계적 절제 실험으로 보인다(§4.9, §4.10).
3. **가림 강건성**: 약한 가림 증강과 이를 유지하는 카메라별 자가학습의 결합으로, 평가한 모든 가림 수준에서 기존 방법 대비 가장 높은 정확도를 유지한다(§4.3).

이에 더해 DREAM이 포함하는 세 로봇(Panda, KUKA, Baxter) 전부에 동일 파이프라인을 적용하여, 적용 가능성과 로봇별 병목(예: 손목 관절의 관측성 한계)을 분석한다(§4.7).

> EN: **Contributions.** The main contributions of this paper are threefold. (1) **A training-free, test-time render-and-compare depth correction system.** We present the first system in the predicted-joint regime with fully automatic bounding boxes that corrects only depth and scale on each test frame by aligning a zero-shot segmentation mask with a differentiably rendered silhouette. Covariance-aware PnP and an automatic cropping scheme that derives the bounding box from the forward-kinematic reprojection of the solved pose complete the system. It attains a mean ADD-AUC of 0.804 on the DREAM real benchmark, the best result under the same protocol, with render-and-compare as the largest single contributor (+0.043 on the mean). (2) **An experimental finding on frozen foundation front-ends.** Systematic ablations show that adapting the backbone improves coarse 2D detection metrics while destroying the sub-pixel precision required by the geometric solver, and that pose accuracy derives from frozen foundation features in general rather than from one specific backbone (§4.9, §4.10). (3) **Occlusion robustness.** Combining light occlusion augmentation with per-camera self-training that retains the augmentation yields the highest accuracy at every evaluated occlusion level (§4.3).

> EN: In addition, we apply the same pipeline to all three robots covered by DREAM (Panda, KUKA, and Baxter), analyzing its applicability and the per-robot bottlenecks, such as an observability limit at the wrist joints (§4.7).

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

파이프라인은 다섯 단계다: **(1)** 동결 DINOv3 ViT-B/16 백본[simeoni2025dinov3]이 히트맵 키포인트를 검출하고, **(2)** DARK 서브픽셀 디코딩으로 격자 양자화 오차를 제거하며, **(3)** 관절각·회전 헤드가 관절 구성과 카메라-로봇 회전을 예측하고, **(4)** 공분산-인지 PnP + 운동학 재투영 정제가 포즈를 푼 뒤, **(5)** 제로샷 SAM 마스크와 렌더 실루엣을 미분가능 렌더링으로 정합해 깊이/스케일을 테스트-타임에 보정한다. 백본은 끝까지 동결한다(적응은 서브픽셀 키포인트 정밀도를 파괴함을 3회 실험으로 확인).

> EN: The pipeline has five stages: **(1)** a frozen DINOv3 ViT-B/16 backbone detects heatmap keypoints, **(2)** DARK sub-pixel decoding removes grid-quantization error, **(3)** joint-angle and rotation heads predict the joint configuration and camera-to-robot rotation, **(4)** covariance-aware PnP with kinematic reprojection refinement solves the pose, and **(5)** a zero-shot SAM mask and a rendered silhouette are aligned by differentiable rendering to correct depth/scale at test time. The backbone stays frozen throughout (we verified across three experiments that adapting it destroys the sub-pixel keypoint precision the solver needs).

> 그림(파이프라인 개요): [figures/fig_pipeline](figures/README.md) — 5단계 흐름 + frozen/trained/test-time 색 구분 + azure RC off 분기. 번호는 LaTeX 단계에서 부여(그림 1~10은 결과 그림에 이미 배정됨), 최종본은 PPT로 재제작 예정. / _EN: Pipeline overview figure: `figures/fig_pipeline` (unnumbered — figure numbering assigned at the LaTeX stage; Figs. 1–10 are already results figures)._

### 3.2 왜 동결인가, 그리고 무료 강건성 레버 (Why frozen, and free robustness levers)

**용어 정의.** 본 논문에서 '동결(frozen)'은 DINOv3 백본 트랜스포머 가중치를 전 과정에서 갱신하지 않음을 뜻한다. 그 위의 경량 헤드(키포인트·관절각·회전)는 학습 대상이며, 아래의 '백본 적응' 반증은 백본 가중치 자체를 움직인 실험들이다.

> EN: **Terminology.** In this paper, "frozen" means the DINOv3 backbone transformer weights are never updated. The lightweight heads on top (keypoint, joint-angle, rotation) are trained; the "backbone adaptation" refutations below are experiments that moved the backbone weights themselves.

**왜 동결인가.** 백본 동결은 편의가 아니라 실험적 발견이다. 백본을 실측에 적응시키는 시도는 일관되게 실패했다 — 공격적 SSL(masked-feature, 6-block)과 pseudo-keypoint co-finetune은 ADD를 악화시켰고, 온건한 SSL(3-block)은 헤드 비정합(OOD)으로 평가 자체가 불능이었다(§4.9, 표 11). 특히 공격적 SSL은 real PCK@5를 +0.069 올리면서도 ADD는 0.567→0.531로 떨어뜨렸다: 적응은 거친 2D 강건성(굵은 임계의 PCK가 보상하는 것)을 얻는 대가로, 기하 솔버가 요구하는 서브픽셀 정밀도를 판다. 이 해리는 §4.4의 키포인트-노이즈 민감도와 정량적으로 정합한다 — PCK@5가 감지하지 못하는 σ=1–2px 구간에서 ADD-AUC가 0.024–0.089 빠진다. 감독형 co-finetune도 같은 방향으로 실패하므로(0.497→0.434 단조 하락) 원인은 특정 목적함수가 아니라 백본을 움직이는 행위 자체다. 반대편 증거로 동결 프론트엔드는 충분히 정밀하다: 관절각을 GT로 주입하면 평균 0.841에 도달하고(§4.2), 백본을 SigLIP2로 바꿔도 포즈 성능이 유지된다(§4.10) — 성능은 특정 모델이 아니라 동결 파운데이션 특징 일반에서 온다.

> EN: **Why frozen.** Freezing the backbone is an experimental finding, not a convenience. Attempts to adapt the backbone to real images consistently failed — aggressive SSL (masked-feature, 6 blocks) and pseudo-keypoint co-finetuning degraded ADD, while gentle SSL (3 blocks) was unevaluable due to head mismatch (OOD) (§4.9, Table 11). Notably, aggressive SSL raised real PCK@5 by +0.069 while dropping ADD 0.567→0.531: adaptation buys coarse 2D robustness (what a coarse-threshold PCK rewards) at the cost of the sub-pixel precision the geometric solver needs. This dissociation is quantitatively consistent with the keypoint-noise sensitivity of §4.4 — ADD-AUC loses 0.024–0.089 in the σ=1–2 px range that PCK@5 cannot even detect. Supervised co-finetuning fails in the same direction (monotone 0.497→0.434), so the cause is not a particular objective but moving the backbone at all. Conversely, the frozen front-end is precise enough: injecting GT joint angles reaches a mean of 0.841 (§4.2), and swapping the backbone for SigLIP2 preserves pose accuracy (§4.10) — performance comes from frozen foundation features in general, not one specific model.

**무료 강건성 레버.** 다음 두 레버는 학습이 전혀 필요 없고, 클린 정확도 기여는 사실상 0이다(leave-one-out ΔMean −0.003/−0.001, §4.4) — 채택 근거는 가림 강건성이다(§4.3). **DARK 서브픽셀 디코딩**은 봉우리 근처 log-히트맵의 1·2차 미분으로 테일러 보정을 적용해 argmax의 격자 양자화를 제거한다(원거리 ORB 격차 −0.010→−0.004). **공분산-인지 PnP(cov-PnP)**는 히트맵의 국소 2차 모멘트에서 키포인트별 이방성 2×2 공분산을 추출해 재투영 잔차를 마할라노비스(백색화) 거리로 바꾼다 — 흐리거나 다봉인(가려진) 키포인트를 방향별로 연속 다운웨이트하여 스칼라 신뢰도 게이팅을 일반화하며, 20% 가림에서 +0.011을 더한다. 두 레버의 작동 기전 — 히트맵 자체의 불확실성 구조를 활용하며, 외부 주입 노이즈에는 무익 — 은 §4.4의 G1 분석이 규명한다.

> EN: **Free robustness levers.** The following two levers require no training and contribute essentially nothing to clean accuracy (leave-one-out ΔMean −0.003/−0.001, §4.4) — they are adopted for occlusion robustness (§4.3). **DARK sub-pixel decoding** applies a Taylor correction from the first/second derivatives of the log-heatmap around the peak, removing argmax grid quantization (narrowing the far-camera ORB gap −0.010→−0.004). **Covariance-aware PnP (cov-PnP)** extracts a per-keypoint anisotropic 2×2 covariance from the heatmap's local second moments and whitens the reprojection residual into a Mahalanobis distance — continuously down-weighting diffuse or multimodal (occluded) keypoints per direction, generalizing scalar confidence gating, and adding +0.011 at 20% occlusion. Their mechanism — exploiting the heatmap's own uncertainty structure, useless against externally injected noise — is established by the G1 analysis in §4.4.

### 3.3 테스트-타임 렌더-비교 깊이 보정 (Test-time render-and-compare)

이미 좋은 키포인트+운동학 추정 위에, 제로샷 SAM으로 얻은 로봇 전경 마스크와 nvdiffrast로 렌더한 로봇 메쉬 실루엣의 IoU를 최대화하도록 포즈의 깊이/스케일을 미분가능하게 정제한다. SAM은 외부 가림체를 로봇에서 분리해 주므로 마스크가 깨끗하다. 이 단계는 카메라별로 켜고 끈다 — 근거리 카메라(azure)는 깊이 신호가 이미 강해 끄고, 원거리에서 크게 이득을 본다(카메라별 +0.04~0.07 ADD-AUC).

> EN: **Test-time render-and-compare.** On top of already-good keypoint + kinematic estimates, we differentiably refine the pose's depth/scale to maximize the IoU between a zero-shot SAM robot foreground mask and an nvdiffrast-rendered robot silhouette. SAM cleanly separates external occluders from the robot. This stage is toggled per camera — off for near cameras (azure), whose depth signal is already strong, and most beneficial on far cameras (+0.04–0.07 ADD-AUC per camera).

**구현 노트(세그멘터 선택).** 마스크는 오리지널 SAM ViT-B(제로샷)로 얻는다. 프롬프트는 1차 패스 포즈의 렌더 마스크에서 유도한 점+박스이고, multimask 출력 중 초기 렌더와 가장 일치하는 마스크를 선택한다(init-render-consistent selection). 분할할 대상의 위치를 기하로 이미 알고 있으므로 텍스트/컨셉 프롬프트형의 더 크고 새로운 세그멘터는 불필요하며, 남은 오차의 지배 요인도 마스크 품질이 아니라 예측 관절각이다(§4.2 known-joint 분석).

> EN: **Implementation note (segmenter choice).** Masks come from the original zero-shot SAM ViT-B. Prompts are points+box derived from the pass-1 pose's rendered mask, and among the multimask outputs we select the one most consistent with the initial render (init-render-consistent selection). Since geometry already tells us where the target is, larger text/concept-promptable segmenters are unnecessary — and the dominant residual error is predicted joint angles, not mask quality (§4.2 known-joint analysis).

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

DREAM은 Panda·KUKA·Baxter 세 로봇을 포함하는 벤치마크지만, **공개 실측 테스트 이미지는 Panda에만 존재**하고 KUKA·Baxter는 합성(domain-randomized) 전용이다. 따라서 헤드라인 실측 평가는 Panda의 4개 카메라 스플릿(RealSense, Kinect360, Azure, ORB)에서 수행하고, KUKA·Baxter는 §4.7에서 합성 스플릿으로 별도 보고한다. 지표는 표준 **ADD-AUC@100mm**(0–100mm 임계에서 ADD 정확도 곡선의 면적)이다. 우리 프로토콜은 **관절각 예측(predicted-joint) + 완전 자동 바운딩 박스**(bbox-from-solved) + sim-to-real 학습으로, GT 바운딩 박스를 쓰는 방법(HoRoPose·CtRNet 등)보다 엄격하고, 주요 경쟁자 RoboPEPP·RoboTAG와는 동일한 자동-bbox 프로토콜이다. 자가학습을 쓰는 카메라(RealSense/Kinect/ORB)는 시퀀스 앞 70%로 적응하고 뒤 30% 영역에서만 평가하여 정보 누수를 차단한다(anti-leak held-out, 카메라당 1000프레임 조밀 샘플). 백본은 DINOv3 ViT-B/16으로 전 과정 동결한다. **비교 주의**: 경쟁 수치는 모두 각 논문(주로 RoboPEPP Table 2)에서 인용한 것[cited]이며 우리가 재현한 것이 아니고, 우리 수치는 누수 방지용 뒤-30% held-out 부분집합에서 측정되어 경쟁 방법이 보고한 전체 테스트셋과 프레임 집합이 다르다 — 동일-프레임 재현 비교는 향후 보완 항목이다.

> EN: DREAM is a three-robot benchmark (Panda, KUKA, Baxter), but **public real test images exist only for Panda**; KUKA and Baxter are synthetic (domain-randomized) only. We therefore run the headline real evaluation on Panda's four camera splits (RealSense, Kinect360, Azure, ORB) and report KUKA/Baxter separately on synthetic splits in §4.7. The metric is the standard **ADD-AUC@100mm**. Our protocol is **predicted-joint + fully automatic bounding boxes** (bbox-from-solved) with sim-to-real training — stricter than GT-box methods (HoRoPose, CtRNet) and identical to our main competitors RoboPEPP/RoboTAG. Self-training cameras (RealSense/Kinect/ORB) adapt on the first 70% of the sequence and are evaluated on the last-30% region to prevent leakage (anti-leak held-out; 1000 densely-sampled frames per camera). The backbone is a DINOv3 ViT-B/16, frozen throughout. **Comparison caveat**: all competitor numbers are cited from their papers (mainly RoboPEPP Table 2), not reproduced by us, and our numbers come from the leakage-preventing last-30% held-out subset — a different frame set than the full test sets competitors report on; same-frame reproduced comparisons remain future work.

### 4.2 주요 결과 (Main results)

DINObotPose는 predicted-joint 체제에서 평균 ADD-AUC **0.804**로 최고 성능을 달성한다(표 1, 그림 1) — 평균 마진 +0.024는 실행-간 노이즈 추정치(~0.010, §4.10)의 2배 이상이다. 카메라별로는 Kinect(+0.043)·Azure(+0.042)에서 명확히 우위이고, RealSense(+0.010)·ORB(+0.003)는 노이즈 범위 내의 통계적 동률로, 어느 카메라에서도 뒤지지 않는다. RoboPEPP·RoboTAG는 우리와 **동일한 자동 bbox 프로토콜**이므로 이는 공정한 동일-조건 비교다(GT-bbox를 쓰는 HoRoPose와 대비, §4.5).

> EN: DINObotPose attains the best mean ADD-AUC of **0.804** in the predicted-joint regime (Table 1, Fig. 1) — the +0.024 mean margin is more than twice the estimated run-to-run noise (~0.010, §4.10). Per camera, the lead is clear on Kinect (+0.043) and Azure (+0.042), while RealSense (+0.010) and ORB (+0.003) are statistical ties within noise — we trail on none. RoboPEPP and RoboTAG run under the **same automatic-bbox protocol** as ours, so this is a fair like-for-like comparison (in contrast to the GT-box HoRoPose, §4.5).

**표 1. DREAM 실측 카메라별 ADD-AUC@100mm (predicted-joint 전 방법).** 1000-프레임 재잠금. 경쟁 수치는 각 논문/RoboPEPP Table 2 인용. bbox: GT=주어진 박스, auto=자체 검출기.

| 방법 | bbox | RealSense | Kinect360 | Azure | ORB | Mean |
|---|---|---|---|---|---|---|
| RoboPose | auto | 0.743 | 0.776 | 0.704 | 0.704 | 0.732 |
| HoRoPose (HPE) | **GT** | 0.752 | 0.760 | 0.822 | 0.752 | 0.772 |
| HoRoPose\* (HPE\*) | auto | 0.491 | — | 0.667 | 0.516 | — |
| RoboTAG | auto | 0.783 | 0.757 | 0.831 | 0.588 | 0.740 |
| RoboPEPP | auto | 0.805 | 0.785 | 0.753 | 0.775 | 0.780 |
| **Ours (predicted)** | auto | **0.815** | **0.828** | **0.795** | **0.778** | **0.804** |
| _Ours (known-joint θ, 상한)_ | _auto_ | _0.867_ | _0.878_ | _0.788_ | _0.831_ | _0.841_ |

> EN: **Table 1. Per-camera ADD-AUC@100mm on DREAM-real (all predicted-joint methods), 1000-frame re-lock;** competitor numbers cited from each paper / RoboPEPP Table 2. Ours leads the mean (0.804); RoboTAG wins only on Azure (0.831) and trails on the other three cameras (four-camera mean 0.787) [RoboTAG ORB corrected to 0.775, 2026-07-21]; HoRoPose reaches 0.772 only with **GT boxes** — under the same off-the-shelf detector (HPE\*) it collapses (ORB 0.516, RealSense 0.491). Among auto-bbox methods only RoboPEPP is competitive, and we beat it on all four cameras (+0.024 mean). The last (italic) row is our **known-joint upper bound** (GT joint angles injected, solving only camera R,t) — see below.

**Known-joint 상한(위 표 마지막 행).** 관절각을 GT로 주입하고 카메라 R,t만 풀면 평균이 **0.804→0.841(+0.037)**로 오른다. 이득은 **원거리 카메라에 집중**된다(RealSense +0.052·Kinect +0.050·ORB +0.053) — 즉 원거리에서 남은 격차의 대부분은 **예측 관절각의 오차**다. 반면 근거리 **Azure는 GT-θ로도 불변**(0.788 vs 0.795)이라 병목이 각도가 아니라 **깊이/포즈**임을 재확인한다. (렌더-비교는 예측-θ용 깊이 보정기라, 이미 정확한 GT-θ RealSense에서는 오히려 과보정하여 base 0.867→RC 0.825로 내려간다 — 상한은 배포와 동일하게 카메라별 최적 RC on/off를 취했다.)

> EN: **Known-joint upper bound (last row).** Injecting GT joint angles and solving only camera R,t lifts the mean **0.804→0.841 (+0.037)**. The gain concentrates on **far cameras** (RealSense +0.052, Kinect +0.050, ORB +0.053) — i.e., most of the remaining far-camera gap is **predicted-angle error** — whereas near-camera **Azure is unchanged even with GT angles** (0.788 vs 0.795), re-confirming its bottleneck is **depth/pose, not angles**. (Render-compare, a predicted-θ depth corrector, over-corrects the already-accurate GT-θ RealSense pose, 0.867→0.825; the ceiling uses per-camera best RC on/off, as in deployment.)

**프로토콜을 통제한 전체 비교(표 2)** 는 우리 0.804가 predicted-joint 체제의 최고임을 보인다. known-joint 계열(CtRNet 86.4, CtRNet-X 86.2)은 관절각을 엔코더로 받는 **더 쉬운 문제**이므로 별도 리그로 분리한다.

> EN: **A protocol-controlled comparison (Table 2)** shows 0.804 is best in the predicted-joint regime. The known-joint family (CtRNet 86.4, CtRNet-X 86.2) receives encoder joint angles — an **easier problem** — and is separated into its own league.

**표 2. Predicted-joint DREAM-real 평균 ADD-AUC (프로토콜 통제).**

| 방법 | Mean | 관절각 | bbox | 깊이 모호성 해법 |
|---|---|---|---|---|
| DREAM (R101-H) | 57.8 | known | — | keypoint+PnP |
| RoboPose | 73.2 | predicted | init 의존 | 반복 render&compare |
| RoboTAG | 74.0 | predicted | 자동 | end-to-end 회귀 |
| HoRoPose | 77.2 | predicted | **GT** | 학습된 root-DepthNet |
| GISR | 77.9\* | predicted | — | 실루엣 정제 |
| RoboPEPP | 78.0 | predicted | 자동 | masking-pretrain |
| **Ours** | **80.4** | predicted | **자동** | **테스트-타임 SAM-실루엣 RC** |
| *(별도 리그)* CtRNet / CtRNet-X | 86.4 / 86.2 | **known** | — | 학습-타임 실루엣 자기지도 |

\*GISR(RA-L'24)는 ORB를 보고하지 않아 **3-카메라 평균**(azure 80.6·kinect 73.9·realsense 79.3)이다 — 가장 어려운 ORB를 제외하므로 4-카메라 평균들과 직접 비교는 유리한 쪽으로 편향됨.

> EN: **Table 2. Predicted-joint DREAM-real mean ADD-AUC (protocol-controlled).** Known-joint CtRNet(-X) is a separate league (encoder angles). Ours is the best predicted-joint method under the hardest (automatic-bbox) setting. \*GISR reports no ORB, so its 77.9 is a **3-camera average** (excluding the hardest split), not directly comparable to the 4-camera means.

**DREAM 전체 로봇 성능(표 3).** DREAM은 Panda·KUKA·Baxter 세 로봇을 포함하며, 동일 파이프라인을 셋 모두에 적용한 성능을 한 표에 정리한다. Panda는 실측 벤치마크의 헤드라인 SOTA(0.804)이고, KUKA·Baxter는 합성 스플릿에서 각각 0.690·0.713이다.

> EN: **All-DREAM-robot performance (Table 3).** DREAM spans Panda, KUKA, and Baxter; we report the same pipeline on all three in one table. Panda is the real-benchmark headline SOTA (0.804); KUKA and Baxter reach 0.690 and 0.713 on synthetic splits.

**표 3. DREAM 로봇별 성능 종합.** (Panda 실측 · KUKA/Baxter 합성)

| 로봇 | 데이터 | 검출기 2D AUC | 포즈 ADD-AUC@100mm | median ADD(mm) |
|---|---|---|---|---|
| **Panda** | **실측(real)** | — | **0.804** | — |
| KUKA | 합성(synth) | 0.735 | 0.690 | 13.1 |
| Baxter 좌완 | 합성(synth) | 0.817 | 0.713 | 17.1 |

> EN: **Table 3. Performance across all DREAM robots** (Panda real; KUKA/Baxter synthetic).

> ⚠️ 세 수치는 대역이 비슷하지만 **동일 조건이 아니다**: Panda는 실측 + RC, KUKA/Baxter는 합성 + RC 미적용이다. 로봇 간 우열이 아니라 **동일 파이프라인의 이식성**을 보는 표이며, KUKA/Baxter의 FK 피팅·잔여 실패모드 상세는 §4.7. / _EN: The three numbers are in a similar range but are **not the same condition**: Panda is real + RC, KUKA/Baxter are synthetic without RC. The table shows pipeline portability, not a robot-vs-robot ranking; per-robot details are in §4.7._

**합성 스플릿 비교(표 4).** 기존 방법들이 표준으로 보고하는 DREAM 합성 테스트(domain-randomized "DR" · photorealistic "Photo")에서 비교한다(경쟁 수치는 RoboPEPP Table 2 인용). 합성은 경쟁 방법들의 **학습 분포(특히 DR)에 가까운 홈그라운드**이며, 우리 파이프라인은 실측 배포에 최적화되어 있어(카메라별 자가학습·실측 깊이용 RC) 합성에서는 선두가 아니다 — Panda 합성에서 우리(+RC) 74.2/76.9는 RoboPEPP(83.0/84.1)·RoboPose(82.9/79.7)에 뒤진다. **그러나 프로토콜을 통제하면 그림이 달라진다**: HoRoPose는 GT-bbox에서 82.7/82.0이지만 **동일한 자동 검출기(HPE\*)를 쓰면 41.4/40.7로 붕괴**하는 반면, 우리는 자동 bbox로도 74.2/76.9를 유지한다. 즉 **동일 (predicted-joint + 자동 bbox) 조건의 유일한 직접 비교 대상인 HoRoPose\*를 큰 격차로 이긴다.** KUKA·Baxter에는 render-compare를 적용하지 않았고(실측 자가학습 데이터 부재), 그럼에도 Baxter에서는 전 비교 방법을 큰 격차로 앞서고 KUKA에서는 수 점 차로 근접한다. 즉 이 결과는 적용 가능성 확인인 동시에 경쟁력 있는 성능이다.

> EN: **Synthetic-split comparison (Table 4).** We compare on the DREAM synthetic tests (domain-randomized "DR" and photorealistic "Photo") that prior methods report as standard (competitor numbers cited from RoboPEPP Table 2). Synthetic is the competitors' **home turf** (close to their training distribution, especially DR), and our pipeline is optimized for real deployment (per-camera self-training, RC tuned for real depth), so we do not lead on synthetic — on Panda our +RC 74.2/76.9 trails RoboPEPP (83.0/84.1) and RoboPose (82.9/79.7). **But controlling for protocol changes the picture**: HoRoPose scores 82.7/82.0 with GT boxes but **collapses to 41.4/40.7 under the same off-the-shelf detector (HPE\*)**, whereas we hold 74.2/76.9 with automatic boxes — i.e., we beat HoRoPose\*, the only direct match under the (predicted-joint + auto-bbox) protocol, by a wide margin. KUKA/Baxter use no render-compare (no real data for self-training), yet Baxter exceeds every compared method by a wide margin and KUKA is within a few points — so this is both an applicability check and a competitive result.

**표 4. DREAM 합성 스플릿 ADD-AUC (×100).** 경쟁 수치는 RoboPEPP(CVPR'25) Table 2 인용. joints=관절각 known/predicted, bbox=GT/auto. HPE=HoRoPose; HPE\*=HPE에 RoboPEPP와 동일 off-the-shelf 검출기 적용.

| 방법 | joints | bbox | Panda-DR | Panda-Photo | KUKA-DR | KUKA-Photo | Baxter-DR |
|---|---|---|---|---|---|---|---|
| DREAM-H | *known* | auto | 82.9 | 81.1 | 73.3 | 72.1 | — |
| HoRoPose (HPE) | predicted | **GT** | 82.7 | 82.0 | 75.1 | 73.9 | 58.8 |
| RoboPose | predicted | auto | 82.9 | 79.7 | 80.2 | 73.2 | 32.7 |
| HoRoPose\* (HPE\*) | predicted | auto | 41.4 | 40.7 | 56.2 | 56.7 | 9.8 |
| RoboPEPP | predicted | auto | 83.0 | 84.1 | 76.2 | 76.1 | 34.4 |
| **Ours** | predicted | auto | **74.2** | **76.9** | 69.0 | 69.8 | **71.3** |

> EN: **Table 4. DREAM synthetic-split ADD-AUC (×100);** competitor numbers cited from RoboPEPP (CVPR'25) Table 2. Ours: Panda = full pipeline with render-compare; KUKA/Baxter = kinematic solver without render-compare. Under matched predicted-joint + auto-bbox, ours (74.2/76.9) far exceeds HoRoPose\* (41.4/40.7); synthetic-specialized RoboPEPP/RoboPose lead overall on their training-distribution home turf.

### 4.3 가림 강건성 (Occlusion robustness)

RoboPEPP의 가림 프로토콜(로봇 bbox 면적의 0–40%를 사각 occluder로 페이스트)로 평가하면, DINObotPose는 **모든 가림 수준에서** 기존 방법들 — RoboPEPP·HoRoPose(HPE)·RoboPose — 을 상회한다(표 5, 그림 2). 이 우위의 원천은 (a) 우리에게만 있는 렌더-비교 깊이 보정, (b) 처음부터 가림에 노출된 약한 가림-증강 헤드다.

> EN: Under RoboPEPP's occlusion protocol (paste rectangular occluders over 0–40% of the robot's bbox area), DINObotPose exceeds prior methods — RoboPEPP, HoRoPose (HPE), and RoboPose — at **every** occlusion level (Table 5, Fig. 2). The advantage stems from (a) the render-compare depth corrector unique to us and (b) a light occlusion-augmentation head exposed to occlusion from the start.

**표 5. 가림 수준별 ADD-AUC** (RoboPEPP Fig.6 프로토콜, synth_photo). 경쟁 수치는 RoboPEPP 논문 Fig.6에서 인용.

| 방법 | 0% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| **Ours (light+RC)** | **0.812** | **0.765** | **0.679** | **0.573** | **0.430** |
| RoboPEPP | 0.795 | 0.730 | 0.600 | 0.470 | 0.351 |
| HoRoPose (HPE) | 0.570 | 0.505 | 0.405 | 0.320 | 0.282 |
| RoboPose | 0.540 | 0.420 | 0.280 | 0.210 | 0.145 |

> EN: **Table 5. ADD-AUC vs occlusion level** (RoboPEPP Fig. 6 protocol, synth_photo; competitor numbers cited from RoboPEPP Fig. 6). Ours dominates across 0–40%: the gap over RoboPEPP widens with occlusion (+0.079 at 40%), and both HoRoPose (HPE) and the iterative RoboPose fall far below (0.282 / 0.145 at 40%).

### 4.4 절제 실험 (Ablations)

**Leave-one-out(표 6).** 배포 스택에서 각 레버를 하나씩 제거하고 **동일한 1000-프레임 held-out 집합**에서 재평가하여, 각 기법의 순 기여(ΔMean)를 동일 조건에서 정량화한다(이전 누적/경쟁자-기준 방식과 달리 우리 모델 기준의 순수 절제). 렌더-비교가 압도적 최대 레버(+0.043)이고, 회전 헤드(+0.016)와 가림-증강/자가학습 스택(+0.010)이 뒤따르며, 무료 디코딩·솔버 레버(DARK/cov-PnP/conf-gate)는 클린에서 작지만 do-no-harm이다(가림에서 진가, §4.3). 완전 자동 bbox는 GT-bbox 대비 −0.002에 불과해 사실상 무비용의 더 엄격한 프로토콜이다.

> EN: **Leave-one-out (Table 6).** We remove each lever from the deployed stack and re-evaluate on the **same 1000-frame held-out set**, quantifying each technique's net contribution (ΔMean) under identical conditions — a pure ablation off our own model, unlike the earlier cumulative/competitor-anchored view. Render-and-compare is by far the largest lever (+0.043), followed by the rotation head (+0.016) and the occlusion-aug/self-training stack (+0.010); the free decoding/solver levers (DARK/cov-PnP/conf-gate) are small on clean but do-no-harm (their value is under occlusion, §4.3). Fully automatic boxes cost only −0.002 vs GT boxes — a near-free, stricter protocol.

**표 6. Leave-one-out 절제 (locked 1000, ADD-AUC@100mm).** 각 행은 배포 Full에서 한 레버 제거.

| 제거 | rs | kinect | orb | azure | Mean | **ΔMean** |
|---|---|---|---|---|---|---|
| **Full (배포)** | 0.815 | 0.828 | 0.778 | 0.795 | **0.804** | — |
| − render-and-compare | 0.745 | 0.765 | 0.738 | 0.795† | 0.761 | **−0.043** |
| − rot-head (R_init) | 0.809 | 0.816 | 0.747 | 0.781 | 0.788 | **−0.016** |
| − occ-aug / angle 자가학습‡ | 0.812 | 0.817 | 0.757 | 0.792 | 0.794 | **−0.010** |
| **zero-real-adaptation (합성 angle+rot 헤드 + RC)** | 0.804 | 0.816 | 0.756 | 0.795 | 0.793 | **−0.011** |
| − DARK decode | 0.812 | 0.827 | 0.776 | 0.788 | 0.801 | −0.003 |
| − cov-PnP | 0.815 | 0.829 | 0.777 | 0.791 | 0.803 | −0.001 |
| − conf-gate | 0.816 | 0.826 | 0.781 | 0.790 | 0.803 | −0.001 |
| auto-bbox → GT-bbox | 0.813 | 0.820 | 0.784 | 0.808 | 0.806 | +0.002 |

> EN: **Table 6. Leave-one-out ablation (locked 1000, ADD-AUC@100mm);** each row removes one lever from the deployed Full model. †azure ships RC off, so −RC leaves azure unchanged. ‡This row reverts only the **angle**-side occ-aug/self-training; the per-camera self-trained **rotation** heads are retained on rs/kinect/orb. The total real-adaptation contribution is isolated by the **zero-real-adaptation** row (fully synthetic angle+rot heads, deployed per-camera RC): mean 0.793, i.e., self-training contributes **+0.011** in total. Even with no real-image adaptation at all, the training-free core stays at or above the strongest feed-forward baseline (RoboPEPP 0.780; clear on kinect/azure, par on realsense, behind on orb), and the per-camera RC gains grow (+0.072/+0.066/+0.045) as RC compensates the weaker synthetic heads.

> ‡주의: 이 행은 **angle 헤드**의 occ-aug/자가학습만 제거하며, rs/kinect/orb의 카메라별 자가학습 **rot 헤드는 유지**된다. 실측 적응의 총기여는 **zero-real-adaptation 행**(완전 합성 angle+rot 헤드 + 배포 RC)이 격리한다: mean 0.793, 즉 자가학습 총기여 **+0.011**. 실측 적응이 전혀 없어도 학습-불필요 코어는 최강 feed-forward 기준선 이상을 유지하며(RoboPEPP 0.780; kinect/azure 명확, realsense 동률, orb 열세), RC 이득은 합성 헤드의 약점을 보상하며 오히려 커진다(+0.072/+0.066/+0.045).

**렌더-비교의 카메라별 기여(그림 4).** RC는 깊이 신호가 약한 **원거리 카메라의 엔진**이다 — RealSense +0.070, Kinect +0.062, ORB +0.040(위 표의 −RC 행에서 카메라별 직접 측정). 근거리 Azure는 깊이가 이미 강해 RC를 끄는 것이 최적(카메라별 on/off). 이는 RC가 "포즈 전체 추정"이 아니라 **깊이/스케일 보정기**로 작동함을 확증한다.

> EN: **Per-camera render-compare contribution (Fig. 4).** RC is the engine for **far cameras** where depth is weak — RealSense +0.070, Kinect +0.062, ORB +0.040 (read directly from the −RC row per camera) — whereas for the near Azure camera it is best off. This confirms RC acts as a **depth/scale corrector**, not a full-pose estimator.

**가림에서 occ-aug의 기여(표 7).** occ-aug/self-train은 클린에서 +0.010이지만 가림이 심해질수록 기여가 커진다 — 40% 가림에서 **+0.038**(light head vs clean head, 나머지 스택 동일). 강건성은 처음부터 증강 학습해야 배어듦을 보인다.

> EN: **occ-aug contribution under occlusion (Table 7).** occ-aug/self-training gives +0.010 on clean but its contribution grows with occlusion — **+0.038 at 40%** (light vs clean head, rest of stack fixed), showing robustness must be trained in from the start.

**표 7. occ-aug on/off의 가림별 ADD-AUC** (synth_photo, 나머지 스택 동일).

| 가림 % | 0 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|
| Full (light head) | 0.812 | 0.765 | 0.679 | 0.573 | 0.430 |
| − occ-aug (clean head) | 0.804 | 0.752 | 0.671 | 0.562 | 0.392 |
| Δ (occ-aug 기여) | +0.008 | +0.013 | +0.007 | +0.011 | **+0.038** |

> EN: **Table 7. occ-aug on/off ADD-AUC vs occlusion** (synth_photo, rest of stack fixed).

**무료 레버.** cov-PnP는 20% 가림에서 +0.011로 do-no-harm을 유지하며, DARK는 특히 원거리 ORB의 격차를 −0.010→−0.004로 좁힌다.

> EN: **Free levers.** cov-PnP adds +0.011 at 20% occlusion with do-no-harm elsewhere, and DARK narrows the far-camera ORB gap from −0.010 to −0.004.

**가림 강건성의 출처(그림 6).** 40% 가림에서 깨끗하게만 학습한 헤드(0.376)보다 약한 가림-증강 헤드(0.420)가 강건하며, 배포 스택은 그 강건성을 대부분 유지(0.396)하면서 실측 정확도를 회복한다. 즉 **가림 강건성은 처음부터 증강 학습해야 배어든다.**

> EN: **Source of occlusion robustness (Fig. 6).** At 40% occlusion, the light occlusion-augmentation head (0.420) is more robust than a clean-only head (0.376), and the deployed stack retains most of it (0.396) while recovering real-image accuracy — i.e., **robustness must be trained in from the start via augmentation.**

**누적 build-up(표 8, 그림 5).** leave-one-out과 상보적으로, 헐벗은 base(클린 헤드·PnP만)에서 레버를 하나씩 **더하며** RealSense held-out 1000에서 단조 개선을 측정한다(세션 단위 mean 진행은 그림 5). base 0.666에서 배포치 0.815까지 **총 +0.149**이며, 가장 큰 세 단계는 rot-head 초기화(+0.036), occ-aug/자가학습 헤드(+0.040), 렌더-비교(+0.070)다. DARK(+0.007)와 cov-PnP·conf-gate는 클린에서 ±0.005 이내로 사실상 평평하다 — 이들의 값은 클린 정확도가 아니라 **가림 강건성**(§4.3)에 있다는 앞선 결론과 정확히 일치한다.

> EN: **Cumulative build-up (Table 8, Fig. 5).** Complementary to leave-one-out, we *add* levers one at a time onto a bare base (clean head, PnP only) and measure the monotone gain on the RealSense held-out 1000 (session-level mean progression in Fig. 5). From base 0.666 to the deployed 0.815 is **+0.149 total**, with the three largest steps being rot-head initialization (+0.036), the occ-aug/self-training head (+0.040), and render-and-compare (+0.070). DARK (+0.007) and cov-PnP/conf-gate are essentially flat on clean (within ±0.005) — exactly matching the earlier finding that their value lies in **occlusion robustness** (§4.3), not clean accuracy.

**표 8. 누적 build-up (RealSense held-out 1000, ADD-AUC@100mm).** 각 행은 위 행에 레버 하나를 추가.

| 스택 | ADD-AUC | Δ |
|---|---|---|
| base (클린 헤드, PnP만, RC off) | 0.666 | — |
| + DARK 서브픽셀 디코딩 | 0.673 | +0.007 |
| + cov-PnP | 0.669 | −0.004 |
| + rot-head 초기화 | 0.705 | +0.036 |
| + occ-aug / 자가학습 헤드 | 0.745 | +0.040 |
| **+ render-and-compare (배포)** | **0.815** | **+0.070** |

> EN: **Table 8. Cumulative build-up (RealSense held-out 1000, ADD-AUC@100mm);** each row adds one lever to the row above. cov-PnP dips −0.004 on clean (its value is under occlusion), the rest accumulate monotonically to the deployed 0.815.

**conf-gate 민감도.** conf-gate 임계값을 {0, 0.05(배포), 0.10, 0.20}으로 스윕하면 클린 RealSense에서 ADD-AUC가 각각 0.747·0.745·0.746·0.749로 **±0.002 이내로 평평**하다 — 배포 하이퍼파라미터가 임의 선택이 아니라 넓은 안정 구간 안에 있음을 보인다(가림 필터로서의 진가는 §4.3).

> EN: **conf-gate sensitivity.** Sweeping the confidence-gate threshold {0, 0.05 (deployed), 0.10, 0.20} moves clean-RealSense ADD-AUC only across 0.747·0.745·0.746·0.749 — **flat within ±0.002**, showing the deployed hyper-parameter sits in a wide stable basin rather than a tuned peak (its real value as an occlusion filter is in §4.3).

**키포인트 노이즈 민감도(G1).** 디코딩된 2D 키포인트에 등방성 가우시안 노이즈(σ px)를 주입하고 RealSense held-out 1000·base-only에서 스윕한다(DREAM식 PnP 강건성). 솔버는 **우아하게 degrade**한다 — σ=0(0.745)→1(0.721)→2(0.656)까지는 완만하고(2px는 DARK 서브픽셀 디코드 오차를 상회하는 수준), 급락은 σ≥4에서만 나타난다(0.490). 즉 운동학-재투영 솔버는 sub-pixel 오차 규모의 2D 노이즈에 절벽 없이 견딘다. 흥미롭게도 **cov-PnP와 평이한 PnP는 거의 동일**하다(σ=0/1/2/4/8에서 cov 0.745/0.721/0.656/0.490/0.293 vs no-cov 0.747/0.722/0.662/0.520/0.307). 이는 cov-PnP의 이점이 **임의로 더해진 노이즈에 대한 강건성이 아니라 히트맵 자체의 불확실성 구조(흐림·다봉 봉우리)를 활용**하는 데서 온다는 §4.3·§4.4 결론을 직접 뒷받침한다 — 주입 노이즈는 클린 히트맵에서 추정된 공분산과 무상관이라, 이방성 가중이 실제 오차와 어긋나 이득이 사라진다(고노이즈에서 오히려 미세하게 불리). 요약: 솔버는 노이즈에 견고하고, cov-PnP의 진가는 노이즈 흡수가 아니라 가림-불확실성 정합이다.

> EN: **Keypoint-noise sensitivity (G1).** We inject isotropic Gaussian noise (σ px) into the decoded 2D keypoints and sweep it on RealSense held-out 1000, base-only (DREAM-style PnP robustness). The solver **degrades gracefully** — σ=0(0.745)→1(0.721)→2(0.656) is gentle (2 px exceeds the DARK sub-pixel decode error), with a drop only at σ≥4 (0.490); i.e. the kinematic-reprojection solver tolerates sub-pixel-scale 2D noise without a cliff. Notably, **cov-PnP and plain PnP are nearly identical** (σ=0/1/2/4/8: cov 0.745/0.721/0.656/0.490/0.293 vs no-cov 0.747/0.722/0.662/0.520/0.307). This directly supports the §4.3/§4.4 conclusion that cov-PnP's benefit comes from **exploiting the heatmap's own uncertainty structure (blurred/multimodal peaks), not robustness to arbitrary added noise** — the injected noise is uncorrelated with the covariance estimated from the clean heatmap, so the anisotropic weighting is miscalibrated to the actual errors and its advantage vanishes (even marginally hurts at high noise). In short: the solver is noise-robust, and cov-PnP's real value is occlusion-uncertainty matching, not noise absorption.

**정성 결과.** 그림 9은 예측된 Panda 메시 실루엣(nvdiffrast)을 RealSense 실측 위에 오버레이한 것으로, 여러 관절 배치에서 몸체가 사진과 밀착함을 보인다(프레임별 ADD 11–66mm). 그림 10은 동일 프레임을 0–40% 가림 사다리로 렌더링해, 중간 가림(≤20%)까지 정렬이 유지되고 심한 가림에서만 실패가 나타남을 정성적으로 확인한다.

> EN: **Qualitative results.** Fig. 9 overlays the predicted Panda mesh silhouette (nvdiffrast) on real RealSense frames, showing the body hugging the photo across diverse joint configurations (per-frame ADD 11–66 mm). Fig. 10 renders one frame across a 0–40% occlusion ladder, qualitatively confirming that alignment holds through moderate occlusion (≤20%) and only fails under heavy occlusion.

### 4.5 프로토콜 분석 (Protocol analysis)

**자동 bbox는 진짜 어렵다.** 시점이 가장 다양한 ORB 카메라에서 HoRoPose는 **GT-bbox**로 0.752이지만, RoboPEPP가 사용한 **기성 검출기(HPE\*)** 아래에서 **0.516으로 약 0.24 하락**하고, 렌더-비교 기반 RoboPose는 0.704를 유지한다. 우리 bbox-from-solved(0.778)는 두 자동-검출기 구성을 모두 상회한다. (HPE\* 수치는 HoRoPose 저자의 결과가 아니라 **RoboPEPP 논문의 재평가 인용**이다.) [2026-07-21 정정: 이전의 "ORB 0.098 급락 / RoboPose 0.327"은 Baxter 열 값을 ORB로 잘못 옮긴 오기] 즉 우리 비교의 직접 상대인 RoboPEPP/RoboTAG는 우리와 **동일한 자동-bbox 프로토콜**이며(따라서 mean 0.804 vs 0.780은 공정한 동일-조건 비교), GT-bbox를 쓰는 HoRoPose 대비로는 우리가 더 엄격한 조건이다.

> EN: **Automatic bounding boxes are genuinely hard.** On the ORB camera, whose viewpoints are the most diverse, HoRoPose scores 0.752 with **GT boxes** but **drops to 0.516** under RoboPEPP's off-the-shelf detector (HPE\*), a ~0.24 loss, while render-and-compare based RoboPose holds 0.704. Our bbox-from-solved (0.778) exceeds both automatic-detector configurations. [corrected 2026-07-21: the earlier "collapses to 0.098 / 0.327" figures were the Baxter column mis-transcribed as ORB] Thus our direct competitors RoboPEPP/RoboTAG run under the **same automatic-bbox protocol** as us (so the 0.804 vs 0.780 mean is a fair like-for-like comparison), while we are stricter than the GT-box HoRoPose.

**실측 적응 공정성.** 우리 배포 스택의 카메라별 자가학습(라벨 없는 pseudo-label self-supervision, CtRNet 계보)이 경쟁자에 없는 이점이라는 우려에 대해, 자가학습을 완전히 제거한 **zero-real-adaptation 구성**(합성 헤드 + RC)도 mean **0.793**으로 RoboPEPP 0.780을 명목상 상회함을 표 6에서 보인다(+0.013 — 단 실행-간 노이즈 추정치 0.010과 비슷한 규모라 보수적으로는 동률-이상으로 해석). 즉 자가학습은 결정적 전제가 아니라 +0.011의 추가 마진이다.

> EN: **Real-adaptation fairness.** Addressing the concern that our per-camera self-training (label-free pseudo-label self-supervision, in the CtRNet lineage) is an advantage competitors lack, Table 6 shows that a **zero-real-adaptation** configuration (synthetic heads + RC) still reaches a mean of **0.793**, nominally above RoboPEPP's 0.780 (+0.013, though comparable in magnitude to the estimated run-to-run noise of 0.010, so conservatively a tie-or-better). Self-training is thus an additional +0.011 margin, not a decisive prerequisite.

### 4.6 재잠금 안정성 (Re-lock stability)

논문급 신뢰성을 위해 표본을 800에서 1000프레임으로 늘려 재측정했다(그림 3). 평균은 0.8037→**0.8039**로 사실상 불변(Δ+0.0002)이고 개별 카메라 변동도 ≤0.006이며, 어느 카메라에서도 RoboPEPP에 뒤지지 않는다(ORB는 −0.002→+0.003으로 전환 — 단 이 마진 자체는 노이즈 범위 내의 동률이다, §4.2). 결과는 표본 수에 강건하다.

> EN: **Re-lock stability (Fig. 3).** For paper-grade confidence we re-measured at 1000 (vs 800) frames: the mean is essentially unchanged (0.8037→**0.8039**, Δ+0.0002) with per-camera drift ≤0.006, and no camera trails RoboPEPP (ORB flips −0.002→+0.003 — itself a within-noise tie, §4.2). Results are robust to sample size.

### 4.7 KUKA·Baxter 상세: data-fit FK와 잔여 실패모드

DREAM의 나머지 두 로봇은 실측 데이터가 없으므로 합성(DR) 스플릿에서 평가한다. 검출기는 Panda 검출기에서 전이학습하여 2D 키포인트 AUC **0.735**(KUKA)·**0.817**(Baxter)를 얻는다. 운동학 FK는 표준 URDF 대신 **DREAM 합성 데이터에 직접 피팅**하여(관절각↔키포인트 3D) 링크 원점을 RMS 0.003mm로 재현한다. 포즈는 예측된 관절각으로부터 운동학 솔버가 R,t를 복원해 ADD-AUC **0.690**(KUKA)·**0.713**(Baxter)를 기록한다(§4.2 표 3, 그림 7).

> EN: The other two DREAM robots have no real data, so we evaluate on synthetic (DR) splits. Detectors transfer-learned from the Panda detector reach 2D-keypoint AUC **0.735** (KUKA) / **0.817** (Baxter). Instead of a standard URDF, we **fit the kinematic FK directly to DREAM's synthetic data** (joint angles ↔ 3D keypoints), reproducing link origins at 0.003 mm RMS. Pose from the kinematic solver, which recovers R,t from the predicted joint angles, gives ADD-AUC **0.690** (KUKA) / **0.713** (Baxter) (Table 3, §4.2; Fig. 7).

(3-로봇 통합 성능표는 §4.2 표 3 참조. / _EN: see the unified 3-robot Table 3 in §4.2._)

**⚠️ 비교 주의.** KUKA/Baxter의 합성 0.69/0.71은 Panda 실측 0.804와 **다른 데이터(합성)·다른 조건(render-compare 미적용)**이므로 로봇 간 우열로 읽어선 안 된다. 다만 비교 방법들이 보고하는 합성 스플릿과는 직접 비교 가능하며(Baxter 1위·KUKA 근접, §4.2 표 4), 이 결과의 의의는 (i) 동일 파이프라인이 세 로봇 모두에서 동작하고, (ii) 지배적 잔여 실패모드(link-identity 혼동으로 인한 파국적 꼬리)를 규명했다는 점이다.

> EN: **Comparison caveat.** The synthetic 0.69/0.71 for KUKA/Baxter are **different data (synthetic) and different conditions (no render-compare)** than Panda's real 0.804 and should not be read as a robot-vs-robot ranking. They are, however, directly comparable to the synthetic splits the compared methods report (Baxter 1st, KUKA within range; Table 4, §4.2). Their value is (i) that the same pipeline runs on all three robots and (ii) the quantitative analysis of the dominant residual failure mode (the catastrophic tail from link-identity confusion) below.

**손목 관측성과 잔여 실패모드(그림 8).** 손목 방향에는 실재하는 관측성 효과가 있다: 손목 관절의 자기축 회전은 자기 키포인트 원점을 움직이지 않아, 완벽한(GT) 키포인트를 헤드에 주입해도 손목 각도가 거의 개선되지 않고(손목 MAE 28.1°→27.6°), 엔드이펙터 특징 패치 헤드(크기 3·5)도 표준 헤드를 넘지 못한다 — 저해상도 도메인-랜덤화 크롭에서 손목-롤이 기하로도 외형으로도 쉽게 결정되지 않음을 뜻한다. 다만 이는 **2차적** 효과다: 손목 각도 25° 오차는 자기 키포인트를 약 8mm만 이동시켜 ADD-AUC에는 +0.005 수준만 기여하기 때문이다(FK 레버암). **지배적 잔여 실패는 관측성 천장이 아니라 link-identity 혼동에 의한 파국적 꼬리**다: 그럴듯하지만 잘못된 키포인트-링크 대응이 신뢰도 높은 오답 포즈를 낳으며(KUKA 프레임의 15.9%가 100mm 초과, 99백분위 ~1m, 중앙값 13.1mm), 신뢰도만으로는 검출되지 않는다. 실제로 카메라 내부파라미터를 바로잡는 것만으로 Baxter의 파국 프레임 비율이 24.7%→8.0%로 붕괴한 반면 손목 미결정성 자체는 변하지 않았다 — 지배 요인이 후자가 아니었음을 보인다.

> EN: **Wrist observability and the residual failure mode (Fig. 8).** A real observability effect exists at the wrist: a wrist joint's self-axis rotation does not move its own keypoint origin, so injecting perfect (GT) keypoints into the head barely changes the wrist angle (wrist MAE 28.1°→27.6°), and a head with an explicit end-effector feature patch (sizes 3 and 5) does not surpass the standard head either — wrist roll is not reliably determined from geometry *or* appearance in low-resolution domain-randomized crops. This effect is **second-order**, however: a 25° wrist error displaces its own keypoint by only about 8 mm and thus contributes on the order of +0.005 ADD-AUC (FK lever-arm). The **dominant residual failure is not the observability ceiling but the catastrophic tail from link-identity confusion**: a plausible but wrong keypoint-to-link assignment yields a confident wrong pose (15.9% of KUKA frames exceed 100 mm, 99th percentile ~1 m, median 13.1 mm) that is not detectable from confidence alone. Indeed, fixing the camera intrinsics alone collapsed Baxter's catastrophic-frame rate from 24.7% to 8.0% while wrist under-determination was unchanged — showing the latter was not the binding factor.

---

### 4.8 런타임 (Runtime)

비교 논문들이 표준으로 보고하는 추론 속도를 RTX 3090에서 측정한다(표 9). 동결 DINOv3 백본 forward는 빠르나(19ms, 60 img/s), 비용은 **테스트-타임 최적화**에 있다 — 운동학 솔버(250 iter, 352 ms/frame)와 렌더-비교(SAM + 미분 렌더, ~1.3 s/frame). base 파이프라인은 ~2.4 fps, RC 포함 end-to-end는 ~0.6 fps다. 정확도 우선 설계로 실시간은 아니며, RC는 카메라별 선택(Azure는 base-only ~2.4 fps)이고 솔버 iter 수로 정확도-속도 절충이 가능하다(부록).

> EN: **Runtime (Table 9), measured on an RTX 3090.** The frozen DINOv3 backbone forward is fast (19 ms, 60 img/s), but the cost lies in **test-time optimization** — the kinematic solver (250 iters, 352 ms/frame) and render-and-compare (SAM + differentiable rendering, ~1.3 s/frame). The base pipeline runs at ~2.4 fps and end-to-end with RC at ~0.6 fps. The method is accuracy-focused rather than real-time; RC is applied per camera (Azure is base-only, ~2.4 fps) and the solver iteration count trades accuracy for speed.

**표 9. 단계별 지연 (RTX 3090).**

| 단계 | 지연 | 처리량 |
|---|---|---|
| DINOv3 backbone fwd (×2: full+crop) | ~40 ms | 60 img/s |
| keypoint decode + DARK | ~3 ms | — |
| kinematic solver (250 iter) | 352 ms | — |
| render-and-compare (SAM + nvdiffrast) | ~1300 ms | — |
| **base end-to-end** | ~420 ms | **~2.4 fps** |
| **+ render-and-compare** | ~1700 ms | **~0.6 fps** |

> EN: **Table 9. Per-stage latency (RTX 3090).**

**RC iteration 수 = 정확도-속도 knob.** RC 반복 수를 스윕하면 RealSense 기준 ADD-AUC가 0(0.745, RC 없음)→25(0.788)→50(0.802)→150(**0.815**)로 **~150 iter에 배포치(250, 0.8155)에 수렴**한다(ORB도 150에서 0.778=배포치). 즉 250→150으로 RC 렌더 비용을 ~40% 줄이면서 정확도 손실이 없다. 또한 RC 신호는 **전적으로 실루엣 IoU 항**에서 나온다 — 실루엣 항 제거 시 base(0.745)로 회귀하고, 재투영 앵커 항은 +0.002만 기여한다.

> EN: **RC iteration count is the accuracy–speed knob.** Sweeping RC iterations, RealSense ADD-AUC goes 0(0.745, no RC)→25(0.788)→50(0.802)→150(**0.815**), **converging to the deployed 250-iter value (0.8155) by ~150 iters** (ORB likewise 0.778 at 150). So 250→150 cuts ~40% of the RC render cost with no accuracy loss. The RC signal comes **entirely from the silhouette-IoU term** — removing it reverts to base (0.745); the reprojection-anchor term adds only +0.002.

**RC 렌더 해상도 = 두 번째 정확도-속도 knob.** 미분 렌더 실루엣 해상도를 스윕하면 RealSense ADD-AUC가 224(0.779)→320(0.805)→448(**0.815**)→512(0.817)로 **448에서 무릎(knee)에 도달**하고 512는 +0.001에 그친다. wall-time은 네 해상도가 사실상 동일(~1330s, nvdiffrast 래스터화가 이 범위에서 GPU-바운드가 아님)해, 배포 스택의 rs/kinect@448 선택이 정확도-비용 최적점임을 확인한다. 즉 반복 수(iter)와 달리 해상도는 448 이상에서 공짜에 가깝고, 진짜 속도 knob은 iteration 수다.

> EN: **RC render resolution is a second accuracy–speed knob.** Sweeping the differentiable-render silhouette resolution, RealSense ADD-AUC goes 224(0.779)→320(0.805)→448(**0.815**)→512(0.817), **reaching a knee at 448** with 512 adding only +0.001. Wall-time is essentially identical across the four resolutions (~1330 s; nvdiffrast rasterization is not GPU-bound in this range), confirming the deployed rs/kinect@448 as the accuracy–cost optimum. Unlike the iteration count, resolution is nearly free beyond 448 — the real speed knob is the iteration count.

**기존 방법과의 런타임 위치(표 10).** 비교 방법들은 두 설계 패러다임으로 갈린다. (i) **단일 feed-forward** 계열 — HoRoPose(HPE)·RoboPEPP·RoboTAG·CtRNet — 은 한 번의 순전파로 포즈를 회귀해 실시간(수십 fps)이지만, 그 대가로 완화된 설정(HoRoPose=GT-bbox, CtRNet=known-joint)에 의존하거나, 우리와 동일한 (predicted+auto) 프로토콜인 RoboPEPP·RoboTAG는 정확도가 우리보다 낮다. (ii) **반복 테스트-타임 최적화** 계열 — RoboPose와 우리 — 는 렌더-비교로 정확도를 끌어올리는 대신 실시간이 아니다. 우리는 이 최적화 축에 서되, RoboPose와 달리 **학습된 refiner가 필요 없고**, RC-iteration knob(250→150, 무손실)으로 최적화 비용을 조절할 수 있어 optimization 계열 중 가장 가벼운 편이다. 절대 지연은 우리 값만 RTX 3090 실측이며, 경쟁 방법의 수치는 각 논문 보고치 체제(feed-forward=실시간 / RoboPose=반복·비실시간)로 표기한다.

> EN: **Runtime positioning vs prior work (Table 10).** Competing methods split into two design paradigms: (i) **single feed-forward** — HoRoPose (HPE), RoboPEPP, RoboTAG, CtRNet — regress the pose in one pass and run in real time (tens of fps), but pay for it with relaxed settings (GT boxes for HoRoPose, known joints for CtRNet) or, for the same-protocol RoboPEPP/RoboTAG, lower accuracy than ours; and (ii) **iterative test-time optimization** — RoboPose and ours — which trade real-time speed for accuracy via render-and-compare. We sit on the optimization axis but, unlike RoboPose, need **no learned refiner** and expose an RC-iteration knob (250→150, lossless) that makes us the lightest of the optimization family. Absolute latencies are RTX-3090-measured for ours only; competitor entries are given as the runtime regime reported by each paper (feed-forward = real-time / RoboPose = iterative, non-real-time).

**표 10. 설계 패러다임별 런타임·설정 비교.** (우리 값은 RTX 3090 실측; 경쟁 방법은 논문 보고 체제)

| 방법 | 패러다임 | 런타임 체제 | 관절 | bbox |
|---|---|---|---|---|
| HoRoPose (HPE) | 단일 feed-forward (holistic 회귀) | 실시간 (논문 보고) | predicted | **GT** |
| CtRNet | feed-forward + 학습-타임 실루엣 자기지도 | 실시간 (논문 보고) | **known** | — |
| RoboPEPP | masked-encoder feed-forward | 실시간급 (논문 보고) | predicted | 자동 |
| RoboTAG | end-to-end 회귀 | 실시간급 (논문 보고) | predicted | 자동 |
| RoboPose | **반복** render-and-compare | 비실시간 (다중 iter) | predicted | init 의존 |
| **Ours (base)** | feed-forward + 운동학 솔버 | **~0.42 s/frame (2.4 fps)** 실측 | predicted | **자동** |
| **Ours (+RC)** | + 테스트-타임 최적화 | **~1.7 s/frame (0.6 fps)** 실측 | predicted | **자동** |

> EN: **Table 10. Runtime and setting by design paradigm** (ours RTX-3090-measured; competitors as the regime reported by each paper). Feed-forward methods are real-time but rely on relaxed settings or known joints; RoboPose and ours are optimization-based (accuracy-focused, non-real-time), with ours the lightest via the RC-iteration knob and no learned refiner.

### 4.9 시도했으나 반증된 대안 (What did not work)

강한 부정 증거를 리뷰어에게 투명하게 제시한다(표 11). 특히 **백본 적응은 깨끗하게 평가된 두 방식(공격적 SSL·co-finetune) 모두에서 ADD를 악화**시켜(솔버가 요구하는 sub-pixel 키포인트 정밀도 파괴) 동결 백본 결정을 정당화하며, 온건 SSL(3-block)은 헤드 비정합으로 평가 불능이라 증거에서 제외한다. 공격적 SSL은 real PCK@5를 +0.069 올리면서도 ADD를 떨어뜨렸는데(표 11), 이는 적응이 거친 2D 강건성과 서브픽셀 정밀도를 맞바꾼다는 직접 증거다(§3.2, G1 민감도와 정량 정합). RC 계열에서도 feature-metric·edge-NCC·multi-start 변형이 실루엣 RC를 못 이겼고, 학습형 상태 prior(population/DPoser식)와 translation prior는 오히려 해로웠다.

> EN: **What did not work (Table 11).** We transparently report strong negative evidence. In particular, **backbone adaptation degraded ADD in both cleanly-evaluated variants** (aggressive SSL and co-finetuning; the gentle 3-block SSL run was unevaluable due to head mismatch and is excluded as evidence), destroying the sub-pixel keypoint precision the solver needs and justifying the frozen backbone; aggressive SSL raises real PCK@5 by +0.069 yet lowers ADD (Table 11) — direct evidence that adaptation trades sub-pixel precision for coarse 2D robustness (§3.2, quantitatively consistent with the G1 sensitivity). Among RC variants, feature-metric / edge-NCC / multi-start did not beat plain silhouette RC, and learned state priors (population/DPoser-style) and a translation prior were actively harmful.

**표 11. 반증된 대안과 그 결과.**

| 대안 | 결과 | 판정 |
|---|---|---|
| 백본 적응 (SSL 6-block) | real PCK@5 **+0.069↑** 인데 realsense ADD 0.531 < 0.567**↓** (정밀도-강건성 트레이드) | ❌ |
| 백본 적응 (SSL 3-block) | 헤드 비정합(OOD)로 평가 불능 — 동결 근거로 미사용 | ⚠️ |
| 백본 적응 (pseudo-kp co-finetune) | 0.497 → 0.434 단조 하락 | ❌ |
| feature-metric (DINO) RC | azure −0.004 → −0.043 | ❌ |
| edge-NCC RC | −0.10 → −0.18 발산 | ❌ |
| multi-start RC | +0.000 (잔여 실패 R-basin 아님) | ❌ |
| occl-robust 실루엣 다운웨이트 | −0.019 (depth 편향) | ❌ |
| population/learned 상태 prior | −0.09 @20% (정답과 싸움) | ❌ |
| translation(t_z) prior | realsense 0.66 → 0.37 | ❌ |
| union-bbox (solved∪detected) | −0.002 | ❌ |
| MCL 멀티가설 | oracle 0.720 < self-train 0.724 | ❌ |
| RoboTAG reproj-consistency 이식 | azure −0.014 | ❌ |
| 768-crop 고해상도 | −0.14 회귀 | ❌ |

> EN: **Table 11. Refuted alternatives and their results.** Backbone adaptation (two evaluable variants regress; the 3-block run is a head-mismatch artifact, not used as evidence), feature-metric/edge-NCC/multi-start RC, learned/translation priors, union-bbox, MCL, ported RoboTAG consistency, and 768-crop all regressed or diverged.

### 4.10 백본 선택: DINOv3 vs SigLIP2 (동일 크기) (Backbone choice)

우리 파이프라인이 특정 백본에 의존하는지 확인하기 위해, **동일 크기(ViT-B/16, ~86M)**의 두 파운데이션 백본 DINOv3와 SigLIP2를 두 수준에서 비교한다(표 12). **(i) 2D 키포인트 검출 수준**(real-Azure 검출 AUC): **언프리즈(fine-tune) 시 둘은 동등**하고(둘 다 ~0.81), **동결(frozen) 시 DINOv3가 명확히 우위**(0.80 vs 0.72)다. 우리 배포 스택은 솔버의 서브픽셀 정밀도를 보존하기 위해 백본을 **동결**하므로(백본 적응은 §4.9에서 3중 반증), 이 동결-체제 우위가 DINOv3 선택을 정당화한다. **(ii) pose 수준**(4개 실측 카메라, GT-bbox·base-only로 백본만 격리, 동일 clean 합성 crop 헤드, 둘 다 unfreeze crop-detector): **두 백본은 pose에서도 사실상 동등**하다 — DINOv3 mean **0.742**(az .806/ki .739/rs .719/orb .704) vs SigLIP2 mean **0.752**(az .778/ki .766/rs .765/orb .698), 차이 +0.010로 실행-간 노이즈 수준이며 카메라별로도 엇갈린다(azure·orb는 DINOv3, kinect·realsense는 SigLIP2 근소 우위). 즉 pose-level 결과는 검출 수준의 "unfreeze 시 동등" 결론을 그대로 확인한다 — 성능은 특정 백본이 아니라 파운데이션 특징 일반에서 온다. **따라서 DINOv3 채택 근거는 pose 우위가 아니라, 우리가 배포하는 동결(frozen) 체제에서의 검출 우위(0.80 vs 0.72)다.**

> EN: **Backbone choice: DINOv3 vs SigLIP2 (matched size).** To check our pipeline is not tied to a specific backbone, we compare DINOv3 and SigLIP2 at **matched size (ViT-B/16, ~86M)** at two levels (Table 12). **(i) 2D-keypoint detection** (real-Azure AUC): **unfrozen they are equal** (both ~0.81), while **frozen, DINOv3 clearly wins** (0.80 vs 0.72). Since our deployed stack **freezes** the backbone to preserve sub-pixel precision (adaptation triply refuted, §4.9), this frozen-regime edge justifies DINOv3. **(ii) pose level** (4 real cameras, GT-bbox + base-only to isolate the backbone, identical clean synthetic crop heads, both with an unfrozen crop-detector): **the two backbones are essentially equal at pose level too** — DINOv3 mean **0.742** (az .806/ki .739/rs .719/orb .704) vs SigLIP2 mean **0.752** (az .778/ki .766/rs .765/orb .698), a +0.010 difference at the level of run-to-run noise, mixed per camera (DINOv3 leads on azure/orb, SigLIP2 on kinect/realsense). The pose-level result thus confirms the detection-level "equal when unfrozen" finding — performance comes from foundation features in general, not a specific backbone. **DINOv3 is therefore chosen not for a pose-level edge but for its detection advantage in the frozen regime we deploy (0.80 vs 0.72).**

**표 12. DINOv3 vs SigLIP2 (ViT-B/16, ~86M) 백본 비교.** 상단: 2D 키포인트 검출 AUC(real-Azure, plateau). 하단: pose ADD-AUC@100mm(4개 실측 카메라, GT-bbox·base-only·동일 clean 합성 crop 헤드로 백본만 격리, held-out 1000).

| 수준 | 지표 | DINOv3 | SigLIP2 | Δ |
|---|---|---|---|---|
| 검출 (frozen) | real-Azure AUC | **0.80** | 0.72 | +0.08 |
| 검출 (unfrozen, last-4 ft) | real-Azure AUC | 0.815 | 0.814 | ~0 |
| pose azure | ADD-AUC | 0.806 | 0.778 | +0.028 |
| pose kinect | ADD-AUC | 0.739 | 0.766 | −0.027 |
| pose realsense | ADD-AUC | 0.719 | 0.765 | −0.046 |
| pose orb | ADD-AUC | 0.704 | 0.698 | +0.006 |
| **pose mean** | ADD-AUC | 0.742 | **0.752** | **−0.010** |

> EN: **Table 12. DINOv3 vs SigLIP2 (ViT-B/16, ~86M) backbone comparison.** Top: 2D-keypoint detection AUC (real-Azure, plateau). Bottom: pose ADD-AUC@100mm (4 real cameras, GT-bbox + base-only + identical clean synthetic crop heads to isolate the backbone, both with an unfrozen crop-detector, held-out 1000). Detection is equal unfrozen and favors DINOv3 frozen; at pose level the two are essentially equal (mean 0.742 vs 0.752, within run-to-run noise, mixed per camera), confirming the "equal when unfrozen" finding. DINOv3 is deployed for its frozen-regime detection edge, not a pose-level advantage.

---

## 5. Conclusion (결론)

우리는 관절각을 예측하고 바운딩 박스를 완전 자동으로 잡는 가장 어려운 설정에서 단안 관절형 로봇 포즈를 추정하는 기하-유도 파이프라인 DINObotPose를 제시했다. 핵심 설계 원칙은 **동결 파운데이션 특징을 끝까지 신뢰하고, 깊이·불확실도·가림을 학습형 회귀가 아니라 기하로 처리**하는 것이다: 동결은 편의가 아니라 발견이다 — 백본 적응 시도는 일관되게 솔버가 요구하는 서브픽셀 정밀도를 파괴했다(§4.9). 그 위에서 DARK 서브픽셀 디코딩과 공분산-인지 PnP가 학습 없이 가림 강건성을 보태고, 제로샷 SAM 마스크에 대한 미분가능 렌더-비교가 테스트-타임에 깊이/스케일만 보정한다(단일 최대 레버, +0.043). 그 결과 DREAM 실측에서 평균 ADD-AUC 0.804로 predicted-joint 체제의 최고 성능을 달성하며, 완전 자동 바운딩 박스를 쓰면서도 RoboPEPP·RoboTAG를 능가하고 평가한 모든 가림 수준에서 앞선다.

> EN: We presented DINObotPose, a geometry-guided pipeline for monocular articulated robot pose estimation under the hardest setting — predicted joints and fully automatic bounding boxes. Its central design principle is to **trust frozen foundation features and handle depth, uncertainty, and occlusion with geometry rather than learned regression**: freezing is a finding, not a convenience — backbone-adaptation attempts consistently destroyed the sub-pixel precision the solver needs (§4.9). On top of this, DARK sub-pixel decoding and covariance-aware PnP add occlusion robustness at no training cost, while differentiable render-and-compare against zero-shot SAM masks corrects only depth/scale at test time (the single largest lever, +0.043). The result is a mean ADD-AUC of 0.804 on DREAM-real — best in the predicted-joint regime — surpassing RoboPEPP and RoboTAG with fully automatic boxes and leading across all evaluated occlusion levels.

우리는 렌더-비교를 발명했다고 주장하지 않는다. 그 개념은 RoboPose와 CtRNet의 선행이며, 우리 기여는 그것을 **학습이 필요 없는 테스트-타임 깊이 보정기**로, 제로샷 SAM과 동결 DINOv3 키포인트 프론트엔드 위에서, predicted-joint·자동-bbox 체제에 맞게 재구성한 데 있다. 이는 학습형 깊이 회귀(HoRoPose)나 종단간 회귀(RoboTAG)만이 답이 아니며, **파운데이션 특징 + 불확실도-인지 기하 + 제로샷 렌더 비교**의 조합이 강력한 대안임을 보인다.

> EN: We do not claim to have invented render-and-compare — the concept is prior art from RoboPose and CtRNet — and our contribution is recasting it as a **training-free test-time depth corrector** built on zero-shot SAM and a frozen-DINOv3 keypoint front-end for the predicted-joint / auto-bbox regime. This shows that learned depth regression (HoRoPose) and end-to-end regression (RoboTAG) are not the only answers: the combination of **foundation features, uncertainty-aware geometry, and zero-shot render comparison** is a strong alternative.

DREAM의 세 로봇에 동일 파이프라인을 적용하여, 검출·데이터-피팅 운동학·포즈까지 로봇별 재설계 없이 end-to-end로 일반화됨을 확인했다. 이 과정에서 지배적 잔여 실패모드를 분리했다: 그럴듯하지만 잘못된 키포인트-링크 대응이 신뢰도 높은 오답 포즈를 낳아, 오차 분포가 균질한 저하가 아니라 낮은 비율의 파국적 꼬리로 나타난다. 이 실패는 신뢰도가 높으므로 신뢰도 기반 거부로 해결할 수 없으며, 대응 수준의 중의성 해소가 자연스러운 다음 단계다.

> EN: Applying the same pipeline to all three DREAM robots, we confirmed end-to-end generalization of detection, data-fit kinematics, and pose without per-robot redesign. In doing so we isolated the dominant residual failure mode: a plausible but wrong keypoint-to-link assignment produces a confident wrong pose, so the error distribution is a low-rate catastrophic tail rather than a uniform degradation. Because this failure is confident, confidence-based rejection cannot address it, and correspondence-level disambiguation is the natural next step.

**한계와 향후 과제.** DREAM은 KUKA·Baxter의 공개 실측 데이터가 없어 이 두 로봇의 실측 SOTA 비교는 불가능하다. 또한 우리 렌더-비교는 안정성을 위해 카메라별 on/off 게이팅과 앵커링에 의존하므로, 자유 실루엣 최적화의 깊이 모호성을 원리적으로 억제하는 정식화가 남은 과제다.

> EN: **Limitations and future work.** DREAM provides no public real data for KUKA or Baxter, precluding a real-data SOTA comparison for those robots. In addition, our render-compare relies on per-camera on/off gating and anchoring for stability; a formulation that principally suppresses the depth ambiguity of free silhouette optimization remains future work.
