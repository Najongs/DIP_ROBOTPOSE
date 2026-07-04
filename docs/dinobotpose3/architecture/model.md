# 모델 구조 (Architecture)

> DINObotPose3: DREAM Panda 단안 6D 포즈 + 관절각 추정. 코드는 `3_pose_models/DINObotPose3/{TRAIN,Eval}`.

## 전체 파이프라인

```
이미지 (512×512)
  │
  ▼ DINOv3 ViT-B/16 백본 (frozen)                    TRAIN/model_v4.py::DINOv3Backbone
  │   → 패치 토큰 (B, 1024, 768)  [32×32 grid @ patch16]
  │
  ├─▶ ViTKeypointHead → per-joint 히트맵 (B,7,H,W)   TRAIN/model.py::ViTKeypointHead
  │     → [DARK sub-pixel 디코드] → 2D 키포인트 + conf   Eval/decode_util.py::dark_decode
  │
  ├─▶ AngleHead (MLP) → 관절각 θ (6개, sin/cos)       TRAIN/model_angle.py::AngleHead
  │
  └─▶ RotationHead → 카메라 회전 R_init (6D)           TRAIN/model_angle.py::RotationHead
        │
        ▼ 운동학 솔버 (PnP init + 재투영 gradient refine)   Eval/solve_pose_kinematic.py::solve_batch
        │   · conf-gate (가림 키포인트 하드컷)
        │   · cov-PnP (히트맵 이방성 공분산 Mahalanobis 가중)
        │   · 미분가능 Panda FK (7 관절 + hand)
        │
        ▼ [render-and-compare] nvdiffrast 정밀 mesh 실루엣 vs SAM 마스크   Eval/rc_refine_from_dump.py
        │   (원거리 카메라만; azure는 off)
        │
        ▼ 출력: 관절각 θ (7) + 카메라 포즈 (R, t)
```

## 구성 요소

### 백본 — DINOv3 ViT-B/16 (frozen)
- `facebook/dinov3-vitb16-pretrain-lvd1689m`, HuggingFace AutoModel
- **완전 동결이 최적**으로 판명 — 적응 계열(SSL, co-finetune, V-JEPA) 3회 반증. 솔버가 요구하는 sub-pixel 키포인트 정밀도를 백본 적응이 파괴함. (V-JEPA 2.1 논문이 "masked-latent 특징은 국소 기하 부정확"으로 독립 확인)
- register 토큰 제거 후 패치 토큰만 사용. SigLIP2도 drop-in 지원(ablation, 동률).

### 2D 키포인트 — ViTKeypointHead + DARK 디코드
- 7개 키포인트: `link0, link2, link3, link4, link6, link7, hand` (Panda FK 인덱스 `[0,2,3,4,6,7,9]`)
- 히트맵 디코드: 기본 windowed soft-argmax(distractor 2nd-mode 강건), **+DARK**(`decode_util.py`) sub-pixel Taylor 보정 — 작은/먼 로봇 정밀도.
- conf = 히트맵 per-joint max.

### 관절각 — AngleHead (MLP)
- 입력: DINO global feat + 키포인트 특징 + K-normalized bearings + conf
- 출력: 6 관절각(sin/cos), J7=0 고정. `mlp_patch`/`transformer`/`mcl` 변형 존재(mlp가 승).
- 학습 손실: sin/cos SmoothL1 + robot-frame FK MSE (+선택 reproj consistency, 실험 중)

### 회전 — RotationHead
- DINO 외관 특징 → 6D 회전 → 솔버 `R_init`. 원거리 카메라의 rotation-basin 모호성 탈출(+0.117 realsense).

### 운동학 솔버 — solve_pose_kinematic.py
- `pnp_init`: 상위 conf 4개 키포인트 EPnP + 폴백/nan 가드 (`pnp_drop=3`)
- 재투영 gradient refine: (θ, R, t) 최적화, IRLS robust reweight, joint-limit sigmoid reparam
- `conf_gate=0.05`: 가림/off-frame 키포인트 하드 제외
- `cov_inv`: 히트맵 2차 모멘트 이방성 공분산 → Mahalanobis 가중 (채택, `heatmap_cov_inv`)

### render-and-compare — nvdiffrast + SAM
- `render_nvdr.py::NVDRSilhouette`: visual mesh(그리퍼 핑거 포함) CUDA 래스터 실루엣/depth/shaded, 미분가능(dr.antialias 경계 그래디언트)
- `rc_refine_from_dump.py`: 배포 포즈에서 시작, SAM ViT-B 마스크(init-render IoU 일관성 선택) vs 렌더 soft-IoU + 재투영 앵커로 refine. **depth/scale 보정기** — 원거리/단축 카메라만 이득, 근거리(azure)는 off.

## 관련
- 데이터/키포인트 정의: [../data/dataset.md](../data/dataset.md)
- 학습 절차: [../training/training.md](../training/training.md)
- 평가/솔버 상세: [../evaluation/evaluation.md](../evaluation/evaluation.md)
- 채택/반증 결정 근거: [../00_overview.md](../00_overview.md)
