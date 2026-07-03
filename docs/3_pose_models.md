# 3_pose_models — 자세추정 모델 학습

## DINObotPose3 (계열 최신·주력)

목적: DINOv3-ViT-B/16 백본 기반 Franka Panda 단안 6D 포즈 — 2D 키포인트 검출 → 관절각/회전 예측 → 운동학 솔버 정제 → nvdiffrast+SAM render-and-compare. **2026-07-03 DREAM 4-real-split SOTA 달성: MEAN ADD-AUC 0.796 vs RoboPEPP 0.780** (완전 자동 bbox 프로토콜), 가림 벤치 20-30% 구간도 승. 진행/로드맵: [robot_pose_next_directions.md](robot_pose_next_directions.md).

**먼저 읽을 것**: `EXPERIMENTS.md`(실험 일지 전체), `SUMMARY.md`(확정 결론 + WORKED/REFUTED 목록 + HOW TO RESUME). 새 실험 전 SUMMARY의 REFUTED 목록을 확인해 중복 실험을 피할 것.

### 환경

- conda env `py312` (torch 2.10, transformers 5.2, albumentations 2.0.8; `requirements.txt`)
- `HF_HOME=/data/public/97_cache` (GPU 서버 캐시 — 로컬 실행 시 조정), SAM 가중치 별도
- wandb 광범위 사용 (`--use-wandb`, 프로젝트: dinov3-pose-estimation / -heatmap-only / -angle-predictor / -3d-pose / -diffusion-angle 등)
- torchrun 멀티GPU (3~5장). **주의: 스크립트 경로가 GPU 서버(`/data/public/NAS/...`)와 NAS(`/home/najo/NAS/DIP/...`) 두 세대로 혼재** — 실행 전 상단 경로 확인.

### 학습 (TRAIN/) — 스테이지 구조

| 스테이지 | 스크립트 | 내용 |
|---|---|---|
| Stage 1: 2D 검출기 | `run_train_heatmap.sh`, `run_train_stage1(_unfrozen).sh`, `run_train_detector_crop/_finetune.sh` | DINOv3 히트맵 키포인트 검출기 (+FDA sim2real). unfreeze 0/4 블록 변형 |
| 백본 ablation | `run_train_siglip(_frozen).sh` | SigLIP2 백본 비교 |
| Stage 1.5: 각도 헤드 | `run_train_angle(.sh, _crop/_jitter/_patch/_tf)` | frozen 검출기 위 관절각 MLP, FK loss |
| 회전 헤드 | `run_train_rotation.sh` | 카메라 회전 prior |
| 3D 통합 | `run_train.sh`(마스터), `run_train_3d(_v2/_v3/_v4).sh` | 각도+FK+히트맵 결합. v4=reproj loss 추가 |
| E2E | `run_train_e2e.sh` | heatmap+angle+camera3D 동시 (LR 1e-5). `CHECKPOINT`의 `XXXXXXXX` 플레이스홀더 채워야 함 |
| Diffusion | `run_train_diffusion(_3).sh` | diffusion 관절각 예측 |
| Stage 3: 자가학습 | `selftrain_pseudo(_rot).py`, `selftrain_detector.py`, `selftrain_mask.py` | 솔버 pseudo-label self-train (배포에 실제 사용됨) |

대표 설정(`run_train.sh`): `facebook/dinov3-vitb16-pretrain-lvd1689m`, IMAGE 512, BATCH 16, LR 1e-4, EPOCHS 30, UNFREEZE 2.
학습 데이터: `datasets/ICRA_multiview/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr`(합성) / 검증 `DREAM_to_DREAM/panda-3cam_azure`(실사).
출력: `TRAIN/outputs_*/.../best_*.pth` (현재 로컬 산출물은 정리로 삭제됨 — 체크포인트는 GPU 서버에 있음).

### 평가 (Eval/)

- **`ab_eval.sh <angle_head.pth> <label> [--crop] [max_frames]`** — 표준 6-split ADD-AUC A/B (realsense/azure/kinect/orb/synth). 실사엔 rot-head, 합성엔 no-rot. 내부적으로 `refine_eval.py` 호출.
- `pck_eval.py --detector --mlp-head --val-dir` — 검출기 PCK를 640×480 기준으로 RoboPEPP/DREAM와 직접 비교
- `mcl_eval.py` — 멀티가설(MCL) 평가: SELECTED/ORACLE/1-HYP 리포트
- `selfbbox_eval.py` — self-bbox crop 파이프라인 (누수 방지 held-out 옵션)
- 진단 probe 다수: `occlusion_diag.py`, `realsense_failure_diag.py`, `depth_ceiling_probe.py`, `silhouette_mesh_probe.py`(render-and-compare), `rc_viz.py`

### 배포 파이프라인 (SUMMARY.md 확정)

full-frame 검출기 + 운동학 솔버 → FK 7kp로 bbox → crop → crop 검출기 → crop 각도/회전 헤드 → 솔버(PnP init + reproj refine). 백본은 frozen 유지, 실사 적응은 헤드 pseudo-label로만.

### Collision/ 데모 (2D/3D 충돌 안전)

- 합성 2D: `conda run -n py312 python Collision/demo.py --out Collision/demo_out --frames 60`
- 실 시퀀스 2D: `Collision/run_video.py --frames-dir <seq> --robot-kps solved_kps.npz --d-safe 45`
- 3D self-test: `Collision/depth3d.py` / 실 시퀀스 3D: `Collision/run_video3d.py --robot-npz solved3d.npz --d-safe 0.15`
- 로봇 = FK 7kp capsule 래스터화, 사람 = Mask R-CNN. 3D는 Depth-Anything 가중치 필요(오프라인 시 `--depth-dir`).

---

## 2025_ICRA_Multi_View_Robot_Pose_Estimation

목적: DINOv3/SigLIP 백본 단일뷰→멀티뷰 로봇 포즈 (FR3/FR5/Meca500 다로봇), DREAM 포맷 통일 학습. DINObotPose 계열의 전신.

- 환경: torch 2.5.1, transformers 4.45.2, kornia, wandb (`requirements.txt`). torchrun DDP.
- **학습 진입점**: `Train/launch.sh` — ablation 순차 런처. 현재 활성: `torchrun --nproc_per_node=4 9th_Single_view_3D_Loss.py --ablation_mode siglip2_only`
  - 스크립트 계보: `1st`(기본) → `2nd`(heatmap) → `3nd`(CNN/DINO ablation) → `6th`(SigLIP) → `7th`(joint angle) → `8th`(heatmap+joint) → `9th/10th`(3D loss)
  - ablation_mode: dino_only / cnn_only / combined / dino_conv_only / siglip_only / siglip2_only / ..._joint 조합
- `Train/Total_ver1/main.py`: 통합 DREAM-robot 학습 (`torchrun --nproc_per_node=3 main.py --robot fr5`)
- `Train/FR5/`, `Train/franka_research3/`: 로봇별 멀티뷰 학습 모듈 + 학습된 `*_best_multiview_model_ddp.pth`
- `DIP_REAL.py`: ZED 실시간 추론 (pyzed)
- **데이터**: `dataset/` → `datasets/ICRA_multiview` 심볼릭링크 (**제거 금지** — 코드가 `__file__` 기준 `<프로젝트>/dataset` 계산)
  - 변환 스크립트: `Converted_dataset/DREAM_to_DREAM_syn.py`(DREAM 합성 라벨 재생성), `FR3_to_DREAM_Fix.py`(경로 픽스), 로봇별 `*_to_DREAM.ipynb`
  - 동기화: `sync/{DREAM,Fr5,franka_research3,Meca500,Meca_insertion}_sync.py`
- 시각화: `visualization/{Fr5,Franka_research3,Meca500,Meca_insertion}_vis.ipynb`
- 출력: `checkpoints_total_{mode}/`, `Train/**/results_ddp/`, wandb 프로젝트 `DINOv3_Ablation_total_{mode}`

---

## Meca500_3D_Pose_Estimation

목적: Meca500 엔코더+ArUco로 3D GT 생성 → YOLO 검출 + ViT 회귀로 단안 3D 키포인트(6kp/7kp) 추정.

노트북 워크플로우 (순서대로):
1. `Encoder_2_Camera.ipynb` — 엔코더 txt를 10ms 보간 후 카메라 타임스탬프와 정렬
2. `250514_Data_preprocessing.ipynb` — ArUco로 카메라→베이스 T 산출 + DH FK로 관절 3D GT 계산·투영 시각화
3. `250514_Yolo_robot_box_model.ipynb` — 로봇 bbox YOLO 학습 (JSON→YOLO 라벨 변환 포함)
4. `250514_3D_pose_estimation.ipynb` — 최종 3D 포즈 추정/시각화 (입력: `synchronized_robot_camera_data_*.csv`, `aruco_final_summary.json`)
5. `250514_3D_pose_estimation_model.py` — timm ViT + DDP + wandb 학습 모듈 (x/y/z별 MSE)

출력: `model_save/{6kp_model,7kp_model}/epoch_*.pth` + 학습곡선/시각화 PNG.
주의: requirements 없음(ultralytics, timm, wandb 필요), 노트북들이 로컬 CSV·절대경로 의존.
