# 4_perception — 세그멘테이션·뎁스·통합 인식

## DINOv3_fine_tunning

목적: DINOv3(frozen) 백본을 로봇포즈/깊이/사람포즈에 파인튜닝하고, 셋을 묶은 통합 추론 파이프라인 제공.

- 환경: conda env `dinov3` (스크립트가 `$HOME/.conda/envs/dinov3/bin/python3` 직접 호출). ultralytics(YOLO-pose), mediapipe, Depth-Anything-3 API, wandb.
- 외부 클론(gitignore, 자체 .git 보유): `dinov3/`(Facebook 공식 — 백본 소스), `Depth-Anything-3/`(ByteDance 공식 — 깊이 모델)

### 학습 스크립트

| 스크립트 | 진입점 | 내용 |
|---|---|---|
| `TRAIN_RUN.sh` | `Single_view_3D_Loss.py --ablation_mode dino_only` | 로봇 포즈 학습 (5 GPU) |
| `train_simple.sh` | `Single_view_3D_Loss_simple.py` | 단순화판 (occlusion aug 등 제거, length loss만). EPOCHS 300, BS 32 → `checkpoints_simple_{mode}/` |
| `train_depth.sh` | `train_depth.py` | DINOv3 깊이 헤드. dinov3-vitb16, DEPTH_SIZE 280×504, EPOCHS 100 → `checkpoints_depth_*/`, wandb `DINOv3_Depth_Estimation` |
| `train_human_pose.sh` | `train_human_pose.py` | COCO 17kp 사람 포즈 히트맵 (frozen 백본) → `checkpoints_human_pose/` — **`coco_dataset/` 로컬 부재, 준비 필요** |

학습 데이터: `datasets/ICRA_multiview/Converted_dataset/**/*.json` (로봇 포즈).

### 통합 파이프라인

- `integrated_pipeline.py` + `run_integrated_pipeline.sh`: Robot Pose(DINOv3) + Depth(DA3 `DA3NESTED-GIANT-LARGE`) + Human Pose(YOLO-pose `yolo11l-pose.pt`, 옵션 MediaPipe hands)
  - 멀티GPU 배치: `ROBOT_GPU=0 / DEPTH_GPU=1 / HUMAN_GPU=2`, `--use_multi_gpu` (3장 ~0.2s, 1장 ~0.36s)
  - `robot_class`: research3 / Fr5 / MecaInsertion / Meca500 / panda
- 단일 이미지: `./run_single_image.sh [IMAGE_PATH] [OUTPUT_PATH]` (기본 ROBOT_CLASS=FR5, 체크포인트 `checkpoints_simple_dino_only_100e/latest_checkpoint.pth`)
- 깊이 추론: `run_depth_inference.sh` → `infer_depth.py --checkpoint ... --depth_root ... --source_root datasets/ICRA_multiview`
- 속도 벤치마크: `benchmark_depth_speed.py`, `test_mediapipe_speed.py`, `Test_speed.ipynb`

### 문서-코드 불일치 주의

- `PIPELINE_SETUP.md`는 사람 포즈를 RTMPose(MMPose)로 설명하지만 **실제 코드는 YOLO-pose + MediaPipe** (문서가 이전 설계). `configs/`는 비어 있음(RTMPose config 미다운로드).
- `TRAINING_SCRIPTS_README.md`는 dinov2-base·EPOCHS 50·3GPU 예시지만 실제 `train_depth.sh`는 dinov3-vitb16·EPOCHS 100·5GPU. 문서의 `train_depth_background.sh`/`train_depth_large.sh`는 존재하지 않음.

---

## Fr5_robot_SegFormer

목적: SegFormer(MiT-b2)로 FR5 로봇팔 세그멘테이션. `5_apps/collision_risk_pipeline`이 이 체크포인트를 사용.

- `robot_segmentation.ipynb` 하나에 학습·평가·추론이 모두 포함:
  - 학습: `SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-b2')` → best val에서 `best_segformer_robot_arm.pth` 저장 (동봉된 329MB 체크포인트가 이 산출물)
  - 평가: Mean IoU / Dice / Pixel Accuracy
  - 추론: `visualize_from_saved_model(image_path, model_path, image_size=512)`
- 테스트 이미지: `fr5.jpeg`, `fr5_2.jpeg`
- **주의**: 학습 데이터 경로가 다른 서버(`/home/ibom002/dataset`) 하드코딩 — 재학습하려면 데이터 확보 + 경로 수정 필요. 체크포인트는 삭제하지 말 것(재생성 데이터가 로컬에 없음).
