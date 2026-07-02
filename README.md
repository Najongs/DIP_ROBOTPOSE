# DIP — Robot Pose Estimation Workspace

로봇 포즈 추정 연구 프로젝트 모음 워크스페이스.
각 프로젝트는 **독립 GitHub repo**(`Najongs/<이름>`)로 관리되며, 이 최상위 repo는
로컬 인덱스(README, 공용 노트북, collision_risk_pipeline)만 추적한다. **원격 없음** — push 금지.

> 2026-07-03 정리: 학습 산출물(wandb/outputs ~13GB) 삭제, DINObotPose v1/v2 로컬 제거(원격 보존), 최상위 remote 오설정 제거.

## 프로젝트 목록

| 디렉터리 | 설명 | 상태 |
|---|---|---|
| [collision_risk_pipeline/](collision_risk_pipeline/) | 로봇-사람 충돌 위험 파이프라인 (SegFormer 로봇 마스크 + YOLO 사람 마스크 → 최소거리 → 위험 점수). **이 repo에서 직접 추적** | **활성** |
| [Fr5_robot_SegFormer/](https://github.com/Najongs/Fr5_robot_SegFormer) | FR5 로봇팔 SegFormer 세그멘테이션. `best_segformer_robot_arm.pth`(328M) 보유 | **활성** |
| [DINObotPose3/](https://github.com/Najongs/DINObotPose3) | DINOv3 백본 로봇 포즈 추정 v3 — 3D 확장 (PnP + FK + diffusion). DINObotPose 계열 최신 주력 | 유휴 |
| [DINOv3_fine_tunning/](https://github.com/Najongs/DINOv3_fine_tunning) | DINOv3 파인튜닝 + Depth-Anything-3 + 사람/로봇 통합 파이프라인. submodule 2개(dinov3, Depth-Anything-3). 브랜치 `master` | 유휴 |
| [2025_ICRA_Multi_View_Robot_Pose_Estimation/](https://github.com/Najongs/2025_ICRA_Multi_View_Robot_Pose_Estimation) | 멀티뷰 카메라 로봇 포즈 추정 (ICRA). `dataset/` 36GB 원본 데이터 보유 | 유휴 |
| [Meca500_3D_Pose_Estimation/](https://github.com/Najongs/Meca500_3D_Pose_Estimation) | Meca500 로봇 3D 포즈 추정. 데이터셋 zip ~3.3GB 보유 | 유휴 |
| [Panda_cap_make_dataset/](https://github.com/Najongs/Panda_cap_make_dataset) | Franka Panda(research3) ArUco 캡처/전처리 | 유휴 |
| [ZED_Cap_make_dataset/](https://github.com/Najongs/ZED_Cap_make_dataset) | ZED 스테레오 카메라 데이터셋 생성 (RGB + 엔코더) | 유휴 |
| [Intertek_Zed_ArUco_Calibration/](https://github.com/Najongs/Intertek_Zed_ArUco_Calibration) | ZED + ArUco 마커 캘리브레이션 | 유휴 |
| [Robot_joint_inference/](https://github.com/Najongs/Robot_joint_inference) | 관절각→좌표 FK 유틸, FR5/Meca500 DH 파라미터 문서 | 참조용 |
| datasets/ | 공용 학습 데이터셋 허브 (FR5, meca500, meca_insertion, intertek, ZED 등, ~7GB). git 미추적 | 데이터 |
| DGIST_IROM_Data_collection/ | DGIST IROM 데이터 수집 (git 미추적) | 데이터 |

### 원격에만 존재 (로컬 삭제됨, 2026-07-03)

- [DINObotPose](https://github.com/Najongs/DINObotPose) — v1: 2D 다중로봇 (SigLIP + LoRA). v3로 대체
- [DINObotPose2](https://github.com/Najongs/DINObotPose2) — v2: heatmap + FDA 도메인 적응. v3로 대체. GPU 서버 작업(`/data/public/NAS/DINObotPose2`)과 merge 완료

필요 시 `git clone https://github.com/Najongs/<이름>.git`으로 복원.

## 프로젝트 간 의존 관계

- `collision_risk_pipeline` → `Fr5_robot_SegFormer/best_segformer_robot_arm.pth` 체크포인트 사용, 최상위 `yolov8n-seg.pt` 사용
- DINObotPose 계열·DINOv3_fine_tunning의 스크립트 다수가 `2025_ICRA_.../dataset/Converted_dataset/` 경로를 하드코딩
- 노트북들이 `datasets/` 절대경로(`/home/najo/NAS/DIP/datasets/...`)를 하드코딩 — 폴더 이동 시 깨짐
- `yolo_v8.ipynb` 일부 셀은 다른 NAS 폴더(`/home/najo/NAS/RobotHuman_Co-work/`)를 참조

## 공용 노트북

| 파일 | 내용 |
|---|---|
| `All_pipeline.ipynb` | 전체 파이프라인 문서화: 세그멘테이션 → 2D 포즈 → 3D 포즈 |
| `Robot_pose_pipeline.ipynb` | 로봇 detection + crop → 2D → 3D 포즈 |
| `Vit_embedding.ipynb` | ViT 밑바닥 구현 학습용 |
| `depth_image_MiDas.ipynb` | MiDaS 단일 이미지 깊이 추정 |
| `yolo_v8.ipynb` | YOLOv8 seg + DeepLabV3 실험 |

## 관리 규칙

- 각 프로젝트의 git 작업은 **해당 디렉터리 안에서** 수행 (최상위 .gitignore가 중첩 repo를 전부 무시)
- 학습 산출물(`wandb/`, `outputs*/`, `results/`, `eval_outputs*/`)은 재생성 가능 — 주기적으로 삭제해 용량 관리
- 대용량 데이터(`datasets/`, ICRA `dataset/`, Meca500 zip)는 git 미추적 — NAS가 유일본이므로 삭제 주의
