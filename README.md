# DIP — Robot Pose Estimation Workspace (Monorepo)

로봇 포즈 추정 연구 모노레포. 기능 흐름(데이터 수집 → 데이터셋 → 로봇/카메라 유틸 → 모델 학습 → 인식 → 응용)에 따라 카테고리로 구성.
**원격 없음** — 이 repo는 로컬 단일 저장소. 과거 프로젝트별 이력은 GitHub 개별 repo(`Najongs/<이름>`)에 아카이브로 보존됨.

> 2026-07-03 재편: 중첩 개별 git 제거 후 모노레포 통합, 카테고리 폴더 그룹핑, ICRA dataset(36G) datasets/로 통합, Meca500 중복 zip 제거(~1.2G).

## 구조

```
DIP/
├── 1_capture/        데이터 수집·캘리브레이션
│   ├── ZED_Cap_make_dataset/            ZED 스테레오 캡처 (RGB+엔코더)
│   ├── Panda_cap_make_dataset/          Franka Panda ArUco 캡처/전처리
│   ├── Intertek_Zed_ArUco_Calibration/  ZED+ArUco 캘리브레이션
│   └── DGIST_IROM_Data_collection/      DGIST 데이터 수집 (데이터, git 미추적)
├── 2_robot/          로봇 유틸
│   └── Robot_joint_inference/           관절각→좌표 FK, FR5/Meca500 DH 파라미터 문서
├── 3_pose_models/    자세추정 모델 학습
│   ├── DINObotPose3/                    DINOv3 백본 3D 포즈 (PnP+FK+diffusion) — 계열 최신
│   ├── 2025_ICRA_Multi_View_Robot_Pose_Estimation/   멀티뷰 포즈 (dataset → datasets/ICRA_multiview 링크)
│   └── Meca500_3D_Pose_Estimation/      Meca500 3D 포즈
├── 4_perception/     세그멘테이션·뎁스·통합
│   ├── Fr5_robot_SegFormer/             FR5 세그멘테이션 (best_segformer_robot_arm.pth) — 활성
│   └── DINOv3_fine_tunning/             DINOv3 파인튜닝+뎁스+통합 파이프라인 (dinov3/, Depth-Anything-3/는 외부 클론)
├── 5_apps/           응용
│   └── collision_risk_pipeline/         로봇-사람 충돌 위험 (SegFormer+YOLO) — 활성
├── datasets/         데이터 허브 (git 미추적, NAS 유일본 — 삭제 주의)
│   ├── ICRA_multiview/  (구 2025_ICRA .../dataset, 36G: DREAM_syn, franka_research3, Converted_dataset 등)
│   ├── FR5_robot/  meca500/  meca_insertion/  intertek_image/ ...
├── notebooks/        공용 노트북 (All_pipeline, Robot_pose_pipeline, Vit_embedding, depth_image_MiDas, yolo_v8)
├── assets/           참조 없는 이미지 모음
└── yolov8n-seg.pt  yolo_train_robot_box.yaml   # 코드가 루트 기준 참조 → 루트 유지
```

## 경로 규칙

- 2026-07-03에 전체 코드의 구 절대경로(`/home/najo/NAS/DIP/<프로젝트>/...`)를 새 구조로 일괄 치환 완료 (73파일). 루트 호환 심볼릭링크는 제거됨.
- 예외 하나: `3_pose_models/2025_ICRA_.../dataset → ../../datasets/ICRA_multiview` 심볼릭링크는 **유지** — ICRA 학습 코드가 `__file__` 기준으로 `<프로젝트>/dataset`을 계산하기 때문 (예: `Train/FR5/fr5_main.py`의 `DATASET_ROOT`).
- 이미 깨져 있던 참조(삭제된 학습 산출물 .pth, 존재한 적 없는 `coco_dataset/`, `3d-robot-pose-estimation/` 등)는 그대로 둠 — 재학습/재준비 시 새 경로에 생성하면 됨.

## git 관리

- 이 repo가 유일한 git. 코드·문서·설정만 추적 (미디어/데이터/가중치/학습 산출물은 .gitignore).
- 과거 이력: GitHub `Najongs/<프로젝트>` 개별 repo에 보존 (읽기 전용 아카이브로 취급, push 금지).
  - `DINObotPose`, `DINObotPose2`: 원격에만 존재 (로컬 삭제됨)
  - `ZED_Cap_make_dataset`의 미완성 ArUco WIP는 원격 `stash-backup-wip-aruco` 브랜치에 백업
- **GPU 서버 주의**: DINObotPose3 최신 실험은 GPU 서버(`/data/public/NAS/...`)에서 개별 repo 기준으로 진행됨.
  GPU 서버 작업을 가져올 땐 GitHub 개별 repo 경유(fetch) 후 이 모노레포에 수동 반영.

## 의존 관계

- `5_apps/collision_risk_pipeline` → `4_perception/Fr5_robot_SegFormer/best_segformer_robot_arm.pth`, 루트 `yolov8n-seg.pt`
- DINOv3_fine_tunning(71곳)·DINObotPose3(16곳) → `datasets/ICRA_multiview` (직접 참조)
- `notebooks/Robot_pose_pipeline.ipynb` → 루트 `yolo_train_robot_box.yaml`
- 노트북들 → `datasets/` 절대경로. `yolo_v8.ipynb` 일부는 다른 NAS 폴더(`RobotHuman_Co-work`) 참조

## 관리 규칙

- 학습 산출물(`wandb/`, `outputs*/`, `results*/`, `eval_outputs*/`)은 재생성 가능 — 주기적으로 삭제
- 새 프로젝트는 카테고리 폴더 아래 생성, 절대경로 하드코딩 대신 인자/설정 파일 사용 권장
