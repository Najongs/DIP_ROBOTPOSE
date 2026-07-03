# 워크스페이스 정리 이력 & 참조 오버뷰

2026-07-03 대규모 재편의 기록. "왜 지금 이 구조인가"와 "무엇이 어디로 갔는가"의 단일 참조점.

## 정리 이력 (2026-07-03, 4단계)

### 1차: 산출물 정리 (~13GB 회수)
- wandb 로그·outputs 체크포인트·시각화 출력 등 재생성 가능한 학습 산출물 전부 삭제 (전부 git 미추적이었음)
- 삭제 범위: DINObotPose3 TRAIN 산출물 9.3G, DINObotPose v1/v2 outputs 2.7G, DINOv3_fine_tunning results·wandb 1G 등
- 예외 보존: `DINObotPose3/Eval/results_diffusion_real`(git 추적 파일 포함)

### 2차: 모노레포 전환 + 카테고리 재편
- 11개 중첩 독립 repo의 dirty 변경분을 각각 커밋·push 후 **개별 .git 제거** → 최상위 DIP가 유일한 git
- `DINObotPose`(v1), `DINObotPose2`(v2)는 push 검증 후 **로컬 삭제** — GitHub에만 존재 (v3로 대체됨)
- DINObotPose2는 GPU 서버 커밋(260309)과 non-fast-forward 충돌 → 원격 우선 merge로 통합
- ZED_Cap의 미적용 stash → 원격 `stash-backup-wip-aruco` 브랜치로 백업
- 폴더를 기능 카테고리(1_capture ~ 5_apps)로 이동, ICRA의 dataset 36G → `datasets/ICRA_multiview`
- Meca500 프로젝트의 zip 7개가 datasets 쪽과 md5 동일 → 삭제 (~1.2G)

### 3차: 경로 치환 + 심볼릭링크 제거
- 73개 파일의 구 절대경로(`DIP/<프로젝트>/`)를 새 구조(`DIP/<카테고리>/<프로젝트>/`)로 일괄 치환
- 임시로 뒀던 루트 호환 심볼릭링크 11개 제거
- **유일하게 남긴 링크**: `3_pose_models/2025_ICRA_.../dataset → ../../datasets/ICRA_multiview` (ICRA 코드가 `__file__` 기준 `<프로젝트>/dataset` 계산 — 제거 금지)

### 4차: 데이터셋 중복 제거 + 통합 (~1.9GB 회수)
- 해제본과 CRC 동일한 zip 5개 삭제 (zip 고유 파일 4개는 해제본에 보충 후)
- 캡처 프로젝트의 ICRA 복사본 삭제 (ZED ArUco_cap1~3, Panda frank_research3_ArUco_pose1/2 — pose2 차이는 quaternion 부호 반전뿐, 의미상 동일)
- franka 라벨을 `franka_research3_to_DREAM_modified`로 일원화 (구버전 삭제, 참조 코드 2곳 수정)
- 남은 원시 캡처 데이터 → `datasets/captures/` 이동 + 원위치 심볼릭링크

**용량 변화**: 69GB → ~54GB (재생성 가능 산출물·중복 ~15GB 제거, 데이터 유실 0)

## 무엇이 어디에 있나 (참조 맵)

### git / 원격

| 위치 | 내용 |
|---|---|
| **`origin = Najongs/DIP_ROBOTPOSE`** | 모노레포 유일 push 대상 (2026-07-03 생성) |
| `Najongs/<구 프로젝트명>` 개별 repo 11개 | 재편 전 이력 아카이브 — **읽기 전용, push 금지** |
| `Najongs/DINObotPose`, `DINObotPose2` | 원격에만 존재 (로컬 삭제됨). 필요 시 재클론 |
| `Najongs/ZED_Cap_make_dataset` `stash-backup-wip-aruco` 브랜치 | 미완성 ArUco 보정 WIP 백업 |
| `Najongs/DINObotPose3` | GPU 서버 작업의 중계 지점 — GPU 서버 → 이 repo → 모노레포 수동 반영 |

### 데이터 (git 미추적, NAS 유일본 — 삭제 금지)

- 전체 지도: [datasets.md](datasets.md)
- 핵심 제약: `ICRA_multiview/Converted_dataset/`의 라벨 JSON ~18만 개가 이미지를 **상대경로 참조** → **ICRA_multiview 내부 구조 변경 금지**
- 유일본 zip: `meca500/meca_Yolo_dataset.zip`(1.6G), `meca_insertion/vla_dataset250509_insertion.zip` 등 — 해제본 없음, 삭제 금지
- 재생성 불가 가중치: `4_perception/Fr5_robot_SegFormer/best_segformer_robot_arm.pth` (학습 데이터가 외부 서버에 있었음)

### 심볼릭링크 현황 (전부 의도적 — 제거 전 확인)

| 링크 | 대상 | 사유 |
|---|---|---|
| `3_pose_models/2025_ICRA_.../dataset` | `datasets/ICRA_multiview` | ICRA 코드가 `__file__` 기준 경로 계산 |
| `1_capture/ZED_Cap.../{1,2,3}_ArUco_cap` | `datasets/captures/zed_meca_aruco/` | 원시 데이터는 datasets, 워크플로우는 프로젝트 |
| `1_capture/Intertek.../ArUco_cap` | `datasets/captures/intertek_fr5/` | 〃 |
| `1_capture/DGIST.../Fr5_ArUco` | `datasets/captures/dgist_fr5/` | 〃 |

### 문서

- 실험 방법: [docs/README.md](README.md) 인덱스 → 카테고리별 가이드
- 작업 규칙: [../CLAUDE.md](../CLAUDE.md)
- DINObotPose3 연구 기록: `EXPERIMENTS.md`(일지) / `SUMMARY.md`(확정 결론·REFUTED 목록)
- SOTA 문헌: [robot_pose_sota_survey.md](robot_pose_sota_survey.md)

## 알려진 이슈 / 이미 깨져 있던 것 (재편과 무관)

- `notebooks/depth_image_MiDas.ipynb` → `datasets/Robot_data/*.tiff` 부재 (RobotHuman_Co-work 시절 데이터)
- `Meca500.../250514_3D_pose_estimation.ipynb` → `3d-robot-pose-estimation/models` 부재
- `4_perception/DINOv3_fine_tunning`: `coco_dataset/` 부재(사람포즈 학습용, 준비 필요), `configs/` 빈 폴더, PIPELINE_SETUP.md·TRAINING_SCRIPTS_README.md가 구버전 설계 서술 (RTMPose ↔ 실제 YOLO-pose)
- `Fr5_robot_SegFormer` 학습 데이터 경로가 외부 서버(`/home/ibom002/`) — 재학습 시 데이터 확보 필요
- DINObotPose3 스크립트의 `/data/public/NAS/...` 경로는 GPU 서버용 — 로컬 경로로 바꾸지 말 것
- `yolo_train_robot_box.yaml` → `datasets/FR5_model/` 참조 (현재 `FR5_robot/`만 존재 — 사용 시 경로 확인)

## GPU 서버 동기화 절차

1. GPU 서버에서 `Najongs/DINObotPose3`(개별 repo)로 push
2. 로컬에서 해당 repo fetch → 변경분 확인
3. 모노레포 경로 규칙에 맞춰 수동 반영 후 커밋 (선례: 모노레포 커밋 `bdd0fc1`)
4. 체크포인트(대용량)는 git 밖 — 필요 시 별도 복사
