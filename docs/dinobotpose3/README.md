# DINObotPose3 — 문서 인덱스

DREAM 벤치마크 단안 로봇 포즈 SOTA (배포 mean **0.804** vs RoboPEPP 0.780, 전 4카메라 가림 강건). 코드는 `3_pose_models/DINObotPose3/`.

> **최종 배포 모델: [FINAL_MODEL.md](FINAL_MODEL.md)** — 카메라별 체크포인트, 파이프라인, 재현 명령, head 계보.
> **먼저 읽을 것: [00_overview.md](00_overview.md)** — 성적표, 채택/반증 맵, 진행 중 실험, 로드맵을 한 페이지에.

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| [FINAL_MODEL.md](FINAL_MODEL.md) | **최종 배포 모델** — 카메라별 체크포인트/파이프라인/재현/head 계보 |
| [PAPER_DRAFT.md](PAPER_DRAFT.md) | 논문화 초안 — abstract, claim framing, 이후 섹션 확장용 |
| [00_overview.md](00_overview.md) | 세션 진행 종합 (성적·결정·현황) |
| [architecture/](architecture/model.md) | 모델 구조 — 백본/헤드/솔버/RC 파이프라인 |
| [data/](data/dataset.md) | 데이터 구조 — DREAM/Panda([dataset](data/dataset.md)) + 멀티로봇 FR5/FR3/Meca500([multi_robot](data/multi_robot.md)) |
| [training/](training/training.md) | 학습 — 스테이지, 손실 항, 자가학습, 체크포인트, GPU 주의 |
| [evaluation/](evaluation/evaluation.md) | 평가 — 프로토콜 3축, 지표, 하네스, 배포 성적, 재현 게이트 |
| [FULL_TEST_EVALUATION.md](FULL_TEST_EVALUATION.md) | Panda/KUKA/Baxter 전체 테스트셋 evaluator·명령·보고 기준 |
| [DEPLOYMENT_PIPELINE.md](DEPLOYMENT_PIPELINE.md) | 공통 bbox-from-2d → crop → solver → RC 배포 파이프라인 |
| [experiments/](experiments/README.md) | 실험별 기록 (날짜별, 판정 포함) |
| [references/](references/sota_survey.md) | 문헌 조사(SOTA 서베이) + [related_work](references/related_work.md)(RoboPose/CtRNet 대비 정직한 포지셔닝) + 다음 방향 로드맵 |

## 원본 로그 (프로젝트 내)

- `3_pose_models/DINObotPose3/EXPERIMENTS.md` — append-only 실험 일지 (전체 이력)
- `3_pose_models/DINObotPose3/SUMMARY.md` — 확정 결론 + WORKED/REFUTED + 재개 방법

## 핵심 한 줄

frozen DINOv3 검출기 → 관절각/회전 head → 운동학 솔버(cov-PnP) → nvdiffrast+SAM render-and-compare.
성능을 올린 건 전부 **학습 불필요 레버**(render-compare, cov-PnP, DARK 디코드). 백본 적응은 반증됨.
