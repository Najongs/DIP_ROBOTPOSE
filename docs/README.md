# DIP 문서 인덱스

카테고리별 실험/사용 방법 문서. 각 문서는 해당 폴더의 프로젝트별 {목적, 환경, 실행 명령, 입출력, 주의사항}을 담는다.

| 문서 | 대상 | 내용 |
|---|---|---|
| [1_capture.md](1_capture.md) | `1_capture/` | 공통 ArUco 캘리브레이션 워크플로우, ZED/Panda/Intertek/DGIST 캡처 절차 |
| [2_robot.md](2_robot.md) | `2_robot/` | FR5·Meca500 DH 파라미터, FK(관절각→좌표), 좌표↔픽셀 변환 |
| [3_pose_models.md](3_pose_models.md) | `3_pose_models/` | DINObotPose3 학습/평가/실험 로그, ICRA 멀티뷰 ablation, Meca500 3D 포즈 |
| [4_perception.md](4_perception.md) | `4_perception/` | DINOv3 파인튜닝(로봇포즈/깊이/사람포즈), 통합 파이프라인, FR5 SegFormer |
| [5_apps.md](5_apps.md) | `5_apps/` | 충돌 위험 파이프라인 실행법 |
| [datasets.md](datasets.md) | `datasets/` | 데이터셋 지도 (무엇이 어디에 있고 어떤 코드가 쓰는지) |

## 빠른 참조

- 워크스페이스 전체 구조·규칙: 루트 [README.md](../README.md), [CLAUDE.md](../CLAUDE.md)
- DINObotPose3 실험 상세 기록: [`3_pose_models/DINObotPose3/EXPERIMENTS.md`](../3_pose_models/DINObotPose3/EXPERIMENTS.md) (실험 일지), [`SUMMARY.md`](../3_pose_models/DINObotPose3/SUMMARY.md) (확정 결론·재개 방법)
- DREAM SOTA 문헌 서베이 + 가림-강건 아이디어: [robot_pose_sota_survey.md](robot_pose_sota_survey.md) (2026-07-03, 프로토콜 3축 주의사항 포함)
- 통합 파이프라인 설치: [`4_perception/DINOv3_fine_tunning/PIPELINE_SETUP.md`](../4_perception/DINOv3_fine_tunning/PIPELINE_SETUP.md)
