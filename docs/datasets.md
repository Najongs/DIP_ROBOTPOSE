# datasets/ — 데이터 지도

**전부 git 미추적, NAS가 유일본 — 삭제 금지.** 총 ~44GB.

| 폴더 | 내용 | 주 사용처 |
|---|---|---|
| `ICRA_multiview/` (36G) | 멀티뷰 포즈 데이터 허브 (아래 상세) | DINObotPose3, ICRA, DINOv3_fine_tunning |
| `FR5_robot/` | FR5 데이터(angle/box_label/depth/image/joint/point_label) + 캡처 코드 + `fr5_yolo_best_model.pt`, `YOLO_Train/` | Robot_joint_inference, yolo_train_robot_box.yaml |
| `meca500/` | Meca500 2D/3D 좌표모델, ArUco, YOLO 학습 노트북, `meca_Yolo_dataset.zip` | Meca500_3D_Pose_Estimation |
| `meca_insertion/` | Meca 바늘삽입 실험 회차별(`Meca_1th~8th`) + vla_dataset zip들 | Meca500, yolo_v8.ipynb |
| `intertek_image/` | Intertek Basler(acA1300) TIFF 원본 | (아카이브) |
| `Fr5_label_output/` | FR5 라벨 출력물(1st~7th) | |
| `ZED/` | ZED 캡처 스크립트만 | |
| `DOWNLOAD.sh` | 다운로드 스크립트 | |

## ICRA_multiview/ 상세 (구 `2025_ICRA_.../dataset`)

`3_pose_models/2025_ICRA_.../dataset` 심볼릭링크가 여기를 가리킴 (링크 제거 금지).

| 하위 | 내용 |
|---|---|
| `Converted_dataset/` | **DREAM 포맷 통일 변환본** — 대부분의 학습 코드가 참조하는 실제 입력. `DREAM_to_DREAM_syn/panda_synth_train_dr`(합성 학습), `DREAM_to_DREAM/panda-3cam_azure` 등(실사 검증) |
| `DREAM_syn/` | DREAM 합성 원본 (panda_synth_train_dr / test_dr / test_photo) |
| `DREAM_real/` | DREAM 실사 원본 (panda-orb, panda-3cam_azure/kinect360/realsense) |
| `Fr5/` | FR5 멀티뷰 (Fr5_1th~7th_250526, ArUco, calib) |
| `franka_research3/` | Franka Research 3 (pose1/pose2, ArUco, Joint_Angle) |
| `Meca500/`, `Meca_insertion/` | Meca 멀티뷰/삽입 |
| `*_preprocessing.py`, `*_Calib_cam_save.py` | 로봇별 전처리·캘리브 스크립트 |

## 규칙

- 새 데이터셋은 여기(`datasets/<이름>/`)에 배치하고 이 문서에 한 줄 추가
- 학습 코드가 참조하는 것은 원본이 아니라 `Converted_dataset/`(DREAM 포맷) — 새 로봇 데이터는 [1_capture.md](1_capture.md)의 변환 절차로 DREAM화 후 사용
- zip 원본과 압축 해제본이 공존하는 경우가 있음(meca_insertion) — 정리 시 md5 대조 후 중복만 제거
