# 1_capture — 데이터 수집·캘리브레이션

4개 프로젝트가 로봇·카메라 대수만 다르고 **동일한 ArUco 캘리브레이션 파이프라인**을 공유한다.
모든 캡처 스크립트는 CLI 인자가 없고 **상단 상수(카메라 시리얼, 저장 경로, 로봇 IP)를 직접 수정**해야 한다.

## 공통 ArUco 워크플로우 (3단계)

1. **내부파라미터 추출 — `Calib_cam_save.py`**
   - ZED SDK `.conf`(예: `camera_conf/SN41182735.conf`)에서 FHD 기준 `fx,fy,cx,cy` + 왜곡계수를 파싱해 JSON 저장. 카메라 연결 불필요.
   - 출력: `Calib_cam_from_conf/{position}_{serial}_{leftcam|rightcam}_calib.json`

2. **ArUco 캡처 — `cap_aruco_image.py`**
   - ZED 라이브 스트림에서 ArUco(`DICT_5X5_50`, 마커 0.05m) 검출 → undistort → solvePnP+RefineLM → EMA(α=0.1, 회전 SLERP) 안정화. `q` 키로 현재 프레임 저장.
   - 1회 실행 = 카메라 1대 · 뷰 1개. 반복 실행으로 `1_ArUco_cap/`, `2_ArUco_cap/`... 폴더를 쌓는다.
   - 출력: `{position}_{serial}_{view}_{ts}.png` + `.json`(마커별 position_m, rotation_quat, corners_pixel)
   - Panda 프로젝트의 `New_cap_aruco_image.py`는 개선판: 4대×양쪽 뷰를 스레드로 **한 번에 단발 캡처**.

3. **보정 후처리 — `*_Preprocessing.ipynb` / `aruco_calculation.ipynb`**
   - N회 캡처 JSON을 마커별로 모아 위치는 평균, 회전은 quaternion 평균(공분산 고유벡터)으로 보정.
   - 출력: `Correct_ArUco/{view}_{cam}_corrected.json`, `aruco_pose_summary.json`

**카메라 시리얼 매핑**
- 4-카메라 세트(RobotE/Panda): `41182735=front(view1)`, `49429257=right(view2)`, `44377151=left(view3)`, `49045152=top(view4)`
- 3-카메라 세트(Intertek/DGIST): `34850673=right`, `38007749=left`, `30779426=top`

**로봇 IP**: Mecademic `192.168.0.100`(`mecademicpy`) / FR5 `192.168.58.2`(`fairino` SDK)

---

## ZED_Cap_make_dataset — Meca(RobotE) RGB+엔코더 수집

- `zed_captrue_image_robotE.py`: 4대 ZED(front/right/left/top) + Mecademic 로봇을 스레드 동기 시작으로 30초 수집.
  - 출력: `vla_dataset/{view}/zed_{serial}_{left|right}_{ts}.jpg` + `vla_dataset/robot_data.txt`(0.1초 간격 timestamp, joint_angles, cartesian_pose)
  - 후처리에서 이미지·로봇 로그를 타임스탬프 최근접 매칭으로 동기화.
- `aruco_calculation.ipynb`: 공통 3단계의 후처리 (입력 `ArUco_cap1~3_250514` → 출력 `Correct_ArUco`, `aruco_pose_summary.json`).

## Panda_cap_make_dataset — Franka Research 3

- `Capture_multi_view.py`: 4대 ZED 30초 이미지 수집(로봇 제어 없음) → `Panda_dataset/view{1-4}/`
- `New_cap_aruco_image.py`: 4대 동시 단발 ArUco 캡처 → `ArUco_capture_dataset/`
- `Panda_Preprocessing.ipynb`: `ArUco_capture_dataset_1~5` 보정 → `Correct_ArUco`
- README에 Panda FR3 DH 파라미터 참고문헌 + `Panda_DH.png`

## Intertek_Zed_ArUco_Calibration — FR5 현장(3-카메라)

순서: `Calib_cam_save.py` → `cap_aruco_image.py`(→ `ArUco_cap/`) → `Fr5_capture_robot.py` → `Fr5_Preprocessng.ipynb`(→ `Correct_ArUco`)

- `Fr5_capture_robot.py`: 3대 ZED(HD1080) + FR5 관절각(`GetActualJointPosDegree`) 30초 동기 수집.
  - 출력: `Fr5_intertek/{right|left|top}/*.jpg` + `Fr5_intertek/joint/joint_{serial}_{ts}.json`
  - FR5 SDK 경로가 `sys.path.append(...fairino)`로 하드코딩 — 환경에 맞게 수정.

## DGIST_IROM_Data_collection — 캡처→동기화→DREAM 변환 엔드투엔드 (git 미추적 데이터 폴더)

가장 완성된 흐름. 캡처(`Fr5_capture_robot.py`) → 동기화 → DREAM 변환:

- `Fr5_sync.py`: `datasets/ICRA_multiview/Fr5/Fr5_*_250526`의 관절 JSON ↔ 이미지 타임스탬프 최근접 매칭 → `fr5_matched_joint_angle.csv`
  - 핵심 파라미터: `MAX_TIME_DIFFERENCE_THRESHOLD=0.05`(50ms 초과 제외), `IMAGE_TIMESTAMP_DELAY=0.0333`(카메라 지연 보정)
- `FR5_to_DREAM.ipynb`: 동기화 CSV + `Fr5_aruco_pose_summary.json` + `Calib_cam_from_conf/` → DREAM 포맷 데이터셋(`Fr5_to_DREAM/`) 생성 (FK로 관절 3D 계산 → 카메라 파라미터로 2D 투영 라벨)

## 새 로봇/현장 데이터셋을 만들 때

1. `.conf` 파일 확보 → `Calib_cam_save.py`로 calib JSON
2. ArUco 마커 촬영 N회 → 보정 노트북으로 `Correct_ArUco` (카메라↔로봇 베이스 외부 파라미터)
3. 로봇 구동하며 이미지+관절각 동기 캡처 (`*_capture_robot.py` 계열)
4. `Fr5_sync.py` 패턴으로 타임스탬프 매칭 CSV 생성
5. `FR5_to_DREAM.ipynb` 패턴으로 DREAM 포맷 변환 → `datasets/`에 배치
