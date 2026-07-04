# 멀티로봇 데이터 구조 (FR5 / FR3 / Meca500 / Meca_insertion)

> DREAM(Panda) 외 로봇들의 데이터 구조. Panda 데이터는 [dataset.md](dataset.md).
> 원본 캡처는 `datasets/`, 변환 코드는 `3_pose_models/2025_ICRA_Multi_View_Robot_Pose_Estimation/`.

## 두 개의 병렬 데이터 생태계

1. **2025_ICRA_Multi_View_Robot_Pose_Estimation** — 멀티뷰, 모든 로봇을 **통일 DREAM 포맷 JSON**으로 변환(`Converted_dataset/`). 주력 파이프라인. DINObotPose3와 동일 로더(`PoseEstimationDataset`/`EvalDataset`) 재사용.
2. **Meca500_3D_Pose_Estimation** — 단일뷰 Meca500 전용, ArUco+엔코더 GT, **CSV 포맷**(DREAM JSON 아님).

원본은 `datasets/`, ICRA 프로젝트는 `dataset → ../../datasets/ICRA_multiview` 심링크로 접근(제거 금지 — `__file__` 기준 경로 계산).

## 통일 DREAM 포맷 스키마 (FR5/FR3/Meca500 공통 목표)

`Converted_dataset/*_to_DREAM.{py,ipynb}` 변환기가 프레임당 JSON 1개 생성 (NDDS/DREAM 스타일):
```json
{
  "camera_data": { "location_worldframe":[x,y,z], "quaternion_xyzw_worldframe":[...],
                   "camera_matrix":[[fx,0,cx],[0,fy,cy],[0,0,1]] },
  "objects": [{
    "class": "robot",
    "keypoints": [
      {"name":"link0", "location":[X,Y,Z],        // 3D, 카메라 프레임 (m)
       "projected_location":[u,v]},               // 2D 픽셀
      ...
    ],
    "joint_angles": [a1,a2,...],                   // rad (로봇별 관절 수)
  }],
  "sim_state"/"camera": { "K": [[...]] }           // 내부 파라미터
}
```
**`location`=카메라 프레임 3D, `projected_location`=2D 픽셀** — Panda/DREAM과 동일 규약이라 다운스트림 로더 공유. 단 `joint_angles` 길이·선택 필드는 변환기별로 약간 다름 → 정확한 필드는 해당 변환기가 ground truth.

## 로봇별 요약

| 로봇 | DOF | 원본 GT 소스 | 키포인트 | DREAM 변환 위치 | 뷰 |
|---|---|---|---|---|---|
| **FR5** (Fairino) | 6 | 엔코더 + DH FK + ArUco | 6 관절 | `Converted_dataset/Fr5_to_DREAM/` | 멀티뷰 ZED |
| **FR3** (Franka Research 3) | 7 | `fr3_matched_joint_angle.csv` + FK + ArUco; 합성도 | 7 관절 + hand (Panda 셋) | `Converted_dataset/franka_research3_to_DREAM/` + `DREAM_syn/franka_research3/` | 멀티뷰(pose1/2/…) |
| **Meca500** (ICRA) | 6 | 엔코더 + DH FK + ArUco | 6 관절 | `Converted_dataset/Meca500_to_DREAM/` | 멀티뷰 |
| **Meca500** (standalone) | 6 | 엔코더 txt + ArUco (CSV) | 6kp / 7kp | `Meca500_3D_Pose_Estimation/` (CSV) | 단일뷰 |
| **Meca_insertion** | 6 | 삽입 태스크 캡처 + 라벨 | Meca500 셋 + 툴팁 | `Converted_dataset/Meca_insertion_to_DREAM/` | — |

## FR5 (6-DOF)

- **원본**: `datasets/FR5_robot/` — `capture.py`, `angle/`(관절각), `box_label/`·`Fr5_label_output/`(2D 라벨). 멀티뷰 ZED (ICRA `Train/FR5/`의 주력 로봇).
- **변환**: `Converted_dataset/FR5_to_DREAM.ipynb`, 전처리 `Fr5_preprocessing.py`, 동기화 `sync/Fr5_sync.py`.
- **DH 파라미터** [alpha°, a(m), d(m), θ_offset] (`2_robot/Robot_joint_inference` 검증):
  ```
  J1:[ 90,  0.000, 0.152, 0]   J2:[  0, -0.425, 0.000, 0]   J3:[  0, -0.395, 0.000, 0]
  J4:[ 90,  0.000, 0.102, 0]   J5:[-90,  0.000, 0.102, 0]   J6:[  0,  0.000, 0.100, 0]
  ```
- **GT**: 엔코더 관절각 → DH FK → 3D 관절좌표 → ArUco 카메라 extrinsic → 2D 투영.

## FR3 / franka_research3 (7-DOF)

- **원본**: `datasets/ICRA_multiview/franka_research3/` — `fr3_matched_joint_angle.csv`(타임스탬프 매칭 관절각), `franka_research3_ArUco_pose1/`·`pose2/`…(멀티뷰 ArUco). 합성: `DREAM_syn/franka_research3/`(DREAM 렌더).
- **변환**: `franka_research3_to_DREAM.ipynb`, `FR3_to_DREAM_Fix.py`(경로 픽스), 전처리 `Franka_research3_preprocessing.py`.
- **키포인트**: 7관절 + hand = **Panda 셋과 동일**(`link0,link2,link3,link4,link6,link7,hand`) — FR3는 Franka 연구용 암, 운동학적으로 Panda. → DINObotPose3 파이프라인 직접 재사용 가능.
- **GT**: csv 관절각 → FK → 3D, ArUco가 뷰별 extrinsic. real(ArUco) + 합성(DREAM_syn) 둘 다.

## Meca500 (6-DOF) — 두 표현

### (a) ICRA 멀티뷰 (DREAM 포맷)
- **원본**: `datasets/meca500/` — `aruco/`, `2D_coordinate_model/`, `3D_coordinate_model/`.
- **변환**: `Meca500_to_DREAM.ipynb`, 전처리 `Meca500_preprocessing.py`, 동기화 `sync/Meca500_sync.py`.
- **DH 파라미터** [alpha°, a(m), d(m), θ_offset°] (검증):
  ```
  J1:[-90, 0.000, 0.135,   0]   J2:[  0, 0.135, 0.000, -90]   J3:[-90, 0.038, 0.000, 0]
  J4:[ 90, 0.000, 0.120,   0]   J5:[-90, 0.000, 0.000,   0]   J6:[  0, 0.000, 0.070, 0]
  ```

### (b) standalone `Meca500_3D_Pose_Estimation/` (CSV, DREAM 아님)
독립 단일뷰 파이프라인, 자체 엔코더+ArUco GT:
- **원본**: 엔코더 `.txt`(관절 읽기), 카메라 타임스탬프 프레임, ArUco 마커
- **중간 CSV**: `synchronized_robot_camera_data_*.csv`(엔코더 10ms 보간 후 카메라 타임스탬프 정렬), `aruco_final_summary.json`(카메라→베이스 변환)
- **GT**: DH FK로 6kp/7kp 3D → 2D 투영
- **노트북 워크플로우**: `Encoder_2_Camera.ipynb` → `250514_Data_preprocessing.ipynb`(ArUco cam→base T + DH FK) → `250514_Yolo_robot_box_model.ipynb`(YOLO bbox) → `250514_3D_pose_estimation.ipynb`
- **모델**: `model_save/{6kp_model,7kp_model}/epoch_*.pth`
- **데이터**: `datasets/meca500/{2D,3D}_coordinate_model/`

## Meca_insertion (Meca500 삽입 태스크 변형)

- **원본**: `datasets/meca_insertion/` — `Meca_1th_25G_insertion/`(25G 니들 삽입, 정밀/의료), `label.zip`·`label_20250514.zip`.
- **변환**: `Meca_insertion_to_DREAM.ipynb`. Meca500 6관절 셋 + 삽입 툴팁. 동일 DREAM JSON.
- **용도**: render-and-compare/가림 연구에서 **실제 텍스처 가림체**(니들 삽입 시 툴/손이 관절 가림)의 미래 타깃으로 지목됨.

## GT 생성 공통 흐름

```
엔코더 관절각 ──DH FK──▶ 로봇 프레임 3D 관절좌표
                              │
ArUco 마커 ──▶ 카메라→베이스 extrinsic (R,t)
                              │
                    카메라 프레임 3D ──K 투영──▶ 2D 픽셀
                              │
                    DREAM JSON (location + projected_location)
```
Panda(DREAM)는 합성 렌더 GT, 나머지 실로봇은 **엔코더+ArUco+DH FK** GT. DH 파라미터 참조: `2_robot/Robot_joint_inference/`(FR5·Meca500 이미지 + `DH_angle_to_coordinate.ipynb`).

## 관련
- Panda(DREAM) 데이터: [dataset.md](dataset.md)
- ICRA 멀티뷰 학습: `docs/3_pose_models.md` §2025_ICRA
- 데이터셋 전체 지도: `docs/datasets.md`
