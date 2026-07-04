# 멀티로봇 데이터 구조 (FR5 / FR3 / Meca500 / Meca_insertion)

> DREAM(Panda) 외 로봇들의 데이터 구조. Panda 데이터는 [dataset.md](dataset.md).
> 원본 캡처는 `datasets/`, 변환 코드는 `3_pose_models/2025_ICRA_Multi_View_Robot_Pose_Estimation/`.

## 두 개의 병렬 파이프라인

1. **2025_ICRA_Multi_View_Robot_Pose_Estimation** — 멀티뷰 DREAM 통일 키포인트 파이프라인. `dataset → ../../datasets/ICRA_multiview` 심링크. FR5/FR3/Meca500/Meca_insertion 원본을 **DREAM 스타일 프레임당 JSON**으로 변환(`Converted_dataset/<robot>_to_DREAM/`). DINObotPose3와 동일 로더 재사용.
2. **Meca500_3D_Pose_Estimation** — Meca500 전용 단일뷰, 3D 키포인트 직접 회귀(ViT+HRNet). GT는 **CSV + `3D_Coordinate_label/` JSON**(DREAM 포맷 아님).

공통: Stereolabs **ZED** 스테레오(파일명 `zed_<serial>_{left,right}_<ts>.jpg`), 내부 파라미터는 `.conf`/`*_calib_cam_from_conf/*.json`. **GT 3D = 엔코더 관절각 → DH FK, ArUco 베이스 마커 포즈로 카메라 프레임에 앵커**; GT 2D = K+dist 핀홀 투영.

## 변환 DREAM JSON 스키마 (Converted_dataset, 로봇 공통)

`*_to_DREAM.ipynb`가 프레임당 JSON 1개 생성 (검증됨 — 4개 로봇 샘플 확인):
```
objects: [{
  class: "Fr5" | "research3" | "Meca500" | "MecaInsertion"
  visibility: 1
  location: [x,y,z]                              // link0(베이스) 3D, 카메라 프레임(m) = ArUco tvec
  keypoints: [{ name, location:[x,y,z], projected_location:[u,v] }, ...]
}]
sim_state: { joints: [{ name, position(rad), velocity }, ...] }
meta: {
  image_path: <상대 .jpg 경로>
  view: <top/left/right/view1..4/Meca 등>,  cam: <left|right|leftcam|rightcam>
  K: [[fx,0,cx],[0,fy,cy],[0,0,1]],  dist_coeffs: [k1,k2,p1,p2,k3]
  aruco_rvec_tvec: { rvec:[...], tvec:[x,y,z] }  // 베이스 ArUco 포즈(카메라 프레임)
}
```
- `location`=카메라 프레임 3D(m), `projected_location`=2D 픽셀. link0의 location == `meta.aruco_rvec_tvec.tvec`(베이스=ArUco 원점).
- **주의**: 이 변환 포맷은 원본 DREAM **합성** 포맷(`camera_data`/`pose_transform`/`cuboid` 등, richer)과 다름 — 변환본은 `objects/keypoints`+`sim_state`+`meta`만 유지. 합성 Panda는 [dataset.md](dataset.md).
- `sim_state` position은 rad, 원본 로봇 파일은 도(FR5/Meca) 또는 rad(FR3 ROS).

## 로봇별 요약

| 로봇 | 원본 이미지(뷰) | 원본 관절 | #KP | 키포인트 이름 | 관절수 | ArUco GT | 변환본(개수) |
|---|---|---|---|---|---|---|---|
| **FR5** | `Fr5/Fr5_{1..7}th_250526/{left,right,top}` | `joint/*.json` 6 (deg) | 7 | `Fr5_link0..6` | 6 | `Fr5_aruco_pose_summary.json` | `Fr5_to_DREAM/` (9,140) |
| **FR3** | `franka_research3_pose{1,2}/Panda_dataset_*/{view1..4}` | CSV pos/vel/effort j1..7+finger (rad) | 7 | `panda_link0,2,3,4,6,7 + panda_hand` | 7+finger | `pose{1,2}_aruco_pose_summary.json` | `franka_research3_to_DREAM_modified/` (16,148) |
| **Meca500** | `Meca500/image/` 1000 (단일 "Meca" 뷰) | `angle/*.json` 6 (deg) | 7 (6 distinct) | `Meca500_link0..6` | 6 | `Meca500_aruco_pose_summary.json` | `Meca500_to_DREAM/` (998) |
| **Meca_insertion** | `Meca_{1th_hand,2th_hand2,3th_insertion}/{left,right,front,top}` | CSV 7 각도 (deg, j7=툴) | 7 (6 distinct) | `Meca500_link0..6` | 6(+원본 j7) | `Meca_insertion_aruco_pose_summary.json`(+std) | `Meca_insertion_to_DREAM/` (1,548) |

## FR5 (Fairino FR5, 6-DOF)

- **원본**: `Fr5/Fr5_{1..7}th_250526/` — 3 ZED 뷰 `left/right/top`. 관절 `joint/joint_<serial>_<ts>.json` = **6각도(도) bare 리스트**. 동기화 `sync/Fr5_sync.py` → `fr5_matched_joint_angle.csv`(image_path, timestamp, joint_1..6).
- **키포인트**: 7 (`Fr5_link0..6`, link0=베이스=ArUco). **DH** [alpha°,a,d,θ_off]:
  ```
  { 90, 0,     0.152, 0}  { 0, -0.425, 0, 0}  { 0, -0.395, 0, 0}
  { 90, 0,     0.102, 0}  {-90, 0,     0.102, 0}  { 0, 0, 0.100, 0}
  ```
- **카메라**: 멀티뷰(top/left/right)×스테레오. 내부 `Fr5_calib_cam_from_conf/`. ArUco `Fr5_aruco_pose_summary.json`.
- **변환**: `Converted_dataset/Fr5_to_DREAM/` (9,140 JSON). 별개 구 데이터셋 `datasets/FR5_robot/`(단일뷰, YOLO 모델 포함)는 DREAM 변환과 무관.

## FR3 / franka_research3 (Franka Research 3, 7-DOF)

- **원본**: `franka_research3_pose{1,2}/Panda_dataset_*[_dark]/{view1..4}` (4 ZED 뷰). 동기화 `sync/franka_research3_sync.py` → `fr3_matched_joint_angle.csv` (ROS 스타일: joint1..7 + finger1,2 각각 position/velocity/effort, rad).
- **키포인트**: 7 = **Panda 셋 동일** `panda_link0,2,3,4,6,7,hand`. `sim_state.joints`: panda_joint1..7 + finger1. 운동학적으로 Panda → DINObotPose3 파이프라인 직접 재사용.
- **DH** (Franka 표준 8행) [a,d,alpha°]:
  ```
  {0,0.333,0} {0,0,-90} {0,0.316,90} {0.0825,0,90} {-0.0825,0.384,-90} {0,0,90} {0.088,0,90} {0,0.107,0}(flange)
  ```
- **변환**: `franka_research3_to_DREAM/`(base) → `..._modified/` (16,148 JSON, `FR3_to_DREAM_Fix.py`가 `meta.image_path`만 재작성). real(ArUco) + 합성 `DREAM_syn/franka_research3/` 둘 다.

## Meca500 (Mecademic, 6-DOF)

- **원본**: `Meca500/image/` 1000장(단일 "Meca" 뷰, mono). `angle/*.json` = **6각도(도)**. 동기화 `sync/Meca500_sync.py` → `Meca500_matched_joint_angle.csv`(image_path, joint_1..6).
- **키포인트**: 7(`Meca500_link0..6`)이나 link4/5 3D가 중복 → **6 distinct**(손목 collapse). **DH** [alpha°,a,d,θ_off°]:
  ```
  {-90, 0,     0.135, 0}  { 0, 0.135, 0, -90}  {-90, 0.038, 0, 0}
  { 90, 0,     0.120, 0}  {-90, 0,     0,  0}  { 0, 0, 0.070, 0}
  ```
- **카메라**: 단일뷰 top-down. K 임베드(`fx 740.29, fy 740.67, cx 970.83, cy 555.39`). ArUco `Meca500_aruco_pose_summary.json`(tvec 0,0.04,0.75; rvec 도).
- **변환**: `Meca500_to_DREAM/` (998 JSON). 별개 대형 워크스페이스 `datasets/meca500/`(2D/3D 모델, YOLO 1.6G)는 실험/원본 측.

## Meca_insertion (Meca500 니들 삽입, 6-DOF + 툴)

- **원본**: `Meca_{1th_hand,2th_hand2,3th_insertion_250514}/{left,right,front,top}` (4 ZED 뷰). 동기화 `sync/Meca_insertion_sync.py` → CSV `joint_1..7`(도, j7=삽입/툴축). 전처리 `Meca_insertion_preprocessing.py`(19.9KB).
- **키포인트**: Meca500 셋 동일(`Meca500_link0..6`, 6 distinct). class `"MecaInsertion"`. `sim_state`는 joint1..6만.
- **카메라**: 멀티뷰(front/left/right/top)×스테레오. K 임베드(`fx 737.12, fy 737.09, ...`, ZED serial 41182735=FR3와 동일). ArUco `Meca_insertion_aruco_pose_summary.json`(가장 richest — tvec + **std_x/y/z** + proj + rvec 도).
- **변환**: `Meca_insertion_to_DREAM/` (1,548 JSON). 원본 삽입 캡처 `datasets/meca_insertion/`(8세션 25G/26G, VLA 데이터).
- **용도**: render-and-compare/가림 연구의 **실제 텍스처 가림체**(니들/손이 관절 가림) 미래 타깃.

## Meca500_3D_Pose_Estimation (독립, Meca500 전용)

`3_pose_models/Meca500_3D_Pose_Estimation/` — DREAM JSON 아님, 3D 직접 회귀.
- **원본**: ZED 이미지(`datasets/meca_insertion/vla_dataset*`), 내부 `settings/SN<serial>.conf`(2K/FHD 섹션별 fx/fy/cx/cy/k1..k3), 엔코더+TCP+ArUco.
- **중간 파일**:
  - `synchronized_robot_camera_data_{1,2,3}.csv` — `image_path, image_timestamp, robot_timestamp, time_difference_s, timestamp, j1..j6, x,y,z,rx,ry,rz, datetime` (**6관절각(도) + TCP 포즈**). 생성: `Encoder_2_Camera.ipynb`, `250514_Data_preprocessing.ipynb`.
  - `aruco_final_summary.json` — 뷰+cam별 베이스 마커 `mean_x/y/z`(평균 tvec, 카메라 프레임 m), `std_x/y/z`, `proj_x/y`, `rvec_*_deg`.
- **GT (3D 키포인트 라벨)**: `3D_Coordinate_label/{left,right,front,top}/<image>.json` = `{ image_path, joint_coords_camera_frame_meters: [[x,y,z]×7] }` — DH FK 링크 원점을 ArUco로 카메라 프레임에 배치. (link5/6 중복 → 6 distinct)
- **DH**: ICRA Meca500 셋과 동일.
- **모델**: `250514_3D_pose_estimation_model.py`(ViT `vit_base_patch16_224` + HRNet 헤드, `RoboKeypointModel(num_kp=6|7)`), `model_save/{6kp_model,7kp_model}/`. 2D 라벨 `labels/`(LabelMe/YOLO), bbox `yolo11{n,x}.pt`.
- **워크플로우**: `Encoder_2_Camera.ipynb` → `250514_Data_preprocessing.ipynb`(ArUco cam→base + DH FK) → `250514_Yolo_robot_box_model.ipynb`(bbox) → `250514_3D_pose_estimation.ipynb`.

## GT 생성 공통 흐름

```
엔코더 관절각(도/rad) ──DH FK──▶ 로봇 프레임 3D 링크 원점
                                       │
ArUco 베이스 마커 ──▶ 카메라→베이스 포즈 (rvec,tvec)
                                       │
                            카메라 프레임 3D ──K+dist 투영──▶ 2D 픽셀
                                       │
                    DREAM JSON (location + projected_location + sim_state + meta)
```
Panda(DREAM_syn)만 합성 렌더 GT, 실로봇 4종은 **엔코더+ArUco+DH FK**. DH 참조: `2_robot/Robot_joint_inference/`(FR5·Meca500 이미지 + `DH_angle_to_coordinate.ipynb`).

## 핵심 스크립트 경로

- 변환: `Converted_dataset/{FR5,franka_research3,Meca500,Meca_insertion}_to_DREAM.ipynb`, `FR3_to_DREAM_Fix.py`, `DREAM_to_DREAM_syn.py`
- 동기화: `sync/{Fr5,franka_research3,Meca500,Meca_insertion,DREAM}_sync.py` (nearest-timestamp, 임계 0.05s)
- Meca500 3D: `Meca500_3D_Pose_Estimation/{Encoder_2_Camera, 250514_Data_preprocessing, 250514_3D_pose_estimation_model.py}`

## 관련
- Panda(DREAM) 데이터: [dataset.md](dataset.md)
- ICRA 멀티뷰 학습: `docs/3_pose_models.md` §2025_ICRA
- 데이터셋 전체 지도: `docs/datasets.md`
