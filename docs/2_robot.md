# 2_robot — 로봇 운동학 유틸

## Robot_joint_inference

목적: 관절각 → 로봇 각 관절의 3D 좌표(FK). FR5·Meca500의 DH 파라미터 문서와 좌표↔픽셀 변환 실험 포함.

### 참조 문서 (`img/`)

| 파일 | 내용 |
|---|---|
| `FR5_model_DH_parameter.png` | FR5 DH 파라미터 표 |
| `FR5_Ontology_Drawing.jpg` | FR5 관절/링크 구조 도면 |
| `meca500_DH_parameter.png` | Meca500 DH 파라미터 표 |
| `ref_Aruco_tvec_rvec2.png` | ArUco tvec/rvec 캘리브레이션 참조도 |

### 코드 (`codes/`)

- **`DH_angle_to_coordinate.ipynb`** — FK 핵심. FR5 관절각(deg) → 각 관절 3D 좌표.
  표준 DH 변환행렬 `get_transformation_matrix(dh, theta)` 체인. FR5 DH 값(m 단위) 하드코딩:
  `d1=0.152, a2=-0.425, a3=-0.395, d4=0.102, ...` — 다른 로봇에 쓰려면 DH 표만 교체.
- `robot_joint_regressor.ipynb` — RGB+depth+angle → 관절 pose 회귀 학습 (PyTorch). 산출물 `multi_label_regressor.pkl`
- `world_to_pixel_train.ipynb` — 3D→2D 픽셀 매핑 회귀 (XGBoost/RandomForest). 입력 `datasets/FR5_robot/joint`, `point_label/`
- `world_to_pixel_coordinate.ipynb` — solvePnP로 rvec/tvec 산출 후 3D→2D 투영 검증

### 다른 프로젝트와의 관계

- DINObotPose3의 운동학 솔버, DGIST의 `FR5_to_DREAM.ipynb`(FK로 GT 라벨 생성), Meca500 전처리가 모두 이 DH 파라미터 체계를 사용한다. DH 값의 출처/검증이 필요하면 여기의 img 문서가 기준.
