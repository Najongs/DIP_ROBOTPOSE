# Robot Pose Visualization Scripts

이 폴더에는 훈련된 DINOv3 모델을 사용하여 각 로봇의 포즈를 시각화하는 스크립트가 포함되어 있습니다.

## 파일 구성

### Python 스크립트
- `visualize_fr5.py` - FR5 로봇 시각화
- `visualize_franka_research3.py` - Franka Research 3 로봇 시각화
- `visualize_meca500.py` - Meca500 로봇 시각화
- `visualize_meca_insertion.py` - Meca Insertion 로봇 시각화

### 실행 스크립트
- `Visualization.sh` - 모든 로봇 시각화 일괄 실행 스크립트
- `RUN_VISUALIZATION.md` - 실행 가이드

## 시각화 내용

각 스크립트는 다음을 시각화합니다:

1. **히트맵 기반 예측 (녹색)**: 모델이 출력한 히트맵에서 추출한 2D 키포인트
   - 녹색 점과 선으로 표시
   - 라벨: `H0`, `H1`, `H2`, ...

2. **FK 기반 예측 (마젠타)**: 예측된 관절 각도로 순방향 기구학(FK)을 계산하고 카메라 좌표계로 투영한 3D 키포인트
   - 마젠타 점과 선으로 표시
   - 라벨: `J0`, `J1`, `J2`, ...

## 사용법

### 🚀 빠른 시작 (권장)

**모든 로봇 시각화를 한 번에 실행하고 결과를 자동 저장:**

```bash
cd visualization
./Visualization.sh
```

결과는 `results/` 폴더에 타임스탬프와 함께 저장됩니다.

특정 체크포인트 사용:
```bash
./Visualization.sh /path/to/checkpoint.pth
```

자세한 사용법은 `RUN_VISUALIZATION.md`를 참조하세요.

---

### 개별 실행

각 로봇을 개별적으로 시각화하려면:

```bash
# Fr5 로봇 시각화
cd visualization
python visualize_fr5.py

# Franka Research 3 로봇 시각화
python visualize_franka_research3.py

# Meca500 로봇 시각화
python visualize_meca500.py

# Meca Insertion 로봇 시각화
python visualize_meca_insertion.py
```

### 옵션

각 스크립트는 다음 옵션을 지원합니다:

```bash
# 특정 체크포인트 사용
python visualize_fr5.py --checkpoint /path/to/checkpoint.pth

# 결과를 파일로 저장
python visualize_fr5.py --output results/fr5_visualization.png

# Meca500의 경우 샘플 개수 지정 가능
python visualize_meca500.py --num_samples 9
```

## 체크포인트 경로

기본 체크포인트 경로: `/home/najo/NAS/DIP/4_perception/DINOv3_fine_tunning/checkpoints_total_dino_conv_only/best_model.pth`

다른 체크포인트를 사용하려면 `--checkpoint` 옵션을 사용하세요.

## 필요 패키지

```
torch
torchvision
opencv-python
numpy
pandas
matplotlib
scipy
pillow
```

## 로봇별 특징

### FR5
- **관절 개수**: 6개
- **카메라 뷰**: left, right, top
- **카메라 타입**: leftcam, rightcam (각 뷰마다 2개)
- **DH 파라미터**: Standard DH
- **뷰별 회전**: 각 뷰마다 다른 베이스 회전 적용

### Franka Research 3
- **관절 개수**: 7개 (+ 1개 finger joint)
- **카메라 뷰**: view1, view2, view3, view4
- **포즈**: pose1, pose2 (각각 다른 ArUco 보정 사용)
- **DH 파라미터**: Modified DH
- **특이사항**: 일부 관절 제외 (exclude_indices = {1, 5})

### Meca500
- **관절 개수**: 6개
- **카메라**: 단일 카메라 (고정)
- **DH 파라미터**: Standard DH
- **샘플링**: 랜덤 샘플링으로 여러 이미지 시각화

### Meca Insertion
- **관절 개수**: 6개
- **카메라 뷰**: left, right, top
- **카메라 타입**: leftcam, rightcam (각 뷰마다 2개)
- **DH 파라미터**: Standard DH
- **베이스 보정**: X축 180도 + Z축 90도 회전

## 문제 해결

### CUDA out of memory
- CPU 모드로 실행: 스크립트에서 `device = 'cpu'`로 수정

### 이미지를 찾을 수 없음
- CSV 파일의 `image_path` 경로 확인
- 상대 경로가 올바른지 확인

### ArUco 데이터 누락
- ArUco JSON 파일이 올바른 경로에 있는지 확인
- JSON 파일에 해당 view/cam 조합이 있는지 확인

## 출력 예시

스크립트는 matplotlib 창을 열거나 (`--output` 미지정 시) 이미지 파일로 저장합니다.

각 subplot에는:
- 원본 이미지 (왜곡 보정 적용)
- 녹색: 히트맵 기반 예측
- 마젠타: FK 기반 예측
- 제목: 뷰/카메라 정보

## 참고사항

- 모든 이미지는 카메라 왜곡이 보정된 상태로 표시됩니다
- FK 기반 예측은 ArUco 마커 보정을 사용하여 로봇 좌표계를 카메라 좌표계로 변환합니다
- 히트맵 예측은 모델이 직접 출력한 2D 좌표입니다
