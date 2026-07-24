# DINObotPose3 canonical checkpoints

로봇별 대표 체크포인트를 한곳에서 찾기 위한 인덱스다. 모든 `.pth`는 기존
`outputs_*` 파일을 가리키는 상대 심볼릭 링크이므로 가중치를 복제하지 않으며,
기존 Eval 스크립트의 경로도 유지된다.

## Panda

`panda/`가 논문의 synthetic-to-real 단일 Panda 모델이다.

- `detector_full.pth`: full-frame 2D 검출기
- `detector_crop.pth`: self-bbox 이후 crop 2D 검출기
- `angle_stage1.pth`, `rotation_stage1.pth`: full-frame 초기 pose head
- `angle_final.pth`: synthetic occlusion-augmentation 최종 angle head
- `rotation_final.pth`: synthetic occlusion-augmentation 최종 rotation head

네 DREAM 실사 카메라 모두 동일한 `angle_final.pth`와
`rotation_final.pth`를 사용하고, 카메라별 intrinsic `K`만 solver에 전달한다.
카메라별 pseudo-label self-training 모델은 primary 모델이 아니므로
`../checkpoints_experimental/panda_camera_adapted/`에 분리했다.

## FR3

`fr3/`에는 공통 detector/rotation과 진단 과정에서 생성된 angle 변형을 보존한다.
FR3는 7-DOF 단안 모호성 때문에 하나의 확정 배포 angle head가 잠기지 않았다.

- `angle_baseline.pth`: 초기 session-split MLP
- `angle_clean.pth`: fresh/head-direct 계열
- `angle_grounded.pth`: geometry-grounded 변형
- `angle_random_split.pth`: random-frame 진단용이며 일반화 성적으로 사용 금지
- `angle_transformer.pth`: transformer 진단 변형

## FR5

`fr5/`의 `detector.pth`, `angle.pth`, `rotation.pth`가 cross-session
head-direct 최종 조합이다.

## Meca500

- `meca500/real/`: 실사 학습 최종 조합
- `meca500/synthetic/`: 합성 in-domain/진단 조합. 실사 배포용이 아님

## KUKA

`kuka/`는 DREAM KUKA synthetic 7-keypoint 조합이다.

## Baxter

`baxter/fullbody_17kp/`는 whole-body 17-keypoint 조합이다. 과거 left-arm
7-keypoint 모델과 혼동하지 않는다.

## 평가 원칙

- Panda primary: 공통 synthetic head를 네 카메라에 동일 적용
- 카메라별 차이: 데이터와 intrinsic `K`
- 카메라별 self-training head: adaptation ablation으로만 표기
- 심볼릭 링크 대상을 이동하거나 삭제하면 이 인덱스도 깨지므로 모델 정리 시
  `find TRAIN/checkpoints* -type l ! -exec test -e {} \; -print`로 확인
