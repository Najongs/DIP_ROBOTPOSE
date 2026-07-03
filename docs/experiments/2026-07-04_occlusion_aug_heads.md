# 2026-07-04 — 가림-증강 head fine-tune (T1 angle / T2 rot) 🔄

## 가설
40% 가림에서 pose 스테이지 붕괴(0.315)의 원인 = angle/rot head가 **가림 입력(퍼진 히트맵, 낮은 conf, 튄 키포인트)을 학습 때 본 적이 없음**. RoboPEPP의 가림 강건성 원천이 학습 시 관절 마스킹이므로, head 레벨에 같은 처방. **백본·detector 동결 유지**(sub-pixel 반증 우회) — 6월의 "detector 가림 재학습 불필요" 판정과도 별개(그건 detector, 이건 head).

## 설정 (둘 다 warm-start fine-tune, lr 2e-4)
| | T1 angle head (GPU4) | T2 crop rot head (GPU0) |
|---|---|---|
| 스크립트 | `train_angle.py` (+`--occlude-aug --kp-drop --init-head` 신규) | `train_rotation.py` (+`--occlude-aug --init-head` 신규) |
| 증강 | occluder 페이스트 p=0.5, ratio U(0.05,0.4) + kp_drop 0.15 | occluder 페이스트 p=0.5, ratio U(0.05,0.4) |
| warm-start | angle_crop_20260605_174740 | rot_crop_20260606_022535 |
| 에폭 | 20 | 15 |
| 출력 | `outputs_angle/angle_occaug_*` | `outputs_rotation/rot_crop_occaug_*` |

증강 구현: `Eval/occl_util.py::paste_random_occluders_` (detector 동결 상태에서 이미지에 페이스트 → head가 자연스럽게 열화된 conf/kp 분포를 학습).

## 검증 계획
1. 가림 벤치 30-40%에서 새 head 페어로 pose/+RC 재측정 (기준: pose 0.481/0.315)
2. 클린 do-no-harm: 0% 벤치 + realsense held-out
3. 통과 시 멀티스타트 RC와 스택

## 결과
(학습 완료 후 기입)
