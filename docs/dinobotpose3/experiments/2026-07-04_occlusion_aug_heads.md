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

## 중간 결과 (Ep1, pose 스테이지 AUC, panda_synth_photo 200)
| 가림 | occ-aug head | 원본 head | Δ |
|---|---|---|---|
| 0% (클린) | 0.711 | 0.720 | −0.009 |
| 20% | 0.575 | 0.561 | +0.014 |
| 40% | 0.333 | 0.315 | +0.018 |

전형적 강건성/정확도 트레이드오프: 가림↑ 개선, 클린 −0.009 회귀. 클린 val angle MAE는 9.09°로 건강(원본과 유사)하나 pose AUC는 소폭 손해. **판정 방향**: 배포 SOTA(0% 평가)엔 원본 head 유지, occ-aug는 **별도 "가림 강건성 config"**로 — RoboPEPP 셀링포인트를 우리 방식으로. Ep1이라 학습 완료 시 재평가(클린 회귀 축소 가능성).

## 수렴 관측 (Ep13-17)
clean val angle MAE 수렴: occaug 8.79°, light 8.51° — **둘 다 배포 crop head(~9.09°)보다 낮음**(clean 정확도 회복). 단 Ep1 pose ADD는 −0.009 클린/+0.014~0.018 가림이었음(proxy≠eval-target). best 체크포인트로 완료 후 가림 벤치 전 구간 최종 A/B 예정.

## ✅ 최종 결과 (수렴 best, +DARK+cov, pose 스테이지 매칭 A/B, synth_photo 200)
| 가림 | base+DARK | occaug+DARK | Δ |
|---|---|---|---|
| 0% | 0.7532 | 0.7551 | **+0.0019 (do-no-harm!)** |
| 20% | 0.6100 | 0.6199 | **+0.0099** |
| 40% | 0.3757 | 0.3923 | **+0.0166** |

**판정: Pareto 승리 (트레이드오프 아님).** Ep1의 클린 회귀(−0.009)는 undertraining 아티팩트였고, **수렴까지 학습하니 0%에서 do-no-harm(+0.002)이면서 가림 구간 개선**. occaug pose 40%=0.392가 **RoboPEPP 40%(0.351)를 pose만으로 넘음**(이전 우리 base+RC 0.328보다도 높음). 채택 후보: (a) 가림 강건성 config, (b) self-train과 결합(occ-aug→self-train). rot occaug 페어와 함께. light 변형은 clean val MAE 더 낮음(8.47°) — 완료 후 비교.

## real 카메라 do-no-harm (pose 스테이지, 300f)
| 카메라 | occaug | 배포 head | Δ |
|---|---|---|---|
| azure (배포=crop 계열) | 0.8010 | 0.7993 | **+0.002 do-no-harm** |
| kinect (배포=self-train) | 0.6841 | 0.746 | **−0.06** (occaug는 self-train 부재) |

**해석**: occaug는 crop 계열 배포 카메라(azure)를 대체 가능하나, self-train head 배포 카메라(kinect/rs/orb)는 real self-train 부재로 뒤짐. **최선 배포 = occ-aug→self-train 스택**(가림 증강 후 카메라별 self-train)으로 둘 다 확보 — 향후 과제. 현재 채택: 가림 강건성 config + azure 대체 후보.

## 🔒 최종 가림 곡선 (occaug head +DARK+cov+RC, synth_photo 200)
| 가림 | occaug+RC | RoboPEPP | 이전(base+RC) | Δ vs RoboPEPP |
|---|---|---|---|---|
| 0% | **0.810** | 0.795 | 0.775 | +0.015 |
| 10% | **0.766** | 0.730 | 0.726 | +0.036 |
| 20% | **0.675** | 0.600 | 0.626 | +0.075 |
| 30% | **0.572** | 0.470 | 0.525 | +0.102 |
| 40% | **0.405** | 0.351 | 0.328 | +0.054 |

**전 가림 구간 RoboPEPP 초과 (0% 포함).** 이전 곡선이 지던 0%(−0.020)·40%(−0.023)를 occaug head가 둘 다 뒤집음. 가림 강건성이 이제 RoboPEPP 대비 명확한 우위 — 셀링포인트 역전.

## 🏆 light(약한 증강)가 최종 승자 — "증강은 적당히"
동일 조건 가림 벤치(pose, +DARK+cov): light가 occaug를 전 지점 이김.
| 가림 | base | occaug(강: ratio≤0.4+kp_drop) | **light(약: ratio≤0.3)** |
|---|---|---|---|
| 0% | 0.7532 | 0.7551 | **0.7584** |
| 20% | 0.6100 | 0.6199 | 0.6200 |
| 40% | 0.3757 | 0.3923 | **0.4199** |

**light+RC 최종 곡선** vs RoboPEPP:
| 가림 | 0% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| **light+RC** | **0.812** | **0.765** | **0.678** | **0.575** | **0.429** |
| RoboPEPP | 0.795 | 0.730 | 0.600 | 0.470 | 0.351 |
| Δ | +0.017 | +0.035 | +0.078 | +0.105 | +0.078 |

**교훈**: 가림 증강은 **강도가 과하면 역효과**. 강한 aug(occaug, ratio≤0.4 + kp_drop 0.15)는 과도 교란으로 클린·가림 둘 다 손해; 약한 aug(light, ratio≤0.3, kp_drop 없음, warm-start)가 클린 최고(base보다도 +0.005)+가림 최고. **light를 채택**(순수 업그레이드, 클린 do-no-harm 초과). "더 많은 증강=더 강건"이 아님.
