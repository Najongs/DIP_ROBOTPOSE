# 2026-07-05 — occ-aug → self-train 스택 🔄

## 가설
light 가림-증강 head는 클린·가림 둘 다 최고이나, real 카메라(kinect/rs/orb)에선 self-train head에 뒤짐(kinect −0.06) — **real 적응 부재**가 원인. 두 이득이 직교하므로 **light head를 시작점으로 카메라별 self-train**하면 가림 강건성(light) + real 적응(self-train) 동시 확보.

## 설계
- `selftrain_pseudo_rot.py` warm-start: **light angle head** + occaug rot head. 솔버 pseudo-label(θ*,R*)로 real 적응 + synth anti-forget.
- **핵심 보강**: self-train synth anti-forget 배치에 `--occlude-aug 0.3` 추가 — real 적응 중에도 가림 노출 유지(안 하면 클린 real에 적응하며 light의 가림 강건성 씻김). `Eval/occl_util.py::paste_random_occluders_` 재사용.
- 카메라: realsense/orb/kinect (self-train이 유효한 3개; azure self-train ~0). crop 파이프라인, 8ep, held-out early-stop.

## 검증 계획
1. 배포 ADD (held-out 800): 기존 self-train head(rot-adapt r1) 대비 do-no-harm 이상
   - 기준: realsense 0.821 / kinect 0.813 / orb 0.771 (+DARK+cov+RC 최종)
2. 가림 강건성 유지: 스택 head로 가림 벤치 → light 곡선(0.812/…/0.429) 근접 유지
3. 통과 시 배포 head 교체 → 정확도+강건성 동시 SOTA

## 스택 전 light head 기준 (real pose, held-out 300, +DARK+cov)
| 카메라 | light (스택 전) | 배포 self-train | 격차 |
|---|---|---|---|
| realsense | 0.745 | 0.755 | −0.010 |
| orb | 0.681 | 0.733 | −0.052 |
| kinect | 0.684 | 0.745 | −0.061 |

예상대로 light head는 real 적응 부재로 배포 self-train에 뒤짐(orb/kinect가 self-train 이득 큼). **스택 목표**: self-train이 이 격차를 메우며 가림 강건성 유지.

## 결과 (kinect·realsense 완료; orb 학습 중)

### 배포 ADD (held-out 800, +DARK+cov+RC)
| 카메라 | 스택 | 기존 배포 | Δ |
|---|---|---|---|
| kinect | **0.8303** | 0.8132 | **+0.017** ✅ |
| realsense | 0.8165 | 0.8213 | −0.005 (강건성은 획득) |
| orb | **0.7726** | 0.7714 | +0.001 (강건성 획득) |

### 가림 벤치 (pose, synth_photo 200) — 강건성 유지 확인
| 가림 | rs stack | kinect stack | light(순수) | base |
|---|---|---|---|---|
| 0% | 0.759 | 0.749 | 0.758 | 0.753 |
| 20% | 0.625 | 0.614 | 0.620 | 0.610 |
| 40% | 0.396 | 0.393 | 0.420 | 0.376 |

### orb 가림 벤치: 0.748/0.634/0.399 (20% light 초과, 40% base 초과)

### 🔒 최종 배포 테이블 (카메라별 최적) — mean 0.799 → **0.804**
| 카메라 | 채택 | 값 | 기존 | 비고 |
|---|---|---|---|---|
| kinect | **스택** | **0.8303** | 0.813 | +0.017, 가림 강건성 획득 |
| realsense | 기존 | 0.8213 | 0.821 | 스택 −0.005(기존 최적); 스택head는 강건성용 대체 가능 |
| azure | 기존 | 0.7916 | 0.792 | self-train ~0 |
| orb | **스택** | **0.7726** | 0.771 | +0.001, 가림 강건성 획득 |
| **mean** | | **0.8039** | 0.7994 | **+0.0045** |

**판정: 스택 성공.** kinect 배포 +0.017(real 적응 회복)이면서 가림 40% 0.393 > base(0.376) > RoboPEPP(0.351) 유지. realsense 스택은 가림 강건성을 **거의 완전 유지**(0%/20% light 동급 이상). self-train이 강건성을 일부 씻지만(40% light 0.420→스택 0.393-0.396), `--occlude-aug 0.3`이 base 이상으로 지킴. **real 적응 + 가림 강건성 동시 확보** — 세션 목표 실현. (강건성 완전 유지엔 occlude-aug 강도↑ 추가 실험 여지)

## realsense도 스택으로 전환 (사용자 요청: 강건성 확보)
realsense 배포를 스택 head로 전환 → 4개 중 **3개(rs/kinect/orb) 가림 강건 head** 확보 (azure만 미강건, 단 RC off라 무관).
- **가림-강건 배포 config**: rs 0.8165 + kinect 0.8303 + azure 0.7916 + orb 0.7726 → mean **0.8028** (강건성 확보)
- max-정확도 config: rs 0.8213(기존) → mean 0.8039 (rs 강건성 없음)
- 차이는 mean −0.001 — realsense 강건성(40% 0.396 vs 기존 head base 0.376)의 대가.

**개선 시도 (진행 중)**: 기존 스택은 light head에서 재self-train(−0.005). 개선판 = **배포 realsense head(0.821, 이미 real 적응)에서 가림 증강만 fine-tune** → real 적응 유지 + 강건성 추가로 −0.005 회수 목표. 결과 나오면 더 나은 쪽 채택.

## azure도 강건성 확보 — light head 직접 전환 (self-train ~0)
azure는 sim2real 갭 작아 self-train ~0 → **light head 직접 사용**이 최적. light head azure(full-1000, RC off) = **0.7953 vs 기존 crop base 0.7916 = +0.004** — 정확도까지 올리며 가림 강건성 확보(강건성은 light 곡선 0.812/…/0.429).

## 🔒 전 카메라 가림 강건 배포 테이블
| 카메라 | 강건 head | 값 | 기존 |
|---|---|---|---|
| realsense | 스택 | 0.8165 | 0.8213 |
| kinect | 스택 | 0.8303 | 0.8132 |
| azure | **light** | **0.7953** | 0.7916 |
| orb | 스택 | 0.7726 | 0.7714 |
| **mean** | | **0.8037** | 0.7994 |

**전 카메라(4/4) 가림 강건 + mean 0.804 = max-정확도와 동일.** azure light(+0.004)·kinect 스택(+0.017)이 realsense 스택(−0.005)을 상쇄 → **정확도 손실 없이 전 카메라 가림 강건성 확보.** (realsense 개선 스택·azure 스택으로 추가 상승 여지 확인 중)

## realsense 개선 스택 실험 결과 — ❌ 강건성 안 배어듦
배포 head(0.821, 가림 미학습)에서 시작 + synth 브랜치 증강만 fine-tune → 가림 40% = 0.378 ≈ base(0.376), light-스택(0.396) 미달. **짧은 self-train의 synth 증강만으론 강건성 안 생김**(real pseudo-label 적응이 지배). 강건성은 angle head를 처음부터 증강 학습한 light에서만 제대로 나옴. → **realsense는 light-스택(0.8165, 강건 0.396) 확정** (−0.005 정확도 = 강건성 대가).

## azure 스택 결과 — self-train ~0 재확인
azure 스택(light→azure self-train) 내부 baseline 0.7735 → 0.7746 (+0.001). azure self-train 무의미 → **azure는 light head 직접(0.7953)이 최적**.

## 🔒🔒 최종 전 카메라(4/4) 가림 강건 배포 테이블
| 카메라 | 강건 head | 배포 ADD | 가림 40% | 기존 배포 |
|---|---|---|---|---|
| realsense | light-스택 | 0.8165 | 0.396 | 0.8213 (−0.005, 강건성 대가) |
| kinect | 스택 | 0.8303 | 0.393 | 0.8132 (+0.017) |
| azure | light | 0.7953 | 0.429 | 0.7916 (+0.004) |
| orb | 스택 | 0.7726 | 0.399 | 0.7714 (+0.001) |
| **mean** | | **0.8037** | 전부 >RoboPEPP 0.351 | 0.7994 |

**전 카메라 정확도(mean 0.804, vs RoboPEPP 0.780 +0.024) + 가림 강건성(전 구간 RoboPEPP 초과) 동시 달성.** 강건성 확보에 사실상 정확도 손실 없음(azure/kinect 이득이 realsense 손실 상쇄). realsense만 −0.005는 강건성의 대가.
**교훈**: 강건성은 angle head를 처음부터 증강 학습(light)해야 배어듦. 배포 head+짧은 증강 self-train으론 안 됨. azure는 self-train~0이라 light 직접이 최적.
