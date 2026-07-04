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
| realsense | (RC 계산 중, pose 0.759) | 0.8213 | — |
| orb | 학습 중 | 0.7714 | — |

### 가림 벤치 (pose, synth_photo 200) — 강건성 유지 확인
| 가림 | rs stack | kinect stack | light(순수) | base |
|---|---|---|---|---|
| 0% | 0.759 | 0.749 | 0.758 | 0.753 |
| 20% | 0.625 | 0.614 | 0.620 | 0.610 |
| 40% | 0.396 | 0.393 | 0.420 | 0.376 |

**판정 (잠정): 스택 성공.** kinect 배포 +0.017(real 적응 회복)이면서 가림 40% 0.393 > base(0.376) > RoboPEPP(0.351) 유지. realsense 스택은 가림 강건성을 **거의 완전 유지**(0%/20% light 동급 이상). self-train이 강건성을 일부 씻지만(40% light 0.420→스택 0.393-0.396), `--occlude-aug 0.3`이 base 이상으로 지킴. **real 적응 + 가림 강건성 동시 확보** — 세션 목표 실현. (강건성 완전 유지엔 occlude-aug 강도↑ 추가 실험 여지)
