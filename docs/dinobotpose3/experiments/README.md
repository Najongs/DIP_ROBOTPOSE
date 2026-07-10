# DINObotPose3 실험 기록 (curated)

> **먼저 읽을 것: [00_overview.md](../00_overview.md)** — 전체 진행 종합(성적표·채택/반증 맵·진행 중·로드맵).

> 원본 실험 일지는 `3_pose_models/DINObotPose3/EXPERIMENTS.md` (append-only, 전체 이력).
> 이 폴더는 **실험 단위의 정리본** — 목적/설정/결과/판정을 실험별 파일로. 새 실험은 `YYYY-MM-DD_이름.md`로 추가하고 이 표를 갱신.

| 실험 | 날짜 | 판정 | 파일 |
|---|---|---|---|
| nvdiffrast+SAM render-and-compare → real 4-split SOTA | 07-03 | ✅ mean **0.796** vs RoboPEPP 0.780 | [2026-07-03_render_compare_sota.md](2026-07-03_render_compare_sota.md) |
| 가림 벤치(RoboPEPP 프로토콜) + 레버 3종 ablation | 07-03 | cov-PnP ✅ / 로버스트IoU ❌ / 모집단prior ❌ | [2026-07-03_occlusion_track.md](2026-07-03_occlusion_track.md) |
| 멀티스타트 RC + SAM-IoU basin 선택 | 07-04 | ❌ 반증 — 잔여 실패는 R-basin 아님 (orb→2D, 40%→θ붕괴) | [2026-07-04_multistart_rc.md](2026-07-04_multistart_rc.md) |
| DARK sub-pixel 디코딩 (2R Idea 3) | 07-04 | ✅ 채택 — 전 카메라 pose +0.005~0.007, 무료 | [2026-07-04_dark_decode.md](2026-07-04_dark_decode.md) |
| 가림-증강 head fine-tune (T1 angle / T2 rot) | 07-04 | 🔄 학습 중 (GPU4/GPU0) | [2026-07-04_occlusion_aug_heads.md](2026-07-04_occlusion_aug_heads.md) |

| RoboTAG식 reproj-consistency angle head | 07-04 | 🔄 학습 중 (azure 격차) | [2026-07-04_robotag_reproj_consistency.md](2026-07-04_robotag_reproj_consistency.md) |

| occ-aug → self-train 스택 | 07-05 | ✅ 전 4카메라 가림 강건 + mean 0.804 (손실 0) | [2026-07-05_occaug_selftrain_stack.md](2026-07-05_occaug_selftrain_stack.md) |

| 멀티로봇 DREAM 검출기 (KUKA / Baxter) | 07-09~10 | ✅ KUKA synth AUC 0.735 / Baxter 0.817 (L2 tail=link혼동, 솔버 복구 예상) | [2026-07-10_multirobot_dream_detectors.md](2026-07-10_multirobot_dream_detectors.md) |

관련 문서: [SOTA 서베이](../references/sota_survey.md) · [다음 시도 로드맵](../references/next_directions.md) · 반증 맵은 로드맵 §3
