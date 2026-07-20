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

| 멀티로봇 DREAM 포즈 (KUKA / Baxter) | 07-09~13 | ✅ 검출기 0.735/0.817, 포즈 ADD 0.34/0.25(synth, direct-pose). 천장: KUKA rot-t, Baxter wrist(관측성). RC 양쪽 막힘. wrist appearance(mlp_patch) 재학습 중 | [2026-07-10_multirobot_dream_detectors.md](2026-07-10_multirobot_dream_detectors.md) |

| 논문 포지셔닝 재구성 + 파이프라인 피규어 | 07-20 | 📝 세션(측정 없음) — 2축 스토리(frozen 기여 승격+PCK/ADD 해리 본문 반영, 무료 레버 격하), fig_pipeline 제작, SAM=v1 확인 | [2026-07-20_paper_positioning.md](2026-07-20_paper_positioning.md) |

| 비평가↔보완자 에이전트 토론 (사전 리뷰) | 07-20 | 🔴 **C8 발견**: 표 6 self-train 절제가 rot 자가학습을 유지 → zero-real-adaptation 수치 미측정(임계 실험 `zero_adapt` 실행) + P0 문구 수정 일괄 반영 | [2026-07-20_critic_debate.md](2026-07-20_critic_debate.md) |

| PAPER_OVERLEAF.tex 논문 조립 | 07-21 | 📝 오버리프 초안 완성 — 전 섹션 작성·무인칭·×100·통합 표(로봇별+Real/Synth)·수식 코드검증·스타일 규칙 확정. 남은 것: 그림·\ref·RoboPEPP 재현 | [2026-07-21_overleaf_paper_assembly.md](2026-07-21_overleaf_paper_assembly.md) |

관련 문서: [SOTA 서베이](../references/sota_survey.md) · [다음 시도 로드맵](../references/next_directions.md) · 반증 맵은 로드맵 §3
