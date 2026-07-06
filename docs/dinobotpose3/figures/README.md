# 논문용 그림 (DREAM SOTA)

`make_figs.py` 실행으로 재생성 (env `dino`, matplotlib). 각 그림은 300-DPI PNG + 벡터 PDF 동시 출력. 이미지 자체는 gitignore(재생성 가능) — 소스(`make_figs.py`, `table_dream.tex`)만 추적.

```bash
python docs/dinobotpose3/figures/make_figs.py
```

| 파일 | 내용 | 출처 데이터 |
|---|---|---|
**결과 (무엇을 달성했나):**
| `fig1_scorecard` | 카메라별 ADD-AUC 막대(Ours/RoboPEPP/RoboTAG) + MEAN, orb auto-bbox 붕괴(RoboPEPP 0.344) 주석 | [FINAL_MODEL.md](../FINAL_MODEL.md) 재잠금 테이블 |
| `fig2_occlusion` | 가림 강건성 곡선 0~40% (RoboPEPP Fig.6 프로토콜), 전 구간 초과 | [experiments/2026-07-05](../experiments/2026-07-05_occaug_selftrain_stack.md), SUMMARY.md |
| `fig3_relock` | 800→1000 재잠금 안정성(Δmean +0.0002), RoboPEPP mean 기준선 | [experiments/2026-07-05](../experiments/2026-07-05_occaug_selftrain_stack.md) §재잠금 |
| `fig4_table` | 결과 표 렌더 이미지 (발표 슬라이드용) | 위와 동일 |
| `table_dream.tex` | LaTeX 표 (논문 본문용, booktabs) | 위와 동일 |

**어트리뷰션 (왜 좋아졌나):**
| `fig5_lever_decomp` | 카메라별 레버 분해 — base(솔버+cov-PnP+DARK+head) + RC 세그먼트. **RC가 원거리 엔진**(+0.070/0.060/0.040), azure는 RC off. **재잠금에서 직접 측정** | 1000-프레임 재잠금 dump vs +RC (직접 측정) |
| `fig6_milestones` | 세션 진행 마일스톤 mean(RoboPEPP 0.780→render-compare 0.796→+DARK 0.799→+stack 0.804), 전부 학습 불필요(self-train 제외) | [00_overview.md](../00_overview.md) 채택 레버, dark_decode 실험 |
| `fig7_occ_mechanism` | 가림 강건성의 출처 — clean head vs occ-aug light vs 배포 스택, 40%에서 light 0.420 vs base 0.376(+0.044). 처음부터 증강 학습해야 배어듦 | [experiments/2026-07-05](../experiments/2026-07-05_occaug_selftrain_stack.md), occlusion_aug_heads |

## 핵심 수치 (2026-07-06 1000-프레임 재잠금)

| cam | Ours | RoboPEPP | RoboTAG |
|---|---|---|---|
| realsense | 0.8153 | 0.805 | 0.783 |
| kinect360 | 0.8275 | 0.785 | 0.757 |
| azure | 0.7945 | 0.753 | 0.831 |
| orb | 0.7784 | 0.775 | 0.588 |
| **mean** | **0.8039** | 0.780 | 0.740 |

가림 곡선(0~40%): ours 0.812/0.765/0.678/0.575/0.429 vs RoboPEPP 0.795/0.730/0.600/0.470/0.351.

프로토콜 주: Ours는 predicted angles + 완전 자동 bbox(bbox-from-solved). 베이스라인은 발표 수치(대부분 GT-bbox). orb는 동일 auto-bbox에서 RoboPEPP가 0.344로 붕괴 — fig1 주석 참조.
