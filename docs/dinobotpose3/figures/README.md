# 논문용 그림 (DREAM SOTA)

`make_figs.py` 실행으로 재생성 (env `dino`, matplotlib). 각 그림은 300-DPI PNG + 벡터 PDF 동시 출력. 이미지 자체는 gitignore(재생성 가능) — 소스(`make_figs.py`, `table_dream.tex`)만 추적.

```bash
python docs/dinobotpose3/figures/make_figs.py
```

| 파일 | 내용 | 출처 데이터 |
|---|---|---|
| `fig1_scorecard` | 카메라별 ADD-AUC 막대(Ours/RoboPEPP/RoboTAG) + MEAN, orb auto-bbox 붕괴(RoboPEPP 0.344) 주석 | [FINAL_MODEL.md](../FINAL_MODEL.md) 재잠금 테이블 |
| `fig2_occlusion` | 가림 강건성 곡선 0~40% (RoboPEPP Fig.6 프로토콜), 전 구간 초과 | [experiments/2026-07-05](../experiments/2026-07-05_occaug_selftrain_stack.md), SUMMARY.md |
| `fig3_relock` | 800→1000 재잠금 안정성(Δmean +0.0002), RoboPEPP mean 기준선 | [experiments/2026-07-05](../experiments/2026-07-05_occaug_selftrain_stack.md) §재잠금 |
| `fig4_table` | 결과 표 렌더 이미지 (발표 슬라이드용) | 위와 동일 |
| `table_dream.tex` | LaTeX 표 (논문 본문용, booktabs) | 위와 동일 |

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
