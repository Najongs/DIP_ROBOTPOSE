# 2026-07-04 — DINO feature-metric render-and-compare (서베이 2R Idea 1) 🔄

## 가설
계획했던 photometric RGB RC는 틀린 버전 — 최신 흐름(MCLoc, AlignPose)은 렌더/실사 비교를 **frozen ViT 특징 공간**에서 수행해 albedo/조명 도메인 갭을 흡수. 우리는 frozen DINOv3 + nvdiffrast 보유 → 붙이면 끝, 학습 불필요. **실루엣 RC가 해로웠던 azure(근거리)를 특징 RC가 이길 수 있는가?**

## 프로브 결과 (go/no-go, azure n=80, GT vs 섭동 판별)
법선-셰이딩 렌더(`render_nvdr.render_shaded`) → DINOv3 패치특징 → 로봇 패치 코사인. edge-NCC(rgb_rc_probe)와 대조:
| 섭동 | feat GT-승률 | edge GT-승률 |
|---|---|---|
| yaw±5 | 91/96% | 85/89% |
| yaw+15 | 100% | 91% |
| depth±5% | 100% | 88/94% |
| J1+10 / J3+10 | 100/95% | 95/85% |
| J4+15 (손목) | 70% | 75% |

**판정: feature-metric이 edge를 전 구간(J4 제외) 압도**, 특히 미세 섭동(yaw±5=refine 구간)에서. margin 단조 증가 → GT가 최적점. **미분 RC 빌드 확정.**

## 구현 (진행 중)
`rc_refine_from_dump.py --feat-w`: RC 내부 루프에서 render_shaded → DINOv3 forward(grad) → (1−masked cosine) 항. azure는 `--no-sil --feat-w`(실루엣 없이 특징+재투영 앵커)로 — 현재 RC OFF인 카메라라 순수 업사이드.

## 결과
(구현·평가 후 기입)
