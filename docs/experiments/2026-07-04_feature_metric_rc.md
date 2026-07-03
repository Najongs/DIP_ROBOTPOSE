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

## ⚠️ 중간 교훈 (edge-NCC를 목적함수로): 판별력 ≠ 최적화가능성
edge-NCC struct-RC를 azure 최적화 목적으로 쓰니 −0.10(w=0.1)~−0.18(w=0.5), nan mean(발산). 프로브에서 GT를 이겼지만(판별) gradient 최적화는 실패 — image gradient는 고주파라 landscape가 노이지, 국소최적. **이게 문헌이 특징 공간을 쓰는 이유**(semantic·저주파·넓은 basin). 특징 항은 재투영 앵커 강하게 유지하며 검증 중.

## 결과 1 — azure 순수 feature-RC (--no-sil): 여지 없음
feat_w 스윕: 1.0→−0.043, 0.3→−0.020, 0.1→−0.004 (단조 수렴, 발산 없음). edge-NCC(발산)와 달리 **잘 조건화됨**(저가중 do-no-harm). 그러나 azure는 순손실 — **재해석: azure는 test-time refinement 여지가 없는 카메라**(이미 0.783=RoboPEPP+0.035, 남은 오차는 상류 2D). 배포는 이미 azure RC OFF이라 회귀 아님. → 특징 항의 시험대를 여지 있는 곳(realsense 스택, 40% 가림)으로 이동.

## 결과 2 — realsense 실루엣+특징 매칭 A/B (150f): 특징 기여 ≈0
| 구성 | +RC Δ |
|---|---|
| 실루엣 단독 | +0.0645 |
| 실루엣 + 특징(0.3) | +0.0665 |
→ 특징 +0.002(노이즈). realsense 문제는 depth인데 **실루엣 면적이 이미 그 신호를 포화** → 특징 무기여. feature-metric RC가 우리 격차엔 부적합으로 수렴: 실루엣 작동처(rs/orb)=신호 포화, 비작동처(azure)=여지 없음. 남은 유일한 여지 = 40% 가림(포즈 붕괴, 실루엣과 직교하는 구조신호 가능성) — 결과 대기.

## 결과 3 — 40% 가림 실루엣+특징: +0.018 (상류 붕괴가 지배)
mean 27000mm(θ 붕괴 프레임), AUC 0.333. 실루엣-단독 대비 미미. 40%는 상류 θ 붕괴가 지배 → RC로 못 고침(멀티스타트 반증과 일치). **40%의 정답은 가림-증강 head(T1/T2)**.

## 🔚 판정 — feature-metric RC 우리 격차엔 부적합 (잘 조건화된 do-no-harm 항으로 코드 유지)
문헌 1순위였으나 실측: azure 여지 없음(−0.004~−0.043), realsense 특징 기여 +0.002(실루엣이 depth 신호 포화), 40% 상류 붕괴 지배. **핵심 이유**: 우리 실루엣 RC가 이미 depth/scale 신호를 포착하므로 특징이 더할 직교 신호가 없음(사람 포즈/BOP는 실루엣 RC가 없어 특징이 큰 이득이지만, 우리는 이미 보유). `--feat-w` 플래그 유지(발산 없음, 향후 실루엣 없는 세팅용). **다음 레버 = 가림-증강 head(근본 원인 직접 공격)로 확정.**
