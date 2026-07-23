# 논문 서사 수정 초안 — 2026-07-22 intrinsics 수정 후속

> **성격: 이력 문서. 대부분 반영 완료.** 아래 항목은 전부 **데이터가 반박하는 주장**이며, 이 파일은 각 위치별 "현재 문장 / 왜 틀렸는지 / 대안"의 이력으로 보존한다(삭제 금지).
>
> 근거 문서: [experiments/2026-07-22_intrinsics_rootcause.md](experiments/2026-07-22_intrinsics_rootcause.md)

## 🟢 적용 상태 (2026-07-22 — 저자가 B-2/B-4를 대안 2로 결정 → 연쇄 항목 확정)

**결정**: "관측성 천장" 서사 삭제 → "link-identity 혼동으로 인한 파국적 꼬리(confident wrong pose라 신뢰도 기반 거부로 못 잡음)"로 교체.

| 항목 | 위치 | 채택 | 상태 |
|---|---|---|---|
| B-1 | `tex:245` 포지셔닝 | **대안 2**(결과를 주장으로 승격) | ✅ 적용 |
| B-2 | `tex:245` 관측성→link-identity | **대안 2**(한계 교체) | ✅ 적용 |
| B-3 | `tex:246` 한국어 대역 | (B-2에 동기화) | ✅ 적용 |
| B-4 | `tex:312` 결론부 | **대안 2**(진단을 기여로 재배치) | ✅ 적용 |
| B-5 | `tex:313` 한국어 대역 | (B-4에 동기화) | ✅ 적용 |
| B-6 | `PAPER_DRAFT.md:195` 경고 블록 | **대안 1**(경고 유지+근거 명시) | ✅ 적용 |
| B-7 | `PAPER_DRAFT.md:197/199` | **대안 1**(사실 교정) — ⚠️ **대안 2 폐기**(아래) | ✅ 적용 |
| B-9 | `PAPER_DRAFT.md:190/191` 표3 병목 열 | **대안 2**(병목 열 삭제→median ADD) | ✅ 적용 |
| B-10 | `make_figs_multirobot.py:51` fig7 주석 | **대안 1**(경고 문구 강화) | ✅ 적용 |
| B-8 | `PAPER_DRAFT.md:328/330` | (수치+경로명 최소 교정) | ✅ 적용(이전 라운드) |

> ⚠️ **B-7 대안 2 폐기 근거**: 대안 2는 "RC를 KUKA로 확장하는 것은 남은 헤드룸"이라 서술하나, 2026-07-22 KUKA RC가 **레버로서 닫혔다**(RC로 0.75 불가, R을 못 고침). 따라서 "남은 헤드룸"은 더 이상 참이 아니어서 **대안 1(RC 헤드룸 주장 없음)**을 채택했다.

### 연쇄로 함께 고친 곳 (B-2/B-4 결정의 직접 파생 — 승인 텍스트를 병렬 위치에 적용)

- `PAPER_DRAFT.md:326` 절 제목 "관측성 병목" → "잔여 실패모드"
- `PAPER_DRAFT.md:334/336` 비교 주의 블록: stale 0.34/0.25 → 0.69/0.71, 병목 지시를 link-identity 꼬리로
- `PAPER_DRAFT.md:338/340` 관측성 분석 문단: 관측을 **존치하되 2차 효과로 강등**, 지배 실패=link-identity 명시 (SUMMARY:116과 동일 처리)
- `PAPER_DRAFT.md:447/449` 결론: B-4 대안 2와 동일 서사로 교체

### ✅ 보류 3곳 — 저자 결정 반영 완료 (2026-07-22)

1. **`PAPER_DRAFT.md:451/453` 한계·향후과제 문단** — **삭제·수렴 완료.** (a) 거짓 메쉬 주장("공개 iiwa7 ~20mm 정합 불가") 삭제, (b) "RC를 KUKA로 확장" 프레이밍 삭제(KUKA RC=닫힌 레버), (c) 손목 관측성 천장 한계 삭제. 결과 = **B-4 대안 2의 생존 한계 두 개로 수렴**: (i) KUKA/Baxter 공개 실측 데이터 부재, (ii) RC의 per-camera 게이팅·앵커 의존(자유 실루엣 깊이 모호성 억제가 향후 과제). **RC 수치 없음.**
2. **`tex:34/41` 서론 기여 항목** — **교체 완료.** "per-robot bottlenecks such as an observability limit at the wrist joints" → "the same pipeline transfers to all three DREAM robots without per-robot redesign, an applicability study with an analysis of the dominant residual failure mode"(국문 대역 동일). 본문·결론과 일관 = 승인된 포지셔닝(로봇별 재설계 없는 전이).
3. **`tex:86`** — **유지(건드리지 않음).** "the final wrist rotation, unobservable from keypoints alone, is fixed to zero"는 참인 방법 설계 서술 = 손목 회전을 0으로 고정. 오류 아님.

---

> **이하 원본 초안(각 항목의 근거·대안 전문). 이력 보존.**

## 이 수정을 강제하는 확정 데이터

| | 이전(논문 게재) | 신규(솔버+참 K, 전체 셋) |
|---|---|---|
| KUKA-DR | 35.7 | **69.0** |
| KUKA-Photo | 31.9 | **69.8** |
| Baxter-DR | 25.2 | **71.3** (원자료 0.7125) |

> 📌 **반올림 확정 = 71.3 (half-up).** 71.25는 정확한 동률이라 더 정밀한 원값으로 동률을 없애려 했으나, **불가능하다**: `Eval/u1_solver_vs_direct.py:266`의 출력 포맷이 `{add_auc(a):>9.4f}`로 **소수 4자리에서 잘리고**, 해당 full-set 런은 `--dump`(per-frame npz) 없이 실행되어 로컬에 재계산할 원배열이 없다. ⇒ **다음 런에 `--dump`를 붙여 두면** 이런 동률을 사후에 깨끗이 해소할 수 있다.

경쟁 대비: KUKA 69.0 vs RoboPEPP 76.2 / RoboPose 80.2 / HoRoPose 75.1 / RoboTAG 75.0 → **사정권**. Baxter 71.3 vs 34.4 / 32.7 / 58.8 / 58.8 → **큰 격차 1위**.

**서사를 무너뜨리는 핵심 관측 3가지**:
1. **Baxter 파국 프레임 비율이 intrinsics 수정만으로 24.7% → 8.0%로 붕괴**했다. 손목 관측성이 지배 요인이었다면 불가능한 변화다.
2. **iiwa7 URDF+메쉬가 존재**하고 우리 FK와 0.04 mm 일치한다 — "정합 메쉬 부재"는 사실이 아니다.
3. 단, **KUKA의 파국 꼬리는 살아남았다**(fail 15.4% → 15.9%, p99 1012 mm). 즉 "한계는 없었다"가 아니라 **한계의 정체가 손목 관측성이 아니라 link-identity 혼동이었다**.

⚠️ 3번 때문에, 아래 대안들은 **"한계 주장을 전부 삭제"하는 방향이 아니라 "한계를 올바른 대상으로 교체"하는 방향**으로 설계했다.

---

## B-1. `PAPER_OVERLEAF.tex:245` — "not comparable / applicability study" 프레이밍

**현재 문장** (사실 수정 반영 후 상태):
> "…poses come from the kinematic solver, reaching ADD-AUCs of 69.0 for KUKA and 71.3 for Baxter; **these numbers are not comparable to the Panda real results and serve as an applicability study with a bottleneck analysis.**"

**왜 문제인가**: "비교 불가"는 두 가지를 뭉뚱그린다. (a) Panda **실측** 대 KUKA/Baxter **합성** 비교 — 이건 여전히 참이고 유지해야 한다. (b) KUKA/Baxter 수치를 **경쟁 방법의 같은 합성 스플릿 수치와** 비교하는 것 — 이건 표 `tab:main`이 이미 하고 있고, **정당하며**, 이제 우리에게 유리하다. 현재 문장은 (a)를 말하면서 (b)까지 차단해 **Baxter 1위·KUKA 사정권이라는 결과를 스스로 무효화**한다. "applicability study"는 25.2일 때는 적절한 겸양이었으나 71.3에서는 **과소주장**이다.

**대안 1 (보수적 — 비교축만 분리)**
> "…reaching ADD-AUCs of 69.0 for KUKA and 71.3 for Baxter. These synthetic results are not directly comparable to the Panda real numbers, but they are measured on the same synthetic splits the compared methods report, where the proposed method is competitive on KUKA and leads on Baxter."

**대안 2 (적극적 — 결과를 주장으로 승격)**
> "…reaching ADD-AUCs of 69.0 for KUKA and 71.3 for Baxter. On Baxter the proposed method exceeds every compared method by a wide margin, and on KUKA it is within a few points of the strongest, which shows that a single pipeline transfers across kinematically distinct robots without per-robot redesign. The comparison is against the synthetic splits the compared methods report; the Panda real numbers are a separate regime."

---

## B-2. `PAPER_OVERLEAF.tex:245` — "observability ceiling … robust limit of keypoint-based estimation"

**현재 문장**:
> "The analysis reveals **an observability ceiling at distal joints**: a wrist joint's rotation about its own axis does not move its own keypoint, so injecting perfect keypoints barely improves the wrist angle, and an end-effector appearance head does not surpass the standard head either, **indicating a robust limit of keypoint-based estimation.**"

**왜 문제인가**: 이 문단은 Baxter 25.2를 **방법의 원리적 한계**로 설명하려고 존재한다. 그런데 Baxter는 카메라 파라미터 수정만으로 **0.2739 → 0.7125**, 파국 프레임 **24.7% → 8.0%**로 바뀌었다. 관측성 천장이 병목이었다면 일어날 수 없다. 개별 관측(GT 2D 주입해도 손목 MAE 28→28°, mlp_patch가 plain mlp를 못 이김)은 **여전히 유효**하지만, 거기서 "따라서 이것이 포즈 정확도의 지배적 한계"라는 **인과 도약이 반증**되었다. 실제로 손목 각도 25° 오차는 키포인트를 8 mm만 움직여 ADD에 거의 기여하지 않는다(§13 FK 민감도).

**대안 1 (축소 — 관측 유지, 인과 주장 제거)**
> "A secondary observability effect is present at distal joints: a wrist joint's rotation about its own axis does not move its own keypoint, so injecting perfect keypoints barely improves the wrist angle. Its effect on pose is small, however, because a wrist error of that magnitude displaces the keypoint by only a few millimetres; the dominant residual failure is link-identity confusion, not distal observability."

**대안 2 (교체 — 실제 잔존 한계로 갈아끼움)**
> "The residual failures are concentrated in a catastrophic tail rather than spread across frames: on KUKA, 15.9 percent of frames exceed 100 mm and the 99th percentile reaches about a metre, while the median is 13.1 mm. These failures are link-identity confusions, where a plausible but wrong keypoint-to-link assignment yields a confident wrong pose, and they are not detectable from confidence alone. This, rather than distal-joint observability, is the limit that remains."

> 📌 **대안 2를 택하면** `tab:refuted`의 관련 항목(신뢰도 기반 거부 반증)과 자연스럽게 연결된다.

---

## B-3. `PAPER_OVERLEAF.tex:246` — 위 두 문장의 한국어 대역

`:245` 확정안이 정해지면 동일 취지로 갱신. 현재 남아 있는 문제 구절: "이 수치는 Panda 실측 결과와 **비교 대상이 아니며** 병목 분석을 동반한 **적용 가능성 연구**다", "분석은 원위 관절의 **관측성 천장**을 드러낸다 … **키포인트 기반 추정의 견고한 한계**를 보인다".

---

## B-4. `PAPER_OVERLEAF.tex:312` — 결론부

**현재 문장**:
> "Applied to all three DREAM robots, the pipeline runs end-to-end on each, and **the analysis quantified an observability ceiling at distal joints**: a wrist joint's rotation about its own axis does not move its own keypoint, so its angle remains under-determined even with perfect keypoints. Remaining limitations include the absence of public real data for KUKA and Baxter, **the need for a benchmark-matched mesh to extend the depth correction**, and the reliance of render-and-compare on per-camera gating and anchoring…"

**왜 문제인가**: 두 개의 독립된 문제가 있다. (1) 관측성 천장을 **논문의 기여로 승격**해 결론에 넣었는데 그 인과가 반증됐다. (2) "benchmark-matched mesh가 필요하다"는 **사실이 거짓**이다 — iiwa7 메쉬는 확보되어 있고 렌더러 건전성까지 확인됐다(GT 포즈 IoU 0.858). 결론에서 존재하지 않는 한계를 향후 과제로 제시하는 셈이다.

**대안 1 (최소 수정)**
> "Applied to all three DREAM robots, the pipeline runs end-to-end on each without per-robot redesign. Remaining limitations include the absence of public real data for KUKA and Baxter, a catastrophic tail from link-identity confusion that confidence-based rejection does not capture, and the reliance of render-and-compare on per-camera gating and anchoring; a formulation that suppresses the depth ambiguity of free silhouette optimization in a principled way is left to future work."

**대안 2 (진단을 기여로 재배치)**
> "Applied to all three DREAM robots, the pipeline runs end-to-end on each without per-robot redesign, and the analysis isolated the dominant residual failure mode: a plausible but wrong keypoint-to-link assignment produces a confident wrong pose, so the error distribution is a low-rate catastrophic tail rather than a uniform degradation. Because this failure is confident, confidence-based rejection cannot address it, and correspondence-level disambiguation is the natural next step. Remaining limitations include the absence of public real data for KUKA and Baxter and the reliance of render-and-compare on per-camera gating and anchoring."

---

## B-5. `PAPER_OVERLEAF.tex:313` — 결론부 한국어 대역

`:312` 확정안에 맞춰 갱신. 현재 남은 문제 구절: "분석은 원위 관절의 **관측성 천장을 정량 규명**했다", "**깊이 보정 확장에 필요한 벤치마크 정합 메쉬**".

---

## B-6. `PAPER_DRAFT.md:194` — "직접 비교 불가" 경고 블록

**현재 문장**:
> "⚠️ Panda는 real ADD-AUC, KUKA/Baxter는 **합성·RC 미적용**이라 서로 **직접 비교 불가** — 통합 성능 뷰일 뿐이며…"

**왜 문제인가**: "합성·RC 미적용"은 **여전히 사실**이므로 경고 자체는 유지 가치가 있다. 다만 세 숫자(0.804 / 0.690 / 0.713)가 이제 비슷한 대역에 있어 독자가 오히려 **동일 지표로 착각하기 쉬워졌다** — 경고의 필요성은 줄지 않고 오히려 커졌다.

**대안 1 (유지 + 근거 명시)**
> "⚠️ 세 수치는 대역이 비슷하지만 **동일 조건이 아니다**: Panda는 실측 + RC, KUKA/Baxter는 합성 + RC 미적용이다. 로봇 간 우열이 아니라 **동일 파이프라인의 이식성**을 보는 표다."

**대안 2 (표에 조건 열 추가)**
표 3에 `데이터` 열 옆에 **`RC 적용` 열(✅/—)**을 추가해 경고문 의존을 줄인다. 각주 대신 표 자체가 조건을 드러내는 편이 오독에 강하다.

---

## B-7. `PAPER_DRAFT.md:206` — "메쉬 부재로 낮음 / 공정 비교 아님 / 성능 주장 아님"

**현재 문장**:
> "KUKA·Baxter는 우리 쪽에 render-compare가 **없어(정합 메쉬 부재, §4.7) 낮으며 공정 비교가 아니다** — 목적은 **성능 주장이 아니라** 동일 파이프라인이 세 로봇에서 end-to-end로 동작함(적용 가능성)을 보이는 것이다."

**왜 문제인가**: 세 겹으로 틀렸다. (1) **원인 귀속이 거짓** — 메쉬는 있다. (2) **"낮다"가 더 이상 사실이 아니다** — Baxter는 1위다. (3) **"성능 주장이 아니다"는 자기 부정** — 표 `tab:main`에 경쟁 수치와 나란히 실린 이상 이미 성능 주장이고, 이제 유리한 주장이다. 참고로 RC를 적용하지 않은 것은 **여전히 사실**이므로, RC 미적용 자체는 남겨도 된다(다만 이제는 "불가능해서"가 아니라 "적용하지 않아서"다).

**대안 1 (사실 교정 중심)**
> "KUKA·Baxter에는 render-compare를 적용하지 않았고(실측 자가학습 데이터 부재), 그럼에도 Baxter에서는 전 비교 방법을 큰 격차로 앞서고 KUKA에서는 수 점 차로 근접한다. 즉 이 결과는 적용 가능성 확인인 동시에 경쟁력 있는 성능이다."

**대안 2 (헤드룸까지 명시)**
> "KUKA·Baxter는 render-compare 없이 측정한 수치다 — 즉 Panda에 적용한 레버 하나를 쓰지 않은 상태에서 Baxter 1위, KUKA 근접이다. RC를 이 두 로봇으로 확장하는 것은 남은 헤드룸이며 본 논문의 범위 밖이다."

> ⚠️ **대안 2 사용 시 주의**: RC 수치를 본문에 넣지 말 것. 선택자 재튜닝이 진행 중이다(현재 KUKA 솔버 0.690 + 선택적 RC = 0.708, oracle 상한 0.745 — **전부 미확정**).

---

## B-8. `PAPER_DRAFT.md:328` / `:330` — §4.7 본문 (🔴 **수치 불일치 발생 중**)

> ✅ **수치 불일치는 해소됨 (2026-07-22).** 아래 "대안 1"에 해당하는 **최소 교정이 적용**되었다. 파일 내부의 두 세대 수치 공존은 사라졌다.
>
> ⚠️ **적용 시 판단 사항 (검토 요망)**: 숫자만 바꾸면 *"direct-pose로 0.690을 기록한다"* 가 되어 **새로운 거짓**이 된다 — 0.690은 direct-pose가 아니라 **솔버**가 낸 값이기 때문이다. 숫자와 경로명이 한 절에 묶여 문법적으로 분리 불가능하므로, **경로명 2어절(`direct-pose로` → `운동학 솔버가`)까지 함께** 교체했다. 이는 승인된 `tex:245`("poses are obtained directly from the heads" → "poses come from the kinematic solver")와 **동일 종류의 사실 교정**이다. 그 문단의 **한계·병목 논증은 손대지 않았다**(애초에 `:328`에는 없고 `:194/:206/:245`에 있다).

**적용 전 문장**:
> ":328 … 포즈는 head 각도 + 회전 헤드의 R,t를 직접 쓰는 **direct-pose로 ADD-AUC 0.357(KUKA)·0.253(Baxter)**를 기록한다"
> ":330 … Pose via a **direct-pose scheme** (head angles + rotation-head R,t) gives ADD-AUC **0.357** (KUKA) / **0.253** (Baxter)"

**적용 후 문장**:
> ":328 … 포즈는 **예측된 관절각으로부터 운동학 솔버가 R,t를 복원**해 ADD-AUC **0.690**(KUKA)·**0.713**(Baxter)를 기록한다"
> ":330 … Pose from the **kinematic solver, which recovers R,t from the predicted joint angles**, gives ADD-AUC **0.690** (KUKA) / **0.713** (Baxter)"

**왜 필요했는가**: 이전 서술은 **방법의 능력을 KUKA +0.32 / Baxter +0.44 ADD-AUC만큼 과소평가**했고, direct-pose를 **설계 선택처럼** 제시했으나 실제로는 솔버 경로가 버그로 막혀 있었을 뿐이다.

**남은 선택지 — 대안 2 (진단을 방법론 교훈으로 서술)**
> "포즈는 운동학 솔버로 복원하여 ADD-AUC 0.690(KUKA)·0.713(Baxter)를 얻는다. 두 로봇의 초기 결과는 이보다 크게 낮았는데, 원인은 방법이 아니라 합성 데이터 트리에서 카메라 내부파라미터가 기본값으로 대체되던 평가 경로의 결함이었다. 이는 재투영 기반 솔버가 내부파라미터 오류에 대해 조용히 실패하며, 그 실패가 모델 품질의 한계처럼 보인다는 점을 보여준다."

> 📌 대안 2는 정직하지만 **논문에 버그 서사를 노출**한다. 통상 이런 내용은 본문이 아니라 재현성 부록이나 각주에 둔다. 저자 판단 필요.

---

## B-9. `PAPER_DRAFT.md:190` / `:191` — 표 3 **병목 열** (수치는 갱신됨, 병목은 미갱신)

**현재 상태** (포즈 수치는 (A)에서 이미 교체):
> `| KUKA | 합성(synth) | 0.735 | 0.690 | 회전 헤드 병진 오차(56mm) |`
> `| Baxter 좌완 | 합성(synth) | 0.817 | 0.713 | 손목 관절 **관측성 천장** |`

**왜 문제인가**: 두 병목 귀속이 모두 반증됐다. KUKA의 "회전 헤드 병진 오차 56 mm"는 참 K 솔버에서 **15.7 mm**로 떨어지므로 병목이 아니었다. Baxter의 "관측성 천장"은 B-2 참조. 지금 표는 **갱신된 수치와 반증된 병목이 한 행에 공존**한다.

**대안 1**
> `| KUKA | 합성(synth) | 0.735 | 0.690 | link-identity 혼동 꼬리(fail 15.9%) |`
> `| Baxter 좌완 | 합성(synth) | 0.817 | 0.713 | — (잔여 오차 균질, fail 8.0%) |`

**대안 2**: 병목 열을 **삭제**하고 `median ADD` 열로 대체(KUKA 13.1 mm / Baxter 17.1 mm). 병목은 본문 서술로 넘긴다 — 한 단어로 병목을 규정하려다 두 번 틀렸으므로, 표에서 빼는 편이 안전하다.

---

## B-10. `figures/make_figs_multirobot.py:51-53` — 그림 7 주석

**현재 주석**: `"SYNTHETIC-ONLY  (no render-compare)\nnot comparable to Panda real"`

**왜 문제인가**: 문구 자체는 여전히 사실이다(합성이고 RC 미적용). 다만 막대 높이가 0.357/0.253 → 0.690/0.713으로 올라와 **Panda 0.804와 시각적으로 비슷해졌으므로**, "not comparable" 경고의 의미가 이전보다 중요해졌다. **데이터·y축은 (A)에서 갱신 완료**했고 문구는 그대로 두었다.

**대안 1 (경고 강화)**
> `"SYNTHETIC-ONLY  (no render-compare)\ndifferent regime — not a robot-vs-robot ranking"`

**대안 2 (비교 대상 표시)**: 합성 두 막대에 **경쟁 최고치를 옅은 참조선**으로 겹쳐 그린다(KUKA 80.2 RoboPose, Baxter 58.8). 그러면 "Panda와 비교하지 말라"가 아니라 **"이 막대는 경쟁 대비로 읽어라"**가 그림 자체로 전달된다.

---

## 적용 순서 제안

1. ✅ ~~**B-8 먼저** — `PAPER_DRAFT.md` 내부 수치 불일치 해소~~ **완료**(수치 + 경로명 최소 교정 적용). 남은 것은 대안 2(버그 서사 노출 여부) 채택 여부뿐이며 **선택 사항**이다.
2. **B-2 / B-4 결정** — 관측성 천장을 어떻게 처리할지가 나머지 문안을 좌우한다(B-1·B-6·B-7·B-9가 여기 딸려온다).
3. **B-1 / B-6 / B-7** — 포지셔닝 문장 일괄.
4. **B-3 / B-5** — 한국어 대역을 확정안에 맞춰 동기화.
5. **B-9 / B-10** — 표·그림 마감.

## 반영 금지 (미확정)

- **RC 수치 전부** — 선택자 재튜닝 진행 중(KUKA 솔버 0.690 + 선택적 RC = 0.708, oracle 상한 0.745).
- **로버스트 outlier 거부의 추가 이득** — 미측정. 이번 개선은 **전적으로 참 K**에서 왔고 outlier 비율은 불변(KUKA 21.1% / Baxter 11.6%)이다.
</content>
