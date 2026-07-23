# 2026-07-22 — 🔴 근본원인: KUKA/Baxter 내부파라미터(K) 버그 — "솔버 발산"은 전부 이 버그의 산물

> **KUKA/Baxter가 경쟁모델 대비 처참했던(ADD-AUC 0.357 / 0.253 vs 75~80 / 33~59) 진짜 이유는 모델도 솔버도 아니라 *카메라 내부파라미터가 항등행렬로 주입되고 있었다는 것*이다.** 스코프는 정확히 이 두 로봇 — Panda는 영향 없음(배포 0.804 유효).
>
> 성격: **평가 시점(eval-time) 버그**. head는 올바르게 학습되어 있으며 **재학습 불필요**((e)절 게이지 논증).
>
> ✅ **2026-07-22 확정**: **전체 테스트셋 실측 완료** (KUKA 5997프레임 / Baxter 5982프레임), **독립 3중 검증** 통과. KUKA ADD-AUC **0.3682 → 0.6901**(+0.322, +87%), Baxter **0.2739 → 0.7125**(+0.439, +160%). 320프레임 subset 수치는 (b)절에서 full-set 수치로 **대체**되었다. 논문 수정 대상 목록은 (h)절 — **본 문서 갱신 범위에서 논문 파일은 편집하지 않는다**(별도 작업).

---

## TL;DR

- **버그**: `datasets/synthetic/{kuka,baxter}_synth_*` 의 프레임 JSON에는 `meta` 키 자체가 없다 → `TRAIN/dataset.py:574-578` 이 조용히 `eye(3)`로 폴백. 그 항등 유래 K가 `Eval/kuka_add_eval.py:156` · `Eval/baxter_add_eval.py:156` 을 거쳐 **진짜 투영을 수행하는 솔버**(`Eval/solve_pose_kinematic.py:105-114`)에 그대로 들어간다.
- **오차 크기**: crop·scale 후 솔버가 받는 값이 **fx ≈ 1.7~1.8**, 참값은 프레임/로봇에 따라 **555~626**(초기 subset 실측 555.4, full-set 대표값 **577 / 626**). **cx = −562 vs 참값 −6.9**. 어긋남 배율은 **정확히 ×320**(native 참값 fx=fy=320, 항등 K는 fx=1 → 배율이 곧 320). 고전적인 **focal/depth 모호성**.
- **파국 확인**: Panda에 완벽한 GT 2D를 넣고 항등 K로 풀면 솔버가 깊이를 **943 mm 대신 4.9 mm**로 복원, ADD-AUC **0.0000**.
- **참 K로 고치면 (전체 셋 확정)**: 출하 중인 `--direct-pose` 대비 KUKA 0.3682 → **0.6901**(+0.322, **+87%**), Baxter 0.2739 → **0.7125**(+0.439, **+160%**).
- **경쟁모델 대비 위치가 뒤집힌다**(Protocol A, ×100): KUKA **69.0** vs RoboPEPP 76.2 / RoboPose 80.2 / HoRoPose 75.1 / RoboTAG 75.0 — "한참 뒤"에서 **사정권**으로. Baxter **71.3** vs RoboPEPP 34.4 / RoboPose 32.7 / HoRoPose·RoboTAG 58.8 — **큰 격차의 1위**.
- **이득의 출처는 오직 참 K다.** 애초 가설이던 **outlier 제거(로버스트 거부)는 기여하지 않았다** — outlier 비율 자체가 불변(KUKA 키포인트의 21.1%, Baxter 11.6%). 로버스트 거부가 그 위에 추가 이득을 주는지는 **미측정**.
- **꼬리(tail) 거동은 두 로봇이 갈린다**: KUKA는 **파국 꼬리가 살아남고**(fail>100mm 15.4% → 15.9%, mean 91.1 mm = median의 7배, p99 = 1012 mm), Baxter는 **꼬리가 대부분 사라진다**(24.7% → **8.0%**). 즉 Baxter의 꼬리는 *이 버그 자체*였고, KUKA의 꼬리는 **별개의 잔존 이슈**(link-identity 혼동)다.
- **무효화되는 과거 결론 4건**: "솔버가 KUKA/Baxter에서 발산한다", "병목은 rot-head 병진오차 56 mm다", "iiwa7에서 재투영 최적화는 해롭다", 그리고 `--direct-pose`가 유일하게 살아남은 이유(= **K를 한 번도 건드리지 않는 유일 경로**였기 때문).
- **수정**: eval-time 단독. 검증된 `Eval/iiwa7_rc_eval.py:81-102` 의 `geometric_K()`를 솔버 호출부에 배선. 원칙 = **모델에는 dataset K, 솔버에는 참 K**.
- **동반 결과**: KUKA render-and-compare 배선 완료·동작 확인(GT-pose IoU 0.858), 50프레임에서 0.2804 → 0.5721. ⚠️ 단 이 RC 튜닝은 **수정 전 baseline 위에서** 이뤄진 것 → **재튜닝 필요(별도 에이전트 진행 중)**, 논문 반영 금지((g)절).
- **논문**: DR 두 셀은 교체 가능하나 **KUKA-Photo는 미측정**이라 표 갱신이 아직 막혀 있다((h)절). **Baxter Photo 스플릿은 DREAM에 존재하지 않는다.**

---

## (a) 버그와 그 정확한 스코프

### 발화 지점 — 데이터셋 폴백

`TRAIN/dataset.py:574-578`:

```python
# Camera intrinsic matrix K (from meta.K)
if 'meta' in data and 'K' in data['meta']:
    keypoints['camera_K'] = np.array(data['meta']['K'], dtype=np.float32)
else:
    # Default fallback (should not happen with proper data)
    keypoints['camera_K'] = np.eye(3, dtype=np.float32)
```

주석이 "should not happen with proper data"라고 적혀 있으나 **실제로는 항상 발생한다** — 해당 트리에서. 실측 확인:

| 데이터셋 트리 | 프레임 JSON 최상위 키 | `meta.K` | dataset이 반환하는 K |
|---|---|---|---|
| `datasets/synthetic/kuka_synth_{train,test}_{dr,photo}` | `camera_data`, `objects`, `sim_state` | **없음** | 🔴 `eye(3)` |
| `datasets/synthetic/baxter_synth_{train,test}_dr` | `camera_data`, `objects`, `sim_state` | **없음** | 🔴 `eye(3)` |
| `datasets/ICRA_multiview/Converted_dataset/DREAM_to_DREAM` (Panda 실측) | … + `meta` | **있음** (orb/realsense/kinect 615.5, azure 399.7) | ✅ 참값 |
| `.../DREAM_to_DREAM_syn` (Panda 합성) | … + `meta` | **있음** (320) | ✅ 참값 |

⇒ **스코프는 정확히 KUKA·Baxter 두 로봇.** Panda는 학습·평가 모두 `DREAM_to_DREAM{,_syn}`을 쓰고 이 트리는 실제 `meta.K`를 싣고 있다. **배포 mean 0.804는 이 버그와 무관하며 유효하다.**

> 📌 문서 정정: `Eval/iiwa7_rc_eval.py:84` 의 docstring은 *"`PoseEstimationDataset` returns `camera_K = eye(3)` on DREAM (frame JSONs carry no meta.K)"* 라고 일반화해 적고 있으나, 이는 **KUKA/Baxter 합성 트리에 한해 참**이고 Panda `DREAM_to_DREAM` 트리에는 **거짓**이다. 이 과잉일반화가 버그를 "정상 동작"처럼 보이게 만든 한 요인이다.

### 전파 지점 — 솔버 호출부

`Eval/kuka_add_eval.py:156` / `Eval/baxter_add_eval.py:156` (두 파일 동일 형태):

```python
refined, kp_cam, reproj = spk.solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters, ...)
```

여기 `K`가 위의 항등 유래 K다. 그리고 솔버는 이걸 **진짜 원근투영**에 쓴다 — `Eval/solve_pose_kinematic.py:105-114`:

```python
def project_points(pts_robot, R, t, K):
    pts_cam = torch.bmm(pts_robot, R.transpose(1, 2)) + t.unsqueeze(1)   # (B,N,3)
    pts_img = torch.bmm(pts_cam, K.transpose(1, 2))                      # (B,N,3)
    z = pts_img[..., 2:3].clamp(min=1e-6)
    return pts_img[..., :2] / z, pts_cam
```

`z`로 나누는 **원근** 투영이므로 fx가 320배 작으면 재투영 잔차를 최소화하는 해는 **깊이를 320배 축소**하는 쪽이다. 이것이 focal/depth 모호성이며, 솔버는 "발산"한 게 아니라 **주어진(틀린) 카메라에 대해 정확히 최적해를 찾고 있었다.**

### 오차 크기 (crop·scale 반영, 실측)

| 항목 | 항등 K 경로 (현행) | 참 K | 비 |
|---|---|---|---|
| fx (post-crop/scale) | 1.736 | 555.4 | **×320** |
| cx (post-crop/scale) | −562 | −6.9 | — |

native `_camera_settings.json` 기준 참값은 `fx=fy=320, cx=320, cy=240` (640×480, hfov 90°) — KUKA·Baxter 양쪽 동일. 항등 K의 fx가 1이므로 배율이 정확히 320이 되는 구조다. crop 블록이 주점을 제자리에서 이동시키므로(`camera_K[0,2] -= bx0`) 항등 위에서는 `K[0,2] = -bx0`가 그대로 남고, 참 K는 여기에 `cx0`(=320)를 더한 값이다.

### 파국의 독립 확인 (sanity check)

**Panda + 완벽한 GT 2D + 항등 K**로 솔버를 돌린 결과: 복원 깊이 **4.9 mm** (참값 943 mm), ADD-AUC **0.0000**. 입력 2D가 완벽해도 카메라가 틀리면 해가 파국이라는 것 — 즉 이 실패는 **검출기·head 품질과 완전히 무관**하다.

---

## (b) ✅ 확정 측정 결과 — 전체 테스트셋

> **상태: 확정.** 전체 테스트셋 실측이며, 독립 3중 검증을 통과했다. 이전 판본의 320프레임 subset 표는 이 표로 **대체**되었다(subset 경향은 방향·크기 모두 일치했다).

### KUKA iiwa7 — DR 전체 5997 프레임

| 모드 | ADD-AUC | mean ADD | median ADD | fail>100mm | median t-err | median R-err |
|---|---|---|---|---|---|---|
| `--direct-pose` (출하·논문 게재 구성) | 0.3682 | 72.1 mm | 60.2 mm | 15.4% | 56.1 mm | 7.42° |
| 솔버 + **참 K** | **0.6901** | 91.1 mm | **13.1 mm** | 15.9% | **15.7 mm** | **5.77°** |

### Baxter 좌완 — DR 전체 5982 프레임

| 모드 | ADD-AUC | mean ADD | median ADD | fail>100mm | median t-err | median R-err |
|---|---|---|---|---|---|---|
| `--direct-pose` (출하·논문 게재 구성) | 0.2739 | 83.1 mm | 73.3 mm | 24.7% | 59.7 mm | 5.65° |
| 솔버 + **참 K** | **0.7125** | **39.5 mm** | **17.1 mm** | **8.0%** | **29.4 mm** | 5.91° |

### KUKA iiwa7 — Photo 전체 5999 프레임 (07-22 추가 측정 — 게이트 해제)

| 모드 | ADD-AUC | median ADD | med t-err | med R-err |
|---|---|---|---|---|
| `--direct-pose` (출하) | 0.3305 | 64.8 mm | 58.9 mm | 8.00° |
| 솔버 + **참 K** | **0.6984** | **12.1 mm** | **15.0 mm** | **5.98°** |

**개선폭**: KUKA-DR **+0.322 (+87%)**, KUKA-Photo **+0.368 (+111%)**, Baxter-DR **+0.439 (+160%)**.

> ✅ **논문 표 게이트 해제.** KUKA-Photo가 DR과 **동일 설정**(솔버+참 K, 전체 셋, 동일 체크포인트)으로 측정되어, 이제 한 행 안에서 파이프라인 구성이 섞이지 않는다. **Baxter Photo는 DREAM에 부재함이 재확인**되어 계속 비워 둔다.

### 경쟁모델 대비 위치 (Protocol A, ×100)

| 로봇 | 우리(참 K 솔버) | RoboPEPP | RoboPose | HoRoPose | RoboTAG | 판정 |
|---|---|---|---|---|---|---|
| KUKA-DR | **69.0** | 76.2 | 80.2 | 75.1 | 75.0 | "한참 뒤"에서 **사정권**으로 (기존 35.7) |
| KUKA-Photo | **69.8** | 76.1 | 73.2 | 73.9 | 76.6 | **사정권** (기존 31.9) |
| Baxter-DR | **71.3** | 34.4 | 32.7 | 58.8 | 58.8 | 🥇 **큰 격차 1위** (기존 25.2) |

> 📌 **반올림**: Baxter 원자료 0.7125 → ×100 = **71.25**로 정확히 중간값이다. 논문 표는 소수 1자리이므로 **half-up으로 71.3**을 채택했다(하위 그룹 최고 → `\textbf{}`). 71.2로 표기할 근거도 동등하므로 저자가 다르게 정하면 표·그림·본문 3곳을 함께 바꿔야 한다.

---

### 🔴 반드시 함께 기록할 단서(qualification) 4건

**① 이득은 100% 참 K에서 온다 — outlier 제거 가설은 기각.**
애초 이 개선을 설명하려던 "로버스트 outlier 제거" 가설은 **성립하지 않는다**. 수정 전후로 outlier 비율이 **변하지 않았다**(KUKA 키포인트의 **21.1%**, Baxter **11.6%**). 즉 솔버는 같은 수의 나쁜 대응을 여전히 먹고 있으며, 단지 **올바른 카메라**로 먹고 있을 뿐이다. 로버스트 거부를 **추가로** 얹었을 때 이득이 더 있는지는 **미측정**.

**② KUKA의 파국 꼬리는 살아남았다 — Baxter의 꼬리는 사라졌다.** (두 로봇의 결정적 대조)

| | KUKA | Baxter |
|---|---|---|
| fail>100mm 변화 | 15.4% → **15.9%** (사실상 불변) | 24.7% → **8.0%** (붕괴) |
| mean vs median | 91.1 mm = median(13.1)의 **7배** | 39.5 mm = median(17.1)의 2.3배 |
| p99 | **1012 mm** | — |
| 해석 | 수정은 **좋은 프레임을 훨씬 좋게** 만들 뿐 **나쁜 프레임은 고치지 못한다**. KUKA의 잔존 꼬리 = **link-identity 혼동**(별개 이슈, 미해결) | Baxter의 꼬리는 **이 버그 자체였다** |

⇒ KUKA에서 mean ADD가 72.1 → 91.1 mm로 **오히려 나빠지는** 것은 모순이 아니다. median이 60.2 → 13.1 mm로 개선되는 동시에 소수 파국 프레임의 오차가 더 커진 결과이며, ADD-AUC@100mm는 median 개선을 반영해 크게 오른다. **KUKA는 "고쳤다"가 아니라 "좋은 프레임만 고쳤다"로 서술해야 한다.**

**③ 재현 caveat — 재측정 direct-pose가 아카이브보다 균일하게 높다.**

| 셀 | 아카이브(논문 게재) | 재측정 direct-pose | 차이 |
|---|---|---|---|
| KUKA-DR | 35.7 | 36.8 | **+1.1** |
| KUKA-Photo | 31.9 | 33.1 | **+1.2** |
| Baxter-DR | 25.3 | 27.4 | **+2.1** |

**세 셀 모두 같은 방향으로 +1~2점**이다(무작위 흔들림이 아니라 계통 차이). 유력 원인은 **`best_*` vs `last_*` 체크포인트 선택**. 아카이브 수치는 **비트 단위로 재현되지 않았다**. 다만 개선폭이 +32~44점 규모이므로 **결론은 영향받지 않는다**.

> ✅ **논문 노출 여부**: 교체된 세 셀은 **전부 새 설정(솔버+참 K)** 값이므로, 옛 direct-pose 수치는 **논문 본문에서 모두 사라졌다**((h)절 적용 결과 확인). 따라서 이 ±2점 계통 차이가 **논문의 어떤 비교에도 들어가지 않는다.** 단 **`PAPER_DRAFT.md:328/330`은 (B)층으로 분류되어 아직 옛 0.357/0.253을 들고 있다** — 그 두 줄이 남아 있는 동안은 문서 내부 수치 불일치가 존재한다.

**④ Panda는 영향 없음 — 배포 0.804 유효.**
Panda는 `Converted_dataset/DREAM_to_DREAM*` 트리를 쓰고 이 트리는 **실제 `meta.K`를 싣는다**. 그리고 `Eval/refine_eval.py`의 일반화된 `geometric_K`는 **참 K가 들어오면 그대로 통과시킨다**(검증 완료). ⇒ **배포 Panda real mean 0.804는 변하지 않는다.**

---

### 독립 3중 검증

1. **direct 모드 포즈 오차가 rot-head 자체 학습 로그와 정확히 일치**한다 — KUKA 56.1 mm / 7.42°. 평가 하네스가 학습 시점 지표를 그대로 재생산한다.
2. **재구성한 K가 데이터셋 자신의 2D와 정합**한다 — GT 3D를 재구성 K로 투영했을 때 데이터셋의 2D 키포인트와 **median 0.0003 px**.
3. **패치된 프로덕션 스크립트가 독립 에이전트의 수치를 비트 단위로 재현**한다.

---

## (c) 이 버그가 무효화하는 과거 결론

| 과거 결론 | 실제 | 판정 |
|---|---|---|
| "KUKA/Baxter에서는 솔버가 발산한다" | 솔버는 **망가진 카메라를 먹고 있었다**. 참 K를 주면 발산하지 않고 direct-pose를 크게 능가 | 🔴 **무효** |
| "병목은 rot-head의 56 mm 병진오차다" | rot-head의 `\|dz\|`는 33.6 / 36.3 mm = **~1 m 장면의 3~4%**. 정상적인 metric 회귀 수준이며 병목이 아님 | 🔴 **무효** |
| "iiwa7에서 재투영 최적화는 해롭다" | 버그의 산물 | 🔴 **무효** |
| "`--direct-pose`가 최선의 포즈 경로다" | direct-pose는 **K를 한 번도 건드리지 않는 유일 경로**라서 살아남았을 뿐. 우월해서가 아니라 **면역이라서** 이겼다 | 🔴 **무효(원인 오귀속)** |
| "Baxter의 병목은 **손목 관측성 천장**이다" | 참 K만으로 0.2739 → **0.7125**, 파국 프레임 24.7% → **8.0%**. 관측성이 지배 요인이었다면 일어날 수 없는 변화 | 🔴 **무효 (full-set 실측)** |
| "KUKA/Baxter에는 정합 메쉬가 없어 RC를 못 쓴다" | iiwa7 URDF+메쉬 **존재**((g)절, FK 0.0048 mm RMS) | 🔴 **사실 오류** |

> ✅ 위 6건은 **full-set 확정으로 판정이 굳어졌다**(이전 판본의 "보류" 해제).
>
> ❌ **반대로, 이 버그가 설명하지 *못하는* 것**: **KUKA의 잔존 파국 꼬리**(fail 15.9%, p99 1012 mm). 이는 K와 무관한 **link-identity 혼동**이며 여전히 미해결이다. 또한 이번 이득은 **outlier 제거와 무관**하다((b)절 단서 ①) — "로버스트 거부가 문제였다"로 서술하면 틀린다.

**교훈(방법론)**: 어떤 우회로가 "이유는 모르겠지만 이것만 된다"는 형태로 살아남을 때, 그 우회로가 **무엇을 건너뛰는지**를 먼저 봐야 한다. `--direct-pose`의 정의적 특징이 바로 "K를 안 쓴다"였고, 그게 단서였다.

---

## (d) 권장 수정 — eval-time 단독, 재학습 없음

### 원칙

> **모델에는 dataset K를, 솔버에는 참 K를.**

두 K를 분리해야 하는 이유는 (e)절 — 학습된 체크포인트가 항등 K 게이지 위에서 bearing feature를 배웠기 때문에 **모델 입력 K를 바꾸면 오히려 망가진다**. 바꿔야 하는 건 **기하 연산에 들어가는 K**뿐이다.

### 구현

이미 검증된 헬퍼가 있다 — `Eval/iiwa7_rc_eval.py:81-102` 의 `geometric_K(val_dir, camera_K, original_size, S)`:

```python
Kt[:, 0, 0] = it['fx']; Kt[:, 1, 1] = it['fy']; Kt[:, 2, 2] = 1.0
Kt[:, 0, 2] = it['cx'] + camera_K[:, 0, 2]      # cx0 - bx0
Kt[:, 1, 2] = it['cy'] + camera_K[:, 1, 2]      # cy0 - by0
return scale_K(Kt, original_size, S)
```

`_camera_settings.json`의 native intrinsics와, 항등 K 위에 남아 있는 crop 오프셋(`-bx0`, `-by0`)을 결합해 **crop·scale 반영된 참 K**를 복원한다. 자체 검증 완료: *GT 3D를 데이터셋 자신의 2D 키포인트로 재투영해 60프레임에서 **<0.09 px***.

**배선 대상**: `Eval/kuka_add_eval.py:156`, `Eval/baxter_add_eval.py:156` 의 `spk.solve_batch(..., K, ...)` 인자를 `geometric_K(...)` 산출물로 교체. 모델 forward에 들어가는 K는 **손대지 않는다**.

**주의 2건**:
1. `geometric_K:95-96` 에 `assert abs(camera_K[0,0,0] - 1.0) < 1e-6` 가드가 있다 — 입력이 항등 폴백임을 전제한다. 공용 헬퍼로 승격할 경우 이 가드를 "항등이면 재구성, 아니면 그대로 통과"로 일반화해야 Panda 경로에서 죽지 않는다.
2. `geometric_K`는 `val_dir`의 `_camera_settings.json`에 의존한다. 카메라가 프레임마다 바뀌는 트리에는 그대로 쓸 수 없다(KUKA/Baxter 합성은 단일 카메라 설정이라 안전).

### 왜 재학습이 필요 없는가 → (e)절

---

## (e) 게이지 논증 — head는 올바르게 학습되어 있다

두 갈래 모두 항등 K에 오염되지 않았다:

1. **rot head의 학습 타깃은 K-free**다. 회전·병진 타깃이 Kabsch(3D↔3D 정렬)로 만들어지므로 내부파라미터가 개입할 여지가 없다.
2. **geo/bearing 게이지는 정확히 affine**이다. 항등 K 하의 bearing과 참 K 하의 bearing 사이에 실측 관계 `identity_bearing = 320 × true + 320`, 상관 **r = 1.00000000** 이 성립한다. 완전 affine이므로 **첫 Linear 층이 스케일·오프셋을 그대로 흡수**한다 — 네트워크는 일관된(단지 다른) 게이지 위에서 학습되었을 뿐 정보 손실이 없다.

⇒ 그래서 **모델에는 계속 dataset K(항등)를 넣어야** 하고(게이지 유지), 바뀌는 건 솔버·렌더러가 쓰는 기하 K뿐이다. 재학습 불필요.

---

## (f) 별도 잠재 이슈 — `zeros(3,3)` 폴백

`Eval/inference_4tier_eval.py:128-131`:

```python
# Camera K
camera_K = np.zeros((3, 3), dtype=np.float32)
if "meta" in data and "K" in data["meta"]:
    camera_K = np.array(data["meta"]["K"], dtype=np.float32)
```

`meta.K`가 없으면 **영행렬**이 나간다. `eye(3)`보다 나쁘다 — 영행렬은 투영 시 `z=0`을 만들어 `clamp(min=1e-6)`에 걸리며, 조용히 무의미한 값을 흘린다.

**권장**: 폴백 대신 **예외를 던질 것**(`raise ValueError(f"{path}: meta.K missing")`). 이번 사건의 본질은 "잘못된 값"이 아니라 **"조용한 폴백"** 이었다. 같은 패턴이 여기 남아 있다.

> 현재 이 경로가 실제로 오염된 결과를 낸 적이 있는지는 **미측정**. 잠재 이슈로만 기록한다.

---

## (g) 동반 결과 — KUKA render-and-compare 배선 완료 (별도 에이전트 측정)

- **구현**: `Eval/iiwa7_render.py` + `Eval/iiwa7_rc_eval.py` 신규.
- **메쉬 출처**: RoboPEPP에 동봉된 iiwa7 URDF + 메쉬. 그 URDF FK가 DREAM kuka를 **0.0048 mm RMS**로 재현 → 기하 정합성 확인.
- **렌더러 건전성**: GT 포즈 렌더 IoU **mean 0.858 / median 0.869**, **100%가 ≥ 0.5**. (Baxter RC가 겪었던 실루엣 붕괴 없음.)
- **RC 효과 (50프레임)**: 재투영 앵커를 건 RC가 해당 subset을 **0.2804 → 0.5721** (mean ADD 94.2 → 69.5 mm)로 개선. **Baxter식 전프레임 발산 없음.**

> ⚠️ 이 설정은 **50프레임에서 튜닝**된 것이고 현재 **500프레임 스윕 중**. 채택 전 단계.

**해석**: (b)의 확정 결과와 합치면 KUKA에는 두 개의 독립된 큰 레버가 있다 — 참 K 솔버(**0.6901**)와 RC.

> 🔴 **중대 caveat (07-22 확정 이후)**: 위 RC 설정은 **수정 전(망가진 솔버) baseline 위에서 튜닝된 것**이다. 기준선이 0.28 → 0.69로 올라간 지금, 이 하이퍼파라미터가 여전히 최적이라고 볼 근거가 없다. **고쳐진 솔버 위에서 재튜닝이 필요하며 별도 에이전트가 진행 중**이다. 재튜닝 전 RC 수치는 **논문에 반영하지 말 것.** 참 K 솔버와 RC의 중복/가산 여부도 여전히 **미측정**이다.

---

## (h) 논문 수정 — **(A) 사실 오류 수정 적용 완료 / (B) 서사 재작성은 초안만**

> 2026-07-22 KUKA-Photo 측정으로 **게이트가 해제**되어, 사실 오류에 해당하는 항목은 **적용했다**. 논증에 해당하는 항목은 **적용하지 않고** 별도 초안 문서로 분리했다: **[../PAPER_REVISION_DRAFT_2026-07-22.md](../PAPER_REVISION_DRAFT_2026-07-22.md)**

### 교체된 셀 (전부 동일 설정: 솔버+참 K, 전체 셋, 동일 체크포인트)

| 논문 셀 | 이전 | 신규 | 상태 |
|---|---|---|---|
| KUKA-**DR** | 35.7 | **69.0** | ✅ 적용 (5997프레임) |
| KUKA-**Photo** | 31.9 | **69.8** | ✅ 적용 (5999프레임) |
| Baxter-**DR** | 25.2 | **71.3** | ✅ 적용 (5982프레임, 0.7125 → half-up) |
| Baxter-**Photo** | (열 없음) | — | ➖ **DREAM에 스플릿 부재 재확인** — 계속 비워 둠 |

### ✅ (A) 적용 완료 — 사실 오류

| 위치 | 적용 내용 |
|---|---|
| `PAPER_OVERLEAF.tex:167` | 우리 행 `35.7 & 31.9 & 25.2` → `69.0 & 69.8 & \textbf{71.3}` |
| `PAPER_OVERLEAF.tex:166` | RoboTAG Baxter `\textbf{58.8}` → `58.8` (볼드 해제, 우리가 그 열 1위) |
| `PAPER_OVERLEAF.tex:171` | 캡션 "use the direct-pose configuration without render-and-compare" → "are evaluated with the kinematic solver but without render-and-compare, which is applied only to the Panda real splits" |
| `PAPER_OVERLEAF.tex:245` | "Because **no matching mesh is available** … poses are obtained **directly from the heads** … 35.7 / 25.2" → "Because no real data exist for self-training, and render-and-compare is reserved for the Panda real splits, poses come from the **kinematic solver** … **69.0 / 71.3**" |
| `PAPER_OVERLEAF.tex:246` | 위 문장의 한국어 대역 동일 갱신 |
| `PAPER_DRAFT.md:181,183` | 0.357·0.253 → **0.690·0.713** (국/영) |
| `PAPER_DRAFT.md:190,191` | 표3 포즈 수치 → 0.690 / 0.713 (**병목 열은 (B)로 이관**) |
| `PAPER_DRAFT.md:210` | 표4 우리 행 → `69.0 | 69.8 | **71.3**` |
| `PAPER_DRAFT.md:212` | 표4 캡션 "direct-pose without render-compare **(no mesh)**" → "kinematic solver without render-compare" (승인된 `tex:171` 수정의 동일 오류·동일 문장이라 확장 적용) |
| `figures/make_figs_multirobot.py:34` | `[0.804, 0.357, 0.253]` → `[0.804, 0.6901, 0.7125]`; 막대가 높아져 `ylim 0.98 → 1.05`, 상단 라벨 `0.90 → 0.94` 동반 조정 |

### 🟡 (B) 초안만 — 적용하지 않음

`tex:245`(applicability/not-comparable 프레이밍 · observability ceiling 서사), `tex:246`, `tex:312`, `tex:313`, `PAPER_DRAFT.md:194, 206, 328, 330`, `PAPER_DRAFT.md:190/191`의 **병목 열**, 그림 7 주석 문구. → 각 위치별 "현재 문장 / 왜 틀렸는지 / 대안 2가지"는 **[../PAPER_REVISION_DRAFT_2026-07-22.md](../PAPER_REVISION_DRAFT_2026-07-22.md)**.

> 🔴 **미해소 부작용**: `PAPER_DRAFT.md:328/330`이 (B)로 분류돼 **옛 0.357/0.253을 그대로 들고 있다.** 같은 파일의 `:181/:183/:190/:191/:210`은 갱신됐으므로 **현재 `PAPER_DRAFT.md` 내부에 두 세대의 수치가 공존**한다. 초안 문서 **B-8**이 이것을 최우선 항목으로 표시하고 있다.

### 1. `docs/dinobotpose3/PAPER_OVERLEAF.tex` (본문 — 최우선)

| 위치 | 현재 | 되어야 할 것 |
|---|---|---|
| **`:167`** (표 `tab:main` 우리 행) | `\textsc{DINObotPose} (Ours) & 74.2 & 76.9 & 79.5 & \textbf{82.8} & \textbf{81.5} & \textbf{77.8} & 35.7 & 31.9 & 25.2 \\` | 마지막 세 셀 = KUKA-DR / KUKA-Photo / Baxter-DR. **`35.7 → 69.0`**, **`25.2 → 71.3`**, **`31.9` = KUKA-Photo 재측정 대기**. Baxter-DR은 하위 그룹 최고이므로 **`\textbf{71.3}`으로 볼드 처리** |
| **`:166`** (RoboTAG 행) | Baxter 셀이 `\textbf{58.8}` | 우리 71.3이 이기므로 **볼드 제거** → `58.8`. (KUKA-DR 볼드는 RoboPose `\textbf{80.2}`(`:163`)에 그대로 유지, KUKA-Photo 볼드는 재측정 결과에 따라 재판정) |
| **`:171`** (캡션) | "KUKA and Baxter use the **direct-pose configuration without render-and-compare**; …" | **이중으로 틀림.** (1) direct-pose를 쓴 진짜 이유는 설계 선택이 아니라 **솔버 경로가 K 버그로 망가져 있었기 때문**, (2) iiwa7 메쉬는 **존재한다**((g)절). → "KUKA and Baxter use the kinematic solver without render-and-compare" 류로 교체하고, direct-pose를 **방법의 특징으로 제시하지 말 것** |
| **`:245`** (본문, 합성 로봇 문단) | "**Because no matching mesh is available** for render-and-compare and no real data exist for self-training, poses are obtained directly from the heads, reaching ADD-AUCs of **35.7 for KUKA and 25.2 for Baxter**; these numbers are **not comparable** to the Panda real results and serve as an **applicability study**" | 세 군데가 바뀐다. (a) "no matching mesh" = **사실이 아님**(iiwa7 메쉬 확보). (b) 수치 → 69.0 / 71.3. (c) **포지셔닝 자체가 바뀐다** — 더 이상 "비교 불가한 적용 가능성 연구"가 아니라 **Baxter에서 명확한 SOTA, KUKA에서 경쟁력 있는 결과**다 |
| **`:245`** (같은 문장 후반) | "The analysis reveals an **observability ceiling at distal joints** … indicating a **robust limit of keypoint-based estimation**" | 이 인과 서사가 **핵심적으로 틀렸다.** Baxter 0.2739 → 0.7125는 손목 관측성이 **지배 요인이 아니었음**을 보인다. 관측성 천장은 잔존 2차 효과로 **격하**하거나 삭제 |
| **`:246`** | `:245`의 한국어 대역 주석 | 위와 동일하게 갱신 |
| **`:137` / `:138`** (평가 프로토콜 문단) | "KUKA and Baxter are evaluated on their synthetic splits" | 수치 자체는 없으나, **KUKA/Baxter 결과의 성격 규정**이 본문 전체와 일관해야 함. 별도 수정은 불필요하나 재작성 시 함께 확인 |
| **`:312` / `:313`** (결론) | "the analysis **quantified an observability ceiling at distal joints** … its angle remains under-determined even with perfect keypoints. Remaining limitations include … **the need for a benchmark-matched mesh** to extend the depth correction" | 두 주장 모두 근거가 무너졌다. (1) 관측성 천장을 결론의 기여로 제시하면 안 됨, (2) "메쉬 부재"는 iiwa7에 대해 **거짓**. 결론 문단 재작성 필요 |

### 2. `docs/dinobotpose3/PAPER_DRAFT.md` (구 초안 — tex의 소스)

| 위치 | 현재 | 되어야 할 것 |
|---|---|---|
| **`:181` / `:183`** | "KUKA·Baxter는 합성 스플릿에서 각각 **0.357·0.253**" | 0.690 / 0.713 (KUKA-Photo 대기) |
| **`:190`** | 표3 행: `| KUKA | 합성(synth) | 0.735 | 0.357 | 회전 헤드 병진 오차(56mm) |` | 포즈 값 **0.357 → 0.690**, **병목 열이 틀림** — `|dz|`는 ~1 m 장면의 3~4%로 정상 회귀 수준. 새 병목 = **잔존 link-identity 혼동 꼬리**(fail 15.9%, p99 1012 mm) |
| **`:191`** | 표3 행: `| Baxter 좌완 | 합성(synth) | 0.817 | 0.253 | 손목 관절 **관측성 천장** |` | **0.253 → 0.713**, 병목 열 **"손목 관측성 천장" 삭제** — 꼬리가 24.7%→8.0%로 붕괴했으므로 관측성이 지배 요인이 아니었음이 실측으로 증명됨 |
| **`:210`** | 표4 우리 행 `… | 35.7 | 31.9 | 25.2 |` | tex `:167`과 동일 교체 |
| **`:212`** | 표4 캡션 "KUKA/Baxter = direct-pose without render-compare (**no mesh**)" | tex `:171`과 동일 — **이중 오류** |
| **`:194`** | "KUKA/Baxter는 **합성·RC 미적용**이라 서로 직접 비교 불가" | 여전히 참(RC 미적용)이나, "비교 불가"라는 방어적 프레이밍은 Baxter 1위·KUKA 사정권에서는 **불필요하게 자기 축소적** |
| **`:206`** | 본문: "KUKA·Baxter는 우리 쪽에 render-compare가 **없어(정합 메쉬 부재, §4.7)** 낮으며 **공정 비교가 아니다** — 목적은 **성능 주장이 아니라** … 적용 가능성" | 원인 귀속(메쉬 부재)이 **거짓**이고, 결과가 뒤집혔으므로 **성능 주장을 해도 되는 상태**. 문단 전면 재작성 |
| **`:328` / `:330`** (§4.7) | "포즈는 head 각도 + 회전 헤드의 R,t를 직접 쓰는 direct-pose로 ADD-AUC **0.357**(KUKA)·**0.253**(Baxter)를 기록한다" | 서술 자체는 실행한 것에 정직하나 **방법의 능력을 KUKA +0.32 / Baxter +0.44 ADD-AUC만큼 과소평가**. 참 K 솔버 경로로 교체 |

### 3. 그림

| 위치 | 현재 | 되어야 할 것 |
|---|---|---|
| **`docs/dinobotpose3/figures/make_figs_multirobot.py:34`** | `pose = [0.804, 0.357, 0.253]   # ADD-AUC@100mm` | `[0.804, 0.6901, 0.7125]`. 3개 막대의 상대 높이가 완전히 달라지므로 **y축 범위·주석·캡션 동시 점검** 필요. 그림 재생성은 env `dino` |

### 요약 — 무엇이 문제인가

게재된 0.357 / 0.253은 **실제로 실행한 파이프라인에 대해 정직한 수치**다. 이것은 **철회(retraction) 사안이 아니라 인과 서사 교체 + 수치 상향** 사안이다. 다만 방향이 크다:

- 현재 논문은 KUKA/Baxter의 저조를 **방법의 원리적 한계**(손목 관측성 천장 · 정합 메쉬 부재)로 설명한다.
- 실제로는 **평가 코드의 구현 버그**였고, 두 설명 근거는 각각 **실측으로 반증**(Baxter 꼬리 붕괴)되고 **사실이 아님**(iiwa7 메쉬 존재)이 확인되었다.
- 결과적으로 **Baxter는 전 경쟁모델을 큰 격차로 이기고(71.3 vs 58.8), KUKA는 사정권에 든다(69.0 vs 75~80)**. 논문의 "적용 가능성 연구" 프레이밍은 **과소 주장**이 되었다.

---

## (h-2) 🔓 EXPERIMENTS.md·SUMMARY 결론 — **보류 아님, 뒤집힘(overturned)**

이전 판본은 이 결론들을 "재시험 전까지 **보류**"로 두었다. full-set 확정으로 **보류 사유가 소멸했고, 판정은 뒤집혔다.**

| 위치 | 현재 기재 | 새 판정 |
|---|---|---|
| `3_pose_models/DINObotPose3/EXPERIMENTS.md:961-968` (2026-07-12/13) | "**direct-pose** mode (trust head angles + rot-head R,t directly, **bypass 2D** → avoids link-confusion): KUKA **0.357**, Baxter **0.253**" 및 여기서 파생된 **"KUKA/Baxter는 솔버 각도정제 금지"** | 🔴 **OVERTURNED.** 솔버는 각도정제를 금지할 이유가 없었다 — **망가진 카메라를 먹고 있었을 뿐**이다. 참 K에서 솔버가 direct-pose를 KUKA +0.32 / Baxter +0.44로 **크게 능가**한다. "금지" 결론은 **철회**하고 **참 K 솔버를 기본 경로로 승격** |
| `3_pose_models/DINObotPose3/SUMMARY.md:53-54` | "**Cross-robot (direct-pose, synthetic-only, no RC)**: KUKA **0.357**, Baxter **0.253**" | 수치·구성 모두 갱신 대상 (참 K 솔버 0.690 / 0.713) |
| `3_pose_models/DINObotPose3/SUMMARY.md:57` | "KUKA **0.357/0.319**, Baxter DR **0.252** — TRAILS synth-specialized RoboPEPP/RoboPose" | "TRAILS"가 **더 이상 참이 아니다**(Baxter 1위). KUKA-Photo(0.319)는 **미측정** |
| `3_pose_models/DINObotPose3/SUMMARY.md:116` (REFUTED 목록) | "mlp_patch appearance angle head … → wrist under-determination은 **observability ceiling**이며 detection/architecture 실패가 아니다" | 🟡 **부분 뒤집힘.** "mlp_patch가 plain mlp를 못 이겼다"는 관측 자체는 유효하나, 여기서 도출한 **"따라서 관측성 천장이 Baxter의 병목"이라는 인과 결론은 무효** — Baxter는 참 K만으로 0.2739 → 0.7125이고 꼬리가 24.7%→8.0%로 붕괴했다. **REFUTED 항목의 근거 문장을 수정할 것** |
| `3_pose_models/DINObotPose3/SUMMARY.md:117` (REFUTED 목록) | "**Baxter render-and-compare**: silhouette-depth + wrist-shape ambiguity degrades pose (77→204 mm)" | ⚪ **유지(독립 사안).** Baxter RC 실패는 *앵커·게이트 부재*로 별도 규명됨([2026-07-22_gap_reexamination.md](2026-07-22_gap_reexamination.md) §13.8). K 버그와의 연관성은 **미측정** |

---

## (i) 측정 상태 정리

| 항목 | 상태 |
|---|---|
| `meta.K` 부재 스코프 (kuka/baxter 있음 / Panda 없음) | ✅ **실측 확인** (JSON 직접 검사) |
| 참 native K = fx,fy 320 / cx 320 / cy 240 | ✅ **실측** (`_camera_settings.json`) |
| post-crop fx 1.736 vs 555.4, cx −562 vs −6.9 | ✅ **실측** |
| Panda GT-2D + 항등 K → 깊이 4.9 mm, AUC 0.0000 | ✅ **실측** (sanity check) |
| **(b) KUKA-DR full-set (5997f): 0.3682 → 0.6901** | ✅ **확정 실측** |
| **(b) Baxter-DR full-set (5982f): 0.2739 → 0.7125** | ✅ **확정 실측** |
| KUKA 꼬리 잔존 (fail 15.4→15.9%, p99 1012 mm) | ✅ **실측** |
| Baxter 꼬리 붕괴 (fail 24.7→8.0%) | ✅ **실측** |
| outlier 비율 불변 (KUKA 21.1% / Baxter 11.6%) | ✅ **실측 — 로버스트 거부 가설 기각** |
| 검증①: direct 모드 56.1 mm/7.42°가 rot-head 학습 로그와 일치 | ✅ **실측** |
| 검증②: 재구성 K의 재투영 오차 median **0.0003 px** | ✅ **실측** |
| 검증③: 패치된 프로덕션 스크립트가 독립 에이전트 수치 비트 재현 | ✅ **실측** |
| Panda 무영향 (`geometric_K`가 참 K 통과 확인, 배포 0.804 불변) | ✅ **검증 완료** |
| bearing affine 관계 r = 1.00000000 | ✅ **실측** |
| KUKA RC: GT-pose IoU 0.858 / FK 0.0048 mm RMS | ✅ **실측** |
| **KUKA-Photo full-set (5999f): 0.3305 → 0.6984** | ✅ **확정 실측 — 논문 표 게이트 해제** |
| direct-pose 재측정이 아카이브 대비 +1~2점(3셀 모두 동일 방향) | ⚠️ **비트 재현 실패** — `best_*` vs `last_*` 추정, 결론 무영향, **논문 노출 없음** |
| KUKA RC: 0.2804 → 0.5721 | ⚠️ **50프레임에서 튜닝** — 재튜닝 중, **논문 반영 금지** |
| KUKA 솔버 0.690 + 선택적 RC = 0.708 (oracle 상한 0.745) | 🔄 **선택자 재튜닝 진행 중 — 미확정, 논문 반영 금지** |
| Baxter-Photo | ➖ **DREAM에 스플릿 자체가 없음** |
| 로버스트 outlier 거부가 참 K **위에** 추가 이득을 주는지 | ❌ **미측정** |
| RC 하이퍼파라미터의 수정 후 재튜닝 | 🔄 **별도 에이전트 진행 중** (현 튜닝은 **수정 전 baseline** 기준) |
| 참 K 솔버 ↔ RC 상호작용(가산/중복) | ❌ **미측정** |
| `inference_4tier_eval.py` zeros(3,3)가 실제 오염을 낸 사례 | ❌ **미측정**(잠재 이슈) |
| Baxter RC REFUTED 판정의 K 버그 연관성 | ❌ **미측정**(독립 사안으로 추정) |

---

## 다음 단계 (미해결 항목)

1. ✅ ~~KUKA-Photo 참 K 재측정~~ **완료(0.6984) — 게이트 해제, (A)층 논문 수정 적용 완료.**
2. 🔴 **(B)층 서사 재작성 — 저자 검토 대기.** 초안 [../PAPER_REVISION_DRAFT_2026-07-22.md](../PAPER_REVISION_DRAFT_2026-07-22.md). **최우선 = B-8**(`PAPER_DRAFT.md:328/330`에 옛 수치가 남아 파일 내부 불일치).
3. 🔄 **RC 재튜닝** — 현재 RC 하이퍼파라미터는 **수정 전(망가진 솔버) baseline 위에서 튜닝**된 것이다. 고쳐진 솔버 위에서 재튜닝해야 하며 **별도 에이전트가 진행 중**. 재튜닝 전 RC 수치를 논문에 반영하지 말 것.
4. **로버스트 outlier 거부의 추가 이득 측정** — 이번 이득에는 기여하지 않았음이 확정됐으나(비율 불변), 참 K **위에** 얹었을 때의 효과는 열려 있다. KUKA 잔존 꼬리(fail 15.9%, link-identity 혼동)의 유일한 유력 후보.
5. `geometric_K()`를 `Eval/` 공용 헬퍼로 승격(항등 아닌 K는 통과시키도록 가드 일반화) — `Eval/refine_eval.py`의 일반화 버전이 이미 Panda 참 K를 통과시킴을 확인.
6. `inference_4tier_eval.py:128-131` 폴백 → `raise`로 전환. 같은 "조용한 폴백" 패턴 전수 점검.
7. 아카이브 기준선 재측정 차이(3셀 모두 +1~2점)의 원인 확정 — `best_*` vs `last_*` 체크포인트 가설 검증.
8. ✅ ~~`SUMMARY.md` 갱신~~ **완료** — `:53-54`(→확정치+OVERTURNED 항목 신설), `:57`(합성 비교 갱신), `:116`(관측 존치 / 인과 추론만 REFUTED 분리), `:117`(유지 + 독립성 근거 명기), 그리고 **신규 REFUTED "head θ 고정/앵커 계열 전체"** 등재.
