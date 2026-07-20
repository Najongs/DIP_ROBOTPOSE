# 2026-07-20 — 논문 포지셔닝 재구성 + 파이프라인 피규어

> 실험이 아니라 **논문 서사·피규어 세션** 정리본. 수치는 전부 기존 실험(§4 ablation·SUMMARY REFUTED)에서 인용, 새 측정 없음.

## 산출물: 파이프라인 개요 피규어 `figures/fig_pipeline`

- 스크립트 `figures/make_fig_pipeline.py` (make_figs.py 컨벤션: serif·300DPI·png+pdf·동일 팔레트). §3.1 개요 그림, 번호 없음(LaTeX 단계 부여). **최종본은 PPT 재제작 예정 — 레이아웃·라벨·수치 참고 초안.**
- 내용: 메인 흐름(입력→Auto-bbox→Frozen DINOv3→Keypoint head+DARK→Angle/Rotation head→cov-PnP 솔버→출력) + **pass-1 피드백 루프**(solved pose→FK 재투영 bbox→2차 패스 = bbox-from-solved의 실체) + 하단 RC 브랜치(SAM+nvdiffrast→soft-IoU, per-camera 토글, azure OFF) + frozen(파랑)/trained(주황)/training-free·test-time(초록) 색 구분.
- 인셋: 입력·디코딩 키포인트·메쉬 오버레이 = 실제(RealSense #2700, ADD 19mm), 히트맵·SAM-IoU = 도식(로컬에 가중치 없음, GPU 서버에서 실물 교체 가능).
- 라벨 주의: "detect → solve → crop"의 detect는 **detector**(디터미넌트 아님). 디터미넌트가 실제 등장하는 곳은 DARK의 2×2 Hessian 역행렬.

## 결정 1 — 2축 스토리로 재구성 (무료 레버 격하)

ablation(표 6 leave-one-out, 표 8 build-up)과 서술 비중을 일치시킨다:

1. **인식 축**: 동결 파운데이션 프론트엔드로 서브픽셀 키포인트 확보 — 적응은 불필요를 넘어 **유해**(아래 증거 체인).
2. **기하 축**: 2D만으론 깊이 제약이 약함 → 테스트타임 RC 깊이 보정 = **최대 레버**(ΔMean +0.043, 원거리 +0.04~0.07) = 경쟁 4방법(RoboPEPP·HoRoPose·RoboKeyGen·RoboTAG)에 없는 구조적 차별점.
3. (보조) 가림 강건성 스택 — RoboPEPP 전 구간 우위의 근거, 유지.

무료 레버(DARK/cov-PnP/conf-gate)는 클린 ΔMean −0.003/−0.001/−0.001로 **"무료 강건성 레버"로 격하**: §3.2 압축, 서론의 "정확도를 공짜로 끌어올린다" 문구 교정(→ 가림 강건성 프레임). G1 분석(기전 규명)은 오히려 전면에.

## 결정 2 — frozen-백본을 명시적 기여로 승격 (증거 체인 4단)

1. **적응하면 나빠진다**: 공격적 SSL(6-block) ADD 0.567→0.531 / 온건 SSL(3-block) 헤드 완전 OOD, ADD 0.0 / pseudo-kp co-finetune 0.497→0.434 단조 하락 (표 11).
2. **왜 나빠지는가 (기전)**: 공격적 SSL에서 **real PCK@5 +0.069↑ 인데 ADD↓** — 적응은 거친 2D 강건성(굵은 임계 PCK가 보는 것)을 얻고 서브픽셀 정밀도(솔버가 요구)를 판다. "2D vs 3D"가 아니라 **"거친 2D vs 정밀 2D"**. G1이 정량 뒷받침: PCK@5가 못 보는 σ=1–2px에서 ADD-AUC −0.024~−0.089. 감독형 co-finetune도 실패 → 원인은 목적함수가 아니라 **백본을 움직이는 행위 자체**.
3. **동결이 충분하다**: known-joint 상한 0.841(§4.2) + G1 우아한 열화.
4. **특정 백본 의존 아니다**: §4.10 unfreeze 시 DINOv3≈SigLIP2(0.742≈0.752). DINOv3 채택 근거 = **frozen 체제 검출 우위(0.80 vs 0.72)**. → "왜 최신 모델 안 썼나" 리뷰 방어와 연결.

⚠️ **주장 경계**: "파운데이션 백본 최초 사용"은 금지 — CtRNet-X(CLIP 파인튜닝, known-joint), RoboPEPP(I-JEPA식 masking 사전학습 인코더) 선행. 신규성은 **완전 동결(frozen)로 충분 + 적응이 유해하다는 반증**이다 (references/related_work.md §차별점 4).

## 결정 3 — SAM 버전 (리뷰 대비)

- 현재 = **오리지널 SAM v1 ViT-B** (`segment_anything.sam_model_registry['vit_b']`, `sam_vit_b_01ec64.pth`). SAM2/SAM3 아님.
- **업그레이드 불필요 판단**: ① 프롬프트가 pass-1 렌더 유도 point+box + init-render-consistent 선택이라 SAM3의 컨셉/텍스트 프롬프트가 놀게 됨 ② 마스크는 병목 아님(known-joint 분석: 잔여 오차 = 예측 관절각; SUMMARY의 "SAM2로 개선" 노트는 구식 스플랫 렌더러 시절 처방 — 진짜 병목은 렌더러였고 nvdiffrast로 해결됨) ③ SAM3(~840M)는 테스트타임 런타임 표에 불리.
- 논문 §3.3에 구현 노트 추가(버전 명시 + 왜 충분한지 한 문단)로 선제 방어.
- **후속 후보 (P1, 미실행)**: `sam_ious` 분포 진단(공짜, rc 로그에 이미 있음) → 낮은-IoU 프레임 비율이 유의하면 SAM 버전 스왑 ablation(rc_dumps 재사용, 카메라당 재평가 1회). 결과 ±0.00x면 "컴포넌트 버전 강건" 주장으로 역이용.

## PAPER_DRAFT 반영 내역 (이 세션)

- §1 서론: 무료 레버 문구 교정(클린 미미·가림 강건성 프레임), **기여 2번 신설**(frozen 프론트엔드 = 발견) 후 재번호(1~5).
- §3.1: fig_pipeline 참조(이전 세션).
- §3.2: "왜 동결인가"(증거 체인+해리+G1 연결) 신설 + DARK/cov-PnP를 "무료 강건성 레버" 한 문단으로 압축.
- §3.3: 구현 노트(SAM v1 ViT-B, 프롬프트 방식, 왜 충분한지).
- §4.9: PCK↑/ADD↓ 해리를 본문·표 11에 수치로 추가 (기존엔 SUMMARY.md에만 있었음).
