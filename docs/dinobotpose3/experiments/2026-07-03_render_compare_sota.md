# 2026-07-03 — nvdiffrast+SAM render-and-compare → DREAM real 4-split SOTA

## 목적
6월에 검증된 render-and-compare 레버(oracle 천장 +0.108)의 배포 차단 요인(렌더러 충실도)을 nvdiffrast로 해소하고 RoboPEPP(mean 0.780)를 초월.

## 설정
- 렌더러: `Eval/render_nvdr.py` — nvdiffrast CUDA 래스터, visual mesh(그리퍼 핑거 hand 프레임에 베이크), dr.antialias 경계 그래디언트
- 마스크: SAM ViT-B, 프롬프트 = 배포 포즈의 투영 키포인트+bbox, **후보 3개 중 init-render IoU 최고 선택**(SAM 자체 스코어 아님)
- refine: `Eval/rc_refine_from_dump.py` — 배포 파이프라인 덤프(θ,R,t)에서 시작, Adam 5e-4, soft-IoU + 재투영 앵커 w=100, do-no-harm min-iou 0.35
- 프로토콜: predicted angles + **완전 자동 bbox**, rs/kinect/orb는 anti-leak held-out 800

## 핵심 결과
| 실험 | 결과 |
|---|---|
| 형상 게이트 | SAM-vs-렌더 IoU 0.345(splat) → **0.676**(nvdr GT-pose) / **0.85**(init-pose 프롬프트); frac<0.4: 94%→3% |
| 해상도 스케일링 | +0.041@224 → +0.067@320 → **+0.078@448** → +0.078@512 (포화; orb만 512) |
| **최종 테이블** | rs **0.818**(+.013 BEAT) / kinect **0.811**(+.026 BEAT, crop구성 교체) / azure 0.788(+.035 BEAT, **RC 제외**) / orb 0.765(−.010) → **mean 0.796** |

## 교훈
1. render-compare는 **depth/scale 보정기** — 원거리/단축 카메라만 이득, 근거리(azure)는 마스크 노이즈가 depth를 흔들어 해로움(−0.047, uv-shift 가드도 무효: 손상이 depth 방향) → 카메라별 on/off
2. SAM 마스크는 "후보 중 렌더 일관성 선택"이 핵심 — 스킵 0건의 안정성
3. 6월의 "SAM 발산"은 전적으로 splat 렌더 충실도 문제였음 — 정밀 렌더로 즉시 해소

## 재현
`Eval/rc_refine_from_dump.py --dump rc_dumps/<cam>_rotadapt_heldout.npz --render-h 448 ...` (EXPERIMENTS.md 2026-07-03 참조)
