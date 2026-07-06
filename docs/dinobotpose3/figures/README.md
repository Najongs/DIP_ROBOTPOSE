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

## 정성 확인 (qualitative overlay) — `qualitative/`

실제 DREAM 프레임 위에 파이프라인 추정 포즈를 겹쳐 **눈으로** 확인. GREEN=GT 스켈레톤, RED=예측 FK 재투영(모델 실제 포즈), YELLOW=검출 원시 2D, CYAN=가림체 뒤라 conf-gate된(운동학이 추론한) 키포인트. PNG는 gitignore(재생성 가능) — 스크립트는 추적됨.

| 파일 | 내용 |
|---|---|
| `qual_{cam}_clean.png` | (스켈레톤) 클린 6장 — RED 예측이 실제 팔에 밀착 (ADD 15~43mm) |
| `qual_{cam}_ladder.png` | (스켈레톤) 가림 0→40% 에스컬레이션 — RED이 GT 추종, 우아한 열화 |
| `qual_{cam}_mesh.png` | **(메쉬 실루엣) 예측 포즈로 렌더한 Panda 메쉬(nvdiffrast, 오렌지)를 실제 이미지에 반투명 오버레이** — 팔 전체가 실제 로봇에 링크 단위로 정합. azure 특히 픽셀-퍼펙트 |
| `qual_{cam}_mesh_ladder.png` | (메쉬) 가림 0→40% — 가림체가 팔을 덮어도 메쉬가 추론된 전신 포즈로 가림체 위에 렌더 = "숨은 팔의 위치를 안다"는 시각 증거 |

관측(kinect ladder, frame #2400): ADD 13→15→34→70→107mm (0/10/20/30/40%), 40%에서 3/7 키포인트가 가림체 뒤(CYAN)인데도 포즈 근사 유지 = **"가려져도 대략 추론"의 시각적 증거**. azure ladder(#3000): 19→27→26→43→141mm. realsense(#2700): 13→14→102→153→137mm.

재생성:
```bash
cd 3_pose_models/DINObotPose3/Eval
G=GPU-<uuid>
DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
# 클린 오버레이
CUDA_VISIBLE_DEVICES=$G python viz_results.py --detector $DET \
  --mlp-head ../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth \
  --val-dir ../Dataset/Converted_dataset/DREAM_real/panda-3cam_azure \
  --indices 40,1000,2000,3000,4000,5000 --out viz_outputs/qual_azure_clean.png
# 가림 에스컬레이션 (가림-강건 light head)
CUDA_VISIBLE_DEVICES=$G python viz_occlusion.py --detector $DET \
  --mlp-head ../TRAIN/outputs_angle/angle_occaug_light_20260704_015400/best_angle_head.pth \
  --val-dir ../Dataset/Converted_dataset/DREAM_real/panda-3cam_azure \
  --ladder "3000:0,0.1,0.2,0.3,0.4" --cols 5 --out viz_outputs/qual_azure_ladder.png
# 메쉬 실루엣 오버레이 (예측 포즈로 Panda 메쉬 렌더 → 실제 이미지 위 반투명)
CUDA_VISIBLE_DEVICES=$G python viz_mesh.py --detector $DET --mlp-head $CLEAN \
  --val-dir ../Dataset/Converted_dataset/DREAM_real/panda-3cam_azure \
  --indices 40,1000,2000,3000,4000,5000 --gt-skel --out viz_outputs/qual_azure_mesh.png
# 메쉬 + 가림 에스컬레이션 (가림-강건 head)
CUDA_VISIBLE_DEVICES=$G python viz_mesh.py --detector $DET \
  --mlp-head ../TRAIN/outputs_angle/angle_occaug_light_20260704_015400/best_angle_head.pth \
  --val-dir ../Dataset/Converted_dataset/DREAM_real/panda-3cam_kinect360 \
  --ladder "2400:0,0.1,0.2,0.3,0.4" --cols 5 --out viz_outputs/qual_kinect_mesh_ladder.png
```
메쉬 오버레이는 pipeline이 푼 키포인트에서 Kabsch로 카메라 포즈(R,t)를 복원 → `render_nvdr.render_shaded`로 정확 메쉬를 예측 포즈에 렌더 → 실제 이미지에 alpha-blend. RC(render-and-compare)가 최적화하는 그 실루엣 정합을 눈으로 확인하는 것.

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
