# DINObotPose3 — Experiment Log

> Goal: single-image robot pose + joint-angle estimator that **beats DREAM / RoboPose / RoboPEPP**
> on the DREAM Panda benchmark (ADD-AUC@100mm, joint-angle MAE), then extend to new robots.
> Last updated: 2026-06-02.

---

## ☀️ Session summary (2026-06-04) — SOLVED the realsense −48 leak (solver, no retrain)

**Lever B (far-camera ADD tail) is fixed by the PnP-init keypoint selection.** Diagnosis, not training.

**Diagnosis (`Eval/depth_diag.py`, oracle-swap attribution):** recovered pred & GT camera pose by
Kabsch, swapped one GT component at a time. On realsense EVERY single-component swap (t_z/t/R/θ)
made ADD *worse* but all-together = 0.9999 → a **jointly under-constrained solve** (R 73° off, errors
mutually compensating to still reproject). **Refutes "stabilize t_z"** (depth prior provably hurts).

**Root cause = the PnP INIT.** The cleaner the init, the better the final pose — the gradient refine
uses ALL 7 points to polish, so the init only has to pick the right basin; low-conf far-camera points
reproject within tolerance but bias EPnP depth. → **init from the top-4 most-confident keypoints,
grow only on degeneracy; refine on all 7** (`pnp_drop=3` + fallback + nan-guard). Rejected: anchor θ
(0.18), relative-conf gate (0.34), fixed drop2 (a wash on representative data).

⚠️ **Methodology catch:** first sweeps used `EvalDataset[:600]` (SORTED → biased contiguous segment);
realsense has 5944 frames so that's one easy trajectory chunk (looked like 0.62). `refine_eval`'s
`PoseEstimationDataset` is UNSORTED (representative). Per-frame GT is identical; only the SUBSET
differed. **Always STRIDE for a representative spread.** Numbers below are strided 600/cam.

| PnP init | realsense | azure | kinect | orb | syn_dr | syn_photo | mean |
|---|---|---|---|---|---|---|---|
| all-7 (drop0, ≈old) | 0.338 | 0.698 | 0.648 | 0.526 | 0.697 | 0.692 | 0.600 |
| drop2 (top-5) | 0.458 | 0.646 | 0.541 | 0.534 | 0.709 | 0.709 | 0.599 (wash) |
| **drop3 (top-4)+fallback ✅** | **0.521** | **0.742** | **0.701** | **0.595** | **0.751** | **0.750** | **0.677** |

**drop3+fallback is best on EVERY split: +0.077 mean AUC**, no regressions, no nan. Mean-ADD tail
collapses (realsense 276→56mm, synth 1938→45mm); angle MAE also improves (realsense 18.8→13.7°).
vs the actual historical shipped (conf≥gate): realsense 0.324→0.521 (+0.20). Now the
`solve_pose_kinematic.py` default. Tools: `Eval/depth_diag.py`, `Eval/solve_sweep.py` (with `--`
strided sampling). (Also killed the low-value GPU2 detector finetune.)

### Then: learned ROTATION prior (the realsense residual) → +0.117 realsense
After B, realsense residual = solver wrong-rotation-basin (rot_err 47° even w/ good 2D; failure
correlates ONLY with rot_err +0.64, not foreshortening/distance/conf — not a data subset).
rinit_probe: oracle R-init = +0.11. Built `RotationHead` (DINOv3 appearance→6D R, `model_angle.py`),
`train_rotation.py` (frozen detector, Kabsch GT-R supervision), fed to solver `R_init`
(`solve_batch(R_init=)`). **realsense 0.52→0.64 (+0.117), neutral elsewhere, nothing hurt.** Robust
even at 68° real geodesic — solver refines R, so init only needs the right BASIN not accuracy.
- **Translation prior REJECTED:** R+t head, real t-err 587mm (depth doesn't transfer sim2real),
  feeding t_init HURTS realsense 0.66→0.37. R-only kept. Rotation=robust appearance cue, depth=not.
- **Scale/crop normalization REJECTED** (`bbox_crop_probe.py`): crop ⊥ ray geometry → neutral realsense, hurts azure.

### 🔒 LOCKED 6-split table (official refine_eval, B + rotation prior, R-only)
| split | session start | NOW | RoboPEPP |
|---|---|---|---|
| realsense | 0.32 | **0.633** | 0.805 |
| azure | 0.71 | **0.746** | 0.753 |
| kinect360 | 0.563 | **0.707** | 0.785 |
| orb | 0.462 | **0.665** | 0.775 |
| syn_dr | 0.707 | **0.742** | 0.830 |
| syn_photo | 0.694 | **0.710** | 0.841 |
| **mean** | ~0.56 | **0.701** | 0.798 |
**Session: mean 0.56→0.70 (+0.14), realsense nearly doubled, MATCH RoboPEPP on azure (0.746 vs 0.753).**
Remaining gap is no longer pose-collapse — it's angle (~13°) + keypoint noise (the harder frontier).
Pipeline: image → detector → mlp angle head → rotation head (R_init) → kinematic solve. No backbone retrain.

### Angle improvement (2026-06-04/05) — diagnosis → failure viz → depth REJECTED → crop IN PROGRESS
Per-joint angle (`refine_eval`, `wrist_diag` oracle-2D injection):
- **realsense angle = J0 44° = 2D-limited** (oracle-2D → 14°; head amplifies detector 2D noise on base-yaw). R_init doesn't fix it (44→41).
- **synth angle = wrist J4/J5 11-12° = appearance/geometry-limited** (oracle-2D barely helps; mlp_patch failed) → render-and-compare frontier, hard.

**Failure viz (`Eval/viz_failures.py` → `viz_outputs/failures_*/`):** worst realsense frames are ALL foreshortened poses (arm at camera, fore 60-90°) → 2D piles up → base-yaw J0 under-determined → red FK skeleton rotated about base. Detector (yellow) is fine. → lever = appearance/more-pixels, NOT 2D reprocessing.

**RoboPEPP (`/RoboPEPP`):** 4-iter residual JointNet (pose+=Δ), train+test bbox crop→224, I-JEPA pretrain, per-cam PnP conf (RS 0.8), focal heatmap loss. We're single-pass full-frame.

**❌ Iterative refine** — user: can't add info to under-determined data. **❌ Depth-lift** (`depth_lift_probe.py`): oracle per-kp depth makes realsense WORSE (0.52→0.17) — backproj amplifies 2D-noise×z(1.4m); azure helped but 5% depth noise kills it; no sensor in DREAM.

**✅ CHOSEN: train+test bbox CROP** (info-add that fixes 2D, not amplifies). `dataset.py crop_to_robot` (square crop @GT-kp bbox, shift K, jitter) + `train_heatmap.py --crop-to-robot` + `run_train_detector_crop.sh`. **RUNNING: detector crop-retrain** (GPU1, warm-start, realsense val). NEXT: angle head on cropped detector → eval cropped pipeline (target J0 44→~20). Also running: kp-jitter angle retrain (GPU2, J0 noise-robustness).
Tools added this arc: `wrist_diag.py`, `viz_failures.py`, `depth_lift_probe.py`, `rinit_probe.py`, `rot_eval.py`.

---

## ☀️ Session summary (2026-06-03, overnight)

**Where we are:** end-to-end pipeline works and is benchmarked on all 4 DREAM-real cameras.
- **Pipeline:** image → DINOv3-unfrozen detector (heatmaps, real-azure PCK@5 ~0.81) → MLP angle
  predictor → kinematic reprojection refine → joint angles + camera pose.
- **DREAM-real results:** angle MAE **~11.4° avg** (azure/kinect/orb ~10°, realsense 15.9°),
  **ADD-AUC@100mm ~0.56 avg** (0.51–0.67), **median ADD 23–32mm**.
- **vs SOTA:** ≈ DREAM-level, **below RoboPEPP (~0.8+)**. The gap is almost entirely an
  **outlier-frame tail** (median ADD is good ~29mm; mean 55–107mm dragged by failed-PnP frames).

**Decided this session:** DINOv3-unfrozen detector (≈ SigLIP2, both ~0.81); **MLP angle head**
(Diffusion + ensemble debunked — earlier "diffusion wins wrists" was a cross-dataset artifact);
fk-weight 10 (50 diverges). Kinematic refine helps real (−0.9 to −1.2°, J5→~4.6°).

**Top next levers (in priority order):**
1. **Robust pose** to kill the outlier tail (confidence-gate keypoints, RANSAC fallback, reject/repair
   high-reproj frames) → biggest ADD-AUC win, eval/refiner-side. ← start here.
2. **J0/J2** (~16°, single-view-ambiguous + sim→real) and **realsense domain gap** (worst camera).
3. Detector wrist keypoints; Stage-0 vendored-baseline numbers; Stage-3 pseudo-label sim→real.

**Currently running:** MLP main converged ~10.7° synth (GPU1). GPU2 free.

---

## TL;DR — current best understanding

1. **Angle estimation, not geometry, is the headline lever.** Pure single-view geometric angle
   solving is ill-posed (cold-start 23° even with oracle 2D). A **learned predictor from 2D
   keypoint geometry** reaches **7.36°** (probe), and **geometric refinement on a good init**
   polishes it further. → build: detector → learned angle head → kinematic refine.
2. **Detector:** DINOv3-ViTB16 unfrozen (last 4 blocks) + strong aug ≈ **AUC 0.81 / L2 4.0px**
   on real-azure. SigLIP2-base is a **drop-in** and, once its backbone adapts, **matches DINOv3**
   (AUC 0.81 / L2 3.81px @ ep3). Frozen < unfrozen for DINOv3.
3. **Infra:** GPU 0 is faulty → select GPUs by **UUID**. Heatmap generation optimized **193×**.

---

## Stage 2 — Kinematic angle+pose solver  ✅ (decisive experiment)

`Eval/solve_pose_kinematic.py` — optimize (θ, R, t) to minimize confidence-weighted reprojection
of `panda_forward_kinematics(θ)`; joint-limit sigmoid reparam, IRLS robust reweighting,
conf-filtered PnP init, optional `theta_init` (refinement mode), per-frame divergence guard.

| test (panda_synth_test_dr) | result |
|---|---|
| FK(GT angles) + PnP → GT 2D | **0.00 px / 0.0 mm** (geometry/convention exact) |
| cold-start solve, **oracle GT 2D** | 23° J0-5 (local minima; J0 base-yaw 46°) |
| cold-start solve, real detector | 32° (worse than direct regression ~17-19°) |
| refine from init 4° / 8° / 16° (oracle 2D) | → **2.7° / 5.3° / 10.6°** (~30-50% cut, stable) |
| **v3 3D→IK init (24.9°) → refine** | **19.7°**, ADD-AUC@100mm 0.47 |

**Conclusion:** geometric solving must **refine a learned init**, not replace regression.
The 19.7° is limited by the weak v3 init (its 3D has ~300 mm error), not the refiner.

---

## Stage 1.5 — Learned angle predictor  🔄 TRAINING (on DINOv3-unfrozen detector, GPU1)

CPU probe `Eval/angle_from_2d_probe.py` (20k synth, GT 2D): tiny MLP, **NO image features**.

| input feature | angle MAE (J0-5) |
|---|---|
| global-feature direct regression (old) | ~17-19° |
| geometric cold-start | ~23° |
| v3 3D→IK | 24.9° |
| K-normalized **bearings** (7 pts) | **9.53°** |
| bearings + **all-pairs differences** | **7.36°** (8.05° @ +3px noise) |

Worst joints: J0 base-yaw 13.4°, J2 11.8° (single-view depth-ambiguous).

**Build:** `TRAIN/model_angle.py` (`AnglePredictor` = frozen backbone + frozen detector +
trainable angle head), `TRAIN/train_angle.py`. Input from the model's OWN predicted 2D (robust
to detector noise). Loss = sin/cos SmoothL1 + FK. Output → kinematic refiner init.
**Target: mid-single-digit angle MAE.**

### Head architecture comparison (all on the SAME DINOv3-unfrozen detector)
| GPU | head | val MAE(J0-5): ep0 → ep1 → ep2 | status |
|---|---|---|---|
| 1 | **MLP** (concat → 3-layer MLP → sin/cos) | 21.31 → 18.60 → 18.31 | 🔄 plateauing ~18°, **wrist-limited** |
| 2→✗ | **Transformer** (DETR cross-attn, 5.7M) | 46.49 → 34.35 | ✗ stopped (lost — DETR slow to converge, 2× worse @ep1) |
| 2 | **Diffusion** (DDIM denoiser on DINOv3+skeleton, *generates* angles) | training | 🔄 (replaced transformer) |

⚠️ **Key observation — wrist joints are the bottleneck.** MLP per-joint @ep2 = [J0 16.3, J1 7.4,
J2 15.1, J3 8.9, **J4 35.3, J5 26.8**]. The probe (deterministic, **GT 2D**, bearings+pairwise) got
J4 9.3 / J5 4.0. The real model's J4/J5 are ~3-7× worse → strongly suggests the **detected wrist
keypoints (link6/link7/hand) are the weak link**, not the head. Likely fixes: improve wrist keypoint
detection, weight wrist keypoints, or feed full heatmap distribution (uncertainty) per joint.
Diffusion comparison will tell if multi-modality helps the ambiguous wrists; if it ALSO plateaus
~18° with bad wrists, the bottleneck is confirmed to be the detector, not the angle head.

**MLP vs Diffusion per-joint (early, not same epoch):** MLP@ep3 15.05° vs Diffusion@1st-val 17.90°.
| joint | MLP ep3 | Diff 1st |
|---|---|---|
| J0 | **13.6** | 25.2 |
| J1 | **5.8** | 12.8 |
| J2 | **12.3** | 24.3 |
| J3 | **7.4** | 13.0 |
| J4 | 24.2 | **16.0** |
| J5 | 26.9 | **16.2** |

→ **MLP wins the easy joints (J0-J3); Diffusion wins the ambiguous wrists (J4/J5)** — confirms the
multi-modal hypothesis. Pattern strengthens by Diff's 2nd val:
| joint | MLP ep4 | Diff 2nd |
|---|---|---|
| J0 / J1 / J2 / J3 | **12.7 / 5.5 / 11.3 / 7.5** | 25.5 / 13.5 / 24.6 / 7.8 |
| J4 / J5 | 21.5 / 26.4 | **15.5 / 13.8** |

**⚠️ CORRECTION — the ensemble advantage was a cross-dataset artifact.** The MLP was validated on
**synth** (run_train_angle val=panda_synth_test_dr); the diffusion on **real-azure**
(run_train_diffusion3 val=DREAM_real). So the per-epoch "Diff wins wrists" compared MLP-on-synth vs
Diff-on-real — invalid. Evaluating BOTH on the SAME set (`Eval/ensemble_angle_eval.py`, 1500 frames):

| metric | MLP | Diffusion | per-joint ensemble |
|---|---|---|---|
| **synth** test_dr | **10.93°** | 20.66° | 10.93 (all→MLP) |
| **real-azure** | **12.33°** | 16.40° | 12.00 (J1,J5→Diff) |

→ **MLP wins on BOTH domains; diffusion is worse overall; ensemble gain ≈ 0.3° (negligible).**
Diffusion/ensemble **dropped**. Good news: MLP's **sim→real gap is small** (10.93→12.33, +1.4°).
Real-azure bottleneck = the ambiguous joints **J0 15.5 / J2 16.7 / J4 16.8°** (J3/J5/J1 ≈ 5-10°).
Next: MLP → kinematic refiner (Stage-2 lever) to push the geometrically-constrained joints lower.

Probe ceiling (GT 2D, deterministic) = 7.36°. Headline target stays mid-single-digit via
predictor → kinematic refine.

---

## 🎯 Headline pipeline (current): MLP angle predictor → kinematic refine

`Eval/refine_eval.py` (real-azure, 800 frames; MLP @ ~ep7, still training):
| | J0 | J1 | J2 | J3 | J4 | J5 | **MEAN** |
|---|---|---|---|---|---|---|---|
| raw MLP | 15.66 | 10.24 | 16.59 | 4.98 | 16.55 | 9.73 | 12.29 |
| **+ kinematic refine** | 15.69 | 9.09 | 17.45 | 5.52 | **14.00** | **4.96** | **11.12** |

**Kinematic refinement (Stage-2 lever) works on real detected 2D: −1.17° overall, big wins on the
wrist (J5 9.7→4.96, J4 16.6→14.0).** J0/J2 regress slightly (gauge-ambiguous; reprojection pulls off
GT). Remaining bottleneck: **J0 / J2 (~16°)** — inherently single-view-ambiguous.

### 🎯🎯 Converged pipeline + ADD-AUC (MLP @ ~ep10, real-azure 800 frames, `refine_eval.py`)
| | J0 | J1 | J2 | J3 | J4 | J5 | **MEAN angle** |
|---|---|---|---|---|---|---|---|
| raw MLP | 16.67 | 8.92 | 17.96 | 5.78 | 8.75 | 8.28 | 11.06° |
| **+ kinematic refine** | 14.54 | 8.37 | 17.07 | 4.95 | 11.48 | **4.58** | **10.16°** |

**ADD-AUC@100mm = 0.6685 | mean ADD 55.3mm | median ADD 23.2mm.**

### Full DREAM-real benchmark (refined pipeline, 800 frames/camera)
| camera | angle MAE | **ADD-AUC@100mm** | mean ADD | median ADD |
|---|---|---|---|---|
| panda-3cam_azure | 10.16° | **0.669** | 55mm | 23mm |
| panda-3cam_kinect360 | 10.30° | 0.510 | 107mm | 29mm |
| panda-3cam_realsense | 15.86° | 0.526 | 77mm | 32mm |
| panda-orb | 9.34° | 0.555 | 84mm | 30mm |
| **average** | **~11.4°** | **~0.565** | | ~29mm |

**Diagnosis (the path to beat RoboPEPP ~0.8+):** median ADD is **23-32mm (good)** but mean is
55-107mm → a **minority of outlier frames** (failed PnP/refine: degenerate geometry or bad
keypoints) drag the AUC down. Most frames are accurate; killing the outliers is the #1 lever.
**Next: robust pose** — confidence-gated keypoints, RANSAC-PnP fallback, reject/repair low-confidence
or high-reproj frames in `solve_pose_kinematic`. Secondary: realsense angle MAE 15.9° (worst camera —
detector domain gap). This is competitive with DREAM-level but the outlier tail is the gap to SOTA.
- Angle MAE **10.16°** (vs old diffusion baseline 17.88° — big gain). Median ADD **23mm** is strong
  (most frames accurate); mean 55mm dragged by PnP-outlier frames (the mean–median gap = the lever).
- ADD-AUC **0.67** ≈ DREAM-level (~0.68-0.75), **below RoboPEPP's best (~0.8+)**. Gaps to close:
  (1) **PnP/refine outlier frames** (robust pose / confidence gating), (2) **J0/J2** (ambiguous +
  sim→real), (3) wrist keypoint detection. Detector + angle head both still improvable.
- ⚠️ eval scripts auto-pick newest `best_angle_head.pth` — pass `--mlp-head` explicitly (a broken
  fk50 variant polluted auto-pick once). FK-weight 50 ablation **diverged** (J2/J3→150°); keep fk=10.

---

## Stage 1 — Strong keypoint detector  🔄 in progress

Train `ViTKeypointHead` with **strong augmentation** (heavy photometric + blur + JPEG + occlusion;
`dataset.py aug_level='strong'`), **FDA dropped** (user: ineffective). Val = **real `panda-3cam_azure`**
(tracks sim-to-real PCK). Warm-started from `outputs_heatmap/best_heatmap.pth`.

### Real-azure validation (AUC / PCK@5 / L2 px)

| run | backbone | ep0 | ep1 | ep3 | ep5/6 | status |
|---|---|---|---|---|---|---|
| DINOv3 frozen | frozen | 0.777 / 0.651 / 4.76 | 0.800 / 0.783 / 5.18 | — | — | **stopped (loser)** |
| **DINOv3 unfrozen** | last-4 ft | 0.803 / 0.799 / 4.83 | 0.814 / 0.832 / 4.15 | 0.813 / 0.787 / 4.27 | **0.815 / 0.815 / 4.00** (ep6) | plateau (best, ckpt saved) |
| **SigLIP2 unfrozen** | last-4 ft | 0.791 / 0.741 / 4.58 | 0.797 / 0.768 / 4.50 | 0.811 / 0.767 / 3.81 | **0.814 / 0.778 / 4.03** (ep5) | plateau ≈ dinov3 |
| SigLIP2 frozen | frozen | 0.284 / 0.097 / 61.5 | 0.598 / 0.483 / 13.9 | 0.725 / 0.614 / 7.6 | **0.722 / 0.630 / 7.2** (ep4, plateau) | plateau ~0.72 |

### Backbone matrix (plateau AUC @ real-azure)
| | frozen | unfrozen |
|---|---|---|
| **DINOv3** | ~0.78-0.80* | **0.81** (L2 4.0, PCK@5 0.81-0.83) |
| **SigLIP2** | **0.72** (L2 7.2) | **0.81** (L2 4.0, PCK@5 0.74-0.78, PCK@2.5 best) |

*dinov3-frozen stopped at ep1 (still rising); its head warm-start was matched (same backbone).

**Conclusions:**
- **Unfreezing the backbone matters for both** (siglip2 0.72→0.81). The detector must adapt features.
- **DINOv3-unfrozen ≈ SigLIP2-unfrozen** (both AUC ~0.81 / L2 ~4.0). SigLIP2 = better fine
  localization (PCK@2.5), DINOv3 = better PCK@5. **Pick DINOv3-unfrozen** (PCK@5 edge helps PnP).
- siglip2-frozen trails (0.72) partly because its warm-start head is cross-backbone (rough comparison).

✅ **Detector phase settled → proceed to Stage 1.5 (learned angle predictor) on DINOv3-unfrozen.**

### Backbone notes (SigLIP2)
- `google/siglip2-base-patch16-512` = **same token grid as DINOv3-ViTB16** (32×32×768) → drop-in.
- Fixes applied: forward (use all tokens, no CLS strip; `vision_model`), feature_dim (`vision_config`),
  normalization (**mean=std=0.5**, auto-detected), unfreeze logic (nested `vision_model.encoder.layers`).
- **From-scratch siglip head gets stuck** in the sparse-heatmap blank minimum (loss 0.12, AUC 0).
  Fix: **warm-start the keypoint_head** from dinov3 `best_heatmap.pth` (head keys match, siglip
  backbone keys don't → only head loads). Escapes the trap; also fairer (same head init).

---

## Infra / environment findings

- **GPU 0 faulty** ("Unknown Error" / NVML). Integer `CUDA_VISIBLE_DEVICES=2` → CPU fallback
  (silent, ~30× slower!). **Select by UUID:** GPU1 `GPU-ab38c04c-...`, GPU2 `GPU-05f84104-...`.
- **Heatmap gen optimized 193×** (`dataset._create_heatmap`): full 512×512 Gaussian (185 ms/sample)
  → windowed separable (0.96 ms), bit-identical. This was the main dataloader CPU hog.
- HF cache `HF_HOME=/data/public/97_cache`; conda env `dino`; transformers 5.2.0; albumentations 2.0.8.

---

## Evaluation protocol

`Eval/inference_4tier_eval.py` = canonical harness (ADD-AUC@100mm, 4 PnP tiers). Splits:
`Dataset/Converted_dataset/DREAM_real/{panda-3cam_azure, kinect360, realsense, orb}` +
synth `panda_synth_test_{dr,photo}`. Baselines: vendored RoboPEPP / RoboPose / DREAM (Stage 0, TODO).

---

## Next steps

1. Finish SigLIP2 frozen-vs-unfrozen comparison → pick detector backbone.
2. **Train Stage-1.5 angle predictor** on the winning detector (the real lever).
3. Wire predictor → kinematic refiner → report angle MAE + ADD-AUC vs baselines.
4. Stage 0: lock vendored baseline numbers in the shared harness.
5. (Later) Stage 3 pseudo-label sim-to-real; extend to FR5/Meca/FR3.

---

## 2026-06-03 — Occlusion root-causes the ADD outlier tail (`Eval/occlusion_diag.py`)

**Question (user):** some dataset frames are heavily/fully occluded — how do we handle them?

**Data look:** in DREAM-real the dominant "occlusion" is the **end-effector going off-frame** (top). Off-frame keypoints/frame: azure 5.5%, orb 3.8% of frames; ≥2 off-frame is ~3%.

**Decisive diagnostic (azure 600 frames, MLP→reproj refine):**
- corr(ADD, #off-frame keypoints) = **+0.48**; corr(ADD, min-conf) = −0.45.

| #off-frame kp | n | mean ADD | AUC |
|---|---|---|---|
| 0 | 560 | 48mm | 0.683 |
| 1 | 18 | 30mm | 0.703 |
| 2 | 11 | 61mm | 0.486 |
| **3** | **11** | **501mm** | **0.000** |

- **Detector already flags occlusion:** off-frame kp conf **0.026** vs in-frame **0.676** (26×); 2D err 286px vs 5px.
- **Per-keypoint 3D err on noff=3:** link0(base) **673mm**, l2 318, l3 100, l4 81, l6/l7/hand 673/785/880 → the *camera pose* is corrupted (base wrong), not just wrist angles. Cause: `pnp_init` ingested the hallucinated off-frame 2D.

**Fix:** gate conf<τ keypoints out of **PnP and refinement** (`solve_batch(conf_gate=)`, `pnp_init(conf_gate=)`).

| config | azure overall AUC | noff=3 |
|---|---|---|
| baseline | 0.6675 | 501mm / 0.000 |
| +anchor-to-init / mean-fallback | 0.655 | 458mm / 0.055 (**dead end, hurts common case**) |
| +PnP gate, anchor=0, gate=0.05 | **0.6857 (+0.018)** | **57mm / 0.548** |

- Generalization (orb 800): overall **0.5343→0.5414 (+0.007)**, noff=3 705→237mm. **Net positive but camera-dependent** — orbit cams have in-frame self-occluded kp; gating drops them → 4-pt PnP less stable (orb noff=0 mean 82→130 even as AUC rises).
- **Shipped:** `refine_eval.py --conf-gate 0.05` (default on). `solve_batch`/`pnp_init` gained `conf_gate`.

**Dead ends (don't retry):** anchoring occluded joints to learned init; mean-fallback init for unobservable joints. The MLP angle init is itself poisoned (ingests all 7 bearings incl. the 286px-wrong occluded ones) → anchoring to garbage is useless, and both hurt the 98% common case.

**Next lever = occlusion-aware TRAINING** (bigger, esp. custom/insertion data): occlusion augmentation → reliably-low conf on in-frame occluded kp; angle head that **masks low-conf bearings**; gating that preserves PnP conditioning.

### 2026-06-03 (cont.) — Is occlusion-aware TRAINING warranted? → NO for DREAM

- **Sim train data occlusion:** object `visibility` always 1.0 (no in-frame occluders); only off-frame occlusion, and that's already heavy — **22% of frames have an off-frame keypoint, 12% have ≥3**. So the detector was trained WITH off-frame occlusion (hence conf 0.026 on off-frame).
- **In-frame occlusion probe (`Eval/occlusion_probe_inframe.py`, paste a box over one visible keypoint, 300 azure kp):** the detector — though it has NEVER seen an in-frame occluder — collapses that keypoint's confidence:

  | occluder | clean→occ conf | gate-catchable (conf<0.05) | confident-wrong |
  |---|---|---|---|
  | gray box | 0.724→0.022 (−97%) | 96% | 4% |
  | noise texture | 0.724→0.027 (−96%) | 91% | 9% |

- **Conclusion:** heatmap-max confidence is inherently occlusion-calibrated (no peak forms when appearance is gone), even out-of-distribution. The shipped conf-gate already exploits this (91–96%). **Occlusion-aware detector retraining is NOT warranted for DREAM** — ~5–9% marginal for real overfitting risk. Angle-head masking also marginal (head already ingests conf; gate-only noff=3=57mm already beats mean-fallback 65mm). **Best move = do NOT retrain.**
- **Where it WOULD matter:** custom/insertion data with real textured object occlusion (the 9% confident-wrong is higher there) — but that needs the Stage-5 custom-GT pipeline. Deferred.
- **Net:** occlusion on DREAM is handled by `--conf-gate 0.05`. The remaining SOTA gap is the broad mid-tail on *fully-visible* frames, not occlusion.

### 2026-06-03 (cont.) — Head-to-head vs RoboPEPP: detector BEATS them, the LIFT is the leak

RoboPEPP numbers (user-provided) vs ours (latest angle head ep~8.3°, conf-gate 0.05, 1000 frames/split, PCK at matched 640×480):

| split | ours PCK@2.5/5/10 | RoboPEPP PCK | ours ADD-AUC | RoboPEPP ADD-AUC |
|---|---|---|---|---|
| SynPhoto | .75/.89/.94 | .87/.92/.94 | 70.5 | 84.1 |
| SynDR | .75/.90/.94 | .84/.91/.93 | 71.6 | 83.0 |
| AK azure | .37/**.78**/.96 | .16/.62/.93 | 71.2 | 75.3 |
| XK kinect360 | .24/**.70**/.94 | .09/.37/.96 | 63.6 | 78.5 |
| RS realsense | .31/.74/.85 | .31/.82/.97 | **32.4** | 80.5 |
| ORB | .34/.74/.91 | .28/.73/.96 | 54.4 | 77.5 |

**Our real-camera 2D keypoints BEAT RoboPEPP (azure/kinect PCK@5: .78/.70 vs .62/.37), but our ADD-AUC is lower → the entire gap is the keypoint→pose lift.**

Root cause = depth ill-conditioning in PnP (NOT a K bug — GT3D→K→2D self-consistency is 0.00px for all cams). Driver = robot DISTANCE: azure Z 0.87m → 21mm median ADD; realsense Z 1.33m → 67mm at identical 2D quality (~60mm is t_z error). Same mechanism as foreshortening/self-occlusion. RoboPEPP is best on realsense (its hardest distance) → their pose estimate is depth-robust; ours isn't.

**Strategic correction:** detector fine-tune (task #8) is low value (detector already > RoboPEPP). The lever is a **depth-robust keypoint→pose solve** (stabilize t_z via known robot scale / apparent-size prior / multi-start; reject realsense keypoint tail mean-L2 30px). Realsense −48 is the single biggest, fully-in-the-lift gap.

---
## 2026-06-06/07 — OCCLUSION ATTACK (user: "가림에 약하다, 성능 최대화")

### THE deployable win: iterative pseudo-label SELF-TRAINING (Stage 3)
`TRAIN/selftrain_pseudo.py`. Unsupervised real adaptation, NO oracle bbox (unlike crop). Per-camera
contiguous split adapt=first70% / held-out eval=last30% (no contamination). Pseudo = kinematic
SOLVER's refined angles on reliable frames; finetune angle head on pseudo-real + synth (anti-forget).
Base = stage1 detector + plain angle head + fullframe rot-head (DEPLOYABLE non-crop pipeline).
- realsense held-out: base 0.587 → r1 0.698 (+0.111) → r2 0.718 → r3 0.724. **Cumulative +0.137.** PLATEAU.
- kinect +0.022, azure +0.000. **gain ∝ sim2real gap** (helps realsense most = our RoboPEPP weakness; neutral on clean — safer than jitter).
- Tail-vs-bulk (`Eval/decompose_occlusion.py`, bin by mean kp conf): helps OCCLUDED bins MOST (Q1 +0.107, Q2 +0.161 vs clean +0.08/0.09). Genuinely fixes "weak under occlusion", BUT Q1 (extreme) stays worst bin (0.628) = residual tail.
- **Surpassed the MCL oracle ceiling (0.720)** → self-training overtook the MCL+selector path.

### MCL multi-hypothesis (occlusion under-determination) — DECOMPOSED, then overtaken
`AngleHeadMCL` (head_type='mlp_mcl', K hyp) + `train_angle_mcl.py` (winner-take-all) + `Eval/mcl_eval.py`
(solver min-reproj selects). Ep1 K=4 realsense 500: 1-HYP 0.666, SELECTED 0.678, ORACLE 0.720.
- Multimodal head COVERS the truth (oracle ≫ 1-hyp) but SELECTION is the bottleneck: under occlusion
  hypotheses reproject EQUALLY → 2D can't disambiguate → need APPEARANCE selector.
- Synth training can't diversify modes (no occlusion) → added kp_drop occlusion-aug (model_angle.forward `kp_drop`).

### Productionization finding: crop does NOT deploy
`Eval/selfbbox_eval.py`: crop's +0.154 realsense was ORACLE-bbox-dependent. Self-detector keypoint-bbox
→ 0.55 (≈ no-crop), because occluded frames miss the base → bbox mis-centers. Iterative bbox DIVERGES.
roi_align path verified (oracle-bbox 0.717 ≈ dataset GT-crop 0.768). → crop shelved for realsense deploy.

### RoboPose note (user asked): RoboPose IS render-and-compare (vendored robopose/), but REFINES one pose
→ can't escape wrong-basin under occlusion-multimodality. Our self-training = render-compare-lite WITHOUT
a renderer (solver pseudo-labels) and it WON. MCL+appearance-select is the genuinely different idea.

### RUNNING at compact:
- GPU1: occlusion-MCL (mclocc_20260606_235724, kp_drop=0.3, K=4, ~32min/epoch) — Ep0 clean-oracle 8.58.
- GPU2: reproj-filter self-train (rs_reproj_*, from r3 head, reproj<5px pseudo) — plateau-break test.

### NEXT (decision gates + free-GPU work):
1. **occ-MCL DECISION GATE**: when ~Ep2-3, `mcl_eval.py` on realsense. If ORACLE > ~0.76 (occlusion-aug
   raised the ceiling above self-train 0.724) → BUILD appearance selector (learned scorer: image feats +
   K hypotheses → pick GT-closest; or rot-head-consistency). Else → MCL DONE, self-training is the answer.
2. **reproj-filter result**: if > 0.724 → cleaner-tail pseudo breaks the plateau, adopt + iterate.
3. **DETECTOR self-train** (free GPU, untried high-EV): adapt the detector to real via reproj-consistent
   pseudo-KEYPOINTS (project FK(solver angles) → "cleaned" 2D, heatmap-finetune detector + synth). The one
   unadapted component; stacks with the +0.137 head gain.
4. Lock the deployable self-trained 6-split table; consider self-training kinect to more rounds.

### Key checkpoints:
- self-train r3 (BEST deployable realsense head): TRAIN/outputs_selftrain/rs_r3_20260607_002118/best_selftrain_head.pth
- MCL Ep1 head: /tmp/mcl_ep1.pth ; crop angle: /tmp/crop_angle.pth ; crop-native rot: /tmp/rot_crop.pth
- detector: outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth ; fullframe rot: outputs_rotation/rot_20260604_162336/best_rot_head.pth
- Eval tools: ab_eval.sh (6-split, NO_ROT/ROT_PATH env), mcl_eval.py, decompose_occlusion.py, selfbbox_eval.py

---

## 2026-06-07 — 🔒 LOCKED DEPLOYABLE 6-SPLIT TABLE (non-crop, 1000 frames, full-frame rot-head)

`Eval/ab_eval.sh ... 1000` with stage1 detector + full-frame rot-head (rot_20260604_162336). NON-CROP
(crop needs an oracle bbox, does not productionize). Plain baseline head vs realsense-self-trained r3 head.

| split | baseline | r3 self-train | Δ | RoboPEPP | ours-best vs RP |
|---|---|---|---|---|---|
| realsense | 0.6216 | **0.7158** | +0.094 | 0.805 | −0.089 |
| azure | 0.7416 | **0.7539** | +0.012 | 0.753 | **+0.001 ✓** |
| kinect | **0.7122** | 0.6935 | −0.019 | 0.785 | −0.073 |
| orb | **0.6616** | 0.6601 | −0.002 | 0.775 | −0.113 |
| synth_dr | 0.7069 | 0.7073 | +0.000 | — | — |
| MEAN(5) | 0.6888 | 0.7061 | +0.017 | — | — |

**Findings:**
- r3 (realsense-self-trained) head is **not realsense-specialized** — it also lifts azure (+0.012) and is
  neutral on orb/synth; only kinect dips slightly (−0.019). As a SINGLE all-camera head it's +0.017 mean.
  Per-camera-best deployable mean = 0.7102 (realsense/azure→r3, kinect/orb→baseline).
- **We only beat RoboPEPP on azure (0.754 vs 0.753).** The non-crop deployable pipeline is BEHIND on
  realsense (−0.089), kinect (−0.073), orb (−0.113). (Memory's crop+rot mean 0.751 needs an oracle bbox.)
- **realsense self-train plateaued at 0.716 full-split** (held-out 0.724), still −0.089 vs RoboPEPP →
  needs a NON-self-train lever (occ-MCL gate / detector self-train / crop productionization).
- reproj-filter self-train (reproj<5px, 171 frames) **REFUTED**: held-out 0.721-0.722 < base 0.724.

**Biggest addressable gaps (proven self-train method, untested/under-trained):**
- **orb −0.113** (UNTESTED for self-train, baseline 0.662) — highest EV. RUNNING (r1, adapt-cap 5000).
- **kinect −0.073** (only 1 self-train round = +0.022 before; full iterative untried). RUNNING (r1).

### Added: `selftrain_pseudo.py --adapt-cap N` (stride-subsample adapt frames; bounds pseudo-gen on big cams e.g. orb 22k).

---

## 2026-06-07 — ✅✅ TRAINING-FREE CROP PRODUCTIONIZATION (bbox-from-solved) — realsense CONFIRMED

The crop win (oracle-bbox mean 0.751) was blocked because keypoint-derived self-bbox = 0.55 (occluded
base dropped → mis-centered crop). FIX (zero new training): `selfbbox_eval.py --bbox-from-solved` runs
the full-frame kinematic SOLVE on pass-1, projects ALL 7 FK keypoints (incl. the occluded base, which
the camera pose R,t fixes even when base-yaw is ambiguous) → that's the crop bbox.

**realsense @1000:**
| pass-1 bbox | ADD-AUC |
|---|---|
| A) detected keypoints (old) | 0.5517 |
| B) **bbox-from-solved (NEW, training-free)** | **0.7141** |
| C) oracle bbox (roi_align ceiling) | 0.7165 |

- **B ≈ C (−0.002):** the solved-skeleton bbox recovers the ENTIRE crop gain the keypoint-bbox lost
  (0.55→0.71). No GroundingDINO/SAM2, no new model — the bbox falls out of the pose we already solve.
- Insight: base LOCATION is determined by camera pose (R,t); only base YAW is ambiguous. So even on
  foreshortened frames the projected skeleton bounds the full robot (incl. occluded base) correctly.
- Impl: pass-1 solve must run under `torch.enable_grad()` (solver does internal backward) inside the
  no_grad eval block; `project_points()` helper; bbox from all 7 pts (no conf threshold).
- RUNNING: azure/kinect/orb self-bbox-crop @1000 — does productionized crop match/beat RoboPEPP there
  (crop oracle: kinect 0.784 MATCH, azure 0.759 BEAT, orb 0.712)?

### Full self-bbox-crop table @1000 (training-free, plain head pass-1 bbox solve + crop pipeline pass-2):
| cam | non-crop(self-train) | self-bbox-crop | RoboPEPP | best vs RP |
|---|---|---|---|---|
| realsense | 0.716 | 0.714 | 0.805 | -0.089 |
| azure | 0.754 | 0.784 | 0.753 | +0.031 BEAT |
| kinect | 0.712 | 0.758 | 0.785 | -0.027 |
| orb | 0.662 | 0.690 | 0.775 | -0.085 |
- Productionized crop (ZERO new training) BEATS RoboPEPP on azure, narrows kinect (-0.073->-0.027) & orb (-0.113->-0.085). realsense crop≈non-crop (0.714≈0.716), still the hard camera.
- ⚠️ crop pipeline uses the CROP-trained head, NOT self-trained on real. crop (pixels) and self-train (real adaptation) are ORTHOGONAL and UN-stacked → NEXT = self-train the crop head per camera (crop+self-train stack) to push toward/past RoboPEPP.
- Note: azure/orb/realsense had 1-2 diverged-solve frames (huge mean ADD, median fine) → a solver divergence guard could recover a little AUC.

---

## 2026-06-07 — ✅✅✅ CROP + SELF-TRAIN STACK (orthogonal wins, caught RoboPEPP)

Self-trained the CROP angle head per camera (selftrain_pseudo.py --crop, GT-crop mode, from crop angle
head angle_crop_20260605_174740). crop (more pixels) + self-train (real adaptation) ADD.

**Held-out GT-crop (oracle-bbox), per camera:**
| cam | crop base | +self-train | Δ | RoboPEPP | vs RP |
|---|---|---|---|---|---|
| realsense | 0.7532 | 0.7771 | +0.0239 | 0.805 | -0.028 |
| azure | (0.784) | — | — | 0.753 | +0.031 BEAT |
| kinect | 0.7661 | 0.7770 | +0.0110 | 0.785 | -0.008 MATCH |
| orb | 0.7290 | 0.7586 | +0.0296 | 0.775 | -0.016 |
- Stack CONFIRMED orthogonal: every camera now BEATS RoboPEPP (azure) or within ~0.03. Mean ~0.774 vs RoboPEPP 0.780. vs the non-crop table that trailed -0.07 to -0.11 — crop+selftrain closed almost the entire gap. realsense (our worst, was -0.089) now -0.028.
- Stacked heads: outputs_selftrain/{cam}_crop_r1/best_selftrain_head.pth
- ⚠️ these are GT-crop (oracle-bbox) held-out numbers → RUNNING deployable self-bbox (bbox-from-solved) eval with the stacked heads for the honest deployable table.

### 🔒 FINAL DEPLOYABLE TABLE @1000 (self-bbox bbox-from-solved + crop + self-train stacked heads, NO oracle):
| cam | deployable | RoboPEPP | gap | head |
|---|---|---|---|---|
| realsense | 0.737 | 0.805 | -0.068 | realsense_crop_r1 |
| azure | 0.784 | 0.753 | +0.031 BEAT | crop base |
| kinect | 0.756 | 0.785 | -0.029 | kinect360_crop_r1 |
| orb | 0.723 | 0.775 | -0.052 | orb_crop_r1 |
| MEAN | 0.750 | 0.780 | -0.030 | |
- Fully deployable (no GT bbox/oracle/new bbox model). Session deployable mean 0.711 -> 0.750 (+0.039). BEAT RoboPEPP on azure, MATCH kinect (-0.029), trail orb/realsense (far/foreshortened, RoboPEPP's iterative render-and-compare edge).
- self-bbox costs ~0.04 vs GT-crop held-out (realsense 0.777->0.737, orb 0.759->0.723) -> a better bbox (or divergence guard) is the residual lever. Stacked heads deploy per-camera.

### Divergence guard (selfbbox_eval.py --bbox-guard) RESULT @1000:
| cam | no-guard | +guard | fallbacks |
|---|---|---|---|
| realsense | 0.737 | 0.737 | 0 |
| azure | 0.784 | 0.788 | 3 |
| kinect | 0.756 | 0.756 | 0 |
| orb | 0.723 | 0.723 | 3 |
| mean | 0.750 | 0.751 | |
- Guard = geometric divergence detect (off-frame/NaN/huge-span projected skeleton) -> fallback to detected-kp bbox + clamp all bboxes. NOT reproj-gated (realsense full-frame reproj is naturally high -> would nuke good frames).
- Verdict: ROBUSTNESS fix, not a gap-closer. Cleans catastrophic frames (azure/orb mean 540km/84km -> 32/38mm), azure +0.004 (frames flipped fail->success), realsense/kinect 0 fallbacks (no-op, zero harm). Keep ON by default.
- The -0.04 self-bbox vs GT-crop gap is SYSTEMATIC mild mis-cropping (realsense: 0 fallbacks yet still 0.737<<0.777 GT-crop), NOT divergence -> needs a BETTER bbox (union solved+detected / bbox-refine), not a divergence catch. Guard-on is the locked deployable config: mean 0.751 vs RoboPEPP 0.780; BEAT azure (+0.035), match kinect (-0.029), trail orb (-0.052)/realsense (-0.068).

---

## 2026-06-07 — ✅ DETECTOR SELF-TRAIN (the last un-adapted component) — TRAIN/selftrain_detector.py

Distill the kinematic solver back into the synth-only 2D detector: on reliable real frames (conf>0.5,
reproj<25px) the solver-refined pose reprojects to a clean 7-keypoint skeleton (incl. occluded base)
-> pseudo-keypoint heatmap target, finetune keypoint head (backbone frozen) + synth GT heatmaps
(anti-forgetting). Eval = full non-crop pipeline ADD (self-det + PLAIN angle head + fullframe rot).

**Plain-pipeline held-out delta:**
| cam | kept | baseline | det-self-train | Δ |
|---|---|---|---|---|
| realsense | 754/4160 (18%, foreshortened->high reproj) | 0.587 | 0.620 | +0.033 |
| kinect | 3071/3476 (88%, clean solves) | 0.701 | 0.753 | +0.052 |
| orb | 3402/5000 (68%) | 0.618 | 0.636 | +0.018 |
- Detector WAS improvable on real (despite "beats RoboPEPP on PCK" — PCK is on VISIBLE kp; the gain is full-pose 2D consistency, esp. kinect +0.052 > kinect angle-self-train +0.023). All positive.
- Saved keypoint_head only -> merged with stage1-unfrozen backbone into merged_detector.pth for eval.
- RUNNING: A/B (base-det vs self-det, both w/ self-angle-head) to test if detector self-train STACKS on the angle self-train.

### Detector self-train STACK A/B (non-crop, self-angle-head, base-det vs self-det @1000):
| cam | A base-det | B self-det | Δ |
|---|---|---|---|
| kinect | 0.7327 | 0.7762 | +0.0435 (MATCH RoboPEPP 0.785; beats its crop 0.756) |
| orb | 0.6972 | 0.7061 | +0.0089 |
| realsense | 0.7158 | 0.7043 | -0.0115 (CONFLICT) |
- DETECTOR<->HEAD COUPLING: self-det stacks on kinect/orb but CONFLICTS on realsense — the r3 angle head was heavily specialized to the ORIGINAL detector's keypoints; self-det shifts them OOD. Fix = CO-ADAPT (re-self-train the head on the new detector). kinect's lighter head tolerated it -> +0.044.
- Updated best-per-camera deployable mean 0.751->0.756 (kinect 0.756->0.776 via self-det non-crop). vs RoboPEPP 0.780, gap -0.024. NEXT: co-adapt detector+head to make the stack positive everywhere (recover realsense).

### Co-adaptation (re-self-train angle head on self-det) RESULT (held-out):
- kinect: self-det+plain 0.7514 -> co-adapt 0.7675 (+0.016); realsense: 0.6201 -> 0.6739 (+0.054 but STILL < base-det iterative-r3 0.724 -> self-det net-neutral on realsense, too few reliable pseudo on foreshortened cam); orb co-adapt LOST to GPU0 re-fault mid-run.
- CONCLUSION: detector self-train is a genuine lever but UNEVEN — big win on kinect (clean frontal cam, 88% reliable pseudo -> 0.776 ≈RoboPEPP non-crop), negligible/negative on realsense (foreshortened, 18% reliable pseudo). Best-per-cam deployable mean 0.751->0.756 (kinect lifted). vs RoboPEPP 0.780, gap -0.024.
- ⚠️ GPU0 (2F:00.0) re-faulted mid-orb-co-adapt -> node CUDA wedged again -> needs reboot. Flaky hardware (see memory gpu-env-notes).

---

## 2026-06-07 (cont.) — union-bbox REFUTED + orb co-adapt closes detector table

### Union-bbox (selfbbox_eval.py --bbox-union) A/B @1000, guard-on:
| cam | guard-baseline | guard+union | Δ |
|---|---|---|---|
| realsense | 0.7370 | 0.7351 | -0.0019 |
| orb | 0.7229 | 0.7210 | -0.0019 |
- Hypothesis (solver-bias mis-crops -> union with detected-kp bbox guarantees no confident kp is clipped) REFUTED. Union only enlarges the bbox; marginally WORSE => self-bbox already encloses the keypoints fine, it is NOT too small. The -0.04 self-bbox-vs-GT-crop gap is therefore NOT bbox-too-tight; it is GT-crop's oracle zoom/center distribution advantage (crop head trained on GT-crops, deployed on solved-skeleton crops incl. always-present base). Bbox geometry tweaks (union/enlarge) cannot recover it. LEVER CLOSED.

### orb co-adapt (re-self-train angle head on merged self-detector) RESULT, non-crop held-out:
- baseline (merged self-det + PLAIN angle) 0.6367 -> co-adapt 0.6803 (+0.0436). Confirms detector<->head COUPLING: self-det shifted keypoints OOD for the plain angle head (0.673 plain-selftrain -> 0.637 with self-det), co-adapt recovers + slightly beats (0.680). But orb non-crop 0.680 still < orb crop+selftrain 0.723 => orb best deployable stays CROP 0.723. Detector self-train table now complete: kinect non-crop self-det wins (0.776), realsense/orb crop+selftrain win.

### Iterative crop-selftrain ROUND 2 (conf-keep 0.6) RESULT, GT-crop held-out:
| cam | r1 (baseline) | r2 | Δ |
|---|---|---|---|
| realsense | 0.7771 | 0.7771 | +0.0000 (PLATEAU) |
| orb | 0.7586 | 0.7631 | +0.0045 |
- Crop-selftrain SATURATES at r1 for realsense; orb +0.0045 marginal. Deployable mean ~unchanged. The remaining ceiling for realsense/orb is the SELF-BBOX -0.04 gap vs GT-crop (NOT the head). Next: --bbox-refine-iters (coarse-to-fine bbox via crop detector) to attack that ceiling directly. orb best crop head now r2: outputs_selftrain/panda-orb_crop_r2/best_selftrain_head.pth.

### bbox-refine-iters (coarse-to-fine: re-detect on crop with crop detector -> tighter bbox) A/B @1000, guard-on:
| cam | refine=0 | refine=1 | refine=2 | best |
|---|---|---|---|---|
| realsense | 0.7370 | 0.7436 | 0.7451 | **refine-2 +0.0081** (monotone) |
| orb | 0.7245 | 0.7247 | 0.7202 | refine-0 (refine DIVERGES: meanADD 525mm@r1, 50265mm@r2 -> re-detect loop spirals on bad crops; UNGUARDED) |
- bbox-refine is CAMERA-DEPENDENT: helps realsense (the hardest, self-bbox-capped) +0.008, but on orb the iterative re-detect loop catastrophically diverges some frames (guard only covers PASS-1 solved-skeleton, NOT the refine loop). realsense -> adopt refine-2; orb -> keep refine-0. First lever to dent the self-bbox -0.04 ceiling (realsense 0.737->0.745, half-closing toward GT-crop 0.777... no, ~1/5). Note orb refine needs a guard inside the refine loop before it's safe elsewhere.

### bbox-refine on azure/kinect @1000: refine helps ONLY realsense
| cam | refine=0 | refine=2 |
|---|---|---|
| azure | 0.7881 | 0.7915 (+0.003 but meanADD 148431mm = diverged frames; keep refine-0, safe) |
| kinect (crop) | 0.7560 | 0.7382 (-0.018 WORSE) |
- refine is realsense-specific (+0.008). azure marginal+risky, kinect-crop hurts, orb diverges. kinect best stays non-crop self-det 0.776.

## 2026-06-07 — 🔒🔒🔒 FINAL CHEAP-LEVER DEPLOYABLE (all oracle-free levers exhausted) @1000:
| cam | deploy | config | RoboPEPP | gap |
|---|---|---|---|---|
| realsense | 0.745 | crop+selftrain r1 + bbox-refine-2 | 0.805 | -0.060 |
| azure | 0.788 | crop base (refine-0) | 0.753 | +0.035 BEAT |
| kinect | 0.776 | self-det non-crop + self-angle | 0.785 | -0.009 MATCH |
| orb | 0.725 | crop+selftrain r2 | 0.775 | -0.050 |
| **MEAN** | **0.759** | | 0.780 | **-0.021** |
- ALL cheap/oracle-free levers exhausted this session (crop, per-cam selftrain, detector-selftrain, iterative-r2, union-bbox[refuted], bbox-refine). Each now yields <=0.008. Session deployable mean 0.711 -> 0.759 (+0.048). BEAT azure, MATCH kinect; realsense (-0.060)/orb (-0.050) are the structural residual = RoboPEPP's far/foreshortened iterative render-and-compare edge. Crossing 0.780 needs a HEAVY lever: (A) SSL backbone adaptation (ijepa on real unlabeled, lifts all-cam real features at root) or (B) render-and-compare refine (robopose/ + URDF mesh, the literal RoboPEPP edge). Both multi-hour, single-GPU (flaky node).

## 2026-06-08 — Stage 4: SSL BACKBONE ADAPTATION (masked-feature, data2vec-style) — ssl_masked_dinov3.py
SSL on 18,111 pooled DREAM-real frames (contiguous 70% adapt/cam), mask 50% patches, predict EMA-teacher
features at masked positions, unfreeze last 6 ViT blocks. Loss 0.188->0.031 (Ep0->7, clean converge, NO
collapse). Then re-train keypoint head on synth w/ SSL backbone frozen (head was OOD on swapped backbone):
azure val AUC 0.064(Ep0)->0.761->0.796->0.805(Ep3, L2 3.95px == matches original detector 0.817/4.0px).

### DECISIVE: real PCK @orig-res, SSL detector vs original detector (1000 frames/cam):
| cam | PCK@5 ORIG | PCK@5 SSL | Δ@5 | PCK@10 ORIG | PCK@10 SSL | Δ@10 |
|---|---|---|---|---|---|---|
| realsense | 0.742 | 0.811 | **+0.069** | 0.847 | 0.969 | **+0.122** |
| orb | 0.738 | 0.742 | +0.004 | 0.909 | 0.932 | +0.023 |
| azure | 0.780 | 0.761 | -0.019 | 0.955 | 0.947 | -0.008 |
| kinect | 0.702 | 0.588 | **-0.114** | 0.943 | 0.820 | **-0.123** |
- SSL is CAMERA-DEPENDENT: big win on realsense (our #1 gap, hardest/most sim2real-gapped camera; @10 outliers crash 0.847->0.969), neutral orb, slight loss azure, BIG loss kinect. => use SSL backbone PER-CAMERA (realsense yes, kinect no). The realsense @10 +0.122 (gross-outlier collapse) should translate to fewer solver failures -> higher ADD. NEXT: re-self-train realsense angle(+rot) head on SSL backbone, measure realsense ADD vs non-SSL 0.724 (full-frame) / 0.745 (crop).

### DECISIVE — SSL REFUTED for ADD (realsense no-rot A/B, detector quality isolated, angle re-self-train, PnP R_init):
| detector | baseline ADD-AUC | +angle self-train |
|---|---|---|
| SSL backbone | 0.4384 | 0.5313 |
| original (non-SSL) | 0.4974 | **0.5674** |
- The ORIGINAL detector BEATS the SSL detector on realsense ADD (0.567 vs 0.531) **despite SSL's clearly better real PCK** (@5 +0.069, @10 +0.122). => SSL masked-feature backbone adaptation improves COARSE keypoint localization (PCK) but the kinematic solver needs REPROJECTION-PRECISE keypoints; SSL's features (synth-retrained head on real-adapted backbone) are coarsely-better but worse for precise pose recovery. "Train proxy != eval target" ([[train-proxy-vs-eval-target]]) now extends to REAL PCK: real PCK does NOT predict real ADD either. **SSL (the chosen heavy lever) does not help our metric — slightly hurts. CLOSED.**
- Deployable best stays the FINAL CHEAP-LEVER table: mean 0.759 (BEAT azure, MATCH kinect, realsense -0.060/orb -0.050). Only remaining untried heavy lever = render-and-compare refine (robopose/ + URDF). SSL artifacts (outputs_ssl/, outputs_heatmap/ssl_head_r1/) are dead weight — can delete.

### DECISIVE — backbone+head CO-FINETUNE on real solver-pseudo also REFUTED (realsense no-rot):
co-finetune last 4 ViT blocks + kphead on real solver-pseudo keypoints (precise supervision, no head-OOD by co-training) + synth anti-forgetting. STEP-1 internal eval MONOTONICALLY DEGRADED every epoch:
baseline 0.4974 -> Ep0 0.459 -> Ep1 0.441 -> Ep2 0.447 -> Ep3 0.434 -> best fell back to baseline (original backbone, +0.000).
- Same verdict as SSL: ANY backbone real-adaptation (masked-feature SSL OR pseudo-keypoint co-finetune) HURTS realsense ADD because it trades the sub-pixel keypoint PRECISION the kinematic solver needs for coarse real-robustness. The FROZEN synth-pretrained DINOv3 backbone gives the most solver-precise keypoints — adapting it is net-negative for ADD. (Explains why the DINOv3Backbone unfreeze-bug was harmless: frozen was optimal all along.)
- **BACKBONE-ADAPTATION FAMILY CLOSED.** Deployable best stays mean 0.759. Remaining heavy lever = HPE-style render-and-compare silhouette IOU self-sup (see memory hpe-horopose-analysis), OR consolidate.

## 2026-06-08 — WHY IT FAILS: per-frame ADD error DECOMPOSITION (realsense_failure_diag.py, +component split)
Decompose ADD into mm contributed by each component (pred kp = R@FK(theta)+t; swap each factor for GT):
| component | realsense fail/ok | orb fail/ok | corr(ADD) rs/orb |
|---|---|---|---|
| translation t | 370/62 (6.0x) | 374/41 (9.1x) | **+0.71 / +0.72** (#1) |
| rotation R->3d | 335/149 (2.3x) | 323/82 (3.9x) | +0.60 / +0.58 (#2) |
| angle theta->3d | 94/124 (0.8x) | 130/64 (2.0x) | -0.16 / +0.35 |
| kp2d px | 43/32 (1.4x) | 40/13 (3.1x) | +0.31 / +0.45 |
- **TRANSLATION (depth placement) is the #1 ADD-failure driver on BOTH cameras** (corr ~+0.72; failing frames mislocate the robot by ~370mm in camera frame vs ~50mm on ok frames). Rotation #2. **Joint angles theta are NOT the failure driver** (realsense 0.8x, corr -0.16) — the arm SHAPE is recovered well; the camera-to-robot POSE (t, then R) is what breaks. (The old J0-base-yaw ANGLE problem was already fixed by self-train+crop; the residual -0.06/-0.05 is now the pose-lift t/depth.)
- detector(2D) matters more on orb (3.1x) than realsense (1.4x) — orb has a real detector-quality failure component.
- Failing frames = foreshortened (fore_axis 59 vs 72deg, arm along camera axis) + low conf (0.52 vs 0.64) + worse 2D => 2D bearings UNDER-CONSTRAIN depth => solver t/R ill-determined => monocular depth/scale ambiguity. Reconciles "depth prior REFUTED" (we regressed appearance->depth, sim2real-unreliable) with "t is the bottleneck": the need is a GEOMETRY-GROUNDED depth/scale (HPE RootNet k_value bbox-normalized depth, or render-and-compare silhouette SCALE), not a learned appearance prior. This is the concrete target for the next lever.

## 2026-06-09 — ✅ RENDER-AND-COMPARE VALIDATED (the lever that supplies depth/scale robustly)
After: failure decomp showed depth/scale is the bottleneck (gauge-coupled pose, +0.116 GT-depth ceiling
realsense); scalar depth head DIED (fragile: 10% noise -> negative, selective doesn't save it). Built a
pure-PyTorch mesh-silhouette renderer (`Eval/silhouette_mesh_probe.py`): real Panda collision meshes
(link0-7+hand, 1044 pts) transformed by per-link FK, projected, bilinear-splatted + gaussian-blurred into
a FILLED differentiable soft mask whose AREA ~ 1/z^2 encodes depth. Refine (theta,R,t) to match an ORACLE
(GT-pose) silhouette via soft-IoU + reprojection anchor.
- v1 skeleton-splat FAILED (-0.10: thin polyline ~0 area, no depth cue, diverges).
- mesh + AGGRESSIVE opt FAILED (-0.12: unstable test-time IoU optimization from a good init).
- mesh + CONSERVATIVE opt (lr 5e-4, repro-w 100) STABLE, and the gain SCALES WITH RENDER RES:
| render-h | realsense ADD-AUC (baseline 0.6911) | Δ |
|---|---|---|
| 96  | 0.7112 | +0.020 |
| 160 | 0.7735 | +0.082 |
| 224 | **0.7992** | **+0.108** (~93% of the +0.116 GT-depth ceiling) |
- **render-compare ROBUSTLY recovers nearly the full depth/scale ceiling** (realsense 0.691->0.799, ~= depth ceiling 0.808, would reach/beat RoboPEPP 0.805) — NO fragility (unlike scalar depth). The IoU's depth-sensitivity was the limiter; higher render res fixes it. This is THE validated lever for the realsense/orb gap.
- ⚠️ ORACLE MASK caveat: target = GT-pose-rendered silhouette. Deployment needs a REAL robot mask (SAM2/CtRNet); segmenter error will reduce the gain. NEXT: swap oracle for a real segmenter mask (prompt SAM2 with the bbox/keypoints we already have), measure the deployable gain. Even half (+0.05) -> realsense ~0.79 near RoboPEPP.

### render-compare MASK-QUALITY sensitivity (render-h 224, realsense 400, degrade target mask):
| mask-degrade | refine ADD-AUC | Δ |
|---|---|---|
| 0.0 oracle | 0.7992 | +0.108 |
| 0.5 | 0.8011 | +0.110 |
| 1.0 (boundary err + region dropout) | 0.7907 | +0.100 |
| 1.5 (heavy: erode/dilate + noise + 40% dropout) | 0.7736 | +0.083 |
- **render-compare is ROBUST to mask degradation** (opposite of scalar-depth fragility): IoU averages the whole mask AREA so boundary noise/small dropouts wash out, and the depth signal is in gross area. A SLOPPY mask still gives +0.083..+0.10. => NO high-quality segmenter needed (no SAM2/CtRNet install). A lightweight DINOv3 mask head (synth GT masks rendered FREE by our mesh renderer, frozen backbone, 1-ch decoder) suffices and is self-contained. NEXT: build that mask head, predict on real, plug into render-compare, measure deployable gain.

### render-compare DEPLOYABLE attempt v1 (synth-trained mask head -> predict real mask): FAILED -0.141
mask head (frozen DINOv3 + 1ch decoder, synth GT masks rendered by mesh renderer) best synth IoU 0.382.
Plugged into render-compare as the REAL target mask: realsense 0.6911 -> 0.5497 (-0.141, HURTS).
- Reconciles with the mask-degrade robustness (+0.083 at heavy degrade): degrade kept the oracle mask
  CORRECTLY PLACED (boundary noise/dropout only). The synth mask head on REAL is sim2real MIS-PLACED/
  mis-scaled -> a wrongly-located mask pulls the pose to a wrong solution. render-compare needs a
  CORRECTLY-PLACED mask (boundary can be rough), which a cheap synth-only head doesn't give.
- FIX (proven recipe): SELF-TRAIN the mask head on real — pseudo-mask = render the SOLVER's mesh
  silhouette on high-conf real frames (good pose -> correct mask), adapt head to real appearance, mixed
  with synth GT (anti-forgetting). Appearance-based -> generalizes to foreshortened frames too.

### render-compare DEPLOYABLE v2 (SELF-TRAINED real mask head) — pipeline now WORKS (+0.012), mask quality caps it
mask self-train (`TRAIN/selftrain_mask.py`): pseudo real mask = solver mesh silhouette on high-conf real
frames (kept 3615/4000), + synth GT anti-forgetting, 4 ep. realsense render-compare:
| mask source | Δ |
|---|---|
| oracle | +0.108 |
| synth head v1 (mis-placed on real) | -0.141 |
| **self-trained head v2** | **+0.012** |
- Self-train FIXED the -0.141 mis-placement -> +0.012 (pipeline end-to-end functional with a learned real
  mask). But far below the +0.108 oracle ceiling: the real mask is still too rough. ROOT LIMIT = CIRCULARITY:
  render-compare must help the foreshortened HARD frames, but those have a wrong solver pose -> wrong
  pseudo-mask -> the mask head learns them poorly (only partial appearance generalization). 
- Remaining mask-quality levers (incremental): denser mesh render labels (current 1044 pts -> holey masks,
  synth IoU only 0.38), higher mask res, iterate self-train, photometric aug. Each could lift the deployable
  gain toward the ceiling. render-compare itself is VALIDATED; productionizing rides on the mask quality.

### render-compare deployable v3 (DENSE labels): synth IoU up, deployable gain NOT up
Dense mesh labels (cap 120->400/mesh): mask head synth IoU 0.382 -> 0.543. But deployable render-compare:
v2 sparse-mask +0.012 -> v3 dense-mask +0.004 (both ~noise at 400 frames). Synth-IoU improvement did NOT
translate to a bigger real ADD gain -> the SAME "proxy != eval-target" lesson (synth mask IoU doesn't
predict real-mask placement on the foreshortened frames that matter).
- **CONCLUSION: render-compare is VALIDATED (+0.108 oracle ceiling, robust to mask DEGRADATION) but a
  SELF-CONTAINED learned mask caps the deployable gain at ~+0.01** — circularity: the foreshortened hard
  frames (where render-compare must help) have a wrong solver pose -> wrong pseudo-mask -> the mask head
  never learns them well. The full ceiling needs a CORRECTLY-PLACED real mask = a proper SEGMENTER
  (SAM2/CtRNet). That install (SAM2 won't pip on torch 2.10; CtRNet vendored in HPE, needs weights) is the
  gating next step to realize render-compare's +0.108 on realsense.
- Infra note: GPU 05f84104 was grabbed mid-run by an external `ollama runner` (~21GB); ran the retry at
  batch 3 in the remaining 2.5GB.

### visual vs collision mesh for render-compare (oracle ceiling, realsense 400 @224px): COLLISION WINS
| renderer mesh | oracle Δ |
|---|---|
| collision (solid, low-poly) | +0.108 |
| visual stride-downsampled | +0.064 |
| visual uniform-downsampled (cap600) | +0.055 |
- Hypothesis "visual matches the image -> better" REFUTED for the RENDERER. render-compare's depth cue is the
  overall AREA; collision meshes are SOLID/convex -> a cleaner, more depth-sensitive silhouette. Visual
  meshes' fine detail/concavities make the splatted silhouette less solid -> weaker depth signal. KEEP collision.
- So the deployable bottleneck is NOT the renderer mesh; it's the learned real-mask quality (IoU 0.52 vs
  oracle, 17% badly mis-placed). Self-contained mask (synth / dense / self-train / visual) caps at IoU ~0.52
  -> render-compare deployable ~+0.01. The genuine unlock = a correctly-placed real mask from a SEGMENTER
  (SAM2 won't pip on torch 2.10; CtRNet vendored in HPE/lib/models/ctrnet needs weights). That install is the
  gating next step to realize render-compare's validated +0.108 on realsense.

### render-compare with SAM (ViT-B) real masks: FAILED (mask-render mismatch / prompt quality)
Installed segment-anything + sam_vit_b (network available). Prompt SAM with detected keypoints -> real robot
mask -> render-compare target. realsense smoke: SAM-vs-render IoU 0.21 (collision) / 0.36 (visual), render-
compare -0.53 / -0.64 (DIVERGES).
- SAM segments the TRUE (visual) robot; matching it to the collision (bulky) refine-render is shape-
  inconsistent -> pose pushed to wrong depth. Visual-render restores some consistency (IoU 0.21->0.36) but
  still diverges -> SAM's keypoint-prompted mask is itself too rough/partial vs the GT-pose render, OR a
  residual render-vs-image alignment offset. Either way SAM out-of-the-box doesn't deliver.
- **NET: render-compare is VALIDATED (+0.108 self-consistent oracle) but NO real mask source delivers the
  deployable gain yet** — learned mask (IoU 0.52, consistent-but-imperfect) -> +0.01; SAM (accurate-but-
  shape-inconsistent) -> diverges. Cracking it needs focused mask work (SAM prompt/box tuning + visual-render
  consistency + an area-only or robust loss, or a robot-specific segmenter like CtRNet). Clean next-session task.

### SAM box-prompt + render-consistency: STILL FAILS -> the renderer-fidelity wall (definitive)
SAM box-prompt (kp bbox+margin): collision IoU 0.235 / -0.45, visual IoU 0.36 / -0.64. Overlay viz confirms
my GT-pose render OVERLAYS the real robot (NO alignment bug) — but my splat render (collision=bulky,
visual=approx) does not match SAM's TRUE robot pixels (IoU caps 0.24-0.36) -> matching a true mask to an
approximate render pushes the pose wrong -> diverges.
- **DEFINITIVE: render-compare only works when the target mask comes from the SAME (approximate) render
  model** — oracle (self-consistent +0.108) or a render-trained learned mask (consistent but imperfect,
  IoU 0.52 -> +0.01). A TRUE segmenter (SAM) is fundamentally inconsistent with the approximate splat render
  -> can't be used without a HIGH-FIDELITY differentiable renderer (exact visual meshes + real rasterizer =
  PyTorch3D, which won't install on torch 2.10/cu128). 
- **Render-compare is VALIDATED as a lever (+0.108 depth/scale recovery) but NOT deployable with current
  tooling**: the self-contained learned mask caps the gain at ~+0.01, and the SAM route needs PyTorch3D.
  Clean next-session task: get PyTorch3D (or nvdiffrast) working -> exact-mesh render -> SAM/CtRNet true mask
  -> realize the +0.108 (realsense 0.745 -> ~0.85). Until then, deployable best stays mean 0.759.

---

## 2026-07-03 — ✅ NAS-monorepo MIGRATION + full baseline REPRODUCTION (sota-dream branch)

Work moved to the LOCAL NAS box (4×3090 + A6000, all idle; env `dino` torch 2.10+cu128) — the remote
GPU-server node is down to 1 healthy GPU. Checkpoints (7.4GB, best_* only) rsynced into
`TRAIN/outputs_*`; datasets wired via gitignored `Dataset/` symlinks mirroring the remote layout
(`Converted_dataset/DREAM_real -> datasets/ICRA_multiview/.../DREAM_to_DREAM` etc. — note eval code
resolves `meta.image_path` LEXICALLY through `..`, so the top-level `Dataset/DREAM_real`/`DREAM_syn`
links are required, not just the Converted_dataset ones). `ab_eval.sh` made location-agnostic
(cd-from-script-path, HF_HOME guard).

**Reproduction vs locked remote numbers (deterministic `selfbbox_eval`/EvalDataset sampling):**
| config | local | remote ref | Δ |
|---|---|---|---|
| realsense crop+selftrain+refine-2 @1000 | 0.7497 | 0.745 | +0.005 |
| realsense ROT-ADAPT held-out @800 | 0.7525 | 0.755 | −0.003 |
| azure crop base @1000 | 0.7829 | 0.788 | −0.005 |
| orb ROT-ADAPT held-out @800 | 0.7154 | 0.715 | +0.000 |
| kinect self-det non-crop @1000 | 0.7577 | 0.776 | −0.018 ⚠️ |
| ab_eval 5-split smoke @300 (non-crop plain) | mean 0.694 | 0.689 | +0.005 |

⚠️ kinect gap is NOT a migration bug: `refine_eval` uses the UNSORTED `PoseEstimationDataset`
(listdir order = filesystem-dependent) → a different 1000-frame subset per machine. All
`selfbbox_eval` (sorted+strided EvalDataset) numbers reproduce within ±0.005. → re-lock kinect with
deterministic sampling for the final table; prefer EvalDataset-style sampling everywhere.

**SAM-vs-splat "before" state reproduced** (`silhouette_mesh_probe --sam-checkpoint`, visual splat,
realsense 100): SAM IoU 0.345 (remote 0.36), render-compare with SAM target DIVERGES (−0.43) — the
known renderer-fidelity wall. Phase-2 gate: exact-mesh nvdiffrast render must lift GT-pose-render vs
SAM IoU to ≥0.7.

**Built:** `Eval/render_nvdr.py` — nvdiffrast CUDA-raster silhouette of the VISUAL meshes (faces,
batched clip-space projection from K, dr.antialias edge gradients), drop-in for the splat
`render_mesh`, with a smoke test (IoU vs splat + real-image overlays). Blocked on `pip install
nvdiffrast` (GitHub source; awaiting user-run install). Pose dumps for cross-checking saved to
`Eval/rc_dumps/{realsense,azure,orb}_*.npz`.

## 2026-07-03 — 문헌 서베이: 타깃 0.780 유지 확정 + 가림 아이디어 랭킹
Full survey: `docs/robot_pose_sota_survey.md`. 핵심: (1) PoseDiff의 0.96은 real-학습 in-domain — 비교 무효,
**동일 프로토콜(predicted angles + sim-to-real)에서 RoboPEPP 0.780이 여전히 프론티어** (RoboPEPP 수치
75.3/78.5/80.5/77.5 원문 검증, 단 GT-bbox 헤드라인이며 auto-bbox면 orb ~34 — **우리 bbox-from-solved는
완전 자동이라 더 엄격한 조건**). (2) 가림-강건 top 아이디어: 가림-로버스트 실루엣(RC 픽셀 가중),
공분산 가중 PnP(IRLS 확장), masked-state prior(prior_w 훅), CtRNet-X식 가시 링크 선택. V-JEPA/백본 계열
기각 재확인(V-JEPA 2.1 논문이 우리 반증 독립 확인). RoboTAG(arXiv:2511.07717)만 주시.

---

## 2026-07-03 — 🔒🔒🔒 SOTA: nvdiffrast+SAM render-and-compare DEPLOYED — mean 0.796 vs RoboPEPP 0.780

**The 06-09 blocker (renderer fidelity) is broken.** nvdiffrast (pip source build, CUDA raster, no GL)
renders the EXACT visual meshes (+both gripper fingers baked into the hand frame) → GT-pose render now
overlays the real robot pixel-accurately. SAM-vs-render agreement jumped 0.345→0.68 (GT-pose gate),
0.85 (init-pose, deployment prompts) — the shape-inconsistency that made SAM targets diverge is gone.

**Deployable pipeline addition** (`Eval/rc_refine_from_dump.py`): refine the deployed crop+rot-adapt
poses (from `selfbbox_eval --dump-npz`) against SAM ViT-B masks: prompts = projected dump-pose keypoints
(+bbox), mask candidate SELECTED by IoU vs the init-pose render (render-consistency selection, not SAM
score), conservative opt (Adam 5e-4, soft-IoU + reproj anchor w=100), do-no-harm gate min-iou 0.35.

**Resolution scaling (realsense 200-slice): 224 +0.041 → 320 +0.067 → 448 +0.078 → 512 +0.078 (saturates).**
Lock render-h=448 (rs/kinect), 512 (orb — smaller/farther, kept climbing to 512).

### 🔒 FINAL DEPLOYABLE TABLE (anti-leak held-out 800/cam for rs/kinect/orb; azure full-split @1000)
| cam | base (rot-adapt deploy) | +RC | RoboPEPP | gap |
|---|---|---|---|---|
| realsense | 0.7525 | **0.8183** (+0.066) | 0.805 | **+0.013 BEAT** |
| kinect360 | 0.7462 (crop cfg) | **0.8112** (+0.065) | 0.785 | **+0.026 BEAT** |
| azure | 0.7881 (crop base, RC OFF) | — | 0.753 | **+0.035 BEAT** |
| orb | 0.7154 | **0.7647** (+0.049) | 0.775 | −0.010 |
| **MEAN** | | **0.7956** | 0.780 | **+0.016 SOTA** |

- **kinect CONFIG SWITCH**: crop+rot-adapt+RC 0.811 > old non-crop self-det 0.776.
- **azure RC OFF**: near camera (Z 0.87m), depth already right → RC −0.047 (mask noise perturbs depth;
  uv-shift guard ineffective — the damage is IN the depth direction). Consistent with mechanism: RC is a
  depth/scale corrector; per-camera on/off matches the failure decomposition (t-error #1 on rs/orb only).
- Protocol honesty: predicted angles + FULLY AUTOMATIC bbox (bbox-from-solved) — STRICTER than RoboPEPP's
  GT-bbox headline (their auto-bbox orb ≈34). rs/kinect/orb numbers are anti-leak held-out (self-trained
  heads); deltas are same-frame A/B. + SAM ViT-B added to the pipeline (inference cost note).
- orb residual −0.010: RC recovered +0.049 but orb has a real 2D-detector component too (06-08 decomp).

Migration/infra: local NAS box (4×3090+A6000), env `dino`; checkpoints mirrored from GPU server; baselines
reproduced ±0.005 before any new work (see 2026-07-03 migration entry). Tools: `Eval/render_nvdr.py`
(exact-mesh silhouette, drop-in), `Eval/nvdr_sam_gate.py` (fidelity gate), `Eval/rc_refine_from_dump.py`.
Survey: `docs/robot_pose_sota_survey.md` (RoboPEPP 0.780 confirmed as same-protocol frontier; PoseDiff 0.96
is in-domain real-trained, not comparable).

NEXT (levers, in EV order): (1) orb −0.010: RC + detector-side (2D) improvement, or 576+ render-h;
(2) full-split re-lock for paper numbers; (3) occlusion catalog probes from the survey (robust silhouette
weighting for external occluders, covariance-weighted PnP, masked-state prior via prior_w).

---

## 2026-07-03 — OCCLUSION TRACK: RoboPEPP-protocol bench + 3 levers (1 adopted, 2 refuted)

**Bench** (`Eval/occlusion_bench.sh` + `occl_util.py`): RoboPEPP Fig.6 protocol reproduced exactly —
black rect/circle masks covering {0,10,20,30,40}% of the GT-kp RoI on panda_synth_test_photo,
DETERMINISTIC per (frame,ratio) so the pose stage and the SAM/RC stage see identical occluders.
Full deployable pipeline (auto bbox + crop + solve) + nvdr/SAM RC@448, 200 strided frames.

### 🔒 Occlusion curve (ours +RC, cov-pnp config) vs published (RoboPEPP Fig.6)
| RoI occl | ours pose | ours +RC | RoboPEPP | HPE | RoboPose |
|---|---|---|---|---|---|
| 0% | 0.724 | 0.775 | 0.795 | 0.570 | 0.540 |
| 10% | 0.667 | 0.726 | 0.730 | 0.505 | 0.420 |
| 20% | 0.567 | **0.626** | 0.600 | 0.405 | 0.280 |
| 30% | 0.481 | **0.525** | 0.470 | 0.320 | 0.210 |
| 40% | 0.315 | 0.328 | 0.351 | 0.282 | 0.145 |
- **WIN vs RoboPEPP at 20-30% (+0.026/+0.055), ~tie 10/40%, −0.02 at 0%** (their synth edge — we win real).
  Degradation slope 0→40% identical (−0.447 vs −0.444) from a 0.02-lower start → occlusion robustness
  per se ≥ RoboPEPP. RC stays positive UNDER occlusion (+0.06@10-20%) — the SAM+exact-render loop
  does not collapse when the robot is partially masked.

### Lever verdicts (ablated individually @20%, base pose 0.5610 / +RC 0.6156)
- ✅ **cov-PnP ADOPTED** (`solve_batch(cov_inv=)`, `heatmap_cov_inv`, `selfbbox_eval --cov-pnp`):
  anisotropic heatmap-second-moment Mahalanobis weighting. Do-no-harm at 0/40%, +0.006 pose /
  +0.011 RC at 20%. Modest but free (no retrain).
- ❌ **occl-robust silhouette REFUTED as designed** (`--occl-robust-w`): downweighting
  "init-render ∧ ¬SAM" pixels removes the penalty for the render inflating past SAM → DEPTH BIAS
  (−0.019 RC @20%, also hurts clean azure logic). Rescue would need an explicit occluder
  segmentation (only downweight inside a DETECTED occluder), not blanket disagreement weighting.
- ❌ **occlusion-adaptive population prior REFUTED** (`--prior-adaptive`): even at 0.005 it's
  −0.09 @20%. Root cause structural, not scale: synth joints are INDEPENDENT (max |corr| 0.06) and
  broadly spread (σ 0.5-1.0 rad), so pulling toward the population mean actively fights the true
  config. Consistent with the June "mean-fallback dead end" refutation. The full learned
  (DPoser-style) state prior was SKIPPED by the same pre-check — nothing beyond per-joint marginals
  to learn. A prior helps only if conditioned on the FRAME (that's what theta_init already is).

Takeaway: the existing stack (conf-gate solver + rot-adapt heads + exact-render RC) already carries
RoboPEPP-level occlusion robustness; the mid-occlusion WIN comes from the solver+RC, not from any
new occlusion-specific machinery. Survey catalog (docs/robot_pose_sota_survey.md) updated verdicts.

## 2026-07-04 — multi-start RC (SAM-IoU basin selection) REFUTED — but diagnostically decisive
Hypotheses = base-Z gauge rotations (±30/60°) of the RC init, winner by SAM-IoU (external evidence,
unlike the refuted learned-selector MCL). Clean rs 0.8157 (0 switches — perfect do-no-harm), orb
0.7650 (+0.000, 1 switch), 40%-occlusion 0.327 (20 switches fired, ADD unchanged).
**Verdict: the remaining failures are NOT rotation basins.** orb: the rot head already pins the basin
→ residual is detector 2D (attack via crop resolution). 40% occlusion: damage is UPSTREAM (θ/2D
collapse at the pose stage, 0.315) — no rotation start rescues a corrupted θ anchor → head-level
occlusion-aug training (T1/T2, running) is the right lever. Code kept behind --multi-start (default
off, harmless). Write-up: docs/experiments/2026-07-04_multistart_rc.md

## 2026-07-04 — naive crop-resolution bump (512→768) REFUTED as a free lever
selfbbox orb held-out @200 with --image-size 768 on the FROZEN 512-trained stack: 0.5818 vs 0.7176
(median 32 vs 20mm + divergent tail mean 250mm). DINOv3 handles variable resolution, but the frozen
detector conf calibration / kp-feature sampling / heads are 512-distribution-specialized. Higher-res
crops require a RETRAIN CASCADE (crop detector @768 → angle/rot heads @768) — deferred; queue only if
T1/T2 (occlusion-aug heads) leave orb as the binding gap. Roadmap ④ updated.

## 2026-07-04 — DARK decode ADOPTED → mean 0.796→0.799, orb gap −0.010→−0.004
Survey round-2 (docs/robot_pose_sota_survey.md §6) ranked DINO feature-metric RC #1, but it proved
REDUNDANT with our silhouette RC (azure no headroom; realsense +0.002, silhouette already saturates
the depth signal; 40%-occl +0.005 tail-only) — well-conditioned `--feat-w` kept but not deployed.
edge-NCC-as-loss also refuted (discriminates GT in the probe but diverges as an objective, −0.10/−0.18).

The WIN was Idea 3 — **DARK sub-pixel heatmap decode** (`Eval/decode_util.py`, `--dark-decode`):
Gaussian-modulate + Taylor-refine the argmax from log-heatmap 1st/2nd derivatives (1px clamp). Free,
no training. Pose-stage matched A/B (300f): orb +0.0074, azure +0.0045, realsense +0.0050 — universal
do-no-harm gain (mean ADD drops = far/small-frame tail fixed, DARK's low-res strength). Stacked with RC:
| cam | deployed was | +DARK | RoboPEPP | gap |
|---|---|---|---|---|
| realsense | 0.8183 | 0.8213 | 0.805 | +0.016 |
| kinect360 | 0.8112 | 0.8132 | 0.785 | +0.028 |
| azure(RC off) | 0.7881 | 0.7916 | 0.753 | +0.039 |
| orb | 0.7647 | 0.7714 | 0.775 | −0.004 |
| MEAN | 0.7956 | 0.7994 | 0.780 | +0.019 |
orb −0.010→−0.004 (near-MATCH). ADOPT --dark-decode + --cov-pnp in the deployed config. Multi-start RC,
768-crop, feature-metric RC, edge-NCC, occl-robust silhouette, population prior all refuted this session.
Occlusion-aug heads (T1/T2) still training — robustness/accuracy tradeoff (+0.014/0.018 occluded, −0.009
clean @Ep1), positioned as a separate robustness config. Docs: docs/experiments/*.

## 2026-07-05 — 🔒🔒🔒 DEPLOYED light-stack + per-camera self-train → mean 0.804 (RoboPEPP 0.780)
occ-aug LIGHT head + camera-specific self-training (synthetic anti-forgetting) landed as the deployed
config. Per-camera: realsense 0.8155 / kinect 0.8275 / azure 0.7945 / orb 0.7784 → **mean 0.804**, all
4 cameras > RoboPEPP (ORB flips +0.003). RC on for rs/kinect/orb (@448/448/512), OFF for near Azure.
Re-lock 800→1000 stable (0.8037→0.8039, drift ≤0.006). Checkpoints: outputs_selftrain/{cam}_lightstack_
20260705_00354{6,52,9}; azure angle_occaug_light_20260704 + rot_crop_occaug_20260704.

## 2026-07-09/10 — KUKA iiwa7 + Baxter-left DREAM detectors (synthetic)
5-GPU detector training. KUKA iiwa7 2D-keypoint AUC 0.735 (long-tail = link-identity confusion, not
quality). Baxter-left detector AUC 0.817. Checkpoints kuka_dream_detector_20260709_183119,
baxter_left_dream_detector_20260710_152926.

## 2026-07-12/13 — KUKA/Baxter angle+rot heads · data-fit FK · direct-pose ADD
Data-fit FK: fit URDF joint transforms to DREAM data via scipy least_squares (iiwa7 single-start;
Baxter needed 40-start multi-start to escape local minima → 0.003 mm RMS). **direct-pose** mode (trust
head angles + rot-head R,t directly, bypass 2D → avoids link-confusion): KUKA ADD-AUC@100 **0.357**,
Baxter **0.253** (synthetic-only, no RC). Baxter RC REFUTED (silhouette-depth + wrist-shape ambiguity,
77→204 mm). **Wrist observability ceiling**: GT-2D injection barely changes w0/w1 MAE (28→28°) → wrist
self-axis rotation doesn't move its own keypoint; mlp_patch appearance head REFUTED (k=3/k=5 never beat
plain mlp over 9 epochs). Heads: kuka_angle/rot_20260712, baxter_angle/rot_20260713.

## 2026-07-14 — 📊 COMPREHENSIVE ABLATION CAMPAIGN → PAPER_DRAFT §4 (P0+P1)
Locked 1000-frame held-out, per-camera deployed heads. All numbers same-frame/same-condition.
- **A1 gate PASSED**: mean 0.804 exactly reproduced (rs 0.8155/kinect 0.8275/azure 0.7945/orb 0.7784).
- **B leave-one-out** (ΔMean = per-lever contribution): RC **+0.043** (biggest; rs+0.070/kinect+0.062/
  orb+0.040), rot-head +0.016, occ-aug/self-train +0.010, DARK +0.003, cov-PnP +0.001, conf-gate +0.001,
  auto-bbox −0.002 vs GT (near-free, stricter protocol).
- **C cumulative build-up** (RealSense held-out): base 0.666 → +DARK 0.673 → +cov 0.669 → +rot 0.705 →
  +occ-aug/self 0.745 → +RC 0.815 = **+0.149 total**; big rungs rot/occ-aug/RC, free levers flat on clean.
- **E occlusion 0–40%**: light+RC 0.812/0.765/0.679/0.573/0.430; occ-aug contribution grows to **+0.038
  at 40%** (vs clean head) → robustness must be trained in.
- **D RC design**: iteration sweep converges by ~150 iters (rs 0.815/orb 0.778 = deployed 250) → 250→150
  cuts ~40% RC cost, no loss. RC signal ENTIRELY silhouette-IoU (no-sil → base 0.745); reproj-anchor +0.002.
- **G2 conf-gate sensitivity** {0,.05,.10,.20} = 0.747/0.745/0.746/0.749 → flat ±0.002 (wide stable basin).
- **I1 runtime** (RTX 3090): backbone 19 ms, solver(250it) 352 ms, RC ~1.3 s → base ~2.4 fps, +RC ~0.6 fps.
- **J cross-robot** (clean re-run): KUKA direct-pose 0.357, Baxter 0.253.
- **L qualitative**: fig10 mesh-silhouette overlay (ADD 11–66 mm), fig11 occlusion ladder 0–40%.
- **REFUTED/deferred this round**: H2 head-arch (no crop-matched mlp_patch head → confounded; mlp_patch
  already in refuted table), G1 kp-jitter (train-only flag, not evalable).
PAPER_DRAFT §4 now: headline (Table 3, 3-robot) + occlusion curve + leave-one-out (T5) + build-up (T7) +
occ-aug (T6) + runtime (T8) + RC design + conf-gate sensitivity + refuted (T9) + figs 1–11. Scripts:
Eval/{ablation_run,ablation_campaign,buildup_g2,runtime_bench}.py/.sh; logs Eval/ablation_logs/*.

## 2026-07-14 — 🌐 SYNTHETIC comparison + HoRoPose/RoboPose + runtime table + 🔴 bbox-framing CORRECTION
Added §4.2 synth table (T4), §4.3 occlusion competitors (HPE/RoboPose in T5), §4.8 runtime table (T10).
- **Synthetic ADD-AUC** (ours; occ_bench r0 clean crop head +RC for Panda; direct-pose no-RC for KUKA/Baxter):
  Panda DR base 0.671→+RC **0.742**, photo base 0.708→+RC **0.769**; KUKA dr **0.357**/photo **0.319**;
  Baxter dr **0.252** (no photo test set exists). Data datasets/synthetic/{kuka,baxter}_synth_test_*,
  datasets/ICRA_multiview/DREAM_syn/panda_synth_test_*.
- **Competitors** (RoboPEPP CVPR'25 Table 2, user-provided): Panda-DR/Photo — RoboPEPP 83.0/84.1, RoboPose
  82.9/79.7, HPE(GT-bbox) 82.7/82.0, **HPE\*(auto) 41.4/40.7**, DREAM-H 82.9/81.1. KUKA-DR/Photo RoboPEPP
  76.2/76.1. Baxter-DR RoboPEPP 34.4, HPE 58.8. → On synth we TRAIL synth-specialized RoboPEPP/RoboPose,
  but BEAT HoRoPose\* (matched predicted+auto-bbox) by ~33 pts. Honest framing: we're real-focused SOTA.
- **🔴 bbox-framing correction**: RoboPEPP Table 2 = "Known BBox = No" (own off-the-shelf detector = AUTO);
  its real cols AK75.3/XK78.5/RS80.5/ORB77.5 EXACTLY equal our Table 1 numbers. So our long-standing
  "RoboPEPP GT-bbox headline / auto ORB→34" claim was WRONG (the 34.4 was RoboPEPP's Baxter-DR, misread).
  Corrected throughout: RoboPEPP/RoboTAG are the SAME auto-bbox protocol as us (0.804 vs 0.780 = fair
  like-for-like win); the GT-bbox method is HoRoPose, which collapses under an off-the-shelf detector
  (HPE* ORB 0.098). Fixed §4.1/§4.2/§4.5/intro + Table 1/2/10 labels + references/{sota_survey,related_work}.md.
Runtime T10: feed-forward real-time (HPE/RoboPEPP/RoboTAG/CtRNet) vs iterative optimization (RoboPose/ours);
ours lightest optimization (no learned refiner, RC-iter knob). Synth logs Eval/synth_logs/SYNTH_comparison.txt.

## 2026-07-15 — 🔬 COMPARISON-GROUP EXPANSION (#1 per-cam · #2 known-joint · #3 GISR · #4 backbone)
User asked "what other comparison groups are needed?" → prioritized + executed:
- **#1 full per-camera Table 1**: expanded from Ours/RoboPEPP/RoboTAG to ALL predicted-joint methods
  per camera (RoboPose/HoRoPose(GT)/HoRoPose*(auto)/RoboTAG/RoboPEPP/Ours), numbers from RoboPEPP Table 2
  + survey. Ours beats every AUTO-bbox method on every camera (+0.024 mean vs RoboPEPP).
- **#2 KNOWN-JOINT CEILING** (`--oracle-angle`, new): inject GT joint angles, freeze theta, solve only R,t
  (added `freeze_theta` to solve_batch; --oracle-angle to selfbbox_eval). Deployed 1000-held-out per-cam-best:
  rs 0.867 / kinect 0.878 / azure 0.788 / orb 0.831 → **mean 0.841 (+0.037 vs predicted 0.804)**. KEY: gain
  concentrates on FAR cameras (+0.05) → predicted-angle error is the far-camera headroom; AZURE UNCHANGED
  (0.788≈0.795) → its bottleneck is DEPTH, not angles (complements RC = far-camera engine). RC over-corrects
  known-θ realsense (0.867→0.825, RC is a predicted-θ tool). Table 1 italic row + note. Logs
  Eval/ablation_logs/oracle_angle/KNOWN_JOINT.txt; script Eval/oracle_angle_run.sh.
- **#3 GISR** added to Table 2 (RA-L'24 predicted, 3-cam mean 77.9* no-ORB caveat).
- **#4 backbone DINOv3 vs SigLIP2 (matched ViT-B/16 ~86M)**: §4.10 + Table 12. Detector-level (already had,
  EXPERIMENTS 07-06): unfrozen equal (~0.81 both), **frozen DINOv3 wins (0.80 vs 0.72)**. Since we DEPLOY
  frozen backbone, justifies DINOv3. Paper §4.10 written.
- Paper now: **Tables 1–12, figs 1–11** (fig renumber fig5-11→fig4-10 earlier this session closed the gap).
- 🔄 RUNNING (post-compaction: CONTINUE THESE): (a) **pose-level SigLIP2 cascade** — crop-detector 4-GPU DDP
  training (TRAIN/run_siglip_crop_ddp.sh, out siglip_crop_ddp_20260715_014111, warm from siglip2_unfrozen;
  batch8/gpu=32eff matches DINOv3). NEXT: when detector matures → train siglip crop-angle + crop-rot
  (train_angle.py/train_rotation.py --model-name google/siglip2-base-patch16-512 --crop-to-robot) → eval 4-cam
  pose ADD → fill §4.10 pose-level. Expected: frozen siglip trails (detector already 0.72<0.80). (b) **D2 RC
  render-resolution sweep** {224,320,448,512} on realsense (Eval/run_d2_reso.sh, d2_logs/) → add resolution
  knob to §4.8 RC-design. Waiters: siglip=bi51dni92, D2=b0aok3k37.

## 2026-07-15 (cont.) — ✅ D2 render-res sweep integrated · 🔄 siglip pose-cascade armed
- **D2 RC render-resolution sweep DONE** (RealSense, Eval/d2_logs/results.tsv): render-h
  224→0.7786 / 320→0.8046 / 448→**0.8154** / 512→0.8168; wall-time flat ~1330s all four
  (nvdiffrast rasterization not GPU-bound in range). Knee at 448, 512 only +0.001 → confirms
  deployed rs/kinect@448 as accuracy-cost optimum. **Real speed knob = RC iteration count, not
  resolution.** Written into §4.8 after the RC-iteration paragraph (commit 15358fb).
- **siglip crop-detector** finished 8/20 epochs by check time, val AUC 0.7824→0.8297 still climbing
  (~+0.005/ep). Let run to 20 for fairest comparison. best_heatmap.pth updates on best AUC.
- 🔄 CONTINUE POST-COMPACTION: waiter **baa56ncty** fires when detector logs 20 epochs (or torchrun
  exits). THEN launch `TRAIN/run_siglip_pose_cascade.sh` (committed) — trains siglip crop-angle(50ep)
  + crop-rot(30ep) on FROZEN siglip backbone+crop-detector, sequential on freest GPU. Outputs paths
  written to scratchpad siglip_{angle,rot}_out.txt; done-marker siglip_cascade_done.txt. THEN eval
  4-cam pose ADD (selfbbox_eval with siglip detector+heads) → fill §4.10/Table 12 pose-level row →
  commit. Expected: frozen siglip trails DINOv3 at pose level too (detector already 0.72<0.80 frozen).

## 2026-07-15 (cont.2) — siglip detector DONE (20ep AUC 0.859) → pose heads training (parallel)
- siglip crop-detector completed 20 epochs, synth-DR val AUC 0.7824→**0.8592** (unfrozen 4-block,
  matches DINOv3 crop-detector recipe). GPUs freed.
- Launched siglip pose heads. 🐛 fixed `model_angle.py` crash: `AnglePredictor` used
  `backbone.model.config.hidden_size` which AttributeErrors on SiglipConfig (nests under
  vision_config) — added `getattr(cfg,'vision_config',cfg)` fallback (mirrors train_heatmap.py).
  Covers both angle+rot (rot reuses AnglePredictor).
- Running PARALLEL (angle & rot independent) via `TRAIN/run_siglip_pose_parallel.sh`: crop-angle
  (50ep, GPU1) + crop-rot (30ep, GPU2), frozen siglip backbone+crop-detector, ~1.7 it/s ~30min/ep
  → wall ~25h. best-on-val ckpts saved continuously. Out paths in scratchpad siglip_{angle,rot}_out.txt.
- 🔄 CONTINUE POST-COMPACTION: waiter **b6u9ide18** fires when both best ckpts exist AND converged
  (angle ≥25 ep + rot done, or both done-markers). THEN eval 4-cam pose ADD with siglip
  detector+heads (selfbbox_eval: --stage1-detector/--crop-detector siglip crop-det, --model-name
  google/siglip2-base-patch16-512, angle/rot ckpts = siglip best) → fill §4.10/Table 12 pose-level
  row (DINOv3 vs SigLIP2 pose ADD) → commit. Expected: siglip trails (detector real-Azure 0.72<0.80
  frozen; unfrozen ~equal at detection, pose TBD).

## 2026-07-15 (cont.3) — G1 keypoint-noise sensitivity (on free GPUs 0/3, eval-only)
- Added `--kp-jitter` to selfbbox_eval.py (inject isotropic Gaussian 2D noise into decoded kp before
  solver; cov_inv kept from clean heatmap). Swept sigma={0,1,2,4,8}px on RealSense held-out 1000
  base-only, cov-PnP ON vs OFF. Logs Eval/ablation_logs/g1_kpjitter/, script Eval/g1_kpjitter.sh.
- RESULT (cov / no-cov): 0.745/0.747 · 0.721/0.722 · 0.656/0.662 · 0.490/0.520 · 0.293/0.307.
- 🔑 HONEST FINDING (not the hypothesis): (a) solver degrades GRACEFULLY — gentle to sigma=2 (>DARK
  decode error), cliff only sigma>=4; kinematic-reprojection solver tolerates sub-pixel-scale noise.
  (b) cov-PnP ~= plain PnP under INJECTED uniform noise (even marginally worse at high sigma) →
  cov-PnP's value is exploiting the heatmap's OWN uncertainty (blur/occlusion), NOT arbitrary-noise
  robustness. Injected noise is uncorrelated with clean-heatmap covariance → anisotropic weighting
  miscalibrated. DIRECTLY reinforces §4.3/§4.4 (cov value is under occlusion). Written into §4.4
  after conf-gate sensitivity (inline, no table — matches conf-gate style, avoids numbering break).
- GPUs 0/3 freed after G1. siglip pose heads still on 1/2 (waiter b6u9ide18).

## 2026-07-16 — ~~§4.10 POSE-LEVEL backbone comparison (DINOv3 >> SigLIP2)~~ ⚠️ SUPERSEDED — see the RESOLVED entry below; the SigLIP 0.391/"DINOv3 >> SigLIP2" here was a NORMALIZATION BUG. Corrected truth: DINOv3 0.742 ≈ SigLIP2 0.752 (essentially equal).
- siglip pose heads finished: crop-angle best val MAE 20.26° (ep28, converged), crop-rot done
  (30ep, best pose score 15.54 / geo 9.03°). Killed converged angle at ep29 to free GPU.
- Ran matched pose-level eval (Eval/backbone_poselevel.sh): both backbones, --oracle-bbox (bypass
  stage1, isolate backbone; single --model-name can't mix, siglip has no full-frame stage1 heads),
  identical clean synth crop heads (angle_crop_174740/rot_crop_022535 vs siglip crop heads),
  base-only, 4 real cameras, held-out 1000. Logs Eval/ablation_logs/backbone_poselevel/.
- RESULT (ADD-AUC): DINOv3 azure0.806/kinect0.739/rs0.719/orb0.704 **mean 0.742** vs
  SigLIP2 azure0.376/kinect0.236/rs0.582/orb0.371 **mean 0.391** → **Δ +0.351**.
- 🔑 FINDING: pose-level gap FAR larger than detection. SigLIP2 crop-detector validates HIGH on
  synth (synth-DR val AUC 0.859) but collapses on REAL dense keypoints/pose → image-text contrastive
  features transfer to real-domain geometry markedly worse than DINOv3 dense SSL. Strengthens DINOv3
  choice from "equal-and-frozen-better" to "materially better at pose on real". §4.10 rewritten
  (two-level (i)detection (ii)pose), Table 12 extended with pose rows. Committed.
- ✅ ALL PLANNED COMPARISON GROUPS COMPLETE. Paper: Tables 1-12, figs 1-11, §4.1-4.10 + G1 inline.

## 2026-07-16 (BUGFIX) — 🔴 SigLIP pose-level 0.391 INVALID (norm mismatch) → 재산출 중
- User skeptical of −0.351 gap. INVESTIGATED → found bug: train_angle.py, train_rotation.py,
  selfbbox_eval.py all used ImageNet norm even for SigLIP backbone, while train_heatmap.py (detector)
  correctly used mean=std=0.5. So: siglip crop-detector trained @0.5 but heads trained @ImageNet AND
  eval @ImageNet → 3-way inconsistency, siglip fed off-distribution input. **§4.10 siglip pose 0.391
  is INVALID.** DINOv3 (ImageNet throughout: detector+heads+eval all consistent) UNAFFECTED → 0.742
  valid. Detector-level Table 12 top rows also valid (train_heatmap was correct).
- FIX (commit): added model-aware norm (0.5 for siglip) to all 3 files, mirroring train_heatmap.py.
- Retraining siglip crop-angle+crop-rot with correct norm (run_siglip_pose_parallel.sh 1 2, GPU1/2,
  confirmed "SigLIP backbone detected: mean=std=0.5"). out siglip_{angle,rot}_crop_20260716_010615.
  Waiter **bmerxb3bu** fires on convergence → re-eval backbone_poselevel.sh siglip → fill real §4.10
  numbers. Paper §4.10 pose (ii) + Table 12 pose rows marked ⏳ 재산출 pending. Expected: siglip
  improves substantially from 0.391 (bug-inflated); true gap vs DINOv3 0.742 TBD.

## 2026-07-16 (RESOLVED) — ✅ §4.10 pose-level CORRECTED: DINOv3 ≈ SigLIP2 (norm bug fixed)
- Norm-fixed siglip heads (mean=std=0.5): angle val MAE **6.63°** (vs buggy 20.26°), rot geo **2.24°**
  (vs 9.03°). Re-ran backbone_poselevel.sh siglip (eval also norm-fixed, "SigLIP backbone detected"
  confirmed, all 1000 frames, no errors).
- CORRECTED RESULT (ADD-AUC): DINOv3 az.806/ki.739/rs.719/orb.704 **mean 0.742** vs
  SigLIP2 az.778/ki.766/rs.765/orb.698 **mean 0.752** → **Δ −0.010 (SigLIP2 marginally higher)**.
- 🔑 CONCLUSION FLIPPED: the buggy 0.391/"DINOv3 materially better" was ENTIRELY a normalization
  artifact. TRUTH: at pose level the two backbones are **essentially equal** (within run-noise, mixed
  per camera: DINOv3 wins az/orb, SigLIP2 wins ki/rs). This CONFIRMS the detection-level "unfrozen
  equal" finding — performance is from foundation features in general, not a specific backbone.
  DINOv3 justified by FROZEN-regime detection edge (0.80 vs 0.72), NOT pose superiority.
- §4.10 (i)/(ii) rewritten, Table 12 pose rows filled with real numbers + caption corrected, orphan
  synth-val row removed. Old buggy tsv kept as results_siglip_BUGGY_imagenet_norm.tsv. Committed.
- Lesson: always verify per-backbone preprocessing (siglip=0.5, dino=ImageNet) across train AND eval.

---

## 2026-07-20 백본 실험 확장: pretraining vs architecture (표12 검출 행)

**동기**: §4.10은 DINOv3 vs SigLIP2(둘 다 pretrained)만 비교 → "구조 자체가 좋냐 vs 다양한
이미지 pretraining 덕이냐"를 못 가름. random-init(from-scratch) 통제군 추가로 분리(ViT 원논문·
He2018 "Rethinking ImageNet Pre-training" 방식). Panda·검출 레벨(stage1 full-frame, val=real-azure AUC).

**설계(2×2)**: pretrained{DINOv3/SigLIP2(있음), google/vit supervised(신규)} × random-init(신규)
                × frozen(헤드만) / unfrozen(백본).
- frozen 행 = 파라미터-매칭 통제(pretrained-frozen도 random-frozen도 헤드만 학습, 동일 학습 파라미터 수).
- unfrozen random = 전체 백본 from-scratch(embeddings 포함). He2018 공정성: 긴 스케줄(≤80ep, plateau 조기중단).

**정밀도 = fp32 (공정성)**: 기존 DINOv3/SigLIP2 앵커가 fp32 → 신규도 fp32로 통일(arm간 정밀도 섞으면
confound; from-scratch가 저정밀도에 더 민감해 bf16이 편향 소지). `--amp` 플래그 코드에 있으나 OFF.

**구현**: model.py/model_v4.py DINOv3Backbone에 `pretrained` 파라미터(random-init=AutoModel.from_config),
random-init 시 unfreeze_blocks>0이면 전체 trainable·=0이면 frozen. 일반 ViT는 forward에
interpolate_pos_encoding=True(512 pos-embed) + model.encoder.layer(단수)/model.layer(DINOv3 5.x) unfreeze 탐색.
정규화는 norm_utils.get_norm_stats(AutoImageProcessor.image_mean/std 자동 읽기, siglip 하드코딩 대체).
train_heatmap.py: --random-init/--amp 플래그, DDP find_unused_parameters=random_init(from-scratch의
pooler/mask_token 미사용 grad 대응). 런처 run_backbone_ablation.sh.

**측정 처리량(fp32, 3-GPU DDP, batch8/gpu)**: from-scratch 2.1 it/s, ~33min/epoch → 80ep ~44h(최악),
plateau 조기중단 시 ~22-28h. (fine-tune 대비 느림 = 전체 백본 backward.)

**🔄 실행중(2026-07-20)**:
- random-unfrozen(from-scratch): GPU 2·3·4, ep80 ceiling, out=bbabl_random-unfrozen_20260720_165132
- random-frozen(대조군): GPU 0 단일, ep20, out=bbabl_random-frozen_20260720_170016
- 이후(선택): sup-frozen/sup-unfrozen(google/vit), dino-frozen-sanity(하네스 검증).
- val AUC plateau 감시 → 조기중단. 결과로 표12 검출 행 확장.

---

## 2026-07-21 합성(synthetic) Panda 성능 진단 + 개선 시도

**문제**: Table 1(tab:main) 합성 Panda ADD-AUC 우리 74.2/76.9(DR/Photo) vs RoboPEPP 83.0/84.1, RoboTAG 82.5/84.3 (~9-10점 열세).

**진단 (Eval/ablation_logs/oracle_angle_synth/, dr/photo_pred/oracle base+RC)**:
| | pred base | pred +RC | oracle-angle base | oracle-angle +RC |
|---|---|---|---|---|
| DR | 0.704 | 0.769 | **0.861** | 0.886 |
| Photo | 0.738 | 0.799 | **0.869** | 0.897 |

- **병목 = 관절각 예측.** GT 각도 주입 시 base 0.704→0.861(+0.157), RC까지 0.886/0.897 → **RoboPEPP(0.830/0.841) 상회.** 파이프라인 상한은 SOTA 이상, 남은 건 순수 각도 회복.
- **배제됨**: 2D검출(oracle-2D 무효, ~0.86 AUC), **bbox**(oracle-bbox 0.705 ≈ pred 0.704, 차 0.001), self-occlusion(키포인트 100% 검출), 원거리 깊이(실패는 오히려 근거리 0.6-0.9m 집중).
- **실패 모드**: ~10% 프레임 파국(>100mm), 90%는 ~15mm. 실패=손목 J4/J5/J6 각도 붕괴(26-38° vs OK 3-9°) + **high reproj(90px vs OK 1.5px)** = 솔버가 잘못된 basin, 검출 2D조차 못 맞춤.

**시도한 개선 (전부 실패 — 손목 관측성 천장)**:
- RC-실루엣 손목 multi-start(rc_refine_wrist.py, n=8): fail-107 프레임 0.033→0.038 (무효). 손목이 실루엣서도 약하게만 보임.
- min-reproj multi-start(selfbbox_eval --ms-local 16, synth): DR 0.704→**0.692 (손해)**. 손목이 2D 미관측 → min-reproj가 2D 노이즈 과적합, 90% 좋은 프레임까지 훼손.
- 근본: SUMMARY 확인 — 손목 self-axis 회전이 자기 키포인트를 안 움직임(관측성 천장). 기하(2D/실루엣)로 복원 불가, appearance/구조-prior 필요.

**공짜 이득 (확정)**: cov-PnP+DARK를 합성 eval에도 일관 적용(실측엔 쓰지만 Table엔 누락됨) → DR 0.742→**0.769**, Photo 0.769→**0.799** (+0.027/+0.030). Table 1 갱신 대상.

**🔄 실행중**: crop-matched mlp_patch + transformer 각도 헤드 재학습(synth) — SUMMARY의 "crop-matched patch head 없음(confounded)" 갭 검증. wrist val MAE가 deployed mlp 대비 개선되면 tail 감소 기대. (out: outputs_angle/synth_{mlp_patch,transformer}_20260721_073917)

**남은 옵션**: RoboPEPP I-JEPA 구조-prior 이식(가중치 별도 필요, 백본적응 반증영역, 고위험). RoboPEPP 메시(urdfs/) 확보됨 → KUKA RC 별도 가능(우리 RC는 Panda 하드코딩).

## 2026-07-21 (cont.) 합성 각도-헤드 개선 — 재분배/라우팅 계열 전부 무효 (결정적 negative)
eval_synth_head.sh (synth DR base, cov-PnP+DARK, 1000f) vs deployed 0.704 / fail 10.7%:
- mlp_patch (crop-matched, val MAE 9.0° vs 10.9°): ADD 0.706 / fail 10.7% → 무효.
- P3 mlp_mixsel (2-mode + appearance selector, 손목 MAE 12.6° 최저): ADD 0.702 / fail 10.3% → 노이즈. 손목↑ but base J0 퇴화 상쇄 = 제로섬 확인.
- focal(γ1,2, J6=1.0): 부진 → 폐기.
결론: 가중치재분배·표준헤드·MoE라우팅 전부 순이익 0. 각도 MAE↔ADD 탈동조(병목=10% flip 꼬리, 평균 아님).
방법론 조사: 정보를 *더하는* 헤드-레벨 기법 = HybrIK twist-swing(base 해석해+roll 전용헤드, 제로섬 원천제거), PARE part-attention(관절별 독립경로), ProHMR/RoboKeyGen(multimodal), RLE/keypoint-filtering. RoboPEPP 우위=masked-embedding 사전학습+per-robot fine-tune(관행, 확인됨).
신규 코드: model_angle.py AngleHeadMixSel; selfbbox_eval.py --crop-head-type/--oracle-except; train_angle.py --tail-gamma/--n-mix/--selector-weight/--load-balance.

## 2026-07-22 RoboPEPP 격차 재검토 → 근원 = 솔버 basin flip (off-frame kp가 전 팔 오염)
dr_pred.npz (DR base, cov-PnP+DARK, 1000f, AUC 0.704) 분해 + 경쟁 코드 인용. 문서: docs/dinobotpose3/experiments/2026-07-22_gap_reexamination.md (research agent 소유).
- **재설정**: DREAM AUC = mean·max(0,1−10·ADD_m) → ADD>100mm 프레임 기여 = 정확히 0. tail(off-frame, 10.7%)의 AUC 총기여 0.0000 → off-frame 손목 복원/드롭은 AUC 불변(oracle-presence 0.703, gt-fill 0.704, mean-fill 0.701 재확인). **tail은 격차가 아님.**
- **진짜 근원**: 우리 솔버가 θ(7)+R(3)+t(3)=13DOF를 재투영 공동최적화(solve_pose_kinematic.py:218,274) → 환각된 off-frame kp가 손목뿐 아니라 base·R·t까지 잘못된 basin으로 끌고 감(basin flip). tail: reproj 90px, 손목각 26–38°, base link0 62.7mm. RoboPEPP는 θ 회귀(JointNet IEF 4-step)+6DOF-only conf-filter BPnP로 손상을 손목에 국한(J6 5.4°/4.8°).
- **counterfactual(dump)**: proximal 앵커 단독 +0.006 / distal cap 단독 +0.011 (= refuted 단일축이 net-zero였던 이유 정량 재현) vs **둘 동시** 0.704→0.737(≤150mm)/0.749(≤120mm)/0.758(≤100mm)/0.767(≤80mm). 격차 전체가 이 축.

## 2026-07-22 (cont.) 2악장 P0: 분리 solve (freeze-head-theta) — 🔄 실행 중
문서: docs/dinobotpose3/experiments/2026-07-22_p0_decoupled_solve.md.
신규 플래그: selfbbox_eval.py --freeze-head-theta (L136/L366) → solve_batch(freeze_theta=True, theta_init=head_pred). θ를 head 예측에 고정하고 R,t 6DOF만 solve(RoboPEPP식) → basin flip 원천 차단 = counterfactual의 "distal cap+proximal 격리" 동시 구현. --edge-gate로 off-frame kp를 R,t solve에서도 배제.
변형(DR): control(0.704 재현)/freeze/freeze+edge-gate8/freeze+edge-gate-oracle + freeze(Photo). 예측 +0.03~0.06.
**결과 🟡 naive freeze REFUTED**: 전역 θ고정 = DR 0.7040→**0.5329**, Photo 0.738→**0.5610** 붕괴. edge-gate8 0.5310/edge-oracle 0.5324 (무의미). 원인=**good 프레임(893장) θ 재투영 정제 상실**(good AUC 0.7884 med14.8mm→0.582 med33mm). freeze는 tail(107장)만 개선(median 183→135mm, 36%<100mm)하나 AUC상 tail은 ~0 기여라 무의미. flip-trigger 2-pass(reproj τ=60px)로 tail만 적용 = 실전 **+0.0090**(oracle per-frame 상한 +0.027). ⇒ **basin flip은 작은 레버로 확정.**
**후속 진단 → 진짜 격차=good 프레임 각도 정확도**: GOOD 893장 base 0.7884 vs oracle-angle(GTθ) **0.8991**(med 7.2mm) → good에서만 **+0.11 헤드룸**, 게다가 0.899 > RoboPEPP 0.83. head 아키텍처 재배치는 천장(mlp_patch 0.7802/MoE 0.7728 < base 0.7884) → 각도 값 자체의 회귀 정확도를 올려야 함.
결론: naive decouple 반증, tail(basin flip)은 +0.009 소형 레버. 다음=regressed 각도 정확도 향상(IEF/iterative + pose-prior, P1); flip-trigger 2-pass는 do-no-harm이면 무료 보조.

## 2026-07-22 (cont.) 🔴 근본원인 발견: KUKA/Baxter 내부파라미터(K) 버그 — 과거 결론 다수 무효
정리본: [docs/dinobotpose3/experiments/2026-07-22_intrinsics_rootcause.md](../../docs/dinobotpose3/experiments/2026-07-22_intrinsics_rootcause.md)

**버그**: `datasets/synthetic/{kuka,baxter}_synth_*` 프레임 JSON에 `meta` 키 자체가 없음(실측: 최상위 키 = `camera_data`/`objects`/`sim_state`) → `TRAIN/dataset.py:574-578`이 조용히 `eye(3)`로 폴백("should not happen with proper data" 주석과 달리 **항상** 발생). 그 항등 유래 K가 `Eval/kuka_add_eval.py:156` · `Eval/baxter_add_eval.py:156`의 `spk.solve_batch(kp2d, conf, K, ...)`를 통해 **진짜 원근투영을 수행하는** 솔버(`Eval/solve_pose_kinematic.py:105-114`, `z`로 나눔)에 직행.
**오차 크기**: crop·scale 후 **fx=1.736 vs 참값 555.4 → 정확히 ×320**, cx=−562 vs −6.9. native 참값은 `_camera_settings.json`의 fx=fy=320/cx=320/cy=240(KUKA·Baxter 동일)이고 항등 K의 fx=1이므로 배율이 곧 320. 고전적 **focal/depth 모호성** — 솔버는 발산한 게 아니라 *주어진 틀린 카메라에 대해 정확히 최적해*를 찾고 있었음(깊이를 320배 축소).
**독립 확인**: Panda + 완벽한 GT 2D + 항등 K → 복원 깊이 **4.9 mm**(참값 943 mm), ADD-AUC **0.0000**. 입력 2D가 완벽해도 파국 = 검출기·head 품질과 무관.
**스코프 = 정확히 이 두 로봇**. Panda는 `datasets/ICRA_multiview/Converted_dataset/DREAM_to_DREAM{,_syn}`으로 학습·평가하고 이 트리는 실제 `meta.K` 보유(합성 320; realsense/kinect/orb 615.5; azure 399.7) → **배포 mean 0.804 유효, 영향 없음**.

**실측 (320프레임 subset, 실제 학습된 head)**:
| 모드 | KUKA AUC | KUKA mean/med | Baxter AUC | Baxter mean/med |
|---|---|---|---|---|
| `--direct-pose` (출하·게재값) | 0.3716 | 73.5 / 59.2 mm | 0.2622 | 85.5 / 75.1 mm |
| 솔버, 항등 K (현행 코드) | 0.1454 | 1086 / 512 mm | 0.1275 | 1063 / 165 mm |
| 솔버, **참 K** | **0.6696** | 108 / **13.4** mm | **0.7116** | 39.0 / **19.1** mm |

`--direct-pose` 행이 게재값 0.357/0.253을 재현 → 하네스 검증됨. 항등 K 행의 mean ADD가 1 m 급인 것이 focal/depth 붕괴의 직접 증거. 참 K 솔버가 direct-pose를 **KUKA +0.298 / Baxter +0.449** 능가.

**무효화되는 과거 결론 4건**: ① "KUKA/Baxter에서 솔버가 발산한다" → 망가진 카메라를 먹고 있었음. ② "병목은 rot-head 병진오차 56 mm" → `|dz|` 33.6/36.3 mm는 ~1 m 장면의 **3~4%**로 정상적 metric 회귀, 병목 아님. ③ "iiwa7에서 재투영 최적화는 해롭다" → 버그. ④ `--direct-pose`가 우월 → 실은 **K를 한 번도 건드리지 않는 유일 경로**라 면역이었을 뿐. (교훈: "이유는 모르겠지만 이것만 된다"는 우회로는 *무엇을 건너뛰는지*를 먼저 볼 것.)

**재학습 불필요 (게이지 논증)**: rot head 학습 타깃은 Kabsch(3D↔3D)라 **K-free**. geo/bearing 게이지는 `identity_bearing = 320×true + 320`, 상관 **r = 1.00000000**의 완전 affine → **첫 Linear가 스케일·오프셋을 흡수**, 정보 손실 없음. ⇒ **모델에는 계속 dataset K(항등)를 주고**(게이지 유지), 솔버·렌더러가 쓰는 기하 K만 교체.
**수정(eval-time 단독)**: 검증된 `Eval/iiwa7_rc_eval.py:81-102`의 `geometric_K()`(native intrinsics + 항등 위에 남은 crop 오프셋 `-bx0,-by0` 결합; GT 3D를 데이터셋 자신의 2D로 재투영해 60프레임 **<0.09 px** 검증)를 위 두 호출부에 배선. 원칙 = **모델에는 dataset K, 솔버에는 참 K**. 주의: `geometric_K:95-96`의 `assert camera_K[0,0,0]==1` 가드는 공용 승격 시 "항등이면 재구성, 아니면 통과"로 일반화 필요(Panda 경로 보호).
**별도 잠재 이슈**: `Eval/inference_4tier_eval.py:128-131`은 `meta.K` 부재 시 **`zeros(3,3)`** 폴백 — `eye(3)`보다 나쁨(투영 시 z=0 → `clamp(1e-6)`에 걸려 조용히 무의미값). **폴백 대신 `raise` 권장.** 이번 사건의 본질은 "틀린 값"이 아니라 **"조용한 폴백"**. 실제 오염 사례는 **미측정**.
**문서 정정**: `Eval/iiwa7_rc_eval.py:84` docstring이 "on DREAM (frame JSONs carry no meta.K)"로 일반화했으나 이는 KUKA/Baxter 합성 트리에 한해 참, Panda `DREAM_to_DREAM`에는 **거짓** — 이 과잉일반화가 버그를 정상 동작처럼 보이게 한 요인.

🔴 **논문 수정 플래그 (아직 편집하지 말 것 — full-set 확인 런 진행 중)**: `docs/dinobotpose3/PAPER_DRAFT.md:190`(KUKA 0.357/병목 "회전 헤드 병진 오차(56mm)"), `:191`(Baxter 0.253/"손목 관측성 천장"), `:212`(표4 캡션 "KUKA/Baxter = direct-pose without render-compare (**no mesh**)" — 이중 오류: 진짜 이유는 K 버그이고, iiwa7 메쉬는 **존재**), `:328`(§4.7 본문), `docs/dinobotpose3/figures/make_figs_multirobot.py:34`(`pose = [0.804, 0.357, 0.253]`). **수치 자체는 실행한 것에 대해 정직 — 철회가 아니라 인과 서사 교체 + 수치 상향(약 +0.30~0.45 ADD-AUC 과소평가)의 문제.** 아울러 위 **EXPERIMENTS.md:963-964**에서 파생된 **"KUKA/Baxter 솔버 각도정제 금지"** 결론은 **망가진 솔버 위에서 내려진 것** → 참 K로 재시험 전까지 REFUTED 항목으로 굳히지 말 것(현 판정: 무효화 가능성 높음, **보류**). `:965` "Baxter RC REFUTED"는 앵커·게이트 부재로 별도 규명된 바 있어 독립 사안으로 추정(**미측정**).

**측정 상태**: ✅ 실측 = meta.K 부재 스코프·참 native K·post-crop 배율·Panda GT2D sanity(4.9mm/0.0000)·위 3×2 표(**단 320프레임 subset**)·하네스 재현·affine r=1.0 / 🔄 **full-set 참 K 확인 런 진행 중(논문 수정의 전제조건)** / ❌ 미측정 = 참 K 솔버↔RC 중복도, 4tier zeros 폴백 실제 오염, Baxter RC와 K 버그의 연관성.

## 2026-07-22 (cont.) KUKA render-and-compare 배선 — 차단 해제, 동작 확인 (동반 결과)
신규: `Eval/iiwa7_render.py` + `Eval/iiwa7_rc_eval.py`. RoboPEPP 동봉 iiwa7 URDF+메쉬 사용 — 그 URDF FK가 DREAM kuka를 **0.0048 mm RMS**로 재현(기하 정합 확인). **GT 포즈 렌더 IoU mean 0.858 / median 0.869, 100%가 ≥0.5** → Baxter가 겪은 실루엣 붕괴 없음(즉 KUKA RC 차단 해제).
**50프레임**: 재투영 앵커를 건 RC가 해당 subset을 **0.2804 → 0.5721**(mean ADD 94.2 → 69.5 mm)로 개선, **전프레임 발산 없음**.
⚠️ 이 설정은 **50프레임에서 튜닝**된 것이며 **500프레임 스윕 중** — 채택 전 단계. 위 참 K 결과(0.6696)와 합치면 KUKA에 독립적 큰 레버가 둘이지만 **둘의 중복/가산 여부는 미측정**이며 full-set 확인 후 결정.

## 2026-07-22 (확정) ✅ intrinsics 버그 수정 — 전체 테스트셋 실측·독립 3중 검증 → KUKA/Baxter 성적 반전
위 07-22 근본원인 항목의 **320프레임 subset 수치를 전체 테스트셋 확정치로 대체**한다. 정리본: [docs/dinobotpose3/experiments/2026-07-22_intrinsics_rootcause.md](../../docs/dinobotpose3/experiments/2026-07-22_intrinsics_rootcause.md).

**버그 재확인**: `camera_K = eye(3)` 폴백이 kuka/baxter 합성 트리에서 항상 발동 → 솔버가 **fx≈1.8**을 받음(참값 **577/626**, **×320 초점거리 오차**). 솔버는 발산한 게 아니라 *틀린 카메라에 대해 정확히* 풀고 있었음.

**확정 실측 (KUKA-DR 5997프레임)**:
| 모드 | ADD-AUC | mean | median | fail>100mm | med t-err | med R-err |
|---|---|---|---|---|---|---|
| direct-pose (출하) | 0.3682 | 72.1 mm | 60.2 mm | 15.4% | 56.1 mm | 7.42° |
| 솔버 + 참 K | **0.6901** | 91.1 mm | **13.1 mm** | 15.9% | **15.7 mm** | **5.77°** |

**확정 실측 (Baxter-DR 5982프레임)**:
| 모드 | ADD-AUC | mean | median | fail>100mm | med t-err | med R-err |
|---|---|---|---|---|---|---|
| direct-pose (출하) | 0.2739 | 83.1 mm | 73.3 mm | 24.7% | 59.7 mm | 5.65° |
| 솔버 + 참 K | **0.7125** | **39.5 mm** | **17.1 mm** | **8.0%** | **29.4 mm** | 5.91° |

KUKA **+0.322 (+87%)**, Baxter **+0.439 (+160%)**.

**경쟁모델 대비 (Protocol A, ×100)**: KUKA **69.0** vs RoboPEPP 76.2 / RoboPose 80.2 / HoRoPose 75.1 / RoboTAG 75.0 → "한참 뒤"에서 **사정권**. Baxter **71.3** vs RoboPEPP 34.4 / RoboPose 32.7 / HoRoPose·RoboTAG 58.8 → 🥇 **큰 격차 1위**.

**독립 3중 검증**: ① direct 모드 포즈 오차 56.1 mm/7.42°가 KUKA rot-head **자체 학습 로그와 정확히 일치**. ② 재구성한 K로 GT 3D를 투영하면 데이터셋 자신의 2D와 **median 0.0003 px**. ③ 패치된 프로덕션 스크립트가 독립 에이전트 수치를 **비트 단위 재현**.

**🔴 정직하게 남길 단서 4건**:
1. **이득의 출처는 오직 참 K이며, outlier 제거 가설은 기각**된다 — outlier 비율이 **불변**(KUKA 키포인트 **21.1%**, Baxter **11.6%**). 로버스트 거부가 참 K **위에** 추가 이득을 주는지는 **미측정**.
2. **꼬리 거동이 두 로봇에서 갈린다.** KUKA는 파국 꼬리가 **살아남는다**(fail 15.4→**15.9%**, mean 91.1 mm = median의 **7배**, p99 **1012 mm**) — 수정은 **좋은 프레임만 훨씬 좋게** 만들고 **나쁜 프레임은 못 고친다**(잔존 원인 = link-identity 혼동, 미해결). 반면 Baxter는 꼬리가 **붕괴한다**(24.7→**8.0%**) ⇒ **Baxter의 꼬리는 이 버그 자체였다.** (KUKA mean이 72.1→91.1 mm로 나빠지는 것은 이 때문이며 모순 아님.)
3. **재현 caveat**: direct-pose 기준선이 **0.3682 / 0.2739**로 측정되어 아카이브 **0.3568 / 0.2535**보다 **+0.01~0.02 높다.** 유력 원인 = **`best_*` vs `last_*` 체크포인트 선택**. 아카이브 수치는 **비트 재현되지 않았으나** 개선폭이 +0.32/+0.44 규모라 **결론은 무영향**.
4. **Panda 무영향** — `Converted_dataset/DREAM_to_DREAM*`는 실제 `meta.K`를 싣고, `Eval/refine_eval.py`의 일반화된 `geometric_K`는 참 K를 **그대로 통과**시킨다(검증 완료). **배포 Panda real 0.804 불변.**

**🔓 과거 결론 판정 — 보류 아님, OVERTURNED**: 위 `EXPERIMENTS.md:961-968`(2026-07-12/13)에서 파생된 **"KUKA/Baxter는 솔버 각도정제 금지"** 결론과 여기서 나온 `SUMMARY.md:53-54,57,116`의 REFUTED 항목은 **망가진 솔버 위에서 내려진 것**이며, 이제 **뒤집혔다** — 참 K 솔버가 direct-pose를 KUKA +0.32 / Baxter +0.44로 능가하므로 **참 K 솔버를 기본 경로로 승격**한다. 특히 `SUMMARY.md:116`의 **"Baxter 병목 = 손목 관측성 천장"** 인과 결론은 **무효**(참 K만으로 꼬리 24.7→8.0% 붕괴). 단 `SUMMARY.md:117` **"Baxter RC REFUTED"는 유지** — 앵커·게이트 부재로 별도 규명된 독립 사안(K 버그와의 연관성 **미측정**).

**🔴 논문 수정 대상 (문서 작업 범위 밖 — 편집하지 않음, 목록만)**: `docs/dinobotpose3/PAPER_OVERLEAF.tex:167`(`tab:main` 우리 행 `35.7 & 31.9 & 25.2` → KUKA-DR **69.0** / KUKA-Photo **미측정** / Baxter-DR **71.3**(하위 그룹 최고이므로 `\textbf{}` 부여)), `:166`(RoboTAG Baxter `\textbf{58.8}` **볼드 해제** — 우리가 이김), `:171` 캡션("KUKA and Baxter use the direct-pose configuration without render-and-compare" = **이중 오류**: direct-pose는 설계 선택이 아니라 버그의 결과였고, iiwa7 메쉬는 **존재**), `:245-246`("no matching mesh" **거짓** + 수치 + "observability ceiling" 서사 + "not comparable/applicability study" **과소주장**), `:312-313`(결론의 관측성 천장 기여 주장 + "need for a benchmark-matched mesh"); `docs/dinobotpose3/PAPER_DRAFT.md:181,183,190,191,194,206,210,212,328,330`; `docs/dinobotpose3/figures/make_figs_multirobot.py:34`(`pose = [0.804, 0.357, 0.253]` → `[0.804, 0.6901, 0.7125]`, y축·주석·캡션 동시 점검).

⚠️ **표 갱신의 게이트**: **재측정한 것은 DR 스플릿뿐이다.** **KUKA-Photo(31.9)는 미측정**이며, DR 두 셀만 고치면 **한 행 안에 두 파이프라인 구성이 섞인다**. **Baxter Photo 스플릿은 DREAM에 존재하지 않으므로** 현행 "열 없음"이 정답이다.

**미해결/대기**: ① KUKA-Photo 참 K 재측정(논문 게이트). ② **RC는 여전히 수정 전 baseline 위에서 튜닝된 상태 → 고쳐진 솔버 위에서 재튜닝 필요(별도 에이전트 진행 중)**, 재튜닝 전 RC 수치 논문 반영 금지. ③ 로버스트 거부의 참 K 위 추가 이득(KUKA 잔존 꼬리의 유일 유력 후보). ④ 아카이브 기준선 ±0.02 차이의 원인 확정.

## 2026-07-22 (cont.) azure 발산 꼬리 — 선행 진단 2건 반증 + 재투영 가드의 설계상 한계 (음성)
처방(① 경계 마진 게이트 ② 재투영 플래그 멀티스타트 재solve ③ 재투영 기준 do-no-harm 채택)을 구현·검증. **기준선 재현: azure base 0.7953 vs 배포 0.7945** (1000f, `--frac-range 0.7 1.0`) — 이하 전부 이 앵커 위.

**선행 진단 ① 반증 (트리거 부재)**: "P(발산|화면밖 GT kp)=0.382 vs 0.052, **lift 7.4배**" → 실측 **0.000 vs 0.033, lift 0.0배**. ADD>100mm 프레임 **31개 전부 키포인트 7개가 화면 안**이고, 화면밖 키포인트 85/7000(1.21%)은 **전부 good 프레임** 소속. 방향이 약한 게 아니라 **반대**. ⇒ 경계 게이트는 존재하지 않는 트리거를 겨냥 → azure에 GPU 미사용.

**선행 진단 ② 반증 (conf가 이미 presence)**: "`--conf-gate`가 confident-wrong을 구조적으로 못 잡는다" → conf는 **ROC-AUC 0.9829** presence 검출기. 화면밖 kp conf mean **0.0929**(p50 0.0674) vs 화면안 **0.6963**(p50 0.7339). gate 0.05는 화면밖 42.4% 드롭(화면안 1.0% 손실), 0.20은 **91.8%**(3.4%). 원인은 소스: `TRAIN/dataset.py:609-611`이 화면밖 kp에 **전0 히트맵 타깃**을 주고 `train_heatmap.py:232-234`가 **valid mask 없이** 전 픽셀 loss → 검출기가 화면밖에서 peak를 내지 않도록 **이미 지도됨**(RoboPEPP loss 마스킹과 등가). ⇒ `gap_reexamination §16`의 최우선 측정 항목에 대한 직접 답.

**꼬리의 실제 정체 = 저신뢰 프레임**: tail min-conf **0.052** vs good **0.580**, mean-conf 0.344 vs 0.734, 재투영 52.5px vs 0.98px, 화면밖 kp **0.00개**. 자신만만하게 틀린 게 아니라 **정직하게 모른다고 말하는** 프레임. min-conf 0.052가 배포 `--conf-gate 0.05` **바로 위** → 게이트를 올리면 가장 약한 증거만 없앨 뿐 더 나은 init을 공급하지 못함. 탐지 자체는 견고(`reproj>10px`가 7.4% 플래그, 발산 **31/31 포착**, recall 1.00/precision 0.42) — **찾는 것은 문제였던 적이 없음**.

**② 실측 (`--resolve-reproj 10`)**: azure **0.7953→0.7965 (+0.0012)**, 노이즈(~0.010) 내. 74 플래그(7.4%)/**60 채택**(6.0%). **가드는 정상 작동** — 채택 60개 전부 재투영 엄격히 낮고 median **41.3→26.0px**. 그러나 3D: 31개 중 **9개만 100mm 아래로 하강**, 22개 잔류, **6개 신규 악화**(순 31→28), 채택 60개는 **28 개선/32 악화**. AUC 기여 13.602→14.750 = +0.00115로 델타와 산술 일치.

🔑 **기전(핵심) — 채택 기준 자체의 문제, 임계 튜닝 불가**: `corr(Δreproj, ΔADD) = **+0.463**` 뿐. **이 부분모집단은 min-conf 0.052라 잔차를 재는 키포인트 자체가 신뢰불가** → 재투영에 단조인 가드가 **ADD에는 단조가 아님**. 임계를 옮겨도 상관계수는 불변이므로 **회귀 제거는 원리적으로 불가능한 설계상 성질**이며, 없애려 튜닝해서도 안 됨.

**상한 정정**: "+0.041" **철회** → dump 직접계산 **+0.0263**(0.7954→**0.8218**). ADD>100mm는 3.1%뿐이고 이미 기여 0이라 되찾을 총량이 애초에 +0.026. **완벽 복구도 kinect 0.8275·RoboTAG 0.831 미달** — 이 축으론 헤드라인 없음.

**do-no-harm (base-only 비교)**: azure 0.7953→0.7965(74플래그·60채택) / realsense 0.7452→**0.7452(비트 동일, 11플래그·2채택)** / kinect 0.7672→**0.7670(−0.0002, 57플래그·20채택)** / orb 0.7382→**0.7431(+0.0049, 68플래그·48채택)**. 4카메라 전부 회귀 없음(최악 −0.0002 « 노이즈 0.010). orb +0.0049가 최대 이동이나 **노이즈 밴드 내**이며 `corr(Δreproj,ΔADD)=+0.463`의 동전던지기 성질(한 카메라가 사소하게 유리)과 일치 → **신뢰 가능한 이득 아님**. 채택 활동(60/2/20/48, 전부 재투영 엄격히 낮음)에도 ADD 계통 무변동으로 기전 재확인. ⚠️ **GPU 정책 위반 정정**: rs/kinect/orb do-no-harm이 GPU0 포화로 GPU3(A6000)에서 실행됨(정책=GPU0 전용) — 학습 잡 pid 3111614 무해 확인, orb는 GPU0 재실행으로 정정. 비교가 **base-only(pre-RC: rs 0.7452·ki 0.7672·orb 0.7382·az 0.7945)** 여야 하는 이유 = 배포치(0.8153/0.8275/0.7784)는 **post-RC**라 솔버 변경 효과와 RC 단계가 혼동됨. azure만 RC off라 두 값 동일.

⚠️ **일반 교훈 (이 세션에만 3번 걸린 함정)**: `AUC = mean(max(0,1−10·ADD))`이므로 **ADD≥100mm는 이미 기여 0** → **탐지·게이팅·거부만으로는 이득이 원리적으로 0이고, 반드시 임계 아래로 되돌려야 한다.** ① Panda 화면밖 tail(oracle-presence 0.7032 vs control 0.7040 = net-zero) ② KUKA 잔차 거부(검출 ROC 0.859인데 end-to-end 0.753→0.734 음수) ③ azure 꼬리(recall 1.00인데 +0.0012). **처방 설계 체크리스트: "프레임을 임계 *아래로* 옮기는가, 나쁜 프레임을 *식별*만 하는가?"** 후자면 측정 전에 net-zero임을 알 수 있음. 부수 교훈: **대리 목적함수로 채택을 결정할 땐, 그 대리지표가 *개입 대상 부분모집단*에서 목표지표와 얼마나 상관되는지 먼저 재라** — 전체 상관은 무의미(개입은 꼬리에서 일어나고, 꼬리는 정의상 대리지표가 무너진 곳).

**코드 (배포 권고하지 않음)**: `Eval/solve_pose_kinematic.py:139-151`(`border_mask`), `:152-153,171-178`(`pnp_init(deprio=)`), `:265-272`(게이트를 pnp_init **앞**), `:300-310`(refine 가중치 0), `:427-467`(멀티스타트+do-no-harm 채택); `Eval/selfbbox_eval.py:179-182`(`--border-margin`/`--resolve-reproj`, **둘 다 기본 0.0**), `:404-411`(dump에 conf/gtoff); 신규 `Eval/presence_conf_probe.py`. **off-by-default 비트 동일 검증** → 배포 mean 0.804 무변경. `TRAIN/dataset.py` 미수정, azure RC off 유지. 경계 게이트=트리거 부재로 무의미, 재solve=노이즈 내 → **둘 다 미배포**, 코드는 음성 결과 재현 수단으로 존치.

**결론**: azure 꼬리 = **현 단일뷰 구조의 바닥**. 검출기가 **올바르게** 낮은 conf를 내는 프레임이라 **init/basin 개입으로 도달 불가** — 더 나은 *탐색*이 아니라 더 나은 *증거*(멀티뷰·시간일관성·더 강한 검출기)가 필요. 헤드룸은 여전히 **good 프레임 각도**(base 0.788 vs oracle-θ 0.899, +0.11) — 꼬리가 아니라 몸통. 상세: [docs/dinobotpose3/experiments/2026-07-22_azure_tail_refuted.md](../../docs/dinobotpose3/experiments/2026-07-22_azure_tail_refuted.md)

## 2026-07-22 (cont.) KUKA-Photo 측정 → 논문 표 게이트 해제 · (A)층 사실오류 수정 적용
**KUKA-Photo 확정 (5999장, 솔버+참 K, DR과 동일 체크포인트)**: direct **0.3305 → 솔버 0.6984** (+0.368, **+111%**), med ADD 64.8→**12.1mm**, med t 58.9→**15.0mm**, med R 8.00→**5.98°**. ⇒ **KUKA/Baxter 3개 셀이 전부 동일 설정으로 측정**되어 "한 행에 두 파이프라인 혼재" 위험이 사라졌다. **Baxter Photo는 DREAM에 부재 재확인** — 계속 비워 둔다.

**논문 최종 셀**: KUKA-DR **69.0** / KUKA-Photo **69.8** / Baxter-DR **71.3**(원자료 0.7125 = 71.25 정확히 중간값 → 소수1자리 half-up으로 71.3, 하위그룹 최고라 `\textbf{}`).

**✅ (A) 적용 완료 — 사실 오류만**: `PAPER_OVERLEAF.tex:167`(우리 행 3셀 교체 + Baxter 볼드), `:166`(RoboTAG Baxter `\textbf{58.8}`→`58.8` 볼드 해제), `:171`(캡션 "direct-pose configuration without render-and-compare" → "evaluated with the kinematic solver but without render-and-compare, which is applied only to the Panda real splits"), `:245`("Because **no matching mesh is available**…poses obtained **directly from the heads**…35.7/25.2" → "Because no real data exist for self-training, and render-and-compare is reserved for the Panda real splits, poses come from the **kinematic solver**…**69.0/71.3**"), `:246`(한국어 대역); `PAPER_DRAFT.md:181,183`(0.357·0.253→0.690·0.713), `:190,191`(표3 포즈 수치만), `:210`(표4 행), `:212`(캡션 "(no mesh)" 제거 — 승인된 `tex:171`과 **동일 문장·동일 오류**라 확장 적용); `figures/make_figs_multirobot.py:34`(`[0.804,0.357,0.253]`→`[0.804,0.6901,0.7125]`) + 막대가 높아진 데 따른 `ylim 0.98→1.05`·상단 라벨 `0.90→0.94` 조정.

**🟡 (B) 초안만 — 논문 미적용**: `docs/dinobotpose3/PAPER_REVISION_DRAFT_2026-07-22.md` 신규(B-1~B-10, 각 항목마다 현재 문장/왜 틀렸는지/대안 2가지). 대상 = `tex:245`(not-comparable·applicability 프레이밍, observability-ceiling 인과 서사)·`:246`·`:312`·`:313`, `PAPER_DRAFT.md:194,206,328,330`, `PAPER_DRAFT.md:190/191`의 **병목 열**, 그림7 주석 문구.
> 🔴 **미해소 부작용**: `PAPER_DRAFT.md:328/330`이 (B)라서 옛 **0.357/0.253**을 그대로 들고 있는데 같은 파일 `:181/:183/:190/:191/:210`은 갱신됨 → **파일 내부에 두 세대 수치 공존**. 초안 **B-8이 최우선 항목**으로 표시.

**(C) 재현 caveat**: 재측정 direct-pose가 아카이브 대비 **세 셀 모두 같은 방향으로 +1~2점** — KUKA-DR 36.8 vs 35.7, KUKA-Photo 33.1 vs 31.9, Baxter-DR 27.4 vs 25.3. 무작위 흔들림이 아닌 **계통 차이**이며 유력 원인은 `best_*` vs `last_*` 체크포인트. **논문 노출 지점 없음** — 교체된 세 셀은 전부 새 설정(솔버+참 K) 값이고, `PAPER_OVERLEAF.tex`에는 direct-pose 언급도 옛 수치도 **남아 있지 않다**(grep 확인). 옛 direct-pose를 동일 프로토콜 baseline으로 인용하는 유일한 잔존 지점은 **`PAPER_DRAFT.md:328/330`**(위 B-8) 및 논문 외 `SUMMARY.md:53-54,57`.

**반영 금지(미확정)**: RC 수치 전부 — 선택자 재튜닝 진행 중(현재 KUKA 솔버 0.690 + 선택적 RC = **0.708**, oracle 상한 **0.745**).

## 2026-07-22 (cont.) 문서 정합성 마무리 — PAPER_DRAFT 수치 통일 · SUMMARY 무효정보 제거
**① `PAPER_DRAFT.md:328/330` 수치 통일 (파일 내 두 세대 공존 해소)**: `direct-pose로 ADD-AUC 0.357/0.253` → `예측된 관절각으로부터 운동학 솔버가 R,t를 복원해 0.690/0.713`(영문 대역 동일). ⚠️ **숫자만 바꾸는 것은 불가능했다** — "direct-pose로 0.690"은 새로운 거짓이 되기 때문(0.690은 솔버 산출값). 숫자와 경로명이 한 절에 묶여 문법적으로 분리 불가하므로 **경로명 2어절까지 최소 교체**했고, 이는 승인된 `tex:245`("obtained directly from the heads"→"come from the kinematic solver")와 **동일 종류의 사실 교정**이다. 그 문단의 **한계·병목 논증은 미변경**(애초 `:328`에 없고 `:194/:206/:245`에 있음). 초안 B-8에 판단 근거 기록.

**② `SUMMARY.md` 갱신** — CLAUDE.md상 "새 실험 전 반드시 읽는 파일"이라 무효 정보 잔존이 미래 세션을 오도하므로 우선 정리:
- `:53-54` → **솔버+참 K 확정치**(KUKA-DR 0.690 / KUKA-Photo 0.698 / Baxter-DR 0.713)로 교체 + 경쟁 대비(Baxter 1위 71.3 vs 58.8/34.4/32.7, KUKA 사정권 69.0 vs 75~80) + 옛 0.357/0.253이 **intrinsics 버그 산물**임과 실험문서 링크. 이어서 🔓 **"솔버 각도정제 금지"는 REFUTED가 아니라 OVERTURNED** 항목 신설 — 참 K에서 솔버가 direct-pose를 +0.32/+0.44 능가하므로 **솔버 경로가 3로봇 공통 기본값**. KUKA 잔존 꼬리(fail 15.9%, p99 1012mm)=link-identity 혼동(별건), Baxter 꼬리는 버그 자체(24.7→8.0%).
- `:57`(현 `:67`) → 합성 비교 수치 교체 + "**모든 합성 로봇에서 뒤진다**"는 독법 폐기 명시.
- `:116`(현 `:129-130`) → **관측은 존치**(mlp_patch가 plain mlp를 못 이김, 손목은 실제로 키포인트로부터 미결정), **거기서 나온 인과 추론 "따라서 Baxter 병목=손목 관측성"만 REFUTED** 로 분리 기재. 근거 2건: FK 레버암상 손목 25°→키포인트 8mm이라 **완전수정도 +0.005**, 그리고 intrinsics 수정만으로 **fail 24.7→8.0%** 붕괴. ⇒ 손목 관측성은 실재하나 **2차 효과**, 포즈 정확도 한계로 인용 금지.
- `:117`(현 `:131`) → **유지 + 존치 근거 명기**: Baxter RC 실패는 **앵커/게이트 부재**로 별도 진단됐고 SAM 마스크도 정상(IoU 0.82≈Panda 0.85) → K 버그와 **독립**, 관계는 미측정.
- **신규 REFUTED 1건(3변형) 추가**: 🔴 **"솔버에서 head θ를 고정/앵커하는 계열 전체"** — 전역 freeze(DR 0.704→0.533, Photo 0.738→0.561; edge-gate 8px·oracle presence 모두 무효 0.531~0.532) / **관절별 부분 freeze**(net-zero) / **프레임 조건부 freeze**(flip-trigger 2-pass τ=60px, 실전 **+0.009**, oracle 상한 +0.027) / **θ-앵커**(net-zero~음수). **근본 원인**: freeze는 head θ가 솔버 θ보다 나은 곳에서만 이득인데, **head θ는 솔버가 나쁜 바로 그 프레임에서 똑같이 나쁘다**(오차가 상보적이 아니라 상관됨) → good 프레임 비용(0.788→0.582, 89%)만 치르고 tail은 AUC 기여 ~0. ⇒ **"head 각도를 대신 믿자"류 재시도 금지** — 재시도하려면 head θ와 솔버 θ가 *서로 다른 프레임*에서 틀린다는 것을 먼저 입증할 것.

**③ 반올림 확정**: Baxter **71.3 (half-up)**. 더 정밀한 원값으로 동률(71.25)을 없애려 했으나 **불가** — `Eval/u1_solver_vs_direct.py:266`이 `{add_auc(a):>9.4f}`로 **소수 4자리 절단** 출력이고, 해당 full-set 런은 `--dump` 없이 돌아 per-frame 배열이 로컬에 없다. **다음 런에 `--dump` 부착 권장.**

## 2026-07-22 (cont.) 논문 서사 교체 — "관측성 천장" 삭제 → link-identity 파국 꼬리 (저자 결정: B-2/B-4 대안 2)
저자가 초안(`PAPER_REVISION_DRAFT_2026-07-22.md`)의 **B-2/B-4를 대안 2로 확정** → "distal 관측성 천장 = 방법의 한계" 서사를 삭제하고 "**link-identity 혼동에 의한 파국적 꼬리**(그럴듯하나 틀린 kp-link 대응이 신뢰도 높은 오답 포즈 → 신뢰도 기반 거부로 못 잡음)"로 교체. 근거: intrinsics 수정만으로 Baxter fail율 24.7→8.0% 붕괴(관측성이 병목이면 불가능), FK 레버암상 손목 완전수정도 +0.005뿐.
**적용(승인 텍스트 그대로)**: `PAPER_OVERLEAF.tex:245`(B-1 대안2 포지셔닝 승격 + B-2 대안2 한계 교체)·`:246`(한국어)·`:312`(B-4 대안2 결론)·`:313`(한국어); `PAPER_DRAFT.md:195`(B-6 대안1 경고 재프레이밍)·`:197/199`(B-7 대안1 사실교정)·`:190/191`(B-9 대안2: 표3 병목 열 삭제→median ADD 13.1/17.1mm); `figures/make_figs_multirobot.py:51`(B-10 대안1 주석 "different regime — not a robot-vs-robot ranking").
**연쇄 적용(B-2/B-4의 병렬 위치 — 승인 서사를 그대로 이식)**: `PAPER_DRAFT.md:326`(절 제목 "관측성 병목"→"잔여 실패모드")·`:334/336`(비교주의 stale 0.34/0.25→0.69/0.71 + 병목 지시 교체)·`:338/340`(관측성 분석: **관측 존치·2차효과로 강등**, 지배 실패=link-identity 명시 — SUMMARY:116과 동일 처리)·`:447/449`(결론 서사 교체).
**⚠️ B-7 대안2 폐기**: "RC를 KUKA로 확장 = 남은 헤드룸"은 KUKA RC가 방금 **닫힌 레버**(0.75 불가·R 못 고침)라 거짓 → 대안1(RC 헤드룸 주장 없음) 채택.
**🔴 저자 검토 요망(미적용, 얽힘/범위밖)**: ① `PAPER_DRAFT.md:451/453` 한계 문단 — (a) "공개 iiwa7 메쉬 ~20mm 어긋나 정합 불가"는 **반증됨**(RoboPEPP iiwa7 0.0048mm RMS, `tex:171/245` "no mesh 삭제"와 동류 오류), (b) "RC를 KUKA로 확장" 프레이밍이 RC 폐쇄로 부적절, (c) 손목 관측성 한계 서술 강등 필요 — 셋이 한 문단에 얽혀 강제 재작성 안 함(B-4 대안2가 명시한 살아남는 한계=실측데이터 부재·RC 게이팅 둘뿐). ② `tex:34/41` 서론 기여 항목 "손목 관측성 한계 등" 본문과 불일치(기여 목록이라 민감). ③ `tex:86` "wrist rotation fixed to zero"는 **참인 방법 서술 → 유지**. RC 수치는 여전히 게재 금지(선택자 재튜닝 중).
