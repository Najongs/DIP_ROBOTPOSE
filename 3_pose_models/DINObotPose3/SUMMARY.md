# DINObotPose3 — Consolidated Summary (2026-07-03)

**Goal:** single-image robot (Panda) pose + joint-angle estimator that beats **RoboPEPP** on the 6 DREAM-Panda splits (metric: ADD-AUC@100mm).

## 🏆 2026-07-06 — mean 0.804 vs RoboPEPP 0.780 (🔒 1000-frame re-lock, confirmed SOTA); ALL-4-CAMS occlusion-robust

The 06-09 renderer blocker was broken with **nvdiffrast** (exact visual-mesh differentiable silhouette,
local build on the NAS box) + **SAM ViT-B** true-robot masks (init-render-consistent mask selection).
Deployable render-and-compare (`Eval/rc_refine_from_dump.py`) on top of the crop+rot-adapt pipeline,
+ **cov-PnP** (heatmap-covariance Mahalanobis) + **DARK sub-pixel decode** (both free, no training):
| cam | re-lock 1000 | 800 (prev) | RoboPEPP | gap |
|---|---|---|---|---|
| realsense | **0.8153** (light-stack, robust) | 0.8165 | 0.805 | **+0.010 BEAT** |
| kinect360 | **0.8275** (stack, robust) | 0.8303 | 0.785 | **+0.043 BEAT** |
| azure | **0.7945** (light head, robust, RC OFF) | 0.7953 | 0.753 | **+0.042 BEAT** |
| orb | **0.7784** (stack, robust) | 0.7726 | 0.775 | **+0.003 BEAT** |
| **MEAN** | **0.8039** | 0.8037 | 0.780 | **+0.024** |

**2026-07-06 full-split re-lock (held-out 800→1000, anti-leak preserved):** mean holds at 0.804 (Δ+0.0002,
robust to sample size), per-cam ≤0.006 drift — **all 4 cams now beat RoboPEPP** (orb flipped −0.002→+0.003).
Confirmed SOTA. Reproduction `--max-frames 1000` in `docs/dinobotpose3/FINAL_MODEL.md`.
**ALL 4 cameras occlusion-robust** (40% occlusion pose 0.39-0.43 all > RoboPEPP 0.351) at NO accuracy cost —
azure light head (+0.004) and kinect stack (+0.017) offset the realsense −0.005 robustness trade. Occlusion
robustness must be baked in by training the angle head from scratch with occlusion-aug (light); short
self-train + synth occ-aug on an already-adapted head does NOT instill it (realsense robust-stack, 40% base-level).
occ-aug→self-train stack (2026-07-05): light occlusion-aug head → per-camera self-train with occlusion-aug
on the synth anti-forget batch — recovers real adaptation WHILE retaining occlusion robustness. kinect +0.017,
orb +0.001, both now occlusion-robust (40% 0.39 > RoboPEPP 0.351); realsense keeps its already-optimal head.
Protocol: predicted angles + fully-automatic bbox (stricter than RoboPEPP's GT-bbox headline); rs/kinect/orb
anti-leak held-out 1000/cam (re-locked 2026-07-06). DARK decode (`--dark-decode`, `Eval/decode_util.py`) lifts pose 2D precision
universally (+0.0035–0.017 per cam, free) — closed the orb gap −0.010→−0.004. RC = depth/scale corrector →
per-camera on/off (helps far cams; azure RC OFF). Details EXPERIMENTS.md 2026-07-04; survey
`docs/robot_pose_sota_survey.md`; roadmap `docs/robot_pose_next_directions.md`. Remaining: orb −0.004.

**Occlusion robustness (RoboPEPP Fig.6 protocol):** with the occlusion-aug head (+DARK+cov+RC) ours =
**0.812/0.765/0.678/0.575/0.429** at 0-40% RoI occlusion vs RoboPEPP 0.795/0.730/0.600/0.470/0.351 —
**BEAT AT EVERY LEVEL** (+0.017 to +0.105). The occ-aug head (2026-07-04) is do-no-harm on clean
(synth +0.002 / real azure +0.002) and flipped the two points we used to lose (0% & 40%). cov-PnP (heatmap-covariance Mahalanobis, `--cov-pnp`) ADOPTED (do-no-harm, +0.011@20%). REFUTED:
occl-robust silhouette downweighting (depth bias) and population-mean adaptive prior (fights the true
config; learned state prior skipped — synth joints independent). Bench: `Eval/occlusion_bench.sh`.

## Pipeline (deployable, oracle-free)
```
image → [self-bbox: full-frame DINOv3 detector + kinematic solve → project FK 7 kp → crop bbox]
      → roi_align crop (K-adjusted)
      → DINOv3 crop detector (heatmaps) → soft-argmax 2D + conf
      → crop angle head (θ init) + crop rot head (R init)
      → kinematic solver (PnP init + reprojection gradient refine, conf-gate 0.05)
      → joint angles θ + camera pose (R,t)
```
Backbone = **frozen** DINOv3-ViT-B/16 (synth-pretrained). All real adaptation is in the heads via solver pseudo-labels — the backbone is deliberately NOT adapted (see Refuted).

## FINAL deployable result vs RoboPEPP @1000 frames/split (best-per-camera config)
| cam | ours | config | RoboPEPP | gap |
|---|---|---|---|---|
| **azure** | **0.788** | crop (guard) | 0.753 | **+0.035 ✅ BEAT** |
| kinect | 0.776 | self-det + self-angle (non-crop) | 0.785 | −0.009 ≈MATCH |
| realsense | 0.745 | crop+selftrain + bbox-refine-2 | 0.805 | −0.060 |
| orb | 0.725 | crop+selftrain r2 | 0.775 | −0.050 |
| **MEAN** | **0.759** | | 0.780 | **−0.021** |

**BEAT RoboPEPP on azure, MATCH on kinect.** realsense/orb remain the structural gap (far/foreshortened — RoboPEPP's iterative render-and-compare edge). Session moved the deployable mean from 0.711 → 0.759 (+0.048).

## 2026-06-29 — ✅ ROT-HEAD self-train DEPLOYED (realsense + orb)
Pseudo-label adaptation extended from the angle head to the **ROTATION head** — the diagnosed R/gauge lever that prior self-train left frozen. `TRAIN/selftrain_pseudo_rot.py` distills the solver's refined R\* (`solve_batch(..., return_pose=True)`) into the rot head (chordal loss) alongside the angle head, per camera, crop pipeline, held-out early-stop.
**Deploy-validated** on the oracle-free self-bbox pipeline, anti-leak held-out (`selfbbox_eval.py --bbox-from-solved --bbox-guard --frac-range 0.7 1.0`, 800 frames), vs the previously-deployed angle-only-selftrain head on **identical frames**:
| cam | deployed (angle-selftrain) | ROT-ADAPT | Δ |
|---|---|---|---|
| **realsense** | 0.745 | **0.755** | **+0.011** |
| **orb** | 0.709 | **0.715** | **+0.007** |
| kinect (crop) | 0.749 | 0.756 | +0.007 |
do-no-harm on every adapted camera; realsense deployed 0.745 == the locked @1000 number → protocol verified.
**DEPLOYED:** realsense + orb now use the per-camera rot-adapt (angle+rot) head **pairs** (see checkpoints). kinect **KEEPS** its non-crop self-det config (0.776 > crop-rot-adapt 0.756); azure unchanged. realsense may still stack bbox-refine-2 (orthogonal, untested with rot-adapt).
**HONEST CAVEAT:** the larger GT-crop held-out gains (+0.024 rs / +0.046 orb / +0.016 kinect) were **oracle-bbox-optimistic**; the real deployable self-bbox gain is the +0.006–0.011 above. We still trail RoboPEPP (~0.03) — the realsense/orb residual is the foreshortened-tail DEPTH limit, proven **single-view-unreachable** this session (silhouette depth-select / render-compare refiner [Phase D] / learned depth-scale all REFUTED). See memory `rot-head-selftrain-wins`.

## What WORKED (the levers that built 0.759)
1. **Kinematic solver** (PnP init from top-4 confident kp + reprojection gradient refine) — solved far-camera collapse (realsense 0.32→0.52).
2. **Train+test crop** productionized oracle-free via **bbox-from-solved** (project all 7 FK keypoints incl. occluded base → crop bbox). +0.04–0.15 per camera.
3. **Rotation head** (learned R_init from appearance) — fixed realsense basin (+0.117).
4. **Per-camera pseudo-label self-training** (distill solver angles on high-conf real frames) — gain ∝ sim2real gap (realsense +0.137, orb +0.057, kinect +0.023, azure ~0).
5. **Detector self-train** (distill solver-reprojected keypoints into the 2D head) — kinect 0.756→0.776 (=RoboPEPP, non-crop).
6. **bbox-refine-2** (coarse-to-fine bbox via crop detector) — realsense 0.737→0.745 (realsense-specific).
7. **Rotation-head pseudo-label self-train** (2026-06-29, `selftrain_pseudo_rot.py`) — adapt the ROT head (not just angle) toward the solver's refined R\* on real; deployable self-bbox +0.011 rs / +0.007 orb / +0.007 kinect over the angle-only-selftrain head. The diagnosed R/gauge bottleneck, the only deployable realization of the gauge headroom (depth headroom stays single-view-unreachable).

## What was REFUTED (negative results, decisively)
- **union-bbox** (solved∪detected): −0.002 — self-bbox already encloses kp; the −0.04 vs GT-crop is oracle-zoom, not bbox defect.
- **iterative crop-selftrain r2**: realsense plateau (+0.000) — head lever saturated.
- **depth / t_z translation prior**: sim2real 587mm — rejected (solver's PnP t is better).
- **MCL multi-hypothesis**: oracle ceiling 0.716 < self-train 0.724 — not worth a selector.
- **🔴 BACKBONE ADAPTATION (entire family) — REFUTED 3 ways:**
  - SSL masked-feature (aggressive 6-block): real PCK ↑ (realsense @5 +0.069) but realsense ADD ↓ (0.531 < 0.567).
  - SSL gentle (3-block): even mild adaptation makes heads fully OOD (ADD 0.0) → 4h cascade to test.
  - pseudo-keypoint co-finetune: realsense ADD monotone ↓ (0.497→0.434).
  - **Root cause:** the kinematic solver needs sub-pixel keypoint PRECISION; adapting the backbone to real trades precision for coarse robustness → net-negative ADD. **Frozen synth-pretrained backbone is optimal.** (Also why the DINOv3Backbone `.layer` unfreeze-bug was harmless — frozen was best all along.) Even "real PCK does not predict real ADD."

## WHY realsense/orb fail (diagnosed 06-08, `Eval/realsense_failure_diag.py`)
Per-frame ADD decomposition: the residual failures are NOT the joint angles (θ recovered well) — they are the
**camera-to-robot POSE, specifically DEPTH/translation**, on foreshortened + low-conf frames (2D bearings
under-constrain depth = monocular scale ambiguity; R/t are gauge-coupled). **Gauge-safe depth ceiling**
(`Eval/depth_ceiling_probe.py`, GT root-depth anchor + re-solve): realsense **+0.116** (0.692→0.808, would
beat RoboPEPP). So depth/scale is the missing constraint. A scalar depth head is DEAD (fragile: 10% noise →
net-negative; the old t_z prior failure was the same).

## RENDER-AND-COMPARE — VALIDATED lever, blocked on the mask (the frontier, 06-09)
Pure-PyTorch mesh-silhouette render-compare (`Eval/silhouette_mesh_probe.py`, NO pytorch3d): render the Panda
collision-mesh silhouette (area ∝ 1/z² encodes depth), refine (θ,R,t) to match a target mask via soft-IoU +
reprojection anchor. **Oracle (self-consistent) ceiling = +0.108 @224px** (realsense 0.691→0.799 ≈ depth
ceiling, would reach RoboPEPP). Robust to mask DEGRADATION (+0.083 even heavily degraded). **This is the
validated path to close realsense/orb.**
- **BLOCKER = a correctly-placed, render-consistent REAL mask.** Render-compare only works when the target
  mask comes from the SAME render model:
  - learned DINOv3 mask head (`TRAIN/train_mask.py` synth + `TRAIN/selftrain_mask.py` real self-train): render-
    consistent but imperfect (IoU 0.52 vs oracle) → deployable only **+0.01**.
  - SAM ViT-B (installed, `/data/public/97_cache/sam/sam_vit_b_01ec64.pth`, wired `--sam-checkpoint`): accurate
    but shape-INCONSISTENT with the approximate splat render → **diverges**. Prompt-finicky on the cluttered
    scene (loose=over-segment, tight+neg=collapse). Visualized in `ViS/rc_viz/`.
- Refuted sub-ideas: visual meshes for the renderer (collision SOLID mesh gives the cleaner area/depth cue:
  oracle +0.108 vs visual +0.055); mesh-shrink toward true thickness (IoU barely moves).
- **To resume:** need a HIGH-FIDELITY differentiable renderer (exact visual mesh + real rasterizer = PyTorch3D
  / nvdiffrast — won't pip on torch 2.10/cu128; a CUDA build) OR a clean render-consistent real mask
  (GroundingDINO "robot" box → SAM2, or a better robot segmenter, or push the learned mask head past IoU 0.52).
  ALL pieces except the renderer/clean-mask are built (renderer, mask head + self-train, SAM, depth probes).

## Status
- **Consolidated at deployable mean 0.759** (BEAT azure, MATCH kinect). All cheap levers + the entire
  backbone-adaptation family exhausted/refuted. render-compare validated (+0.108) but not deployable with
  current tooling (mask/renderer).
- Infra: use **only GPU `GPU-05f84104-40d4-c675-91bf-5427bc0fd5e9`**; `ab38c04c` (2F:00.0) is flaky (node-wide
  CUDA wedge under load). The shared GPU also gets grabbed by an external `ollama runner` (~21GB) — check free
  mem first. Always kill jobs by EXACT PID (shell-wrapper kills leave orphaned python holding the GPU).

## Key files / checkpoints
- Eval: `Eval/ab_eval.sh` (6-split), `Eval/selfbbox_eval.py` (`--bbox-from-solved --bbox-guard [--bbox-refine-iters N]`), `Eval/pck_eval.py`.
- Train: `TRAIN/selftrain_pseudo.py` (angle self-train, `--crop`), `TRAIN/selftrain_detector.py` (detector self-train, `--unfreeze-backbone` co-finetune [refuted]), `TRAIN/ssl_masked_dinov3.py` (SSL [refuted]).
- Detectors: stage1 `outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth`; crop `outputs_heatmap/crop_20260605_010622/best_heatmap.pth`. Heads: full-frame rot `outputs_rotation/rot_20260604_162336`; crop rot `outputs_rotation/rot_crop_20260606_022535`; kinect self-det `outputs_selftrain_det/panda-3cam_kinect360_r1/merged_detector.pth`.
- **🔒 DEPLOYED per-camera heads (2026-06-29):** realsense + orb use the ROT-ADAPT pairs `outputs_selftrain/{realsense,orb}_rot_r1/best_selftrain_head.pth` (crop angle) **+** `outputs_selftrain/{realsense,orb}_rot_r1/best_selftrain_rot.pth` (crop rot) — pass BOTH (`--crop-angle <head> --rot-head <rot>`). kinect = non-crop self-det (unchanged); azure = synth crop angle + `rot_crop_20260606_022535` (unchanged). SUPERSEDES the old per-cam angle-only `outputs_selftrain/{cam}_crop_r{1,2}/best_selftrain_head.pth` for realsense/orb. Eval flag added: `selfbbox_eval.py --frac-range LO HI` (anti-leak held-out). Train: `TRAIN/selftrain_pseudo_rot.py`. Env: `dino` env was replaced by `py312` (torch2.10 + albumentations) this session.
- Render-compare frontier: `Eval/silhouette_mesh_probe.py` (renderer + refine + `--mask-head`/`--sam-checkpoint`/`--mask-degrade`/`--bbox-refine-iters`), `Eval/depth_ceiling_probe.py`, `Eval/realsense_failure_diag.py` (ADD decomposition + oracle ceilings), `Eval/rc_viz.py` (mask viz → `ViS/rc_viz/`). `TRAIN/train_mask.py` + `TRAIN/selftrain_mask.py` (mask head + real self-train); heads in `TRAIN/outputs_mask/`. Meshes `ViS/Panda/meshes/{collision,visual}/`. SAM weights `/data/public/97_cache/sam/sam_vit_b_01ec64.pth`.
- Full experiment log: `EXPERIMENTS.md`. Strategy memories: `robopepp-target-numbers`, `ssl-backbone-refuted`, `hpe-horopose-analysis`, `render-compare-validated`, `realsense-failures-are-foreshortened-j0`, `pseudo-label-selftrain-works`, `gpu-env-notes`.

## HOW TO RESUME (next session, in priority order)
1. **Clean render-consistent mask** (highest EV, no heavy install): improve the mask via GroundingDINO("robot arm")→SAM2 box prompt, or train a better robot segmenter, or push `selftrain_mask.py` past IoU 0.52 (iterate noisy-student rounds; the foreshortened-frame circularity is the limit). Then re-run `silhouette_mesh_probe.py --mask-head <new>` (or `--sam-checkpoint` once masks are clean) → target realsense 0.745→~0.80.
2. **High-fidelity renderer**: if PyTorch3D/nvdiffrast can be built (needs a torch/cu combo with prebuilt wheels — torch 2.10/cu128 has none), exact-mesh render + SAM true mask realizes the +0.108 directly.
3. Either path: validate on the locked 6-split `ab_eval.sh` @1000 before claiming. The render-compare gain is currently proven only at the oracle/degraded-mask level.
