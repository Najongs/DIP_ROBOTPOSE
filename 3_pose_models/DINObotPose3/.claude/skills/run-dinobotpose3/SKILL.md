---
name: run-dinobotpose3
description: Run, evaluate, and visualize the DINObotPose3 robot pose pipeline. Use when asked to run DINObotPose3, reproduce its DREAM SOTA numbers (mean ADD-AUC 0.804), evaluate a camera split, render a mesh-overlay screenshot, sanity-check the solver/model internals after editing TRAIN/, or check which GPU to use.
---

DINObotPose3 estimates Panda joint angles + camera pose from a single image
(frozen DINOv3 detector → angle/rot heads → kinematic solver → nvdiffrast+SAM
render-and-compare). There is no server and no UI — the "app" is a two-stage
eval, and you drive it with **`.claude/skills/run-dinobotpose3/driver.py`**.

All paths below are relative to `3_pose_models/DINObotPose3/`.

The driver exists because the deployed configuration is *not* reconstructible
from the scripts in `Eval/`: the per-camera checkpoint table lives only in
`docs/dinobotpose3/FINAL_MODEL.md`, `Eval/verify_sota.sh` still encodes the
**older 0.799** config, and picking a GPU by integer index silently lands you
on the wrong card. The driver hardcodes the correct table and picks GPUs by UUID.

## Prerequisites

Nothing to install — this is a configured research box. Use the `dino` conda env
directly by path (do **not** rely on `conda activate`; see Gotchas):

```bash
/home/najo/.conda/envs/dino/bin/python --version   # Python 3.10.19
```

It already has torch 2.10.0+cu128, nvdiffrast, segment_anything, transformers 5.2.0,
opencv, trimesh. Verify everything in one shot:

```bash
/home/najo/.conda/envs/dino/bin/python .claude/skills/run-dinobotpose3/driver.py doctor
```

That checks deps, lists GPUs with free memory (by UUID), and confirms all 13
deployed checkpoints and all 4 DREAM real splits are present. It exits non-zero
if anything is missing. Expect `DOCTOR: all green`.

## Run (agent path)

```bash
cd 3_pose_models/DINObotPose3
P=/home/najo/.conda/envs/dino/bin/python
D=.claude/skills/run-dinobotpose3/driver.py

$P $D doctor                      # env / GPU / checkpoint / dataset check
$P $D smoke                       # fast end-to-end, 20 kinect frames  (~80 s)
$P $D fwd                         # model internals only, no eval loop  (~9 s)
$P $D eval --cam azure --frames 20
$P $D viz  --cam kinect           # mesh-overlay PNG                   (~41 s)
$P $D sota                        # all 4 cams x 1000 frames  (LONG, tens of min)
```

Every command takes `--gpu GPU-<uuid>` to pin a card (default: most free;
`DINOBOT_GPU` env var also works). Outputs — dumps, per-stage logs, PNGs —
land in `Eval/driver_out/` (gitignored).

**Start with `smoke`.** It runs the real deployed pipeline end-to-end (base pose
→ render-compare) and prints measured-vs-deployed side by side:

```
-- base pose [kinect] 20 frames
     [Pose] ADD-AUC@100mm: 0.8210 | mean ADD 17.9mm | median 15.4mm (20 frames)
   [32s, exit 0]
-- render-compare [kinect] @448
     SAM-vs-init-render IoU: mean 0.844  median 0.847  frac>=0.5: 1.00
     + nvdr/SAM render-compare ADD-AUC@100mm 0.8514
     Δ: +0.0304
   [49s, exit 0]

camera          base      +RC   deployed  (n=20)
kinect        0.8210   0.8514     0.8275
```

`deployed` is the locked 1000-frame number from FINAL_MODEL.md. At `--frames 20`
expect it to be *close, not equal* — 20 frames is a smoke test, not a
reproduction. Only `sota` (1000 frames) should reproduce the locked values.

### Which command for which change

| You edited | Run |
|---|---|
| `TRAIN/model.py`, `model_v4.py`, `norm_utils.py`, `model_angle.py` | `fwd` first (9 s — catches shape drift and checkpoint incompatibility), then `smoke` |
| `Eval/selfbbox_eval.py`, solver, decode | `smoke`, then `eval --cam <cam> --frames 200` |
| `Eval/rc_refine_from_dump.py`, `render_nvdr.py` | `smoke` (its second stage *is* render-compare) |
| anything, before claiming a number moved | `sota` — nothing smaller is comparable to the locked table |

`fwd` imports `TRAIN/` directly and asserts a PnP round-trip recovers known
camera-frame keypoints (the A0 gate in miniature — expect `0.0001 mm`), then
loads the deployed detector into `AnglePredictor` and forwards a random batch:

```
  solve_pnp_batch -> (2, 7, 3) valid [True, True] reproj [0.0, 0.0]
  round-trip error 0.0001 mm
  detector: 261/261 tensors loaded
     joint_angles (1, 7)     keypoints_2d (1, 7, 2)   heatmaps_2d (1, 7, 512, 512)
     sin_cos (1, 6, 2)       confidence (1, 7)        global_feat (1, 768)
FWD OK
```

If `261/261` drops or the assert fires, your `TRAIN/` edit broke checkpoint
compatibility — fix that before trusting any eval number.

### The visual check

`viz` renders the predicted Panda mesh (nvdiffrast, orange) over the real photo
with the GT skeleton (green) on top, and prints per-frame `ang`/`ADD`:

```bash
$P $D viz --cam kinect --indices 50,800,1600,2400,3200,4000
# -> Eval/driver_out/viz_mesh_kinect.png
```

**Open the PNG and look at it.** The mesh should sit on the robot and hug the
green skeleton. Verified this session: 5 of 6 frames align tightly (ADD 13–19 mm);
frame #800 visibly drifts at the wrist (ADD 47 mm, ang 12.1°) — that is the
expected failure mode, not a broken run. A blank or wildly offset mesh means the
angle head or the solver is broken.

## Run (human path)

`Eval/verify_sota.sh` is the hand-written 4-camera script. **It reproduces the
older mean 0.799**, not the current 0.804 — see Gotchas. Prefer `driver.py sota`.

## Gotchas

- **Integer `CUDA_VISIBLE_DEVICES` does not match `nvidia-smi` indices on this
  box.** Verified: `nvidia-smi` lists index 0 as an RTX 3090 and index 3 as the
  A6000, but `CUDA_VISIBLE_DEVICES=0` yields the **A6000** and `=3` yields a 3090.
  Always select by UUID. The driver does this for you; if you run `Eval/` scripts
  by hand, pass `CUDA_VISIBLE_DEVICES=GPU-<uuid>`.

- **`Eval/verify_sota.sh` is stale.** Its header says `mean ADD-AUC 0.799`, it
  uses `NF=800`, and it points at the `outputs_selftrain/{cam}_rot_r1/` heads.
  The deployed 0.804 config uses the **`*_lightstack_*`** heads at **1000**
  frames. `docs/dinobotpose3/FINAL_MODEL.md` is the authority; the driver's
  `CAMS` table mirrors it. `docs/dinobotpose3/training/training.md` also lists
  the older heads in its checkpoint table — same trap.

- **azure runs with render-compare OFF.** RC is a depth/scale corrector and only
  helps far cameras; azure is near-field. It also uses different heads
  (`angle_occaug_light` + `rot_crop_occaug`, not a selftrain pair). The driver
  encodes this — `eval --cam azure` prints `RC [azure] SKIPPED` by design.

- **`--frac-range 0.7 1.0` is an anti-leak guard, not a speed knob.** rs/kinect/orb
  self-trained on the earlier part of each sequence, so evaluation must use the
  held-out tail. Dropping it inflates results. The driver always passes it.

- **`Eval/rc_viz.py` is broken on this box** — it hardcodes the GPU-server SAM
  path `/data/public/97_cache/sam/sam_vit_b_01ec64.pth`, which does not exist
  locally (the real one is `weights_sam/sam_vit_b_01ec64.pth`). Use
  `driver.py viz` (wraps `viz_mesh.py`) instead. GPU-server paths under
  `/data/public/NAS/...` are left in these scripts intentionally per CLAUDE.md —
  do not "fix" them wholesale.

- **HuggingFace floods stdout** with a per-tensor "Loading weights" progress bar
  (211 lines of `it/s` spam) on every backbone construction. The driver filters
  it; raw runs need `| grep -v 'it/s'` or the log is unreadable.

- **All 5 GPUs typically sit at 100% utilization** from other training jobs, but
  have free *memory*. That's fine — select on free memory (what the driver does),
  not utilization, or you'll conclude the box is unusable.

- **Do not `conda activate` in scripts.** `/opt/anaconda3` exists but the `dino`
  env lives in `/home/najo/.conda/envs/`. Call the interpreter by absolute path.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no GPU with >=6000 MiB free` | Every card is memory-full. Wait, or `--gpu GPU-<uuid>` to force one. `doctor` lists free memory per UUID. |
| `FileNotFoundError: /data/public/97_cache/sam/...` | You ran `Eval/rc_viz.py`. Use `driver.py viz`. |
| `torch.cuda.OutOfMemoryError` during RC | Render-compare at `--render-h 512` (orb) is the memory peak. Pick the A6000 (`GPU-05b804ff…`, ~45 GB free) or lower `--frames`. |
| `step failed: selfbbox_eval.py` | The driver prints the last 25 lines above the failure. Most often a checkpoint path — run `doctor`. |
| Numbers well below the `deployed` column | Expected at low `--frames`. Confirm with `sota` before concluding a regression. |
| `AttributeError` / missing keys from `TRAIN/` edits | Run `fwd` — it isolates model-layer breakage from eval-harness breakage. |

## Reference

- `docs/dinobotpose3/FINAL_MODEL.md` — authoritative deployed checkpoint table
- `docs/dinobotpose3/00_overview.md` — scoreboard, adopted levers, **REFUTED list**
- `SUMMARY.md` / `EXPERIMENTS.md` — confirmed conclusions / append-only log

Per CLAUDE.md: check the REFUTED list before starting a new experiment — backbone
SSL adaptation, co-finetuning, and union-bbox are already disproven.
