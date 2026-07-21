#!/usr/bin/env python3
"""
DINObotPose3 driver — launch and drive the deployed pose pipeline.

The deployed model is a 2-stage eval, not a service:
    selfbbox_eval.py  (base pose + --dump-npz)  ->  rc_refine_from_dump.py  (nvdiffrast+SAM refine)

This driver encodes the AUTHORITATIVE per-camera checkpoint table from
docs/dinobotpose3/FINAL_MODEL.md (the mean-0.804 config) so you never have to
re-derive it, auto-picks a GPU BY UUID (integer CUDA_VISIBLE_DEVICES is
scrambled on this box), and gives you one command per thing you'd want to do.

    python driver.py doctor              # env / GPU / checkpoint / dataset check
    python driver.py smoke               # fast e2e (base+RC, 20 frames) — ~1.5 min
    python driver.py eval --cam kinect   # deployed eval for one camera
    python driver.py sota                # all 4 cameras (the 0.804 reproduction)
    python driver.py viz --cam kinect    # mesh-overlay PNG (the visual check)
    python driver.py fwd                 # direct model invocation (no full eval)

Run it with the `dino` env python. Paths are resolved relative to the project
root (the DINObotPose3/ directory), so it works from any cwd.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ── layout ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]      # .../DINObotPose3
EVAL = ROOT / "Eval"
TRAIN = ROOT / "TRAIN"
DATA = ROOT / "Dataset/Converted_dataset/DREAM_real"
SAM = ROOT / "weights_sam/sam_vit_b_01ec64.pth"
PY = "/home/najo/.conda/envs/dino/bin/python"

# Shared across all cameras (FINAL_MODEL.md "공용").
S1DET = TRAIN / "outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth"
S1ANG = TRAIN / "outputs_angle/angle_20260603_013948/best_angle_head.pth"
S1ROT = TRAIN / "outputs_rotation/rot_20260604_162336/best_rot_head.pth"
CROPDET = TRAIN / "outputs_heatmap/crop_20260605_010622/best_heatmap.pth"

_ST = TRAIN / "outputs_selftrain"

# The deployed table. THIS is the 0.804 config — Eval/verify_sota.sh is STALE
# (it still points at the *_rot_r1 heads = the older 0.799 result).
CAMS = {
    "realsense": dict(
        val=DATA / "panda-3cam_realsense",
        angle=_ST / "realsense_lightstack_20260705_003546/best_selftrain_head.pth",
        rot=_ST / "realsense_lightstack_20260705_003546/best_selftrain_rot.pth",
        rc=448, expect=0.8153,
    ),
    "kinect": dict(
        val=DATA / "panda-3cam_kinect360",
        angle=_ST / "kinect_lightstack_20260705_003552/best_selftrain_head.pth",
        rot=_ST / "kinect_lightstack_20260705_003552/best_selftrain_rot.pth",
        rc=448, expect=0.8275,
    ),
    "orb": dict(
        val=DATA / "panda-orb",
        angle=_ST / "orb_lightstack_20260705_003549/best_selftrain_head.pth",
        rot=_ST / "orb_lightstack_20260705_003549/best_selftrain_rot.pth",
        rc=512, expect=0.7784,
    ),
    # azure runs RC OFF (near-field camera — RC is a depth/scale corrector and
    # only helps far cameras). Its heads are the light occ-aug pair, not selftrain.
    "azure": dict(
        val=DATA / "panda-3cam_azure",
        angle=TRAIN / "outputs_angle/angle_occaug_light_20260704_015400/best_angle_head.pth",
        rot=TRAIN / "outputs_rotation/rot_crop_occaug_20260704_002102/best_rot_head.pth",
        rc=None, expect=0.7945,
    ),
}

OUT = EVAL / "driver_out"


# ── GPU selection (by UUID — integer indices are scrambled on this box) ───────
def gpus():
    """(uuid, name, free_MiB) per GPU, ordered most-free-first."""
    q = "--query-gpu=uuid,name,memory.used,memory.total --format=csv,noheader,nounits"
    out = subprocess.run(f"nvidia-smi {q}", shell=True, capture_output=True, text=True).stdout
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        uuid, name, used, total = parts[0], parts[1], int(parts[2]), int(parts[3])
        rows.append((uuid, name, total - used))
    return sorted(rows, key=lambda r: -r[2])


def pick_gpu(explicit=None, need=6000):
    if explicit:
        return explicit
    if os.environ.get("DINOBOT_GPU"):
        return os.environ["DINOBOT_GPU"]
    for uuid, name, free in gpus():
        if free >= need:
            return uuid
    raise SystemExit(f"no GPU with >={need} MiB free; pass --gpu GPU-<uuid> to force")


def run(cmd, env_gpu, cwd=EVAL, tee=None, echo_all=False):
    """Run a pipeline step, echoing only the lines worth reading (or all of them)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=env_gpu, PYTHONUNBUFFERED="1")
    t0 = time.time()
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace")
    keep = []
    for line in p.stdout:
        # HF's weight-materialization progress bar floods stdout; drop carriage-return spam.
        if "it/s" in line or "\r" in line.rstrip("\n"):
            continue
        keep.append(line)
        if echo_all or re.search(r"ADD-AUC|IoU|Δ|saved|Error|Traceback|frames|guard", line):
            print("   " + line.rstrip())
    p.wait()
    if tee:
        Path(tee).write_text("".join(keep))
    print(f"   [{time.time() - t0:.0f}s, exit {p.returncode}]")
    if p.returncode != 0:
        print("".join(keep[-25:]))
        raise SystemExit(f"step failed: {cmd.split()[1]}")
    return "".join(keep)


def auc(text, pat=r"ADD-AUC@100mm[: ]+([0-9.]+)"):
    m = re.findall(pat, text)
    return float(m[-1]) if m else None


# ── commands ─────────────────────────────────────────────────────────────────
def cmd_doctor(a):
    ok = True
    print("== python / deps ==")
    print(f"   interpreter {PY}")
    mods = ["torch", "nvdiffrast", "segment_anything", "transformers", "cv2", "trimesh", "scipy"]
    code = ("import importlib.util as u\n"
            f"for m in {mods!r}:\n"
            "    print('   OK  ' if u.find_spec(m) else '   MISS', m)\n")
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    print((r.stdout + r.stderr).rstrip())
    ok &= (r.returncode == 0 and "MISS" not in r.stdout)

    print("== GPUs (use the UUID — integer CUDA_VISIBLE_DEVICES is scrambled here) ==")
    for uuid, name, free in gpus():
        print(f"   {uuid}  {name:24s} {free:6d} MiB free")

    print("== checkpoints ==")
    shared = [S1DET, S1ANG, S1ROT, CROPDET, SAM]
    for p in shared:
        print(f"   {'OK  ' if p.exists() else 'MISS'} {p.relative_to(ROOT)}")
        ok &= p.exists()
    for cam, c in CAMS.items():
        for k in ("angle", "rot"):
            e = c[k].exists()
            print(f"   {'OK  ' if e else 'MISS'} [{cam}/{k}] {c[k].relative_to(ROOT)}")
            ok &= e

    print("== datasets ==")
    for cam, c in CAMS.items():
        n = len(list(c["val"].iterdir())) if c["val"].exists() else 0
        print(f"   {'OK  ' if n else 'MISS'} [{cam}] {n} files  {c['val'].relative_to(ROOT)}")
        ok &= bool(n)

    print("\n" + ("DOCTOR: all green" if ok else "DOCTOR: problems above"))
    return 0 if ok else 1


def _base(cam, frames, gpu, tag):
    c = CAMS[cam]
    OUT.mkdir(exist_ok=True)
    dump = OUT / f"{tag}.npz"
    cmd = (
        f"{PY} selfbbox_eval.py"
        f" --stage1-detector {S1DET} --stage1-angle {S1ANG} --stage1-rot {S1ROT}"
        f" --crop-detector {CROPDET} --crop-angle {c['angle']} --rot-head {c['rot']}"
        f" --bbox-from-solved --bbox-guard --cov-pnp --dark-decode"
        f" --frac-range 0.7 1.0 --max-frames {frames}"
        f" --val-dir {c['val']} --dump-npz {dump}"
    )
    print(f"-- base pose [{cam}] {frames} frames")
    return auc(run(cmd, gpu, tee=OUT / f"{tag}_base.log")), dump


def _rc(cam, frames, gpu, dump, tag):
    c = CAMS[cam]
    if c["rc"] is None:
        print(f"-- RC [{cam}] SKIPPED (azure is near-field — deployed config is RC off)")
        return None
    cmd = (
        f"{PY} rc_refine_from_dump.py --dump {dump} --val-dir {c['val']}"
        f" --sam-checkpoint {SAM} --render-h {c['rc']} --max-frames {frames}"
    )
    print(f"-- render-compare [{cam}] @{c['rc']}")
    txt = run(cmd, gpu, tee=OUT / f"{tag}_rc.log")
    return auc(txt, r"render-compare ADD-AUC@100mm ([0-9.]+)")


def cmd_eval(a):
    gpu = pick_gpu(a.gpu)
    print(f"GPU {gpu}\n")
    cams = list(CAMS) if a.cam == "all" else [a.cam]
    results = {}
    for cam in cams:
        tag = f"{cam}_{a.frames}"
        b, dump = _base(cam, a.frames, gpu, tag)
        r = None if a.no_rc else _rc(cam, a.frames, gpu, dump, tag)
        results[cam] = (b, r)
        print()
    print("=" * 62)
    print(f"{'camera':<11}{'base':>9}{'+RC':>9}{'deployed':>11}  (n={a.frames})")
    print("-" * 62)
    finals = []
    for cam, (b, r) in results.items():
        f = r if r is not None else b
        if f is not None:
            finals.append(f)
        print(f"{cam:<11}{b if b else float('nan'):>9.4f}"
              f"{(f'{r:.4f}' if r is not None else '  off'):>9}"
              f"{CAMS[cam]['expect']:>11.4f}")
    if len(finals) == 4:
        print("-" * 62)
        print(f"{'MEAN':<11}{'':>9}{sum(finals) / 4:>9.4f}{0.8039:>11.4f}")
    print("=" * 62)
    print(f"logs + dumps: {OUT}")
    return 0


def cmd_smoke(a):
    print("SMOKE: deployed pipeline end-to-end, few frames. Expect base~0.82 -> RC~0.85 on kinect.\n")
    return cmd_eval(a)


def cmd_sota(a):
    print("SOTA reproduction: 4 cameras x 1000 held-out frames. Tens of minutes.")
    print("Target: rs .8153 / kinect .8275 / azure .7945 / orb .7784 -> mean .8039\n")
    a.no_rc = False          # RC is part of the deployed config; never skip it here
    return cmd_eval(a)


def cmd_viz(a):
    """Mesh-silhouette overlay — the 'does the estimate match reality' check."""
    gpu = pick_gpu(a.gpu)
    OUT.mkdir(exist_ok=True)
    out = Path(a.out) if a.out else OUT / f"viz_mesh_{a.cam}.png"
    c = CAMS[a.cam]
    cmd = (
        f"{PY} viz_mesh.py --detector {S1DET} --mlp-head {S1ANG}"
        f" --val-dir {c['val']} --indices {a.indices} --gt-skel --out {out}"
    )
    print(f"GPU {gpu}\n-- mesh overlay [{a.cam}]")
    run(cmd, gpu)
    print(f"\nwrote {out}")
    print("ORANGE = predicted mesh silhouette (nvdiffrast). GREEN = GT skeleton.")
    print("Read the PNG: mesh should sit on the robot and hug the green skeleton.")
    return 0


def cmd_fwd(a):
    """Direct invocation of the TRAIN/ internals — no eval loop, no dataset.

    This is the layer most edits touch (model.py / model_v4.py / norm_utils.py).
    Catches shape drift and checkpoint incompatibility in seconds instead of
    after a 40-second eval spin-up.
    """
    gpu = pick_gpu(a.gpu, need=3000)
    # Inject paths as a header so the body below stays a PLAIN string — an
    # f-string body would swallow the script's own f-string braces.
    header = (
        f"TRAIN_DIR = {str(TRAIN)!r}\n"
        f"EVAL_DIR  = {str(EVAL)!r}\n"
        f"S1DET     = {str(S1DET)!r}\n"
        f"S1ANG     = {str(S1ANG)!r}\n"
        f"MODEL     = 'facebook/dinov3-vitb16-pretrain-lvd1689m'\n"
    )
    body = '''
import sys, torch
sys.path.insert(0, TRAIN_DIR); sys.path.insert(0, EVAL_DIR)
dev = "cuda"
import model as M
from norm_utils import get_norm_stats
from model_angle import AnglePredictor

print("== norm stats (backbone-specific; a mismatch silently corrupts results) ==")
print("  dinov3 ->", get_norm_stats(MODEL, verbose=False))

print("== FK (pure, no weights) ==")
kp0 = M.panda_forward_kinematics(torch.zeros(2, 7))
print("  panda_forward_kinematics(zeros)", tuple(kp0.shape), "link0", kp0[0, 0].tolist())
assert kp0.shape[1:] == (7, 3), kp0.shape

print("== solver round-trip (the A0 gate, in miniature) ==")
# Project known camera-frame keypoints, solve PnP back, demand mm-level recovery.
K = torch.tensor([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]]).expand(2, 3, 3)
kp_r = M.panda_forward_kinematics(torch.randn(2, 7) * 0.4)
cam = kp_r + torch.tensor([0., 0., 1.5])
uv = (K @ (cam / cam[..., 2:]).transpose(1, 2)).transpose(1, 2)[..., :2]
kp_cam, valid, reproj = M.solve_pnp_batch(uv, kp_r, K)
err_mm = (kp_cam - cam).norm(dim=-1).mean().item() * 1000
print("  solve_pnp_batch ->", tuple(kp_cam.shape), "valid", valid.tolist(),
      "reproj", [round(v, 4) for v in reproj.tolist()])
print("  round-trip error %.4f mm" % err_mm)
assert valid.all() and err_mm < 5.0, "PnP round-trip broken: %.2f mm" % err_mm

print("== build + load the deployed detector/angle pair ==")
m = AnglePredictor(MODEL, 512, head_type="mlp").to(dev).eval()
sd = {k.replace("module.", ""): v for k, v in torch.load(S1DET, map_location=dev).items()}
ref = m.state_dict()
keep = {k: v for k, v in sd.items() if k in ref and v.shape == ref[k].shape}
dropped = [k for k in sd if k not in keep]
m.load_state_dict(keep, strict=False)
m.angle_head.load_state_dict(torch.load(S1ANG, map_location=dev))
print("  detector: %d/%d tensors loaded" % (len(keep), len(sd)))
if dropped:
    print("  dropped (shape/name mismatch):", dropped[:6], "..." if len(dropped) > 6 else "")
assert any("keypoint_head" in k for k in keep), "no keypoint_head weights loaded — model/ckpt mismatch"

print("== forward a random 512x512 batch ==")
with torch.no_grad():
    out = m(torch.randn(1, 3, 512, 512, device=dev), K[:1].to(dev))
items = out.items() if isinstance(out, dict) else enumerate(
    out if isinstance(out, (tuple, list)) else [out])
for k, v in items:
    print("  ", k, tuple(v.shape) if torch.is_tensor(v) else type(v).__name__)
print("")
print("FWD OK")
'''
    print(f"GPU {gpu}\n-- direct invocation (TRAIN internals)")
    run(f"{PY} -c {shquote(header + body)}", gpu, echo_all=True)
    return 0


def shquote(s):
    return "'" + s.replace("'", "'\\''") + "'"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, frames=None, cam=None):
        p.add_argument("--gpu", help="GPU UUID (default: most free). Env: DINOBOT_GPU")
        if frames is not None:
            p.add_argument("--frames", type=int, default=frames)
        if cam is not None:
            p.add_argument("--cam", default=cam, choices=list(CAMS) + ["all"])

    common(sub.add_parser("doctor", help="env/GPU/checkpoint/dataset check"))
    p = sub.add_parser("smoke", help="fast end-to-end (base+RC, 20 frames)")
    common(p, frames=20, cam="kinect")
    p.add_argument("--no-rc", action="store_true")
    p = sub.add_parser("eval", help="deployed eval for a camera")
    common(p, frames=200, cam="kinect")
    p.add_argument("--no-rc", action="store_true", help="skip render-compare")
    p = sub.add_parser("sota", help="all 4 cameras, the 0.804 reproduction")
    common(p, frames=1000, cam="all")
    p = sub.add_parser("viz", help="mesh-overlay PNG")
    common(p, cam="kinect")
    p.add_argument("--indices", default="50,800,1600,2400,3200,4000")
    p.add_argument("--out")
    common(sub.add_parser("fwd", help="direct model invocation, no eval loop"))

    a = ap.parse_args()
    return {"doctor": cmd_doctor, "smoke": cmd_smoke, "eval": cmd_eval,
            "sota": cmd_sota, "viz": cmd_viz, "fwd": cmd_fwd}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
