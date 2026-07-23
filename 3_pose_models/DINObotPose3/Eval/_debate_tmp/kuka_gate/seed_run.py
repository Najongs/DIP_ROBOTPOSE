"""Seed-injection wrapper: set cv2/np/torch RNG seed, then run a target eval script's main.
Usage: python seed_run.py <SEED> <path/to/eval_script.py> [script args...]
The RANSAC in solve_pose_kinematic.pnp_init uses cv2.solvePnPRansac (unseeded in the
eval scripts); this wrapper fixes cv2.theRNG so each run is reproducible for a given seed.
"""
import sys, os
seed = int(sys.argv[1])
script = os.path.abspath(sys.argv[2])
sys.argv = [script] + sys.argv[3:]
import cv2, numpy as np, torch
cv2.setRNGSeed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print(f"[seed_run] seed={seed} script={script}", flush=True)
import runpy
runpy.run_path(script, run_name='__main__')
