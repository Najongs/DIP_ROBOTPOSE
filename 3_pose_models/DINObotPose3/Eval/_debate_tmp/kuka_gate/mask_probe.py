"""Compare per-keypoint validity conventions on KUKA/Baxter synth-DR test sets:
  vmask   = batch['valid_mask']          (u1_solver_vs_direct.py uses this for ADD)
  gt3dnz  = (keypoints_3d != 0).any      (kuka_add_eval/baxter_add_eval uses this)
  allkp   = every keypoint               (RoboPEPP uses this: dist3d.mean over all C)
Reports how often they diverge -> whether the ADD averaging SET differs between scripts."""
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); EVAL = os.path.abspath(os.path.join(HERE, '../..'))
TRAIN = os.path.abspath(os.path.join(EVAL, '../TRAIN')); sys.path += [EVAL, TRAIN]
from dataset import PoseEstimationDataset
from torch.utils.data import DataLoader

def probe(name, val_dir, kp, joints):
    ds = PoseEstimationDataset(val_dir, keypoint_names=kp, image_size=(512,512), heatmap_size=(512,512),
                               augment=False, include_angles=True, sigma=2.5, crop_to_robot=True,
                               crop_margin=1.5, angle_joint_names=joints)
    ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
    tot=0; vsum=0; g3sum=0; both_diff=0; frames=0; vm_lt_all=0
    for b in ld:
        vm = (b['valid_mask']>0)                       # (B,N)
        g3 = (b['keypoints_3d'].abs().sum(-1)>0)       # (B,N)
        B,N = vm.shape
        tot += B*N; vsum += int(vm.sum()); g3sum += int(g3.sum())
        both_diff += int((vm!=g3).sum()); frames += B
        vm_lt_all += int((vm.sum(1) < N).sum())        # frames where vmask drops >=1 kp
    print(f"\n[{name}] {frames} frames x {N} kp = {tot} kp-slots")
    print(f"  valid_mask>0        : {vsum}/{tot} = {vsum/tot:.4f}  (u1/paper ADD set)")
    print(f"  gt3d!=0             : {g3sum}/{tot} = {g3sum/tot:.4f}  (kuka/baxter_add_eval ADD set)")
    print(f"  all keypoints       : {tot}/{tot} = 1.0000  (RoboPEPP ADD set)")
    print(f"  vmask != gt3d slots : {both_diff} ({100*both_diff/tot:.2f}%)")
    print(f"  frames where vmask drops >=1 kp: {vm_lt_all}/{frames} = {100*vm_lt_all/frames:.1f}%")

probe('KUKA', '../../../datasets/synthetic/kuka_synth_test_dr',
      [f'iiwa7_link_{i}' for i in range(1,8)], [f'iiwa7_joint_{i}' for i in range(1,8)])
probe('BAXTER-left', '../../../datasets/synthetic/baxter_synth_test_dr',
      ['left_s0','left_s1','left_e0','left_e1','left_w0','left_w1','left_w2'],
      ['left_s0','left_s1','left_e0','left_e1','left_w0','left_w1','left_w2'])
