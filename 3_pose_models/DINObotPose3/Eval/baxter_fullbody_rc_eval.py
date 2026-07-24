"""Baxter 17kp full-body render-and-compare evaluation.

Crop detector/heads provide theta(12), R and t. A 17kp reprojection solve is applied first,
then a canonical torso + both-arm + gripper silhouette refines theta, R and t against SAM.
The future full-frame detector only replaces the initial localization stage.
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(__file__)
TRAIN = os.path.abspath(os.path.join(HERE, "../TRAIN"))
sys.path.extend((TRAIN, HERE))

from baxter_fullbody_add_eval import refine_full
from baxter_fullbody_render import (
    ANG12,
    KP17,
    baxter_fullbody_all_link_transforms,
    make_baxter_fullbody_renderer,
)
from dataset import PoseEstimationDataset
from model_angle import AnglePredictor, rot6d_to_matrix
from model_v4 import _BAXTER_FB_JOINT_LIMITS, baxter_forward_kinematics
from refine_eval import add_auc, geometric_K, scale_K
from segment_anything import SamPredictor, sam_model_registry


def matrix_to_rot6d(rotation):
    return torch.cat((rotation[..., 0], rotation[..., 1]), dim=-1)


def soft_iou(rendered, target):
    intersection = (rendered * target).sum()
    union = (rendered + target - rendered * target).sum().clamp(min=1e-6)
    return intersection / union


def full_intrinsics(data_dir, device):
    settings = json.load(open(os.path.join(data_dir, "_camera_settings.json")))
    item = settings["camera_settings"][0]["intrinsic_settings"]
    width = int(item["resolution"]["width"])
    height = int(item["resolution"]["height"])
    side = max(width, height)
    pad_x, pad_y = (side - width) // 2, (side - height) // 2
    matrix = torch.tensor([
        [item["fx"], 0.0, item["cx"] + pad_x],
        [0.0, item["fy"], item["cy"] + pad_y],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)
    return matrix, width, height, side, pad_x, pad_y


def project(points, rotation, translation, intrinsics):
    camera = torch.einsum("bij,bnj->bni", rotation, points) + translation.unsqueeze(1)
    depth = camera[..., 2].clamp(min=1e-3)
    u = camera[..., 0] / depth * intrinsics[:, 0, 0:1] + intrinsics[:, 0, 2:3]
    v = camera[..., 1] / depth * intrinsics[:, 1, 1:2] + intrinsics[:, 1, 2:3]
    return torch.stack((u, v), dim=-1), camera


def main():
    parser = argparse.ArgumentParser()
    checkpoint_root = os.path.join(TRAIN, "checkpoints/baxter/fullbody_17kp")
    parser.add_argument("--detector", default=os.path.join(checkpoint_root, "detector.pth"))
    parser.add_argument("--angle-head", default=os.path.join(checkpoint_root, "angle.pth"))
    parser.add_argument("--rot-head", default=os.path.join(checkpoint_root, "rotation.pth"))
    parser.add_argument("--val-dir", default="/home/najo/NAS/DIP/datasets/synthetic/baxter_synth_test_dr")
    parser.add_argument("--sam-checkpoint", default=os.path.join(HERE, "../weights_sam/sam_vit_b_01ec64.pth"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--reproj-iters", type=int, default=150)
    parser.add_argument("--rc-iters", type=int, default=60)
    parser.add_argument("--crop-margin", type=float, default=1.5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DINO, SAM and nvdiffrast")
    device = torch.device("cuda")
    size = args.image_size
    lower = torch.tensor([x[0] for x in _BAXTER_FB_JOINT_LIMITS], device=device)
    upper = torch.tensor([x[1] for x in _BAXTER_FB_JOINT_LIMITS], device=device)

    model = AnglePredictor(
        "facebook/dinov3-vitb16-pretrain-lvd1689m", size,
        fix_joint7_zero=False, head_type="mlp", num_kp=17, num_ang=12,
        with_rotation=True, with_translation=True,
    ).to(device).eval()
    detector = torch.load(args.detector, map_location=device)
    detector = {key.replace("module.", ""): value for key, value in detector.items()}
    state = model.state_dict()
    model.load_state_dict(
        {key: value for key, value in detector.items() if key in state and value.shape == state[key].shape},
        strict=False,
    )
    model.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))
    model.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))

    renderer = make_baxter_fullbody_renderer(device)
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint).to(device).eval()
    segmenter = SamPredictor(sam)
    k_full, width, height, side, pad_x, pad_y = full_intrinsics(args.val_dir, device)

    dataset = PoseEstimationDataset(
        args.val_dir, keypoint_names=KP17, image_size=(size, size), heatmap_size=(size, size),
        augment=False, include_angles=True, sigma=2.5, crop_to_robot=True,
        crop_margin=args.crop_margin, angle_joint_names=ANG12,
    )
    indices = list(range(0, len(dataset), max(1, len(dataset) // args.max_frames)))[:args.max_frames]
    before, after, mask_ious = [], [], []

    for count, index in enumerate(indices, 1):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        gt3d = sample["keypoints_3d"].unsqueeze(0).to(device)
        camera_k = sample["camera_K"].unsqueeze(0)
        original_size = sample["original_size"].unsqueeze(0)
        k_model = scale_K(camera_k, original_size, size).to(device)
        k_crop = geometric_K(args.val_dir, camera_k, original_size, size).to(device)
        with torch.no_grad():
            output = model(image, k_model)

        confidence = output["confidence"].clamp(min=1e-3)
        with torch.enable_grad():
            theta0, rotation0, translation0 = refine_full(
                output["joint_angles"].float(), output["rot_matrix"].float(), output["trans"].float(),
                k_crop, output["keypoints_2d"], confidence, lower, upper,
                iters=args.reproj_iters,
            )
        fk0 = baxter_forward_kinematics(theta0)
        _, camera0 = project(fk0, rotation0, translation0, k_full.unsqueeze(0))
        valid = gt3d.abs().sum(-1) > 1e-6
        before.append(float((camera0 - gt3d).norm(dim=-1)[valid].mean()))

        rgb_path = sample["annotation_path"].replace(".json", ".rgb.jpg")
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        padded = np.zeros((side, side, 3), dtype=np.uint8)
        padded[pad_y:pad_y + height, pad_x:pad_x + width] = rgb
        keypoints_full, _ = project(fk0, rotation0, translation0, k_full.unsqueeze(0))
        prompts = keypoints_full[0].detach().cpu().numpy()
        in_bounds = (
            (prompts[:, 0] >= 0) & (prompts[:, 0] < side)
            & (prompts[:, 1] >= 0) & (prompts[:, 1] < side)
        )
        if in_bounds.sum() < 5:
            after.append(before[-1])
            continue
        selected = prompts[in_bounds]
        low = selected.min(0)
        high = selected.max(0)
        margin = 0.12 * float(np.max(high - low))
        box = np.array([
            max(0.0, low[0] - margin), max(0.0, low[1] - margin),
            min(side - 1.0, high[0] + margin), min(side - 1.0, high[1] + margin),
        ], dtype=np.float32)
        segmenter.set_image(padded)
        masks, scores, _ = segmenter.predict(
            point_coords=selected.astype(np.float32),
            point_labels=np.ones(len(selected), dtype=np.int32),
            box=box, multimask_output=True,
        )
        target = torch.tensor(masks[int(np.argmax(scores))], dtype=torch.float32, device=device)

        theta = theta0.detach().clone().requires_grad_(True)
        rotation6d = matrix_to_rot6d(rotation0).detach().clone().requires_grad_(True)
        translation = translation0.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([
            {"params": [theta], "lr": 8e-3},
            {"params": [rotation6d], "lr": 2e-3},
            {"params": [translation], "lr": 2e-3},
        ])
        for _ in range(args.rc_iters):
            bounded_theta = torch.maximum(torch.minimum(theta, upper), lower)
            rotation = rot6d_to_matrix(rotation6d)
            vertices = renderer.robot_verts(bounded_theta, baxter_fullbody_all_link_transforms)
            rendered = renderer(vertices, rotation, translation, k_full.unsqueeze(0), side, side)[0]
            pose_prior = (
                0.05 * ((bounded_theta - theta0) ** 2).mean()
                + 0.02 * ((rotation - rotation0) ** 2).mean()
                + 0.10 * ((translation - translation0) ** 2).mean()
            )
            loss = 1.0 - soft_iou(rendered, target) + pose_prior
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        theta1 = torch.maximum(torch.minimum(theta, upper), lower).detach()
        rotation1 = rot6d_to_matrix(rotation6d).detach()
        translation1 = translation.detach()
        fk1 = baxter_forward_kinematics(theta1)
        _, camera1 = project(fk1, rotation1, translation1, k_full.unsqueeze(0))
        after.append(float((camera1 - gt3d).norm(dim=-1)[valid].mean()))
        with torch.no_grad():
            vertices = renderer.robot_verts(theta1, baxter_fullbody_all_link_transforms)
            rendered = renderer(vertices, rotation1, translation1, k_full.unsqueeze(0), side, side)[0]
            mask_ious.append(float(soft_iou(rendered, target)))
        if count % 10 == 0:
            print(f"{count}/{len(indices)} ADD {np.mean(before)*1000:.1f} -> {np.mean(after)*1000:.1f} mm")

    before_array, after_array = np.asarray(before), np.asarray(after)
    print(f"\nBaxter full-body 17kp RC ({len(before_array)} frames)")
    print(f"before: AUC {add_auc(before_array):.4f}, mean {before_array.mean()*1000:.1f}mm")
    print(f"after : AUC {add_auc(after_array):.4f}, mean {after_array.mean()*1000:.1f}mm")
    if mask_ious:
        print(f"final SAM/render IoU mean {np.mean(mask_ious):.3f}")


if __name__ == "__main__":
    main()
