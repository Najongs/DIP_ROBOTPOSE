from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

from mask_distance import (
    MaskDistanceResult,
    draw_distance_overlay,
    minimum_mask_distance,
    risk_from_distance_px,
)


def normalize_torch_device(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


@dataclass(frozen=True)
class StaticRiskResult:
    distance: MaskDistanceResult
    risk_score: float
    robot_inference_ms: float
    human_inference_ms: float


class SegFormerForRobotArm(nn.Module):
    def __init__(self, num_classes: int = 2, model_name: str = "nvidia/mit-b2"):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.segformer(pixel_values=pixel_values)
        return nn.functional.interpolate(
            outputs.logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class RobotSegFormerPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str = "nvidia/mit-b2",
        image_size: int = 512,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.image_size = image_size
        self.device = normalize_torch_device(device)
        self.model = self._load_model()

    def _load_model(self) -> SegFormerForRobotArm:
        model = SegFormerForRobotArm(num_classes=2, model_name=self.model_name)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def predict_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        input_tensor = self._preprocess(image_rgb)
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = torch.argmax(output, dim=1)[0].cpu().numpy().astype(np.uint8)
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


class HumanYoloSegPredictor:
    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        confidence: float = 0.25,
        device: Optional[str] = None,
    ):
        from ultralytics import YOLO

        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model = YOLO(model_path)

    def predict_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        results = self.model.predict(image_rgb, conf=self.confidence, device=self.device, verbose=False)
        result = results[0]
        h, w = image_rgb.shape[:2]
        combined = np.zeros((h, w), dtype=bool)

        if result.masks is None or result.boxes is None:
            return combined

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for idx, cls_id in enumerate(classes):
            if cls_id != 0:
                continue
            mask = cv2.resize(masks[idx], (w, h), interpolation=cv2.INTER_NEAREST)
            combined |= mask > 0.5

        return combined


class StaticCollisionPipeline:
    def __init__(
        self,
        robot_predictor: RobotSegFormerPredictor,
        human_predictor: HumanYoloSegPredictor,
        danger_px: float = 20.0,
        caution_px: float = 80.0,
    ):
        self.robot_predictor = robot_predictor
        self.human_predictor = human_predictor
        self.danger_px = danger_px
        self.caution_px = caution_px

    def run(self, image_rgb: np.ndarray) -> tuple[StaticRiskResult, np.ndarray, np.ndarray]:
        start = time.perf_counter()
        robot_mask = self.robot_predictor.predict_mask(image_rgb)
        robot_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        human_mask = self.human_predictor.predict_mask(image_rgb)
        human_ms = (time.perf_counter() - start) * 1000.0

        distance = minimum_mask_distance(robot_mask, human_mask)
        risk_score = risk_from_distance_px(distance.distance_px, self.danger_px, self.caution_px)

        return (
            StaticRiskResult(
                distance=distance,
                risk_score=risk_score,
                robot_inference_ms=robot_ms,
                human_inference_ms=human_ms,
            ),
            robot_mask,
            human_mask,
        )


def load_rgb(path: str | Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Static robot-human mask distance pipeline")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--robot-checkpoint",
        default="/home/najo/NAS/DIP/4_perception/Fr5_robot_SegFormer/best_segformer_robot_arm.pth",
        help="SegFormer robot checkpoint path",
    )
    parser.add_argument("--human-model", default="yolov8n-seg.pt", help="Ultralytics person segmentation model")
    parser.add_argument("--out-dir", default="collision_risk_pipeline/outputs", help="Output directory")
    parser.add_argument("--danger-px", type=float, default=20.0)
    parser.add_argument("--caution-px", type=float, default=80.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    image_rgb = load_rgb(args.image)
    robot = RobotSegFormerPredictor(args.robot_checkpoint, device=args.device)
    human = HumanYoloSegPredictor(args.human_model, device=args.device)
    pipeline = StaticCollisionPipeline(robot, human, danger_px=args.danger_px, caution_px=args.caution_px)

    result, robot_mask, human_mask = pipeline.run(image_rgb)
    overlay = draw_distance_overlay(image_rgb, robot_mask, human_mask, result.distance)

    out_dir = Path(args.out_dir)
    save_rgb(out_dir / "overlay.png", overlay)
    cv2.imwrite(str(out_dir / "robot_mask.png"), robot_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "human_mask.png"), human_mask.astype(np.uint8) * 255)

    result_dict = asdict(result)
    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    print(json.dumps(result_dict, indent=2, ensure_ascii=False))
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
