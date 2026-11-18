"""Person detections for clear individuals, with an aerial density fallback."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2


def detection_scene_check(detections, image_size, density_count):
    """Require large, consistent detections and reject a likely missed crowd."""
    if not detections:
        return False, "no_person_detections"
    if len(detections) >= 100:
        return False, "detector_output_limit"
    width, height = image_size
    scale = min(800 / min(width, height), 1333 / max(width, height))
    heights = np.array([d["box"][3] - d["box"][1] for d in detections]) * scale
    if heights.min() < 64 or np.median(heights) < 80:
        return False, "tiny_or_distant_people"
    if not density_count * 0.5 <= len(detections) <= max(2, density_count * 2):
        return False, "models_disagree"
    return True, "large_visible_people"


def density_coverage(density, detections):
    total = float(density.sum(dtype=np.float64))
    if total <= 1e-8:
        return 1.0
    mask = np.zeros(density.shape, dtype=bool)
    for detection in detections:
        x0, y0, x1, y1 = detection["box"]
        # The density counter has stride 8; include one cell around each body.
        left, top = max(0, int(x0 // 8) - 1), max(0, int(y0 // 8) - 1)
        right = min(mask.shape[1], int(np.ceil(x1 / 8)) + 1)
        bottom = min(mask.shape[0], int(np.ceil(y1 / 8)) + 1)
        mask[top:bottom, left:right] = True
    return float(density[mask].sum(dtype=np.float64) / total)


def annotate_people(image, detections):
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    font_size = max(16, round(max(image.size) / 50))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    line_width = max(2, round(max(image.size) / 500))
    for number, detection in enumerate(detections, 1):
        x0, y0, x1, y1 = detection["box"]
        draw.rectangle((x0, y0, x1, y1), outline=(25, 120, 225), width=line_width)
        label = str(number)
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width = bounds[2] - bounds[0] + 10
        top = max(0, y0 - font_size - 8)
        draw.rectangle((x0, top, x0 + text_width, top + font_size + 8), fill=(25, 120, 225))
        draw.text((x0 + 5, top + 1), label, fill="white", font=font)
    return overlay


class PersonDetector:
    def __init__(self, checkpoint_path, device, score_threshold=0.8):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        if not 0 < score_threshold < 1:
            raise ValueError("Person detection threshold must be between zero and one")
        self.score_threshold = score_threshold
        # Startup never downloads files or changes the user's Torch cache.
        self.model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        with self.checkpoint_path.open("rb") as stream:
            self.checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

    @torch.inference_mode()
    def detect(self, image):
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).to(self.device)
        output = self.model([tensor])[0]
        keep = (output["labels"] == 1) & (output["scores"] >= self.score_threshold)
        boxes = output["boxes"][keep].cpu().numpy()
        scores = output["scores"][keep].cpu().tolist()
        detections = []
        width, height = image.size
        for box, score in zip(boxes, scores):
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if box[2] > box[0] and box[3] > box[1]:
                detections.append({"box": box.tolist(), "score": score})
        return sorted(detections, key=lambda d: (d["box"][1], d["box"][0]))


class HybridCounterService:
    def __init__(self, counter, detector, mode="auto", min_coverage=0.75):
        if mode not in ("auto", "detection"):
            raise ValueError("Hybrid counting mode must be auto or detection")
        self.counter, self.detector = counter, detector
        self.device, self.count_policy = counter.device, counter.count_policy
        self.mode, self.min_coverage = mode, min_coverage

    def predict(self, image):
        start = time.perf_counter()
        result = self.counter.predict(image, return_density=True)
        density = result.pop("density_array")
        detections = self.detector.detect(image)
        use_detector, reason = detection_scene_check(detections, image.size, result["raw_count"])
        coverage = density_coverage(density, detections)
        if use_detector and coverage < self.min_coverage:
            use_detector, reason = False, "unexplained_density"
        if self.mode == "detection":
            use_detector, reason = True, "explicit_detection_mode"
        result["density_predicted_count"] = result["predicted_count"]
        result["detector_candidate_count"] = len(detections)
        result["detector_selection_reason"] = reason
        result["density_coverage"] = coverage
        result["detections"] = detections if use_detector else []
        if use_detector:
            result.update(predicted_count=len(detections), estimated_count=float(len(detections)),
                          count_method="person_detection", detection_overlay=annotate_people(image, detections),
                          model="Faster R-CNN ResNet50 FPN v2 (COCO person)")
        result["inference_time_ms"] = (time.perf_counter() - start) * 1000
        return result


def configure_detector(counter, mode, detector_path, policy_path, density_path):
    if mode not in ("auto", "density", "detection"):
        raise ValueError("COUNTING_MODE must be auto, density, or detection")
    if mode == "density":
        return counter, "disabled"
    policy_path, detector_path = Path(policy_path), Path(detector_path)
    if mode == "auto" and not policy_path.exists():
        return counter, "not_validated"
    policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}
    if mode == "auto":
        if policy.get("algorithm") != "large_people_v1":
            return counter, "unsupported_policy"
        with Path(density_path).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if policy.get("density_checkpoint_sha256") != digest:
            return counter, "density_checkpoint_mismatch"
    if not detector_path.exists():
        if mode == "detection":
            raise FileNotFoundError(f"Person detector weights not found: {detector_path}")
        return counter, "weights_missing"
    detector = PersonDetector(detector_path, counter.device, policy.get("score_threshold", 0.8))
    if mode == "auto" and detector.checkpoint_sha256 != policy.get("detector_checkpoint_sha256"):
        return counter, "detector_checkpoint_mismatch"
    coverage = float(policy.get("min_coverage", 0.75))
    if not 0 <= coverage <= 1:
        raise ValueError("Person detector density coverage must be in [0,1]")
    return HybridCounterService(counter, detector, mode, coverage), "enabled"
