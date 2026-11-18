"""Evaluate clear-person detection handoff while preserving aerial density counts."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from finetune_low_vram import ROOT, CrowdCounterService
from backend.detection import PersonDetector, detection_scene_check, density_coverage
from backend.localization import choose_count


def metrics(rows, key):
    target = np.array([r["target"] for r in rows])
    error = np.abs(np.array([r[key] for r in rows]) - target)
    sparse = (target > 0) & (target <= 5)
    return {"images": len(rows), "image_mae": float(error.mean()),
            "sparse_image_mae": float(error[sparse].mean()),
            "empty_image_mae": float(error[target == 0].mean()),
            "exact_images": int(np.sum(error == 0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "error_aware/checkpoints_balanced_low_vram/coarse_validated.pth")
    parser.add_argument("--detector", type=Path, default=ROOT / "checkpoints/person_detection/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "error_aware/evaluation/person_detection_handoff.json")
    parser.add_argument("--min-coverage", type=float, default=0.75)
    args = parser.parse_args()
    counter = CrowdCounterService(args.checkpoint, count_policy_path=ROOT / "aerial-crowd-backend/backend/count_policy.json")
    detector = PersonDetector(args.detector, counter.device)
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report = {"density_checkpoint_sha256": checkpoint_sha,
              "detector_checkpoint_sha256": detector.checkpoint_sha256,
              "score_threshold": detector.score_threshold, "min_coverage": args.min_coverage}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        features = json.loads((ROOT / "outputs/localization_cache" / f"{split}.json").read_text())
        if features["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("Density cache belongs to a different checkpoint")
        cache = ROOT / "outputs/person_detection_benchmark" / f"{split}.json"
        rows = []
        if cache.exists():
            saved = json.loads(cache.read_text())
            if all(saved.get(k) == v for k, v in report.items() if k not in ("validation", "test")):
                rows = saved["rows"]
        if len(rows) > len(features["images"]):
            raise ValueError("Unexpected benchmark cache length")
        for index in range(len(rows), len(features["images"])):
            record = features["images"][index]
            baseline = choose_count(record["raw_count"], record["masses"], counter.count_policy)[0]
            with Image.open(ROOT / "dataset/VisDrone2019-DET-train" / record["image"]) as source:
                image = source.convert("RGB")
            detections = detector.detect(image)
            selected, reason = detection_scene_check(detections, image.size, record["raw_count"])
            coverage = None
            if selected:
                result = counter.predict(image, return_density=True)
                baseline = result["predicted_count"]
                coverage = density_coverage(result["density_array"], detections)
                selected = coverage >= args.min_coverage
                if not selected:
                    reason = "unexplained_density"
            rows.append({"image": record["image"], "target": record["target"], "baseline": baseline,
                         "prediction": len(detections) if selected else baseline, "handoff": selected,
                         "candidate_count": len(detections), "reason": reason, "coverage": coverage})
            if (index + 1) % 25 == 0 or index + 1 == len(features["images"]):
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({**{k: v for k, v in report.items() if k not in ("validation", "test")}, "rows": rows}))
                print(f"{split}: {index+1}/{len(features['images'])} images; handoffs={sum(r['handoff'] for r in rows)}", flush=True)
        report[split] = {"baseline": metrics(rows, "baseline"), "auto": metrics(rows, "prediction"),
                         "handoffs": sum(r["handoff"] for r in rows), "reasons": dict(Counter(r["reason"] for r in rows))}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(split, report[split], flush=True)
    report["accepted"] = all(report[s]["auto"]["image_mae"] <= report[s]["baseline"]["image_mae"]
                             and report[s]["auto"]["sparse_image_mae"] <= report[s]["baseline"]["sparse_image_mae"]
                             for s in ("validation", "test"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("ACCEPTED", report["accepted"], flush=True)


if __name__ == "__main__":
    main()
