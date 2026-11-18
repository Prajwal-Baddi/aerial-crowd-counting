"""Compare a fine-tuned candidate with the deployed model on untouched test data."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from finetune_low_vram import ROOT, CrowdCounterService, ErrorAwareCrowdDataset, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "error_aware/checkpoints_stage1_holdout80/coarse_best.pth")
    parser.add_argument("--candidate", type=Path, default=ROOT / "error_aware/checkpoints_low_vram/coarse_last.pth")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "error_aware/evaluation/low_vram_comparison.json")
    parser.add_argument("--resume", action="store_true", help="Reuse completed evaluations of unchanged checkpoint files.")
    args = parser.parse_args()
    dataset = ErrorAwareCrowdDataset(ROOT / "error_aware/processed/test.jsonl")
    report = json.loads(args.output.read_text(encoding="utf-8")) if args.resume and args.output.exists() else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, checkpoint in (("baseline", args.baseline), ("candidate", args.candidate)):
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        sample_path = str(args.image.resolve()) if args.image else None
        cached = report.get(name, {})
        if cached.get("checkpoint_sha256") == digest and cached.get("sample_image") == sample_path:
            print(f"Reusing completed {name} evaluation", flush=True)
            continue
        service = CrowdCounterService(checkpoint)
        metrics = evaluate(service.model, dataset, service.device, batch_size=1)
        report[name] = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                        "sample_image": sample_path, "test": metrics}
        if args.image:
            with Image.open(args.image) as image:
                result = service.predict(image)
            report[name]["sample"] = {k: v for k, v in result.items() if k != "density_map"}
        print(name, report[name], flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        del service
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
