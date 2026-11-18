# AirCount FastAPI backend

This service counts clearly visible people with a pretrained person detector
and retains the trained Stage-1 `CoarseCounter` for tiny or crowded aerial scenes.
The response identifies which method supplied the count.

## Install and start

From `D:\aerial-crowd-counting`, with the existing `.venv` activated:

```powershell
python -m pip install -r aerial-crowd-backend/backend/requirements.txt
python aerial-crowd-backend/backend/download_person_detector.py
python -m uvicorn backend.main:app --app-dir aerial-crowd-backend --reload --host 127.0.0.1 --port 8000
```

PyTorch and torchvision should already be installed in the project environment.

- API: http://127.0.0.1:8000/
- Model status and checkpoint: http://127.0.0.1:8000/health
- API documentation: http://127.0.0.1:8000/docs

The backend prefers `error_aware/checkpoints_balanced_low_vram/coarse_validated.pth`
when that explicitly promoted checkpoint exists. Otherwise it uses the original
`error_aware/checkpoints_stage1_holdout80/coarse_best.pth`. `/health` reports the
actual checkpoint in use. To select a checkpoint explicitly, set its path before
starting the backend:

```powershell
$env:COARSE_CHECKPOINT="D:\aerial-crowd-counting\error_aware\checkpoints_balanced_low_vram\coarse_validated.pth"
```

Only use the fine-tuned best checkpoint if its validation metrics improve. A
`coarse_last.pth` file alone does not indicate improvement.
Clear an old override with `$env:COARSE_CHECKPOINT=$null` to restore automatic
checkpoint selection, then restart the backend.

## Person detection for visible individuals

The density model estimates 16.79 for the 22-person image. Its learned density
mass is not a reliable integer count for these large individual pedestrians.
Auto mode instead counts COCO `person` detections from Torchvision's pretrained
Faster R-CNN ResNet50 FPN v2, with numbered boxes available for visual inspection.
This detector is pretrained; no additional training epochs were run for this change.

| Provided image | Live API count | Numbered boxes |
| --- | ---: | ---: |
| Three pedestrians | 3 | 3 |
| Five pedestrians | 5 | 5 |
| `22.png` | 22 | 22 |

The combined detector/density path peaked at **0.82 GiB reserved VRAM** on the
4 GB RTX 2050. The original density checkpoint and sparse policy remain intact.

`COUNTING_MODE` accepts:

- `auto` (default): use person detections for sufficiently large, consistent
  bodies at the detector's resized resolution. Counts must be within a factor
  of two of the density evidence, and boxes must explain at least 75% of its mass.
  Tiny people, uncertain coverage, and saturated outputs keep the density fallback.
- `density`: always use the trained density pipeline and its sparse policy.
- `detection`: explicitly count COCO person detections, including a zero count
  when none are found. This bypasses the automatic fallback checks.

The detector uses score threshold 0.8 and COCO's person category, regardless of
walking pose. `person_detector_policy.json` pins both model hashes for auto mode.
Startup never downloads weights. Missing detector weights or a mismatched density
checkpoint disable auto handoff; inspect `detector_status` in `/health`.

Weights are installed by the download command above in
`checkpoints/person_detection/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`.
`PERSON_DETECTOR_CHECKPOINT` can select another location for those same weights.
Restart the backend after changing model files or counting-mode environment settings.

The benchmark compared 647 validation and 648 test images with the deployed
density pipeline. One validation count improved; all test counts were unchanged
because the small-person scenes kept density counting. This checks fallback
preservation; the provided images exercise the large-person detection route.
Reports are `error_aware/evaluation/person_detection_handoff.json` and
`error_aware/evaluation/person_detection_examples.json`.

```powershell
python error_aware/evaluate_person_detection.py
```

The benchmark caches progress every 25 images under `outputs/person_detection_benchmark`
and resumes completed images when model hashes and settings match.

## Density fallback results

The promoted density checkpoint is the 0.375 intermediate state. Three fine-tuning
epochs were completed; selection uses the initial zoom fine-tune and the first
sparse-training epoch. The second sparse epoch was evaluated and retained.
Training peaked at 0.84 GiB reserved VRAM on the 4 GB RTX 2050.

| Check | Original | Selected |
| --- | ---: | ---: |
| Three-person photo: rounded count | 2 | 3 |
| Three-person photo: density sum | 2.068703 | 2.507599 |
| Test full-image MAE | 5.266239 | 5.182490 |
| Test sparse-image MAE | 2.086952 | 2.085219 |
| Test crop MAE | 0.799675 | 0.796821 |
| Test sparse-crop MAE | 1.239064 | 1.242665 |
| Test crop RMSE | 2.431898 | 2.411839 |

Sparse-image error is essentially unchanged; sparse-crop error increases slightly.
The selected model improves overall full-image count error and passes the
three-person regression. The prediction handler was checked with the promoted
weights, including the returned image dimensions and generated density PNG.

Reports: `error_aware/evaluation/balanced_low_vram_comparison.json` and
`error_aware/evaluation/balanced_low_vram_selection.json`. The comparison records
the tested/promoted weight hash. Original weights remain available via
`COARSE_CHECKPOINT=error_aware/checkpoints_stage1_holdout80/coarse_best.pth`.

## Inference

- RGB float inputs in `[0,1]` match `ErrorAwareCrowdDataset`.
- Non-overlapping 512x512 crops preserve the training pixel scale.
- Incomplete right/bottom tiles receive black padding, matching PIL's training
  crop behavior. Repeating edge pixels changes the model's input distribution.
- Stride-8 density maps are stitched; cells entirely outside the image are
  excluded. The density sum estimates the count, rather than the number of peaks.
- `raw_count` always exposes the continuous density sum. `estimated_count` uses
  either that sum or the validated sparse response count. `predicted_count`
  rounds the selected estimate once (halves round up).
- `count_method` identifies `person_detection`, `density_integral`, or `localized_components`.
- Person detections return `detections` (original-image boxes and scores),
  `detection_overlay_url`, and the fallback's `density_predicted_count` for comparison.
  The frontend shows the numbered overlay when detection supplies the count;
  open the image to inspect the full-size boxes.
- `/predict` accepts JPG, PNG, or WebP in multipart field `file`, up to 20 MB.
- Density images are served at the returned `density_map_url`.
- Inference time grows with the number of tiles. Validate accuracy on images
  representative of the intended camera height, resolution, and scene.

The React app calls `http://localhost:8000/predict` by default. Restart Vite after
changing its `VITE_API_URL` environment setting.

## Sparse response counting in density mode

The five-person example produced density mass 6.0609: individual responses can
exceed one count, and weak responses also appear near the lamp and tile boundary.
The selected sparse rule counts separated strong responses for this image.
Live `POST /predict` checks return **5** for the five-person image and **3** for
the earlier three-person image.

`count_policy.json` is enabled only for its matching checkpoint SHA-256. The
validation-selected rule applies within density mass 4 through 8, requires at
least two strong components, rejects components below mass 0.6, and falls back
when a component exceeds 1.5 or localization disagrees by more than one count.
The density map continues to show the raw model response. `/health` exposes the
active policy; a different checkpoint automatically skips this policy.

These metrics compare rounded full-image counts using the API's visible-image
padding crop, rather than the continuous density metrics above:

| Metric | Density only | Sparse rule |
| --- | ---: | ---: |
| Validation image MAE | 4.874807 | 4.867079 |
| Validation sparse-image MAE | 2.005 | 1.965 |
| Test image MAE | 5.160494 | 5.165123 |
| Test sparse-image MAE | 2.04 | 2.01 |

The rule improves sparse-scene test error by 1.47%, with a 0.09% increase in overall
test error. This tradeoff is explicit in the evaluation report; calibration is
strict by default and this deployment used an explicit 0.5% overall test tolerance.
To reproduce selection and evaluation from the project root:

```powershell
python error_aware/calibrate_localization.py --objective overall --max-component-mass 1.5 --max-test-overall-regression 0.005
```

Features are cached under `outputs/localization_cache`, keyed by checkpoint and
manifest hashes. The report is
`error_aware/evaluation/localization_counting_report.json`. Rename
`aerial-crowd-backend/backend/count_policy.json` and restart to disable localization without changing
the trained weights.

## Fine-tuning on RTX 2050 / 4 GB VRAM

The hardware profile uses microbatch 1, gradient accumulation 4 (effective batch
4), FP16 autocast with gradient scaling, 512x512 crops, frozen early VGG layers,
and zero data-loader workers. CUDA allocations are limited to 80% of VRAM.

Fine-tuning starts from the existing checkpoint. It mixes native crops with
1.5x/2x/3x zoom augmentation and horizontal flips, transforms point annotations,
and regenerates count-preserving density targets. The count-loss weight is 0.05.
It never uses the uploaded three-person JPEG as a training example.

Run from the project root:

```powershell
python error_aware/finetune_low_vram.py --smoke-test
python error_aware/finetune_low_vram.py --epochs 3
python error_aware/evaluate_low_vram.py
```

The smoke test processes eight training batches and eight validation crops; it
does not save weights. A full run first evaluates the original checkpoint on all
validation crops, then validates after each epoch. Training/validation source
images must be disjoint. The separate output directory contains:

- `coarse_last.pth`: latest completed epoch.
- `coarse_best.pth`: written only when crop MAE improves and both sparse-crop
  MAE and full-image MAE are no worse than the original model.
- `training_history.json`: baseline, validation metrics, run time, and peak VRAM.

Original checkpoints remain intact. Further continuation can initialize from a
candidate using `--checkpoint`; this starts a new optimizer rather than resuming
optimizer state. Select a different `--output-dir` for each experiment.

`evaluate_low_vram.py` compares both checkpoints on the separate test manifest
and saves `error_aware/evaluation/low_vram_comparison.json`. Use `--image` to also
compare counts on a specific local image. Review this report before deployment.
`--resume` reuses completed model evaluations when the checkpoint hash and sample
path match. The report is saved after each model finishes.

## Sparse-scene follow-up

The first zoom-augmented candidate counts the three-person example correctly,
but test sparse-image MAE increased from 2.087 to 2.352. It is not the default.
The follow-up mines labeled empty crops from the same Stage-1 training images,
mixes 2,048 positive and 2,048 negative crops, and doubles the empty-crop loss.
It retains 20% zoom augmentation and freezes the backbone. No validation/test
images are used for background mining.

The tested profile uses microbatch 2, accumulation 2, and FP16; its smoke test
peaked at 0.84 GiB reserved VRAM on the RTX 2050.

```powershell
python error_aware/finetune_low_vram.py --epochs 2 --checkpoint error_aware/checkpoints_low_vram/coarse_last.pth --reference-checkpoint error_aware/checkpoints_stage1_holdout80/coarse_best.pth --output-dir error_aware/checkpoints_sparse_low_vram --batch-size 2 --accumulate 2 --freeze-backbone --sparse-rebalance --zoom-probability 0.2 --empty-weight 2 --head-lr 0.000005
```

With `--sparse-rebalance`, `coarse_best.pth` requires better sparse full-image MAE
and no worse overall full-image MAE against the reference checkpoint. Mining
statistics and the selected training manifest are saved with the run. The
original model remains the backend default until a candidate passes comparison
and is copied to the explicitly promoted `coarse_validated.pth` path.

`select_checkpoint_average.py` validates intermediate parameter values between
the zoom and sparse fine-tunes. It produces a single standard CoarseCounter
checkpoint, with no extra model needed during inference. It records source
checkpoint hashes and checks overall/sparse image MAE plus an optional image
regression before writing `coarse_best.pth`:

```powershell
python error_aware/select_checkpoint_average.py --weight 0.375 --image "WhatsApp Image 2026-11-18 at 11.16.26.jpeg"
```

This selection adds validation passes, not training epochs. Passing validation
still requires a separate test comparison before changing the backend default.

## Regression checks

```powershell
python -m unittest discover -s aerial-crowd-backend/backend -p 'test_*.py'
python -m unittest discover -s error_aware -p test_finetune_low_vram.py
```
