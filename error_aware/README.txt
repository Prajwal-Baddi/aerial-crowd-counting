# Stage-2 held-out image split

Copy `create_stage2_holdout_split.py` into `error_aware/`.

From the project root, run:

```powershell
python error_aware/create_stage2_holdout_split.py --data-dir error_aware/processed --seed 42 --holdout-fraction 0.20
```

The script creates:
- `error_aware/processed/stage1_train80_balanced.jsonl`
- `error_aware/processed/stage2_train20_holdout.jsonl`
- `error_aware/processed/stage2_holdout_split_summary.json`

The split is grouped by original image. It filters the existing balanced training manifest for Stage 1 and uses all crops from the held-out images for Stage 2. Validation and test manifests are untouched.

After inspecting the summary, train a new Stage-1 checkpoint using:

```powershell
python error_aware/train.py --data-dir error_aware/processed --train-manifest stage1_train80_balanced.jsonl --output-dir error_aware/checkpoints_stage1_holdout80 --epochs 15 --batch-size 1
```

Do not start Stage 2 until the new Stage-1 validation result has been inspected.
