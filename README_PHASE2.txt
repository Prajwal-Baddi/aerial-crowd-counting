RESIDUAL PATCH REFINEMENT — PHASE 2

Purpose
-------
Replace the old independently predicted patch density with a learned signed
correction to the coarse model's density map.

Architecture
------------
Input 1: 160x160 RGB crop.
Input 2: corresponding 10x10 coarse density patch.
The coarse patch is upsampled and concatenated with the image crop.
The refiner predicts a signed 10x10 residual using a linear output layer.
Final patch = clamp(coarse patch + predicted residual, min=0).

Important data-integrity change
-------------------------------
Ground-truth residual targets are formed from the corresponding 10x10 slice
of the FULL 80x80 ground-truth density map. Do not regenerate a Gaussian
density map independently inside each crop. Slicing the full map preserves
Gaussian mass at patch boundaries and aligns the target with the coarse map.

New split
---------
model_split.py creates non-overlapping partitions:
  65% coarse-model training
   5% coarse-model checkpoint selection
  17% residual-refiner training
   3% residual-refiner checkpoint selection
  10% final test

The coarse model never trains on residual-refiner or final-test images.
The refiner never trains on final-test images.
This is a new split; the old mc_multiscale_crowdcounter.pth must NOT be used
for this experiment because it was trained on the former 80% training split.

Files to place alongside your existing project files
-----------------------------------------------------
model_split.py
residual_model.py
prepare_residual_cache.py
residual_data.py
train_residual_refiner.py
evaluate_residual_refinement.py

Required existing project files
-------------------------------
dataset_loader.py
model.py

Execution order (from project root, with venv activated)
---------------------------------------------------------
1. Train a new coarse model and create the split:
     python model_split.py

2. Generate aligned coarse and full-image GT density caches:
     python prepare_residual_cache.py

3. Train the residual patch refiner:
     python train_residual_refiner.py

4. Verify partitioning, cache alignment and tensor shapes:
     python check_residual_pipeline.py

5. Run the final evaluation once all checkpoints exist:
     python evaluate_residual_refinement.py

Expected generated files
------------------------
residual_phase2_split.json
residual_phase2_coarse.pth
residual_phase2_cache.pt
residual_patch_refiner_best.pth
residual_patch_refiner_last.pth
residual_refinement_test_results.csv

Hardware notes
--------------
The scripts use batch size 2 for the 4 GB RTX 2050 in coarse training and
batch size 8 for patch training. If CUDA runs out of memory, lower the relevant
batch size to 1 or 4 respectively.

Evaluation notes
----------------
Patch-level validation MAE is a training diagnostic only. The final comparison
must use full-image MAE and RMSE on final_test. Do not tune hyperparameters or
select checkpoints using final_test results.

The 10% final-test partition is still a holdout from this project-specific
random split, not a sequence-aware split. VisDrone contains video sequences;
a sequence-aware split is a stronger future evaluation if sequence metadata
is available.

Metric/runtime note
-------------------
The evaluator's uncertainty-guided timing includes coarse inference, 20 MC
passes and residual refinement. Random timing includes coarse inference and
refinement. Baseline timing includes coarse inference only.
