# AirCount FastAPI backend

This service uses the trained Stage-1 `CoarseCounter` checkpoint. It deliberately
does not apply the gated RefineNet because the current ErrorHead checkpoint trains
the error classifier; a validated trained refiner checkpoint has not been confirmed.

## Expected project layout

```text
aerial-crowd-counting/
├── error_aware/
│   ├── model_gated.py
│   └── checkpoints_stage1_holdout80/
│       └── coarse_best.pth
└── backend/
    ├── main.py
    ├── inference.py
    └── requirements.txt
```

## Install

Run from the project root, with your existing `.venv` activated:

```powershell
python -m pip install -r backend/requirements.txt
```

PyTorch and torchvision should already be installed in the project's environment.
Do not reinstall them unless your current environment is missing them.

## Start

From `D:\aerial-crowd-counting`:

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Check:
- http://127.0.0.1:8000/
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

The checkpoint default is:
`error_aware/checkpoints_stage1_holdout80/coarse_best.pth`

Override it in PowerShell if needed:

```powershell
$env:COARSE_CHECKPOINT="D:\aerial-crowd-counting\error_aware\checkpoints_stage1_holdout80\coarse_best.pth"
```

## Frontend

The React app already calls `POST http://localhost:8000/predict` and sends the
image in a multipart field named `file`. Its density-map display uses the returned
`density_map_url`.

If the frontend `.env` is set to another API URL, restart Vite after changing it.

## Inference behavior and caveats

- Input is RGB, converted to float `[0,1]`, matching `ErrorAwareCrowdDataset`.
- Images are tiled into non-overlapping 512x512 crops; right/bottom edges are
  replicated to complete a tile. This preserves the training pixel scale.
- Per-tile stride-8 density maps are stitched and summed.
- `predicted_count` is rounded from the summed density mass.
- For very large images, inference time grows with the number of tiles.
- This is an initial deployment pipeline. Validate predictions on representative
  full-resolution images before reporting model accuracy. The tile layout and
  edge handling should be evaluated against the training crop distribution.
