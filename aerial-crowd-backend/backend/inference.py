
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# --------------------------------------------------
# Project path configuration
# --------------------------------------------------

# Resolve D:\aerial-crowd-counting
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make the main project directory importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from error_aware.model_gated import CoarseCounter


# --------------------------------------------------
# Checkpoint loading utilities
# --------------------------------------------------

def _extract_state_dict(checkpoint):
    """Support the checkpoint containers used by the training scripts."""

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)

            if isinstance(value, dict) and value:
                return value

    if isinstance(checkpoint, dict):
        return checkpoint

    raise RuntimeError("Checkpoint does not contain a state dictionary.")


def _clean_state_dict(state):
    """Try common wrappers/prefixes without silently dropping unmatched tensors."""

    state = {
        str(k): v
        for k, v in state.items()
        if torch.is_tensor(v)
    }

    candidates = [state]

    for prefix in (
        "module.",
        "model.",
        "coarse.",
        "module.coarse.",
        "model.coarse.",
    ):
        stripped = {
            k[len(prefix):]: v
            for k, v in state.items()
            if k.startswith(prefix)
        }

        if stripped:
            candidates.append(stripped)

    # Some checkpoints contain both a complete ESRNet
    # and a coarse submodule.
    for candidate in list(candidates):
        if any(k.startswith("coarse.") for k in candidate):
            nested = {
                k[len("coarse."):]: v
                for k, v in candidate.items()
                if k.startswith("coarse.")
            }

            if nested:
                candidates.append(nested)

    return candidates


def load_coarse_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    raw_state = _extract_state_dict(checkpoint)
    expected = model.state_dict()
    errors = []

    for candidate in _clean_state_dict(raw_state):

        compatible = {
            k: v
            for k, v in candidate.items()
            if (
                k in expected
                and tuple(v.shape) == tuple(expected[k].shape)
            )
        }

        if len(compatible) == len(expected):
            model.load_state_dict(compatible, strict=True)
            return checkpoint, len(compatible)

        errors.append(
            f"{len(compatible)}/{len(expected)} matching tensors; "
            f"sample checkpoint keys={list(candidate)[:5]}"
        )

    raise RuntimeError(
        "Could not match checkpoint weights to CoarseCounter. "
        "The checkpoint may belong to a different architecture. "
        + " | ".join(errors)
    )


# --------------------------------------------------
# Crowd counting inference service
# --------------------------------------------------

class CrowdCounterService:

    def __init__(
        self,
        checkpoint_path: Path,
        device_name="auto",
        tile_size=512,
    ):

        # Device selection
        if device_name == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device_name)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "MODEL_DEVICE=cuda was requested, but CUDA is unavailable."
            )

        self.tile_size = int(tile_size)

        if self.tile_size != 512:
            raise ValueError(
                "This checkpoint was trained on 512x512 crops; "
                "TILE_SIZE must be 512."
            )

        # Initialize coarse counter
        self.model = CoarseCounter(
            pretrained=False,
            freeze_early=True,
        ).to(self.device)

        # Load trained checkpoint
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        metadata, matched = load_coarse_checkpoint(
            self.model,
            checkpoint_path,
            self.device,
        )

        self.model.eval()

        print(f"[AirCount] Device: {self.device}")
        print(f"[AirCount] Checkpoint: {checkpoint_path}")
        print(f"[AirCount] Loaded {matched} coarse-counter tensors.")

        if isinstance(metadata, dict):
            print(
                "[AirCount] Checkpoint metadata: "
                f"epoch={metadata.get('epoch', 'unknown')}"
            )

    # --------------------------------------------------
    # Density heatmap generation
    # --------------------------------------------------

    @staticmethod
    def _make_heatmap(
        density: np.ndarray,
        output_size: tuple[int, int],
    ) -> Image.Image:

        density = np.maximum(
            density.astype(np.float32),
            0,
        )

        if density.max() > 0:
            normalized = (
                np.log1p(density)
                / np.log1p(density.max())
            )
        else:
            normalized = density

        gray = Image.fromarray(
            np.uint8(
                np.clip(normalized * 255, 0, 255)
            ),
            mode="L",
        )

        gray = gray.resize(
            output_size,
            Image.Resampling.BILINEAR,
        )

        x = np.asarray(
            gray,
            dtype=np.float32,
        ) / 255.0

        # Piecewise blue -> cyan -> yellow -> red palette
        stops = np.array(
            [
                [12, 20, 110],
                [0, 150, 255],
                [30, 220, 170],
                [255, 220, 40],
                [220, 35, 30],
            ],
            dtype=np.float32,
        )

        position = x * (len(stops) - 1)

        lo = np.floor(position).astype(np.int32)
        hi = np.minimum(
            lo + 1,
            len(stops) - 1,
        )

        frac = (position - lo)[..., None]

        rgb = (
            stops[lo] * (1 - frac)
            + stops[hi] * frac
        )

        return Image.fromarray(
            np.uint8(np.clip(rgb, 0, 255)),
            mode="RGB",
        )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    @torch.inference_mode()
    def predict(self, image: Image.Image):

        start = time.perf_counter()

        image = image.convert("RGB")

        original_w, original_h = image.size
        tile = self.tile_size

        # Pad right/bottom by replicating edge pixels.
        # No resizing: preserve the pixel scale used during training.

        padded_w = (
            (original_w + tile - 1) // tile
        ) * tile

        padded_h = (
            (original_h + tile - 1) // tile
        ) * tile

        canvas = Image.new(
            "RGB",
            (padded_w, padded_h),
        )

        canvas.paste(image, (0, 0))

        if padded_w > original_w:

            edge = image.crop(
                (
                    original_w - 1,
                    0,
                    original_w,
                    original_h,
                )
            ).resize(
                (
                    padded_w - original_w,
                    original_h,
                )
            )

            canvas.paste(edge, (original_w, 0))

        if padded_h > original_h:

            edge = canvas.crop(
                (
                    0,
                    original_h - 1,
                    padded_w,
                    original_h,
                )
            ).resize(
                (
                    padded_w,
                    padded_h - original_h,
                )
            )

            canvas.paste(edge, (0, original_h))

        # Full density map at stride 8
        full_density = np.zeros(
            (
                padded_h // 8,
                padded_w // 8,
            ),
            dtype=np.float32,
        )

        tile_count = 0

        # Process image tile by tile
        for y in range(0, padded_h, tile):

            for x in range(0, padded_w, tile):

                crop = canvas.crop(
                    (
                        x,
                        y,
                        x + tile,
                        y + tile,
                    )
                )

                array = (
                    np.asarray(
                        crop,
                        dtype=np.float32,
                    ) / 255.0
                )

                tensor = torch.from_numpy(
                    array.transpose(2, 0, 1).copy()
                )

                tensor = tensor.unsqueeze(0).to(self.device)

                output = self.model(tensor)

                density = (
                    output["density"]
                    .float()
                    .squeeze(0)
                    .squeeze(0)
                )

                density_np = density.cpu().numpy()

                gy = y // 8
                gx = x // 8

                full_density[
                    gy:gy + 64,
                    gx:gx + 64,
                ] = density_np

                tile_count += 1

        # Exclude density cells belonging only to padding.
        visible_density_h = (original_h + 7) // 8
        visible_density_w = (original_w + 7) // 8

        display_density = full_density[
            :visible_density_h,
            :visible_density_w,
        ]

        # Density values represent count mass per stride-8 cell.
        count = float(display_density.sum())

        heatmap = self._make_heatmap(
            display_density,
            (original_w, original_h),
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return {
            "predicted_count": max(
                0,
                int(round(count)),
            ),
            "inference_time_ms": elapsed_ms,
            "density_map": heatmap,
            "tiles_processed": tile_count,
        }
