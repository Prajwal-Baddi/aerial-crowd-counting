import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from finetune_low_vram import ScaleAugmentedDataset


class AugmentationTests(unittest.TestCase):
    def test_zoom_transforms_labels_and_preserves_density_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            Image.new("RGB", (512, 512), "white").save(Path(directory) / "input.png")
            dataset = ScaleAugmentedDataset.__new__(ScaleAugmentedDataset)
            dataset.dataset_root = Path(directory)
            dataset.crop_size = 512
            dataset.density_stride = 8
            dataset.density_size = 64
            dataset.gaussian_sigma = 3.0
            dataset.records = [{"image": "input.png", "crop_x": 0, "crop_y": 0,
                                "points": [[10, 10], [255, 255], [300, 300]]}]
            # Zoom top-left 256x256 to 512x512; the third point is outside.
            with patch("finetune_low_vram.random.choice", return_value=2.0), patch(
                "finetune_low_vram.random.randint", return_value=0
            ), patch("finetune_low_vram.random.random", side_effect=[0.0, 1.0, 1.0]):
                image, density, count = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 512, 512))
            self.assertEqual(count.item(), 2)
            self.assertAlmostEqual(density.sum().item(), 2.0, places=5)
            self.assertGreater(density[0, 2, 2].item(), 0)
            self.assertGreater(density[0, 63, 63].item(), 0)


if __name__ == "__main__":
    unittest.main()
