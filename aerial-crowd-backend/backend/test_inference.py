import unittest

import numpy as np
import torch
from PIL import Image

from inference import CrowdCounterService


class RecordingModel:
    def __init__(self, mass):
        self.mass = mass
        self.inputs = []

    def __call__(self, tensor):
        self.inputs.append(tensor.clone())
        density = torch.zeros((1, 1, 64, 64))
        density[0, 0, 0, 0] = self.mass
        return {"density": density}


class InferenceTests(unittest.TestCase):
    def service(self, mass=1.0):
        service = CrowdCounterService.__new__(CrowdCounterService)
        service.device = torch.device("cpu")
        service.tile_size = 512
        service.model = RecordingModel(mass)
        return service

    def test_edge_padding_matches_training_crop(self):
        image = Image.new("RGB", (513, 499), (60, 120, 180))
        service = self.service()
        result = service.predict(image)
        for x, actual in zip((0, 512), service.model.inputs):
            expected = np.asarray(image.crop((x, 0, x + 512, 512)), dtype=np.float32) / 255
            np.testing.assert_array_equal(actual[0].numpy(), expected.transpose(2, 0, 1))
        self.assertEqual(result["tiles_processed"], 2)
        self.assertEqual(result["raw_count"], 2.0)
        self.assertEqual(result["density_map"].size, image.size)

    def test_sum_before_rounding_and_keep_raw_estimate(self):
        result = self.service(mass=1.3).predict(Image.new("RGB", (1024, 512)))
        self.assertAlmostEqual(result["raw_count"], 2.6, places=5)
        self.assertEqual(result["estimated_count"], result["raw_count"])
        self.assertEqual(result["predicted_count"], 3)

    def test_empty_density_stays_zero(self):
        self.assertEqual(self.service(mass=0).predict(Image.new("RGB", (32, 32)))["predicted_count"], 0)


if __name__ == "__main__":
    unittest.main()
