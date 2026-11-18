import unittest

import numpy as np
import torch
from PIL import Image

from detection import HybridCounterService, density_coverage, detection_scene_check


def people():
    return [{"box": box, "score": 0.99} for box in
            ([40, 40, 90, 120], [104, 104, 150, 184], [168, 168, 220, 248])]


class FakeCounter:
    device = torch.device("cpu")
    count_policy = None

    def predict(self, image, return_density=False):
        assert return_density
        density = np.zeros((32, 32), dtype=np.float32)
        density[8, 8] = density[16, 16] = density[24, 24] = 0.8
        return {"predicted_count": 2, "raw_count": 2.4, "estimated_count": 2.4,
                "count_method": "density_integral", "density_array": density,
                "density_map": Image.new("RGB", image.size), "tiles_processed": 1}


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image):
        return self.detections


class DetectionTests(unittest.TestCase):
    def test_large_people_are_eligible(self):
        self.assertEqual(detection_scene_check(people(), (256, 256), 2.4), (True, "large_visible_people"))

    def test_tiny_people_preserve_aerial_counting(self):
        small = [{"box": [10, 10, 20, 30], "score": 0.99}]
        self.assertEqual(detection_scene_check(small, (1600, 900), 1)[1], "tiny_or_distant_people")

    def test_original_pixels_do_not_hide_detector_downscaling(self):
        detections = [{"box": [100, 100, 200, 300], "score": 0.99}]
        self.assertFalse(detection_scene_check(detections, (8000, 4500), 1)[0])

    def test_partial_crowd_detections_are_rejected(self):
        self.assertEqual(detection_scene_check(people(), (256, 256), 50)[1], "models_disagree")

    def test_output_saturation_is_rejected(self):
        self.assertEqual(detection_scene_check(people() * 34, (256, 256), 102)[1], "detector_output_limit")

    def test_unexplained_density_is_measured(self):
        density = np.zeros((32, 32), dtype=np.float32)
        density[2, 2] = density[30, 30] = 1
        self.assertEqual(density_coverage(density, [{"box": [0, 0, 32, 32]}]), 0.5)

    def test_handoff_returns_box_count_and_preserves_raw_estimate(self):
        image = Image.new("RGB", (256, 256), "white")
        result = HybridCounterService(FakeCounter(), FakeDetector(people())).predict(image)
        self.assertEqual(result["predicted_count"], 3)
        self.assertEqual(result["density_predicted_count"], 2)
        self.assertEqual(result["raw_count"], 2.4)
        self.assertEqual(result["count_method"], "person_detection")
        self.assertEqual(len(result["detections"]), 3)
        self.assertEqual(result["detection_overlay"].size, image.size)
        self.assertNotIn("density_array", result)
        self.assertEqual(image.getpixel((40, 40)), (255, 255, 255))

    def test_uncertain_scene_keeps_existing_prediction(self):
        result = HybridCounterService(FakeCounter(), FakeDetector([])).predict(Image.new("RGB", (256, 256)))
        self.assertEqual(result["predicted_count"], 2)
        self.assertEqual(result["detections"], [])
        self.assertNotIn("detection_overlay", result)

    def test_explicit_detection_mode_can_return_zero(self):
        result = HybridCounterService(FakeCounter(), FakeDetector([]), "detection").predict(Image.new("RGB", (256, 256)))
        self.assertEqual(result["predicted_count"], 0)
        self.assertEqual(result["count_method"], "person_detection")


if __name__ == "__main__":
    unittest.main()
