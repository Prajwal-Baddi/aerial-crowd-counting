import unittest

import numpy as np

from localization import component_masses, choose_count


class LocalizationTests(unittest.TestCase):
    policy = {"mass_threshold": 0.5, "raw_min": 3.0, "raw_max": 12.0}

    def test_counts_separated_people_once_and_rejects_weak_artifact(self):
        density = np.zeros((24, 24), dtype=np.float32)
        for y, x in ((2, 2), (2, 12), (12, 2), (12, 12), (20, 20)):
            density[y, x] = 1.18
        density[20, 2] = 0.17
        raw = float(density.sum())
        self.assertEqual(round(raw), 6)
        self.assertEqual(choose_count(raw, component_masses(density), self.policy), (5, "localized_components"))

    def test_preserves_low_mass_density_estimate(self):
        self.assertEqual(choose_count(2.51, [0.31, 1.1, 1.1], self.policy), (3, "density_integral"))

    def test_large_disagreement_falls_back_to_integral(self):
        self.assertEqual(choose_count(6.0, [3.2, 2.8], self.policy), (6, "density_integral"))

    def test_diagonal_fragments_form_one_component(self):
        density = np.zeros((8, 8), dtype=np.float32)
        density[2, 2] = density[3, 3] = 0.6
        masses = component_masses(density)
        self.assertEqual(len(masses), 1)
        self.assertAlmostEqual(masses[0], 1.2, places=5)

    def test_merged_crowd_response_is_not_counted_as_one_person(self):
        policy = self.policy | {"max_component_mass": 1.5}
        self.assertEqual(choose_count(5.9, [1.9, 1.1, 0.9, 1.1, 0.9], policy), (6, "density_integral"))

    def test_empty_density_remains_zero(self):
        density = np.zeros((8, 8), dtype=np.float32)
        self.assertEqual(choose_count(0, component_masses(density), self.policy), (0, "density_integral"))


if __name__ == "__main__":
    unittest.main()
