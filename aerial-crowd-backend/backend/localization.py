"""Conservative sparse-scene counting from separated density responses."""
import math

import numpy as np
from scipy.ndimage import label


def component_masses(density):
    density = np.asarray(density, dtype=np.float32)
    if density.ndim != 2 or not np.isfinite(density).all():
        raise ValueError("Expected a finite 2-D density map")
    if density.size == 0 or density.max() <= 0:
        return np.empty(0, dtype=np.float64)
    mask = density > max(float(density.max()) * 0.005, 1e-5)
    regions, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.empty(0, dtype=np.float64)
    return np.bincount(regions.ravel(), weights=density.ravel())[1:]


def choose_count(raw_count, masses, policy):
    rounded = max(0, math.floor(raw_count + 0.5))
    if policy is None:
        return rounded, "density_integral"
    masses = np.asarray(masses)
    localized = int(np.count_nonzero(masses >= policy["mass_threshold"]))
    strong = int(np.count_nonzero(masses >= 0.5))
    if (policy["raw_min"] <= raw_count <= policy["raw_max"]
            and strong >= 2 and abs(localized - rounded) <= 1
            and masses.max(initial=0) <= policy.get("max_component_mass", np.inf)):
        return localized, "localized_components"
    return rounded, "density_integral"
