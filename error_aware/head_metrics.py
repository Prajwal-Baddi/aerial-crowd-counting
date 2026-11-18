"""Pure-NumPy metrics for the 3-way ErrorHead."""
import numpy as np
from selection_baselines import DELTA, types, topk_mask


def average_precision(score, label):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=bool).reshape(-1)
    if score.shape != label.shape:
        raise ValueError(f"score/label shape mismatch: {score.shape} vs {label.shape}")
    positives = int(label.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = label[order].astype(np.float64)
    tp = np.cumsum(y)
    ranks = np.arange(1, len(y) + 1, dtype=np.float64)
    return float(((tp / ranks) * y).sum() / positives)


def class_distribution(C, G, delta=DELTA):
    """Cell- and crop-level class balance; C and G have shape [N, cells]."""
    C, G = np.asarray(C, dtype=np.float64), np.asarray(G, dtype=np.float64)
    if C.shape != G.shape or C.ndim != 2:
        raise ValueError(f"Expected matching [N,cells] arrays; got {C.shape}, {G.shape}")
    err = C - G
    inc, dec = err <= -delta, err >= delta
    fine = ~(inc | dec)
    inc_crop, dec_crop = inc.any(1), dec.any(1)
    both = inc_crop & dec_crop
    total = err.size
    n_inc, n_dec, n_fine = int(inc.sum()), int(dec.sum()), int(fine.sum())
    return {
        "crops": int(C.shape[0]), "cells_per_crop": int(C.shape[1]),
        "total_cells": int(total), "fine_cells": n_fine,
        "increase_cells": n_inc, "decrease_cells": n_dec,
        "material_cells": n_inc + n_dec,
        "fine_pct": 100.0 * n_fine / max(total, 1),
        "increase_pct": 100.0 * n_inc / max(total, 1),
        "decrease_pct": 100.0 * n_dec / max(total, 1),
        "material_pct": 100.0 * (n_inc + n_dec) / max(total, 1),
        "increase_to_decrease_ratio": n_inc / n_dec if n_dec else None,
        "crops_with_increase": int(inc_crop.sum()),
        "crops_with_decrease": int(dec_crop.sum()),
        "crops_with_any_material_error": int((inc_crop | dec_crop).sum()),
        "crops_with_both_directions": int(both.sum()),
        "increase_only_crops": int((inc_crop & ~dec_crop).sum()),
        "decrease_only_crops": int((dec_crop & ~inc_crop).sum()),
        "neither_direction_crops": int((~(inc_crop | dec_crop)).sum()),
    }


def print_class_distribution(r, title):
    print(f"\\n{title}\\n" + "-" * 72)
    print(f"Crops: {r['crops']:,} | Cells: {r['total_cells']:,}")
    print(f"{'Class':<18}{'Cells':>14}{'Share':>12}")
    print("-" * 44)
    for name, count, pct in (
        ("Fine", r["fine_cells"], r["fine_pct"]),
        ("Increase", r["increase_cells"], r["increase_pct"]),
        ("Decrease", r["decrease_cells"], r["decrease_pct"]),
        ("Material total", r["material_cells"], r["material_pct"]),
    ):
        print(f"{name:<18}{count:>14,}{pct:>11.3f}%")
    ratio = r["increase_to_decrease_ratio"]
    print("Increase/decrease cell ratio:", "undefined" if ratio is None else f"{ratio:.4f}")
    print("Crop counts:")
    for key, label in (
        ("crops_with_increase", "with increase errors"),
        ("crops_with_decrease", "with decrease errors"),
        ("crops_with_any_material_error", "with either direction"),
        ("crops_with_both_directions", "with both directions"),
        ("increase_only_crops", "increase only"),
        ("decrease_only_crops", "decrease only"),
        ("neither_direction_crops", "neither direction"),
    ):
        print(f"  {label:<27}{r[key]:>8,}")


def head_report(C, G, P, budgets=(3, 6, 13, 26), seed=0):
    """C,G: [N,256]; P: [N,3,256] probabilities."""
    C, G, P = np.asarray(C), np.asarray(G), np.asarray(P)
    if C.shape != G.shape:
        raise ValueError(f"C/G shape mismatch: {C.shape} vs {G.shape}")
    if P.ndim != 3 or P.shape[1] != 3 or P.shape[0] != C.shape[0] or P.shape[2] != C.shape[1]:
        raise ValueError(f"Incompatible probability shape {P.shape} for C shape {C.shape}")

    err = C - G
    inc, dec = err <= -DELTA, err >= DELTA
    material = inc | dec
    ap = lambda s, y: average_precision(s, y)

    # Direction-aware baselines: low density for under-counting; high density
    # for over-counting. Material errors use absolute coarse-vs-GT error.
    result = {
        "base_rate": float(material.mean()) if material.size else 0.0,
        "class_distribution": class_distribution(C, G),
        "ap": {
            "material": {"head": ap(1.0 - P[:, 0], material), "density": ap(np.abs(err), material)},
            "increase": {"head": ap(P[:, 1], inc), "density": ap(-C, inc)},
            "decrease": {"head": ap(P[:, 2], dec), "density": ap(C, dec)},
        },
        "coverage": [],
    }
    rng = np.random.default_rng(seed)
    error_types, abs_err = types(C, G), np.abs(err)
    scores = {"head": 1.0 - P[:, 0], "density": np.abs(C), "oracle": abs_err}
    for k in budgets:
        for name, score in scores.items():
            mask = topk_mask(score, k, rng)
            result["coverage"].append({
                "k": int(k), "rule": name,
                "precision": float(material[mask].mean()) if mask.any() else 0.0,
                "abs_err_share": float(abs_err[mask].sum() / abs_err.sum()) if abs_err.sum() else 0.0,
                **{f"recall_{t}": float(v[mask].sum() / max(v.sum(), 1)) for t, v in error_types.items()},
            })
    return result


def print_head_report(r):
    print(f"Base rate of material-error cells: {100*r['base_rate']:.2f}%")
    print(f"{'average precision':<22}{'head':>8}{'baseline':>10}")
    for name, values in r["ap"].items():
        print(f"  {name:<20}{values['head']:>8.3f}{values['density']:>10.3f}")
    print(f"\\n{'cells/crop':>10} {'rule':<9}{'precision':>10}{'abs err':>9}{'FP_empty':>10}{'miss':>8}{'under':>8}{'over':>8}")
    for row in r["coverage"]:
        print(
            f"{row['k']:>10} {row['rule']:<9}"
            f"{100*row['precision']:>9.1f}%{100*row['abs_err_share']:>8.1f}%"
            f"{100*row.get('recall_FP_empty', 0):>9.1f}%"
            f"{100*row.get('recall_miss', 0):>7.1f}%"
            f"{100*row.get('recall_under', 0):>7.1f}%"
            f"{100*row.get('recall_over', 0):>7.1f}%"
        )
