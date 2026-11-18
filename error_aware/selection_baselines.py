"""Which cells do simple selection rules catch? Uses the cached cells.npz (no GPU, seconds).

    python error_aware/selection_baselines.py --cells error_aware/analysis/cells.npz

For each budget (cells per crop) and rule, prints precision (share of selected cells with a material error)
and recall of material cells BY TYPE. A learned ErrorHead has to beat the best simple rule, especially on 'miss'.
"""
import argparse
import numpy as np

DELTA = 0.25


def types(coarse, gt):
    err = coarse - gt
    return {"FP_empty": (gt < 0.1) & (err >= DELTA),
            "miss": (gt >= 0.5) & (coarse < 0.1) & (err <= -DELTA),
            "under": (gt >= 0.1) & (coarse >= 0.1) & (err <= -DELTA),
            "over": (gt >= 0.1) & (err >= DELTA)}


def neighbour_max(coarse):
    """Max coarse count in each cell's 3x3 neighbourhood (finds cells next to detected people)."""
    n = coarse.shape[0]
    g = np.pad(coarse.reshape(n, 16, 16), ((0, 0), (1, 1), (1, 1)))
    stack = [g[:, i:i + 16, j:j + 16] for i in range(3) for j in range(3)]
    return np.max(stack, axis=0).reshape(n, 256)


def topk_mask(score, k, rng):
    score = score + rng.random(score.shape) * 1e-9            # random tie-breaking (many cells are exactly equal)
    idx = np.argpartition(-score, k - 1, axis=1)[:, :k]
    m = np.zeros(score.shape, bool)
    np.put_along_axis(m, idx, True, axis=1)
    return m


def run(coarse, gt, budgets=(3, 6, 13, 26), seed=0):
    rng = np.random.default_rng(seed)
    err = np.abs(coarse - gt)
    ty = types(coarse, gt)
    material = np.abs(coarse - gt) >= DELTA
    scores = {"random": rng.random(coarse.shape), "density": coarse,
              "density+nbr": np.maximum(coarse, 0.5 * neighbour_max(coarse)), "oracle": err}
    rows = []
    for k in budgets:
        for name, s in scores.items():
            m = topk_mask(s, k, rng)
            rows.append({"k": k, "rule": name,
                         "precision": float(material[m].mean()),
                         "abs_err_share": float(err[m].sum() / err.sum()),
                         **{f"recall_{t}": float(v[m].sum() / max(v.sum(), 1)) for t, v in ty.items()}})
    return rows, float(material.mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", required=True)
    a = ap.parse_args()
    z = np.load(a.cells, allow_pickle=False)
    rows, base = run(z["coarse"], z["gt"])
    print(f"Base rate of material-error cells: {100 * base:.2f}% (precision of random selection)\n")
    print(f"{'cells/crop':>10} {'rule':<12}{'precision':>10}{'abs err':>9}{'FP_empty':>10}{'miss':>8}{'under':>8}{'over':>8}")
    for r in rows:
        print(f"{r['k']:>10} {r['rule']:<12}{100*r['precision']:>9.1f}%{100*r['abs_err_share']:>8.1f}%"
              f"{100*r['recall_FP_empty']:>9.1f}%{100*r['recall_miss']:>7.1f}%{100*r['recall_under']:>7.1f}%{100*r['recall_over']:>7.1f}%")
    print("\n(cells/crop 3, 6, 13, 26 = about 1%, 2.5%, 5%, 10% of the 256 cells in a crop)")


if __name__ == "__main__":
    main()