"""Where does Stage 1 make errors, and how much could each correction mechanism recover?

    python error_aware/analyze_error_budget.py `
      --manifest error_aware/processed/stage2_train20_holdout.jsonl `
      --checkpoint error_aware/checkpoints_stage1_holdout80/coarse_best.pth `
      --output-dir error_aware/analysis

Only the frozen Stage-1 counter is run. Cell arrays are cached to cells.npz so the analysis can be re-run instantly.
"""
import argparse, json, zlib
from pathlib import Path
import numpy as np

DELTA = 0.25      # a cell error counts as "material" above this many people
EDGES = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, np.inf])


def crop_stats(pred, gt):
    e, empty = np.abs(pred - gt), gt == 0
    f = lambda a: float(a.mean()) if a.size else float("nan")
    return {"mae": f(e), "empty_mae": f(e[empty]), "nonempty_mae": f(e[~empty])}


def analyze(coarse, gtc, counts, names):
    """coarse, gtc: [N,256] cell counts; counts: [N] true crop counts; names: image id per crop."""
    err = coarse - gtc
    pred_crop = coarse.sum(1)
    base = crop_stats(pred_crop, counts)
    cats = {
        "FP_empty": (gtc < 0.1) & (err >= DELTA),
        "miss":     (gtc >= 0.5) & (coarse < 0.1) & (err <= -DELTA),
        "under":    (gtc >= 0.1) & (coarse >= 0.1) & (err <= -DELTA),
        "over":     (gtc >= 0.1) & (err >= DELTA),
    }
    material = np.abs(err) >= DELTA
    tot_abs = np.abs(err).sum()
    out = {"baseline": base, "n_crops": int(len(counts)), "n_cells": int(err.size),
           "material_cells": int(material.sum()), "categories": {}}
    for k, m in cats.items():
        fixed = pred_crop - (err * m).sum(1)           # replace those cells by ground truth
        s = crop_stats(fixed, counts)
        out["categories"][k] = {
            "cells": int(m.sum()), "share_of_material_cells": float(m.sum() / max(material.sum(), 1)),
            "share_of_total_abs_cell_error": float(np.abs(err[m]).sum() / tot_abs),
            "crop_mae_if_fixed": s["mae"], "nonempty_mae_if_fixed": s["nonempty_mae"],
            "empty_mae_if_fixed": s["empty_mae"],
            "mae_recovered_pct": 100 * (base["mae"] - s["mae"]) / base["mae"]}
    # is GAIN_MAX = 3 enough for the 'under' cells?
    u = cats["under"]
    ratio = gtc[u] / np.maximum(coarse[u], 1e-6)
    out["under_gt_over_coarse_ratio"] = {"p25": float(np.percentile(ratio, 25)), "median": float(np.median(ratio)),
                                         "p75": float(np.percentile(ratio, 75)), "p95": float(np.percentile(ratio, 95)),
                                         "frac_above_3": float((ratio > 3).mean())} if u.any() else None
    # calibration baseline: per-bin ratio sum(gt)/sum(coarse), fit on half the images, tested on the other half
    half = np.array([zlib.crc32(n.encode()) % 2 for n in names])
    fit, ev = half == 0, half == 1
    bins = np.digitize(coarse, EDGES[1:-1])
    ratios = np.ones(len(EDGES) - 1)
    for b in range(len(ratios)):
        sel = fit[:, None] & (bins == b)
        if coarse[sel].sum() > 1e-6:
            ratios[b] = gtc[sel].sum() / coarse[sel].sum()
    cal = coarse * ratios[bins]
    out["calibration"] = {"bin_edges": EDGES[:-1].tolist(), "ratios": ratios.tolist(),
                          "eval_before": crop_stats(pred_crop[ev], counts[ev]),
                          "eval_after": crop_stats(cal.sum(1)[ev], counts[ev])}
    return out


def print_report(r):
    b = r["baseline"]
    print(f"\nStage-1 on {r['n_crops']} crops: MAE {b['mae']:.4f} | empty {b['empty_mae']:.4f} | non-empty {b['nonempty_mae']:.4f}")
    print(f"Material cell errors (|err| >= {DELTA}): {r['material_cells']} of {r['n_cells']} cells\n")
    print(f"{'type':<10}{'cells':>8}{'%material':>11}{'%abs err':>10}{'MAE if fixed':>14}{'recovered':>11}")
    for k, v in r["categories"].items():
        print(f"{k:<10}{v['cells']:>8}{100*v['share_of_material_cells']:>10.1f}%{100*v['share_of_total_abs_cell_error']:>9.1f}%"
              f"{v['crop_mae_if_fixed']:>14.4f}{v['mae_recovered_pct']:>10.1f}%")
    g = r["under_gt_over_coarse_ratio"]
    if g:
        print(f"\n'under' cells, GT/coarse ratio: median {g['median']:.2f}, p75 {g['p75']:.2f}, p95 {g['p95']:.2f}; "
              f"{100*g['frac_above_3']:.1f}% need more than 3x (GAIN_MAX=3)")
    c = r["calibration"]
    print(f"\nCalibration baseline (held-out images): MAE {c['eval_before']['mae']:.4f} -> {c['eval_after']['mae']:.4f} | "
          f"non-empty {c['eval_before']['nonempty_mae']:.4f} -> {c['eval_after']['nonempty_mae']:.4f} | "
          f"empty {c['eval_before']['empty_mae']:.4f} -> {c['eval_after']['empty_mae']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", default="dataset/VisDrone2019-DET-train")
    ap.add_argument("--output-dir", default="error_aware/analysis")
    ap.add_argument("--max-crops", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    cache = out / "cells.npz"
    if cache.exists() and a.max_crops is None:
        z = np.load(cache, allow_pickle=False)
        coarse, gtc, counts, names = z["coarse"], z["gt"], z["counts"], z["names"].tolist()
        print(f"Loaded cached cells: {cache}")
    else:
        import torch, torch.nn.functional as F
        from torch.utils.data import DataLoader
        from data import ErrorAwareCrowdDataset
        from model_multiplicative import CoarseCounter
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CoarseCounter(pretrained=False, freeze_early=True).to(dev)
        ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
        model.load_state_dict(ck.get("model_state_dict", ck), strict=True); model.eval()
        ds = ErrorAwareCrowdDataset(Path(a.manifest), dataset_root=a.dataset_root, crop_size=512,
                                    density_stride=8, gaussian_sigma=3.0, return_metadata=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        cell = lambda d: (F.avg_pool2d(d.float(), 4) * 16).flatten(1).cpu().numpy()
        C, G, N, names = [], [], [], []
        with torch.no_grad():
            for i, (img, den, cnt, meta) in enumerate(loader):
                if a.max_crops and i >= a.max_crops:
                    break
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    d = model(img.to(dev))["density"]
                C.append(cell(d)); G.append(cell(den.to(dev))); N.append(float(cnt)); names.append(str(meta["image"][0]))
                if (i + 1) % 1000 == 0:
                    print(f"  {i + 1} crops")
        coarse, gtc, counts = np.concatenate(C), np.concatenate(G), np.array(N)
        if a.max_crops is None:
            np.savez_compressed(cache, coarse=coarse, gt=gtc, counts=counts, names=np.array(names))
    res = analyze(coarse, gtc, counts, names)
    print_report(res)
    json.dump(res, open(out / "error_budget.json", "w"), indent=2)
    print(f"\nSaved {out / 'error_budget.json'}")


if __name__ == "__main__":
    main()