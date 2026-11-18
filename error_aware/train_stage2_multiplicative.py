"""Stage 2C: train multiplicative-additive RefineNet on the frozen Stage-1 counter.

    python error_aware/train_stage2_multiplicative.py --smoke-test
    python error_aware/train_stage2_multiplicative.py --epochs 30

Uses model_multiplicative.py and refine_losses_multiplicative.py next to this file.
"""
import argparse, json, math, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ErrorAwareCrowdDataset, get_loader
from model_multiplicative import ESRNet, cell_sum, select_topk
from refine_losses_multiplicative import build_training_mask, refinement_loss


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description="Stage-2C multiplicative-additive refinement training.")
    p.add_argument("--data-dir", default="error_aware/processed")
    p.add_argument("--dataset-root", default="dataset/VisDrone2019-DET-train")
    p.add_argument("--train-manifest", default=None,
                   help="Optional jsonl of crops for Stage 2 (e.g. a split the counter was NOT trained on).")
    p.add_argument("--checkpoint", default="error_aware/checkpoints/coarse_best.pth")
    p.add_argument("--output-dir", default="error_aware/checkpoints_stage2_multiplicative")
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4, help="Coarse net is frozen/no_grad: bs 4 should fit; use 2 or 1 if not.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refine-width", type=int, default=96)
    # selection during training
    p.add_argument("--budget-min", type=float, default=0.10)
    p.add_argument("--budget-max", type=float, default=0.40)
    p.add_argument("--background-frac", type=float, default=0.25,
                   help="Extra random (mostly easy/empty) cells refined in training so the refiner learns to do nothing.")
    # loss weights
    p.add_argument("--w-err", type=float, default=1.0)
    p.add_argument("--w-rank", type=float, default=0.5)
    p.add_argument("--w-cell", type=float, default=1.0)
    p.add_argument("--w-pix", type=float, default=1.0)
    p.add_argument("--w-harm", type=float, default=0.5)
    p.add_argument("--w-empty", type=float, default=3.0)
    p.add_argument("--w-add", type=float, default=0.1)
    p.add_argument("--w-glob", type=float, default=0.05)
    # validation / model selection
    p.add_argument("--val-budget", type=float, default=0.25)
    p.add_argument("--taus", default="0,0.05,0.1,0.2,0.4", help="Predicted-error thresholds swept at validation.")
    p.add_argument("--empty-tol", type=float, default=0.05,
                   help="Max allowed rise in empty-crop MAE over the coarse model for a config to be eligible.")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--gap-check-batches", type=int, default=50)
    # misc
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--amp-init-scale", type=float, default=128.0)
    p.add_argument("--debug-grads", action="store_true")
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-validation-batches", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- setup helpers
def load_coarse_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.coarse.load_state_dict(state, strict=True)
    for prm in model.coarse.parameters():
        prm.requires_grad_(False)
    model.coarse.eval()
    print(f"Loaded Stage-1 checkpoint strictly: {path}  (coarse frozen)")


def make_train_loader(args):
    if args.train_manifest is None:
        return get_loader(data_dir=args.data_dir, split="train", batch_size=args.batch_size,
                          num_workers=args.num_workers, dataset_root=args.dataset_root)
    ds = ErrorAwareCrowdDataset(Path(args.train_manifest), dataset_root=args.dataset_root,
                                crop_size=512, density_stride=8, gaussian_sigma=3.0)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=args.num_workers > 0)


def true_cell_error(coarse_density, target_density):
    """Error-head target: |predicted - true| count in every 32x32 px cell."""
    return (cell_sum(coarse_density.detach().float()) - cell_sum(target_density.float())).abs()


def error_loss(pred, target):
    """Cell errors are tiny and heavy-tailed (most crops hold ~1 person). A log-compressed L1 keeps
    the small-error regime informative; plain SmoothL1(beta=1) is almost flat there."""
    t = lambda x: torch.log1p(10.0 * x)
    return F.smooth_l1_loss(pred, target, beta=0.1) + F.l1_loss(t(pred), t(target))


def ranking_loss(pred, target, pairs=1024, margin_ignore=0.05):
    p, t = pred.flatten(1), target.flatten(1)
    B, N = p.shape
    if N < 2:
        return p.sum() * 0.0
    i = torch.randint(0, N, (B, pairs), device=p.device)
    j = torch.randint(0, N, (B, pairs), device=p.device)
    d = t.gather(1, i) - t.gather(1, j)
    valid = d.abs() >= margin_ignore
    if not valid.any():
        return p.sum() * 0.0
    return F.softplus(-(d.sign() * (p.gather(1, i) - p.gather(1, j)))[valid]).mean()


# ----------------------------------------------------------------------------- sanity checks
@torch.no_grad()
def coarse_mae(model, loader, device, amp, n_batches):
    model.eval(); errs = []
    for k, (img, _, cnt) in enumerate(loader):
        if k >= n_batches:
            break
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            d = model.coarse(img.to(device))["density"]
        errs += (d.float().sum((1, 2, 3)).cpu() - cnt.float()).abs().tolist()
    return float(np.mean(errs)) if errs else float("nan")


@torch.no_grad()
def identity_check(model, loader, device, amp):
    """A freshly initialised refiner must reproduce the coarse map (gain=1, add~0)."""
    model.eval()
    img, _, _ = next(iter(loader))
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        c = model.coarse(img.to(device))
        d_c = c["density"].float()
        mask = torch.ones(d_c.shape[0], 1, 16, 16, dtype=torch.bool, device=device)
        final = model.apply_refine(d_c, c["stride4_features"], mask)
    diff = (final.float() - d_c).abs().max().item()
    dc = (final.float().sum() - d_c.sum()).item() / d_c.shape[0]
    identity_ok = diff < 1e-4 and abs(dc) < 1e-2
    print(f"Identity check: max|refined-coarse| = {diff:.2e}, mean count change = {dc:+.4f} "
          f"({'OK' if identity_ok else 'WARNING: initial additive mass is not negligible'})")
    if not identity_ok:
        raise RuntimeError(
            "Identity initialization failed: reduce ADD_BIAS_INIT or inspect gain/add scaling."
        )


# ----------------------------------------------------------------------------- metrics
def _avg_rank(x):
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    start = np.cumsum(cnt) - cnt
    return (start + (cnt - 1) / 2.0)[inv]


def spearman(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(_avg_rank(a), _avg_rank(b))[0, 1])


def summarize(pred, gt, coarse):
    err, empty = pred - gt, gt == 0
    d = pred - coarse
    f = lambda v: float(v) if v is not None else None
    return {
        "mae": f(np.abs(err).mean()), "rmse": f(np.sqrt((err ** 2).mean())),
        "empty_mae": f(np.abs(err[empty]).mean()) if empty.any() else None,
        "nonempty_mae": f(np.abs(err[~empty]).mean()) if (~empty).any() else None,
        "mean_delta": f(d.mean()),
        "empty_delta": f(d[empty].mean()) if empty.any() else None,
    }


@torch.no_grad()
def validate(model, loader, device, args, amp, taus, max_batches=None):
    """Coarse baseline vs refined (predicted-error selection at several thresholds) vs density / random /
    oracle selection, plus error-localisation quality. All at the same budget."""
    model.eval(); model.coarse.eval()
    configs = [("pred", t) for t in taus] + [("oracle", 0.0), ("density", 0.0), ("random", 0.0)]
    name = lambda m, t: f"pred@tau{t:g}" if m == "pred" else m
    gt, coarse = [], []
    refined = {name(m, t): [] for m, t in configs}
    area = {name(m, t): [] for m, t in configs}
    cover = {m: [] for m in ("pred", "oracle", "density", "random")}
    sp = []
    for step, (img, gt_den, cnt) in enumerate(loader, 1):
        if max_batches is not None and step > max_batches:
            break
        img, gt_den = img.to(device, non_blocking=True), gt_den.to(device, non_blocking=True).float()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            c = model.coarse(img)
            e_pred = model.error_head(c["context"], c["density"])
        d_c, f4, e_pred = c["density"].float(), c["stride4_features"], e_pred.float()
        e_true = true_cell_error(d_c, gt_den)
        scores = {"pred": e_pred, "oracle": e_true, "density": cell_sum(d_c), "random": torch.rand_like(e_pred)}
        gt += cnt.float().tolist(); coarse += d_c.sum((1, 2, 3)).cpu().tolist()
        for b in range(img.shape[0]):
            et, ep = e_true[b].flatten().cpu().numpy(), e_pred[b].flatten().cpu().numpy()
            r = spearman(ep, et)
            if r is not None:
                sp.append(r)
        for m, t in configs:
            mask = select_topk(scores[m], args.val_budget, min_score=t if t > 0 else None)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                fin = model.apply_refine(d_c, f4, mask)
            k = name(m, t)
            refined[k] += fin.float().sum((1, 2, 3)).cpu().tolist()
            area[k] += mask.float().mean((1, 2, 3)).cpu().tolist()
            if t == 0:   # coverage: share of total coarse error lying inside the selected cells
                tot = e_true.sum((1, 2, 3))
                cov = ((e_true * mask.float()).sum((1, 2, 3)) / tot.clamp(min=1e-6))[tot > 1e-4]
                cover[m] += cov.cpu().tolist()
    gt, coarse = np.array(gt), np.array(coarse)
    res = {"samples": len(gt), "coarse": summarize(coarse, gt, coarse), "configs": {},
           "spearman": float(np.mean(sp)) if sp else None,
           "coverage": {m: (float(np.mean(v)) if v else None) for m, v in cover.items()}}
    for k, v in refined.items():
        res["configs"][k] = {**summarize(np.array(v), gt, coarse), "area": float(np.mean(area[k]))}
    return res


def choose_config(res, empty_tol):
    """Best predicted-error config that does not hurt empty crops; else the most conservative one."""
    base_empty = res["coarse"]["empty_mae"]
    pred = {k: v for k, v in res["configs"].items() if k.startswith("pred@")}
    ok = {k: v for k, v in pred.items()
          if base_empty is None or v["empty_mae"] is None or v["empty_mae"] <= base_empty + empty_tol}
    if ok:
        k = min(ok, key=lambda x: ok[x]["mae"])
        return k, float(k.split("tau")[1]), ok[k]["mae"], True
    k = max(pred, key=lambda x: float(x.split("tau")[1]))
    return k, float(k.split("tau")[1]), pred[k]["mae"], False


def print_validation(res, chosen):
    c = res["coarse"]
    fmt = lambda v: "  n/a " if v is None else f"{v:6.3f}"
    print(f"  {'config':<14}{'MAE':>7}{'empty':>8}{'nonempty':>9}{'dCount':>8}{'area':>7}")
    print(f"  {'coarse':<14}{fmt(c['mae'])} {fmt(c['empty_mae'])}  {fmt(c['nonempty_mae'])}  {'':>6}  {'':>5}")
    for k, v in res["configs"].items():
        star = " <-" if k == chosen else ""
        print(f"  {k:<14}{fmt(v['mae'])} {fmt(v['empty_mae'])}  {fmt(v['nonempty_mae'])} {v['mean_delta']:+7.3f} {v['area']:6.2f}{star}")
    cov = {m: (f"{v:.2f}" if v is not None else "n/a") for m, v in res["coverage"].items()}
    sp = res["spearman"]
    print(f"  Spearman(pred,true cell error) = {('n/a' if sp is None else f'{sp:.3f}')} | "
          f"error coverage@{res['configs'][chosen]['area']:.2f}-ish budget: {cov}")


# ----------------------------------------------------------------------------- training
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, args, amp):
    model.train(); model.coarse.eval()
    p_true = 0.8 - 0.6 * (epoch - 1) / max(args.epochs - 1, 1)     # scheduled sampling 0.8 -> 0.2
    keys = ["total", "err", "rank", "cell", "pix", "harm", "add", "glob", "gain", "add_mass", "dcount"]
    sums = {k: torch.zeros((), device=device) for k in keys}
    n = skipped = 0
    t0 = time.time()
    for step, (img, gt_den, _) in enumerate(loader, 1):
        if args.max_train_batches is not None and step > args.max_train_batches:
            break
        img, gt_den = img.to(device, non_blocking=True), gt_den.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            c = model.coarse(img)                                  # frozen: no graph, no activations kept
        d_c, ctx, f4 = c["density"].float(), c["context"], c["stride4_features"]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            e_pred = model.error_head(ctx, d_c)
        e_pred = e_pred.float()
        e_true = true_cell_error(d_c, gt_den)

        mask = build_training_mask(e_true, e_pred.detach(), random.uniform(args.budget_min, args.budget_max),
                                   p_true, args.background_frac)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            final, aux = model.apply_refine(d_c, f4, mask, return_aux=True)

        l_err, l_rank = error_loss(e_pred, e_true), ranking_loss(e_pred, e_true)
        l_ref, parts = refinement_loss(final, d_c, gt_den, mask, aux, w_cell=args.w_cell, w_pix=args.w_pix,
                                       w_harm=args.w_harm, w_empty=args.w_empty, w_add=args.w_add,
                                       w_glob=args.w_glob)
        loss = args.w_err * l_err + args.w_rank * l_rank + l_ref
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        params = list(model.error_head.parameters()) + list(model.refiner.parameters())
        gnorm = torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        after = scaler.get_scale()
        if after < before:                            # AMP skipped this update
            skipped += 1
            print(f"WARNING: AMP skipped update (epoch {epoch}, step {step}, grad norm {float(gnorm):.3g}, "
                  f"scale {before} -> {after}); scheduler not advanced")
        else:
            scheduler.step()

        with torch.no_grad():
            vals = {"total": loss, "err": l_err, "rank": l_rank, **parts,
                    "gain": aux["gain"].float().mean() if aux else torch.zeros((), device=device),
                    "add_mass": aux["add"].float().sum((1, 2, 3)).mean() if aux else torch.zeros((), device=device),
                    "dcount": (final.float().sum((1, 2, 3)) - d_c.sum((1, 2, 3))).mean()}
            for k in keys:
                sums[k] += vals[k].detach().float()
        n += 1
        if step == 1 or step % args.log_interval == 0:
            print(f"Epoch {epoch:02d} Step {step:04d} | loss {loss.item():.4f} | err {l_err.item():.4f} | "
                  f"rank {l_rank.item():.3f} | cell {parts['cell'].item():.4f} | harm {parts['harm'].item():.4f} | "
                  f"dCount {vals['dcount'].item():+.3f} | gain {vals['gain'].item():.3f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} | teacher {p_true:.2f}")
    avg = {k: (v / max(n, 1)).item() for k, v in sums.items()}
    return avg, time.time() - t0, n, skipped


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_score, best_tau, args, extra=None):
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    payload = {"epoch": epoch, "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
               "best_val_mae": best_score, "best_tau": best_tau, "val_budget": args.val_budget,
               "args": vars(args), **(extra or {})}
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device} | AMP: {amp}")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    taus = sorted(set([0.0] + [float(t) for t in args.taus.split(",") if t.strip()]))

    model = ESRNet(pretrained=False, freeze_early=True, refine_width=args.refine_width).to(device)
    load_coarse_checkpoint(model, args.checkpoint, device)
    train_loader = make_train_loader(args)
    val_loader = get_loader(data_dir=args.data_dir, split="validation", batch_size=1, shuffle=False,
                            num_workers=args.num_workers, dataset_root=args.dataset_root)

    if args.gap_check_batches > 0 and not args.smoke_test:
        tr = coarse_mae(model, train_loader, device, amp, args.gap_check_batches)
        va = coarse_mae(model, val_loader, device, amp, args.gap_check_batches)
        print(f"Coarse MAE: train crops {tr:.3f} | validation crops {va:.3f}")
        if tr < 0.7 * va:
            print("WARNING: the counter is much better on Stage-2 training crops than on validation crops. "
                  "The error head will under-estimate real errors. Prefer --train-manifest with crops the "
                  "Stage-1 model was not trained on.")

    optimizer = torch.optim.AdamW(list(model.error_head.parameters()) + list(model.refiner.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    epochs = 1 if args.smoke_test else args.epochs
    steps_per_epoch = min(len(train_loader), args.max_train_batches or len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch)
    lr_fn = lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)) * \
        max(0.01, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler = torch.amp.GradScaler("cuda", init_scale=args.amp_init_scale, enabled=amp)

    start_epoch, best_score, best_tau, history, bad = 1, float("inf"), 0.0, [], 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"]); optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch, best_score, best_tau = ck["epoch"] + 1, ck["best_val_mae"], ck.get("best_tau", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
    else:
        identity_check(model, val_loader, device, amp)

    for epoch in range(start_epoch, epochs + 1):
        tr, secs, n_upd, skipped = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                                                   device, epoch, args, amp)
        val = validate(model, val_loader, device, args, amp, taus,
                       max_batches=2 if args.smoke_test else args.max_validation_batches)
        chosen, tau, score, guard_ok = choose_config(val, args.empty_tol)
        base = val["coarse"]["mae"]
        print(f"\nEpoch {epoch:02d} | train loss {tr['total']:.4f} (dCount {tr['dcount']:+.3f}, gain {tr['gain']:.3f}, "
              f"add/cell {tr['add_mass']:.4f}) | skipped updates {skipped}/{n_upd} | {secs / 60:.1f} min")
        print_validation(val, chosen)
        print(f"  selected {chosen}: MAE {score:.4f} vs coarse {base:.4f} "
              f"({'improves' if score < base else 'does NOT beat coarse'}"
              f"{'' if guard_ok else '; empty-crop guard failed'})\n")
        history.append({"epoch": epoch, "train": tr, "validation": val, "chosen": chosen, "tau": tau,
                        "guard_ok": guard_ok, "seconds": secs})
        if args.smoke_test:
            continue
        # a config that fails the empty-crop guard is never promoted to "best"
        if guard_ok and score < best_score:
            best_score, best_tau, bad = score, tau, 0
            save_checkpoint(out / "stage2_best.pth", epoch, model, optimizer, scheduler, best_score, best_tau, args)
            json.dump({"epoch": epoch, "tau": tau, "chosen": chosen, "validation": val},
                      open(out / "stage2_best_validation.json", "w"), indent=2)
            print(f"Saved best checkpoint (tau={tau:g}, MAE {best_score:.4f}).")
        else:
            bad += 1
        save_checkpoint(out / "stage2_last.pth", epoch, model, optimizer, scheduler, best_score, best_tau, args)
        json.dump(history, open(out / "stage2_training_history.json", "w"), indent=2)
        if bad >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    if args.smoke_test:
        print("SMOKE TEST COMPLETED (no checkpoint saved).")
    else:
        print(f"Stage 2 finished. Best validation MAE {best_score:.4f} at tau={best_tau:g}.")


if __name__ == "__main__":
    main()