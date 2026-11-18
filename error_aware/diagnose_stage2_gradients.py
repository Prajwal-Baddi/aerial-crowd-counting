"""One-batch Stage-2 gradient diagnostic. Run from project root:
python error_aware/diagnose_stage2_gradients.py --no-amp
No checkpoint is saved; the temporary optimizer update is discarded.
"""
import argparse, random
import numpy as np
import torch
from data import get_loader
from model_multiplicative import ESRNet, cell_sum
from refine_losses_multiplicative import build_training_mask, refinement_loss


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='error_aware/processed')
    p.add_argument('--dataset-root', default='dataset/VisDrone2019-DET-train')
    p.add_argument('--checkpoint', default='error_aware/checkpoints/coarse_best.pth')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--refine-width', type=int, default=96)
    p.add_argument('--budget', type=float, default=0.25)
    p.add_argument('--background-frac', type=float, default=0.25)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-search-batches', type=int, default=200,
                   help='Maximum training batches to scan for a nonempty crop')
    p.add_argument('--nonempty-threshold', type=float, default=0.5)
    p.add_argument('--no-amp', action='store_true')
    return p.parse_args()


def stats(g):
    if g is None:
        return (0., 0., 0.)
    g = g.detach().float()
    return (g.norm().item(), g.abs().mean().item(), g.abs().max().item())


def show(label, g):
    n, m, x = stats(g)
    print(f'{label:<25} norm={n:.6e} mean|g|={m:.6e} max|g|={x:.6e}')


def main():
    a = args_parser()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda' and not a.no_amp
    print(f'Device: {device} | AMP: {amp}')

    model = ESRNet(pretrained=False, freeze_early=True, refine_width=a.refine_width).to(device)
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model.coarse.load_state_dict(ckpt.get('model_state_dict', ckpt.get('model', ckpt)), strict=True)
    for p in model.coarse.parameters(): p.requires_grad_(False)
    loader = get_loader(data_dir=a.data_dir, split='train', batch_size=a.batch_size,
                        num_workers=a.num_workers, dataset_root=a.dataset_root)
    selected_batch = None
    searched = 0
    for batch in loader:
        searched += 1
        batch_images, batch_gt, batch_counts = batch
        if bool((batch_counts > a.nonempty_threshold).any()):
            selected_batch = (batch_images, batch_gt, batch_counts)
            break
        if searched >= a.max_search_batches:
            break
    if selected_batch is None:
        raise RuntimeError(f'No nonempty crop found in {searched} batches. Increase --max-search-batches.')
    images, gt, counts = selected_batch
    images, gt = images.to(device), gt.to(device).float()
    model.train(); model.coarse.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        c = model.coarse(images)
    coarse, ctx, f4 = c['density'].float(), c['context'], c['stride4_features']
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        e_pred = model.error_head(ctx, coarse)
    e_true = (cell_sum(coarse.detach()) - cell_sum(gt)).abs()
    mask = build_training_mask(e_true, e_pred.float().detach(), a.budget, 0.8, a.background_frac)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        final, aux = model.apply_refine(coarse, f4, mask, return_aux=True)

    conv = model.refiner.body[-1]
    total, parts = refinement_loss(final, coarse, gt, mask, aux)
    print('\n--- Batch ---')
    print(f'Selected after scanning {searched} training batch(es).')
    print('GT counts:', counts.tolist())
    n_nonempty = int((counts > a.nonempty_threshold).sum().item())
    print(f'Nonempty crops: {n_nonempty}/{len(counts)} (GT count > {a.nonempty_threshold})')
    print(f'Selected cells: {int(mask.sum())}/{mask.numel()}')
    print('Coarse counts:', coarse.sum((1,2,3)).detach().cpu().tolist())
    print('Refined counts:', final.sum((1,2,3)).detach().cpu().tolist())
    print(f'Mean gain: {aux["gain"].detach().float().mean().item():.9e}')
    print(f'Mean add: {aux["add"].detach().float().mean().item():.9e}')
    add_mass = aux['add'].detach().float().sum((1,2,3)).cpu().tolist()
    print('Add mass/crop:', [f'{v:.6e}' for v in add_mass])

    print('\n--- Per-loss gradients at final output convolution ---')
    for name, loss in parts.items():
        gw, gb = torch.autograd.grad(loss, (conv.weight, conv.bias), retain_graph=True, allow_unused=True)
        print(f'\n{name} loss={loss.detach().item():.8g}')
        show('gain-channel weights', None if gw is None else gw[0])
        show('add-channel weights', None if gw is None else gw[1])
        show('gain-channel bias', None if gb is None else gb[0])
        show('add-channel bias', None if gb is None else gb[1])

    print('\n--- Total refinement gradients ---')
    total.backward()
    show('gain-channel weights', None if conv.weight.grad is None else conv.weight.grad[0])
    show('add-channel weights', None if conv.weight.grad is None else conv.weight.grad[1])
    show('gain-channel bias', None if conv.bias.grad is None else conv.bias.grad[0])
    show('add-channel bias', None if conv.bias.grad is None else conv.bias.grad[1])
    show('feature_reduce weights', model.refiner.feature_reduce.weight.grad)
    show('first body conv weights', model.refiner.body[0].weight.grad)

    print('\n--- Temporary AdamW update ---')
    opt = torch.optim.AdamW(list(model.error_head.parameters()) + list(model.refiner.parameters()),
                            lr=a.lr, weight_decay=1e-4)
    old_w, old_b = conv.weight.detach().clone(), conv.bias.detach().clone()
    opt.step()
    dw, db = conv.weight.detach() - old_w, conv.bias.detach() - old_b
    show('gain-channel weight delta', dw[0])
    show('add-channel weight delta', dw[1])
    print(f'gain bias delta: {db[0].item():+.6e}')
    print(f'add bias delta:  {db[1].item():+.6e}')
    print('\nDiagnostic complete. No checkpoint or project file was modified.')

if __name__ == '__main__': main()
