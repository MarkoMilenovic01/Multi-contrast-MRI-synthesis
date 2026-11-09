"""CLI: prepare, train, evaluate, infer and compare section 8.3.1 models."""
import argparse
import hashlib
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from mri_synthesis.data import Slices, prepare, normalize, slices_from_volume, stack
from mri_synthesis.models import Generator, System


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; install a CUDA-enabled PyTorch build or use --device cpu')
    return torch.device(name)


def checkpoint(path):
    # Checkpoints consist of tensors and primitive metadata, load with safe mode.
    return torch.load(path, map_location='cpu', weights_only=True)


def load_generator(path, device):
    saved = checkpoint(path)
    config = saved['config']
    generator = Generator(config['slices'], 1, config['width'], config['blocks']).to(device)
    generator.load_state_dict(saved['networks']['G'])
    return generator.eval(), config


def save_checkpoint(path, system, optimizers, config, epoch, history):
    payload = dict(config=config, epoch=epoch, history=history,
                   networks={k: v.state_dict() for k, v in system.nets.items()},
                   optimizers=[opt.state_dict() for opt in optimizers],
                   torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   pools={k: [item.cpu() for item in pool.items] for k, pool in system.pools.items()})
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    tracker = None
    if (Path(args.output)/'last.pt').exists() and not args.resume:
        raise ValueError('Existing experiment: use --resume or a new output directory')
    if getattr(args, 'mlflow', False):
        from mri_synthesis.tracking import Tracker
        tracker = Tracker(args.output, bool(args.resume))
    try:
        _train(args, tracker)
    except BaseException:
        if tracker:
            tracker.finish('interrupted_or_failed', failed=True)
        raise


def _train(args, tracker=None):
    started = time.monotonic()
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    config = dict(model=args.model, slices=args.slices, branch=args.branch, width=getattr(args, 'width', 64),
                  blocks=getattr(args, 'blocks', 9), lr=args.lr, constant=args.constant, decay=args.decay,
                  batch_size=args.batch_size, seed=args.seed, data=str(Path(args.data).resolve()))
    manifest = json.loads((Path(args.data)/'manifest.json').read_text())
    if 'image_size' in manifest:
        config['image_size'] = manifest['image_size']
        config['dataset_manifest_sha256'] = hashlib.sha256((Path(args.data)/'manifest.json').read_bytes()).hexdigest()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    resumed = checkpoint(args.resume) if args.resume else None
    if resumed:
        if resumed['config'] != config:
            raise ValueError('Resume configuration must match checkpoint (including data path and schedule)')
    elif (output/'last.pt').exists():
        raise ValueError('Existing experiment: use --resume or a new output directory')
    (output/'config.json').write_text(json.dumps(config, indent=2))
    dataset = Slices(args.data, 'train', args.model, args.branch, args.slices)
    if tracker:
        tracker.params(config, dataset)
    system = System(args.model, args.slices, device, width=config['width'], blocks=config['blocks'])
    optimizers = system.optimizers(args.lr)
    start, history = 0, []
    if resumed:
        for key, net in system.nets.items():
            net.load_state_dict(resumed['networks'][key])
        for opt, state in zip(optimizers, resumed['optimizers']):
            opt.load_state_dict(state)
        for key, items in resumed['pools'].items():
            system.pools[key].items = [item.to(device) for item in items]
        start, history = resumed['epoch'], resumed['history']
        torch.set_rng_state(resumed['torch_rng'])
        random.setstate(resumed['python_rng'])
        if device.type == 'cuda' and resumed['cuda_rng']:
            torch.cuda.set_rng_state_all(resumed['cuda_rng'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    max_seconds = getattr(args, 'max_minutes', 0)*60
    stop_reason = 'epochs_complete'
    for epoch in range(start, args.constant+args.decay):
        factor = 1 if epoch < args.constant else (args.constant+args.decay-epoch)/args.decay
        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = args.lr*factor
        totals, batches = {}, 0
        for x, y, _, _ in loader:
            if max_seconds and time.monotonic()-started >= max_seconds:
                stop_reason = 'time_budget'
                break
            logs = system.step(x.to(device), y.to(device), optimizers)
            if not all(np.isfinite(list(logs.values()))):
                raise FloatingPointError('Nonfinite losses; stop and inspect data')
            for key, value in logs.items():
                totals[key] = totals.get(key, 0)+value
            batches += 1
        if not batches:
            break
        row = dict(epoch=epoch+1, lr=args.lr*factor, batches=batches,
                   epoch_complete=int(batches == len(loader)),
                   elapsed_seconds=time.monotonic()-started,
                   **{k: v/batches for k, v in totals.items()})
        history.append(row)
        print(json.dumps(row), flush=True)
        if tracker:
            tracker.epoch(row)
        with (output/'losses.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        save_checkpoint(output/'last.pt', system, optimizers, config, epoch+1, history)
        if stop_reason == 'time_budget':
            break
    if history:
        save_checkpoint(output/'final.pt', system, optimizers, config, history[-1]['epoch'], history)
        plot_losses(history, output/'losses.png')
    summary = dict(stop_reason=stop_reason, elapsed_seconds=time.monotonic()-started,
                   epochs_recorded=len(history), max_minutes=getattr(args, 'max_minutes', 0),
                   partial_final_epoch=bool(history and history[-1].get('epoch_complete', 1) == 0),
                   note='Budget checked between batches; saving and plotting add a short overhead.')
    (output/'training_summary.json').write_text(json.dumps(summary, indent=2))
    if tracker:
        tracker.finish(stop_reason)


def plot_losses(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for key in history[0]:
        if key not in ['epoch', 'lr', 'batches', 'epoch_complete', 'elapsed_seconds']:
            ax.plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (individual terms unweighted)')
    ax.legend()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def metrics(prediction, reference):
    from skimage.metrics import structural_similarity
    error = np.mean((prediction-reference)**2)
    return dict(mae=float(np.abs(prediction-reference).mean()),
                psnr=float(10*np.log10(1/error)) if error > 0 else None,
                ssim=float(structural_similarity(reference, prediction, data_range=1)))


def evaluate(args):
    device = device_for(args.device)
    generator, config = load_generator(args.checkpoint, device)
    dataset = Slices(args.data, args.split, 'pgan', 'registered', config['slices'])
    grouped = {}
    with torch.inference_mode():
        for x, y, subject, _ in dataset:
            prediction = ((generator(x[None].to(device))[0, 0].cpu().numpy()+1)/2).clip(0, 1)
            row = metrics(prediction, (y[0].numpy()+1)/2)
            grouped.setdefault(subject, []).append(row)
    scores = {}
    for subject, rows in grouped.items():
        scores[subject] = {}
        for key in ['mae', 'psnr', 'ssim']:
            # None represents +infinity (identical images); do not silently omit it.
            scores[subject][key] = (None if any(r[key] is None for r in rows)
                                    else float(np.mean([r[key] for r in rows])))
    aggregate = {}
    for key in ['mae', 'psnr', 'ssim']:
        values = [r[key] for r in scores.values()]
        aggregate[key] = dict(mean=None, std=None) if None in values else dict(
            mean=float(np.mean(values)), std=float(np.std(values)))
    result = dict(split=args.split, checkpoint=str(args.checkpoint),
                  protocol=f"Full {config.get('image_size', 256)}x{config.get('image_size', 256)} slices, [0,1], mean within subject then equal subject weight",
                  psnr_null_means='positive infinity for identical images', subjects=scores, aggregate=aggregate)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    from mri_synthesis.tracking import log_evaluation
    log_evaluation(args.checkpoint, args.split, args.output, aggregate)
    print(json.dumps(aggregate, indent=2))


def infer(args):
    import nibabel as nib
    from skimage.transform import resize
    device = device_for(args.device)
    generator, config = load_generator(args.checkpoint, device)
    original = nib.load(args.source)
    canonical = nib.as_closest_canonical(original)
    volume = canonical.get_fdata(dtype=np.float32)
    size = config.get('image_size', 256)
    images = slices_from_volume(normalize(volume), size)
    result = np.zeros_like(volume, dtype=np.float32)
    scale = size/max(volume.shape[:2])
    shape = tuple(max(1, round(n*scale)) for n in volume.shape[:2])
    start = [(size-n)//2 for n in shape]
    with torch.inference_mode():
        for z in range(len(images)):
            x = stack(images, z, config['slices'])[None].to(device)
            prediction = ((generator(x)[0, 0].cpu().numpy()+1)/2).clip(0, 1)
            crop = prediction[start[0]:start[0]+shape[0], start[1]:start[1]+shape[1]]
            result[:, :, z] = resize(crop, volume.shape[:2], preserve_range=True)
    # Undo canonical axis permutation/flips without interpolating intensities.
    transform = nib.orientations.ornt_transform(nib.io_orientation(canonical.affine),
                                                nib.io_orientation(original.affine))
    restored = nib.orientations.apply_orientation(result, transform)
    header = original.header.copy()
    header.set_data_dtype(np.float32)
    output = nib.Nifti1Image(restored, original.affine, header)
    output.set_qform(original.get_qform(), int(original.header['qform_code']))
    output.set_sform(original.get_sform(), int(original.header['sform_code']))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, args.output)


def compare(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    device = device_for(args.device)
    manifest = json.loads((Path(args.data)/'manifest.json').read_text())
    if args.subject not in manifest['splits']['test'] or args.subject not in manifest['subjects']:
        raise ValueError('Comparison subject must be a successfully prepared held-out test subject')
    subject_dir = Path(args.data)/'test'/args.subject
    x = np.load(subject_dir/'x.npy', mmap_mode='r')
    y = np.load(subject_dir/'y.npy', mmap_mode='r')
    with np.load(subject_dir/'indices.npz') as data:
        valid = data['paired']
    z = args.slice if args.slice is not None else int(valid[len(valid)//2])
    if z not in valid:
        raise ValueError('Select a valid paired slice')
    paths = [args.pgan1, args.cgan_reg1, args.cgan_unreg1,
             args.pgan3, args.cgan_reg3, args.cgan_unreg3]
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    with torch.inference_mode():
        for row, count in enumerate([1, 3]):
            images = [x[z]]
            for col in range(3):
                generator, config = load_generator(paths[row*3+col], device)
                expected_model = 'pgan' if col == 0 else 'cgan'
                expected_branch = 'unregistered' if col == 2 else 'registered'
                if (config['slices'], config['model'], config['branch']) != (count, expected_model, expected_branch):
                    raise ValueError(f'Checkpoint in wrong comparison slot: {paths[row*3+col]}')
                images.append((generator(stack(x, z, count)[None].to(device))[0, 0].cpu().numpy()+1)/2)
                del generator
            images.append(y[z])
            for col, (image, title) in enumerate(zip(images, ['T1 source', 'pGAN', 'cGAN registered', 'cGAN unregistered', 'T2 reference'])):
                axes[row, col].imshow(image, cmap='gray', vmin=0, vmax=1)
                axes[row, col].set_title(f'{title}\n{count}-slice input')
                axes[row, col].axis('off')
    fig.suptitle(f'{args.subject}, axial slice {z} — generated results')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--raw', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(function=lambda a: prepare(a.raw, a.output, a.seed))
    p = commands.add_parser('train')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', choices=['pgan', 'cgan'], required=True)
    p.add_argument('--branch', choices=['registered', 'unregistered'], default='registered')
    p.add_argument('--slices', type=int, choices=[1, 3], default=1)
    p.add_argument('--constant', type=int, default=50)
    p.add_argument('--decay', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume')
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--blocks', type=int, default=9)
    p.add_argument('--max-minutes', type=float, default=0, help='0 disables wall-clock budget')
    p.add_argument('--mlflow', action='store_true')
    p.set_defaults(function=train)
    p = commands.add_parser('evaluate')
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--output', required=True)
    p.set_defaults(function=evaluate)
    p = commands.add_parser('infer')
    p.add_argument('--source', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(function=infer)
    p = commands.add_parser('compare')
    p.add_argument('--data', required=True)
    p.add_argument('--subject', required=True)
    p.add_argument('--slice', type=int)
    for name in ['pgan1', 'cgan-reg1', 'cgan-unreg1', 'pgan3', 'cgan-reg3', 'cgan-unreg3']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(function=compare)
    for subparser in commands.choices.values():
        if subparser.prog.endswith('prepare'):
            continue
        subparser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.command == 'train' and (args.constant < 0 or args.decay < 0 or args.constant+args.decay < 1 or args.batch_size < 1):
        parser.error('Use nonnegative schedule lengths, at least one epoch, and positive batch size')
    if args.command == 'train' and (args.width < 4 or args.blocks < 1 or args.max_minutes < 0):
        parser.error('Use width >= 4, blocks >= 1 and max-minutes >= 0')
    args.function(args)


if __name__ == '__main__':
    main()
