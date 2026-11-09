"""Explicit launcher: python train_quick.py --start. Without --start: preflight only."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from mri_synthesis.paths import ROOT, CONFIG_PATH


def preflight(root, spec):
    from mri_synthesis.data import Slices
    import torch
    manifest = json.loads((root/spec['data']/'manifest.json').read_text())
    if manifest['failures'] or len(manifest['subjects']) != 28 or manifest['image_size'] != 128:
        raise ValueError('Quick dataset is incomplete or has unexpected resolution')
    groups = [set(manifest['splits'][name]) for name in ['train', 'val', 'test']]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Patient leakage between splits')
    for split, expected in [('train', 20), ('val', 4), ('test', 4)]:
        if len(manifest['splits'][split]) != expected:
            raise ValueError(f'Unexpected {split} split')
        for subject in manifest['splits'][split]:
            if manifest['subjects'][subject]['paired_slices'] != 20:
                raise ValueError(f'Unexpected slice count: {subject}')
        for item in spec['experiments']:
            dataset = Slices(root/spec['data'], split, item['model'], item['branch'], item['slices'])
            if len(dataset) != expected*20:
                raise ValueError(f'Unexpected example count in {split}')
            x, y, _, _ = dataset[0]
            if x.shape != (item['slices'], 128, 128) or y.shape != (1, 128, 128):
                raise ValueError('Unexpected tensor shapes')
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError('Nonfinite data')
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required for the quick experiment launcher')
    if not (Path(torch.hub.get_dir())/'checkpoints'/'vgg16-397923af.pth').exists():
        raise ValueError('Pre-cache pretrained VGG16 weights before the timed suite')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true', help='Actually start all six training runs')
    args = parser.parse_args()
    root = ROOT
    spec = json.loads(CONFIG_PATH.read_text())
    manifest = preflight(root, spec)
    print('Ready: 400 training, 80 validation, 80 test examples; 128x128; six runs.', flush=True)
    if not args.start:
        print('Preflight only. Training has NOT started. Use --start when ready.')
        return
    started = time.monotonic()
    for item in spec['experiments']:
        remaining = spec['suite_minutes']-(time.monotonic()-started)/60
        if remaining <= 1:
            raise RuntimeError('Suite time budget exhausted; remaining experiments were not started')
        output = root/'runs'/'quick'/item['name']
        if (output/'final.pt').exists():
            print(f"Skipping completed {item['name']}", flush=True)
            continue
        minutes = min(spec['max_minutes_per_experiment'], remaining-1)
        command = [sys.executable, 'run.py', 'train', '--data', spec['data'],
                   '--output', str(output), '--model', item['model'], '--branch', item['branch'],
                   '--slices', str(item['slices']), '--width', str(spec['width']),
                   '--blocks', str(spec['blocks']), '--constant', str(spec['constant']),
                   '--decay', str(spec['decay']), '--device', 'cuda', '--mlflow', '--max-minutes', str(minutes)]
        if (output/'last.pt').exists():
            command += ['--resume', str(output/'last.pt')]
        subprocess.run(command, cwd=root, check=True)
        for split in ['val', 'test']:
            subprocess.run([sys.executable, 'run.py', 'evaluate', '--data', spec['data'],
                '--checkpoint', str(output/'final.pt'), '--split', split,
                '--output', f'results/quick/{item["name"]}_{split}.json', '--device', 'cuda'], cwd=root, check=True)
    subject = manifest['splits']['test'][0]
    command = [sys.executable, 'run.py', 'compare', '--data', spec['data'], '--subject', subject,
               '--output', 'results/quick/comparison.png', '--device', 'cuda']
    for item in spec['experiments']:
        option = item['name'].replace('_', '-')
        command += ['--'+option, f'runs/quick/{item["name"]}/final.pt']
    subprocess.run(command, cwd=root, check=True)
    from mri_synthesis.tracking import log_evaluation
    for item in spec['experiments']:
        log_evaluation(root/'runs'/'quick'/item['name']/'final.pt', 'comparison',
                       root/'results'/'quick'/'comparison.png', {})


if __name__ == '__main__':
    main()
