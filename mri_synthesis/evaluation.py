"""Check or evaluate all six trained models. Never starts or resumes training."""
import argparse
import csv
import json
from pathlib import Path
from mri_synthesis.paths import ROOT, CONFIG_PATH
from types import SimpleNamespace

from mri_synthesis.cli import compare, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify files and splits without inference')
    parser.add_argument('--split', choices=['val', 'test', 'both'], default='test')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    root = ROOT
    spec = json.loads(CONFIG_PATH.read_text())
    data = root/spec['data']
    manifest = json.loads((data/'manifest.json').read_text())
    groups = [set(manifest['splits'][name]) for name in ['train', 'val', 'test']]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Subject overlap between splits')
    if manifest['failures']:
        raise ValueError('Dataset has preparation failures')
    from mri_synthesis.data import Slices
    for split, expected in [('train', 400), ('val', 80), ('test', 80)]:
        if len(Slices(data, split)) != expected:
            raise ValueError(f'Unexpected number of {split} examples')
    for item in spec['experiments']:
        path = root/'runs/quick'/item['name']/'final.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing completed checkpoint: {path}')
    print('All six final checkpoints and shared 400/80/80 dataset found; no patient overlap.', flush=True)
    if args.check_only:
        print('Check complete. No training or inference performed.')
        return
    output = root/'results/quick'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    from mri_synthesis.cli import checkpoint
    for item in spec['experiments']:
        path = root/'runs/quick'/item['name']/'final.pt'
        metadata = checkpoint(path)
        config = metadata['config']
        if (config['model'], config['branch'], config['slices']) != (item['model'], item['branch'], item['slices']):
            raise ValueError(f'Wrong checkpoint configuration for {item["name"]}')
        epoch = metadata['epoch']
        del metadata
        for split in (['val', 'test'] if args.split == 'both' else [args.split]):
            report = output/f'{item["name"]}_{split}.json'
            evaluate(SimpleNamespace(device=args.device, checkpoint=str(path), data=str(data),
                                     split=split, output=str(report)))
            aggregate = json.loads(report.read_text())['aggregate']
            rows.append(dict(experiment=item['name'], split=split, epoch=epoch,
                             **{f'{metric}_{stat}': value for metric, values in aggregate.items()
                                for stat, value in values.items()}))
    summary = output/f'summary_{args.split}.csv'
    with summary.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    options = dict(device=args.device, data=str(data), subject=manifest['splits']['test'][0],
                   slice=None, output=str(output/'comparison.png'))
    for item in spec['experiments']:
        options[item['name']] = str(root/'runs/quick'/item['name']/'final.pt')
    compare(SimpleNamespace(**options))
    from mri_synthesis.tracking import log_evaluation
    for item in spec['experiments']:
        path = root/'runs/quick'/item['name']/'final.pt'
        log_evaluation(path, 'comparison', output/'comparison.png', {})
        log_evaluation(path, 'summary', summary, {})
    print(f'Scores: {summary}\nExamples: {output/"comparison.png"}\nNo training performed.')


if __name__ == '__main__':
    main()
