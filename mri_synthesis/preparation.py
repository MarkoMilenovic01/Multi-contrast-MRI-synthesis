"""Prepare the fixed 20/4/4-subject quick dataset; never train models."""
import argparse
import json
from pathlib import Path
from mri_synthesis.paths import ROOT
from mri_synthesis.data import discover, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT/'data'/'raw')
    parser.add_argument('--output', type=Path, default=ROOT/'data'/'quick')
    args = parser.parse_args()
    baseline = ROOT/'data'/'prepared'
    selection = json.loads((ROOT/'configs'/'quick_subjects.json').read_text())
    splits = selection['splits']
    t1, t2 = discover(args.raw, 'T1'), discover(args.raw, 'T2')
    missing = sorted({s for ids in splits.values() for s in ids} - (t1.keys() & t2.keys()))
    if missing:
        raise ValueError(f'Missing paired T1/T2 scans under {args.raw}: {", ".join(missing)}')
    prepare(args.raw, args.output, seed=selection['seed'], splits=splits,
            size=128, slice_count=20, reuse=baseline if baseline.exists() else None)
    quick = json.loads((args.output/'manifest.json').read_text())
    if quick['failures'] or len(quick['subjects']) != 28:
        raise RuntimeError('Some subjects failed; inspect the quick manifest before training')


if __name__ == '__main__':
    main()
