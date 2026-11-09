"""Subject-safe IXI preparation, physical-space registration and slice loading."""
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize


def normalize(volume):
    volume = np.nan_to_num(volume, nan=0, posinf=0, neginf=0).astype(np.float32)
    foreground = volume[volume > 0]
    if not foreground.size:
        raise ValueError('Volume contains no positive foreground')
    low, high = np.percentile(foreground, [0.5, 99.5])
    if high <= low:
        raise ValueError('Volume has no usable intensity range')
    return np.clip((volume-low)/(high-low), 0, 1).astype(np.float32)


def fit_slice(image, size=256):
    scale = size / max(image.shape)
    shape = tuple(max(1, round(n*scale)) for n in image.shape)
    small = resize(image, shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
    out = np.zeros((size, size), np.float32)
    start = [(size-n)//2 for n in shape]
    out[start[0]:start[0]+shape[0], start[1]:start[1]+shape[1]] = small
    return out


def slices_from_volume(volume, size=256):
    return np.stack([fit_slice(volume[:, :, i], size) for i in range(volume.shape[2])])


def rigid_register(fixed_path, moving_path):
    import SimpleITK as sitk
    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                               sitk.CenteredTransformInitializerFilter.GEOMETRY)
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.1, 42)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=0.001,
                                                    numberOfIterations=200,
                                                    relaxationFactor=0.5)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    transform = method.Execute(fixed, moving)
    if not np.isfinite(method.GetMetricValue()):
        raise ValueError('Nonfinite registration metric')
    aligned = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0, sitk.sitkFloat32)
    return aligned, transform, method.GetOptimizerStopConditionDescription()


def discover(root, contrast):
    found = {}
    for path in Path(root).rglob('*.nii*'):
        if re.search(rf'-{contrast}\.nii(\.gz)?$', path.name, re.I):
            match = re.match(r'(IXI\d+)', path.name, re.I)
            if match:
                key = match[1].upper()
                if key in found:
                    raise ValueError(f'Duplicate {contrast} for {key}: {path}')
                found[key] = path
    return found


def split_subjects(ids, seed=42):
    ids = sorted(ids)
    if len(ids) < 7:
        raise ValueError('At least seven paired subjects are required for three splits')
    random.Random(seed).shuffle(ids)
    a, b = int(len(ids)*0.7), int(len(ids)*0.85)
    return dict(train=ids[:a], val=ids[a:b], test=ids[b:])


def central_indices(indices, count):
    """Evenly sample central 60% of usable slices, retaining true neighbor indices."""
    indices = np.asarray(indices)
    if count is None:
        return indices
    margin = int(len(indices)*0.2)
    central = indices[margin:len(indices)-margin] if margin else indices
    if len(central) < count:
        raise ValueError(f'Only {len(central)} central slices; need {count}')
    return central[np.linspace(0, len(central)-1, count).round().astype(int)]


def prepare(raw, output, seed=42, splits=None, size=256, slice_count=None, reuse=None):
    import nibabel as nib
    import SimpleITK as sitk
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise ValueError('Output already contains a manifest; choose a new output directory')
    t1, t2 = discover(raw, 'T1'), discover(raw, 'T2')
    splits = splits or split_subjects(t1.keys() & t2.keys(), seed)
    manifest = dict(seed=seed, splits=splits, subjects={}, failures={},
                    image_size=size, slices_per_subject=slice_count,
                    missing_t1=sorted(t2.keys()-t1.keys()), missing_t2=sorted(t1.keys()-t2.keys()))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for split, ids in splits.items():
        for subject in ids:
            try:
                cached = Path(reuse)/'registration'/subject if reuse else None
                if cached and (cached/'T2_registered.nii.gz').exists():
                    aligned = sitk.ReadImage(str(cached/'T2_registered.nii.gz'))
                    transform = None
                    stop = 'Reused existing registration; inspect alignment preview'
                else:
                    aligned, transform, stop = rigid_register(t1[subject], t2[subject])
                work = output/'registration'/subject
                work.mkdir(parents=True, exist_ok=True)
                aligned_path = work/'T2_registered.nii.gz'
                sitk.WriteImage(aligned, str(aligned_path))
                # SimpleITK resampling transforms map fixed/output points to moving/input points.
                if transform is not None:
                    sitk.WriteTransform(transform, str(work/'T1_to_T2_resampling.tfm'))
                volumes = [nib.as_closest_canonical(nib.load(str(p))).get_fdata(dtype=np.float32)
                           for p in [t1[subject], t2[subject], aligned_path]]
                x, raw_y, y = [slices_from_volume(normalize(v), size) for v in volumes]
                valid_x = np.flatnonzero((x > 0.05).mean((1, 2)) >= 0.05)
                valid_y = np.flatnonzero((y > 0.05).mean((1, 2)) >= 0.05)
                paired = np.intersect1d(valid_x, valid_y)
                raw_valid = np.flatnonzero((raw_y > 0.05).mean((1, 2)) >= 0.05)
                paired = central_indices(paired, slice_count)
                raw_valid = central_indices(raw_valid, slice_count)
                if slice_count is not None:
                    valid_x = paired.copy()  # Same source examples in all six experiments.
                if not len(paired) or not len(raw_valid):
                    raise ValueError('No usable foreground slices after preparation')
                dest = output/split
                dest.mkdir(exist_ok=True)
                subject_dir = dest/subject
                subject_dir.mkdir(exist_ok=True)
                for name, array in dict(x=x, y=y, raw_y=raw_y).items():
                    np.save(subject_dir/f'{name}.npy', array)
                np.savez_compressed(subject_dir/'indices.npz', paired=paired,
                                    valid_x=valid_x, raw_valid=raw_valid)
                mid = int(paired[len(paired)//2])
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                for ax, im, title in zip(axes, [x[mid], y[mid], (x[mid]+y[mid])/2],
                                         ['T1', 'Registered T2', 'Overlay — inspect alignment']):
                    ax.imshow(im, cmap='gray', vmin=0, vmax=1)
                    ax.set_title(title)
                    ax.axis('off')
                fig.savefig(work/'alignment.png', bbox_inches='tight')
                plt.close(fig)
                manifest['subjects'][subject] = dict(split=split, t1=str(t1[subject].resolve()),
                    t2=str(t2[subject].resolve()), registration_stop=stop,
                    paired_slices=len(paired), source_slices=len(valid_x), unregistered_target_slices=len(raw_valid))
                print(f'{subject}: prepared {len(paired)} paired slices', flush=True)
            except Exception as error:
                manifest['failures'][subject] = str(error)
                print(f'{subject}: FAILED: {error}', flush=True)
            (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for split in splits:
        if not any(v['split'] == split for v in manifest['subjects'].values()):
            raise ValueError(f'No successfully prepared subjects in {split}; inspect manifest')


def stack(array, index, count):
    indices = np.clip(np.arange(index-count//2, index+count//2+1), 0, len(array)-1)
    return torch.from_numpy(np.asarray(array[indices], dtype=np.float32).copy()*2-1)


class Slices(Dataset):
    def __init__(self, root, split, model='pgan', branch='registered', count=1):
        self.root, self.model, self.branch, self.count = Path(root), model, branch, count
        manifest = json.loads((self.root/'manifest.json').read_text())
        self.sources, self.targets = [], []
        self.paths = {}
        for subject in manifest['splits'][split]:
            if subject not in manifest['subjects']:
                continue
            path = self.root/split/subject
            self.paths[subject] = path
            with np.load(path/'indices.npz') as arrays:
                source = arrays['paired'] if branch == 'registered' else arrays['valid_x']
                target = arrays['paired'] if branch == 'registered' else arrays['raw_valid']
                self.sources.extend((subject, int(i)) for i in source)
                self.targets.extend((subject, int(i)) for i in target)
        if not self.sources or not self.targets:
            raise ValueError(f'Empty {split} dataset')
        if model == 'pgan' and branch != 'registered':
            raise ValueError('pGAN requires registered paired data')

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        subject, z = self.sources[index]
        target_subject, target_z = ((subject, z) if self.model == 'pgan'
                                     else random.choice(self.targets))
        x = stack(np.load(self.paths[subject]/'x.npy', mmap_mode='r'), z, self.count)
        target_name = 'y' if self.branch == 'registered' else 'raw_y'
        y = stack(np.load(self.paths[target_subject]/f'{target_name}.npy', mmap_mode='r'), target_z, 1)
        return x, y, subject, z
