"""CPU smoke/validation: python -m unittest -v tests.test_pipeline (no MRI/download needed)."""
import argparse
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mri_synthesis.data import Slices, fit_slice, normalize, prepare, rigid_register, split_subjects, stack
from mri_synthesis.models import Generator, Perceptual, System
from mri_synthesis.cli import (checkpoint, compare, evaluate, infer, load_generator,
                 metrics, save_checkpoint, seed_all)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_subject_split_and_normalization(self):
        splits = split_subjects([f'IXI{i:03}' for i in range(20)])
        self.assertEqual([len(splits[k]) for k in ['train', 'val', 'test']], [14, 3, 3])
        self.assertFalse(set(splits['train']) & set(splits['test']))
        self.assertFalse(set(splits['train']) & set(splits['val']))
        self.assertFalse(set(splits['val']) & set(splits['test']))
        self.assertEqual(splits, split_subjects([f'IXI{i:03}' for i in range(20)]))
        volume = normalize(np.arange(1000).reshape(10, 10, 10))
        self.assertTrue(np.isfinite(volume).all())
        self.assertGreaterEqual(volume.min(), 0)
        self.assertLessEqual(volume.max(), 1)
        self.assertEqual(fit_slice(volume[:, :, 0]).shape, (256, 256))
        with self.assertRaises(ValueError):
            normalize(np.zeros((4, 4, 4)))

    def test_metrics_and_boundary_stack(self):
        image = np.random.default_rng(42).uniform(0, 1, (32, 32)).astype('float32')
        same = metrics(image, image)
        self.assertEqual(same, dict(mae=0, psnr=None, ssim=1))
        changed = metrics(image*0.9, image)
        self.assertGreater(changed['mae'], 0)
        self.assertLess(changed['ssim'], 1)
        values = np.stack([image, image*0.5])
        stacked = stack(values, 0, 3)
        self.assertTrue(torch.equal(stacked[0], stacked[1]))

    def test_perceptual_gradients(self):
        # Random weights ONLY in tests: production pGAN always requests ImageNet weights.
        loss = Perceptual(pretrained=False)
        x = torch.rand(2, 1, 32, 32, requires_grad=True)
        value = loss(x, torch.zeros_like(x))
        value.backward()
        self.assertGreater(x.grad[0].abs().sum().item(), 0)
        self.assertGreater(x.grad[1].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in loss.parameters()))

    def test_six_training_steps_and_checkpoint(self):
        seed_all(42)
        with tempfile.TemporaryDirectory() as temp:
            for model, branch in [('pgan', 'registered'), ('cgan', 'registered'), ('cgan', 'unregistered')]:
                for count in [1, 3]:
                    with self.subTest(model=model, branch=branch, slices=count):
                        system = System(model, count, torch.device('cpu'), width=4, blocks=9, pretrained=False)
                        optimizers = system.optimizers(0.0002)
                        x, y = torch.rand(2, count, 32, 32)*2-1, torch.rand(2, 1, 32, 32)*2-1
                        before = system.nets['G'][1].weight.detach().clone()
                        losses = system.step(x, y, optimizers)
                        self.assertTrue(all(np.isfinite(list(losses.values()))))
                        self.assertFalse(torch.equal(before, system.nets['G'][1].weight))
                        self.assertEqual(system.nets['G'](x).shape, y.shape)
                        if model == 'cgan':
                            self.assertEqual(system.nets['F'](system.nets['G'](x)).shape, x.shape)
                            self.assertEqual(system.nets['G'](system.nets['F'](y)).shape, y.shape)
                            self.assertTrue(any(p.grad is not None for p in system.nets['F'].parameters()))
                        config = dict(model=model, slices=count, branch=branch, width=4, blocks=9)
                        path = Path(temp)/'last.pt'
                        save_checkpoint(path, system, optimizers, config, 1, [losses])
                        generator, loaded_config = load_generator(path, torch.device('cpu'))
                        system.nets['G'].eval()
                        with torch.no_grad():
                            torch.testing.assert_close(generator(x), system.nets['G'](x))
                        self.assertEqual(config, loaded_config)
                        saved = checkpoint(path)
                        resumed = System(model, count, torch.device('cpu'), width=4, blocks=9, pretrained=False)
                        resumed_opts = resumed.optimizers(0.0002)
                        for name, net in resumed.nets.items():
                            net.load_state_dict(saved['networks'][name])
                        for opt, state in zip(resumed_opts, saved['optimizers']):
                            opt.load_state_dict(state)
                        self.assertTrue(all(np.isfinite(list(resumed.step(x, y, resumed_opts).values()))))

    def test_data_evaluation_comparison_and_nifti_inference(self):
        import nibabel as nib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'test').mkdir()
            image = np.random.default_rng(0).uniform(0, 1, (3, 32, 32)).astype('float32')
            directory = root/'test'/'IXI001'
            directory.mkdir()
            for name, array in dict(x=image, y=image*0.8, raw_y=image*0.7).items():
                np.save(directory/f'{name}.npy', array)
            np.savez_compressed(directory/'indices.npz', paired=np.arange(3),
                                valid_x=np.arange(3), raw_valid=np.arange(3))
            (root/'manifest.json').write_text(json.dumps(dict(splits=dict(test=['IXI001']), subjects=dict(IXI001={}))))
            paired = Slices(root, 'test', count=3)
            x, y, subject, z = paired[0]
            self.assertEqual(x.shape, (3, 32, 32))
            for branch in ['registered', 'unregistered']:
                unpaired = Slices(root, 'test', 'cgan', branch, 3)
                self.assertEqual(unpaired[0][1].shape, (1, 32, 32))
            paths = {}
            for model, branch in [('pgan', 'registered'), ('cgan', 'registered'), ('cgan', 'unregistered')]:
                for count in [1, 3]:
                    config = dict(model=model, branch=branch, slices=count, width=4, blocks=9)
                    path = root/f'{model}_{branch}_{count}.pt'
                    # Untrained test fixtures, never experimental results.
                    torch.save(dict(config=config, networks={'G': Generator(count, 1, 4, 9).state_dict()}), path)
                    paths[model, branch, count] = str(path)
            path = paths['pgan', 'registered', 1]
            evaluate(argparse.Namespace(device='cpu', checkpoint=path, data=str(root), split='test', output=str(root/'metrics.json')))
            self.assertIn('IXI001', json.loads((root/'metrics.json').read_text())['subjects'])
            compare(argparse.Namespace(device='cpu', data=str(root), subject='IXI001', slice=None,
                pgan1=path, pgan3=paths['pgan', 'registered', 3],
                cgan_reg1=paths['cgan', 'registered', 1], cgan_reg3=paths['cgan', 'registered', 3],
                cgan_unreg1=paths['cgan', 'unregistered', 1], cgan_unreg3=paths['cgan', 'unregistered', 3],
                output=str(root/'comparison.png')))
            self.assertTrue((root/'comparison.png').exists())
            volume = np.random.default_rng(2).uniform(1, 100, (16, 20, 3)).astype('float32')
            affine = np.diag([-1.5, 2.0, 3.0, 1.0])
            source = root/'source.nii.gz'
            nib.save(nib.Nifti1Image(volume, affine), source)
            infer(argparse.Namespace(device='cpu', checkpoint=path, source=str(source), output=str(root/'prediction.nii.gz')))
            prediction = nib.load(root/'prediction.nii.gz')
            self.assertEqual(prediction.shape, volume.shape)
            np.testing.assert_allclose(prediction.affine, affine)
            self.assertTrue(np.isfinite(prediction.get_fdata()).all())

    def test_physical_registration(self):
        import SimpleITK as sitk
        with tempfile.TemporaryDirectory() as temp:
            grid = np.indices((32, 32, 32), dtype=np.float32)
            radius = sum((grid[i]-(12+i*3))**2 for i in range(3))
            array = np.exp(-radius/45).astype(np.float32)
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((1.1, 1.2, 1.3))
            path = Path(temp)/'volume.nii.gz'
            sitk.WriteImage(image, str(path))
            aligned, transform, stop = rigid_register(path, path)
            self.assertEqual(aligned.GetSize(), image.GetSize())
            np.testing.assert_allclose(aligned.GetSpacing(), image.GetSpacing(), rtol=1e-6)
            self.assertTrue(np.isfinite(sitk.GetArrayFromImage(aligned)).all())
            self.assertTrue(stop)
            self.assertLess(np.abs(sitk.GetArrayFromImage(aligned)-array).mean(), 0.02)

    def test_prepare_end_to_end(self):
        import nibabel as nib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root/'raw'
            raw.mkdir()
            grid = np.indices((32, 32, 32), dtype=np.float32)
            radius = sum((grid[i]-(12+i*3))**2 for i in range(3))
            volume = (100*np.exp(-radius/45)).astype(np.float32)
            for i in range(7):
                for contrast in ['T1', 'T2']:
                    nib.save(nib.Nifti1Image(volume, np.eye(4)), raw/f'IXI{i:03}-Test-{contrast}.nii.gz')
            prepare(raw, root/'prepared')
            manifest = json.loads((root/'prepared'/'manifest.json').read_text())
            self.assertEqual(len(manifest['subjects']), 7)
            self.assertFalse(manifest['failures'])
            for split in ['train', 'val', 'test']:
                dataset = Slices(root/'prepared', split, count=3)
                x, y, subject, z = dataset[0]
                self.assertEqual(x.shape, (3, 256, 256))
                self.assertEqual(y.shape, (1, 256, 256))
                self.assertTrue(torch.isfinite(x).all())


if __name__ == '__main__':
    unittest.main()
