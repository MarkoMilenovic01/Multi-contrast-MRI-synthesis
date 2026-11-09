"""Preparation checks only: no optimizer updates or MRI model training."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mri_synthesis.data import central_indices, stack


class QuickSetupTests(unittest.TestCase):
    def test_public_subject_list(self):
        root = Path(__file__).resolve().parents[1]
        selection = json.loads((root/'configs/quick_subjects.json').read_text())
        splits = selection['splits']
        self.assertEqual([len(splits[k]) for k in ['train', 'val', 'test']], [20, 4, 4])
        self.assertEqual(len({s for ids in splits.values() for s in ids}), 28)

    def test_preparation_missing_inputs_does_not_write(self):
        from mri_synthesis import preparation
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'must_not_exist'
            with patch('sys.argv', ['preparation', '--raw', temp, '--output', str(output)]), \
                 patch.object(preparation, 'prepare') as prepare:
                with self.assertRaisesRegex(ValueError, 'Missing paired T1/T2'):
                    preparation.main()
                prepare.assert_not_called()
            self.assertFalse(output.exists())

    def test_center_selection_preserves_neighbors(self):
        selected = central_indices(np.arange(100), 20)
        self.assertEqual(len(np.unique(selected)), 20)
        self.assertGreaterEqual(selected.min(), 20)
        self.assertLessEqual(selected.max(), 79)
        volume = np.arange(100, dtype=np.float32)[:, None, None]/100
        result = stack(volume, int(selected[5]), 3)
        np.testing.assert_allclose(np.diff(result[:, 0, 0].numpy()), [0.02, 0.02], atol=1e-6)
        with self.assertRaises(ValueError):
            central_indices(np.arange(10), 20)

    def test_six_configuration_matrix(self):
        spec = json.loads((Path(__file__).resolve().parents[1]/'configs'/'quick_experiments.json').read_text())
        actual = {(e['model'], e['branch'], e['slices']) for e in spec['experiments']}
        expected = {(m, b, s) for m, b in [('pgan', 'registered'),
                    ('cgan', 'registered'), ('cgan', 'unregistered')] for s in [1, 3]}
        self.assertEqual(actual, expected)
        self.assertEqual(spec['width'], 32)
        self.assertEqual(spec['blocks'], 9)
        self.assertLessEqual(len(actual)*spec['max_minutes_per_experiment'], 150)

    def test_preflight_does_not_launch_training(self):
        import mri_synthesis.suite as train_quick
        manifest = {'splits': {'test': ['IXI000']}}
        with patch('sys.argv', ['train_quick.py']), patch.object(train_quick, 'preflight', return_value=manifest), \
                patch.object(train_quick.subprocess, 'run') as launch:
            train_quick.main()
        launch.assert_not_called()

    def test_tracking_logging_and_resume(self):
        import mlflow
        from mlflow.tracking import MlflowClient
        from mri_synthesis.tracking import Tracker, log_evaluation
        # Isolated test database: fixture metrics never enter the real project experiment.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root = Path(temp)
            uri = 'sqlite:///'+(root/'fixture.db').as_posix()

            def setup_fixture():
                mlflow.set_tracking_uri(uri)
                experiment = mlflow.get_experiment_by_name('setup-test-fixtures')
                if experiment is None:
                    mlflow.create_experiment('setup-test-fixtures', artifact_location=(root/'artifacts').as_uri())
                mlflow.set_experiment('setup-test-fixtures')
                return mlflow

            data = root/'data'
            data.mkdir()
            (data/'manifest.json').write_text('{"fixture": true}')
            output = root/'run'
            class FixtureDataset:
                def __init__(self, folder):
                    self.root = folder

                def __len__(self):
                    return 1

            with patch('mri_synthesis.tracking.setup', setup_fixture), patch('mri_synthesis.tracking.tracking_uri', return_value=uri):
                tracker = Tracker(output)
                tracker.params({'fixture': True}, FixtureDataset(data))
                tracker.epoch({'epoch': 1, 'fixture_metric': 0.5})
                first_id = tracker.run.info.run_id
                tracker.finish('fixture_no_training')
                resumed = Tracker(output, resume=True)
                self.assertEqual(first_id, resumed.run.info.run_id)
                resumed.finish('fixture_resumed_no_training')
            report = root/'scores.json'
            report.write_text('{"fixture": true}')
            log_evaluation(output/'final.pt', 'val', report, {'mae': {'mean': 0.1, 'std': 0}})
            record = MlflowClient(tracking_uri=uri).get_run(first_id)
            self.assertEqual(record.data.metrics['val_mae_mean'], 0.1)
            self.assertEqual(record.data.metrics['fixture_metric'], 0.5)
            self.assertEqual(record.info.status, 'FINISHED')

    def test_budget_stop_records_summary_without_optimizer_updates(self):
        import argparse
        import mri_synthesis.cli as run
        import torch
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root/'data'
            data.mkdir()
            (data/'manifest.json').write_text('{"image_size": 128}')
            output = root/'out'
            args = argparse.Namespace(seed=42, threads=2, device='cpu', model='cgan',
                slices=1, branch='registered', width=4, blocks=9, lr=.0002,
                constant=1, decay=0, batch_size=1, data=str(data), output=str(output),
                resume=None, max_minutes=0.001)
            system = MagicMock()
            system.optimizers.return_value = [MagicMock(param_groups=[{}]), MagicMock(param_groups=[{}])]
            dataset = [(torch.zeros(1, 128, 128), torch.zeros(1, 128, 128), 'fixture', 0)]
            with patch.object(run, 'Slices', return_value=dataset), \
                 patch.object(run, 'System', return_value=system), \
                 patch.object(run.time, 'monotonic', side_effect=[0, 1, 1]):
                run._train(args)
            system.step.assert_not_called()
            summary = json.loads((output/'training_summary.json').read_text())
            self.assertEqual(summary['stop_reason'], 'time_budget')


if __name__ == '__main__':
    unittest.main()
