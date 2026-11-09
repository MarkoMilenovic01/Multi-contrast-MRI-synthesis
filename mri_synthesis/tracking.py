"""Local MLflow tracking without a required background server."""
import json
from pathlib import Path
from mri_synthesis.paths import ROOT


def tracking_uri():
    return 'sqlite:///' + (ROOT/'tracking'/'mlflow.db').resolve().as_posix()


def setup():
    import mlflow
    directory = ROOT/'tracking'
    directory.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri())
    name = 'MRI section 8.3.1 - quick six experiments'
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        mlflow.create_experiment(name, artifact_location=(directory/'artifacts').resolve().as_uri())
    mlflow.set_experiment(name)
    return mlflow


class Tracker:
    def __init__(self, output, resume=False):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.mlflow = setup()
        info = self.output/'mlflow.json'
        previous = json.loads(info.read_text()) if resume and info.exists() else None
        self.run = self.mlflow.start_run(run_id=previous['run_id'] if previous else None,
                                        run_name=self.output.name)
        info.write_text(json.dumps(dict(run_id=self.run.info.run_id,
                                       tracking_uri=tracking_uri()), indent=2))

    def params(self, config, dataset):
        self.mlflow.log_params(config)
        self.mlflow.log_param('training_examples', len(dataset))
        self.mlflow.log_artifact(str(dataset.root/'manifest.json'), 'data')
        for name in ['cli.py', 'data.py', 'models.py', 'tracking.py', 'paths.py', '__init__.py']:
            self.mlflow.log_artifact(str(ROOT/'mri_synthesis'/name), 'implementation')
        self.mlflow.log_artifact(str(ROOT/'configs'/'quick_experiments.json'), 'implementation')
        self.mlflow.log_artifact(str(ROOT/'configs'/'quick_subjects.json'), 'implementation')

    def epoch(self, row):
        self.mlflow.log_metrics({k: v for k, v in row.items() if k != 'epoch'}, step=row['epoch'])

    def finish(self, reason, failed=False):
        self.mlflow.set_tag('stop_reason', reason)
        for name in ['config.json', 'losses.csv', 'losses.png', 'training_summary.json']:
            if (self.output/name).exists():
                self.mlflow.log_artifact(str(self.output/name))
        self.mlflow.set_tag('checkpoint_path', str((self.output/'final.pt').resolve()))
        self.mlflow.end_run(status='FAILED' if failed else 'FINISHED')


def log_evaluation(checkpoint_path, split, result_path, aggregate):
    info = Path(checkpoint_path).parent/'mlflow.json'
    if not info.exists():
        return
    import mlflow
    from mlflow.tracking import MlflowClient
    saved = json.loads(info.read_text())
    client = MlflowClient(tracking_uri=saved['tracking_uri'])
    for key, values in aggregate.items():
        for stat, value in values.items():
            if value is not None:
                client.log_metric(saved['run_id'], f'{split}_{key}_{stat}', value)
    client.log_artifact(saved['run_id'], str(result_path), artifact_path='evaluation')
