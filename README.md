# Medical MRI synthesis — six quick experiments

Implementation of section 8.3.1 of *Deep Learning for Medical Image Analysis*, second edition: T1-to-T2 synthesis using paired pGAN and cycle-consistent cGAN. All six quick experiments now have final checkpoints.

| Model | Training data | Input modes |
|---|---|---|
| pGAN | Registered paired T1/T2 | One slice, three slices |
| cGAN | Registered data, independently sampled | One slice, three slices |
| cGAN | Unregistered data, independently sampled | One slice, three slices |

Quick data: **20/4/4 subjects**, **400/80/80 selected examples** for training/validation/test, **128x128** inputs. Networks use width 32 and nine residual blocks. The model families and losses follow the book; dataset size, resolution, model width, and training duration are reduced for this educational project.

## Setup from GitHub

The repository contains source code, configuration, tests, and documentation. MRI scans, prepared arrays, trained weights, generated images, and the local MLflow database are excluded by `.gitignore`. Completed models/results described below refer to the local project, not files bundled with a fresh clone.

Use Python 3.11 or 3.12. From the cloned repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install matching CUDA torch/torchvision builds from https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -m unittest -v tests.test_quick_setup
```

Download IXI T1 and T2 archives from the dataset website below, comply with its license, and extract them into `data/raw`. Prepare only the selected 28 subjects:

```powershell
python -m mri_synthesis.preparation
```

The tracked `configs/quick_subjects.json` records the exact split; no local full-data manifest is required. Inspect registration previews before training. Cache the official pretrained VGG16 feature weights (approximately 528 MB, hash checked), then check/start the suite:

```powershell
python -c "import torch; torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth', map_location='cpu', check_hash=True)"
python train_quick.py
python train_quick.py --start
```

Training requires CUDA for this quick launcher; model evaluation also supports CPU. The original 2–3 hour budget is a target for the local RTX 4060 setup, not a speed guarantee on other hardware.

## Test the trained models

From this directory:

```powershell
python test_models.py --check-only
python test_models.py --device cuda
```

These commands never train models. Open `results/quick/summary_test.csv` for all six scores and `results/quick/comparison.png` for generated examples.

To view every selected example, open `results/quick/all_examples/index.html`. Rebuild the full local gallery with `python visualize_all.py --device cuda` (inference only): 560 comparisons and 112 contact sheets across all three splits, with all six models on every row.

**[Testing instructions](docs/TESTING.md)** explain evaluation, metrics, images, MLflow, new-volume inference, and code checks.

## Folder structure

```text
.
├── mri_synthesis/              # Application implementation
│   ├── data.py                # Preprocessing and slice datasets
│   ├── models.py              # Generators, discriminators and losses
│   ├── cli.py                 # Individual prepare/train/evaluate/infer commands
│   ├── suite.py               # Six-experiment training orchestration
│   ├── evaluation.py          # Test all six completed models
│   ├── visualization.py       # All-example gallery and contact sheets
│   ├── tracking.py            # MLflow integration
│   ├── paths.py               # Shared project/configuration paths
│   └── preparation.py         # Optional original-data rebuilding helper
├── configs/
│   ├── quick_experiments.json
│   └── quick_subjects.json     # Exact 20/4/4 subject selection
├── tests/
│   ├── test_quick_setup.py
│   └── test_pipeline.py
├── docs/                      # Guides, cleanup checklist and validation records
├── data/quick/                # Prepared data; keep in place
├── runs/quick/                # Trained checkpoints and logs; keep in place
├── results/quick/             # Scores, plots and all_examples/index.html
├── tracking/                  # MLflow database and artifacts; keep in place
├── logs/                      # Disposable setup/check logs
├── run.py                     # Small command launcher
├── train_quick.py             # Small training launcher
├── test_models.py             # Small evaluation launcher
├── visualize_all.py           # Small gallery launcher
├── mlflow_ui.ps1
├── requirements.txt
└── README.md
```

Root Python files are compatibility launchers; implementation lives in the package. No installation step is needed when running from this project directory. Data and experiment folders retain their original paths so existing checkpoints, gallery links, and MLflow history continue to work.

## Repository map

| Location | Purpose |
|---|---|
| `data/quick/` | Shared prepared data, fixed subject split and slice indices |
| `runs/quick/<experiment>/` | Final models, configuration, training logs and MLflow link |
| `results/quick/` | Scores, summary tables and images |
| `tracking/` | Local MLflow SQLite database and artifacts |
| `mri_synthesis/` | Model, data, CLI, evaluation, visualization and tracking implementation |
| `train_quick.py`, `configs/quick_experiments.json` | Six-experiment training launcher and settings |
| `test_models.py` | All-six evaluation and comparison; no training |
| `visualize_all.py` | All-example image gallery and contact sheets; no training |
| `mri_synthesis/tracking.py`, `mlflow_ui.ps1` | Experiment logging and local dashboard |
| `tests/` | Setup and implementation checks |

**[Cleanup checklist](docs/DELETE_CANDIDATES.md)** marks redundant checkpoints and optional cleanup. Nothing is automatically deleted. Keep the final checkpoints, selected data, MLflow history, and results.

## MLflow

```powershell
powershell -ExecutionPolicy Bypass -File .\mlflow_ui.ps1
```

Open http://127.0.0.1:5000 and select **MRI section 8.3.1 - quick six experiments**. The dashboard runs with one worker to avoid the observed Windows multi-worker startup failure. Training and evaluation log directly to the local SQLite database; a dashboard is not required for logging.

## Training and reproducibility

See **[Quick start](docs/QUICK_START.md)** for training configuration and budget details. `python train_quick.py` is preflight only; `--start` explicitly starts the suite and skips models with existing final checkpoints. Testing uses final checkpoints and does not restart training. New experiments with changed settings should use new output directories.

`requirements.txt` lists dependencies; the local-only `docs/environment_quick.json` records the original prepared environment and is ignored by Git. [Validation records](docs/VALIDATION.md) describe checks. The historical [full-data workflow](docs/FULL_WORKFLOW_REFERENCE.md) is retained for reference. The quick preparation helper now rebuilds the selected dataset directly from raw IXI scans and the tracked subject list.

Check setup without training:

```powershell
python -m unittest -v tests.test_quick_setup
```

## Sources

- Dar et al., *Image Synthesis in Multi-Contrast MRI With Conditional Generative Adversarial Networks*, IEEE TMI 2019: https://doi.org/10.1109/TMI.2019.2901750
- Authors' implementation: https://github.com/icon-lab/pGAN-cGAN
- IXI dataset (acknowledge its source and comply with CC BY-SA 3.0): https://brain-development.org/ixi-dataset/

Synthetic images can distort anatomy. Small-data demonstration results do not establish clinical reliability or reproduce the book's published performance.
