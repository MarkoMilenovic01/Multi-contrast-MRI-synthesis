# Prepared six-experiment project

Training is deliberately **not started** by setup or by preflight.

Open PowerShell in this directory and check readiness:

```powershell
python train_quick.py
```

When you choose to start training later:

```powershell
python train_quick.py --start
```

In another PowerShell window, open the local tracking dashboard:

```powershell
.\mlflow_ui.ps1
```

Visit **http://127.0.0.1:5000** and select **MRI section 8.3.1 - quick six experiments**. It remains empty until training is started; no placeholder runs or fabricated scores are created during preparation.

## Prepared dataset

| Split | Subjects | Source/target examples per subject | Total |
|---|---:|---:|---:|
| Training | 20 | 20 | 400 |
| Validation | 4 | 20 | 80 |
| Test | 4 | 20 | 80 |

`data/quick/manifest.json` records the exact subjects. They are selected deterministically within the original dataset's splits; no subject moves between splits. Existing completed registrations are reused when available. The separate `data/prepared` directory and all raw scans/archives are preserved.

The input resolution is **128x128**. Twenty evenly spaced center slices are selected from the central 60% of foreground-containing slices for each subject. Neighboring inputs are fetched from the full volume, not from other selected center slices, so three-slice inputs retain true anatomical adjacency. Registered pGAN and both cGAN modes share exactly the same T1 center examples. Unregistered T2 targets are selected independently within their own native volumes; they are never used for paired evaluation.

Inspect the alignment previews in `data/quick/registration/<subject>/alignment.png` before running experiments. The preparation validation checks geometry, finite tensors, counts, and split isolation; it cannot certify all anatomical alignments.

## Faster training configuration

`configs/quick_experiments.json` lists all six runs: pGAN, cGAN registered, and cGAN unregistered, each with one- and three-slice inputs.

- 20 epochs maximum: 10 constant-rate + 10 decay epochs.
- Batch size 1, seed 42, learning rate 0.0002.
- Base channel width **32 instead of 64**; keep all **nine residual blocks**, the PatchGAN discriminators, and each model's original loss terms.
- **22-minute training budget per model**, checked before each batch. Initialization counts toward this budget. Saving checkpoints, finalizing tracking, and plotting incur additional overhead.
- Up to **132 minutes of budgeted training** across all six configurations; the launcher reserves a **150-minute suite budget** and adjusts the next training budget for elapsed time. Validation, test evaluation, checkpoint writes, subprocess startup, and comparison rendering add overhead. This targets a 2–3 hour session on the available RTX 4060; it is not an exact runtime guarantee. No training-speed benchmark is run during this preparation.
- If a time budget is reached mid-epoch, save the available model as `final.pt` and mark the partial epoch, actual batch count, elapsed time, and stop reason. Runs can therefore complete different numbers of updates; report this alongside fidelity scores.

These are small-data, reduced-resolution project experiments. They preserve the six methods but should not be described as reproducing the book's image quality or exact original model capacity.

## MLflow and outputs

Tracking uses a local SQLite database at `tracking/mlflow.db` with local artifacts; no account, remote service, or running tracking server is needed for logging. The UI is a separate optional process. MLflow is already installed in the current Python environment; `requirements.txt` also declares it for a future installation.

One MLflow run is created per actual training experiment. It records configuration, dataset manifest, training-example count, epoch losses, learning rate, batches, partial-epoch status, and elapsed time. It stores configuration/log/curve/summary artifacts and tags the local checkpoint path. Checkpoints remain in `runs/quick/<name>` rather than being duplicated into MLflow storage. A persisted `mlflow.json` links interrupted training to its original run.

The launcher automatically evaluates final checkpoints on validation and test splits, attaches MAE/PSNR/SSIM and JSON reports to the same runs, and creates `results/quick/comparison.png`. Test results are for final reporting; do not use them to tune model settings. Model weights use source data only during inference. Evaluation and inference read the image size from each checkpoint, so quick models use the same 128x128 preprocessing throughout.

Rerunning `--start` skips configurations already containing `final.pt`, and resumes interrupted ones containing `last.pt`. A time-limited run with `final.pt` is considered complete; it is not automatically extended. Time-limited partial epochs are recorded as the final attempted epoch; resumption proceeds to the next epoch rather than replaying that partial epoch. Change settings only into a new output directory; resume requires matching architecture/data/schedule.

Pretrained VGG16 weights are cached during preparation so their initial download is outside the training budget. No synthetic MRI weights have been trained during setup.

To rebuild on another machine, download and extract IXI T1/T2 into `data/raw`, then run `python -m mri_synthesis.preparation`. The tracked `configs/quick_subjects.json` preserves the exact 20/4/4 subject selection. No original full-data manifest is needed. Existing registrations in `data/prepared` are reused if available; otherwise selected subjects are registered from raw NIfTI. An existing quick manifest is never silently overwritten.

MLflow API reference used: https://mlflow.org/docs/latest/ml/tracking/tracking-api/
