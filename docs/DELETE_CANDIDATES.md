# Cleanup checklist — marked, not deleted

The project was checked after all six quick experiments completed. No data, checkpoints, or results were deleted by this cleanup.

## Can delete now

The following checkpoint pairs were checked by comparing their ZIP entry names, sizes, and CRCs; each `last.pt` has the same serialized contents as its adjacent `final.pt`. Keep every **final.pt**. Deleting these `last.pt` copies reclaims approximately **462 MiB**:

- [ ] `runs/quick/pgan1/last.pt`
- [ ] `runs/quick/pgan3/last.pt`
- [ ] `runs/quick/cgan_reg1/last.pt`
- [ ] `runs/quick/cgan_reg3/last.pt`
- [ ] `runs/quick/cgan_unreg1/last.pt`
- [ ] `runs/quick/cgan_unreg3/last.pt`

This recommendation applies to these completed runs. Do not delete `last.pt` from an interrupted or future active run. Keep `final.pt` even if deleting `last.pt`; the quick launcher uses final checkpoints to recognize completed runs.

Other optional cleanup:

- [ ] `logs/setup_checks.log` — old setup-test output.
- [ ] `__pycache__/` — regenerated automatically by Python.
- [ ] `mri_synthesis/preparation.py` — optional rebuilding helper, not used for training/evaluation of the existing prepared data. Keep it for a reusable source repository: with raw IXI scans and `configs/quick_subjects.json`, it can recreate the quick dataset without a full-data manifest.
- [ ] `results/quick/examples_now.png` — intermediate five-model preview; `comparison.png` contains all six models. Keep the preview if useful in your project history.

## Optional larger reductions

- [ ] `data/quick/registration/` — approximately **571 MiB** of saved registration volumes and alignment previews; quick training/evaluation load the prepared arrays in train/val/test instead. Keep these until you have inspected alignment or if documenting preprocessing. Once removed, visual alignment review from these files requires restoring/rebuilding them.

## Keep

- Runtime: `mri_synthesis/` (except the optional preparation helper noted above), `run.py`, `train_quick.py`, `test_models.py`, `visualize_all.py`, `configs/quick_experiments.json`, `mlflow_ui.ps1`, `requirements.txt`.
- `data/quick/manifest.json` and the complete `data/quick/train/`, `val/`, `test/` trees (including all `.npy` and `indices.npz` files).
- All six `runs/quick/*/final.pt` files; `mlflow.json` connects each run to its MLflow history. Keep config, training summaries, loss logs/plots to document results.
- `tracking/` — real experiment database, local artifacts, and active dashboard logs. Do not remove database files or their WAL/SHM companions while the dashboard is running.
- `results/quick/` evaluation JSON/CSV reports and `comparison.png`.
- `tests/test_pipeline.py` and `tests/test_quick_setup.py` — small verification suites. Some pipeline tests perform synthetic optimizer updates; use the no-training commands in TESTING.md when only testing trained models.
- Documentation and environment record: small and useful for reproducibility.
- `.gitignore`.

After the folder reorganization, all disposable check logs live in `logs/`. The complete `logs/` directory can be deleted. Python may regenerate `__pycache__/` under the root, package, and tests; all are disposable caches.

The large `data/raw/`, `data/prepared/`, and IXI archives are no longer present inside this project. No further large raw-data cleanup is needed here. Cached pretrained VGG16 weights outside this repository remain needed for any future pGAN training, but not for final-checkpoint inference.
