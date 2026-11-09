# Validation record

## Folder reorganization (November 8, 2026)

- Moved implementation into `mri_synthesis/`, settings into `configs/`, test suites into `tests/`, documentation/environment record into `docs/`, and disposable check logs into `logs/`.
- Kept root launchers for `run.py`, `train_quick.py`, `test_models.py`, and `visualize_all.py`. Updated package imports, configuration discovery, source-artifact logging, and documentation commands.
- Did not move or modify dataset arrays, trained checkpoints, results/gallery files, or MLflow storage. The local database URI is unchanged; historical source artifacts remain intact.
- Eight non-training checks passed after the move. Verified all four launchers, dataset/checkpoint preflight, and loading all six real final generators. The 560-example gallery remains in its existing location. No model training was started.

## Post-training cleanup and evaluation audit (November 8, 2026)

- Found all six completed `final.pt` checkpoints; their paired `last.pt` copies have matching serialized ZIP entry names, sizes, and CRCs. Marked redundant copies and optional files in DELETE_CANDIDATES.md. No data/results/checkpoints were deleted.
- Added `test_models.py` and TESTING.md; both `--check-only` and the default CUDA evaluation completed successfully. Evaluation used 80 selected slices from four held-out subjects per model, updated the existing MLflow runs, and generated `results/quick/summary_test.csv` and the complete six-model `comparison.png`.
- This audit used inference only; it did not start or resume training. Successful execution means the evaluation pipeline ran, not that every model has acceptable quality. In particular, the three-slice unregistered model has poor fidelity in this run; retain its results for an honest six-model comparison.
- The main README now describes the active quick project; the original full-data README is archived under docs/FULL_WORKFLOW_REFERENCE.md. Historical preparation records below describe their state at the time and are superseded by this audit where applicable.

## Quick six-experiment preparation (November 8, 2026)

- Prepared real IXI data separately in `data/quick`: 20 training, 4 validation, and 4 test subjects; 20 selected center slices per subject; 128x128 resolution. All 28 succeeded, with zero recorded failures. Existing raw data and the partial full-data preparation remain intact.
- Preflight verified all six configurations, 400/80/80 example counts, source/target tensor shapes, finite values, no patient overlap, CUDA availability, and cached pretrained VGG16 weights. Weights were downloaded and hash-checked without training a model. One newly registered test-subject preview was visually inspected; inspect the remaining previews before training.
- Five new preparation tests passed: slice selection/true adjacency, configuration matrix, no-training launcher default, MLflow parameter/artifact/metric logging and resume in an isolated test database, and a mocked time-budget stop with no optimizer updates. Three existing checks for data/evaluation/comparison/NIfTI inference, subject splitting, and metrics also passed without training updates.
- Initialized the real local MLflow experiment and verified it contains **zero runs**. Test fixture runs were isolated from that database. The current environment includes MLflow 3.9.0; full package and GPU details are recorded in `docs/environment_quick.json`.
- Quick configurations retain nine residual blocks and use width 32. They have a maximum of 20 epochs and a 22-minute per-experiment budget checked between batches. Runtime and real-image quality have **not** been measured by training in this preparation. See `QUICK_START.md` for runtime scope and commands.

The historical validation below predates the explicit instruction not to train. **No model optimization was performed during this quick-workflow preparation.**

## Original implementation checks

Checked on November 8, 2026, using Python 3.11.0, PyTorch 2.9.1+cu130, torchvision 0.24.1+cu130, NumPy 2.2.6, NiBabel 5.3.3, SimpleITK 2.5.3, scikit-image 0.25.2, and matplotlib 3.10.7.

- `python -m unittest -v tests.test_pipeline`: **7 tests passed** (about 34 seconds).
- Tests cover all six configurations with nine residual blocks and reduced channel widths; both cycle paths; generator/reverse-generator updates; frozen perceptual-network parameters with gradients through both batch examples; checkpoint and optimizer restoration; deterministic subject splits; registration on a synthetic volume; complete preparation of seven synthetic subjects; both data branches; metrics; comparison rendering; and NIfTI shape/affine preservation.
- `python -m py_compile models.py data.py run.py test_pipeline.py` and CLI help passed.
- Additional full-width, batch-one, three-slice, 256x256 CUDA training updates passed for both model families on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB). Measured peak allocated tensor memory was approximately 1.7 GB for cGAN and 0.6 GB for pGAN; this is not a guarantee of total GPU memory use.
- The full-width CUDA training entrypoint was interrupted after epoch one, resumed from its checkpoint, and completed epoch two with loss logs and a final checkpoint, using synthetic 32x32 fixtures.

All numerical smoke-test inputs were synthetic. Perceptual tests and the CUDA pGAN update used randomly initialized VGG16 features to avoid downloading weights; production pGAN requests frozen pretrained ImageNet VGG16. The test suite's generated images and scores are fixtures, not research results, and its temporary artifacts are removed automatically.

**Not yet performed:** IXI download, real-data registration review, production training with pretrained VGG16, completed six-experiment training, or held-out medical-image quality assessment. No claim is made to reproduce the book's images or published scores. Instructions for those steps are in README.md.
