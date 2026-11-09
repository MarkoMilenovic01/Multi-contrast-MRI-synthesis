# Test the quick project

Run these commands from this project directory. They do not train or resume models.

## 1. Check the prepared data and completed checkpoints

```powershell
python test_models.py --check-only
```

Checks all six final checkpoint files, the 400/80/80 example counts, and patient separation. Missing files or invalid splits stop with an error.

## 2. Evaluate all six models on held-out subjects

```powershell
python test_models.py --device cuda
```

Uses all **80 selected test slices from four unseen subjects**. For each model it loads `final.pt`, generates synthetic T2 images, compares them against aligned real T2 references, and saves MAE, PSNR, and SSIM. Lower MAE and higher PSNR/SSIM indicate closer image agreement. These metrics alone do not establish clinical correctness.

Outputs:

- `results/quick/<experiment>_test.json` — subject-level metrics and aggregate means/standard deviations.
- `results/quick/summary_test.csv` — all six models in one spreadsheet-friendly table.
- `results/quick/comparison.png` — source T1, pGAN, registered cGAN, unregistered cGAN, and real T2; rows compare one- versus three-slice inputs.
- Existing MLflow runs receive updated scores and report/figure artifacts; no new training runs are created.

Means are calculated within each subject first, then subjects receive equal weight. All images use [0,1] intensity and 128x128 quick preprocessing; padding/background are included. A PSNR JSON `null` means positive infinity for identical images. Evaluation compares every method with registered references, including models trained on unregistered data.

To evaluate validation and test data together:

```powershell
python test_models.py --split both --device cuda
```

This also creates `summary_both.csv`. Use validation for development and reserve test results for final reporting. Repeated evaluation does not retrain models, but tuning based on test results would compromise the held-out comparison.

## 3. View images and MLflow

Open `results/quick/comparison.png` or run:

```powershell
Invoke-Item results/quick/comparison.png
```

If the dashboard is not already running:

```powershell
powershell -ExecutionPolicy Bypass -File .\mlflow_ui.ps1
```

Open http://127.0.0.1:5000 and select **MRI section 8.3.1 - quick six experiments**. Select a run to view its test metrics and **Artifacts → evaluation** for figures/reports.

## 4. Synthesize a new T1 volume

### Browse every selected example

```powershell
python visualize_all.py --device cuda
Invoke-Item results/quick/all_examples/index.html
```

Creates **560 individual comparisons** (400 train, 80 validation, 80 test) and **112 contact sheets**, each displaying all six final models beside input T1 and real T2. The local HTML gallery has split/subject filters, a slice slider, and Previous/Next buttons; it needs no server. Only test subjects represent held-out final performance. The gallery uses final-checkpoint inference only and never trains. For test-only output, use `--split test`, which writes `results/quick/test_examples/index.html` instead.

### Apply a model to a new volume

If you have a separate T1 NIfTI scan:

```powershell
python run.py infer --source "C:\path\to\T1.nii.gz" --checkpoint runs/quick/pgan3/final.pt --output results/quick/new_synthetic_T2.nii.gz --device cuda
```

Inference needs only T1. It preserves the source grid and writes normalized synthetic T2 intensities. It cannot calculate fidelity metrics without a real, anatomically registered T2 reference. The original raw IXI directory has been removed, so replace the example source path with an actual existing scan. Trained checkpoint inference does not require pretrained VGG weights.

## 5. Check code without training

```powershell
python -m unittest -v tests.test_quick_setup
```

These five setup tests use fixture data and an isolated MLflow database; they do not update MRI model weights or create project training runs. `tests/test_pipeline.py` additionally contains synthetic optimizer-update tests, so do not run its full suite if you want zero optimizer updates.
