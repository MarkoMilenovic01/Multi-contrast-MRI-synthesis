# Section 8.3.1: multi-contrast MRI synthesis

**Historical reference only.** Raw and partial full-data directories have been removed. Commands assume execution from the project root. For the active small-data workflow use [QUICK_START.md](QUICK_START.md), or [TESTING.md](TESTING.md) to test the trained models.

Modern educational implementation of the **pGAN** and cycle-consistent **cGAN** described in the supplied pages of *Deep Learning for Medical Image Analysis*, second edition, section 8.3.1, equations 8.30–8.35 and Figure 8.5. This directory contains implementation instructions, not report prose.

Sources:

- Dar et al., *Image Synthesis in Multi-Contrast MRI With Conditional Generative Adversarial Networks*, IEEE TMI 2019. https://doi.org/10.1109/TMI.2019.2901750
- Authors' reference implementation: https://github.com/icon-lab/pGAN-cGAN
- IXI data and attribution/license: https://brain-development.org/ixi-dataset/

## Setup on Windows / NVIDIA

Use Python 3.11 or 3.12. Open PowerShell in this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install a matching torch/torchvision CUDA build using https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m unittest -v tests.test_pipeline
```

Tests use random VGG weights exclusively as offline fixtures, small channel widths, and synthetic inputs. Production pGAN downloads ImageNet VGG16 weights on first use (about 528 MB); weights remain frozen, but generated-image gradients propagate through the feature extractor. No pretrained synthesis model is included. Full training is computationally expensive, especially for all six configurations; no fixed GPU-memory or runtime guarantee is assumed.

## Obtain and prepare IXI

Download **IXI-T1.tar** and **IXI-T2.tar** through the IXI website above. The archives total roughly 8 GB; budget 100 GB free for the complete dataset, prepared arrays, and six experiments (actual use depends on slice counts). Prepared float32 arrays are uncompressed and memory-mapped to avoid repeatedly decompressing whole volumes during shuffled training. Review and comply with the dataset's license and attribution terms. Extract both archives under `data/raw`, preserving filenames such as `IXI002-Guys-0828-T1.nii.gz`. Downloads are deliberately explicit, not automatically started by installation.

```powershell
New-Item -ItemType Directory -Force data/raw
tar -xf path/to/IXI-T1.tar -C data/raw
tar -xf path/to/IXI-T2.tar -C data/raw
python run.py prepare --raw data/raw --output data/prepared
```

Preparation pairs by IXI subject ID, rejects duplicate scans, and splits paired subjects 70/15/15 with seed 42 **before** processing slices. One manifest is shared by every experiment. Failed subjects remain assigned to their original split and are recorded rather than silently reassigned. Missing counterparts are listed. Use a new output directory to rerun preparation.

T2 is rigidly registered to T1 in physical space with multiresolution Mattes mutual information and linearly resampled onto the T1 grid. The unregistered T2 branch retains its native geometry (only canonical axis orientation and image resizing are applied). Each volume uses positive-foreground 0.5/99.5-percentile normalization to [0,1]. Canonical axial slices are aspect-preserving resized/padded to 256x256; network inputs are mapped to [-1,1]. Slices with less than 5% foreground above 0.05 are excluded. Registered pairs require foreground in both images. Three-slice stacks retain adjacency and repeat edge slices at volume boundaries.

Inspect `data/prepared/registration/<subject>/alignment.png`, the saved registered NIfTI images, and `manifest.json` before training. Optimizer convergence alone does not establish anatomical alignment. Registration failures and unusable volumes are logged; preprocessing stops with an error if any split has no successful subjects. Review or correct unacceptable alignments and regenerate the dataset before making quantitative claims. Do not use an input directory containing previously registered copies as raw data.

## Train the six configurations

```powershell
python run.py train --data data/prepared --model pgan --slices 1 --output runs/pgan1 --device cuda
python run.py train --data data/prepared --model pgan --slices 3 --output runs/pgan3 --device cuda
python run.py train --data data/prepared --model cgan --branch registered --slices 1 --output runs/cgan_reg1 --device cuda
python run.py train --data data/prepared --model cgan --branch registered --slices 3 --output runs/cgan_reg3 --device cuda
python run.py train --data data/prepared --model cgan --branch unregistered --slices 1 --output runs/cgan_unreg1 --device cuda
python run.py train --data data/prepared --model cgan --branch unregistered --slices 3 --output runs/cgan_unreg3 --device cuda
```

`cGAN registered` means images were anatomically registered during preparation; source and target slices are still independently sampled during training. Neither cGAN variant uses paired pixel/perceptual supervision. Its target can come from another training subject. Held-out subjects never appear in the training sampler.

Each run saves configuration, all individual loss terms, loss curves, `last.pt` each epoch, and `final.pt` at completion. Resume an interrupted run with the same original arguments plus `--resume runs/<name>/last.pt`. Model/optimizer states, replay pools, CPU/CUDA random states, and Python sampler state are restored at epoch boundaries. The data directory must be unchanged. Resumption restarts an interrupted epoch; partial-epoch progress is not saved. Checkpoints should be loaded only from trusted sources, using the safe tensor loader built into this implementation.

Defaults: nine residual blocks, generator/discriminator base width 64, instance normalization, residual dropout 0.5, normal weight initialization (standard deviation 0.02), batch size 1, Adam lr 0.0002 and betas (0.5,0.999), 50 constant-rate epochs then 50 decreasing-rate epochs. The decay reaches zero after the final epoch; the last training epoch uses 1/50 of the initial rate. Replay pools hold up to 50 fake examples. Full precision is used to preserve a straightforward reference implementation; mixed precision is not enabled.

For a real-data short execution check, use a separate output with `--constant 1 --decay 0`. For an offline CPU smoke run, `python -m unittest -v tests.test_pipeline` exercises all six variants without downloading data or pretrained weights. The smoke test's reduced width is not a trained scientific model.

### Losses and channel arrangement

- **pGAN:** `G: k -> 1`; conditional discriminator sees concatenated `k+1` channels. Generator loss is `mean((D(x,G(x))-1)^2) + 100*L1(G(x),y) + 100*L1(V(G(x)),V(y))`. V is frozen ImageNet VGG16 through relu2_2, immediately before its second maxpool. Grayscale images repeat across RGB channels using the authors' [-1,1] convention, without adding ImageNet mean/std normalization.
- **cGAN:** `G: k -> 1`, `F: 1 -> k`; target discriminator sees one channel and source discriminator sees k. Generator loss sums the two least-squares adversarial losses plus `100*L1(F(G(x)),x) + 100*L1(G(F(y)),y)`. No identity loss is added.
- Each discriminator minimizes half the sum of its real-to-one and detached-fake-to-zero squared errors. The negative maximization expression printed in the book is implemented through the equivalent separate discriminator/generator minimization objectives; do not minimize that negative expression directly.

## Evaluate, infer, and compare

```powershell
python run.py evaluate --data data/prepared --checkpoint runs/pgan1/final.pt --output results/pgan1.json --device cuda
python run.py infer --source data/raw/IXI002-Guys-0828-T1.nii.gz --checkpoint runs/pgan3/final.pt --output results/synthetic_T2.nii.gz --device cuda
```

Run evaluation for each of the six final checkpoints with a distinct output filename. `--split val` enables development-set evaluation; default `test` is for final reporting. Training does not select checkpoints using the test set; comparisons use the fixed final epoch. MAE, PSNR (data range 1), and SSIM are calculated over the full normalized 256x256 slice, including padding/background. Scores are averaged within each subject, then across subjects with equal subject weight; standard deviations are across subjects. An identical-image PSNR is represented by JSON `null` meaning positive infinity. These metrics are not clinical validation and may be influenced by background.

All evaluation uses the same aligned held-out T1/T2 references, including evaluation of the model trained on unregistered images. Raw unregistered target slices must never be used for voxel-wise fidelity scoring. This evaluation protocol is an explicit adaptation and does not reproduce the paper's original unregistered-data scoring policy.

```powershell
python run.py compare --data data/prepared --subject IXI001 --pgan1 runs/pgan1/final.pt --pgan3 runs/pgan3/final.pt --cgan-reg1 runs/cgan_reg1/final.pt --cgan-reg3 runs/cgan_reg3/final.pt --cgan-unreg1 runs/cgan_unreg1/final.pt --cgan-unreg3 runs/cgan_unreg3/final.pt --output results/comparison.png --device cuda
```

Replace `IXI001` with a successfully prepared subject from the manifest's test split. `--slice N` selects a valid canonical axial slice; otherwise the middle valid slice is used. The resulting two-row figure compares single- and three-slice inputs with a shared source/reference. Checkpoint slots are validated to prevent mislabeled figures.

Inference accepts a T1 NIfTI only. It predicts slices, removes padding, resizes to the native in-plane dimensions, reverses canonical orientation, and preserves source shape, affine, qform, and sform. Output is **normalized synthesized T2 intensity [0,1]**, not an estimate of scanner-calibrated intensity. No target image, registration, or target-derived normalization is used at inference. Slice-wise generation can produce inconsistency through the volume and invent or remove structures; these outputs are for the project, not diagnosis.

## Fidelity and differences from the legacy code

Architecture, loss families, VGG feature location, neighboring-slice channel arrangement, and demonstration hyperparameters follow the authors' repository. The book calls its cycle-consistent model cGAN; this is distinct from pGAN's conditional discriminator setup. This implementation is independently organized for current Python/PyTorch, directly reads NIfTI rather than the legacy MATLAB arrays, and removes obsolete `Variable`, `volatile`, and forced-CUDA APIs. It fixes VGG batch indexing so every sample contributes, freezes VGG parameters while retaining input gradients, and disables dropout for deterministic inference. Stochastic noise z is not separately sampled: conditioning is by the MRI input, following the reference implementation.

The new IXI split, rigid registration, preprocessing, and evaluation protocol are adaptations, not an exact reproduction of the original data preparation or published results. The original experiments also investigated other datasets and synthesis directions; this project implements T1-to-T2 only. It includes no reproduced book figures, pretrained synthesis weights, or fabricated numerical results.
