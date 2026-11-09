# README example assets

These two figures come from the existing final checkpoints and local all-example gallery. No models were retrained to create this documentation update.

- Selection: middle selected axial slice from each of the first two test subjects, IXI236 and IXI186. Both subjects belong exclusively to the held-out test split in `configs/quick_subjects.json`. Exact indices and filenames are recorded in `examples.json`.
- Columns: input T1; pGAN single-slice; pGAN three-slice; registered cGAN single-slice; registered cGAN three-slice; unregistered cGAN single-slice; unregistered cGAN three-slice; real T2 reference.
- Source/reference images were normalized, registered where applicable, resized/padded, and assembled alongside generated predictions and labels. The generator receives only T1 at inference; the real T2 is a visual/evaluation reference.
- Figures intentionally retain model failures and artifacts. They are illustrative examples; the CSV summarizes all 80 test examples from four subjects.
- `test_metrics.csv` is copied from the completed local evaluation. Its means and standard deviations are computed across subject-level scores, not individual slices. The underlying dataset and model weights are not included in Git.

## Attribution and figure license

Original source: **IXI Dataset — Information eXtraction from Images (EPSRC GR/S21533/02)**, via https://brain-development.org/ixi-dataset/.

The IXI data is provided under **Creative Commons Attribution-ShareAlike 3.0 Unported**: https://creativecommons.org/licenses/by-sa/3.0/ (legal text: https://creativecommons.org/licenses/by-sa/3.0/legalcode).

The adapted example PNG figures in this folder are distributed under **CC BY-SA 3.0**, with the same source attribution. This figure-specific notice does not assign a software license to the rest of the repository.
