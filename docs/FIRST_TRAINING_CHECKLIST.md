# First IQA Training Checklist

Official, step-by-step checklist for the first real Image Quality Assessment training run --
and for every subsequent training run of any stage, once implemented. Run through this in order;
do not skip to "Full 50-epoch launch" without completing the smoke test steps first.

**Stage 1 has already completed this checklist successfully** -- see "Completed Run Record"
at the end of this document for the verified baseline results. This checklist remains the
official process for every future training run (Stage 1 re-runs, or any other stage once
implemented), not a one-time record.

Every step below maps to a real cell in `colab/notebooks/stage01_iqa.ipynb` and a real function in
`colab/common/`. Nothing here is aspirational -- if a step's underlying code doesn't exist yet
for a future stage's notebook, that stage isn't ready for this checklist yet.

---

## 1. Repository Clone

- [ ] Open `colab/notebooks/stage01_iqa.ipynb` in Google Colab.
- [ ] `Runtime > Change runtime type > Hardware accelerator > GPU` (T4 or better), **before**
      running any cell -- changing runtime type after cells have run resets the VM.
- [ ] Run the **Bootstrap** cell. Confirm output ends with `Bootstrap complete: /content/diabetic_retinoplasty`.
- [ ] If it fails: check for a git error in the output (network issue, wrong branch, or the repo
      URL in the Bootstrap cell no longer matches `colab/common/colab_config.py`'s `REPO_URL`).

## 2. Drive Mount

- [ ] Run **Section 1 (Setup)** -- calls `setup.setup()`.
- [ ] Accept the Google Drive authorization prompt when it appears.
- [ ] Confirm the printed summary shows `repo_dir`, `drive_mount_point`, `env_vars`, and
      `session_log` -- and that `session_log` points at a real path under
      `MyDrive/DiabeticRetinopathy/logs/`.
- [ ] Confirm your Drive actually contains `MyDrive/DiabeticRetinopathy/datasets/EyeQ/raw/{train,test}`
      **before** proceeding -- see `colab/README.md`'s Google Drive layout if unsure.

## 3. GPU Verification

- [ ] Run **Section 2 (Verification)**'s first cell -- calls `verify_environment.verify_all(...)`.
- [ ] Confirm `[OK] GPU available: <device name> (1 device(s))` appears in the output.
- [ ] If it raises "No GPU detected": go back to Step 1's runtime-type change, then re-run from
      the Bootstrap cell (a runtime change resets everything after it).

## 4. Environment Verification

- [ ] Same cell as Step 3 -- confirm every one of these lines appears with `[OK]`, not `[WARN]`
      or a raised exception: Python version, TensorFlow version, repository path, Google Drive
      mounted, required packages, GPU available, CUDA version.
- [ ] `env_report` (the cell's return value) should be a dict with `python_version`,
      `tensorflow_version`, `gpu_count`, `gpu_name`, `cuda_version`, `mixed_precision_policy`.

## 5. Mixed Precision

- [ ] Still within Step 3/4's verification output: confirm `[OK] Mixed precision policy: mixed_float16`
      (on a GPU runtime -- `float32` is expected and correct only if you deliberately ran without
      a GPU, which should not be the case for a real training run).

## 6. Dataset Verification

- [ ] Run **Section 2 (Verification)**'s second cell -- calls `verify_dataset.verify_eyeq_dataset()`.
- [ ] Confirm both `[train]` and `[test]` lines report `missing images: 0` and
      `corrupted (of 50 sampled): 0` (a small nonzero corrupted count is not automatically fatal --
      it aborts on its own past a 20% threshold -- but investigate before proceeding if you see any).
- [ ] Confirm the printed train/test class distributions look plausible (`Good` should be the
      large majority in both splits, per the dataset's known imbalance).
- [ ] Confirm "Dataset verification passed." is the last line.

## 7. Model Creation

- [ ] Run **Section 3 (Dataset Loading)** -- confirm `Experiment root:` prints a real, new,
      timestamped path under `experiments/IQA/` on Drive, and `Class weights (train split):`
      prints a dict with keys `0`, `1`, `2`.
- [ ] Run **Section 4 (Model Creation)** -- confirm `model.summary()` prints the full
      EfficientNetB0 + head architecture with no errors, and note the trainable/non-trainable
      parameter counts for sanity (should be a few million trainable, ~4M+ total).

## 8. 2-Epoch Smoke Test

**Do this before the full run.** In **Section 3**, temporarily set `EPOCHS = 2`, then run
**Section 5 (Training)**.

- [ ] Confirm training starts (`Epoch 1/2` appears) without an exception.
- [ ] Confirm both epochs complete and `Best model exported to: ...` prints a path ending in
      `exported_models/IQA/best_model.keras`.
- [ ] Confirm no `NaN`/`Inf` loss values appear in the per-epoch output.
- [ ] **Reset `EPOCHS = 50`** (or your intended value) before re-running Section 5 for the real
      run -- do not forget this, or the "full launch" will silently only run 2 epochs.

## 9. Checkpoint Verification

- [ ] In Google Drive, open the experiment folder printed in Step 7 (`experiments/IQA/<timestamp>/`).
- [ ] Confirm `checkpoints/best.keras`, `checkpoints/last.keras`, `checkpoints/metrics.csv`, and
      `checkpoints/epoch_state.json` all exist and are non-empty.
- [ ] Open `epoch_state.json` and confirm `last_completed_epoch` matches the number of epochs the
      smoke test actually ran.

## 10. TensorBoard Verification

- [ ] Run the **TensorBoard** cell immediately after Section 5. Confirm the TensorBoard UI loads
      inline and shows a `loss`/`val_loss` curve with 2 data points (from the smoke test).
- [ ] Confirm `experiments/IQA/<timestamp>/logs/` on Drive contains real TensorBoard event files
      (non-empty, `events.out.tfevents.*` naming).

## 11. Evaluation Verification

- [ ] Run **Section 6 (Evaluation)**. Confirm `evaluate()` completes and prints real accuracy/
      precision/recall/F1/AUC/QWK numbers (not zero across the board, which would suggest a
      broken label/prediction alignment rather than just "an undertrained 2-epoch model").
- [ ] Confirm `confusion_matrix.png`, `roc_curves.png`, `calibration_curve.png`,
      `evaluation_report.json`, and `training_history.png` all appear under
      `experiments/IQA/<timestamp>/evaluation/` on Drive, and that the sample-predictions grid
      saves to `experiments/IQA/<timestamp>/predictions/sample_predictions.png`.
- [ ] This is still the 2-epoch smoke-test model -- expect mediocre metrics. The goal here is
      confirming the evaluation *pipeline* runs end-to-end without error, not good numbers yet.

## 12. Full 50-Epoch Launch

Only after every step above has passed:

- [ ] Confirm `EPOCHS` is reset to your real intended value (Step 8's reminder).
- [ ] Decide whether to keep `RESUME_EXPERIMENT_DIR = None` (a fresh experiment folder) or start
      from the smoke-test experiment -- **recommended: leave it `None`** and let a new,
      independent experiment folder be created, since the smoke test's 2-epoch checkpoint isn't a
      meaningful starting point.
- [ ] Re-run **Section 3 (Dataset Loading)** to create the new experiment folder, then **Section 4
      (Model Creation)**, then **Section 5 (Training)**.
- [ ] Expect early stopping (`patience=8` on `val_loss`) to end the run before 50 epochs if
      validation loss plateaus -- this is expected behavior, not a failure.
- [ ] **Known limitation to be aware of, not a blocker:** if you use `RESUME_EXPERIMENT_DIR` on a
      *future* run to continue this one, model weights restore correctly but the Adam optimizer's
      momentum/variance state does not (a Keras 3 optimizer-variable-count mismatch in
      `training/trainer.py`'s resume path) -- expect a small, temporary training-quality dip right
      at the resume boundary, not a crash or wrong results.

## 13. Expected Outputs

After a completed full run, confirm every one of these exists on Drive under
`experiments/IQA/<timestamp>/`:

- [ ] `checkpoints/best.keras`, `checkpoints/last.keras`, `checkpoints/metrics.csv`, `checkpoints/epoch_state.json`
- [ ] `logs/` (TensorBoard event files) and `tensorboard/` (archival copy, populated by the
      Training section's `archive_tensorboard_logs()` call)
- [ ] `evaluation/confusion_matrix.png`, `evaluation/roc_curves.png`, `evaluation/calibration_curve.png`,
      `evaluation/training_history.png`, `evaluation/evaluation_report.json`
- [ ] `predictions/sample_predictions.png`
- [ ] `metadata.json` with a real (non-null) `git_commit_hash`, `gpu_model`, and your actual
      hyperparameters
- [ ] `exported_models/IQA/best_model.keras` on Drive (stable path, outside the experiment folder)

## 14. Post-Training Review

- [ ] Read `metadata.json` -- confirm `git_commit_hash` matches the commit you intended to train
      (i.e., you didn't accidentally train against uncommitted local changes never pushed to
      GitHub, since Colab only ever sees what's on the remote).
- [ ] Compare `evaluation_report.json`'s test-split metrics against the smoke test's (Step 11) --
      confirm they're meaningfully better, not just different.
- [ ] Skim `predictions/sample_predictions.png` for any obviously wrong predictions on images that
      look unambiguous by eye -- a useful sanity check beyond aggregate metrics.
- [ ] Decide whether this run's `exported_models/IQA/best_model.keras` should be committed back
      into the repository (see `colab/README.md`'s optional, disabled-by-default commit/push
      step) -- review the file yourself first; this is a deliberate, manual decision, never
      automatic.
- [ ] Once satisfied with real results, update `README.md`'s Results section (currently
      explicitly flagged as placeholder/illustrative baseline data) -- do not leave fabricated or
      stale numbers in place of real ones.

---

## Completed Run Record -- Stage 1 Baseline (2026-08-05)

This checklist has been followed through to a completed, verified Stage 1 run. Recorded here as
a real example, and as the official Stage 1 baseline -- this section documents what happened, it
does not replace the checklist above for future runs (of Stage 1 or any other stage).

**Experiment:** `experiments/IQA/2026-08-05_09-11-28` (Google Drive)

**Held-out test-split metrics** (Step 11/13, EyeQ test split, 16,249 images):

| Metric | Value |
|---|---|
| Accuracy | 88.05% |
| F1 Score | 86.12% |
| AUC | 96.48% |
| Quadratic Weighted Kappa (QWK) | 0.8987 |

No additional metrics beyond these four are claimed as part of this baseline.

**Training behavior:**
- EfficientNetB0 with ImageNet-pretrained weights (first 100 layers frozen); 4,378,278 total
  parameters.
- Mixed precision (`mixed_float16`) enabled throughout (GPU: Tesla T4).
- Best validation loss (0.17268) at **Epoch 2**; `ReduceLROnPlateau` reduced the learning rate
  twice (`1e-4 -> 5e-5` at Epoch 6, `5e-5 -> 2.5e-5` at Epoch 10); `EarlyStopping` (patience 8,
  monitoring `val_loss`) stopped training at Epoch 10 and restored the Epoch 2 weights.
- The exported model corresponds exactly to the Epoch 2 best-validation checkpoint, not the
  final (Epoch 10) weights.

**Output locations (Google Drive):**

```
MyDrive/
└── DiabeticRetinopathy/
    ├── exported_models/
    │   └── IQA/
    │       └── best_model.keras          -- 85.26 MB, this run's best checkpoint
    └── experiments/
        └── IQA/
            └── 2026-08-05_09-11-28/
                ├── checkpoints/          best.keras, last.keras, metrics.csv, epoch_state.json
                ├── logs/                 TensorBoard event files
                ├── tensorboard/          archival copy of logs/
                ├── evaluation/           confusion matrix, ROC curves, calibration, training history
                ├── predictions/          sample-prediction visualizations
                └── metadata.json         run metadata (git commit, hyperparameters, versions)
```

**Repository status:** Stage 01 (Image Quality Assessment) -- Completed, Verified, Baseline
Established. Stage 02 (Image Preprocessing) -- Ready to Begin.
