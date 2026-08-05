# Colab Training Workflow

Official Google Colab training environment for this repository. Every trainable pipeline module
(Image Quality Assessment today; Vessel Segmentation, Lesion Segmentation, and Final
Classification once implemented) trains here, against datasets stored in Google Drive, with all
outputs written back to Drive so nothing important is lost when a Colab VM recycles.

This directory holds only Colab-specific orchestration code. Actual model/dataset/training/
evaluation logic lives in the repository root and in `training/` / `evaluation/` -- nothing here
duplicates it; every module below calls into that existing code.

---

## Folder structure

```
colab/
├── README.md              -- this file
├── colab_config.py         -- every path this workflow uses, in one place
├── drive_paths.py          -- the Google Drive folder layout, as data (no side effects)
├── environment.py          -- environment inspection/setup primitives (GPU, TF, CUDA, packages)
├── setup.py                -- one-call setup: mount Drive, clone repo, install deps, chdir, env vars
├── verify_environment.py   -- aborting environment checks, built on environment.py
├── verify_dataset.py       -- aborting EyeQ dataset checks (structure, counts, corruption)
├── experiment_manager.py   -- isolated, timestamped experiment folders + metadata.json
└── notebooks/
    └── 01_image_quality_assessment.ipynb   -- the actual training notebook
```

### Purpose of each file

| File | Purpose |
|---|---|
| `colab_config.py` | Single source of truth for every path (Drive root, dataset root, experiment root, TensorBoard root, exported-model root, repository root). Pure data -- no Drive mount, no cloning. Every other module and the notebook import paths from here instead of hardcoding them. |
| `drive_paths.py` | Defines the verified Google Drive directory tree (see below) and resolves it into concrete paths. Pure path logic, no `google.colab` import, so it's importable and testable outside Colab. |
| `environment.py` | Low-level, non-aborting info-gathering and setup actions: Python/TensorFlow/Keras/CUDA versions, GPU device/name/memory, git commit hash, missing-package detection. Reuses `training.check_gpu` / `training.enable_mixed_precision` rather than reimplementing them. |
| `setup.py` | The one function (`setup.setup()`) the notebook calls to mount Drive, clone/update the repository, install `requirements.txt`, `cd` into the repository, and point `EYEQ_RAW_DIR` at Drive. |
| `verify_environment.py` | Pass/fail checks built on `environment.py`: Python version, TensorFlow version, GPU availability, CUDA, mixed precision, repository path, Google Drive mount, required packages. Every check raises `RuntimeError` with a specific message on failure -- nothing here is caught or swallowed. |
| `verify_dataset.py` | Verifies the EyeQ dataset: directory/`labels.csv`/image-folder existence, image counts, missing images, a corrupted-image spot check, and class distribution. Reuses `image_quality_dataset._read_labels` / `_decode_image` directly rather than re-implementing file/JPEG checks. Returns a structured `EyeQVerificationReport`; raises on any failed check. |
| `experiment_manager.py` | Creates a new, isolated, timestamped folder per training run under `experiments/IQA/`, with `checkpoints/`, `logs/`, `tensorboard/`, `evaluation/`, `predictions/` subfolders and a `metadata.json` (git commit, timestamp, TF/Python versions, GPU model, and the run's hyperparameters). Also supports resuming into an existing experiment folder. |
| `notebooks/01_image_quality_assessment.ipynb` | The actual notebook. Thin orchestration only: Setup -> Verification -> Dataset Loading -> Model Creation -> Training -> Evaluation -> Export. |

---

## Google Drive layout

This workflow assumes your Google Drive already contains **exactly** this structure. Nothing
here creates, assumes, or depends on any folder beyond it:

```
MyDrive/
└── DiabeticRetinopathy/
    ├── datasets/                      (real data -- never created or modified by this workflow)
    │   ├── EyeQ/
    │   │   ├── raw/
    │   │   │   ├── train/{images/, labels.csv}
    │   │   │   └── test/{images/, labels.csv}
    │   │   └── processed/
    │   ├── APTOS2019/
    │   └── IDRiD/
    │
    ├── experiments/                   (auto-created)
    │   ├── IQA/
    │   │   └── YYYY-MM-DD_HH-MM-SS/   (one folder per training run, never overwritten)
    │   │       ├── checkpoints/       best.keras, last.keras, metrics.csv, epoch_state.json
    │   │       ├── logs/              TensorBoard event files (the live --logdir)
    │   │       ├── tensorboard/       archival copy of logs/, written after training finishes
    │   │       ├── evaluation/        confusion matrix, ROC curves, calibration, training history
    │   │       ├── predictions/       sample-prediction visualizations
    │   │       └── metadata.json      run metadata (see experiment_manager.py)
    │   ├── VesselSegmentation/
    │   ├── LesionSegmentation/
    │   └── FinalClassification/
    │
    ├── tensorboard/                   (auto-created; global root, reserved for cross-experiment use)
    │
    ├── exported_models/               (auto-created)
    │   └── IQA/
    │       └── best_model.keras       (the current best model for this module -- stable path,
    │                                    overwritten by each run, distinct from the per-experiment
    │                                    checkpoints/ archive above)
    │
    └── logs/                          (auto-created; session-level setup logs, see setup.py)
```

`datasets/*/raw` is **never** written to by anything in `colab/` -- it must already contain real,
uploaded data. `experiments/`, `tensorboard/`, `exported_models/`, and `logs/` are created
automatically (`drive_paths.ensure_output_directories()`, called by `setup.setup()`) if missing.

**Why `logs/` and `tensorboard/` both exist inside each experiment:** the existing
`training.trainer.TrainingConfig` (which this refactor intentionally does not modify) always
writes TensorBoard's event files to `<run_dir>/logs`. `tensorboard/` is filled by
`experiment_manager.archive_tensorboard_logs()` as a one-time copy after training completes, for
anyone who expects a folder literally named `tensorboard/`. **`logs/` is the live directory** --
point TensorBoard there during or immediately after a run (the notebook does this automatically).

---

## How to open the notebook

1. Upload `colab/notebooks/01_image_quality_assessment.ipynb` to Google Colab (File > Upload
   notebook), or open it directly from GitHub (File > Open notebook > GitHub, paste this repo's
   URL and select the file).
2. `Runtime > Change runtime type > Hardware accelerator > GPU` (T4 or better recommended).

## How to start training

Run the notebook's cells top to bottom. No manual edits are required if your Drive matches the
layout above. The notebook will:

1. **Setup** -- mount Drive, clone the repo, install requirements, `cd` into it.
2. **Verification** -- abort with a clear error if the environment or dataset isn't ready.
3. **Dataset Loading** -- pick a batch size, create a new experiment folder, load EyeQ.
4. **Model Creation** -- build and preview the EfficientNetB0 architecture.
5. **Training** -- call `train_image_quality.train()`, writing checkpoints and TensorBoard logs
   directly to this run's Drive experiment folder.
6. **Evaluation** -- evaluate on the held-out test split, save plots and sample predictions.
7. **Export** -- confirm the best model landed in `exported_models/IQA/best_model.keras` and
   print a summary of every artifact this run produced.

## Where outputs are saved

Everything lands on Google Drive, isolated per run under
`experiments/IQA/<timestamp>/` (see the layout above), plus the single stable
`exported_models/IQA/best_model.keras`. **Nothing important is left on the Colab VM** -- if the
runtime disconnects or recycles, only the ephemeral repository clone is lost, never your data.

## How to resume training

Every run gets a brand-new, isolated experiment folder -- resuming means pointing back at a
**specific previous** one instead of starting fresh. In the notebook's "Dataset Loading" section,
set:

```python
RESUME_EXPERIMENT_DIR = "/content/drive/MyDrive/DiabeticRetinopathy/experiments/IQA/2026-08-05_10-30-00"
```

before running that cell, then run the rest of the notebook as usual. `experiment_manager` will
validate that folder (instead of creating a new one) and `train_image_quality.train()` will pick
up from `checkpoints/last.keras` and the recorded epoch in `checkpoints/epoch_state.json`.

## How to launch TensorBoard

The notebook launches it automatically right after training, pointed at the current experiment's
`logs/` folder (`%tensorboard --logdir $logs_dir`). To compare multiple runs, or reopen
TensorBoard in a later session, point it at any experiment's `logs/` directory directly, e.g.:

```python
%load_ext tensorboard
%tensorboard --logdir "/content/drive/MyDrive/DiabeticRetinopathy/experiments/IQA/2026-08-05_10-30-00/logs"
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `verify_google_drive_mounted` raises "Google Drive does not appear to be mounted" | `setup.setup()`'s Drive mount step didn't complete -- re-run Section 1, and accept Colab's authorization prompt. |
| `verify_eyeq_dataset` raises "EyeQ raw dataset directory not found" | Your Drive doesn't match the layout above. Confirm `MyDrive/DiabeticRetinopathy/datasets/EyeQ/raw/{train,test}` exists exactly, with correct capitalization. |
| `verify_gpu_available` raises "No GPU detected" | `Runtime > Change runtime type > Hardware accelerator > GPU`, then re-run from Section 1 (a runtime change resets the VM). |
| `verify_required_packages` raises "Missing required package(s)" | Re-run Section 1 (`setup.setup()`) -- it installs `requirements.txt` before this check runs; if it still fails, the package may need a Colab restart to take effect (`Runtime > Restart session`). |
| Training resumes from epoch 0 despite setting `RESUME_EXPERIMENT_DIR` | Confirm the path points at an experiment folder that actually contains `checkpoints/last.keras` and `checkpoints/epoch_state.json` -- a folder from a run that failed before its first epoch completed won't have either. |
| `ImportError` on `setup`, `colab_config`, etc. in the Bootstrap cell's *next* cell | The Bootstrap cell didn't finish (clone failed, or `colab/` wasn't added to `sys.path`) -- check its output for a git error before re-running. |
| Colab session disconnects mid-training | Nothing is lost except the current epoch's un-checkpointed progress -- checkpoints and logs are already on Drive. Set `RESUME_EXPERIMENT_DIR` to the interrupted run's folder and re-run the notebook. |

## Expected workflow

```
Open notebook in Colab
        |
        v
Run Bootstrap + Section 1 (Setup)  --------->  Drive mounted, repo cloned, deps installed
        |
        v
Run Section 2 (Verification)  -------------->  aborts here if environment/dataset isn't ready
        |
        v
Run Sections 3-4 (Dataset Loading, Model)  -->  new experiment folder created on Drive
        |
        v
Run Section 5 (Training)  ------------------->  checkpoints + TensorBoard logs stream to Drive
        |
        v
Run Section 6 (Evaluation)  ----------------->  plots + sample predictions written to Drive
        |
        v
Run Section 7 (Export)  --------------------->  best_model.keras confirmed in exported_models/IQA/
        |
        v
Session ends / VM recycles  ----------------->  every output already safe on Drive
```
