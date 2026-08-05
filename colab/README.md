# Colab Training Workflow

Official Google Colab training environment for this repository, and the reusable infrastructure
for **every** stage of the 11-stage target pipeline (`PROJECT_CODE.md`, `PROJECT_STRUCTURE.md`) --
not just Image Quality Assessment. Every stage trains against datasets stored in Google Drive,
with all outputs written back to Drive so nothing important is lost when a Colab VM recycles.

`colab/common/` holds only Colab-specific orchestration code -- mounting Drive, verifying the
environment/dataset, managing experiment folders. Actual model/dataset/training/evaluation logic
lives in the repository root and in `training/` / `evaluation/`; nothing here duplicates it, and
nothing in `training/` or `evaluation/` was modified to build this workflow.

---

## Folder structure

```
colab/
├── README.md                   -- this file
├── common/                      -- shared by every stage notebook
│   ├── colab_config.py            every path this workflow uses, in one place
│   ├── drive_paths.py             the Google Drive folder layout, as data (no side effects)
│   ├── environment.py             environment inspection/setup primitives (GPU, TF, CUDA, packages)
│   ├── setup.py                   one-call setup: mount Drive, clone repo, install deps, chdir, env vars
│   ├── verify_environment.py      aborting environment checks, built on environment.py
│   ├── verify_dataset.py          aborting EyeQ dataset checks (structure, counts, corruption)
│   └── experiment_manager.py      isolated, timestamped experiment folders + metadata.json
└── notebooks/
    ├── stage01_iqa.ipynb                       Image Quality Assessment -- fully implemented
    ├── stage02_preprocessing.ipynb              template
    ├── stage03_vessel_segmentation.ipynb        template
    ├── stage04_lesion_segmentation.ipynb        template
    ├── stage05_local_feature_extraction.ipynb   template
    ├── stage06_global_feature_extraction.ipynb  template
    ├── stage07_feature_fusion.ipynb             template
    ├── stage08_corn_classifier.ipynb            template
    ├── stage09_uncertainty_estimation.ipynb     template
    ├── stage10_explainability.ipynb             template
    └── stage11_evaluation.ipynb                 template
```

Only `stage01_iqa.ipynb` is a complete, working notebook today. Every other stage notebook is a
lightweight template (title, objective, expected inputs/outputs, dependencies, implementation
status, a working Bootstrap + `colab/common/` import cell) -- see "How to build out a stage
notebook" below before writing training code into one of them.

### Purpose of each helper module (`colab/common/`)

| File | Purpose |
|---|---|
| `colab_config.py` | Single source of truth for every path (Drive root, dataset roots for EyeQ/APTOS2019/IDRiD, experiment root, TensorBoard root, exported-model root, repository root), plus Stage 1's resolved `IQA_*` constants. Pure data -- no Drive mount, no cloning. Every other module and every notebook import paths from here instead of hardcoding them. |
| `drive_paths.py` | Defines the verified Google Drive directory tree (see below) and resolves it into concrete, POSIX-correct paths (Colab always runs Linux, so paths are built with `posixpath` regardless of the OS this module happens to be imported from). Pure path logic, no `google.colab` import, so it's importable and testable outside Colab. `experiment_dir(module)` / `exported_model_dir(module)` are generic across all four independently-trained modules (`IQA`, `VesselSegmentation`, `LesionSegmentation`, `FinalClassification`), not IQA-specific. |
| `environment.py` | Low-level, non-aborting info-gathering and setup actions: Python/TensorFlow/Keras/CUDA versions, GPU device/name/memory, git commit hash, missing-package detection. Reuses `training.check_gpu` / `training.enable_mixed_precision` rather than reimplementing them. |
| `setup.py` | The one function (`setup.setup()`) every stage notebook calls to mount Drive, clone/update the repository, install `requirements.txt`, `cd` into the repository, and point `EYEQ_RAW_DIR` at Drive. Also writes a session log to Drive's global `logs/`. |
| `verify_environment.py` | Pass/fail checks built on `environment.py`: Python version, TensorFlow version, GPU availability, CUDA, mixed precision, repository path, Google Drive mount, required packages. Every check raises `RuntimeError` with a specific message on failure -- nothing here is caught or swallowed. |
| `verify_dataset.py` | Verifies the EyeQ dataset: directory/`labels.csv`/image-folder existence, image counts, missing images, a corrupted-image spot check, and class distribution. Reuses `image_quality_dataset._read_labels` / `_decode_image` directly rather than re-implementing file/JPEG checks. Returns a structured `EyeQVerificationReport`; raises on any failed check. EyeQ-specific today (only Stage 1 is implemented) -- see "How to build out a stage notebook" for what a future stage's dataset verification should look like. |
| `experiment_manager.py` | Creates a new, isolated, timestamped folder per training run under `experiments/<Module>/`, with `checkpoints/`, `logs/`, `tensorboard/`, `evaluation/`, `predictions/` subfolders and a `metadata.json`. Also supports resuming into an existing experiment folder. Generic across modules -- pass any `colab_config.DRIVE.experiment_dir("<Module>")`. |

### Purpose of each notebook

| Notebook | Stage | Status |
|---|---|---|
| `stage01_iqa.ipynb` | 1. Image Quality Assessment | **Implemented.** Setup -> Verification -> Dataset Loading -> Model Creation -> Training -> Evaluation -> Export, calling `image_quality_dataset.py` / `image_quality_model.py` / `train_image_quality.py` / `evaluate_image_quality.py` / `image_quality_inference.py`. |
| `stage02_preprocessing.ipynb` | 2. Image Preprocessing | Template. `image_preprocessing.py` (repository root) already implements the transforms this stage needs -- not yet orchestrated here. |
| `stage03_vessel_segmentation.ipynb` | 3. Vessel Segmentation | Template. Per `SEGMENTATION_ARCHITECTURE.md`, inference-only against a pretrained U-Net -- likely will not need `experiment_manager.py`'s training-run tracking at all. |
| `stage04_lesion_segmentation.ipynb` | 4. Lesion Segmentation | Template. Trains an Attention U-Net on IDRiD -- see `SEGMENTATION_ARCHITECTURE.md`. |
| `stage05_local_feature_extraction.ipynb` | 5. Local Feature Extraction | Template. Likely trained jointly with Stages 6-8 -- confirm before implementing independently. |
| `stage06_global_feature_extraction.ipynb` | 6. Global Feature Extraction | Template. Likely trained jointly with Stages 5, 7, 8. |
| `stage07_feature_fusion.ipynb` | 7. Feature Fusion | Template. Likely trained jointly with Stages 5, 6, 8. |
| `stage08_corn_classifier.ipynb` | 8. CORN Classification | Template. Very likely the actual trainable checkpoint boundary for Stages 5-8 combined -- see `colab_config.DRIVE.experiment_dir("FinalClassification")`. |
| `stage09_uncertainty_estimation.ipynb` | 9. Uncertainty Estimation | Template. Inference-only against the Stage 8 model -- no new training run. |
| `stage10_explainability.ipynb` | 10. Explainability | Template. Inference-only against the Stage 8 model -- no new training run. |
| `stage11_evaluation.ipynb` | 11. Evaluation | Template. Runs after every upstream stage is trained -- never fill this in with placeholder numbers. |

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
    │   │       └── metadata.json      run metadata (see "How metadata.json is generated")
    │   ├── VesselSegmentation/
    │   ├── LesionSegmentation/
    │   └── FinalClassification/
    │
    ├── tensorboard/                   (auto-created; global root, reserved for cross-experiment use)
    │
    ├── exported_models/               (auto-created)
    │   ├── IQA/best_model.keras
    │   ├── VesselSegmentation/
    │   ├── LesionSegmentation/
    │   └── FinalClassification/       (the current best model per module -- stable path,
    │                                    overwritten by each run, distinct from the per-experiment
    │                                    checkpoints/ archive above)
    │
    └── logs/                          (auto-created; session-level setup logs, see setup.py)
```

`experiments/` currently has four module buckets (`IQA`, `VesselSegmentation`, `LesionSegmentation`,
`FinalClassification`), not eleven -- this matches `PROJECT_CODE.md`'s Models table, since
Stages 5-8 are very likely trained as one combined `FinalClassification` model, and Stages 3, 9,
10 don't train a new model at all (inference-only against an upstream checkpoint). See
`PROJECT_STRUCTURE.md`'s Pipeline Overview for the full stage-to-module mapping.

`datasets/*/raw` is **never** written to by anything in `colab/` -- it must already contain real,
uploaded data. `experiments/`, `tensorboard/`, `exported_models/`, and `logs/` (and their module
subfolders) are created automatically (`drive_paths.ensure_output_directories()`, called by
`setup.setup()`) if missing.

**Why `logs/` and `tensorboard/` both exist inside each experiment:** the existing
`training.trainer.TrainingConfig` (which this workflow intentionally does not modify) always
writes TensorBoard's event files to `<run_dir>/logs`. `tensorboard/` is filled by
`experiment_manager.archive_tensorboard_logs()` as a one-time copy after training completes, for
anyone who expects a folder literally named `tensorboard/`. **`logs/` is the live directory** --
point TensorBoard there during or immediately after a run (`stage01_iqa.ipynb` does this
automatically).

---

## How to open a notebook

1. Upload the notebook (e.g. `colab/notebooks/stage01_iqa.ipynb`) to Google Colab (File > Upload
   notebook), or open it directly from GitHub (File > Open notebook > GitHub, paste this repo's
   URL and select the file).
2. `Runtime > Change runtime type > Hardware accelerator > GPU` (T4 or better recommended).

## How to start training

Run `stage01_iqa.ipynb`'s cells top to bottom (the only stage with real training code today). No
manual edits are required if your Drive matches the layout above. The notebook will:

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

Everything lands on Google Drive, isolated per run under `experiments/<Module>/<timestamp>/` (see
the layout above), plus the module's single stable `exported_models/<Module>/best_model.keras`.
**Nothing important is left on the Colab VM** -- if the runtime disconnects or recycles, only the
ephemeral repository clone is lost, never your data.

## How to resume training

Every run gets a brand-new, isolated experiment folder -- resuming means pointing back at a
**specific previous** one instead of starting fresh. In `stage01_iqa.ipynb`'s "Dataset Loading"
section, set:

```python
RESUME_EXPERIMENT_DIR = "/content/drive/MyDrive/DiabeticRetinopathy/experiments/IQA/2026-08-05_10-30-00"
```

before running that cell, then run the rest of the notebook as usual. `experiment_manager` will
validate that folder (instead of creating a new one) and `train_image_quality.train()` will pick
up from `checkpoints/last.keras` and the recorded epoch in `checkpoints/epoch_state.json`.

## How to launch TensorBoard

`stage01_iqa.ipynb` launches it automatically right after training, pointed at the current
experiment's `logs/` folder (`%tensorboard --logdir $logs_dir`). To compare multiple runs, or
reopen TensorBoard in a later session, point it at any experiment's `logs/` directory directly:

```python
%load_ext tensorboard
%tensorboard --logdir "/content/drive/MyDrive/DiabeticRetinopathy/experiments/IQA/2026-08-05_10-30-00/logs"
```

## How metadata.json is generated

`experiment_manager.create_experiment()` writes `metadata.json` into the experiment root the
moment the folder is created (before training starts), combining:

- **Automatically gathered:** `timestamp`, `git_commit_hash` (via `environment.get_git_commit_hash`),
  `tensorflow_version`, `python_version`, `gpu_model` (`None` on a CPU-only run).
- **Passed in by the calling notebook** as keyword arguments -- `stage01_iqa.ipynb` passes
  `dataset_path`, `batch_size`, `learning_rate`, `epochs`, `image_size`, `freeze_layers`,
  `random_seed`.

Any future stage notebook should pass whatever hyperparameters are relevant to *that* stage the
same way -- `experiment_manager.create_experiment()` accepts arbitrary keyword arguments and
records them verbatim, so there's nothing to modify in `experiment_manager.py` itself to support
a new stage's metadata.

## How to create new stage notebooks

The 10 non-IQA notebooks under `colab/notebooks/` are templates, not empty files -- each already
has a working Bootstrap cell and a `colab/common/` import cell. To turn one into a real
implementation:

1. Read `PROJECT_CODE.md`'s Development Workflow first: explain the existing implementation,
   explain why it needs to change, propose a plan, wait for approval, implement, verify
   integration, stop. Do not skip straight to writing training code.
2. Implement that stage's dataset loader, model, and training entry point as top-level repository
   modules (mirroring `image_quality_dataset.py` / `image_quality_model.py` /
   `train_image_quality.py` / `evaluate_image_quality.py` / `image_quality_inference.py`), using
   `training.Trainer` / `evaluation.Evaluator` internally -- do not reimplement checkpointing,
   early stopping, or evaluation metrics inside the notebook.
3. Follow `stage01_iqa.ipynb`'s section structure: Setup -> Verification -> Dataset Loading ->
   Model Creation -> Training -> Evaluation -> Export. Reuse `colab/common/setup.py` and
   `colab/common/verify_environment.py` unchanged; write a stage-specific dataset verification
   function (mirroring `verify_dataset.verify_eyeq_dataset()`'s shape) only if that stage's
   dataset needs different checks than EyeQ's.
4. Use `experiment_manager.resolve_experiment(colab_config.DRIVE.experiment_dir("<Module>"), ...)`
   for experiment isolation -- do not invent a different output-folder scheme.
5. Update this README's "Purpose of each notebook" table and `PROJECT_STRUCTURE.md`'s matching
   pipeline-stage entry to reflect the new status.

## Common Troubleshooting

### GPU issues

| Symptom | Likely cause / fix |
|---|---|
| `verify_gpu_available` raises "No GPU detected" | `Runtime > Change runtime type > Hardware accelerator > GPU`, then re-run from Setup (a runtime change resets the VM). |
| Training is far slower than expected on a GPU runtime | Check `verify_mixed_precision()`'s output -- if it reports `float32` despite a GPU being present, the mixed precision policy didn't activate; re-run Setup/Verification and check for an exception earlier in the cell. |
| `verify_cuda_available` raises "CPU-only build" | The installed TensorFlow wheel has no CUDA support -- restart the runtime with a GPU accelerator selected *before* `pip install`s run, so the correct wheel is resolved. |

### Drive issues

| Symptom | Likely cause / fix |
|---|---|
| `verify_google_drive_mounted` raises "Google Drive does not appear to be mounted" | `setup.setup()`'s Drive mount step didn't complete -- re-run Setup, and accept Colab's authorization prompt. |
| `verify_eyeq_dataset` raises "EyeQ raw dataset directory not found" | Your Drive doesn't match the layout above. Confirm `MyDrive/DiabeticRetinopathy/datasets/EyeQ/raw/{train,test}` exists exactly, with correct capitalization (`DiabeticRetinopathy`, `EyeQ`). |
| Experiment folders appear to go missing between sessions | Confirm you're mounting the same Google account's Drive every session -- `experiments/` lives under that account's `MyDrive`, not shared/organization Drive unless explicitly configured. |

### Resume issues

| Symptom | Likely cause / fix |
|---|---|
| Training resumes from epoch 0 despite setting `RESUME_EXPERIMENT_DIR` | Confirm the path points at an experiment folder that actually contains `checkpoints/last.keras` and `checkpoints/epoch_state.json` -- a run that failed before its first epoch completed has neither. |
| `resume_experiment` raises "missing expected subfolder(s)" | The path isn't an `experiment_manager`-created folder (e.g. a manually created Drive folder, or a folder from before this workflow existed) -- point at a real experiment folder instead. |
| Optimizer state seems reset after resuming | Known, documented limitation -- `train_image_quality.train()`'s resume path restores model weights correctly but not Adam's momentum/variance state (a Keras 3 optimizer-variable-count mismatch). See `docs/FIRST_TRAINING_CHECKLIST.md` and the IQA training-pipeline inspection notes for detail; not something `colab/` introduces or can fix without changing `training/`. |

### Other

| Symptom | Likely cause / fix |
|---|---|
| `verify_required_packages` raises "Missing required package(s)" | Re-run Setup (`setup.setup()`) -- it installs `requirements.txt` before this check runs; if it still fails, the package may need a Colab restart to take effect (`Runtime > Restart session`). |
| `ImportError` on `setup`, `colab_config`, etc. in the Bootstrap cell's *next* cell | The Bootstrap cell didn't finish (clone failed, or `colab/common/` wasn't added to `sys.path`) -- check its output for a git error before re-running. |
| Colab session disconnects mid-training | Nothing is lost except the current epoch's un-checkpointed progress -- checkpoints and logs are already on Drive. Set `RESUME_EXPERIMENT_DIR` to the interrupted run's folder and re-run the notebook. |

## Expected workflow

```
Open a stage notebook in Colab
        |
        v
Run Bootstrap + Setup  ---------------------->  Drive mounted, repo cloned, deps installed
        |
        v
Run Verification  ---------------------------->  aborts here if environment/dataset isn't ready
        |
        v
Run Dataset Loading + Model Creation  -------->  new experiment folder created on Drive
        |
        v
Run Training  --------------------------------->  checkpoints + TensorBoard logs stream to Drive
        |
        v
Run Evaluation  -------------------------------->  plots + sample predictions written to Drive
        |
        v
Run Export  -------------------------------------->  best_model.keras confirmed in exported_models/<Module>/
        |
        v
Session ends / VM recycles  ------------------------>  every output already safe on Drive
```
