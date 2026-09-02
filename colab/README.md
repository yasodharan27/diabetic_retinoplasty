# Colab Training Workflow

Official Google Colab training environment for this repository, and the reusable infrastructure
for **every** stage of the 11-stage target pipeline (`PROJECT_CODE.md`, `PROJECT_STRUCTURE.md`) --
not just Image Quality Assessment. Every dataset's master copy lives on Google Drive; each
session **stages** (copies) the dataset it needs onto the Colab VM's local SSD once, so per-epoch
training I/O never crosses the slow Drive mount (see "Dataset Staging" below). All outputs are
written back to Drive so nothing important is lost when a Colab VM recycles.

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
│   ├── dataset_staging.py         copies a dataset from Drive to the local SSD, verifies the copy
│   └── experiment_manager.py      isolated, timestamped experiment folders + metadata.json
└── notebooks/
    ├── stage01_iqa.ipynb                       Image Quality Assessment -- fully implemented
    ├── stage02_preprocessing.ipynb              template
    ├── stage03_vessel_segmentation.ipynb        template
    ├── stage04_lesion_segmentation.ipynb        template
    ├── stage05_local_feature_extraction.ipynb   template
    ├── stage06_global_feature_extraction.ipynb  template
    ├── stage07_feature_fusion.ipynb             template
    ├── stage08_corn_classifier.ipynb            joint Stage 05-08+RACAF training notebook -- infrastructure implemented, RUN_TRAINING=False
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
| `verify_dataset.py` | Verifies the EyeQ dataset: directory/`labels.csv`/image-folder existence, image counts, missing images, a corrupted-image spot check, and class distribution. Reuses `image_quality_dataset._read_labels` / `_decode_image` directly rather than re-implementing file/JPEG checks. Returns a structured `EyeQVerificationReport`; raises on any failed check. EyeQ-specific today (only Stage 1 is implemented), but it doesn't care whether the `raw_dir` it's pointed at is on Drive or the local SSD -- see "Dataset Staging" below. |
| `dataset_staging.py` | Copies a dataset's Drive master directory to the Colab VM's local SSD once per session (`stage_dataset()`), and verifies the copy (`verify_staged_copy()`). Dataset-agnostic: the caller passes which Drive directory to stage and what to name it locally -- nothing here is hardcoded to EyeQ, so the same functions work for APTOS2019/IDRiD once those stages exist. Read-only with respect to Drive; never writes back to it. See "Dataset Staging" below. |
| `experiment_manager.py` | Creates a new, isolated, timestamped folder per training run under `experiments/<Module>/`, with `checkpoints/`, `logs/`, `tensorboard/`, `evaluation/`, `predictions/` subfolders and a `metadata.json`. Also supports resuming into an existing experiment folder. Generic across modules -- pass any `colab_config.DRIVE.experiment_dir("<Module>")`. |

### Purpose of each notebook

| Notebook | Stage | Status |
|---|---|---|
| `stage01_iqa.ipynb` | 1. Image Quality Assessment | **Completed -- Verified -- Baseline Established.** Trained end-to-end (held-out test accuracy 88.05%, F1 86.12%, AUC 96.48%, QWK 0.8987 -- see Section 7 of the notebook and `docs/FIRST_TRAINING_CHECKLIST.md`'s completed-run record). Setup -> Verification -> Dataset Loading -> Model Creation -> Training -> Evaluation -> Export, calling `image_quality_dataset.py` / `image_quality_model.py` / `train_image_quality.py` / `evaluate_image_quality.py` / `image_quality_inference.py`. |
| `stage02_preprocessing.ipynb` | 2. Image Preprocessing | Template -- **Ready to Begin.** `image_preprocessing.py` (repository root) already implements the transforms this stage needs -- not yet orchestrated here. |
| `stage03_vessel_segmentation.ipynb` | 3. Vessel Segmentation | Template. Integrates a pretrained LWNet checkpoint for inference only -- see `SEGMENTATION_ARCHITECTURE.md`. Does **not** follow `stage01_iqa.ipynb`'s Dataset Staging -> Training -> Evaluation lifecycle: no dataset to stage, no training loop, no `experiment_manager.py` training run. Scoped to Setup -> Verification -> Checkpoint Integration -> Inference Verification -> Export. |
| `stage04_lesion_segmentation.ipynb` | 4. Lesion Segmentation | Template. Trains an Attention U-Net on IDRiD -- see `SEGMENTATION_ARCHITECTURE.md`. |
| `stage05_local_feature_extraction.ipynb` | 5. Local Feature Extraction | Template. **Locked:** trained jointly with Stages 6-8 + RACAF, not independently -- see `JOINT_TRAINING_ARCHITECTURE.md`. |
| `stage06_global_feature_extraction.ipynb` | 6. Global Feature Extraction | Template. **Locked:** trained jointly with Stages 5, 7, 8 + RACAF -- see `JOINT_TRAINING_ARCHITECTURE.md`. |
| `stage07_feature_fusion.ipynb` | 7. Feature Fusion | Template. **Locked:** trained jointly with Stages 5, 6, 8 + RACAF -- see `JOINT_TRAINING_ARCHITECTURE.md`. |
| `stage08_corn_classifier.ipynb` | 8. CORN Classification | **Repurposed as the joint Stage 05-08+RACAF training notebook** (CORN has no standalone training path of its own) -- see `JOINT_TRAINING_ARCHITECTURE.md` §27. Infrastructure cells (Drive/environment/dataset/checkpoint verification, authoritative split, joint model construction, a synthetic-tensor smoke test) are implemented, plus optional **Phase 1 / Phase 1b cache-precomputation cells** (`RUN_CACHE_PRECOMPUTATION = False` by default -- populates Stage 03/04/RACAF's cache one image at a time, independent of training, safe to interrupt/resume; §32). Phase 1 runs against a **local** cache directory, not Drive directly -- a real T4 run against the Drive-mounted cache directly was measured to be impractically slow (Drive's per-file-open latency on every check/write); see §33. An entry already cached from a PRIOR run (on Drive) is detected via a cheap existence-only check against `config.LOCAL_FEATURE_RESULTS_DIR`/`config.RACAF_RESULTS_DIR` -- **never** a bulk content pull, which previously crashed a real run's Drive FUSE mount with `OSError: [Errno 107] Transport endpoint is not connected` (thousands of files copied concurrently right after another already-heavy Drive copy); newly computed entries are written locally only and pushed to Drive by the Phase 1b cell (any time, safe to re-run); see §35. A `CACHE_DIAGNOSTIC_MAX_IMAGES` control lets Phase 1's real code path be tried on a small number of images first, with per-image RSS/GPU-memory logging (§35). Phase 1 also calls `training.check_gpu()` before either the Stage 03 or Stage 04 model loads, so TensorFlow requests incremental GPU memory growth instead of claiming ~all VRAM on first use and starving Stage 03's separate PyTorch CUDA allocator on the same GPU -- root-caused (with real checkpoints/images, ruling out a CPU-side leak) after a real run still crashed from RAM exhaustion even with local caching, even though Colab's CPU RAM graph never showed growth; see §34. Dataset-loading and `Trainer` cells (**Phase 2**) are gated behind `RUN_TRAINING = False`. No real training has completed, no checkpoint exists. Checkpoint boundary: `colab_config.DRIVE.experiment_dir("FinalClassification")`. No second, competing joint-training notebook exists or should be created. |
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
    │   ├── EyeQ/                      (Stage 1 only)
    │   │   ├── raw/
    │   │   │   ├── train/{images/, labels.csv}
    │   │   │   └── test/{images/, labels.csv}
    │   │   └── processed/
    │   ├── APTOS2019/
    │   └── IDRiD/                     (no DRIVE/ or CHASE_DB1/ -- Stage 3 uses a vendored
    │                                    pretrained checkpoint, not a staged dataset)
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
Stages 5-8 are very likely trained as one combined `FinalClassification` model, and Stages 9-10
don't train a new model at all (inference-only against the Stage 8 checkpoint). Of the two
segmentation stages, only Stage 4 (`LesionSegmentation`) actually produces a training run under
`experiments/`; Stage 3 (`VesselSegmentation`) never does; its bucket under `exported_models/`
holds a vendored pretrained checkpoint instead of a training output. See `PROJECT_STRUCTURE.md`'s
Pipeline Overview for the full stage-to-module mapping.

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

## Dataset Staging

### Why

Google Drive's FUSE mount (`/content/drive`) is latency-bound per file open, not
bandwidth-bound. This project's `tf.data` pipelines (`image_quality_dataset.py`) don't cache
decoded images, so every training epoch re-reads every training image from its source directory
-- reading directly from Drive means paying that per-file latency tax on every single epoch, not
just once. Copying the dataset to the Colab VM's local SSD once, at the start of a session, turns
every subsequent epoch's reads into fast local I/O instead.

### How it works

1. Each stage notebook calls `dataset_staging.stage_dataset(<drive_dir>, "<name>")` once, near
   the top of its Dataset Loading section -- `stage01_iqa.ipynb` calls
   `dataset_staging.stage_dataset(colab_config.EYEQ_RAW_DIR, "EyeQ")`.
2. Every file under the Drive source directory is copied to `/content/datasets/<name>/` using a
   16-worker thread pool by default -- Drive's FUSE mount tolerates meaningful concurrency, so
   overlapping many small-file copies cuts wall-clock staging time substantially versus a naive
   sequential copy, where each blocking file open is serialized behind the last.
3. `dataset_staging.verify_staged_copy()` re-counts files and total bytes on the Drive side and
   compares against the local copy -- a generic, dataset-structure-agnostic check that raises
   immediately on any mismatch.
4. The stage's own dataset verifier (e.g. `verify_dataset.verify_eyeq_dataset()`) is run again,
   this time against the staged local directory, for a deeper check (`labels.csv` presence,
   per-split image counts, a corrupted-image spot check) -- completely unmodified, since it
   already accepts any `raw_dir` path and has no idea whether that path is on Drive or local disk.
5. Every downstream call that reads dataset images (`load_eyeq_datasets()`, `train()`,
   `evaluate()`, sample predictions) is pointed at the staged local directory
   (`staged_eyeq.local_dir`), not `colab_config.EYEQ_RAW_DIR`.

Staging is **idempotent**: if `/content/datasets/<name>/` already exists, the copy is skipped
(pass `force=True` to re-copy from scratch). This matters because a Colab session reconnect, or
simply re-running the Dataset Loading cell, should not re-copy tens of thousands of files it
already has.

The Drive master copy is **read-only** from `stage_dataset()`'s perspective -- it only ever
reads from the Drive source and writes to the local destination, never the reverse. The staged
local copy itself is disposable (it lives under `/content`, wiped whenever the Colab VM recycles)
and is never treated as an output -- checkpoints, logs, evaluation results, and the exported model
all still go to Google Drive exactly as before (see "Where outputs are saved").

**`dataset_staging.sync_missing_files(source_dir, dest_dir)`** is the incremental, direction-
agnostic sibling used where `stage_dataset()`'s all-or-nothing "already staged, skip everything"
check doesn't fit -- a directory that grows one file at a time across many runs, like the joint
training notebook's per-image inference cache (`JOINT_TRAINING_ARCHITECTURE.md` §33). It copies
only whatever is missing at the destination, in either direction (Drive -> local to resume, or
local -> Drive to persist), reusing the same latency-tolerant thread-pool copy. **Do not use it to
bulk-pull an entire persistent Drive cache down "just in case"** -- a real run doing exactly that
crashed with `OSError: [Errno 107] Transport endpoint is not connected` under the resulting
concurrent-copy burst (§35); checking whether a specific cache entry already exists needs only a
cheap `os.path.exists` stat against its known path, not its content. Each individual copy is
atomic (temp file + size-verified rename) and retries transient Drive/FUSE errors (ENOTCONN and
similar) with backoff; a file that still fails after retries is reported in the returned
`(copied_count, already_present_count, failures)` tuple rather than aborting the whole call, and
remains missing at the destination so a later call retries it automatically.

### Making this reusable for a future stage

Nothing in `dataset_staging.py` is EyeQ-specific -- which dataset a notebook stages is entirely
up to that notebook, not hardcoded in the staging module. A future `stage04_lesion_segmentation.ipynb`
would stage IDRiD the same way:

```python
staged_idrid = dataset_staging.stage_dataset(colab_config.IDRID_DATASET_DIR, "IDRiD")
dataset_staging.verify_staged_copy(staged_idrid)
# + that stage's own dataset verifier, if one exists, against staged_idrid.local_dir
```

`colab_config.py` already exposes `EYEQ_RAW_DIR`, `APTOS2019_DATASET_DIR`, and
`IDRID_DATASET_DIR` for exactly this -- no changes to `dataset_staging.py` itself are needed to
stage a different dataset. Stage 3 (Vessel Segmentation) does not need a dataset constant here at
all: it integrates a vendored pretrained checkpoint rather than staging a dataset, so
`dataset_staging.py` is not part of its notebook.

### Expected speed improvement (estimate)

**This is an estimate, not a measured benchmark.** It's reasoned from this project's actual,
verified EyeQ dataset size (12,543 train-pool images, 16,249 held-out test images -- see
`docs/FIRST_TRAINING_CHECKLIST.md`) and commonly-reported Google Drive FUSE mount latency
characteristics on Colab, not from a real training run on this specific data (this development
environment has no GPU/Colab access to measure it directly). `stage_dataset()` prints its own
real elapsed time and files/s rate every time it runs -- treat that as ground truth once you
actually run it, and update this section with real numbers from a real session.

| | Reading from Drive (before) | Reading from local SSD (after) |
|---|---|---|
| Typical per-file open latency (commonly reported for Colab's Drive FUSE mount vs. local disk) | ~50-200 ms | well under 1 ms |
| Images read per training epoch | ~12,543 (full train pool: shuffled train split + validation split, re-read every epoch since nothing is cached) | same |
| Estimated per-epoch I/O-wait cost | on the order of many minutes if requests were fully serialized; `tf.data`'s parallel reads overlap some of this, but Drive FUSE mounts commonly throttle/serialize small-file concurrency more than local disk does, so real-world impact is typically still a large multiple of local-disk time | a small fraction of a second -- I/O effectively stops being the bottleneck; epoch time becomes compute-bound (decode + forward + backward pass) instead |
| One-time cost this change adds | none | staging ~28,792 files (train + test) once per session -- expect this to take on the order of minutes with the default 16-worker parallel copy; the exact number is printed by `stage_dataset()` |

**Net effect over a full 50-epoch run:** even under a conservative reading, a few minutes spent
staging once is much smaller than a recurring per-epoch Drive I/O tax paid 50 times over -- this
should be a large net win for any run beyond a handful of epochs. Exact numbers depend on your
specific Colab runtime, network conditions, and Drive account tier; run the notebook once and
read `stage_dataset()`'s printed output for a real, session-specific number.

---

## How to open a notebook

1. Upload the notebook (e.g. `colab/notebooks/stage01_iqa.ipynb`) to Google Colab (File > Upload
   notebook), or open it directly from GitHub (File > Open notebook > GitHub, paste this repo's
   URL and select the file).
2. `Runtime > Change runtime type > Hardware accelerator > GPU` (T4 or better recommended).

## How to start training

**Stage 1 has already been trained and verified** -- held-out test accuracy 88.05%, F1 86.12%,
AUC 96.48%, QWK 0.8987 (experiment `2026-08-05_09-11-28`; full detail in the notebook's Section 7
and `docs/FIRST_TRAINING_CHECKLIST.md`'s completed-run record). The steps below remain the
official process for re-running Stage 1 or training any future stage.

Run `stage01_iqa.ipynb`'s cells top to bottom (the only stage with real training code today). No
manual edits are required if your Drive matches the layout above. The notebook will:

1. **Setup** -- mount Drive, clone the repo, install requirements, `cd` into it.
2. **Verification** -- abort with a clear error if the environment or dataset isn't ready (checked
   against the Drive master copy).
3. **Dataset Loading** -- pick a batch size, create a new experiment folder, **stage EyeQ onto the
   local SSD and verify the copy** (see "Dataset Staging" above), then load EyeQ from the staged
   local copy.
4. **Model Creation** -- build and preview the EfficientNetB0 architecture.
5. **Training** -- call `train_image_quality.train()`, writing checkpoints and TensorBoard logs
   directly to this run's Drive experiment folder.
6. **Evaluation** -- evaluate on the held-out test split, save plots and sample predictions.
7. **Export** -- confirm the best model landed in `exported_models/IQA/best_model.keras` and
   print a summary of every artifact this run produced.

## Where outputs are saved

Everything lands on Google Drive, isolated per run under `experiments/<Module>/<timestamp>/` (see
the layout above), plus the module's single stable `exported_models/<Module>/best_model.keras`
(TensorFlow-based modules) -- `exported_models/VesselSegmentation/best_model.pth` is the vendored
LWNet checkpoint, uploaded once rather than produced by a training run, per
`SEGMENTATION_ARCHITECTURE.md` §6.
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
   `training.Trainer` / `evaluation.Evaluator` internally if that stage is implemented in
   TensorFlow -- do not reimplement checkpointing, early stopping, or evaluation metrics inside
   the notebook. Stage 3 (Vessel Segmentation) is the one exception to this whole step: it has no
   dataset loader or training entry point at all, only a model-loading + inference module wrapping
   a vendored pretrained checkpoint (PyTorch, fixed -- see `SEGMENTATION_ARCHITECTURE.md` §6),
   without changing the stage's documented `pipeline.SegmentationStage` contract.
3. Follow `stage01_iqa.ipynb`'s section structure: Setup -> Verification -> Dataset Loading ->
   Model Creation -> Training -> Evaluation -> Export (Stage 3 instead follows the reduced
   Setup -> Verification -> Checkpoint Integration -> Inference Verification -> Export shape noted
   in the notebook table above). Reuse `colab/common/setup.py` and
   `colab/common/verify_environment.py` unchanged; write a stage-specific dataset verification
   function (mirroring `verify_dataset.verify_eyeq_dataset()`'s shape) only if that stage's
   dataset needs different checks than EyeQ's.
4. In Dataset Loading, stage that stage's dataset onto local disk with
   `dataset_staging.stage_dataset(colab_config.<DATASET>_DIR, "<Name>")` before loading it --
   don't read training images directly from Drive (see "Dataset Staging" above). Reuse
   `dataset_staging.py` unchanged; it takes the Drive directory and a name, nothing stage-specific.
5. Use `experiment_manager.resolve_experiment(colab_config.DRIVE.experiment_dir("<Module>"), ...)`
   for experiment isolation -- do not invent a different output-folder scheme.
6. Update this README's "Purpose of each notebook" table and `PROJECT_STRUCTURE.md`'s matching
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

### Dataset staging issues

| Symptom | Likely cause / fix |
|---|---|
| `stage_dataset` raises "does not exist on Drive" | The Drive source directory is wrong or Drive isn't mounted -- confirm `colab_config.EYEQ_RAW_DIR` (or whichever dataset constant you passed) resolves correctly and Section 2's Verification already passed against it. |
| `verify_staged_copy` raises a file-count or byte-count mismatch | The copy was interrupted (e.g. a Drive disconnect mid-copy) -- re-run the staging cell with `force=True` to discard the partial local copy and re-stage from scratch. |
| Staging seems to hang or is much slower than expected | Drive may be throttling under high concurrency -- try a lower `max_workers` (e.g. `stage_dataset(..., max_workers=4)`); very large datasets will also simply take longer the first time regardless of concurrency. |
| Local SSD fills up (`OSError: No space left on device`) | The Colab VM's local disk is finite and shared with the repository clone, pip packages, etc. -- a single dataset should fit comfortably, but staging multiple large datasets in the same session may not; stage only what the current notebook actually needs. |
| Training still seems Drive-bound after staging | Confirm the training/evaluation calls were actually updated to use `staged_<name>.local_dir`, not the original `colab_config.<DATASET>_DIR` -- staging alone does nothing if downstream code still reads from Drive. |

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
Run Dataset Loading  -------------------------->  dataset staged Drive -> local SSD + verified,
        |                                          new experiment folder created on Drive
        v
Run Model Creation  ---------------------------->  architecture previewed
        |
        v
Run Training  --------------------------------->  reads staged local data; checkpoints + TensorBoard
        |                                          logs stream to Drive
        |
        v
Run Evaluation  -------------------------------->  plots + sample predictions written to Drive
        |
        v
Run Export  -------------------------------------->  best_model[.ext] confirmed in exported_models/<Module>/
        |
        v
Session ends / VM recycles  ------------------------>  every output already safe on Drive
```
