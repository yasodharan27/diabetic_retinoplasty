"""
Defines the verified Google Drive directory layout used by every Colab
training notebook in this repository, and resolves it into concrete paths.

Pure path logic only -- no `google.colab` import, no filesystem access at
import time, so this module can be imported (and its path resolution
tested) outside Colab. Mounting the drive itself is `setup.py`'s job.

Verified layout (do not assume any folder beyond this exists):

    MyDrive/
    +-- DiabeticRetinopathy/
        +-- datasets/
        |   +-- EyeQ/           (raw/{train,test}/{images/,labels.csv}, processed/)
        |   +-- APTOS2019/
        |   +-- IDRiD/
        +-- experiments/
        |   +-- IQA/
        |   +-- VesselSegmentation/
        |   +-- LesionSegmentation/
        |   +-- FinalClassification/
        +-- tensorboard/
        +-- exported_models/
        +-- cache/
        +-- logs/

`cache/` is a minimal, additive extension of this same verified layout (added
for the Stage 05-08+RACAF joint-training design; see `JOINT_TRAINING_ARCHITECTURE.md`),
not a redesign of it -- it holds the same *category* of content
`experiments/`/`tensorboard/`/`exported_models/`/`logs/` already hold ("pipeline
outputs... safe to create automatically", see below), just for a kind of
output none of those four buckets fit: small, derived, per-image arrays
(frozen Stage 03/04 predictions, RACAF's reliability signal) that must be
reused ACROSS every training run and resumed Colab session -- not scoped to
one timestamped `experiments/<module>/<run>/` folder (which would defeat
reuse across runs), and not a final trained-weight artifact
(`exported_models/`). Locally, this same content already lives under
`results/<module>/` (`config.py`'s `LOCAL_FEATURE_RESULTS_DIR`,
`RACAF_RESULTS_DIR`); Drive's verified layout has no equivalent top-level
bucket, so one is added here, following the exact same per-module dict
pattern `experiment_dirs`/`exported_model_dirs` already use.

`datasets/` holds real, irreplaceable, already-uploaded data -- nothing in
this module ever creates or writes into it (see `DATASET_DIRS` vs.
`OUTPUT_DIRS` below, and `ensure_output_directories()`). `experiments/`,
`tensorboard/`, `exported_models/`, and `logs/` are pipeline outputs and
are safe to create automatically.

`exported_models/` and `experiments/` are namespaced per pipeline module
(`IQA`, `VesselSegmentation`, ...) so multiple modules never collide on the
same `best_model.keras` -- `experiments/<module>/` is spelled out in the
verified layout; `exported_models/<module>/` mirrors that same convention
for consistency, since the diagram only shows the shared root but every
trainable module needs its own final-model slot.

--- APTOS2019 / IDRiD raw+processed sub-resolution: epistemic status ---

The layout above (`datasets/APTOS2019/`, `datasets/IDRiD/`) is the directly
verified one -- it does not by itself say whether Drive's copies mirror
this project's local `raw/`/`processed/` convention, or (for IDRiD) the
three-subset `grading/`/`localization/`/`segmentation/` split confirmed in
this repository's own local `datasets/IDRiD/` directory. `aptos2019_raw_dir`
/ `aptos2019_processed_dir` and the `idrid_*_raw_dir` / `idrid_*_processed_dir`
fields below extend the layout by convention -- the same raw/processed
scaffolding `config.py`'s own module comment says "every dataset under
datasets/ follows" locally, and the same three subset names
`lesion_segmentation_dataset.py` (frozen, Stage 04) already depends on via
`config.dataset_raw_dir("IDRiD/segmentation")`. This is a reasonable,
non-fabricated inference, NOT a directly verified fact about Drive's actual
current contents -- this module cannot check Drive from outside Colab, and
no one has confirmed these specific subpaths against the live Drive folder
as of this addition. `verify_dataset.verify_idrid_dataset_dir()` checks
this explicitly (and fails clearly, rather than silently) the first time a
real Colab session mounts Drive; until that has run at least once, treat
these specific fields as unverified-from-code.
"""

import os
import posixpath
from dataclasses import dataclass

DRIVE_PROJECT_FOLDER_NAME = "DiabeticRetinopathy"

PIPELINE_MODULES = ("IQA", "VesselSegmentation", "LesionSegmentation", "FinalClassification")

# Persistent, per-image derived caches (Step 3/Step 4 of the joint-training
# infrastructure correction) -- keyed separately from PIPELINE_MODULES,
# which enumerates independently *trained/checkpointed* modules. Neither
# cache belongs to an independently-trained module: "LocalFeatureExtraction"
# caches Stage 03/04's frozen output (consumed by Stage 05, which trains
# only as part of "FinalClassification"), and "RACAF" caches its own
# reliability signal (kappa/r) -- both are derived, regenerable artifacts,
# not trained weights.
CACHE_MODULES = ("LocalFeatureExtraction", "RACAF")


@dataclass(frozen=True)
class DrivePaths:
    """Every resolved Drive path this project's Colab notebooks use."""

    drive_mount_point: str
    project_root: str

    # datasets/ -- real data, read-only from this project's perspective
    datasets_root: str
    eyeq_dataset_dir: str
    eyeq_raw_dir: str
    eyeq_processed_dir: str

    aptos2019_dataset_dir: str
    aptos2019_raw_dir: str
    aptos2019_processed_dir: str

    # IDRiD is not flat -- three named subsets, each with its own raw/processed
    # pair (grading/localization/segmentation; Stage 04 uses only "segmentation"
    # today, via config.dataset_raw_dir("IDRiD/segmentation")). idrid_dataset_dir
    # is the bare root; the six *_raw_dir/*_processed_dir fields below are the
    # per-subset paths every dataset loader actually needs. See this module's
    # docstring for what is/isn't directly verified about Drive here.
    idrid_dataset_dir: str
    idrid_grading_raw_dir: str
    idrid_grading_processed_dir: str
    idrid_localization_raw_dir: str
    idrid_localization_processed_dir: str
    idrid_segmentation_raw_dir: str
    idrid_segmentation_processed_dir: str

    # experiments/<module>/ -- one isolated, timestamped folder per training run
    experiments_root: str
    experiment_dirs: dict  # module name -> experiments_root/<module>

    # tensorboard/, exported_models/<module>/, cache/<module>/, logs/ -- other outputs
    tensorboard_root: str
    exported_models_root: str
    exported_model_dirs: dict  # module name -> exported_models_root/<module>
    cache_root: str
    cache_dirs: dict  # module name -> cache_root/<module>, see CACHE_MODULES
    logs_root: str

    def experiment_dir(self, module):
        try:
            return self.experiment_dirs[module]
        except KeyError:
            raise ValueError(f"Unknown pipeline module {module!r}; expected one of {PIPELINE_MODULES}") from None

    def exported_model_dir(self, module):
        try:
            return self.exported_model_dirs[module]
        except KeyError:
            raise ValueError(f"Unknown pipeline module {module!r}; expected one of {PIPELINE_MODULES}") from None

    def cache_dir(self, module):
        try:
            return self.cache_dirs[module]
        except KeyError:
            raise ValueError(f"Unknown cache module {module!r}; expected one of {CACHE_MODULES}") from None


def build_drive_paths(drive_mount_point="/content/drive"):
    """Resolve every path in the verified Drive layout from a mount point.

    Builds paths with `posixpath` rather than `os.path`, deliberately --
    Colab always runs Linux, so a Drive path is always POSIX-style
    regardless of the OS this module happens to be imported from (e.g.
    during local, offline validation on Windows). Does not touch the
    filesystem -- see `ensure_output_directories()`."""
    project_root = posixpath.join(drive_mount_point, "MyDrive", DRIVE_PROJECT_FOLDER_NAME)
    datasets_root = posixpath.join(project_root, "datasets")
    experiments_root = posixpath.join(project_root, "experiments")
    exported_models_root = posixpath.join(project_root, "exported_models")
    cache_root = posixpath.join(project_root, "cache")

    eyeq_dataset_dir = posixpath.join(datasets_root, "EyeQ")
    aptos2019_dataset_dir = posixpath.join(datasets_root, "APTOS2019")
    idrid_dataset_dir = posixpath.join(datasets_root, "IDRiD")
    idrid_grading_dir = posixpath.join(idrid_dataset_dir, "grading")
    idrid_localization_dir = posixpath.join(idrid_dataset_dir, "localization")
    idrid_segmentation_dir = posixpath.join(idrid_dataset_dir, "segmentation")

    return DrivePaths(
        drive_mount_point=drive_mount_point,
        project_root=project_root,
        datasets_root=datasets_root,
        eyeq_dataset_dir=eyeq_dataset_dir,
        eyeq_raw_dir=posixpath.join(eyeq_dataset_dir, "raw"),
        eyeq_processed_dir=posixpath.join(eyeq_dataset_dir, "processed"),
        aptos2019_dataset_dir=aptos2019_dataset_dir,
        aptos2019_raw_dir=posixpath.join(aptos2019_dataset_dir, "raw"),
        aptos2019_processed_dir=posixpath.join(aptos2019_dataset_dir, "processed"),
        idrid_dataset_dir=idrid_dataset_dir,
        idrid_grading_raw_dir=posixpath.join(idrid_grading_dir, "raw"),
        idrid_grading_processed_dir=posixpath.join(idrid_grading_dir, "processed"),
        idrid_localization_raw_dir=posixpath.join(idrid_localization_dir, "raw"),
        idrid_localization_processed_dir=posixpath.join(idrid_localization_dir, "processed"),
        idrid_segmentation_raw_dir=posixpath.join(idrid_segmentation_dir, "raw"),
        idrid_segmentation_processed_dir=posixpath.join(idrid_segmentation_dir, "processed"),
        experiments_root=experiments_root,
        experiment_dirs={m: posixpath.join(experiments_root, m) for m in PIPELINE_MODULES},
        tensorboard_root=posixpath.join(project_root, "tensorboard"),
        exported_models_root=exported_models_root,
        exported_model_dirs={m: posixpath.join(exported_models_root, m) for m in PIPELINE_MODULES},
        cache_root=cache_root,
        cache_dirs={m: posixpath.join(cache_root, m) for m in CACHE_MODULES},
        logs_root=posixpath.join(project_root, "logs"),
    )


def output_directories(paths: DrivePaths):
    """Every directory that is safe to auto-create (pipeline outputs only --
    never a dataset directory)."""
    return [
        paths.experiments_root,
        *paths.experiment_dirs.values(),
        paths.tensorboard_root,
        paths.exported_models_root,
        *paths.exported_model_dirs.values(),
        paths.cache_root,
        *paths.cache_dirs.values(),
        paths.logs_root,
    ]


def ensure_output_directories(paths: DrivePaths):
    """Create every output directory (`experiments/`, `tensorboard/`,
    `exported_models/`, `logs/`, and their module subfolders) if missing.
    Never creates or touches anything under `datasets/` -- those must
    already exist with real, uploaded data (see `verify_dataset.py`)."""
    for directory in output_directories(paths):
        os.makedirs(directory, exist_ok=True)
