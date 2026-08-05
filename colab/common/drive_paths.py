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
        +-- logs/

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
"""

import os
import posixpath
from dataclasses import dataclass

DRIVE_PROJECT_FOLDER_NAME = "DiabeticRetinopathy"

PIPELINE_MODULES = ("IQA", "VesselSegmentation", "LesionSegmentation", "FinalClassification")


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
    idrid_dataset_dir: str

    # experiments/<module>/ -- one isolated, timestamped folder per training run
    experiments_root: str
    experiment_dirs: dict  # module name -> experiments_root/<module>

    # tensorboard/, exported_models/<module>/, logs/ -- other outputs
    tensorboard_root: str
    exported_models_root: str
    exported_model_dirs: dict  # module name -> exported_models_root/<module>
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

    eyeq_dataset_dir = posixpath.join(datasets_root, "EyeQ")

    return DrivePaths(
        drive_mount_point=drive_mount_point,
        project_root=project_root,
        datasets_root=datasets_root,
        eyeq_dataset_dir=eyeq_dataset_dir,
        eyeq_raw_dir=posixpath.join(eyeq_dataset_dir, "raw"),
        eyeq_processed_dir=posixpath.join(eyeq_dataset_dir, "processed"),
        aptos2019_dataset_dir=posixpath.join(datasets_root, "APTOS2019"),
        idrid_dataset_dir=posixpath.join(datasets_root, "IDRiD"),
        experiments_root=experiments_root,
        experiment_dirs={m: posixpath.join(experiments_root, m) for m in PIPELINE_MODULES},
        tensorboard_root=posixpath.join(project_root, "tensorboard"),
        exported_models_root=exported_models_root,
        exported_model_dirs={m: posixpath.join(exported_models_root, m) for m in PIPELINE_MODULES},
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
        paths.logs_root,
    ]


def ensure_output_directories(paths: DrivePaths):
    """Create every output directory (`experiments/`, `tensorboard/`,
    `exported_models/`, `logs/`, and their module subfolders) if missing.
    Never creates or touches anything under `datasets/` -- those must
    already exist with real, uploaded data (see `verify_dataset.py`)."""
    for directory in output_directories(paths):
        os.makedirs(directory, exist_ok=True)
