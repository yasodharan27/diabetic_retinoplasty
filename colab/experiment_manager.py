"""
Reusable, isolated experiment folders for Colab training runs.

Every call to `create_experiment()` makes a new, timestamped, never-reused
directory under `experiments/<module>/` with the exact subfolder layout the
project's training run:

    experiments/<module>/YYYY-MM-DD_HH-MM-SS/
        checkpoints/   -- best.keras, last.keras, metrics.csv, epoch_state.json
                           (written by training.Trainer -- see note below)
        logs/          -- TensorBoard event files (also training.Trainer)
        tensorboard/   -- archival copy of logs/, populated by
                           archive_tensorboard_logs() after training finishes
        evaluation/    -- confusion matrix / ROC / calibration plots, report JSON
        predictions/   -- sample-prediction visualizations
        metadata.json  -- run metadata (git commit, versions, hyperparameters, ...)

Note on logs/ vs. tensorboard/: `training.trainer.TrainingConfig` (which
this refactor intentionally does not modify -- see PROJECT_CODE.md) always
derives its checkpoint and TensorBoard directories as `<run_dir>/checkpoints`
and `<run_dir>/logs`. Passing this experiment's root as `run_dir` therefore
makes `checkpoints/` and `logs/` exactly what Trainer produces, with zero
changes to training/. `tensorboard/` has no natural producer under that
fixed convention, so it starts empty and is filled by
`archive_tensorboard_logs()` (a real, one-time copy of the live `logs/`
event files) once training completes, for anyone who expects a
`tensorboard/` folder to hold TensorBoard data.
"""

import datetime
import json
import os
import posixpath
import shutil
import uuid
from dataclasses import dataclass

import environment

METADATA_FILENAME = "metadata.json"
EXPERIMENT_SUBFOLDERS = ("checkpoints", "logs", "tensorboard", "evaluation", "predictions")


@dataclass(frozen=True)
class ExperimentPaths:
    root: str
    checkpoints_dir: str
    logs_dir: str
    tensorboard_dir: str
    evaluation_dir: str
    predictions_dir: str
    metadata_path: str


def _experiment_paths_for(root):
    # posixpath, not os.path: `root` is always a Colab (Linux) Drive path
    # regardless of the OS this module happens to be imported from.
    return ExperimentPaths(
        root=root,
        checkpoints_dir=posixpath.join(root, "checkpoints"),
        logs_dir=posixpath.join(root, "logs"),
        tensorboard_dir=posixpath.join(root, "tensorboard"),
        evaluation_dir=posixpath.join(root, "evaluation"),
        predictions_dir=posixpath.join(root, "predictions"),
        metadata_path=posixpath.join(root, METADATA_FILENAME),
    )


def _build_metadata(repo_dir, **run_metadata):
    gpus = environment.get_gpu_devices()
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit_hash": environment.get_git_commit_hash(repo_dir),
        "tensorflow_version": environment.get_tensorflow_version(),
        "python_version": environment.get_python_version(),
        "gpu_model": environment.get_gpu_name() if gpus else None,
    }
    metadata.update(run_metadata)
    return metadata


def create_experiment(experiments_root, repo_dir, **run_metadata):
    """Create a new, isolated, timestamped experiment folder under
    `experiments_root` (e.g. `colab_config.IQA_EXPERIMENTS_DIR`).

    `repo_dir` is used only to look up the current git commit hash for
    `metadata.json`. Any additional keyword arguments (e.g. `dataset_path`,
    `batch_size`, `learning_rate`, `epochs`, `image_size`, `random_seed`)
    are recorded verbatim into `metadata.json` alongside the automatically
    gathered fields (git commit, timestamp, TensorFlow/Python versions, GPU
    model).

    Never overwrites a previous experiment: the timestamp has one-second
    resolution, so on the rare chance of a collision (two runs started in
    the same second) a short random suffix is appended instead of reusing
    the existing folder.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = posixpath.join(experiments_root, timestamp)
    if os.path.exists(root):
        root = f"{root}_{uuid.uuid4().hex[:6]}"

    paths = _experiment_paths_for(root)
    for subfolder in EXPERIMENT_SUBFOLDERS:
        os.makedirs(posixpath.join(root, subfolder), exist_ok=True)

    metadata = _build_metadata(repo_dir, **run_metadata)
    with open(paths.metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created experiment: {root}")
    return paths


def resume_experiment(experiment_root):
    """Return `ExperimentPaths` for an *existing* experiment directory
    (one previously returned by `create_experiment()`), for resuming an
    interrupted run into the same folder rather than starting a new one.
    Validates the expected subfolders and metadata.json are present."""
    if not os.path.isdir(experiment_root):
        raise RuntimeError(f"Cannot resume: experiment directory not found at {experiment_root}.")

    paths = _experiment_paths_for(experiment_root)
    missing = [s for s in EXPERIMENT_SUBFOLDERS if not os.path.isdir(posixpath.join(experiment_root, s))]
    if missing:
        raise RuntimeError(
            f"Cannot resume: {experiment_root} is missing expected subfolder(s) {missing} "
            "-- is this really a directory created by create_experiment()?"
        )
    if not os.path.isfile(paths.metadata_path):
        raise RuntimeError(f"Cannot resume: {paths.metadata_path} not found.")

    print(f"Resuming experiment: {experiment_root}")
    return paths


def resolve_experiment(experiments_root, repo_dir, resume_from=None, **run_metadata):
    """`resume_from=None` (default) creates a new experiment; passing an
    existing experiment's root path resumes into it instead. The single
    entry point the notebook's RESUME_TRAINING / RESUME_EXPERIMENT_DIR
    cell should call."""
    if resume_from is None:
        return create_experiment(experiments_root, repo_dir, **run_metadata)
    return resume_experiment(resume_from)


def archive_tensorboard_logs(experiment: ExperimentPaths):
    """Copy the live TensorBoard event files from `logs/` into
    `tensorboard/` (see the module docstring for why these two folders
    exist separately). Safe to call multiple times; safe to call on an
    empty `logs/` (e.g. before training has produced any events yet)."""
    if not os.path.isdir(experiment.logs_dir):
        return
    shutil.copytree(experiment.logs_dir, experiment.tensorboard_dir, dirs_exist_ok=True)
    print(f"Archived TensorBoard logs: {experiment.logs_dir} -> {experiment.tensorboard_dir}")
