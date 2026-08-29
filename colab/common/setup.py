"""
One-call Colab environment setup: mount Drive, clone/update the repository,
install requirements, enter the repository, and configure environment
variables. The notebook should call `setup()` once instead of repeating any
of this inline.

`google.colab` is only imported inside `mount_drive()`, not at module level,
so this module (and everything that imports it) stays importable outside
Colab -- e.g. for the static validation this project's workflow requires.

Bootstrapping note: this module itself lives inside the repository (under
`colab/common/`), so it cannot be imported until *after* the repository has
been cloned. Every stage notebook's first cell necessarily duplicates a
minimal version of `clone_or_update_repo()` for that one bootstrap step --
see `colab/notebooks/stage01_iqa.ipynb`. `setup()` calls the real
`clone_or_update_repo()` again afterwards (idempotent -- a `git pull` if
already cloned), so the two never drift apart.
"""

import json
import os
import posixpath
import subprocess
import sys
from datetime import datetime

import colab_config


def mount_drive(mount_point=colab_config.DRIVE_MOUNT_POINT):
    from google.colab import drive
    drive.mount(mount_point)
    return mount_point


def clone_or_update_repo(repo_url=colab_config.REPO_URL, repo_dir=colab_config.REPO_DIR,
                          branch=colab_config.REPO_BRANCH):
    if not os.path.isdir(posixpath.join(repo_dir, ".git")):
        print(f"Cloning {repo_url} (branch={branch}) into {repo_dir} ...")
        subprocess.run(["git", "clone", "--branch", branch, repo_url, repo_dir], check=True)
    else:
        print(f"Repository already present at {repo_dir}; pulling latest {branch} ...")
        subprocess.run(["git", "-C", repo_dir, "pull", "origin", branch], check=True)

    for path in (repo_dir, posixpath.join(repo_dir, "colab", "common")):
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo_dir


def install_requirements(repo_dir=colab_config.REPO_DIR, extra_packages=None):
    requirements_path = posixpath.join(repo_dir, "requirements.txt")
    if os.path.exists(requirements_path):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_path],
            check=True,
        )
    if extra_packages:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *extra_packages], check=True)
    print("Dependencies installed.")


def enter_repository(repo_dir=colab_config.REPO_DIR):
    os.chdir(repo_dir)
    print("Entered repository root:", os.getcwd())
    return repo_dir


def configure_environment_variables():
    """Points `config.py`'s generic `dataset_raw_dir()`/`dataset_processed_dir()`
    helpers (and `EYEQ_RAW_DIR` specifically) at Drive, for every dataset a
    downstream stage's dataset loader actually resolves by name:
    `dataset_raw_dir(name)` / `dataset_processed_dir(name)` check the
    environment variable `f'{name.upper()}_RAW_DIR'` /
    `f'{name.upper()}_PROCESSED_DIR'` before falling back to a path inside
    the cloned repo's own (dataset-free) `datasets/` folder -- previously
    only `EYEQ_RAW_DIR` was set here, so a fresh Colab session had no way to
    resolve APTOS2019 or IDRiD from Drive at all (any Stage 05+ dataset
    loader would silently fall back to a nonexistent in-repo path instead).

    IDRiD is not flat -- `lesion_segmentation_dataset.py` (frozen, Stage 04)
    already calls `config.dataset_raw_dir("IDRiD/segmentation")`, so the
    matching environment variable name is literally `IDRID/SEGMENTATION_RAW_DIR`
    (a `/`-containing key -- valid as an in-process `os.environ` entry, even
    though it could not be `export`ed from a shell; nothing here needs to).
    Only `grading`/`localization`/`segmentation` are wired -- IDRiD's own
    bare root has no `raw/`/`processed/` of its own to point at (verified
    locally: `datasets/IDRiD/` contains only the three subset folders), so
    no `IDRID_RAW_DIR`/`IDRID_PROCESSED_DIR` is fabricated here.

    This also wires the frozen Stage 1/3/4 CHECKPOINT paths, and the
    persistent Stage 05/RACAF derived-cache and Stage 05-08+RACAF exported-model
    paths -- a gap found auditing the joint-training design: previously only
    dataset directories were set here, so `image_quality_inference.load_iqa_model()`
    / `vessel_segmentation_inference.load_vessel_model()` /
    `lesion_segmentation_model.load_lesion_model()`'s own `DEFAULT_MODEL_PATH`s
    (each `config.py`-resolved) would fall back to a path inside the just-cloned,
    ephemeral Colab VM checkout -- which never contains these already-trained
    checkpoints -- in a fresh Colab session, exactly the same class of gap this
    function already fixed for datasets. `LOCAL_FEATURE_RESULTS_DIR`/
    `RACAF_RESULTS_DIR` are the two per-image derived-prediction caches that
    must persist across resumed Colab sessions (`config.py`'s
    `LOCAL_FEATURE_EXTRACTION`/`RACAF` dataclasses); `GLOBAL_FEATURE_RESULTS_DIR`/
    `FEATURE_FUSION_RESULTS_DIR`/`CORN_RESULTS_DIR` are intentionally NOT wired
    here -- per their own `config.py` docstrings, none of those three stages
    has any frozen-upstream inference of its own to cache, so nothing is ever
    written there.

    Training/evaluation output paths otherwise are intentionally *not* set
    here -- the notebook passes the current experiment's Drive paths
    explicitly to `train_image_quality.train()` / `evaluate_image_quality.evaluate()`
    instead (see `experiment_manager.py`), which every one of those
    functions already accepts as an argument."""
    env_vars = {
        "EYEQ_RAW_DIR": colab_config.EYEQ_RAW_DIR,
        "EYEQ_PROCESSED_DIR": colab_config.EYEQ_PROCESSED_DIR,
        "APTOS2019_RAW_DIR": colab_config.APTOS2019_RAW_DIR,
        "APTOS2019_PROCESSED_DIR": colab_config.APTOS2019_PROCESSED_DIR,
        "IDRID/GRADING_RAW_DIR": colab_config.IDRID_GRADING_RAW_DIR,
        "IDRID/GRADING_PROCESSED_DIR": colab_config.IDRID_GRADING_PROCESSED_DIR,
        "IDRID/LOCALIZATION_RAW_DIR": colab_config.IDRID_LOCALIZATION_RAW_DIR,
        "IDRID/LOCALIZATION_PROCESSED_DIR": colab_config.IDRID_LOCALIZATION_PROCESSED_DIR,
        "IDRID/SEGMENTATION_RAW_DIR": colab_config.IDRID_SEGMENTATION_RAW_DIR,
        "IDRID/SEGMENTATION_PROCESSED_DIR": colab_config.IDRID_SEGMENTATION_PROCESSED_DIR,
        # Frozen Stage 1/3/4 checkpoints (existing artifacts -- never
        # retrained, never written to by this project).
        "IQA_MODEL_DIR": colab_config.IQA_MODEL_DIR,
        "VESSEL_SEG_MODEL_DIR": colab_config.VESSEL_SEG_MODEL_DIR,
        "LESION_SEG_MODEL_DIR": colab_config.LESION_SEG_MODEL_DIR,
        # Persistent, per-image derived caches (Stage 03/04 predictions,
        # RACAF reliability) -- must survive a Colab VM restart.
        "LOCAL_FEATURE_RESULTS_DIR": colab_config.LOCAL_FEATURE_CACHE_DIR,
        "RACAF_RESULTS_DIR": colab_config.RACAF_CACHE_DIR,
        # Stage 05-08 + RACAF joint ("FinalClassification") exported models --
        # each stage's own save()/load() keeps working independently once the
        # joint checkpoint is sliced back out per stage.
        "LOCAL_FEATURE_MODEL_DIR": colab_config.LOCAL_FEATURE_MODEL_DIR,
        "GLOBAL_FEATURE_MODEL_DIR": colab_config.GLOBAL_FEATURE_MODEL_DIR,
        "FEATURE_FUSION_MODEL_DIR": colab_config.FEATURE_FUSION_MODEL_DIR,
        "RACAF_MODEL_DIR": colab_config.RACAF_MODEL_DIR,
        "CORN_MODEL_DIR": colab_config.CORN_MODEL_DIR,
    }
    os.environ.update(env_vars)
    return env_vars


def _write_session_log(repo_dir, env_vars):
    """Records that a setup() ran, and with what resolved paths, into the
    Drive-global logs/ root -- useful for auditing which Colab sessions
    trained what and when, across all experiments. `env_vars` is
    `configure_environment_variables()`'s own return value, so this log
    always reflects every dataset path actually configured that session,
    not just EyeQ's."""
    os.makedirs(colab_config.LOGS_ROOT, exist_ok=True)
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = posixpath.join(colab_config.LOGS_ROOT, f"setup_{session_id}.json")
    with open(log_path, "w") as f:
        json.dump(
            {
                "session_id": session_id,
                "repo_dir": repo_dir,
                "repo_url": colab_config.REPO_URL,
                "env_vars": env_vars,
            },
            f, indent=2,
        )
    return log_path


def setup(repo_url=colab_config.REPO_URL, repo_dir=colab_config.REPO_DIR,
          branch=colab_config.REPO_BRANCH, drive_mount_point=colab_config.DRIVE_MOUNT_POINT):
    """Run every setup step in order and return a summary dict. This is the
    single function the notebook should call."""
    import drive_paths as _drive_paths

    mount_drive(drive_mount_point)
    clone_or_update_repo(repo_url, repo_dir, branch)
    install_requirements(repo_dir)
    enter_repository(repo_dir)
    env_vars = configure_environment_variables()
    _drive_paths.ensure_output_directories(colab_config.DRIVE)
    log_path = _write_session_log(repo_dir, env_vars)

    summary = {
        "repo_dir": repo_dir,
        "drive_mount_point": drive_mount_point,
        "env_vars": env_vars,
        "session_log": log_path,
    }
    print("\nSetup complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary
