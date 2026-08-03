"""
Shared, model-agnostic training infrastructure for the diabetic retinopathy
Colab notebooks (Image Quality Assessment, Vessel Segmentation, Lesion
Segmentation, Final Classification).

This module intentionally contains NO model architectures and NO
dataset-specific loading code -- it only provides the reusable scaffolding
every training notebook needs: GPU/mixed-precision setup, dataset-folder
resolution, checkpointing + resume-training support, callback wiring
(early stopping, LR scheduling, TensorBoard), and exporting a trained
model's weights back into the repository.

Each notebook clones the repository into the Colab VM and imports this
module from `notebooks/colab_utils.py` -- see any of the four training
notebooks in this folder for the expected usage sequence.
"""

import json
import os
import shutil
import subprocess
import sys

import tensorflow as tf


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def is_colab():
    """True when running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def check_gpu():
    """Print GPU availability and enable memory growth. Returns the GPU device list."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU available: {[g.name for g in gpus]}")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError as e:
                print(f"Could not set memory growth on {g.name}: {e}")
    else:
        print(
            "No GPU detected. In Colab: Runtime > Change runtime type > "
            "Hardware accelerator > GPU, then re-run this cell."
        )
    return gpus


def setup_mixed_precision():
    """Enable mixed_float16 when a GPU is present; otherwise keep float32
    (mixed precision only helps on GPU/TPU, so it's skipped on CPU-only runtimes)."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Mixed precision enabled: {policy.name}")
    else:
        print("No GPU detected -- keeping the default float32 policy.")
    return tf.keras.mixed_precision.global_policy()


# ---------------------------------------------------------------------------
# Project setup (repo clone/update + dependency install)
# ---------------------------------------------------------------------------
# Note: the first notebook cell that clones the repository necessarily
# duplicates a minimal version of `clone_or_update_repo` inline, because
# this module only becomes importable *after* the repo has been cloned.
# The versions here exist so later cells (or a future re-run in the same
# session) can reuse the same logic without re-deriving it.

def clone_or_update_repo(repo_url, repo_dir, branch="main"):
    """Clone the repository if it isn't present yet, otherwise pull the latest branch."""
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        print(f"Cloning {repo_url} (branch={branch}) into {repo_dir} ...")
        subprocess.run(["git", "clone", "--branch", branch, repo_url, repo_dir], check=True)
    else:
        print(f"Repository already present at {repo_dir}; pulling latest {branch} ...")
        subprocess.run(["git", "-C", repo_dir, "pull", "origin", branch], check=True)

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    notebooks_dir = os.path.join(repo_dir, "notebooks")
    if notebooks_dir not in sys.path:
        sys.path.insert(0, notebooks_dir)
    return repo_dir


def install_requirements(repo_dir, extra_packages=None):
    """pip install the repo's requirements.txt plus any notebook-specific extras."""
    requirements_path = os.path.join(repo_dir, "requirements.txt")
    if os.path.exists(requirements_path):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_path],
            check=True,
        )
    if extra_packages:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *extra_packages],
            check=True,
        )
    print("Dependencies installed.")


# ---------------------------------------------------------------------------
# Dataset path configuration
# ---------------------------------------------------------------------------

def resolve_dataset_dir(repo_dir, dataset_subfolder):
    """Resolve (and create if missing) `<repo_dir>/datasets/<dataset_subfolder>`.

    Datasets are expected to be uploaded directly into the Colab VM's copy of
    the repository (e.g. via the Colab file browser, `files.upload()`, or
    unzipping an uploaded archive) -- NOT mounted from Google Drive.
    """
    dataset_dir = os.path.join(repo_dir, "datasets", dataset_subfolder)
    os.makedirs(dataset_dir, exist_ok=True)
    contents = os.listdir(dataset_dir)
    if not contents:
        print(f"[WARNING] {dataset_dir} is empty.")
        print("Upload the dataset into this exact folder before running the training cell, e.g.:")
        print("  - drag/drop files into it using the Colab file browser (left sidebar), or")
        print("  - from google.colab import files; files.upload()  # then move the result here, or")
        print(f"  - !unzip -q <archive>.zip -d \"{dataset_dir}\"")
    else:
        print(f"Found {len(contents)} item(s) in {dataset_dir}")
    return dataset_dir


# ---------------------------------------------------------------------------
# Resume-training support
# ---------------------------------------------------------------------------

class EpochStateLogger(tf.keras.callbacks.Callback):
    """Writes the last completed epoch to a small JSON file after every epoch,
    so a later session can resume training at the correct `initial_epoch`."""

    def __init__(self, state_path):
        super().__init__()
        self.state_path = state_path

    def on_epoch_end(self, epoch, logs=None):
        with open(self.state_path, "w") as f:
            json.dump({"last_completed_epoch": epoch + 1}, f)


def get_resume_epoch(state_path):
    """Return the epoch to resume from (0 if no prior run state exists)."""
    if not os.path.exists(state_path):
        return 0
    with open(state_path) as f:
        state = json.load(f)
    return state.get("last_completed_epoch", 0)


# ---------------------------------------------------------------------------
# Callbacks: checkpointing, early stopping, LR scheduling, TensorBoard
# ---------------------------------------------------------------------------

def build_training_callbacks(
    run_dir,
    monitor="val_loss",
    mode="min",
    early_stopping_patience=8,
    reduce_lr_patience=4,
    reduce_lr_factor=0.5,
    min_lr=1e-6,
    save_weights_only=True,
):
    """Build the standard callback set used by every training notebook.

    `save_weights_only=False` saves complete models (`.keras`) instead of
    weights-only checkpoints (`.weights.h5`) -- Keras infers the format from
    the file extension, so the checkpoint filenames below switch to match.

    Returns (callbacks, paths) where `paths` is a dict with the resolved
    best/last checkpoint paths, the epoch-state file, and the TensorBoard
    log directory, so the calling notebook can reference them later
    (resume checks, export step, etc.) without re-deriving them.
    """
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    ext = "weights.h5" if save_weights_only else "keras"
    best_path = os.path.join(checkpoints_dir, f"best.{ext}")
    last_path = os.path.join(checkpoints_dir, f"last.{ext}")
    state_path = os.path.join(run_dir, "epoch_state.json")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_path, monitor=monitor, mode=mode,
            save_best_only=True, save_weights_only=save_weights_only, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            last_path, save_best_only=False, save_weights_only=save_weights_only, verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode=mode, patience=early_stopping_patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode=mode, factor=reduce_lr_factor,
            patience=reduce_lr_patience, min_lr=min_lr, verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1),
        EpochStateLogger(state_path),
    ]
    paths = {
        "best_weights": best_path,
        "last_weights": last_path,
        "epoch_state": state_path,
        "logs_dir": logs_dir,
    }
    return callbacks, paths


# ---------------------------------------------------------------------------
# Validation / reporting
# ---------------------------------------------------------------------------

def evaluate_and_report(model, val_ds):
    """Run model.evaluate on the validation set and print each metric."""
    results = model.evaluate(val_ds, return_dict=True, verbose=1)
    print("Validation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    return results


def plot_history(history, output_path=None):
    """Generic training-curve plot: every non-'val_' metric in `history.history`
    alongside its validation counterpart, if present. No model-specific assumptions."""
    import matplotlib.pyplot as plt

    metrics = [k for k in history.history if not k.startswith("val_")]
    if not metrics:
        print("No metrics found in history to plot.")
        return

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(history.history[metric], label=f"train_{metric}")
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label=val_key)
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved training curves to {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Exporting the trained model back into the repository
# ---------------------------------------------------------------------------

def export_trained_model(best_weights_path, repo_dir, module_key,
                          filename="best_model.weights.h5", also_offer_download=True):
    """Copy the best checkpoint into `<repo_dir>/models/<module_key>/` and,
    when running in Colab, trigger a browser download of the same file."""
    export_dir = os.path.join(repo_dir, "models", module_key)
    os.makedirs(export_dir, exist_ok=True)
    dest_path = os.path.join(export_dir, filename)
    shutil.copy2(best_weights_path, dest_path)
    print(f"Copied best model to {dest_path}")

    if also_offer_download and is_colab():
        from google.colab import files
        files.download(dest_path)
        print("Triggered browser download of the exported model.")

    return dest_path


def git_commit_and_push(repo_dir, message, paths=None, push=False):
    """Optional helper to commit exported weights back into the repo.

    Nothing is pushed unless `push=True` is passed explicitly -- review the
    diff yourself (`git -C <repo_dir> status` / `git -C <repo_dir> diff --cached`)
    before pushing trained weights from a Colab session.
    """
    paths = paths or ["."]
    subprocess.run(["git", "-C", repo_dir, "add", *paths], check=True)
    status = subprocess.run(
        ["git", "-C", repo_dir, "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    if not status.stdout.strip():
        print("Nothing to commit.")
        return

    subprocess.run(["git", "-C", repo_dir, "commit", "-m", message], check=True)
    print("Committed locally.")
    if push:
        subprocess.run(["git", "-C", repo_dir, "push"], check=True)
        print("Pushed.")
    else:
        print(
            "push=False -- commit created locally but NOT pushed. "
            "Review with `git -C <repo_dir> log -1` and push manually if it looks right."
        )
