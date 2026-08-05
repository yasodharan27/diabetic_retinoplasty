"""
Low-level environment inspection & setup primitives for the Colab training
workflow. Gathers information and performs setup actions but never raises
on a "bad" result -- `verify_environment.py` builds pass/fail checks (that
do raise) on top of these.

GPU memory growth and mixed precision are already implemented in
`training.trainer` (`check_gpu`, `enable_mixed_precision`); this module
reuses them rather than reimplementing the same `tf.config` calls.
"""

import platform
import subprocess

import tensorflow as tf

from training import check_gpu as _check_gpu
from training import enable_mixed_precision as _enable_mixed_precision


def is_colab():
    """True when running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_python_version():
    return platform.python_version()


def get_tensorflow_version():
    return tf.__version__


def get_keras_version():
    """Standalone `keras` package version if installed, else None (some TF
    builds ship only `tf.keras`, with no separately importable `keras`)."""
    try:
        import keras
        return keras.__version__
    except ImportError:
        return None


def get_gpu_devices():
    """List of physical GPU devices TensorFlow can see (empty on CPU-only runtimes)."""
    return tf.config.list_physical_devices("GPU")


def get_gpu_name(gpu=None):
    """Device name of the first GPU (or a given device), or None if no GPU."""
    gpus = get_gpu_devices()
    if not gpus:
        return None
    device = gpu or gpus[0]
    details = tf.config.experimental.get_device_details(device)
    return details.get("device_name")


def get_gpu_memory_mib():
    """Total memory (MiB) of the first GPU via `nvidia-smi`, or None if
    unavailable (no GPU, or `nvidia-smi` not on PATH)."""
    if not get_gpu_devices():
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 -- nvidia-smi may be missing/unavailable
        return None


def get_cuda_version():
    """CUDA version this TensorFlow build was compiled against, or None
    (e.g. a CPU-only TF build has no CUDA version)."""
    build_info = tf.sysconfig.get_build_info()
    return build_info.get("cuda_version")


def check_gpu():
    """Print GPU availability and enable memory growth (delegates to
    `training.check_gpu`). Returns the GPU device list."""
    return _check_gpu()


def enable_mixed_precision(enabled=True):
    """Enable `mixed_float16` when a GPU is present, else `float32`
    (delegates to `training.enable_mixed_precision`). Returns the resulting policy."""
    return _enable_mixed_precision(enabled)


def get_git_commit_hash(repo_dir):
    """Short commit hash of `repo_dir`'s current `HEAD`, or None if it
    isn't a git repository (or git isn't available)."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


# Package name (as pip/requirements.txt spells it) -> the name it's actually
# imported under, for the handful of this project's dependencies where the
# two differ.
_IMPORT_NAME_OVERRIDES = {
    "opencv-python": "cv2",
    "python-dotenv": "dotenv",
    "scikit-learn": "sklearn",
}


def _parse_requirement_name(requirement_line):
    """Extract the bare package name from a `requirements.txt` line such as
    `tensorflow>=2.9.0` -> `tensorflow`."""
    name = requirement_line.strip()
    for marker in (">=", "<=", "==", ">", "<", "~="):
        name = name.split(marker, 1)[0]
    return name.strip()


def get_missing_packages(requirements_path):
    """Returns the list of package names from `requirements_path` that
    cannot currently be imported."""
    import importlib

    missing = []
    with open(requirements_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            package_name = _parse_requirement_name(line)
            import_name = _IMPORT_NAME_OVERRIDES.get(package_name, package_name.replace("-", "_"))
            try:
                importlib.import_module(import_name)
            except ImportError:
                missing.append(package_name)
    return missing
