"""
Low-level environment inspection & setup primitives for the Colab training
workflow. Gathers information and performs setup actions but never raises
on a "bad" result -- `verify_environment.py` builds pass/fail checks (that
do raise) on top of these.

GPU memory growth and mixed precision are already implemented in
`training.trainer` (`check_gpu`, `enable_mixed_precision`); this module
reuses them rather than reimplementing the same `tf.config` calls.
"""

import importlib
import importlib.metadata
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
# imported under. Kept only as a last-resort fallback for the rare case
# `_distribution_import_names()` below finds nothing (e.g. a distribution
# installed without the metadata Python's own tooling normally relies on) --
# the primary resolution mechanism is `importlib.metadata`, not this table,
# specifically so a new mismatched package (like scikit-image -> skimage)
# does not require a new hardcoded entry here to be recognized correctly.
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


def _normalize_distribution_name(name):
    """PEP 503 normalization (case- and separator-insensitive), so
    'scikit-image', 'scikit_image', and 'Scikit-Image' are all recognized
    as the same distribution when matching a requirements.txt entry against
    installed package metadata."""
    return name.strip().lower().replace("_", "-")


def _packages_distributions_index():
    """Inverted index {normalized distribution name -> {import names}},
    built from `importlib.metadata.packages_distributions()` -- the
    standard library's own reverse mapping from importable top-level module
    name to the PyPI distribution(s) that installed it (Python >= 3.10).
    Not cached across calls: this scans currently-installed distributions,
    and `setup.py` may `pip install` new ones between verification runs
    within the same process, so a stale cache could silently under-report.
    Returns an empty dict on Python < 3.10, where this stdlib function
    doesn't exist -- callers fall back to per-distribution metadata lookup."""
    packages_distributions = getattr(importlib.metadata, "packages_distributions", None)
    if packages_distributions is None:
        return {}
    index = {}
    for import_name, dist_names in packages_distributions().items():
        for dist_name in dist_names:
            index.setdefault(_normalize_distribution_name(dist_name), set()).add(import_name)
    return index


def _distribution_import_names(distribution_name, index):
    """The set of top-level import module names the installed PyPI
    distribution `distribution_name` actually provides, resolved via real
    package metadata (not guessed from the distribution's own name) -- this
    is what makes 'scikit-image' -> {'skimage'}, 'opencv-python' -> {'cv2'},
    'python-dotenv' -> {'dotenv'}, etc. resolve correctly for any
    distribution/import-name mismatch, not just ones this project happens
    to hardcode.

    Tries, in order:
      1. `index` (from `_packages_distributions_index()`) -- reflects what's
         actually importable in this environment right now.
      2. The distribution's own `top_level.txt` metadata file via
         `importlib.metadata.distribution()` -- covers Python < 3.10 (no
         `packages_distributions()`) and any distribution the inverted
         index above didn't pick up.
    Returns an empty set if the distribution isn't installed, or neither
    mechanism finds an import name for it -- callers fall back to the
    manual `_IMPORT_NAME_OVERRIDES` table and finally a naive
    dash-to-underscore guess.
    """
    from_index = index.get(_normalize_distribution_name(distribution_name))
    if from_index:
        return from_index

    try:
        dist = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return set()

    top_level = dist.read_text("top_level.txt")
    if not top_level:
        return set()
    return {line.strip() for line in top_level.splitlines() if line.strip()}


def _import_name_candidates(package_name, index):
    """Every plausible import module name for `package_name`, most
    authoritative first: real installed-package metadata, then this
    project's small manual-override table, then the legacy naive
    dash-to-underscore guess -- so a package metadata resolution misses is
    never worse off than before this function existed."""
    candidates = list(_distribution_import_names(package_name, index))
    override = _IMPORT_NAME_OVERRIDES.get(package_name)
    if override and override not in candidates:
        candidates.append(override)
    naive = package_name.replace("-", "_")
    if naive not in candidates:
        candidates.append(naive)
    return candidates


def _is_importable(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def get_missing_packages(requirements_path):
    """Returns the list of package names from `requirements_path` that
    cannot currently be imported under any of their plausible import
    names (see `_import_name_candidates`)."""
    index = _packages_distributions_index()

    missing = []
    with open(requirements_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            package_name = _parse_requirement_name(line)
            candidates = _import_name_candidates(package_name, index)
            if not any(_is_importable(name) for name in candidates):
                missing.append(package_name)
    return missing
