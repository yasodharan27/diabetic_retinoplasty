"""
Reusable environment verification for the Colab training workflow.

Every `verify_*` function raises `RuntimeError` with a clear, specific
message on failure -- callers (the notebook, or `verify_all`) should let
that exception propagate and abort the run rather than catching it, per
this project's "abort immediately" verification policy.

Built entirely on `environment.py`'s info-gathering primitives; nothing
here talks to `tf.config` or `subprocess` directly.
"""

import os

import environment


def _version_tuple(version_string):
    return tuple(int(part) for part in version_string.split(".")[:3] if part.isdigit())


def verify_python_version(min_version="3.8"):
    current = environment.get_python_version()
    if _version_tuple(current) < _version_tuple(min_version):
        raise RuntimeError(f"Python {current} is older than the minimum required {min_version}.")
    print(f"[OK] Python version {current} >= {min_version}")
    return current


def verify_tensorflow_version(min_version="2.9.0"):
    """2.9.0 matches this project's requirements.txt (`tensorflow>=2.9.0`) --
    the whole training stack (training/, image_quality_model.py) needs Keras 3 APIs."""
    current = environment.get_tensorflow_version()
    if _version_tuple(current) < _version_tuple(min_version):
        raise RuntimeError(
            f"TensorFlow {current} is older than the minimum required {min_version} "
            "(see requirements.txt)."
        )
    print(f"[OK] TensorFlow version {current} >= {min_version}")
    return current


def verify_gpu_available(required=True):
    gpus = environment.get_gpu_devices()
    if not gpus:
        message = (
            "No GPU detected. In Colab: Runtime > Change runtime type > "
            "Hardware accelerator > GPU, then re-run from this cell."
        )
        if required:
            raise RuntimeError(message)
        print(f"[WARN] {message}")
        return gpus
    name = environment.get_gpu_name() or "unknown model"
    print(f"[OK] GPU available: {name} ({len(gpus)} device(s))")
    return gpus


def verify_cuda_available(required=True):
    gpus = environment.get_gpu_devices()
    if not gpus:
        print("[SKIP] No GPU present -- CUDA check not applicable.")
        return None
    cuda_version = environment.get_cuda_version()
    if not cuda_version:
        message = "A GPU is present but this TensorFlow build reports no CUDA version (CPU-only build?)."
        if required:
            raise RuntimeError(message)
        print(f"[WARN] {message}")
        return None
    print(f"[OK] CUDA version: {cuda_version}")
    return cuda_version


def verify_mixed_precision():
    gpus = environment.get_gpu_devices()
    policy = environment.enable_mixed_precision(True)
    expected_name = "mixed_float16" if gpus else "float32"
    if policy.name != expected_name:
        raise RuntimeError(f"Unexpected mixed precision policy: {policy.name} (expected {expected_name}).")
    print(f"[OK] Mixed precision policy: {policy.name}")
    return policy


def verify_repository_path(repo_dir, required_files=("config.py", "train_image_quality.py")):
    if not os.path.isdir(repo_dir):
        raise RuntimeError(f"Repository directory not found at {repo_dir}.")
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        raise RuntimeError(f"{repo_dir} exists but is not a git repository (no .git/).")
    missing = [f for f in required_files if not os.path.isfile(os.path.join(repo_dir, f))]
    if missing:
        raise RuntimeError(
            f"{repo_dir} does not look like the expected repository -- missing: {missing}."
        )
    print(f"[OK] Repository present at {repo_dir}")
    return repo_dir


def verify_google_drive_mounted(drive_mount_point):
    my_drive = os.path.join(drive_mount_point, "MyDrive")
    if not os.path.isdir(my_drive):
        raise RuntimeError(
            f"Google Drive does not appear to be mounted at {drive_mount_point} "
            "(MyDrive/ not found). Run setup.mount_drive() first."
        )
    print(f"[OK] Google Drive mounted at {drive_mount_point}")
    return drive_mount_point


def verify_required_packages(requirements_path):
    missing = environment.get_missing_packages(requirements_path)
    if missing:
        raise RuntimeError(
            f"Missing required package(s) from {requirements_path}: {missing}. "
            "Run setup.install_requirements() first."
        )
    print(f"[OK] All packages in {requirements_path} are importable")
    return []


def verify_all(repo_dir, drive_mount_point, requirements_path, require_gpu=True):
    """Run every check in sequence, aborting on the first failure. Returns a
    small report dict once every check has passed."""
    python_version = verify_python_version()
    tensorflow_version = verify_tensorflow_version()
    verify_repository_path(repo_dir)
    verify_google_drive_mounted(drive_mount_point)
    verify_required_packages(requirements_path)
    gpus = verify_gpu_available(required=require_gpu)
    cuda_version = verify_cuda_available(required=require_gpu and bool(gpus))
    mixed_precision_policy = verify_mixed_precision()

    return {
        "python_version": python_version,
        "tensorflow_version": tensorflow_version,
        "gpu_count": len(gpus),
        "gpu_name": environment.get_gpu_name(),
        "cuda_version": cuda_version,
        "mixed_precision_policy": mixed_precision_policy.name,
    }
