"""Colab setup for Stage-4 v2 (cache generation and training). Call `setup_stage4_v2()` from a notebook.

  1. The project's standard setup (setup.setup): mounts Drive, clones/updates the repo, installs
     requirements.txt (which pins segmentation-models-pytorch==0.5.0), and wires the dataset/cache env vars.
  2. TJDR_RAW_DIR / TJDR_PROCESSED_DIR -> the verified Drive copy (record §43).
  3. Verifies smp == 0.5.0 (installs exactly that version if not).
  4. Downloads the pinned SE-ResNet-101 ImageNet weights (HF smp-hub, fixed revision) into a Drive cache
     and verifies the exact size + SHA-256. On any failure it raises: there is no fallback encoder.
  5. Verifies the TJDR raw copy (443/110 usable) and the IDRiD segmentation paths.
"""
import os
import posixpath
import subprocess
import sys


def configure_tjdr_env(colab_config):
    tjdr = posixpath.join(colab_config.DATASET_ROOT, "TJDR")
    env = {"TJDR_RAW_DIR": posixpath.join(tjdr, "raw"), "TJDR_PROCESSED_DIR": posixpath.join(tjdr, "processed")}
    os.environ.update(env)
    return env


def ensure_smp():
    import pipeline_v2_config as v2cfg
    try:
        import segmentation_models_pytorch as smp
        ok = smp.__version__ == v2cfg.SMP_VERSION
    except ImportError:
        ok = False
    if not ok:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"segmentation-models-pytorch=={v2cfg.SMP_VERSION}"],
                       check=True)
        raise RuntimeError("segmentation-models-pytorch was (re)installed -- restart the runtime and rerun setup.")
    return v2cfg.SMP_VERSION


def verify_encoder_weights(weights_cache_dir, downloader=None):
    """Fetch + verify the pinned weights. Raises stage4_v2.Stage4WeightsUnavailableError on any mismatch."""
    import stage4_v2
    record = stage4_v2.fetch_pinned_encoder_weights(cache_dir=weights_cache_dir, downloader=downloader)
    print(f"  SE-ResNet-101 weights OK: {record['bytes']} bytes, sha256 {record['sha256']}")
    return record


def verify_datasets():
    import config
    import pipeline_v2_config as v2cfg
    import stage4_v2_data as data
    import tjdr_dataset as tj
    data.idrid_split()                                   # asserts the pinned 44/10 split and its sha
    report = {"tjdr_usable": {s: len(tj.usable_ids(s)) for s in ("train", "test")},
              "idrid_split_sha256": v2cfg.IDRID_V2_SPLIT_SHA256}
    seg_raw = config.dataset_raw_dir("IDRiD/segmentation")
    seg_proc = config.dataset_processed_dir("IDRiD/segmentation")
    missing = [p for p in (seg_raw, seg_proc) if not os.path.isdir(p)]
    if missing:
        raise RuntimeError(f"IDRiD segmentation paths missing on Drive: {missing}")
    report["idrid_segmentation"] = {"raw": seg_raw, "processed": seg_proc}
    return report


def setup_stage4_v2(fetch_weights=True):
    import setup as colab_setup
    info = colab_setup.setup()
    import colab_config
    info["tjdr_env"] = configure_tjdr_env(colab_config)
    info["smp"] = ensure_smp()
    if fetch_weights:
        weights_dir = posixpath.join(colab_config.EXPORTED_MODELS_ROOT, "pretrained_weights", "smp_hub")
        info["encoder_weights"] = verify_encoder_weights(weights_dir)
    info["datasets"] = verify_datasets()
    print("Stage-4 v2 setup complete:", {k: v for k, v in info.items() if k in ("tjdr_env", "smp", "datasets")})
    return info
