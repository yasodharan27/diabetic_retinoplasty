"""Colab setup for Stage-4 v2 (cache generation and training). Call `setup_stage4_v2()` from a notebook.

  1. The project's standard setup (setup.setup): mounts Drive, clones/updates the repo, installs
     requirements.txt (which pins segmentation-models-pytorch==0.5.0), and wires the dataset/cache env vars.
  2. TJDR_RAW_DIR / TJDR_PROCESSED_DIR -> the verified Drive copy (record §43).
  3. Verifies smp == 0.5.0 (installs exactly that version if not).
  4. (training only) Downloads the pinned SE-ResNet-101 ImageNet weights (HF smp-hub, fixed revision) to
     LOCAL disk and verifies the exact size + SHA-256. On any failure it raises: there is no fallback encoder.
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


def _remount_drive():
    from google.colab import drive
    import colab_config
    drive.mount(colab_config.DRIVE_MOUNT_POINT, force_remount=True)


def stage_files(pairs, max_remounts=6, remount=None, log=print, label="files"):
    """Copies (src, dst, sha256) pairs from Drive to local disk, robust to Drive FUSE drops
    (`[Errno 107] Transport endpoint is not connected`): atomic per-file copies (dataset_staging._copy_one,
    which also retries short hiccups), SHA-256 check of every copy, skip of local files that already verify
    (so a rerun resumes), and an automatic Drive remount when the mount itself has dropped.
    Returns {"copied", "skipped", "remounts"}."""
    import dataset_staging
    import stage34_cache_v2 as cache
    remount = remount or _remount_drive
    copied = skipped = remounts = 0
    i = 0
    while i < len(pairs):
        src, dst, want = pairs[i]
        if os.path.exists(dst) and cache.sha256_file(dst) == want:
            skipped += 1
            i += 1
            continue
        try:
            dataset_staging._copy_one(src, dst)
        except OSError as exc:
            if not dataset_staging._is_transient_os_error(exc) or remounts >= max_remounts:
                raise
            remounts += 1
            log(f"  Drive dropped at {os.path.basename(src)} ({exc}); remount {remounts}/{max_remounts}, resuming")
            remount()
            continue
        if cache.sha256_file(dst) != want:
            os.remove(dst)
            raise RuntimeError(f"{dst}: copied file does not match its recorded SHA")
        copied += 1
        i += 1
        if (copied + skipped) % 200 == 0:
            log(f"  {label}: staged {copied + skipped}/{len(pairs)}")
    return {"copied": copied, "skipped": skipped, "remounts": remounts}


def _read_drive_bytes(path, remount, max_remounts, log):
    import dataset_staging
    for attempt in range(max_remounts + 1):
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except OSError as exc:
            if not dataset_staging._is_transient_os_error(exc) or attempt == max_remounts:
                raise
            log(f"  Drive dropped reading {os.path.basename(path)} ({exc}); remounting")
            (remount or _remount_drive)()


def _write_last(local_path, payload):
    import shutil
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp = local_path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(payload)
    shutil.move(tmp, local_path)


def stage_training_cache(drive_dir, local_dir, max_remounts=6, remount=None, log=print):
    """The Stage-4 training cache (manifest["files"] = {file name: {"sha256": ...}}); manifest.json is
    written LAST, byte-for-byte, so its presence means the copy finished."""
    import json
    payload = _read_drive_bytes(os.path.join(drive_dir, "manifest.json"), remount, max_remounts, log)
    manifest = json.loads(payload.decode("utf-8"))
    pairs = [(os.path.join(drive_dir, n), os.path.join(local_dir, n), v["sha256"])
             for n, v in sorted(manifest["files"].items())]
    os.makedirs(local_dir, exist_ok=True)
    res = stage_files(pairs, max_remounts, remount, log, "training cache")
    _write_last(os.path.join(local_dir, "manifest.json"), payload)
    log(f"  training cache staged: {res['copied']} copied, {res['skipped']} already present, {res['remounts']} remounts")
    return res


def stage_bundle(bundle_id, drive_roots, local_roots, max_remounts=6, remount=None, log=print):
    """Copies one v2 bundle (its three generations + bundle manifest) from Drive to local disk with per-file
    SHA checks against the generation manifests; manifests are written last. Returns per-kind results."""
    import json
    import stage34_cache_v2 as cache
    names = {"stage2_rgb_v2": cache.rgb_filename, "stage3_cache_v2": cache.vessel_filename,
             "stage4_cache_v2": cache.pathology_filename}
    bpath = os.path.join(drive_roots["bundle_v2"], bundle_id, cache.MANIFEST_NAMES["bundle_v2"])
    bpayload = _read_drive_bytes(bpath, remount, max_remounts, log)
    bundle = json.loads(bpayload.decode("utf-8"))
    gens = {"stage2_rgb_v2": bundle["stage2_generation"], "stage3_cache_v2": bundle["stage3_generation"],
            "stage4_cache_v2": bundle["stage4_generation"]}
    out = {}
    for kind, gen in gens.items():
        src_gen, dst_gen = os.path.join(drive_roots[kind], gen), os.path.join(local_roots[kind], gen)
        mpayload = _read_drive_bytes(os.path.join(src_gen, cache.MANIFEST_NAMES[kind]), remount, max_remounts, log)
        files = json.loads(mpayload.decode("utf-8"))["files"]
        sub = cache.DATA_SUBDIRS[kind]
        pairs = [(os.path.join(src_gen, sub, names[kind](i)), os.path.join(dst_gen, sub, names[kind](i)), sha)
                 for i, sha in sorted(files.items())]
        out[kind] = stage_files(pairs, max_remounts, remount, log, kind)
        _write_last(os.path.join(dst_gen, cache.MANIFEST_NAMES[kind]), mpayload)
    _write_last(os.path.join(local_roots["bundle_v2"], bundle_id, cache.MANIFEST_NAMES["bundle_v2"]), bpayload)
    log(f"  bundle {bundle_id} staged: {out}")
    return out


def setup_stage4_v2(fetch_weights=True):
    import setup as colab_setup
    info = colab_setup.setup()
    import colab_config
    info["tjdr_env"] = configure_tjdr_env(colab_config)
    info["smp"] = ensure_smp()
    if fetch_weights:
        # Local disk only: Drive cannot hold the Hugging Face cache (no symlinks; FUSE copies fail with EIO).
        # ~200 MB, a few seconds; the SHA/size check is what matters, not where it is cached.
        info["encoder_weights"] = verify_encoder_weights("/content/hf_cache")
    info["datasets"] = verify_datasets()
    print("Stage-4 v2 setup complete:", {k: v for k, v in info.items() if k in ("tjdr_env", "smp", "datasets")})
    return info
