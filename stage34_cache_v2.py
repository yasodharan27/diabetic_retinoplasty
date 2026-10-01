"""Stage 3/4 v2 cache lineage: file formats, manifests, guards and the C2 pyramid features
(research record §40; `research/Grade3vs4_Architecture_Research/IMPLEMENTATION_SPEC_STAGE3_8_V2.md`
§2, §10, §11). numpy only -- imported by the PyTorch Stage-4 side and the Keras Stage-5..8 side,
which is what makes the cache the framework boundary.

Rules enforced here (the user's cache rule, 2026-10-01):
  * the v2 pipeline NEVER reads a legacy Stage-4 cache or the legacy Stage-4 model: legacy
    directories and legacy file names are refused, the legacy Stage-4 SHA is deny-listed, and every
    v2 reader requires the expected SHA/generation explicitly (no defaults, no fallback);
  * legacy caches are never written: every writer refuses legacy locations;
  * the Stage-3 vessel cache may be reused, but only as `_vessel_` files and only after the parity
    check (`stage3_parity`).

Pathology file: `APTOS_<id>_pathology-s4v2_512x512.npz`, keys
    maps           uint8 (512, 512, 2K)  = round(p * 255), channels [c:mean, c:max for c in classes]
    channels       str   (2K,)
    stage4_sha256, stage3_sha256, gen_id, image_id, schema_version
"""
import datetime
import hashlib
import json
import os
import re

import numpy as np

import pipeline_v2_config as v2cfg


class CacheLineageError(RuntimeError):
    """Base class: a v2 cache/model does not have the lineage the caller required."""


class LegacyArtifactError(CacheLineageError):
    """A legacy cache/model location or file was about to be read or written by the v2 pipeline."""


class DenyListedModelError(CacheLineageError):
    """The deny-listed legacy Stage-4 model (or a cache produced by it) was encountered."""


class ManifestMismatchError(CacheLineageError):
    """A file, its embedded metadata and the manifest disagree."""


class IncompleteCacheError(CacheLineageError):
    """A cache generation does not cover the required population."""


class StaleCacheError(CacheLineageError):
    """A freshness canary or the Stage-3 parity check failed."""


DENY_LISTED_SHA256 = frozenset({v2cfg.LEGACY_STAGE4_SHA256})
PATHOLOGY_SUFFIX = "_pathology-s4v2_{size}x{size}.npz"
_LEGACY_LESION_NAME = re.compile(r"_lesion_\d+x\d+\.npy$|_lesion_prob|lesion_maps", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------- hashing and guards

def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_sha(value, what):
    if not isinstance(value, str) or not _SHA256.match(value):
        raise ValueError(f"{what} must be a full lowercase SHA-256 hex digest, got {value!r}")
    return value


def assert_not_deny_listed(sha256, what="model"):
    if sha256 in DENY_LISTED_SHA256:
        raise DenyListedModelError(f"{what} sha256 {sha256} is the deny-listed legacy Stage-4 model; "
                                   "the v2 pipeline never uses it or anything derived from it.")


def _norm(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path)).replace("\\", os.sep)))


def _is_within(path, root):
    path, root = _norm(path), _norm(root)
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:          # different drives on Windows
        return False


def assert_not_legacy_path(path, legacy_dirs=v2cfg.LEGACY_DIRS):
    """Refuse any path inside a legacy cache/model directory or carrying a legacy lesion-cache name.
    Every v2 reader and writer calls this before touching the file system."""
    for root in legacy_dirs:
        if root and _is_within(path, root):
            raise LegacyArtifactError(f"{path} lies inside the legacy location {root}; the v2 pipeline "
                                      "never reads or writes legacy caches/models.")
    if _LEGACY_LESION_NAME.search(os.path.basename(str(path))):
        raise LegacyArtifactError(f"{path} has a legacy Stage-4 lesion-cache file name.")
    if "cache_archive" in str(path).replace("\\", "/").split("/"):
        raise LegacyArtifactError(f"{path} is inside a legacy mixed cache archive.")
    return path


def assert_legacy_vessel_source(path):
    """The ONE permitted read from a legacy location: a loose Stage-3 vessel file
    (`APTOS_<id>_vessel_512x512.npy`) copied into `Stage3/s3-91f0cada/` after the parity check.
    Never a lesion file, never an archive shard."""
    name = os.path.basename(str(path))
    if not re.fullmatch(r"APTOS_[0-9a-f]+_vessel_512x512\.npy", name):
        raise LegacyArtifactError(f"{path} is not a loose Stage-3 vessel cache file.")
    if "cache_archive" in str(path).replace("\\", "/").split("/"):
        raise LegacyArtifactError(f"{path} is inside a legacy mixed cache archive.")
    return path


def assert_legacy_rgb_source(path):
    """The other permitted legacy read: a loose canonical Stage-2 RGB file (`APTOS_<id>_rgb_512x512.npy`,
    = joint_training_dataset._resize_rgb_01 of the Stage-2 output), copied into `Stage2/rgb-v1/` after its
    parity check. It is Stage-2 output, not Stage-4; never a lesion file, never an archive shard."""
    name = os.path.basename(str(path))
    if not re.fullmatch(r"APTOS_[0-9a-f]+_rgb_512x512\.npy", name):
        raise LegacyArtifactError(f"{path} is not a loose canonical RGB cache file.")
    if "cache_archive" in str(path).replace("\\", "/").split("/"):
        raise LegacyArtifactError(f"{path} is inside a legacy mixed cache archive.")
    return path


def _atomic_write_bytes(path, writer):
    assert_not_legacy_path(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.part"
    writer(tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- channels and ids

def channel_names(classes):
    classes = tuple(classes)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError(f"classes must be a non-empty tuple of unique names, got {classes}")
    for c in classes:
        if c not in v2cfg.CANONICAL_CLASSES:
            raise ValueError(f"unknown Stage-4 class {c!r}; canonical: {v2cfg.CANONICAL_CLASSES}")
    return tuple(f"{c}:{p}" for c in classes for p in v2cfg.POOLINGS)


def classes_from_channels(channels):
    channels = tuple(str(c) for c in channels)
    if len(channels) % len(v2cfg.POOLINGS):
        raise ManifestMismatchError(f"{len(channels)} channels is not a multiple of "
                                    f"{len(v2cfg.POOLINGS)}")
    classes = tuple(channels[i].split(":")[0] for i in range(0, len(channels), len(v2cfg.POOLINGS)))
    if channel_names(classes) != channels:
        raise ManifestMismatchError(f"channel list {channels} is not in the [c:mean, c:max] order")
    return classes


def stage4_generation_id(stage4_sha256, num_classes):
    _check_sha(stage4_sha256, "stage4_sha256")
    assert_not_deny_listed(stage4_sha256)
    return f"s4v2-{stage4_sha256[:12]}-K{int(num_classes)}"


def bundle_id(stage2_generation, stage3_generation, stage4_generation):
    return f"{stage2_generation}__{stage3_generation}__{stage4_generation}"


def pathology_filename(image_id, size=v2cfg.CACHE_SIZE):
    return f"APTOS_{image_id}" + PATHOLOGY_SUFFIX.format(size=size)


def vessel_filename(image_id, size=v2cfg.CACHE_SIZE):
    return f"APTOS_{image_id}_vessel_{size}x{size}.npy"


def rgb_filename(image_id, size=v2cfg.CACHE_SIZE):
    """The canonical RGB cache name (`joint_training_dataset._canonical_rgb_cache_path`):
    float32 (512, 512, 3) = `_resize_rgb_01(native Stage-2 image, 512)`."""
    return f"APTOS_{image_id}_rgb_{size}x{size}.npy"


# Generation directory layout: <root>/<generation_id>/{MANIFEST_NAMES[kind], DATA_SUBDIRS[kind]/...}
MANIFEST_NAMES = {"stage2_rgb_v2": "stage2_manifest.json", "stage3_cache_v2": "stage3_manifest.json",
                  "stage4_cache_v2": "stage4_manifest.json", "bundle_v2": "bundle_manifest.json"}
DATA_SUBDIRS = {"stage2_rgb_v2": "rgb", "stage3_cache_v2": "vessel", "stage4_cache_v2": "pathology"}


def stage4_generation_dir(generation_id, root=v2cfg.STAGE4_ROOT):
    return os.path.join(root, generation_id)


# --------------------------------------------------------------------------- pooling and packing

def block_pool_mean_max(probabilities, factor=v2cfg.POOL_FACTOR):
    """Exact non-overlapping factor x factor block mean and max of (H, W, K) probabilities
    (H, W divisible by factor). Returns (mean, max), each (H/f, W/f, K) float32."""
    p = np.asarray(probabilities, dtype=np.float32)
    if p.ndim != 3:
        raise ValueError(f"expected (H, W, K), got {p.shape}")
    h, w, k = p.shape
    if h % factor or w % factor:
        raise ValueError(f"{p.shape} is not divisible by the pooling factor {factor}")
    blocks = p.reshape(h // factor, factor, w // factor, factor, k)
    return blocks.mean(axis=(1, 3), dtype=np.float64).astype(np.float32), blocks.max(axis=(1, 3))


def to_uint8(p):
    p = np.asarray(p, dtype=np.float32)
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities contain non-finite values")
    return np.rint(np.clip(p, 0.0, 1.0) * 255.0).astype(np.uint8)


def from_uint8(maps):
    maps = np.asarray(maps)
    if maps.dtype != np.uint8:
        raise ValueError(f"expected uint8 maps, got {maps.dtype}")
    return maps.astype(np.float32) / 255.0


def pack_pathology_maps(probabilities, factor=v2cfg.POOL_FACTOR):
    """(H, W, K) Stage-4 probabilities at the inference size -> uint8 (H/f, W/f, 2K) interleaved
    [c:mean, c:max] per class, the order `channel_names` gives."""
    mean, maximum = block_pool_mean_max(probabilities, factor)
    stacked = np.stack([mean, maximum], axis=-1)            # (h, w, K, 2)
    return to_uint8(stacked.reshape(*stacked.shape[:2], -1))


# --------------------------------------------------------------------------- pathology npz I/O

def write_pathology_npz(path, maps, *, channels, image_id, stage4_sha256, stage3_sha256, gen_id):
    maps = np.asarray(maps)
    channels = tuple(channels)
    classes_from_channels(channels)
    _check_sha(stage4_sha256, "stage4_sha256")
    _check_sha(stage3_sha256, "stage3_sha256")
    assert_not_deny_listed(stage4_sha256)
    if gen_id != stage4_generation_id(stage4_sha256, len(channels) // len(v2cfg.POOLINGS)):
        raise ManifestMismatchError(f"gen_id {gen_id!r} does not match stage4 sha / K")
    expected_shape = (v2cfg.CACHE_SIZE, v2cfg.CACHE_SIZE, len(channels))
    if maps.dtype != np.uint8 or maps.shape != expected_shape:
        raise ValueError(f"maps must be uint8 {expected_shape}, got {maps.dtype} {maps.shape}")
    if os.path.basename(path) != pathology_filename(image_id):
        raise ValueError(f"{path} is not named {pathology_filename(image_id)}")

    def _write(tmp):
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, maps=maps, channels=np.asarray(channels),
                                stage4_sha256=np.asarray(stage4_sha256),
                                stage3_sha256=np.asarray(stage3_sha256), gen_id=np.asarray(gen_id),
                                image_id=np.asarray(str(image_id)),
                                schema_version=np.asarray(v2cfg.CACHE_SCHEMA_VERSION))

    _atomic_write_bytes(path, _write)
    return sha256_file(path)


def read_pathology_npz(path, *, expected_stage4_sha256, expected_stage3_sha256, expected_gen_id,
                       expected_channels, expected_file_sha256=None):
    """Verified read. Every expectation is REQUIRED (no defaults): the caller states which
    generation it wants and gets exactly that or an exception. `expected_file_sha256` (from the
    manifest) additionally pins the bytes. Returns the uint8 maps."""
    assert_not_legacy_path(path)
    assert_not_deny_listed(expected_stage4_sha256, "requested Stage-4 generation")
    if not str(path).endswith(".npz"):
        raise LegacyArtifactError(f"{path} is not a v2 pathology npz")
    if expected_file_sha256 is not None and sha256_file(path) != expected_file_sha256:
        raise ManifestMismatchError(f"{path}: file sha256 differs from the manifest")
    with np.load(path, allow_pickle=False) as data:
        required = {"maps", "channels", "stage4_sha256", "stage3_sha256", "gen_id", "image_id",
                    "schema_version"}
        missing = required - set(data.files)
        if missing:
            raise ManifestMismatchError(f"{path}: missing keys {sorted(missing)}")
        stage4 = str(data["stage4_sha256"])
        assert_not_deny_listed(stage4, f"{path} producer")
        checks = {"stage4_sha256": (stage4, expected_stage4_sha256),
                  "stage3_sha256": (str(data["stage3_sha256"]), expected_stage3_sha256),
                  "gen_id": (str(data["gen_id"]), expected_gen_id),
                  "channels": (tuple(str(c) for c in data["channels"]), tuple(expected_channels)),
                  "schema_version": (int(data["schema_version"]), v2cfg.CACHE_SCHEMA_VERSION)}
        for key, (found, wanted) in checks.items():
            if found != wanted:
                raise ManifestMismatchError(f"{path}: {key} is {found!r}, expected {wanted!r}")
        maps = np.array(data["maps"])
    if maps.dtype != np.uint8 or maps.shape != (v2cfg.CACHE_SIZE, v2cfg.CACHE_SIZE,
                                                 len(expected_channels)):
        raise ManifestMismatchError(f"{path}: maps are {maps.dtype} {maps.shape}")
    return maps


# --------------------------------------------------------------------------- Stage 3 vessel

def read_stage3_vessel(path, *, expected_file_sha256):
    """A v2 Stage-3 vessel map (512, 512, 1) float32 in [0, 1], bytes pinned by the manifest."""
    assert_not_legacy_path(path)
    if sha256_file(path) != expected_file_sha256:
        raise ManifestMismatchError(f"{path}: file sha256 differs from the Stage-3 manifest")
    vessel = np.load(path, allow_pickle=False).astype(np.float32)
    if vessel.ndim == 2:
        vessel = vessel[..., None]
    if vessel.shape != (v2cfg.CACHE_SIZE, v2cfg.CACHE_SIZE, 1):
        raise ManifestMismatchError(f"{path}: vessel map shape {vessel.shape}")
    if not (np.all(np.isfinite(vessel)) and vessel.min() >= 0.0 and vessel.max() <= 1.0):
        raise ManifestMismatchError(f"{path}: vessel values outside [0, 1]")
    return vessel


def read_stage2_rgb(path, *, expected_file_sha256):
    """A v2 Stage-2 canonical RGB frame (512, 512, 3) float32 in [0, 1], bytes pinned by the manifest."""
    assert_not_legacy_path(path)
    if sha256_file(path) != expected_file_sha256:
        raise ManifestMismatchError(f"{path}: file sha256 differs from the Stage-2 manifest")
    rgb = np.load(path, allow_pickle=False).astype(np.float32)
    if rgb.shape != (v2cfg.CACHE_SIZE, v2cfg.CACHE_SIZE, 3):
        raise ManifestMismatchError(f"{path}: RGB shape {rgb.shape}")
    if not (np.all(np.isfinite(rgb)) and rgb.min() >= 0.0 and rgb.max() <= 1.0):
        raise ManifestMismatchError(f"{path}: RGB values outside [0, 1]")
    return rgb


def stage3_parity(cached, recomputed, tol=1e-4):
    """§2: max |cached - recomputed| <= tol for a parity id, else StaleCacheError."""
    delta = float(np.max(np.abs(np.asarray(cached, np.float32).squeeze()
                                - np.asarray(recomputed, np.float32).squeeze())))
    if not delta <= tol:
        raise StaleCacheError(f"Stage-3 parity failed: max |delta| {delta:.3g} > {tol}")
    return delta


def freshness_canary(cached_uint8, recomputed_uint8, tol_counts=2):
    """§11: a recomputed image must match its cached pathology maps within 2/255."""
    a, b = np.asarray(cached_uint8), np.asarray(recomputed_uint8)
    if a.dtype != np.uint8 or b.dtype != np.uint8 or a.shape != b.shape:
        raise StaleCacheError(f"canary shape/dtype mismatch: {a.dtype}{a.shape} vs {b.dtype}{b.shape}")
    delta = int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))
    if delta > tol_counts:
        raise StaleCacheError(f"freshness canary failed: max |delta| {delta}/255 > {tol_counts}/255")
    return delta


# --------------------------------------------------------------------------- manifests

def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _population(ids):
    ids = sorted(str(i) for i in ids)
    if len(set(ids)) != len(ids):
        raise ValueError("population ids are not unique")
    return ids


def stage4_cache_manifest(*, stage4_sha256, stage3_sha256, classes, split_sha256, files,
                          model_manifest_sha256=None):
    """`files`: {image_id: sha256 of its npz}. The population is exactly `files`' keys."""
    _check_sha(stage4_sha256, "stage4_sha256")
    _check_sha(stage3_sha256, "stage3_sha256")
    assert_not_deny_listed(stage4_sha256)
    channels = channel_names(classes)
    return {"kind": "stage4_cache_v2", "schema_version": v2cfg.CACHE_SCHEMA_VERSION,
            "generation_id": stage4_generation_id(stage4_sha256, len(classes)),
            "stage4_sha256": stage4_sha256, "stage3_sha256": stage3_sha256,
            "model_manifest_sha256": model_manifest_sha256,
            "classes": list(classes), "channels": list(channels),
            "pooling": f"exact {v2cfg.POOL_FACTOR}x{v2cfg.POOL_FACTOR} block mean+max from "
                       f"{v2cfg.STAGE4_INFERENCE_SIZE}^2",
            "dtype": "uint8 (round(p*255))", "size": v2cfg.CACHE_SIZE,
            "preproc_version": v2cfg.PREPROC_VERSION, "split_sha256": split_sha256,
            "population": _population(files), "files": {str(k): v for k, v in sorted(files.items())},
            "deny_list": sorted(DENY_LISTED_SHA256), "created_utc": _now()}


def stage3_cache_manifest(*, lwnet_sha256, tta, parity, split_sha256, files):
    _check_sha(lwnet_sha256, "lwnet_sha256")
    if lwnet_sha256 != v2cfg.STAGE3_LWNET_SHA256:
        raise ManifestMismatchError(f"Stage-3 model {lwnet_sha256} is not the pinned LWNet")
    if not parity or not parity.get("passed"):
        raise StaleCacheError("a Stage-3 v2 generation requires a passed parity check")
    return {"kind": "stage3_cache_v2", "schema_version": v2cfg.CACHE_SCHEMA_VERSION,
            "generation_id": v2cfg.STAGE3_GENERATION, "lwnet_sha256": lwnet_sha256,
            "vessel_seg_tta": bool(tta), "parity": parity, "size": v2cfg.CACHE_SIZE,
            "split_sha256": split_sha256, "population": _population(files),
            "files": {str(k): v for k, v in sorted(files.items())}, "created_utc": _now()}


def stage2_cache_manifest(*, split_sha256, files):
    return {"kind": "stage2_rgb_v2", "schema_version": v2cfg.CACHE_SCHEMA_VERSION,
            "generation_id": v2cfg.STAGE2_RGB_GENERATION, "preproc_version": v2cfg.PREPROC_VERSION,
            "frame": f"full native -> direct resize {v2cfg.CACHE_SIZE}^2", "split_sha256": split_sha256,
            "population": _population(files), "files": {str(k): v for k, v in sorted(files.items())},
            "created_utc": _now()}


def bundle_manifest(stage2, stage3, stage4):
    """Binds the three generations; the Stage-3 lineage lives here (Stage 4 is RGB-only)."""
    if stage2["kind"] != "stage2_rgb_v2" or stage3["kind"] != "stage3_cache_v2" \
            or stage4["kind"] != "stage4_cache_v2":
        raise ManifestMismatchError("bundle needs a Stage-2, a Stage-3 and a Stage-4 v2 manifest")
    if stage4["stage3_sha256"] != stage3["lwnet_sha256"]:
        raise ManifestMismatchError("Stage-4 cache records a different Stage-3 sha than the Stage-3 cache")
    splits = {stage2["split_sha256"], stage3["split_sha256"], stage4["split_sha256"]}
    if len(splits) != 1:
        raise ManifestMismatchError(f"generations disagree on the split: {sorted(splits)}")
    population = sorted(set(stage2["population"]) & set(stage3["population"])
                        & set(stage4["population"]))
    return {"kind": "bundle_v2", "schema_version": v2cfg.CACHE_SCHEMA_VERSION,
            "bundle_id": bundle_id(stage2["generation_id"], stage3["generation_id"],
                                   stage4["generation_id"]),
            "stage2_generation": stage2["generation_id"], "stage3_generation": stage3["generation_id"],
            "stage4_generation": stage4["generation_id"], "stage3_sha256": stage3["lwnet_sha256"],
            "stage4_sha256": stage4["stage4_sha256"], "channels": stage4["channels"],
            "split_sha256": splits.pop(), "population": population, "created_utc": _now()}


def write_manifest(path, manifest):
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")

    def _write(tmp):
        with open(tmp, "wb") as fh:
            fh.write(payload)

    _atomic_write_bytes(path, _write)
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path, *, expected_kind, expected_generation_id=None):
    assert_not_legacy_path(path)
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("kind") != expected_kind:
        raise ManifestMismatchError(f"{path}: kind {manifest.get('kind')!r} != {expected_kind!r}")
    if manifest.get("schema_version") != v2cfg.CACHE_SCHEMA_VERSION:
        raise ManifestMismatchError(f"{path}: schema version {manifest.get('schema_version')}")
    gen_key = "bundle_id" if expected_kind == "bundle_v2" else "generation_id"
    if expected_generation_id is not None and manifest.get(gen_key) != expected_generation_id:
        raise ManifestMismatchError(f"{path}: {gen_key} {manifest.get(gen_key)!r} != "
                                    f"{expected_generation_id!r}")
    for sha in (manifest.get("stage4_sha256"),):
        if sha:
            assert_not_deny_listed(sha, f"{path} Stage-4")
    return manifest


def assert_complete(manifest, required_ids, directory=None, filename=None):
    """Every required id is in the manifest (and, when `directory` is given, on disk)."""
    listed = set(manifest["population"])
    missing = sorted(set(str(i) for i in required_ids) - listed)
    if directory is not None:
        on_disk_missing = [i for i in sorted(listed & set(str(r) for r in required_ids))
                           if not os.path.exists(os.path.join(directory, filename(i)))]
        missing += on_disk_missing
    if missing:
        raise IncompleteCacheError(f"{manifest.get('generation_id', manifest.get('bundle_id'))}: "
                                   f"{len(missing)} required ids missing, e.g. {missing[:5]}")
    return True


# --------------------------------------------------------------------------- C2 pyramid features

PYRAMID_GRIDS = (1, 2, 4)


def pyramid_feature_names(channels, grids=PYRAMID_GRIDS):
    return [f"{c}|g{g}|r{r}c{col}|{stat}" for g in grids for r in range(g) for col in range(g)
            for c in channels for stat in ("mean", "max")]


def pyramid_features(maps, grids=PYRAMID_GRIDS):
    """§10 Q-pyr / V-pyr: mean and max of every channel over each cell of 1x1 + 2x2 + 4x4 grids.
    `maps` (H, W, C) uint8 (converted to [0, 1]) or float. Returns C * 2 * 21 float32 values in the
    order of `pyramid_feature_names`."""
    x = from_uint8(maps) if np.asarray(maps).dtype == np.uint8 else np.asarray(maps, np.float32)
    h, w, c = x.shape
    out = []
    for g in grids:
        if h % g or w % g:
            raise ValueError(f"{x.shape} not divisible by grid {g}")
        cells = x.reshape(g, h // g, g, w // g, c)
        mean = cells.mean(axis=(1, 3), dtype=np.float64)          # (g, g, c)
        maximum = cells.max(axis=(1, 3))
        out.append(np.stack([mean, maximum], axis=-1).reshape(-1))   # r, col, c, stat
    return np.concatenate(out).astype(np.float32)


# --------------------------------------------------------------------------- legacy fingerprint

def legacy_fingerprint(paths=v2cfg.LEGACY_DIRS):
    """(relative path, size, mtime) digest of every legacy location -- recorded before and after a
    v2 job to prove no legacy cache/model was written."""
    import pl_convnext
    return pl_convnext.directory_fingerprint(paths)
