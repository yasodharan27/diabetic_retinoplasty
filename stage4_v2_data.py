"""Stage-4 v2 training data (research record §40, §42-§44).

  * IDRiD v2 loader: the 54 lesion-segmentation TRAINING images only, with a pinned 44/10 split
    stratified by SE presence (`pipeline_v2_config.IDRID_V2_*`). The 27 test images are refused here;
    they are reachable only through `stage4_v2_gate`. The grading set is never read.
  * Stage 2: the project's canonical DR preprocessing (`image_preprocessing.preprocess_image`, profile
    "DR"). IDRiD uses its existing Drive `processed/` output; TJDR's is created once into
    `datasets/TJDR/processed` by the cache notebook. The geometry is untouched (no crop).
  * Training-input cache `s4train-v1`: Stage-2 image directly resized to 1536^2 (uint8) plus the 4
    binary class masks at 1536 (current rule, threshold 0.5). Roles are train and val only. It is
    built once and read by `PatchDataset`. It is not a Stage-4 prediction cache.
  * Dataset-balanced batches: every batch holds batch_size/2 patches from IDRiD and from TJDR.
"""
import hashlib
import json
import os
import re

import numpy as np

import config
import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache
import stage4_v2 as s4
import tjdr_dataset as tj

CLASSES = v2cfg.STAGE4_V2A_CLASSES
IDRID_MASK_DIRS = {"MA": ("1. Microaneurysms", "MA"), "HE": ("2. Haemorrhages", "HE"),
                   "EX": ("3. Hard Exudates", "EX"), "SE": ("4. Soft Exudates", "SE")}
_IDRID_SEG_ID = re.compile(r"^IDRiD_\d{2}$")           # grading ids are IDRiD_001..516 (3 digits)


class Stage4DataError(ValueError):
    """A Stage-4 v2 data contract was violated (split, role, test isolation, cache lineage)."""


# --------------------------------------------------------------------------- Stage 2

def stage2_rgb(raw_path):
    """Canonical Stage-2 DR preprocessing of one image, exactly as `preprocess_image` writes it
    (cv2 BGR in -> preprocess_array(profile="DR") -> BGR out), returned as RGB uint8, native size."""
    import cv2
    from image_preprocessing import preprocess_array
    bgr = cv2.imread(raw_path)
    if bgr is None:
        raise Stage4DataError(f"cannot read {raw_path}")
    return cv2.cvtColor(preprocess_array(bgr, profile="DR"), cv2.COLOR_BGR2RGB)


def read_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def stage2_parity(raw_path, processed_path):
    """Recompute Stage 2 from raw and compare with a stored processed file. Returns max |diff| and PSNR
    (IDRiD's processed files are JPEG, so they are not bit-exact; TJDR's are PNG, so they are exact)."""
    a = stage2_rgb(raw_path).astype(np.float64)
    b = read_rgb(processed_path).astype(np.float64)
    if a.shape != b.shape:
        raise Stage4DataError(f"Stage-2 shape mismatch {a.shape} vs {b.shape} for {processed_path}")
    mse = float(np.mean((a - b) ** 2))
    return {"max_abs": float(np.abs(a - b).max()), "psnr": float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)}


# --------------------------------------------------------------------------- IDRiD v2

def idrid_raw_dir():
    return config.dataset_raw_dir("IDRiD/segmentation")


def idrid_processed_dir():
    return config.dataset_processed_dir("IDRiD/segmentation")


def _split_payload(train, val):
    return json.dumps({"train": list(train), "val": list(val)}, sort_keys=True, separators=(",", ":"))


def idrid_split_sha256(train, val):
    return hashlib.sha256(_split_payload(train, val).encode("utf-8")).hexdigest()


def compute_idrid_split(se_positive_ids, all_ids=v2cfg.IDRID_SEG_TRAIN_IDS, seed=v2cfg.IDRID_V2_SPLIT_SEED, n_val=10):
    """Deterministic SE-stratified draw (the rule that produced the pinned split)."""
    ids = sorted(all_ids)
    pos = [i for i in ids if i in set(se_positive_ids)]
    neg = [i for i in ids if i not in set(se_positive_ids)]
    n_pos = round(n_val * len(pos) / len(ids))
    rng = np.random.default_rng(seed)
    val = sorted([str(x) for x in rng.choice(pos, n_pos, replace=False)] +
                 [str(x) for x in rng.choice(neg, n_val - n_pos, replace=False)])
    return tuple(i for i in ids if i not in val), tuple(val)


def idrid_split():
    """The pinned IDRiD v2 split: (train 44, val 10). Asserts its sha and its disjointness from the
    test set on every call."""
    val = tuple(v2cfg.IDRID_V2_VAL_IDS)
    train = tuple(i for i in v2cfg.IDRID_SEG_TRAIN_IDS if i not in val)
    if (len(train), len(val)) != (44, 10) or set(val) - set(v2cfg.IDRID_SEG_TRAIN_IDS):
        raise Stage4DataError("IDRiD v2 split is not 44/10 within the 54 training images")
    if set(train + val) & set(v2cfg.IDRID_SEG_TEST_IDS):
        raise Stage4DataError("IDRiD test ids leaked into the v2 split")
    if idrid_split_sha256(train, val) != v2cfg.IDRID_V2_SPLIT_SHA256:
        raise Stage4DataError("IDRiD v2 split sha differs from the pinned value")
    return {"train": train, "val": val}


def idrid_se_positive_ids(raw_dir=None):
    folder = os.path.join(raw_dir or idrid_raw_dir(), "2. All Segmentation Groundtruths", "a. Training Set",
                          IDRID_MASK_DIRS["SE"][0])
    return sorted(f[:-len("_SE.tif")] for f in os.listdir(folder) if f.endswith("_SE.tif"))


def assert_idrid_training_id(image_id):
    """Only the 54 lesion-segmentation training images may enter Stage-4 training / validation."""
    if not _IDRID_SEG_ID.match(str(image_id)):
        raise Stage4DataError(f"{image_id} is not an IDRiD lesion-segmentation id (grading ids are refused)")
    if image_id in v2cfg.IDRID_SEG_TEST_IDS:
        raise Stage4DataError(f"{image_id} is an IDRiD TEST image; only stage4_v2_gate may read it")
    if image_id not in v2cfg.IDRID_SEG_TRAIN_IDS:
        raise Stage4DataError(f"{image_id} is not one of the 54 IDRiD training images")


def _assert_segmentation_path(path):
    parts = str(path).replace("\\", "/").lower()
    if "/grading/" in parts or "disease grading" in parts:
        raise Stage4DataError(f"{path}: the IDRiD grading set never enters Stage 4")
    return path


def read_idrid_masks(image_id, split_dir, raw_dir=None, classes=CLASSES):
    """(K, H, W) uint8 {0,1}; a missing mask file means the lesion is absent (IDRiD annotates all four)."""
    from PIL import Image
    root = _assert_segmentation_path(raw_dir or idrid_raw_dir())
    masks, shape = [], None
    for c in classes:
        folder, suffix = IDRID_MASK_DIRS[c]
        path = os.path.join(root, "2. All Segmentation Groundtruths", split_dir, folder, f"{image_id}_{suffix}.tif")
        if os.path.exists(path):
            with Image.open(path) as im:
                m = np.asarray(im)
            if m.ndim == 3:
                m = m[..., 0]
            values = set(np.unique(m).tolist())
            if not (values <= {0, 1} or values <= {0, 255}):
                raise Stage4DataError(f"{path}: unexpected mask values {sorted(values)[:6]}")
            masks.append((m > 0).astype(np.uint8))
            shape = m.shape
        else:
            masks.append(None)
    if shape is None:
        raise Stage4DataError(f"{image_id}: no lesion mask files at all")
    return np.stack([m if m is not None else np.zeros(shape, np.uint8) for m in masks])


def load_idrid_pair(image_id, raw_dir=None, processed_dir=None, classes=CLASSES):
    """Stage-2 RGB (Drive processed/ JPEG) + native (K, H, W) masks for one of the 54 training images."""
    assert_idrid_training_id(image_id)
    proc = _assert_segmentation_path(processed_dir or idrid_processed_dir())
    path = os.path.join(proc, "1. Original Images", "a. Training Set", f"{image_id}.jpg")
    if not os.path.exists(path):
        raise Stage4DataError(f"Stage-2 output missing: {path}")
    rgb = read_rgb(path)
    masks = read_idrid_masks(image_id, "a. Training Set", raw_dir, classes)
    if masks.shape[1:] != rgb.shape[:2]:
        raise Stage4DataError(f"{image_id}: mask {masks.shape[1:]} vs image {rgb.shape[:2]}")
    return rgb, masks


# --------------------------------------------------------------------------- source entries

def source_entries():
    """Every (dataset, role, split, id, camera) that the training cache must contain: IDRiD 44/10,
    TJDR 443 (train) / 110 (official test, used as validation). Test images never appear."""
    split = idrid_split()
    entries = [("IDRiD", role, "a. Training Set", i, "IDRiD-Kowa") for role in ("train", "val") for i in split[role]]
    for tj_split, role in (("train", "train"), ("test", "val")):
        entries += [("TJDR", role, tj_split, i, None) for i in tj.usable_ids(tj_split)]
    return entries


def tjdr_camera(width):
    return {2048: "TJDR-TRC50DX", 3912: "TJDR-CLARUS500"}.get(int(width), f"TJDR-{width}")


# --------------------------------------------------------------------------- training-input cache

def cache_dir(root=v2cfg.STAGE4_TRAIN_CACHE_ROOT, generation=v2cfg.STAGE4_TRAIN_CACHE_GENERATION):
    return os.path.join(root, generation)


def sample_filename(dataset, role, image_id):
    return f"{dataset}__{role}__{image_id}.npz"


def make_cache_sample(rgb_native, masks_native, size=v2cfg.STAGE4_INFERENCE_SIZE):
    """Stage-2 RGB (native) -> 1536^2 uint8 via stage4_v2.resize_full_frame; each binary class mask
    -> stage4_v2.resize_mask_full_frame (unchanged rule, threshold 0.5)."""
    rgb = np.rint(s4.resize_full_frame(rgb_native, size) * 255.0).astype(np.uint8)
    masks = np.stack([s4.resize_mask_full_frame(m, size) for m in masks_native])
    return rgb, masks


def write_cache_sample(directory, entry, rgb, masks):
    dataset, role, split, image_id, camera = entry
    if role not in v2cfg.STAGE4_TRAIN_ROLES:
        raise Stage4DataError(f"role {role!r} is not allowed in the training cache")
    if dataset == "IDRiD":
        assert_idrid_training_id(image_id)
    elif dataset == "TJDR":
        tj.assert_usable(split, image_id)
    else:
        raise Stage4DataError(f"unknown dataset {dataset}")
    path = os.path.join(directory, sample_filename(dataset, role, image_id))
    cache.assert_not_legacy_path(path)
    meta = {"dataset": dataset, "role": role, "split": split, "image_id": image_id, "camera": camera,
            "classes": list(CLASSES), "annotated": [True] * len(CLASSES)}
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, rgb=rgb, masks=masks, meta=np.asarray(json.dumps(meta, sort_keys=True)))
    os.replace(tmp, path)
    return path, cache.sha256_file(path), meta


def build_training_cache(directory, entries, loader, log=print):
    """Resumable: files already listed with a matching sha in `progress.json` are skipped. `loader(entry)`
    returns (Stage-2 RGB native, masks native). Writes `manifest.json` at the end."""
    progress_path = os.path.join(directory, "progress.json")
    cache.assert_not_legacy_path(directory)
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as fh:
            done = json.load(fh)
    else:
        done = {}
    for n, entry in enumerate(entries):
        name = sample_filename(entry[0], entry[1], entry[3])
        path = os.path.join(directory, name)
        if name in done and os.path.exists(path) and cache.sha256_file(path) == done[name]["sha256"]:
            continue
        rgb_n, masks_n = loader(entry)
        camera = entry[4] or tjdr_camera(rgb_n.shape[1])
        entry = (*entry[:4], camera)
        rgb, masks = make_cache_sample(rgb_n, masks_n)
        _, sha, meta = write_cache_sample(directory, entry, rgb, masks)
        done[name] = {"sha256": sha, **meta, "native_shape": list(rgb_n.shape[:2]),
                      "positive_pixels_1536": masks.reshape(len(CLASSES), -1).sum(axis=1).tolist()}
        if n % 25 == 0:
            with open(progress_path, "w") as fh:
                json.dump(done, fh)
            log(f"  cache {n + 1}/{len(entries)}")
    with open(progress_path, "w") as fh:
        json.dump(done, fh)
    return write_training_manifest(directory, done, entries)


def write_training_manifest(directory, done, entries):
    names = {sample_filename(e[0], e[1], e[3]) for e in entries}
    files = {k: v for k, v in done.items() if k in names}
    counts = {}
    for v in files.values():
        counts[f"{v['dataset']}/{v['role']}"] = counts.get(f"{v['dataset']}/{v['role']}", 0) + 1
    expected = {f"{d}/{r}": n for (d, r), n in v2cfg.STAGE4_TRAIN_EXPECTED.items()}
    manifest = {"kind": "stage4_train_cache", "generation": v2cfg.STAGE4_TRAIN_CACHE_GENERATION,
                "classes": list(CLASSES), "size": v2cfg.STAGE4_INFERENCE_SIZE,
                "preproc": "Stage-2 DR profile (image_preprocessing.preprocess_image) -> "
                           "stage4_v2.resize_full_frame -> uint8; masks: stage4_v2.resize_mask_full_frame(0.5)",
                "idrid_split_sha256": v2cfg.IDRID_V2_SPLIT_SHA256,
                "tjdr_excluded": {k: list(v) for k, v in v2cfg.TJDR_EXCLUDED.items()},
                "tjdr_source_listing_sha256": v2cfg.TJDR_SOURCE_LISTING_SHA256,
                "counts": counts, "complete": counts == expected, "files": files}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(
        {k: v["sha256"] for k, v in sorted(files.items())}, sort_keys=True).encode()).hexdigest()
    cache.write_manifest(os.path.join(directory, "manifest.json"), {**manifest, "schema_version": 1})
    return manifest


class TrainingCache:
    """Read side. Verifies the manifest (generation, complete counts, split sha, no test ids) and, with
    verify_files=True, every file's sha once at start-up."""

    def __init__(self, directory, verify_files=True, require_complete=True):
        self.directory = directory
        cache.assert_not_legacy_path(directory)
        with open(os.path.join(directory, "manifest.json")) as fh:
            m = json.load(fh)
        if m.get("kind") != "stage4_train_cache" or m.get("generation") != v2cfg.STAGE4_TRAIN_CACHE_GENERATION:
            raise Stage4DataError(f"{directory}: not a {v2cfg.STAGE4_TRAIN_CACHE_GENERATION} training cache")
        if m["idrid_split_sha256"] != v2cfg.IDRID_V2_SPLIT_SHA256:
            raise Stage4DataError("training cache was built with a different IDRiD split")
        if require_complete and not m.get("complete"):
            raise Stage4DataError(f"training cache incomplete: {m.get('counts')}")
        for name, v in m["files"].items():
            if v["role"] not in v2cfg.STAGE4_TRAIN_ROLES:
                raise Stage4DataError(f"{name}: forbidden role {v['role']}")
            if v["dataset"] == "IDRiD":
                assert_idrid_training_id(v["image_id"])
            else:
                tj.assert_usable(v["split"], v["image_id"])
            if verify_files and cache.sha256_file(os.path.join(directory, name)) != v["sha256"]:
                raise Stage4DataError(f"{name}: sha differs from the manifest")
        self.manifest = m
        self.fingerprint = m["fingerprint"]

    def names(self, role, dataset=None):
        return sorted(k for k, v in self.manifest["files"].items()
                      if v["role"] == role and (dataset is None or v["dataset"] == dataset))

    def load(self, name):
        with np.load(os.path.join(self.directory, name), allow_pickle=False) as d:
            return d["rgb"], d["masks"], json.loads(str(d["meta"]))


# --------------------------------------------------------------------------- sampling

def balanced_batch_plan(names_by_dataset, batch_size, step, seed):
    """The batch for `step`: batch_size/len(datasets) images drawn uniformly (with replacement) from each
    dataset, plus a per-item seed. Counter-based, so any step can be reproduced (e.g. after a resume)."""
    datasets = sorted(names_by_dataset)
    if batch_size % len(datasets):
        raise Stage4DataError(f"batch_size {batch_size} is not divisible by {len(datasets)} datasets")
    rng = np.random.default_rng([int(seed), int(step)])
    plan = []
    for d in datasets:
        names = names_by_dataset[d]
        for k in rng.integers(0, len(names), batch_size // len(datasets)):
            plan.append((names[int(k)], int(rng.integers(0, 2 ** 31 - 1))))
    return plan


class PatchDataset:
    """Map-style dataset keyed by (cache file name, item seed): the Stage-2 1536 image and masks are read
    from the cache, a class-aware patch is drawn (stage4_v2.sample_patch_origin, p_lesion), flips are
    applied (label-preserving), and the ImageNet-normalised tensors are returned."""

    def __init__(self, training_cache, patch=v2cfg.STAGE4_TRAIN_PATCH, p_lesion=0.5, flips=True):
        self.cache, self.patch, self.p_lesion, self.flips = training_cache, patch, p_lesion, flips

    def __getitem__(self, key):
        import torch
        name, item_seed = key
        rgb, masks, meta = self.cache.load(name)
        rng = np.random.default_rng(int(item_seed))
        annotated = np.asarray(meta["annotated"], dtype=bool)
        y0, x0, _ = s4.sample_patch_origin(rng, masks, annotated, self.patch, self.p_lesion)
        p = self.patch
        img = rgb[y0:y0 + p, x0:x0 + p].astype(np.float32) / 255.0
        m = masks[:, y0:y0 + p, x0:x0 + p]
        if self.flips:
            if rng.random() < 0.5:
                img, m = img[:, ::-1], m[:, :, ::-1]
            if rng.random() < 0.5:
                img, m = img[::-1], m[:, ::-1]
        fov = s4.fov_mask(img)
        return {"x": s4.to_model_input(np.ascontiguousarray(img))[0],
                "y": torch.from_numpy(np.ascontiguousarray(m).astype(np.float32)),
                "fov": torch.from_numpy(fov[None].astype(np.float32)),
                "annotated": torch.from_numpy(annotated),
                "dataset": meta["dataset"]}


def collate(items):
    import torch
    return {"x": torch.stack([i["x"] for i in items]), "y": torch.stack([i["y"] for i in items]),
            "fov": torch.stack([i["fov"] for i in items]),
            "annotated": torch.stack([i["annotated"] for i in items]),
            "dataset": [i["dataset"] for i in items]}
