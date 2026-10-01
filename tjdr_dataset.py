"""TJDR access for Stage-4 v2 training (research record §42 R1-R3, §43).

  * R1: the pinned exclusions (`pipeline_v2_config.TJDR_EXCLUDED`) are applied by `usable_ids` and
    enforced by `load_pair`. An excluded id cannot be loaded for training, validation or evaluation.
  * R2: masks are palette-indexed PNGs, read with `np.asarray(Image.open(p))` and never
    `.convert()`-ed. The reader requires mode "P" and values in {0..4}. `split_index_mask` maps codes
    explicitly (MA 3, HE 2, EX 1, SE 4) into K binary masks BEFORE any resize.
    `stage4_v2.resize_mask_full_frame` refuses non-binary input.
  * R3: `verify_tjdr` re-checks the whole dataset; `DatasetSpec("TJDR").verified` rests on it.
"""
import os

import numpy as np

import config
import pipeline_v2_config as v2cfg

SPLITS = ("train", "test")


class TJDRDataError(ValueError):
    """A TJDR file or id violates the verified dataset contract."""


def raw_root():
    return config.dataset_raw_dir(v2cfg.TJDR_DATASET_NAME)


def image_path(split, image_id, root=None):
    return os.path.join(root or raw_root(), split, "image", f"{image_id}.png")


def mask_path(split, image_id, root=None):
    return os.path.join(root or raw_root(), split, "annotation", f"{image_id}.png")


def official_ids(split, root=None):
    if split not in SPLITS:
        raise TJDRDataError(f"unknown TJDR split {split!r}")
    folder = os.path.join(root or raw_root(), split, "image")
    return sorted(os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(".png"))


def usable_ids(split, root=None, check_counts=True):
    """The official ids minus the pinned exclusions; the only ids any Stage-4 split may use."""
    ids = official_ids(split, root)
    excluded = set(v2cfg.TJDR_EXCLUDED[split])
    missing = excluded - set(ids)
    if missing:
        raise TJDRDataError(f"excluded ids not present in {split}: {sorted(missing)}")
    kept = [i for i in ids if i not in excluded]
    if check_counts and (len(ids), len(kept)) != (v2cfg.TJDR_OFFICIAL_COUNTS[split], v2cfg.TJDR_USABLE_COUNTS[split]):
        raise TJDRDataError(f"{split}: {len(ids)} official / {len(kept)} usable, expected "
                            f"{v2cfg.TJDR_OFFICIAL_COUNTS[split]} / {v2cfg.TJDR_USABLE_COUNTS[split]}")
    return kept


def assert_usable(split, image_id):
    if image_id in v2cfg.TJDR_EXCLUDED.get(split, ()):
        raise TJDRDataError(f"{image_id} is excluded ({v2cfg.TJDR_EXCLUSION_REASONS[image_id]})")
    if not str(image_id).startswith(f"TJDR_{split}_"):
        raise TJDRDataError(f"{image_id} does not belong to the TJDR {split} split")


def read_index_mask(path):
    """The raw palette-index mask (H, W) uint8. No `.convert()`: that would turn palette indices
    into luminance values."""
    from PIL import Image
    with Image.open(path) as im:
        if im.mode != v2cfg.TJDR_MASK_MODE:
            raise TJDRDataError(f"{path}: mask mode {im.mode!r}, expected {v2cfg.TJDR_MASK_MODE!r}")
        mask = np.asarray(im)
    validate_index_mask(mask, path)
    return mask


def validate_index_mask(mask, where="mask"):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise TJDRDataError(f"{where}: expected a 2-D uint8 index mask, got {mask.dtype} {mask.shape}")
    bad = np.setdiff1d(np.unique(mask), v2cfg.TJDR_MASK_VALUES)
    if bad.size:
        raise TJDRDataError(f"{where}: values {bad.tolist()} outside {v2cfg.TJDR_MASK_VALUES}")
    return mask


def split_index_mask(index_mask, classes=v2cfg.STAGE4_V2A_CLASSES):
    """(H, W) index mask -> (K, H, W) uint8 {0, 1}, one channel per class in `classes` order, via
    the explicit code map. Must happen BEFORE any resize."""
    mask = validate_index_mask(index_mask)
    unknown = [c for c in classes if c not in v2cfg.TJDR_MASK_CODES]
    if unknown:
        raise TJDRDataError(f"TJDR does not annotate {unknown}")
    return np.stack([(mask == v2cfg.TJDR_MASK_CODES[c]).astype(np.uint8) for c in classes])


def resized_class_masks(index_mask, classes=v2cfg.STAGE4_V2A_CLASSES, size=v2cfg.STAGE4_INFERENCE_SIZE):
    """Per-class binary masks resized with the unchanged Stage-4 rule (threshold 0.5)."""
    import stage4_v2
    return np.stack([stage4_v2.resize_mask_full_frame(m, size) for m in split_index_mask(index_mask, classes)])


def processed_root():
    return config.dataset_processed_dir(v2cfg.TJDR_DATASET_NAME)


def processed_image_path(split, image_id, root=None):
    """Stage-2 DR output (image_preprocessing.preprocess_image, lossless PNG) of one TJDR image."""
    return os.path.join(root or processed_root(), split, "image", f"{image_id}.png")


def load_pair(split, image_id, root=None, classes=v2cfg.STAGE4_V2A_CLASSES, stage2=False, processed=None):
    """(RGB uint8 (H, W, 3), class masks uint8 (K, H, W)) at native resolution, for a usable id only.
    `stage2=True` reads the Stage-2 DR output from `processed/`. There is no fallback to raw: a missing
    processed file raises."""
    from PIL import Image
    assert_usable(split, image_id)
    path = processed_image_path(split, image_id, processed) if stage2 else image_path(split, image_id, root)
    if stage2 and not os.path.exists(path):
        raise TJDRDataError(f"Stage-2 output missing: {path} (run the Stage-4 cache notebook's Stage-2 step)")
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    masks = split_index_mask(read_index_mask(mask_path(split, image_id, root)), classes)
    if masks.shape[1:] != rgb.shape[:2]:
        raise TJDRDataError(f"{image_id}: mask {masks.shape[1:]} vs image {rgb.shape[:2]}")
    return rgb, masks


def verify_tjdr(root=None):
    """Full check behind `DatasetSpec("TJDR").verified`: counts, exclusions, pairing, mask mode and
    values, image/mask sizes, and per-class image counts (official + usable). Returns a report;
    raises TJDRDataError on any failure."""
    from PIL import Image
    report = {"root": root or raw_root(), "splits": {}}
    for split in SPLITS:
        ids = official_ids(split, root)
        ann = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(root or raw_root(), split, "annotation")))
        if ids != ann:
            raise TJDRDataError(f"{split}: image/annotation ids differ")
        usable = set(usable_ids(split, root))
        counts = {"official": {c: 0 for c in v2cfg.TJDR_MASK_CODES}, "usable": {c: 0 for c in v2cfg.TJDR_MASK_CODES}}
        sizes = {}
        for image_id in ids:
            mask = read_index_mask(mask_path(split, image_id, root))
            with Image.open(image_path(split, image_id, root)) as im:
                if im.size[::-1] != mask.shape:
                    raise TJDRDataError(f"{image_id}: image {im.size[::-1]} vs mask {mask.shape}")
                sizes[im.size[0]] = sizes.get(im.size[0], 0) + 1
            present = set(np.unique(mask).tolist())
            for c, code in v2cfg.TJDR_MASK_CODES.items():
                if code in present:
                    counts["official"][c] += 1
                    counts["usable"][c] += image_id in usable
        if counts["official"] != v2cfg.TJDR_CLASS_IMAGE_COUNTS_OFFICIAL[split]:
            raise TJDRDataError(f"{split}: class image counts {counts['official']} differ from §42")
        report["splits"][split] = {"official": len(ids), "usable": len(usable),
                                   "excluded": list(v2cfg.TJDR_EXCLUDED[split]),
                                   "class_image_counts": counts, "image_sizes": sizes}
    report["passed"] = True
    return report
