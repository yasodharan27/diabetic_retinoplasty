"""
Reusable dataset verification for the Colab training workflow.

`verify_eyeq_dataset()` is EyeQ-specific (Stage 1): it reuses
`image_quality_dataset`'s own loading/decoding functions (`_read_labels`,
`_decode_image`) instead of reimplementing file-existence or JPEG-decoding
logic, so a pass here means "the exact code path
`image_quality_dataset.load_eyeq_datasets` will use also works," not just
"these files happen to exist."

`verify_image_folder()` is generic and dataset-agnostic (Stage 2 and beyond):
unlike EyeQ, most datasets Stage 2 preprocesses (APTOS2019, IDRiD, ...) are
plain image folders with no `labels.csv` / quality-class structure to check
-- it verifies only what's true of any such folder (exists, non-empty,
a corruption spot check), and is reused for both a raw input folder before
preprocessing and a processed output folder after.

Every `verify_*` function raises `RuntimeError` with a clear message on
failure and returns a structured report once every check passes.
"""

import os
import random
from dataclasses import dataclass, field

import cv2
import pandas as pd

import image_quality_dataset as iqd

SAMPLE_SIZE = 50
CORRUPTION_ABORT_FRACTION = 0.20
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class SplitReport:
    split: str
    csv_rows: int
    usable_rows: int
    missing_images: int
    corruption_sampled: int
    corruption_failures: int
    class_distribution: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EyeQVerificationReport:
    raw_dir: str
    train: SplitReport
    test: SplitReport
    expected_train_split_count: int
    expected_val_split_count: int


def _verify_split_structure(raw_dir, split):
    csv_path = os.path.join(raw_dir, split, "labels.csv")
    images_dir = os.path.join(raw_dir, split, "images")

    if not os.path.isfile(csv_path):
        raise RuntimeError(f"Dataset verification failed: missing {csv_path}.")
    if not os.path.isdir(images_dir):
        raise RuntimeError(f"Dataset verification failed: missing {images_dir}.")
    return csv_path, images_dir


def _verify_split_images(raw_dir, split, images_dir):
    """Missing-image count reuses `image_quality_dataset._read_labels`,
    which already drops (and reports) rows whose image file is missing."""
    raw_df = pd.read_csv(os.path.join(raw_dir, split, "labels.csv"))
    verified_df = iqd._read_labels(raw_dir, split)
    missing = len(raw_df) - len(verified_df)

    if len(verified_df) == 0:
        raise RuntimeError(
            f"Dataset verification failed: split '{split}' has 0 usable images "
            f"after checking for missing files under {images_dir}."
        )
    return raw_df, verified_df, missing


def _spot_check_corruption(raw_dir, split, sample_size=SAMPLE_SIZE):
    """Lightweight spot check (fixed random sample), not a full-dataset
    decode -- a full scan would cost minutes-to-tens-of-minutes for no real
    benefit. Reuses the exact JPEG decode path the training pipeline uses."""
    df = iqd._read_labels(raw_dir, split)
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    failures = []
    for _, row in sample.iterrows():
        try:
            iqd._decode_image(row["path"], (224, 224)).numpy()
        except Exception as exc:  # noqa: BLE001 -- want to catch and report any decode failure
            failures.append((row["path"], str(exc)))

    if len(sample) > 0 and len(failures) / len(sample) > CORRUPTION_ABORT_FRACTION:
        raise RuntimeError(
            f"Dataset verification failed: {len(failures)}/{len(sample)} sampled images in "
            f"split '{split}' failed to decode ({len(failures) / len(sample):.0%} > "
            f"{CORRUPTION_ABORT_FRACTION:.0%} threshold). The dataset may be corrupted or "
            "incompletely uploaded to Drive."
        )
    return len(sample), failures


def verify_eyeq_dataset(raw_dir, val_split=0.15, sample_size=SAMPLE_SIZE):
    """Verify the EyeQ dataset at `raw_dir` (directory, labels.csv, image
    counts, missing images, a corruption spot check, and class
    distribution) for both the `train` and `test` splits. Aborts with a
    `RuntimeError` on the first failed check; returns a structured
    `EyeQVerificationReport` once everything passes."""
    if not os.path.isdir(raw_dir):
        raise RuntimeError(
            f"EyeQ raw dataset directory not found at {raw_dir}. Verify Drive is mounted "
            "and DRIVE_PROJECT_ROOT / EYEQ_RAW_DIR resolve to your actual Drive layout."
        )

    split_reports = {}
    for split in ("train", "test"):
        _, images_dir = _verify_split_structure(raw_dir, split)
        raw_df, verified_df, missing = _verify_split_images(raw_dir, split, images_dir)
        sampled, failures = _spot_check_corruption(raw_dir, split, sample_size=sample_size)

        class_distribution = {
            iqd.QUALITY_CLASSES[i]: int((verified_df["quality"] == i).sum())
            for i in range(len(iqd.QUALITY_CLASSES))
        }

        split_reports[split] = SplitReport(
            split=split,
            csv_rows=len(raw_df),
            usable_rows=len(verified_df),
            missing_images=missing,
            corruption_sampled=sampled,
            corruption_failures=len(failures),
            class_distribution=class_distribution,
        )
        print(
            f"[{split}] csv rows: {len(raw_df)} | usable: {len(verified_df)} | "
            f"missing images: {missing} | corrupted (of {sampled} sampled): {len(failures)}"
        )

    expected_val = round(split_reports["train"].usable_rows * val_split)
    expected_train = split_reports["train"].usable_rows - expected_val
    print(
        f"\nExpected stratified split of the train pool (val_split={val_split}): "
        f"~{expected_train} train / ~{expected_val} val"
    )
    print(f"Held-out test split (never used in training): {split_reports['test'].usable_rows} images")
    print("\nDataset verification passed.")

    return EyeQVerificationReport(
        raw_dir=raw_dir,
        train=split_reports["train"],
        test=split_reports["test"],
        expected_train_split_count=expected_train,
        expected_val_split_count=expected_val,
    )


@dataclass(frozen=True)
class ImageFolderReport:
    path: str
    image_count: int
    corruption_sampled: int
    corruption_failures: int


def verify_image_folder(path, min_images=1, sample_size=SAMPLE_SIZE):
    """Generic, dataset-structure-agnostic verification for a flat folder of
    images -- used by Stage 2 (Image Preprocessing) and any future stage
    that consumes a plain image folder (APTOS2019, IDRiD, DRIVE, CHASE_DB1,
    ...), none of which have EyeQ's `labels.csv` / quality-class structure
    for `verify_eyeq_dataset()` to check.

    Checks, in order: the directory exists; it contains at least
    `min_images` files with a recognized image extension (matching
    `image_preprocessing.IMAGE_EXTENSIONS`'s own filter, non-recursive,
    same as `preprocess_folder()`'s own listing); and a random sample of up
    to `sample_size` of them decode successfully via `cv2.imread` (a spot
    check, not a full-folder scan -- mirrors `verify_eyeq_dataset()`'s own
    `_spot_check_corruption` approach and abort threshold).

    Reusable for both a raw input folder (before preprocessing) and a
    processed output folder (after) -- it has no opinion on which.

    Raises `RuntimeError` with a specific message on any failed check;
    returns a structured `ImageFolderReport` once every check passes.
    """
    if not os.path.isdir(path):
        raise RuntimeError(f"Dataset verification failed: {path} is not a directory.")

    image_files = sorted(
        f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if len(image_files) < min_images:
        raise RuntimeError(
            f"Dataset verification failed: {path} contains {len(image_files)} image file(s), "
            f"expected at least {min_images}."
        )

    sample = random.Random(42).sample(image_files, k=min(sample_size, len(image_files)))
    failures = []
    for name in sample:
        if cv2.imread(os.path.join(path, name)) is None:
            failures.append(name)

    if len(sample) > 0 and len(failures) / len(sample) > CORRUPTION_ABORT_FRACTION:
        raise RuntimeError(
            f"Dataset verification failed: {len(failures)}/{len(sample)} sampled images in "
            f"{path} failed to decode ({len(failures) / len(sample):.0%} > "
            f"{CORRUPTION_ABORT_FRACTION:.0%} threshold). The folder may be corrupted or "
            "incompletely staged/uploaded."
        )

    print(
        f"[{path}] {len(image_files)} images | corrupted (of {len(sample)} sampled): {len(failures)}"
    )
    return ImageFolderReport(
        path=path,
        image_count=len(image_files),
        corruption_sampled=len(sample),
        corruption_failures=len(failures),
    )
