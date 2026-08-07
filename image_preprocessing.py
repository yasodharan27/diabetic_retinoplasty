"""
Preprocessing module for pipeline Step 2.

Implements only the two approved, currently-missing operations from
PROJECT_CODE.md's preprocessing list -- CLAHE and Gamma Correction. Green
Channel Extraction, Ben Graham Preprocessing, Median Denoising, Resizing,
and Data Augmentation already exist in pre_process_with_dataset_download.py
/ pre_process_test_and_train.py and are intentionally left untouched and
unduplicated here.

Operates on raw RGB fundus images from any dataset folder (EyePACS,
APTOS2019, ...) and writes results into a separate output location --
`raw/` inputs are only ever read, never modified or overwritten, per the
project's dataset policy.

Configuration: `gamma`, `clip_limit`, and `tile_grid_size` are fully
configurable, first-class parameters on every public function in this
module (apply_gamma_correction, apply_clahe, preprocess_array,
preprocess_image, preprocess_folder, preprocess_dataset) and on the CLI.
Each defaults to `None`, which resolves to the centralized default in
`config.PREPROCESSING` (config.py's `PreprocessingConfig` -- itself
overridable via the DEFAULT_GAMMA / DEFAULT_CLAHE_CLIP_LIMIT /
DEFAULT_CLAHE_TILE_GRID_WIDTH / DEFAULT_CLAHE_TILE_GRID_HEIGHT environment
variables). An explicit non-None argument always takes precedence over the
config default. No preprocessing parameter is hardcoded in this file.

Profiles: `preprocess_array`, `preprocess_image`, `preprocess_folder`, and
`preprocess_dataset` also accept a `profile=` argument -- a
config.PreprocessingProfile instance (e.g. `config.PREPROCESSING_PROFILES.IQA`)
or its name as a string ("IQA" / "DR"). A profile supplies gamma/clip_limit/
tile_grid_size fallback values *and* decides whether CLAHE runs at all
(`clahe_enabled`). Resolution priority for gamma/clip_limit/tile_grid_size is:

    explicit function argument > profile value > config.PREPROCESSING default

The IQA profile disables CLAHE and defaults gamma to 1.0 (an identity
transform), so it performs no preprocessing by default -- the Image
Quality Assessment model (Step 1) is trained on, and must continue to see,
the original unprocessed RGB EyeQ images. The DR profile enables CLAHE and
uses config.PREPROCESSING's centralized Gamma+CLAHE defaults. Omitting
`profile` (the default, `None`) runs the same unconditional Gamma+CLAHE
pipeline this module has always run -- existing callers are unaffected.

Not used by the Image Quality Assessment module (Step 1) -- besides being
available as the `IQA` no-op profile above -- which is required to see
only the original, unprocessed RGB images.

Execution logging: `preprocess_folder` / `preprocess_dataset` accept an
optional `log_file=None`. When given, a CSV row (image_name, input_path,
output_path, profile, status, error_message, processing_time_ms) is
written per file encountered -- successes, failures, and skipped
non-image files alike -- so a full run over EyePACS/APTOS2019 is fully
auditable. The underlying file-processing behavior (what gets read/
written, per-file error handling, overwrite protection on originals) is
identical whether or not `log_file` is supplied -- only the CSV
side-effect is conditional.

Both functions always return a `PreprocessingResult`: `summary` (a
`PreprocessingSummary` -- total_images, processed, skipped, failed,
total_processing_time) plus `processed_files` / `skipped_files` /
`failed_files`, the actual `pathlib.Path` lists behind those counts
(`processed_files` holds each success's *output* path; `skipped_files` /
`failed_files` hold *input* paths, since nothing was written for those).
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import PREPROCESSING, PREPROCESSING_PROFILES, dataset_processed_dir, dataset_raw_dir

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
LOG_COLUMNS = (
    "image_name", "input_path", "output_path", "profile",
    "status", "error_message", "processing_time_ms",
)


@dataclass(frozen=True)
class PreprocessingSummary:
    """Aggregate statistics only -- see PreprocessingResult for the
    per-file path lists of a preprocess_folder() / preprocess_dataset()
    run. `total_processing_time` is wall-clock seconds for the whole run;
    each logged CSV row's `processing_time_ms` is per-file, in
    milliseconds. Invariant: total_images == processed + skipped + failed."""
    total_images: int
    processed: int
    skipped: int
    failed: int
    total_processing_time: float


@dataclass(frozen=True)
class PreprocessingResult:
    """Full result of a preprocess_folder() / preprocess_dataset() run:
    aggregate counts (`summary`) plus the actual per-file paths for every
    outcome. `processed_files` holds each successful run's *output* path
    (the produced artifact); `skipped_files` / `failed_files` hold *input*
    paths (nothing was written for those)."""
    summary: PreprocessingSummary
    processed_files: list[Path]
    skipped_files: list[Path]
    failed_files: list[Path]


def _profile_label(profile):
    """Human-readable value for the CSV log's `profile` column."""
    if profile is None:
        return "none"
    if isinstance(profile, str):
        return profile
    return "custom"


def _open_log_writer(log_file, overwrite_log):
    """Validates overwrite protection, opens `log_file` for writing, and
    returns (file_handle, csv.DictWriter) with the header already
    written. Caller is responsible for closing the file handle."""
    if os.path.exists(log_file) and not overwrite_log:
        raise FileExistsError(
            f"Log file already exists at {log_file} -- pass overwrite_log=True "
            "to overwrite it, or choose a different path."
        )
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    handle = open(log_file, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
    writer.writeheader()
    return handle, writer


def _resolve_gamma(gamma):
    """None -> config.PREPROCESSING.DEFAULT_GAMMA; an explicit value always passes through unchanged."""
    return PREPROCESSING.DEFAULT_GAMMA if gamma is None else gamma


def _resolve_clip_limit(clip_limit):
    """None -> config.PREPROCESSING.DEFAULT_CLAHE_CLIP_LIMIT; an explicit value always passes through unchanged."""
    return PREPROCESSING.DEFAULT_CLAHE_CLIP_LIMIT if clip_limit is None else clip_limit


def _resolve_tile_grid_size(tile_grid_size):
    """None -> config.PREPROCESSING.DEFAULT_CLAHE_TILE_GRID_SIZE; an explicit value always passes through unchanged."""
    return PREPROCESSING.DEFAULT_CLAHE_TILE_GRID_SIZE if tile_grid_size is None else tile_grid_size


def _resolve_profile(profile):
    """Accepts a config.PreprocessingProfile instance, a profile name
    string ("IQA"/"DR"), or None. Returns a PreprocessingProfile, or None
    when no profile was given (the pre-profile, unconditional pipeline).

    Dispatches on type via `isinstance(profile, str)` rather than checking
    for `config.PreprocessingProfile` specifically -- a profile object only
    needs to expose `.gamma` / `.clahe_enabled` / `.clip_limit` /
    `.tile_grid_size` (duck typing), which also keeps this working if
    `config` is ever reloaded (e.g. `importlib.reload`), since that
    produces a new dataclass *type* object that a nominal `isinstance`
    check against this module's own import-time class reference would no
    longer match."""
    if profile is None or not isinstance(profile, str):
        return profile
    try:
        return getattr(PREPROCESSING_PROFILES, profile)
    except AttributeError:
        available = [f.name for f in fields(PREPROCESSING_PROFILES)]
        raise ValueError(f"Unknown preprocessing profile '{profile}'. Available: {available}")


def apply_gamma_correction(image, gamma=None):
    """Power-law (gamma) tone correction via a 255-entry lookup table.
    gamma > 1 brightens midtones, gamma < 1 darkens them. Applied
    identically across every channel of the input array.

    `gamma=None` (the default) resolves to config.PREPROCESSING.DEFAULT_GAMMA;
    an explicit value always takes precedence."""
    gamma = _resolve_gamma(gamma)
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(image, table)


def apply_clahe(image, clip_limit=None, tile_grid_size=None):
    """Contrast Limited Adaptive Histogram Equalization. For color images,
    applied to the L (lightness) channel in LAB space so hue/saturation are
    preserved (equivalent to plain CLAHE for single-channel/grayscale input).

    `clip_limit=None` / `tile_grid_size=None` (the defaults) resolve to
    config.PREPROCESSING.DEFAULT_CLAHE_CLIP_LIMIT / DEFAULT_CLAHE_TILE_GRID_SIZE;
    an explicit value always takes precedence."""
    clip_limit = _resolve_clip_limit(clip_limit)
    tile_grid_size = _resolve_tile_grid_size(tile_grid_size)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_array(image, profile=None, gamma=None, clip_limit=None, tile_grid_size=None):
    """Apply the approved Step 2 pipeline (Gamma Correction, then CLAHE --
    if enabled) to an in-memory image array. Gamma runs first so CLAHE's
    local contrast enhancement operates on an already tonally-corrected
    image.

    Resolution priority per parameter: explicit function argument >
    `profile` value > config.PREPROCESSING default. `profile` additionally
    gates whether CLAHE runs at all (its `clahe_enabled` flag); omitting
    `profile` always runs CLAHE, matching this module's pre-profile
    behavior exactly."""
    resolved_profile = _resolve_profile(profile)

    if resolved_profile is not None:
        if gamma is None:
            gamma = resolved_profile.gamma
        if clip_limit is None:
            clip_limit = resolved_profile.clip_limit
        if tile_grid_size is None:
            tile_grid_size = resolved_profile.tile_grid_size

    image = apply_gamma_correction(image, gamma=gamma)
    if resolved_profile is None or resolved_profile.clahe_enabled:
        image = apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    return image


def preprocess_image(input_path, output_path, profile=None, gamma=None,
                      clip_limit=None, tile_grid_size=None):
    """
    Process a single image file and write the result to `output_path`.
    Never touches `input_path` -- reads it, writes a new file elsewhere.
    See preprocess_array for the profile/parameter resolution rules.
    """
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError(
            f"Refusing to preprocess {input_path} in place -- output_path must "
            "differ from input_path (original images are never overwritten)."
        )

    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to load image: {input_path}")

    processed = preprocess_array(
        image, profile=profile, gamma=gamma, clip_limit=clip_limit, tile_grid_size=tile_grid_size
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not cv2.imwrite(output_path, processed):
        raise IOError(f"Failed to write processed image to {output_path}")
    return output_path


def preprocess_folder(input_dir, output_dir, profile=None, gamma=None, clip_limit=None,
                       tile_grid_size=None, log_file=None, overwrite_log=False):
    """
    Process every image directly inside `input_dir` (non-recursive) and
    write each result under `output_dir`, preserving filenames. `input_dir`
    is only ever read from -- its contents are never modified. A file that
    fails to process is logged and skipped; the rest of the folder still
    runs. See preprocess_array for the profile/parameter resolution rules.

    log_file: optional CSV path (see module docstring for columns; format
    unchanged by the presence of PreprocessingResult below). Left at None
    (the default), no CSV is written and behavior is identical to before
    this feature existed. Raises FileExistsError if log_file already
    exists, unless overwrite_log=True.

    Always returns a PreprocessingResult (summary + processed_files /
    skipped_files / failed_files path lists).
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"{input_dir} is not a directory.")
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        raise ValueError(
            f"Refusing to preprocess {input_dir} in place -- output_dir must "
            "differ from input_dir (original images are never overwritten)."
        )

    all_filenames = sorted(os.listdir(input_dir))
    image_filenames = [f for f in all_filenames if f.lower().endswith(IMAGE_EXTENSIONS)]
    if not image_filenames:
        raise FileNotFoundError(f"No images found in {input_dir}.")

    os.makedirs(output_dir, exist_ok=True)

    log_handle, log_writer = (None, None)
    if log_file is not None:
        log_handle, log_writer = _open_log_writer(log_file, overwrite_log)

    profile_label = _profile_label(profile)
    processed_files = []
    skipped_files = []
    failed_files = []
    run_start = time.perf_counter()

    try:
        for filename in tqdm(all_filenames, desc=f"Preprocessing {input_dir}"):
            input_path = os.path.join(input_dir, filename)

            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                skipped_files.append(Path(input_path))
                if log_writer:
                    log_writer.writerow({
                        "image_name": filename, "input_path": input_path, "output_path": "",
                        "profile": profile_label, "status": "skipped",
                        "error_message": "unsupported file extension", "processing_time_ms": "",
                    })
                continue

            output_path = os.path.join(output_dir, filename)
            file_start = time.perf_counter()
            try:
                preprocess_image(
                    input_path, output_path,
                    profile=profile, gamma=gamma, clip_limit=clip_limit, tile_grid_size=tile_grid_size,
                )
                elapsed_ms = (time.perf_counter() - file_start) * 1000
                processed_files.append(Path(output_path))
                if log_writer:
                    log_writer.writerow({
                        "image_name": filename, "input_path": input_path, "output_path": output_path,
                        "profile": profile_label, "status": "processed",
                        "error_message": "", "processing_time_ms": f"{elapsed_ms:.3f}",
                    })
            except (ValueError, IOError) as e:
                elapsed_ms = (time.perf_counter() - file_start) * 1000
                failed_files.append(Path(input_path))
                print(f"[image_preprocessing] Skipping {filename}: {e}")
                if log_writer:
                    log_writer.writerow({
                        "image_name": filename, "input_path": input_path, "output_path": "",
                        "profile": profile_label, "status": "failed",
                        "error_message": str(e), "processing_time_ms": f"{elapsed_ms:.3f}",
                    })
    finally:
        if log_handle is not None:
            log_handle.close()

    total_processing_time = time.perf_counter() - run_start

    print(
        f"Processed {len(processed_files)}/{len(all_filenames)} images into {output_dir}"
        + (f" ({len(failed_files)} failed, {len(skipped_files)} skipped)"
           if (failed_files or skipped_files) else "")
    )

    summary = PreprocessingSummary(
        total_images=len(all_filenames),
        processed=len(processed_files),
        skipped=len(skipped_files),
        failed=len(failed_files),
        total_processing_time=total_processing_time,
    )
    return PreprocessingResult(
        summary=summary,
        processed_files=processed_files,
        skipped_files=skipped_files,
        failed_files=failed_files,
    )


def preprocess_dataset(dataset_name, subfolder=None, profile=None, gamma=None, clip_limit=None,
                        tile_grid_size=None, log_file=None, overwrite_log=False):
    """
    Convenience wrapper for the datasets/<name>/raw -> datasets/<name>/processed
    convention shared by EyePACS, APTOS2019, etc. (see config.dataset_raw_dir /
    config.dataset_processed_dir). `subfolder`, if given, is appended to both
    the raw and processed dirs -- e.g. subfolder="train_images" for APTOS2019
    (datasets/APTOS2019/raw/train_images) or subfolder="train" for EyePACS
    (datasets/EyePACS/raw/train). See preprocess_array for the
    profile/parameter resolution rules and preprocess_folder for the
    log_file/overwrite_log/PreprocessingResult contract.
    """
    raw_dir = dataset_raw_dir(dataset_name)
    processed_dir = dataset_processed_dir(dataset_name)
    if subfolder:
        raw_dir = os.path.join(raw_dir, subfolder)
        processed_dir = os.path.join(processed_dir, subfolder)
    return preprocess_folder(
        raw_dir, processed_dir,
        profile=profile, gamma=gamma, clip_limit=clip_limit, tile_grid_size=tile_grid_size,
        log_file=log_file, overwrite_log=overwrite_log,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply approved Step 2 preprocessing (Gamma Correction + CLAHE) "
                     "to a single image or a folder of images."
    )
    parser.add_argument("input", help="Path to an image file or a folder of images.")
    parser.add_argument("output", help="Output file path (single image) or output folder (folder mode).")
    parser.add_argument("--profile", choices=[f.name for f in fields(PREPROCESSING_PROFILES)], default=None,
                         help="Named preprocessing profile from config.py's PREPROCESSING_PROFILES "
                              "(IQA = no preprocessing, DR = Gamma Correction + CLAHE). Omit to run "
                              "the unconditional Gamma+CLAHE pipeline (this module's default behavior).")
    parser.add_argument("--gamma", type=float, default=None,
                         help=f"Gamma correction value (default from config.py: "
                              f"{PREPROCESSING.DEFAULT_GAMMA}, override via the DEFAULT_GAMMA "
                              "env var). gamma > 1 brightens, gamma < 1 darkens. Overrides --profile.")
    parser.add_argument("--clip-limit", type=float, default=None,
                         help=f"CLAHE clipLimit (default from config.py: "
                              f"{PREPROCESSING.DEFAULT_CLAHE_CLIP_LIMIT}, override via the "
                              "DEFAULT_CLAHE_CLIP_LIMIT env var). Overrides --profile.")
    parser.add_argument("--tile-grid-size", type=int, nargs="+", default=None, metavar="N",
                         help="CLAHE tileGridSize: one value for a square grid "
                              "(e.g. 8 -> (8, 8)), or two for width/height "
                              f"(e.g. 8 16 -> (8, 16)). Default from config.py: "
                              f"{PREPROCESSING.DEFAULT_CLAHE_TILE_GRID_SIZE} (override via the "
                              "DEFAULT_CLAHE_TILE_GRID_WIDTH / DEFAULT_CLAHE_TILE_GRID_HEIGHT env vars). "
                              "Overrides --profile.")
    parser.add_argument("--log-file", default=None,
                         help="Folder mode only: write a per-image CSV execution log to this path.")
    parser.add_argument("--overwrite-log", action="store_true",
                         help="Allow --log-file to overwrite an existing file (refused by default).")
    args = parser.parse_args()

    if args.tile_grid_size is None:
        tile_grid_size = None
    elif len(args.tile_grid_size) == 1:
        tile_grid_size = (args.tile_grid_size[0], args.tile_grid_size[0])
    elif len(args.tile_grid_size) == 2:
        tile_grid_size = tuple(args.tile_grid_size)
    else:
        parser.error("--tile-grid-size accepts either 1 value (square) or 2 values (width height).")

    if os.path.isdir(args.input):
        result = preprocess_folder(
            args.input, args.output, profile=args.profile,
            gamma=args.gamma, clip_limit=args.clip_limit, tile_grid_size=tile_grid_size,
            log_file=args.log_file, overwrite_log=args.overwrite_log,
        )
        print(result.summary)
    else:
        preprocess_image(
            args.input, args.output, profile=args.profile,
            gamma=args.gamma, clip_limit=args.clip_limit, tile_grid_size=tile_grid_size,
        )
        print(f"Processed image saved to {args.output}")


if __name__ == "__main__":
    main()
