# ============================================================================================
# READ-ONLY VISUAL REVIEW SHEET -- APTOS validation Grade-4 images decoded <= 2 by all three frozen
# NO_RACAF seeds (research record section 15, Audit 5), next to correctly graded Grade-4 images.
#
# Produces ONE PNG contact sheet and prints its path plus the ordered list of all 31 image IDs.
# Nothing else is written. No model is loaded, nothing is trained or inferred, no label, cache or
# checkpoint is touched, and no automated diagnosis or interpretation is performed. Images are
# shown as stored (original APTOS PNG, aspect preserved, no enhancement).
#
# GROUPS
#   Failure cases (21): the exact IDs printed by label_noise_ceiling_audit.py, Audit 5, in that
#     order. Each is re-checked against the frozen per-sample predictions: true grade 4 and
#     predicted grade <= 2 in all three NO_RACAF seeds (42, 123, 2026), BEST checkpoint.
#   Comparison (10): validation images with true grade 4 that at least one of the three seeds
#     predicts as 4, the first 10 by image ID in sorted order. No image-quality or disease rule is
#     used. ("Correct in all three seeds" was not used: the frozen seeds get only 6 / 3 / 19 PDR
#     images right, so too few would qualify.)
#   Every case is labelled: image ID | true grade | predictions of seeds 42/123/2026.
#
# TWO VIEWS PER CASE (both unenhanced; the only operation is resampling, LANCZOS)
#   1. Full original fundus, aspect ratio preserved, fitted into a FULL_TILE x FULL_TILE box.
#   2. "central crop — unenhanced": a fixed geometric crop, identical rule for every image -- a
#      square of side CROP_FRACTION x (shorter image side), centred on the image centre, shown at
#      CROP_TILE px. No crop is chosen automatically, and nothing is detected, segmented, sharpened
#      or contrast-adjusted. Because the crop is the geometric centre, it contains the optic disc
#      only when the disc happens to lie near the image centre.
#
# RUN: a Colab runtime with Drive (CPU is enough). Paste into one cell or %run.
# Local/testing overrides: DR_REPO_DIR, DR_EXPERIMENTS_ROOT, DR_APTOS_IMAGE_DIR.
# ============================================================================================

import datetime
import os
import posixpath
import subprocess
import sys

FAILURE_IDS = (  # Audit 5 order (ascending mean p_gt_2)
    "eaa0dfbd5024", "b37aae3c8fe1", "d1a24527a15d", "8bed09514c3b", "6cd606dc52e9",
    "d48178e4a49b", "63b4d030b016", "82bb8a01935f", "e3ab63dc9a60", "1bf30c84bbad",
    "cd54d022e37d", "fce93caa4758", "f03d3c4ce7fb", "65c958379680", "887c26fc0e1f",
    "211518c46162", "789434d095d1", "84c663f39632", "6cfb7b44ef6f", "4bd941611343",
    "b3819a805dca",
)
N_COMPARISON = 10
SEEDS = (42, 123, 2026)
PDR = 4
FULL_TILE = 960       # box for the full fundus (square, aspect preserved inside)
CROP_TILE = 960       # displayed size of the central crop (square)
CROP_FRACTION = 0.5   # crop side = 0.5 x shorter image side (~1,070-1,420 px on typical APTOS images)
VIEW_GAP = 8          # gap between the two views of one case
LABEL_H = 64          # label band under each case
COLUMNS = 3           # cases per row
HEADER_H = 70
MARGIN = 16

REPO_URL = "https://github.com/yasodharan27/diabetic_retinoplasty.git"
REPO_DIR = os.environ.get("DR_REPO_DIR", "/content/diabetic_retinoplasty")


def setup():
    on_colab = os.path.isdir("/content") and "DR_EXPERIMENTS_ROOT" not in os.environ
    if on_colab and not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
    sys.path.insert(0, os.path.join(REPO_DIR, "colab", "common"))
    import colab_config
    experiments_root = os.environ.get("DR_EXPERIMENTS_ROOT", colab_config.DRIVE.experiments_root)
    image_dir = os.environ.get("DR_APTOS_IMAGE_DIR",
                               os.path.join(colab_config.APTOS2019_RAW_DIR, "train_images"))
    return experiments_root, image_dir


def load_predictions(experiments_root):
    import pandas as pd
    frames = {}
    for seed in SEEDS:
        path = posixpath.join(experiments_root, "ImprovedTraining", "improved_multiseed_2026_09",
                              "NO_RACAF", f"seed_{seed}", "evaluation", "per_sample_best.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"frozen per-sample predictions not found: {path}")
        frames[seed] = pd.read_csv(path, dtype={"image_id": str}).set_index("image_id")
    reference = frames[SEEDS[0]]
    for seed, frame in frames.items():
        if not (frame.index.sort_values().equals(reference.index.sort_values())
                and (frame.loc[reference.index, "true_grade"] == reference["true_grade"]).all()):
            raise RuntimeError(f"seed {seed}: validation population differs from seed {SEEDS[0]}")
    return frames


def select(frames):
    reference = frames[SEEDS[0]]
    preds = {i: tuple(int(frames[s].loc[i, "predicted_grade"]) for s in SEEDS) for i in reference.index}
    truth = {i: int(reference.loc[i, "true_grade"]) for i in reference.index}
    for i in FAILURE_IDS:
        if i not in truth:
            raise RuntimeError(f"{i} is not in the frozen validation population")
        if truth[i] != PDR or max(preds[i]) > 2:
            raise RuntimeError(f"{i} does not meet the Audit-5 criterion: grade {truth[i]}, "
                               f"predictions {preds[i]}")
    eligible = sorted(i for i in reference.index
                      if truth[i] == PDR and PDR in preds[i] and i not in FAILURE_IDS)
    if len(eligible) < N_COMPARISON:
        raise RuntimeError(f"only {len(eligible)} grade-4 images are predicted 4 by at least one "
                           f"seed; the rule needs {N_COMPARISON}. No other rule is substituted.")
    return preds, truth, eligible[:N_COMPARISON], len(eligible)


def load_font(size):
    from PIL import ImageFont
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                      "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def views(path):
    """(full view, central crop, crop box in original pixels). Resampling only; no enhancement."""
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        side = int(round(CROP_FRACTION * min(width, height)))
        left, top = (width - side) // 2, (height - side) // 2
        box = (left, top, left + side, top + side)
        crop = image.crop(box).resize((CROP_TILE, CROP_TILE), Image.LANCZOS)
        full = image.copy()
        full.thumbnail((FULL_TILE, FULL_TILE), Image.LANCZOS)
        canvas = Image.new("RGB", (FULL_TILE, FULL_TILE), (0, 0, 0))
        canvas.paste(full, ((FULL_TILE - full.width) // 2, (FULL_TILE - full.height) // 2))
        return canvas, crop, box


def main():
    from PIL import Image, ImageDraw

    experiments_root, image_dir = setup()
    frames = load_predictions(experiments_root)
    preds, truth, comparison, n_eligible = select(frames)
    groups = [("FAILURE CASES: true grade 4, predicted <= 2 by all three NO_RACAF seeds (n=21)",
               list(FAILURE_IDS)),
              (f"COMPARISON: true grade 4, predicted 4 by >= 1 seed; first {N_COMPARISON} of "
               f"{n_eligible} by sorted ID (n={N_COMPARISON})", comparison)]
    for _, ids in groups:
        for i in ids:
            if not os.path.exists(os.path.join(image_dir, f"{i}.png")):
                raise FileNotFoundError(os.path.join(image_dir, f"{i}.png"))

    view_h = max(FULL_TILE, CROP_TILE)
    cell_w, cell_h = FULL_TILE + VIEW_GAP + CROP_TILE + MARGIN, view_h + LABEL_H + MARGIN
    rows = [(len(ids) + COLUMNS - 1) // COLUMNS for _, ids in groups]
    width = MARGIN + COLUMNS * cell_w
    height = sum(HEADER_H + r * cell_h for r in rows) + MARGIN
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    header_font, label_font = load_font(34), load_font(24)

    y = 0
    ordered = []
    for (title, ids), n_rows in zip(groups, rows):
        draw.text((MARGIN, y + 18), title, fill=(0, 0, 0), font=header_font)
        y += HEADER_H
        for k, i in enumerate(ids):
            x0 = MARGIN + (k % COLUMNS) * cell_w
            y0 = y + (k // COLUMNS) * cell_h
            full, crop, box = views(os.path.join(image_dir, f"{i}.png"))
            crop_x = x0 + FULL_TILE + VIEW_GAP
            sheet.paste(full, (x0, y0))
            sheet.paste(crop, (crop_x, y0))
            label = (f"{i} | true grade {truth[i]} | "
                     f"pred s42/s123/s2026: {'/'.join(map(str, preds[i]))}")
            draw.text((x0 + 4, y0 + view_h + 16), label, fill=(0, 0, 0), font=label_font)
            draw.text((crop_x + 4, y0 + view_h + 16), "central crop — unenhanced",
                      fill=(0, 0, 0), font=label_font)
            ordered.append((len(ordered) + 1, "failure" if ids is groups[0][1] else "comparison",
                            i, truth[i], preds[i], box))
            print(f"  placed {len(ordered)}/31 {i}")
        y += n_rows * cell_h

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = posixpath.join(experiments_root, "Grade3vs4_VisualReview", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    out_path = posixpath.join(out_dir, "pdr_review_contact_sheet.png")
    sheet.save(out_path, optimize=True)

    print("\nCONTACT SHEET:", out_path)
    print(f"size {width} x {height} px; full view {FULL_TILE} px box; central crop "
          f"{CROP_FRACTION:g} x shorter side, centred, shown at {CROP_TILE} px; original images, "
          "no enhancement (resampling only)")
    print("\nORDERED IDS (position | group | image_id | true grade | pred s42/s123/s2026 | "
          "crop box in original pixels)")
    for pos, group, i, g, p, box in ordered:
        print(f"{pos:>2} | {group:<10} | {i} | {g} | {'/'.join(map(str, p))} | {box}")
    print("\nNo automated interpretation was performed.")


if __name__ == "__main__":
    main()
