# TJDR pre-training audit (Step 2, 2026-10-01; research record §42)

Nothing was trained or cached. Every number below was measured on the downloaded files.

**Tags:**
- **[V]** measured / verified on the data or the source page.
- **[vis]** visual inspection.
- **[I]** inference.

**Audit script:** `research/Grade3vs4_Architecture_Research/tjdr_preflight_audit.py`.

**Outputs (sha256):** `datasets/TJDR/audit/`
- `inventory.csv` 6ef4e8dd…
- `components.csv` c7fc0230…
- `idrid_seg_inventory.csv` 903afde0…
- `overlap.json` d1d5fed3…
- `overlap_stage2.json` 2e192b6a…
- `overlays/`

## 1. Source, download and licence

**Source [V]:** `github.com/NekoPii/TJDR`. It has no code: only README + LICENSE. Last push 2025-12-09.
- The README links the data on OneDrive and Google Drive (folder `1RBAtPPAvX1KXiJrNsrJz7jSMCL3AgAJT`).
- Downloaded from Google Drive with rclone: 1,122 files, 4.85 GiB.
- `rclone check` against the source md5: **0 differences**.
- The source listing with md5s is kept at `datasets/TJDR/audit/source_listing_md5.txt` (sha256 31cdefa2…).

**Licence [V]:**

| Where | What it says |
|---|---|
| GitHub repo `LICENSE` (commit "Create LICENSE", 2025-02-27; GitHub API spdx) | **Apache-2.0** |
| arXiv 2312.15389 abstract page (`rel` = "Rights to this article") | **CC BY-NC-ND 4.0**, which covers the *paper*, not the data |
| Google Drive data folder | **no licence file** (only `train/`, `test/`) |

**Is "Apache-2.0" supported?** Partly.
- It is the only licence the authors attach to the dataset release: the repository whose sole purpose is to distribute the data link.
- But the data files themselves carry no licence, the LICENSE file was added 14 months after release, and Apache-2.0 is a software licence.
- **Conservative reading [I]:** use for non-commercial research with citation (both candidate licences allow that), and do not redistribute the images.
- Our Drive copy is private. This is not a blocker.

## 2. Structure and counts

| | train | test | total |
|---|---|---|---|
| images (PNG, RGB) | 448 | 113 | **561** |
| annotation PNGs | 448 | 113 | **561** |
| image without annotation / annotation without image | 0 / 0 | 0 / 0 | 0 |
| decode errors / image–mask size mismatch | 0 / 0 | 0 / 0 | 0 |
| Topcon TRC-50DX, 2048×2048 | 202 | 55 | 257 |
| Zeiss CLARUS 500, 3912×3912 | 246 | 58 | 304 |

These match the paper (448/113; 257 × 2048², 304 × 3912²).

**Masks [V]:**
- One single-channel **palette ("P") PNG per image**, at the same size as the image.
- Values **{0, 1, 2, 3, 4}**; the palette is the Labelme/VOC colormap, identical in all 561 files.
- Value 0 appears in 561 files; 1 in 322; 2 in 318; 3 in 174; 4 in 192.
- They are Labelme polygons already rasterised: **no JSON is released**.

**Class codes [V]:** paper: EX(1), HE(2), MA(3), SE(4). Confirmed on the data:
- mean RGB inside EX = (153, 106, 40): bright yellow.
- HE = (114, 60, 32) and MA = (120, 61, 29): dark red.
- SE = (139, 95, 46): pale.
- MA components are the smallest.

**The class order differs from ours (MA, HE, EX, SE); a loader must map explicitly.**

## 3. Class / annotation table [V]

| class | images train / test | components | pixels (native) | outside-FOV pixels |
|---|---|---|---|---|
| MA | 137 / 37 (174) | 1,507 | 332,466 | 0 |
| HE | 249 / 69 (318) | 3,204 | 9,984,369 | 10 |
| EX | 255 / 67 (322) | 3,352 | 4,520,700 | 0 |
| SE | 151 / 41 (192) | 503 | 2,371,366 | 0 |

**Per camera (images):**

| camera | MA | HE | EX | SE |
|---|---|---|---|---|
| CLARUS | 75 | 170 | 219 | 127 |
| TRC-50DX | 99 | 148 | 103 | 65 |

**Other counts:**
- 57 images have no lesion; all are TRC-50DX (48 train, 9 test).
- Classes per image: 0 → 57, 1 → 194, 2 → 158, 3 → 112, 4 → 40.
- For comparison, IDRiD segmentation (81 images): MA 81, HE 80, EX 81, SE 40.
- TJDR adds 2.1× (MA), 4.0× (HE, EX) and 4.8× (SE) more annotated images than IDRiD.

## 4. Definitions vs ours (paper text [V])

- **HE:** "deep-layer dot/blot … superficial streak/flame-shaped … **large sub-internal-limiting-membrane or subretinal** haemorrhages".
  - This includes preretinal (sub-ILM) haemorrhage, so it is consistent with **HE = any haemorrhage**.
  - Vitreous haemorrhage is not mentioned (unverifiable; expected to be rare in this cohort).
- **MA:** "well-defined, smooth-edged red / dark-red spots of varying sizes". Same concept as IDRiD.
  - The MA/small-HE boundary is annotator-specific (§6).
- **EX:** yellow punctate/patchy lesions with clear boundaries = hard exudates. ✓
- **SE:** cotton-wool, ill-defined greyish-white. ✓ (SE ≡ CWS)

## 5. Duplicates and overlap [V]

**Method:**
- Stage 1: exact file sha256 and pixel sha256.
- Stage 2: normalised correlation of the high-pass (vessel/lesion) texture of the FOV-cropped green channel at 128², mirror-aware.
- (A whole-image pHash was also computed, but it cannot separate fundus images: unrelated APTOS pairs collide. It is kept in `overlap.json` only as a record.)

**Calibration:**
- Positive control (Stage-2 DR preprocessing + downscale to 1024 + JPEG q85, 24 TJDR images): r = **0.83–0.98**.
- APTOS-internal: it finds APTOS's known duplicates (156 pairs ≥ 0.9), with a null median of 0.01 and a 99th percentile of 0.20.

| TJDR vs | n | exact sha | max r | pairs ≥ 0.5 | verdict |
|---|---|---|---|---|---|
| APTOS train (our train + validation) | 3,662 | 0 | 0.50 | 1 (visual: different eyes, mirror artefact) | **no overlap** |
| APTOS test | 1,928 | 0 | 0.46 | 0 | **no overlap** |
| IDRiD grading | 516 | 0 | 0.40 | 0 | **no overlap** |
| IDRiD segmentation (includes the old Stage-4 training data = IDRiD seg train) | 81 | 0 | 0.31 | 0 | **no overlap** |

**Inside TJDR:**
- **6 byte-identical image pairs**, with **different masks**. Two of them **cross the official split**:
  - train_101 = test_023
  - train_092 = test_021
  - train_039 = train_041
  - train_040 = train_042
  - train_089 = train_091
  - train_173 = train_174
- **2 same-eye near-duplicates** (visually the same fundus recaptured/recoloured):
  - train_100 ≈ train_105 (r = 0.96)
  - test_002 ≈ test_003 (r = 0.82)
- All other pairs are ≤ 0.67, and the visually checked pairs at 0.55–0.67 are different eyes. No patient IDs are released, so patient-level disjointness beyond this cannot be proven.

**Annotation consistency, measured on the 6 twice-annotated images (Dice between the two masks):**

| class | Dice |
|---|---|
| MA | 0.36–0.55 |
| HE | 0.62–0.78 |
| EX | 0.69–0.83 |
| SE | 0.88–0.94 |

One pair has SE (825 px) annotated in one copy and absent in the other.

## 6. Completeness and suitability for the partial-label loss

- **Format:** each mask annotates all four classes (0 = negative), so A_TJDR = {MA, HE, EX, SE}.
- **Single label per pixel:** the index mask is mutually exclusive, but IDRiD's independent masks overlap on < 0.1 % of pixels for every class pair (max HE∩SE 0.07 %). Nothing material is lost.
- **Completeness:**
  - [vis] obvious lesions are annotated; a few faint dots are not. Polygons are generous.
  - [V] from the duplicate pairs: MA labels are noisy (Dice about 0.45), and one SE omission.
  - This is label noise, not missing classes, so treating absence as negative is correct for the partial-label loss.
  - MA from TJDR is weaker supervision than IDRiD MA.
- **FOV:**
  - The FOV is a circle inscribed in a square; all corners are exactly 0. The Step-1 `fov_mask` (> 10/255) gives an FOV fraction of 0.785 (= π/4).
  - Ten CLARUS images have 0.69–0.78 (dark rim/eyelid shading inside the circle).
  - Only 10 annotated pixels (HE) fall outside it.

## 7. Geometry vs the 1536 → 512 pipeline

- **Aspect:** TJDR is square, so the direct resize to 1536² is isotropic. APTOS (aspect 1.0–1.51, median 1.33) and IDRiD (1.51) are squashed horizontally by up to 1.5×. The Stage-4 training mix covers both.
- **Scale (optic-disc diameter in the 1536 frame, [vis] small sample):**

  | source | OD diameter at 1536 |
  |---|---|
  | TRC-50DX | about 140–170 px |
  | IDRiD | about 185 px |
  | APTOS | about 120–225 px |
  | **CLARUS 500** | **about 100 px** (wide field) |

  - CLARUS lesions are therefore about 0.5–0.6× APTOS pixel scale.
  - MA equivalent diameter at 1536 (median): CLARUS 5.4, TRC 12.3, IDRiD 7.6 px.
  - TRC MA outlines are about 2× wider relative to the frame than IDRiD's (coarser polygons).
- **Assessment [I]:** a real but non-blocking mismatch. 54 % of TJDR is at about half scale, which acts as scale augmentation. Report Stage-4 validation per camera.

## 8. Tiny-lesion survival under the CURRENT rule (threshold 0.5, unchanged)

**Rule:** native binary mask → `stage4_v2.resize_mask_full_frame` (area-weighted, ≥ 0.5) → exact 3×3 block mean + max (`pack_pathology_maps`) → 512 uint8. Identity with the Step-1 function was asserted in the script.

| group | class | components | median eq-diam @1536 | survive @1536 | survive 512-max | lost |
|---|---|---|---|---|---|---|
| IDRiD (81) | MA | 3,497 | 7.6 | 100 % | 100 % | 0 |
| IDRiD | HE / EX / SE | 1,900 / 11,642 / 150 | 19.9 / 7.3 / 46.1 | 100 % | 100 % | 0 |
| TJDR TRC-50DX | MA | 742 | 12.3 | 99.7 % | 99.7 % | 2 (1–2 px specks) |
| TJDR TRC-50DX | HE / EX / SE | 1,192 / 1,130 / 216 | 27.4 / 19.4 / 47.7 | 99.7 / 100 / 100 % | same | 4 HE (1-px fragments) |
| TJDR CLARUS 500 | MA | 765 | 5.4 | 100 % | 100 % | 0 |
| TJDR CLARUS 500 | HE / EX / SE | 2,012 / 2,222 / 287 | 12.8 / 6.1 / 23.6 | 99.9 / 99.6 / 100 % | same | 2 HE, 8 EX (4–12 px specks) |

**Summary:**
- 16 of 8,566 TJDR components are lost, and 0 of 17,189 IDRiD components.
- Every lost component has 1–12 native pixels; lost pixel mass is about 0.
- **No image loses an entire class.**
- Pixel retention at 1536 vs the area-expected value: median 1.00–1.02.
- 512 mean channel: the minimum possible value for a surviving component is 28/255 (1/9 of a block), so it is never quantised away.
- **Conclusion:** there is no evidence for changing the MA threshold.
- (Two components differ between the 1536 and 512-max survival checks, because a neighbouring lesion shares their 3×3 block.)

## 9. Step-1 implementation vs the actual TJDR structure

What fits unchanged:
- `resize_full_frame` / `resize_mask_full_frame` / `fov_mask` / `sample_patch_origin` work on 2048² and 3912² inputs.
- `DatasetSpec("TJDR")` has the correct A_d.
- The partial-label loss needs no change.

What must be handled explicitly in training preparation (not done now):

- **R1 — exclusion list (pinned):**
  - train: exclude 041, 042, 091, 174 (exact duplicates) and 105 (near-duplicate), keeping the lower id → **443**.
  - test (used as the second validation set): exclude 021 and 023 (copies of train images) and 003 (near-duplicate) → **110**.
- **R2 — TJDR mask reader:**
  - `np.asarray(Image.open(p))` **without** `.convert()`: `convert("L")` would turn palette indices into luminance.
  - Assert mode `P`, size equal to the image, and values ⊆ {0..4}.
  - Map {MA: 3, HE: 2, EX: 1, SE: 4}.
  - Binarise **per class before** `resize_mask_full_frame`. That function does `mask > 0`, so passing the index mask would merge all four classes.
- **R3 — `DatasetSpec("TJDR")`:** `verified=True`, with the label codes, counts, exclusion list and source-listing sha recorded.
- **R4 — stage TJDR to Drive** (`gdrive:DiabeticRetinopathy/datasets/TJDR`, md5-checked) before Colab training.
- **R5 (recommended):** report Stage-4 validation per camera, because of §7.

## 10. Decision

**GO for v2-a**, with R1–R4 applied in training preparation:
- No overlap with APTOS, IDRiD or the old Stage-4 data.
- Classes and definitions are compatible.
- Tiny-lesion survival is 99.8 % under the unchanged rule.
- The loss is compatible.

The official-split defects (2 cross-split duplicates, 4 within-train duplicates, 2 near-duplicates) are fully handled by the pinned exclusion list. MA label noise and the CLARUS scale are known limitations, not blockers.
