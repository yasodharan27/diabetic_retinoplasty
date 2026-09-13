# RACAF vs NO-RACAF — Exact Duplicate-Contamination Audit

**Scope:** a read-only, per-image audit of the 41 APTOS train/validation duplicate images
identified in the RACAF (`2026-09-12_02-45-05`) vs NO-RACAF (`2026-09-13_04-11-07`) validation
population, using the two experiments' saved per-image prediction CSVs, the authoritative split
manifest, and the raw APTOS 2019 training images. Executed once in Google Colab, 2026-09-13, as a
single self-contained read-only cell: it mounted Drive, read files, computed statistics, and wrote
nothing — no training, no model load, no inference, no file or checkpoint modification. This
supersedes the confusion-matrix-based sensitivity scenarios in
[RACAF_vs_NO_RACAF_Analysis.md](RACAF_vs_NO_RACAF_Analysis.md) §9 (2026-09-13, first version) with
an exact, per-image result.

**Result in one line:** the exact duplicate-excluded comparison confirms the finding — removing
the same 41 images from both models moves ΔQWK from +0.0828302 to +0.0828904 (n = 689), i.e. **the
NO-RACAF advantage is not explained by, or dependent on, train/validation duplication.**

---

## 1. Method

1. **File identification.** The two per-image CSVs were located at their expected paths —
   `experiments/FinalClassification/2026-09-12_02-45-05/evaluation/per_sample_corn_predictions.csv`
   (RACAF) and `.../2026-09-13_04-11-07/evaluation/per_sample_corn_predictions.csv` (NO-RACAF) —
   and each was confirmed against its own experiment's `metadata.json`,
   `evaluation/evaluation_manifest.json` and `evaluation/final_diagnostic_report.json`, all of
   which carry the matching `config_hash` (`3f549e1638d9409f7862f1e799bd7052` for RACAF,
   `adf02bb3f8aa6ab93843036bad6b339e` for NO-RACAF); the NO-RACAF diagnostic report additionally
   names `joint_stage05_08_no_racaf`. Sizes: RACAF CSV 294,047 bytes; NO-RACAF CSV 298,088 bytes;
   730 rows and 27 columns each.
2. **Alignment.** Rows were merged on `image_id` (never on row order, which differs between the
   two files) with `validate="one_to_one"`. No duplicate IDs within either file; the ID sets are
   identical (730/730 intersect); ground truth agrees for all 730 matched images; the `correct`
   column in each CSV is internally consistent with `true_grade == predicted_grade` for all 730
   rows in both files.
3. **Split verification.** `dataset_splits/aptos2019_train_val_split.csv` was read (3,662 rows:
   2,929 train / 733 val). Its SHA-256, computed inside the audit cell after normalising line
   endings, is `bc80fd450340b09307fbd80a1b00553e70e34d64a3cdf94635162b6c1e99aca5`. **Independently
   re-verified for this report** against the exact committed blob (`git show HEAD:dataset_splits/
   aptos2019_train_val_split.csv | sha256sum`) and against the file served from
   `raw.githubusercontent.com/yasodharan27/diabetic_retinoplasty/main/...`: both equal
   `bc80fd45…9aca5`. The 730 evaluated images are exactly "validation minus the 3 known
   empty-FOV exclusions" (`262ad704319c`, `26453eb7e989`, `3a122851e526`), and every evaluated
   image's CSV ground truth equals the split manifest's `diagnosis` — no mismatch.
4. **Duplicate identification — independently reproduced, not read from a prior list.** For every
   validation image, training images sharing its exact **byte size** were found first (a
   byte-identical file must have an identical size); only that reduced candidate set was hashed.
   This produced 42 validation candidates and 85 files (candidates plus their same-size training
   counterparts) requiring MD5. Each MD5 match was then confirmed with a **full byte-for-byte
   comparison** (`filecmp.cmp(..., shallow=False)`), not MD5 alone. 41 of the 42 candidates were
   confirmed byte-identical to at least one training image (the 42nd was a same-size, different-
   content false candidate, correctly excluded). Zero of the 3 empty-FOV exclusions have a
   training twin.
5. **Cross-check.** The reproduced 41-ID set was compared against the earlier content-hash audit's
   own list (`dup41.json`, produced independently in an earlier session): **identical, ID for
   ID.** This is two independent computations of the same result, not one computation reported
   twice.

## 2. The 41 duplicate images

True-grade distribution: `{0: 5, 1: 6, 2: 21, 3: 1, 4: 8}` — matches the previously reported and
expected distribution exactly. One validation image (`51131b48f9d4`) has two distinct training
twins (both grade 2); every other duplicate has exactly one training twin. All 41 are present in
both prediction CSVs.

**9 of the 41 (22%) have a training twin with a *different* label than the validation image's own
ground truth** — the same 9 identified previously: `14e3f84445f7` (val 3, twin 4),
`4a44cc840ebe` (val 2, twin 3), `51131b48f9d4` (val 1, twins 2, 2), `7005be54cab1` (val 1, twin 4),
`80964d8e0863` (val 4, twin 2), `8446826853d0` (val 2, twin 1), `b9127e38d9b9` (val 2, twin 4),
`f03d3c4ce7fb` (val 4, twin 2), `f066db7a2efe` (val 0, twin 1). A model that simply reproduced its
training twin's label would be *wrong* on these 9 by construction, regardless of which architecture
it is.

| val_id | true grade | train twin(s) [grade] | RACAF pred | NO-RACAF pred |
|---|---|---|---|---|
| `034cb07a550f` | 4 | `c8d2d32f7f29` [4] | 3 | 4 |
| `04ac765f91a1` | 1 | `3044022c6969` [1] | 1 | 1 |
| `05a5183c92d0` | 1 | `63a03880939c` [1] | 1 | 1 |
| `1006345f70b7` | 2 | `435d900fa7b2` [2] | 2 | 2 |
| `14e3f84445f7` | 3 | `f0f89314e860` [4] | 4 | 4 |
| `1638404f385c` | 4 | `576e189d23d4` [4] | 2 | 1 |
| `1a1b4b2450ca` | 2 | `92b0d27fc0ec` [2] | 2 | 2 |
| `23d7ca170bdb` | 2 | `ea9e0fb6fb0b` [2] | 1 | 2 |
| `38487e1a5b1f` | 2 | `b376def52ccc` [2] | 2 | 2 |
| `3f44d749cd0b` | 0 | `f9e1c439d4c8` [0] | 1 | 1 |
| `4a44cc840ebe` | 2 | `0cb14014117d` [3] | 2 | 2 |
| `51131b48f9d4` | 1 | `42a850acd2ac`, `8cb6b0efaaac` [2, 2] | 2 | 1 |
| `65c958379680` | 4 | `11242a67122d` [4] | 2 | 2 |
| `6c3745a222da` | 4 | `eadc57064154` [4] | 2 | 2 |
| `6cb98da77e3e` | 2 | `48c49f662f7d` [2] | 2 | 2 |
| `7005be54cab1` | 1 | `3ee4841936ef` [4] | 1 | 1 |
| `75a7bc945b7d` | 2 | `98104c8c67eb` [2] | 2 | 2 |
| `7a0cff4c24b2` | 2 | `86baef833ae0` [2] | 2 | 2 |
| `80964d8e0863` | 4 | `ab50123abadb` [2] | 3 | 4 |
| `8446826853d0` | 2 | `8ef2eb8c51c4` [1] | 2 | 3 |
| `8d7bb0649a02` | 2 | `7550966ef777` [2] | 2 | 2 |
| `8fc09fecd22f` | 1 | `d1cad012a254` [1] | 1 | 1 |
| `91cbe1c775ef` | 2 | `7d261f986bef` [2] | 2 | 2 |
| `9b32e8ef0ca0` | 2 | `a15652b22ab8` [2] | 1 | 2 |
| `9f4132bd6ed6` | 2 | `9b7b6e4db1d5` [2] | 2 | 3 |
| `a3b2e93d058b` | 2 | `3fd7df6099e3` [2] | 2 | 2 |
| `a4012932e18d` | 0 | `906d02fb822d` [0] | 2 | 0 |
| `a8b637abd96b` | 0 | `e2c3b037413b` [0] | 1 | 1 |
| `b8ac328009e0` | 2 | `ff0740cb484a` [2] | 2 | 2 |
| `b9127e38d9b9` | 2 | `e39b627cf648` [4] | 2 | 2 |
| `ba2624883599` | 2 | `14515b8f19b6` [2] | 2 | 3 |
| `c9f0dc2c8b43` | 2 | `9c5dd3612f0c` [2] | 3 | 3 |
| `cac40227d3b2` | 2 | `0161338f53cc` [2] | 2 | 2 |
| `d28bd830c171` | 2 | `f9ecf1795804` [2] | 1 | 2 |
| `d51b3fe0fa1b` | 4 | `df4913ca3712` [4] | 2 | 4 |
| `d51c2153d151` | 0 | `7b691d9ced34` [0] | 1 | 1 |
| `d801c0a66738` | 1 | `68332fdcaa70` [1] | 1 | 1 |
| `d85ea1220a03` | 4 | `bfefa7344e7d` [4] | 2 | 3 |
| `f03d3c4ce7fb` | 4 | `9a3c03a5ad0f` [2] | 2 | 2 |
| `f066db7a2efe` | 0 | `278aa860dffd` [1] | 0 | 1 |
| `f920ccd926db` | 2 | `bcdc8db5423b` [2] | 2 | 2 |

Neither model reproduces its training twin's label reliably even when the twin's label matches the
validation ground truth (e.g. `65c958379680`, `6c3745a222da`, `d51b3fe0fa1b`: same-label grade-4
twins, both models predict 2). This is evidence against simple memorisation as the explanation for
either model's behaviour on this subset.

## 3. Duplicate-subset vs non-duplicate metrics

| Subset | n | RACAF QWK | RACAF acc | RACAF MAE | NO-RACAF QWK | NO-RACAF acc | NO-RACAF MAE | ΔQWK |
|---|---|---|---|---|---|---|---|---|
| Duplicate (41) | 41 | 0.5208 | 0.5610 (23/41) | 0.6098 | 0.6737 | 0.6585 (27/41) | 0.4634 | **+0.1529** |
| Non-duplicate (689) | 689 | 0.7656 | 0.7489 | 0.3672 | 0.8485 | 0.7692 | 0.3019 | +0.0829 |
| All (730) | 730 | 0.7633 | 0.7384 | 0.3808 | 0.8461 | 0.7630 | 0.3110 | +0.0828 |

**Both models perform markedly worse on the 41 duplicates than on the rest of the validation set**
(RACAF QWK 0.52 vs 0.77; NO-RACAF QWK 0.67 vs 0.85; accuracy 56–66% vs 74–77%). This is consistent
with §2: 21/41 (51%) of the duplicates are grade 2 — the most confusable class for both models — and
9/41 (22%) carry a conflicting train/val label that caps achievable accuracy regardless of
architecture. **Neither model shows the inflated, near-perfect accuracy that pure memorisation of
an identical training image would predict.**

NO-RACAF's ΔQWK is *larger* within the duplicate subset (+0.1529) than in the non-duplicate subset
(+0.0829) or the full set (+0.0828). Because the duplicate subset is only 41/730 (5.6%) of the
population, this does not materially change the aggregate figure (§4) — but it is reported here
without adjustment, exactly as observed, per Part E of the audit's own instructions: this
subset-level pattern is descriptive and does **not** establish that duplication favours NO-RACAF
causally; it is equally consistent with the duplicated images (grade-2-heavy, partly mislabeled
relative to their twin) simply being a harder-than-average sample on which NO-RACAF's higher
severe/mid-grade recall (see the main analysis) generalises the same way it does elsewhere.

## 4. Exact duplicate-excluded comparison (n = 689)

Removing the **same** 41 IDs from both models' predictions (never a different 41 per model):

| Metric | RACAF | NO-RACAF | Δ (NO − RACAF) |
|---|---|---|---|
| QWK | 0.7656318 | 0.8485222 | **+0.0828904** |
| Accuracy | 0.7489115 | 0.7692308 | +0.0203193 |
| Balanced accuracy | 0.4874637 | 0.5947067 | +0.1072430 |
| Macro F1 | 0.4861897 | 0.5795934 | +0.0934037 |
| MAE | 0.3671988 | 0.3018868 | −0.0653120 |

**Paired bootstrap of the duplicate-excluded ΔQWK** (5,000 resamples, deterministic seed
20260913, same resampled indices applied to both models each replicate): mean **+0.0828047**, 95%
CI **[+0.0453, +0.1232]**, P(Δ ≤ 0) = **0.0000** (0/5,000 replicates), 0 invalid replicates.

**Comparison with the all-image result:** full-population ΔQWK +0.0828302, 95% CI [+0.0460,
+0.1202], P(Δ ≤ 0) = 0.0000 (0/5,000). Excluding the duplicates shifts ΔQWK by **+0.0000602**
(≈ +0.0001) — the advantage is essentially unchanged, and if anything infinitesimally larger. The
95% CI's lower bound moves from +0.0460 to +0.0453 (wider on that side, because n drops from 730 to
689), but both intervals cleanly exclude zero.

## 5. Paired discordance (correctness only, not QWK)

|  | NO-RACAF correct | NO-RACAF incorrect |
|---|---|---|
| **RACAF correct** | 488 | 51 |
| **RACAF incorrect** | 69 | 122 |

Improved by NO-RACAF (RACAF wrong → NO-RACAF right): 69. Worsened (RACAF right → NO-RACAF wrong):
51. Net: +18 images (+2.47 pp accuracy, matching the accuracy delta above exactly). **Exact
two-sided binomial McNemar test on the 120 discordant pairs: p = 0.1203** — the exact-grade
correctness swing alone is *not* statistically significant at α = 0.05, even though the QWK
difference is (§4). This is expected and not a contradiction: QWK weights the *distance* of an
error quadratically, so NO-RACAF's advantage is driven more by converting large ordinal errors into
small ones (§ "Confusion-matrix analysis" in the main report) than by a large net swing in exact
correctness.

## 6. Cross-check against the previously reported (confusion-matrix-derived) values

Every headline figure in [RACAF_vs_NO_RACAF_Analysis.md](RACAF_vs_NO_RACAF_Analysis.md) and
[RACAF_Ablation_Experiment_1_Report.md](RACAF_Ablation_Experiment_1_Report.md) — both QWKs,
balanced accuracy, macro F1, grade-3/4 recall, ΔQWK, and both confusion matrices — was recomputed
directly from the per-image CSVs and matched the previously recorded values to within floating-point
rounding (largest discrepancy 6.9e-06 on RACAF macro F1; both confusion matrices identical
cell-for-cell). **No error, inconsistency, or data problem was found** in this audit: file
identification succeeded unambiguously by config-hash provenance, the ID alignment was exact,
ground truth agreed for all 730 images and matched the split manifest, the duplicate list was
independently reproduced twice with an identical result, and every historical figure it could check
reproduced exactly.

## 7. Conclusion

**The 41 train/validation duplicate images do not explain, and are not required for, the NO-RACAF
advantage found in the main ablation analysis.** The exact duplicate-excluded ΔQWK (+0.0829, 95% CI
excluding zero) is statistically indistinguishable from the all-image ΔQWK (+0.0828, 95% CI
excluding zero). This directly resolves the open item in
[RACAF_vs_NO_RACAF_Analysis.md](RACAF_vs_NO_RACAF_Analysis.md) §9 (v1): the theoretical "adversarial
allocation" that could have erased the gap under the confusion-matrix-based sensitivity scenarios
does not correspond to what the actual per-image data show.

**What remains unresolved.** This is still a comparison of **one training run per architecture**.
The paired bootstrap here quantifies uncertainty from *which 730 (or 689) validation images* were
evaluated — it says nothing about training-seed variance, and the ablation's own overfitting
(higher validation loss, worse calibration; see the main analysis §6, §11) is a separate, unresolved
caveat on any architectural claim. A multi-seed repeat remains the recommended next step before any
claim that RACAF is not a beneficial component in general.

---

## 8. Provenance

| | |
|---|---|
| Analysis type | Read-only, single self-contained Colab cell; no training, no inference, no model load, no file writes |
| Executed | 2026-09-13 |
| Inputs | `experiments/FinalClassification/2026-09-12_02-45-05/evaluation/per_sample_corn_predictions.csv` (RACAF); `experiments/FinalClassification/2026-09-13_04-11-07/evaluation/per_sample_corn_predictions.csv` (NO-RACAF); `dataset_splits/aptos2019_train_val_split.csv`; `datasets/APTOS2019/raw/train_images/` |
| Duplicate method | byte-size prefilter → MD5 → full byte comparison (`filecmp.cmp`, `shallow=False`) |
| Bootstrap | 5,000 paired resamples, seed 20260913, `cohen_kappa_score(y_true, y_pred, weights="quadratic")` |
| McNemar | exact two-sided binomial test on discordant pairs (`scipy.stats.binomtest`) |
| Cross-checked against | the earlier independent content-hash audit (`dup41.json`) — identical 41-ID result |
| Repository state | no `.py`, notebook, checkpoint, or cache file modified by this audit |
