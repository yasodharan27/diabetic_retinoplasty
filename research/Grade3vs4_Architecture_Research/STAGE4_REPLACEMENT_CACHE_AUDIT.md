# Stage 4 replacement — cache dependency and isolation audit (2026-10-01)

This is an audit only: nothing was implemented, trained, deleted, moved or regenerated.
- **Sources:** the code (`lesion_segmentation_model.py`, `local_feature_extraction_dataset.py` (lfed),
  `joint_training_dataset.py` (jtd), `racaf.py`, `improved_training_data.py`, `joint_cache_archive.py`,
  `joint_cache_diagnostics.py`, `joint_cache_staging.py`, `icdr_two_route_experiment.py`,
  `pl_convnext.py`, `config.py`, `colab/common/*`), the notebooks, and a Drive inventory via rclone.

## 1. Current Stage 4 → cache dependency graph

```
exported_models/LesionSegmentation/best_model.keras   (OLD Stage 4; sha256 64b3c0468bdc…ce76; 157.5 MB)
  │  default path everywhere: lesion_segmentation_model.DEFAULT_MODEL_PATH
  │    → lfed.DEFAULT_LESION_MODEL_PATH → racaf.load_frozen_stage4_model() → jtd resolved_stage4_model
  │
  ├─ jtd._get_or_compute_joint_frozen_outputs()   [Stage 3 + OLD Stage 4 run together, cached by NAME only]
  │     ├─ vessel_map  = resize(LWNet(native RGB))                → APTOS_<id>_vessel_512x512.npy   (Stage 3 only)
  │     ├─ lesion_maps = OLD Stage 4 identity TTA view            → APTOS_<id>_lesion_512x512.npy   (OLD Stage 4)
  │     └─ kappa, r    = racaf.compute_reliability(4 TTA views)   → APTOS_<id>_racaf_reliability.npz (OLD Stage 4)
  ├─ jtd._get_or_compute_canonical_rgb()                          → APTOS_<id>_rgb_512x512.npy      (Stage 2 only)
  │
  ├─ Drive loose persistent cache:
  │     cache/LocalFeatureExtraction/   3,651 lesion (OLD) · 3,651 vessel · 3,662 rgb     (flat)
  │     cache/RACAF/                    3,651 racaf_reliability.npz (OLD)
  ├─ Drive archive: cache/cache_archive/cache_shard_000…007.tar (28.5 GB, 14,604 files = 3,651 × 4 kinds;
  │     EVERY shard mixes OLD lesion + OLD reliability with vessel + rgb; no manifest file on Drive)
  ├─ Local runtime (after extraction): /content/cache/local_feature_extraction, /content/cache/racaf,
  │     marker /content/cache/.multiseed_archive_extracted.json   (shared by 5+ notebooks)
  │
  └─ consumers:
        improved_training_data.missing_local_artifacts / locally_cached_entries / load_cached_sample
          → jtd._build_joint_sample (stage5_input = rgb ‖ vessel ‖ lesion, + reliability scalar)
        icdr_two_route_experiment.ensure_local_cache / prepare_population
        joint_cache_archive.index_persistent_cache / build_archive / extract_archive
        joint_cache_staging, complete_local_cache (can GENERATE missing entries with the default = OLD Stage 4)
        joint_cache_diagnostics.artifact_paths
        notebooks: multiseed_racaf_improved_training, pl_convnext_priors_experiment,
                   stage08_corn_classifier, grade4_aux_loss_experiment, stage08_icdr_two_route_head
        research artefacts derived from OLD lesion maps (S4 / S5 burden CSVs, Phase-0 lesion statistics,
                   label-noise audit statistics, PL / RACAF / NO_RACAF / C1 / G0 runs)
```

**Root hazards.**
- **H1 — name-only cache identity.** `lfed._cache_path` = `APTOS_{id}_{kind}_{H}x{W}.npy`. The key has no
  model identity, so any file named `…_lesion_512x512.npy` is accepted as "the" lesion map, whichever
  Stage 4 produced it.
- **H2 — silent defaults.** Every loader defaults to the OLD checkpoint path. Any code path that
  computes Stage 4 without an explicit path uses the OLD model.
- **H3 — persistent fallback.** A missing local file is silently filled from the Drive loose cache
  (OLD lesion files) and mirrored locally (`_load_persistent_array`).
- **H4 — mixed archive.** Extracting any legacy shard writes OLD lesion and reliability files into the
  shared local directory. The marker file makes later notebooks skip re-checking.
- **H5 — shared local paths.** `/content/cache/local_feature_extraction` is used by every notebook. A
  runtime that ever ran an old notebook holds OLD lesion files under the canonical names.
- **H6 — the reliability contract.** `_build_joint_sample` always emits the OLD-Stage-4 reliability
  scalar.

## 2. Old artefacts that must never be reused by the new pipeline

| Artefact | Location | Why |
|---|---|---|
| OLD Stage 4 checkpoint (**deny-list sha256 `64b3c0468bdc77204e7126c4ff06aa3fbbdab1430349217e5d6baaa69888ce76`**) | `exported_models/LesionSegmentation/best_model.keras` (+ `training_run/`) | Replaced model |
| OLD lesion maps | `cache/LocalFeatureExtraction/APTOS_*_lesion_512x512.npy` (3,651) | OLD Stage 4 output |
| OLD RACAF reliability | `cache/RACAF/APTOS_*_racaf_reliability.npz` (3,651) | Derived from OLD Stage 4 TTA; RACAF is closed |
| Legacy archive shards | `cache/cache_archive/cache_shard_000…007.tar` | Contain OLD lesion + reliability (their vessel/rgb members are reusable — §6) |
| Local legacy dirs and marker | `/content/cache/local_feature_extraction`, `/content/cache/racaf`, `/content/cache/.multiseed_archive_extracted.json` | May hold OLD lesion files |
| Features or statistics computed from OLD lesion maps | S4/S5 burden CSVs, Phase-0 `lesion_statistics_grade34.csv`, label-noise audit statistics, PL / RACAF / NO_RACAF / C1 / G0 checkpoints | Historical results only |

## 3. Recommended new cache structure and naming (a separate generation namespace)

```
Drive:  DiabeticRetinopathy/
  exported_models/LesionSegmentation_v2/<train_stamp>/
      best_model.keras
      stage4_v2_manifest.json   (sha256, arch, encoder, input contract, training datasets + counts,
                                 IDRiD-test metrics, git commit, Stage 3 sha used for training inputs)
  cache/Stage34_v2/<GEN_ID>/          GEN_ID = "s4v2-<first 12 of Stage-4 v2 sha256>"
      generation_manifest.json  (GEN_ID, stage4_sha256, stage3_sha256 = 91f0cada4b26…18de1,
                                 input contract, image_size 512, population ids + split sha bc80fd45…,
                                 per-file sha256 list, created, git commit, deny-list sha)
      lesion/  APTOS_<id>_lesion-s4v2_512x512.npz   keys: maps (512,512,4) float16/32,
                                                          stage4_sha256, stage3_sha256, gen_id
      vessel/  APTOS_<id>_vessel_512x512.npy          (legacy Stage-3 values, re-homed; see §6)
      rgb/     APTOS_<id>_rgb_512x512.npy             (legacy Stage-2 values, re-homed; see §6)
      archive/ s34v2_shard_000….tar + archive_manifest.json   (members only from this generation)
Local:  /content/cache_s34v2/<GEN_ID>/{lesion,vessel,rgb}/   marker /content/cache_s34v2/<GEN_ID>/.extracted.json
```

- A different **kind string** (`lesion-s4v2`), a different **file type** (`.npz` with identity keys) and
  a different **root** (`Stage34_v2/<GEN_ID>`). Old code can never find the new files, and new code can
  never match an old filename.
- **No reliability artefact.** RACAF is closed, and the new sample contract has no reliability scalar.
- A new Stage 4 checkpoint produces a new `GEN_ID`, and therefore a new directory. Generations cannot
  mix.

## 4. Artefacts that must be regenerated

1. **Stage 4 v2 checkpoint** (training).
2. **Stage 4 v2 lesion maps** for the 3,651 cached APTOS images (2,921 training + 730 validation),
   single pass, no TTA. About 20–40 min on a T4.
3. **The Stage34_v2 generation manifest and archive** (vessel + rgb re-homed, lesion v2).
4. **Stage 3 vessel maps for the *new Stage-4 training images*** (Retinal-Lesions, MAPLES-DR), if Stage 4
   v2 keeps the RGB + vessel input. Namespaced per dataset.
5. **Later:** IDRiD-test Stage 3/4-v2 maps for the single-use external evaluation (a namespaced
   `IDRiDtest_*` generation).
6. **All downstream Stage 5–8 models** (Architecture 1) — trained only on the v2 generation.

## 5. Artefacts that can remain unchanged (reused, with verification)

- **Stage 2 canonical RGB** (`*_rgb_512x512.npy`). It depends only on Stage 2; recomputation parity was
  already verified (max |Δ| 0.0, §31 extraction).
- **Stage 3 vessel maps** (`*_vessel_512x512.npy`).
  - They depend only on LWNet (`91f0cada…`) and the native RGB: `prepare_stage4_input`'s channel 3 is
    computed before any Stage 4 call.
  - Reuse them after a spot-check: recompute Stage 3 for K = 25 random images and require
    max |Δ| ≤ 1e-5.
- **Stage 3 LWNet checkpoint** (frozen); the **IDRiD vessel cache** used for Stage 4 training inputs.
- **Split, population and the 11 empty-FOV exclusions**, pinned as before.

## 6. Preserving the old caches (no delete, move or modify)

- New code **never writes under** `cache/LocalFeatureExtraction`, `cache/RACAF`, `cache/cache_archive`,
  `exported_models/LesionSegmentation` or `/content/cache/{local_feature_extraction,racaf}`. Every writer
  asserts its destination is under a v2 root.
- Vessel and rgb are **copied** into the v2 generation. The recommended source is streaming the legacy
  shards with a member-name filter (`*_vessel_*`, `*_rgb_*` only): tar members are *read*; OLD lesion
  and reliability members are skipped by name and never written.
- Record `pl.directory_fingerprint` (path, size, mtime) of every legacy directory before and after each
  v2 job, and assert equality.

## 7. Can the existing archive infrastructure be used?

**Not unchanged.** `joint_cache_archive` hard-codes the artefact kinds {vessel, lesion, rgb} + RACAF
reliability, the shard name `cache_shard_%03d.tar`, `ARCHIVE_DIR_NAME = "cache_archive"`, and extraction
into the shared local dirs.

Reuse its streaming, temp-dir and validation helpers, but add:
- an `artifact_kinds` / member-filter parameter (so legacy shards can be read for vessel/rgb only);
- a generation-scoped archive dir, shard prefix (`s34v2_shard_`) and **`archive_manifest.json`**
  (gen id, per-member sha256);
- extraction that verifies every lesion member's embedded `stage4_sha256` against the manifest.

## 8. Checkpoint identity to record

- **In the Stage 4 v2 manifest:** checkpoint sha256; the encoder name and its pretrained-weights sha;
  input contract (channels, 512, normalisation); training datasets with image counts and hashes; IDRiD
  test metrics; git commit.
- **In every lesion file:** `stage4_sha256`, `stage3_sha256`, `gen_id`.
- **In every downstream run manifest:** `stage4_sha256`, `stage3_sha256`, `gen_id`, the generation
  manifest sha. Training **refuses to start** if `stage4_sha256` equals the deny-listed OLD sha.

## 9. Safety checks (fail loudly)

1. **Checkpoint loader** `load_stage4_v2(path, expected_sha256)`. No default path. It raises if:
   - the sha differs from the expected one;
   - the path lies under `exported_models/LesionSegmentation/`;
   - the sha equals the deny-list `64b3c046…`.
2. **Cache reader** accepts only `.npz` files named `*_lesion-s4v2_*`, with keys
   {maps, stage4_sha256, stage3_sha256, gen_id} equal to the active generation. It raises
   `LegacyStage4CacheError` on any `*_lesion_512x512.npy`, missing key or mismatch.
3. **Directory guard.** The v2 cache and local roots must not equal or lie inside any legacy directory.
   It raises at import/config time.
4. **No legacy fallback.** The v2 path has no persistent fallback to legacy dirs, and never calls
   `jtd._get_or_compute_joint_frozen_outputs`, `jtd._build_joint_sample`, `ensure_local_cache`,
   `complete_local_cache` or `lfed._cache_path`.
5. **Generation completeness.** The population count equals the manifest count; per-file sha256 list
   matches; the split sha is `bc80fd45…`.
6. **Freshness canary (start of every training run).** Recompute Stage 4 v2 on K = 8 fixed images and
   require max |Δ| ≤ 1e-3 against the cache. This proves the cache is the new model's output.
7. **Non-legacy canary (once per generation).** For K = 25 images, the v2 maps must differ from the
   legacy maps: mean |Δ| above a pre-set floor. The legacy files are read-only, and read only for this
   check.
8. **Legacy fingerprint** unchanged before and after every v2 job (§6).
9. **Unit tests** covering:
   - a sample built from a dir containing only legacy files raises;
   - loading the old checkpoint raises;
   - a mismatched `gen_id` raises;
   - a writer targeting a legacy dir raises.

## 10. Code, notebook and config changes

- **`config.py`, `colab/common/colab_config.py` (+ `drive_paths`):** add `LESION_SEG_V2_MODEL_DIR`,
  `STAGE34_V2_CACHE_ROOT`, `LOCAL_STAGE34_V2_ROOT`, `LEGACY_STAGE4_SHA256` (deny-list) and
  `STAGE3_SHA256`. Keep the old constants untouched.
- **New `stage4_v2.py`:** a pretrained-encoder U-Net builder, training, inference, and
  `load_stage4_v2`. `lesion_segmentation_model.py` stays untouched (historical).
- **New `stage34_cache_v2.py`:** path scheme, npz writer and reader with identity, manifest
  build/validate, directory guards, both canaries, and legacy-fingerprint helpers.
- **`joint_cache_archive.py`:** add *optional* parameters (artifact kinds, member filter, archive dir,
  shard prefix, manifest), with defaults preserving legacy behaviour exactly — or a thin
  `archive_v2.py` reusing its helpers.
- **New `arch1_data.py`:** builds the Architecture-1 sample (rgb ‖ vessel ‖ lesion-v2, no reliability)
  from the v2 generation only.
- **Notebooks (new; old ones untouched):**
  - `stage04_v2_lesion_segmentation.ipynb` (train + IDRiD-test gate + export);
  - `stage34_v2_cache_generation.ipynb` (filtered legacy vessel/rgb re-home + Stage 4 v2 inference +
    manifest + archive + canaries);
  - later, the Architecture-1 training notebook.
- **Tests:** `tests/test_stage34_cache_v2.py` (guards), `tests/test_stage4_v2.py`.
- **Docs, after validation:** `SEGMENTATION_ARCHITECTURE.md`, `PROJECT_CODE.md` (Stage 4 v2; the legacy
  generation marked historical).

## 11. Implementation plan (in order; each step verified before the next)

1. **Freeze legacy identities:** record the deny-list sha, the LWNet sha and legacy directory
   fingerprints in a committed constants file.
2. **Implement** `stage34_cache_v2.py` + guards + tests, and `stage4_v2.py`. Laptop only; run the tests.
3. **Train Stage 4 v2 (Colab).**
   - Data: IDRiD train (54) + Retinal-Lesions (1,593; request) ± MAPLES-DR (200).
   - Excluded: DDR (reserved) and the IDRiD test set.
   - Gate on IDRiD-test Dice/AUPR against the legacy mean Dice 0.131, with a pre-set bar.
   - Export to `LesionSegmentation_v2/<stamp>/` with its manifest.
4. **Generate the v2 generation (Colab).**
   - Stream the legacy shards with a vessel/rgb-only filter into the local v2 root.
   - Spot-check Stage 3 parity.
   - Run Stage 4 v2 on all 3,651 images; write identity npz files.
   - Build the generation manifest; pack the v2 archive to Drive.
   - Run the non-legacy canary and the legacy fingerprint check.
5. **Only then** train the Stage 5–8 Architecture 1 on `GEN_ID`, with the freshness canary at every run
   start.

**Consequence for the §36 plan.** Its **E1** (Architecture 1 on the *old* Stage 3/4 caches) conflicts
with this rule for the final pipeline. It is either **dropped**, or run only as an explicitly labelled,
non-final mechanism diagnostic in the legacy namespace. The recommended order becomes: Stage 4 v2
(train + gate) → v2 generation → Architecture 1 on v2 → soft targets → IDRiD external evaluation.
