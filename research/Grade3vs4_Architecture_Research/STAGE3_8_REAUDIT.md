# Stage 3–8 redesign — re-audit before implementation (2026-10-01)

Audit only: nothing was trained, regenerated, modified, moved or committed.

**Tags:**
- **[V]** verified in the repository or on Drive in this session.
- **[L]** literature, with source.
- **[I]** my inference or recommendation.

**User note (2026-10-01):** new datasets may be used if they are downloaded and placed on Drive; the
project is not restricted to IDRiD.

---

## 1. Stage 3 — vessel

- **[V] Model:** LWNet `wnet` (in_c = 3, n_classes = 1), vendored and inference only.
  - Checkpoint `best_model.pth`, sha256 **`91f0cada4b26ece63464b05be60f9b3a51f1bcd1f764081d07d3a35961118de1`**,
    identical in `models/vessel_segmentation/` and `exported_models/VesselSegmentation/` on Drive.
  - `config.VESSEL_SEG_TTA` defaults to True.
- **[V] Output:** `predict_vessel_mask()` runs an FOV circle-fit, crops, resizes to the 512 working size,
  applies optional TTA, and pastes back a native-resolution probability map in [0, 1].
- **[V] Cached value:** `APTOS_<id>_vessel_512x512.npy`, 3,651 files in `cache/LocalFeatureExtraction/`
  (plus copies inside the legacy archive shards).
  - It is channel 3 of `racaf.prepare_stage4_input(native_rgb, native_vessel, (512, 512))`: a joint
    4-channel anti-aliased resize.
  - skimage applies no smoothing across the channel axis, so the vessel channel depends only on the
    vessel map.
- **[V] Independence from Stage 4:** in `jtd._get_or_compute_joint_frozen_outputs`, the vessel map is
  computed and resized *before* `racaf.tta_views(stage4_model, …)`. No Stage-4 value enters it. It depends
  on LWNet (`91f0cada…`), the Stage-2 native RGB and `VESSEL_SEG_TTA`.
- **[I] Reuse:** yes — do not regenerate Stage 3. Reuse is conditional on a parity check (below).
- **Parity check (exact):**
  1. Choose K = 25 image ids with a fixed seed from the 3,651 population.
  2. For each: `raw = lfed._load_raw_bgr(lfed.DEFAULT_TRAIN_IMAGE_DIR, id)`;
     `nat = lfed._resolve_processed_rgb(raw, lfed.DEFAULT_PROCESSED_DIR, id)`;
     `v = predict_vessel_mask(nat, model=<LWNet loaded after its sha is asserted>)["probability_map"]`;
     `ref = racaf.prepare_stage4_input(nat, v, (512, 512))[0, ..., 3:4]`.
  3. Require `max|ref − cached| ≤ 1e-4` (GPU non-determinism allowance) for all 25, with
     `VESSEL_SEG_TTA` at its default (True).
  4. Any failure stops the pipeline; there is no silent regeneration.
- **Where Stage 3 enters the new pipeline:** a **Stage-3 generation** `cache/Stage3/s3-91f0cada/vessel/…`
  (copies of the verified files), consumed directly by the Stage 5 prior encoder.
  - [I] Stage 3 **no longer feeds Stage 4.** The v2 Stage 4 is RGB-only (§2).
  - Stage 3's downstream contribution is *measured* (§7), not assumed.

## 2. Stage 4 — encoder selection

**[L] Evidence on IDRiD (area under the precision-recall curve, IDRiD protocol).**

| method | EX | HE | SE | MA | mean | setting |
|---|---|---|---|---|---|---|
| HRNetV2 + HRDecoder (MICCAI 2024) | 0.876 | 0.708 | 0.777 | 0.492 | **0.713** | mmseg; 1440×960 training, 2880×1920 inference; 10–22 GB |
| ConvNeXt backbone (same table) | — | — | — | — | 0.699 | mmseg (decoder not stated in the summary) |
| M2MRF (HRNet-W48) | — | — | — | — | 0.672 | mmseg 0.8 / PyTorch 1.6 |
| **SE-ResNet-101 U-Net (Playout & Cheriet 2024)** | 0.825 | 0.620 | 0.713 | 0.451 | **0.652** | **plain U-Net; 1,536 px; random 512 px patches; IDRiD-only training; won a 1,000-trial search optimised for cross-dataset generalisation** |
| *Our current Stage 4* | Dice 0.357 | 0.127 | 0.024 | 0.017 | Dice 0.131 | Attention U-Net from scratch, **512 px input**, 54 images |

**[I] Selected encoder: SE-ResNet-101 (ImageNet) in a U-Net.** Why this one:
- It is the only candidate with a published result from exactly the simple recipe we can reproduce: a
  U-Net at 1,536 px trained on 512 patches, from IDRiD alone.
- It was selected *for cross-dataset generalisation*. Our deployment domain (APTOS) is a different
  dataset.
- It is available in MIT-licensed `segmentation_models_pytorch` (not currently installed; install
  pinned on Colab).
- It is T4-feasible.
- HRNet/HRDecoder and the ConvNeXt entry score 0.05–0.06 higher mean AUPR, but need mmsegmentation and
  custom decoders. The table differences rest on only 27 test images (noise of roughly ±0.03).
- It is deliberately **not** the same encoder as the Stage 6 grading trunk: the choice is made on
  segmentation evidence, not convenience.

**Specification:**
- `smp.Unet(encoder_name="se_resnet101", encoder_weights="imagenet", classes=4)`, sigmoid multi-label
  output (MA, HE, EX, SE).
- About 51 M params [I; confirm at build]. The encoder is **fine-tuned**, with a lower encoder LR.
- **Input: RGB only** (Stage-2 processed, FOV-cropped, resized to 1,536²). Train on random 512 px
  patches; infer on the full 1,536² image.
- Output downsampled to 512² with the legacy anti-aliased bilinear convention for the cache.
  [I] MA maps are lossy at 512; this matches the resolution of the Stage 5–7 input.
- Loss: BCE + Dice.
- **Expected T4 cost:** about 1.5–3 h for IDRiD-only training; about 3–5 h with added datasets.
  Inference on 3,651 APTOS images about 20–30 min.

**Caution.** Better Dice or AUPR is a *segmentation gate only*. It does not show that grading improves;
that is tested downstream (gates C2 and D).

## 3. Stage 4 datasets and access

| dataset | status | images / classes | split | overlap / leakage | access / licence |
|---|---|---|---|---|---|
| **IDRiD segmentation** | **[V] on Drive** | 81 JPG (4288×2848) + TIF masks. Train 54 (MA 54, HE 53, EX 54, SE 26); test 27 (MA 27, HE 27, EX 27, SE 14); + optic disc | official 54/27; [I] internal validation = 10 of the 54 (fixed seed, stratified by SE presence); **test 27 held out for the gate only** | [V] 0 of 81 appear in the IDRiD grading test set (content hash, IDRiD report). APTOS overlap *not yet checked* → hash check (§6) | already in use in the project |
| **DDR lesion subset** | [L] public; **not on Drive** | 757 images; MA / HE / EX / SE (the same 4 classes) | official train/val/test [L; exact split to verify on download] | **DDR was reserved as an external grading set.** Training Stage 4 on it removes DDR's independence for the full pipeline | [L] CC BY 4.0 (HF card) |
| **Retinal-Lesions** | [L] free on request; **not on Drive** | 1,593 EyePACS images; MA, intraretinal HE, EX, CWS (+ NV, preretinal/vitreous haemorrhage, fibrous proliferation) | none official [L]; make a fixed split | EyePACS-sourced, so not APTOS [L]; carries DR grades (irrelevant to APTOS) | Google-form request; **licence terms unverified** |
| **MAPLES-DR** | [L]; **not on Drive** | 198 Messidor images (138 train / 60 test); MA, HE, exudates, CWS, NV, vessels, … | official | Messidor (France) | [L] annotations CC-BY on figshare; **images require a MESSIDOR consortium request** |
| **FGADR Seg-set** | agreement **pending (user)** | 1,842; MA / HE / EX / SE / IRMA / NV | none official | — | signed agreement; clause 6 "no derivative works" to be clarified |

**[I] Data plan.**
- **Stage 4 v2-a:** IDRiD only — available today, with a published recipe at 0.652 mean AUPR.
- **Request now, in parallel:** Retinal-Lesions (form) and MAPLES-DR (annotations + MESSIDOR images).
  Add them as **v2-b** only if v2-a passes the gates, or fails on generalisation.
- **DDR:** its 757 lesion images are the easiest download, but **using them costs DDR as an independent
  external set.** That is a user decision; my recommendation is not to use DDR.

## 4–5. Caches: never reuse, new structure, dependency graph

**Never reuse** (as in §37, verified):
- the OLD Stage 4 checkpoint `64b3c0468bdc…ce76`;
- `cache/LocalFeatureExtraction/*_lesion_512x512.npy` (3,651);
- `cache/RACAF/*_racaf_reliability.npz` (3,651);
- `cache/cache_archive/cache_shard_000–007.tar` (they contain old lesion and reliability members);
- local `/content/cache/{local_feature_extraction, racaf}` and its marker;
- every result derived from old lesion maps.

**Remain untouched and reused (read-only):**
- `cache/LocalFeatureExtraction/*_vessel_512x512.npy` (3,651) and `*_rgb_512x512.npy` (3,662);
- the LWNet checkpoint;
- the split, population and exclusions.

**Regenerate:** the Stage 4 v2 checkpoint; the lesion-v2 maps for 3,651 APTOS images (and later the
IDRiD grading test); the v2 manifests; all Stage 5–8 models.

**[I] Namespaces** (Stage 3 and Stage 4 are separate generations, so Stage 3 is never recreated when
Stage 4 changes):

```
cache/Stage2/rgb-v1/APTOS_<id>_rgb_512x512.npy                   (verified copies; parity Δ 0.0 already shown §31)
cache/Stage3/s3-91f0cada/vessel/APTOS_<id>_vessel_512x512.npy    (verified copies; parity check §1)
    stage3_manifest.json {lwnet_sha256, VESSEL_SEG_TTA, split sha bc80fd45…, population, per-file sha256}
exported_models/LesionSegmentation_v2/<stamp>/{best_model.pt, stage4_v2_manifest.json}
cache/Stage4/s4v2-<sha12>/lesion/APTOS_<id>_lesion-s4v2_512x512.npz  {maps(512,512,4), stage4_sha256, gen_id}
    stage4_manifest.json {stage4_sha256, encoder, input contract (RGB-only, 1536→512), training datasets+counts,
                          IDRiD-test metrics, split sha, population, per-file sha256, legacy deny-list sha}
cache/Stage34_bundle/<s3-id>__<s4-id>/archive/*.tar + bundle_manifest.json   (optional fast-load archive;
                          members from these two generations only)
```

- [I] Vessel and RGB copies are made from the **loose Drive files**, one-time, CPU only, about 1–2 h.
  They never touch the legacy tar shards that contain old lesion files.
  - A faster alternative streams the legacy shards with a vessel/rgb-only name filter. It reads tar
    bytes that include old lesion members (skipped, never written). Conservative choice: the loose copy.
- **Guards:** everything in §37 §9 applies — sha-checked loaders with no defaults; a deny-list; identity
  `.npz` files; directory guards; no legacy fallback; completeness, freshness and non-legacy canaries;
  a legacy-fingerprint invariance check; unit tests.

**Dependency graph:**

```
Stage 2 (canonical RGB, rgb-v1) ──────────────┬───────────────────────────────► Stage 6 RGB trunk
      │                                        │
      ├─► Stage 3 LWNet 91f0cada (frozen) ─► s3-91f0cada vessel cache ─┐
      │                                                                 ├─► Stage 5 prior encoder ─► Stage 7 fusion ─► Stage 8 CORN
      └─► NEW Stage 4 (SE-ResNet-101 U-Net, RGB-only) ─► s4v2-<sha> lesion cache ─┘
```

## 6. Pre-flight data checks

- Hash the 81 IDRiD segmentation images (and later any added set) against the 3,662 APTOS images;
  require 0 overlap.
- Confirm again that the IDRiD grading test has 0 overlap with any Stage-4 training image.

## 7. Stage 5–7 — Architecture 1, re-evaluated

**Spatial check [V/I].**
- The prior maps enter at 512².
- The ConvNeXt-T stage outputs at 512 input are 128²/96, 64²/192, 32²/384, 16²/768 [V, S1].
- Injection at 128² (stride 4) retains HE/EX/SE extent and MA *presence* through learned convolutions.
  MA shape is not retained, and was already lossy at 512.
- So yes — it uses spatial vessel and lesion information at useful resolutions. It is not pooled first.

**Material differences from closed routes [I].**
- **PL:** priors entered only as a *linear* 4×4-stride-4 projection added at the stem, through
  zero-init kernels.
- **Architecture 1:** a *nonlinear* prior encoder, additive **and** multiplicative injection at **four
  depths**, and (new) far better lesion maps.
- It shares none of RACAF's scalar reliability gating, and none of the old Stage 7's random-init
  encoders or early GAP.
- DINO, RETFound, 1024 px and G0 are unrelated.
- **Risk (honest):** PL showed that the *old* priors added about 0 beyond RGB, and lesion burden is
  already recoverable from P (§§24–27). The expected gain is modest and uncertain.

**It remains the leading candidate because:**
- zero-init makes it start ≡ P, the proven baseline;
- it has the lowest cost;
- it has the cleanest attribution;
- it consumes all streams spatially.

**Specification [I]:**
- **Inputs:** RGB 512²×3 (rgb-v1), vessel 512²×1 (s3 cache), lesion-v2 512²×4 (s4v2 cache).
- **Stage 5 — prior encoder (about 1–1.5 M params, GroupNorm because batch = 2):**
  - a **vessel stem** (1 → 16 ch) and a **lesion stem** (4 → 32 ch), each conv stride 2;
  - concatenated, then 4 stages producing 128²×32, 64²×64, 32²×128, 16²×256.
  - Separate stems let vessel and lesion contributions be tested separately.
- **Stage 6:** ImageNet ConvNeXt-Tiny on RGB (the P trunk; drop_path 0, as P, for comparability).
- **Stage 7:** after each ConvNeXt stage s (before its downsampling layer):
  F_s ← F_s ⊙ (1 + α_s ⊙ tanh(W_a·Pr_s)) + γ_s ⊙ (W_b·Pr_s).
  - α_s and γ_s are per-channel and zero-initialised; W_a and W_b are 1×1 convs.
  - The modified features propagate into later stages.
  - Final GAP → LN → CORN.
- **Stage 8:** CORN Dense(768→4).
- **Size and compute:** about 29.5 M params (P 27.8 M + about 1.7 M); about +10 % compute; about
  3.5–4 T4-h per run at batch 2.

**Role of each stream:**
- **RGB:** the whole pretrained trunk.
- **Vessel map:** its own stem, then shared prior features that modulate all four stages.
- **MA / HE / EX / SE maps:** the lesion stem (all 4 channels together), then the same multi-scale
  modulation.
- **Contribution measurement:** pre-specified inference-time channel-group permutation (vessel only /
  lesion only / both). If vessel contributes nothing, it is still retained (requirement) and reported
  honestly.

## 8. CORN, EMA, soft targets

- [V §25] CORN decoding is faithful, and the task is 5-class ordinal. **Keep CORN.**
- **EMA:** a shadow from the first run; it never changes the raw weights.
- **EMA evaluation rule (fixed in advance):**
  - primary architecture comparison: raw weights at BEST by validation QWK — the legacy protocol,
    comparable with P;
  - EMA at the same BEST epoch, and EMA at the final epoch — both descriptive;
  - no selection among these after seeing results.
- **Soft ordinal targets:** only after the architecture passes (step F). No evidence justifies bundling
  them earlier.

## 9–10. Cheapest gated sequence (each step pre-registered; stop at any failed gate)

| step | work | T4 cost | gate |
|---|---|---|---|
| A | this audit + data requests (Retinal-Lesions, MAPLES-DR) + the DDR decision | 0 | — |
| B | implement the v2 cache and guards + Stage 4 v2; train **v2-a** (IDRiD only) | about 2–3 h | **Segmentation gate:** IDRiD-test AUPR (IDRiD protocol) beats the *old* Stage 4 — re-scored with the same protocol, read-only — in all 4 classes, and mean AUPR ≥ 0.55 |
| C | Stage-3 parity + copies; Stage-4 v2 inference on 3,651 APTOS images; manifests; canaries | about 1 h GPU + 1–2 h CPU | all canaries and fingerprints pass |
| **C2** | **CPU-only grading-relevance check:** the 10 lesion statistics from v2 maps vs from legacy maps, training-split CV (Stage 4 v2 never saw APTOS labels, so this is clean) | 0 | v2 statistics ≥ legacy statistics on ≥ 3 vs ≤ 2 AUROC (pre-set margin). **If v2 carries no more grading information, reconsider before spending on D** |
| D | Architecture 1, **one seed (42)**, EMA shadow; compared with the existing P-42 | about 4 h | usage: lesion-group permutation ΔQWK ≤ −0.01; non-inferiority: QWK ≥ P-42 − 0.02 and AUROC ≥ P-42 − 0.01; guardrails. **One seed cannot show a small gain; it only screens for failure or non-use** |
| E | 2 more seeds (3-seed confirmation) | about 8 h | pre-registered effect rule vs the 3 P runs |
| F | soft ordinal targets (3 seeds); EMA from the shadows | about 12 h | pre-registered |
| G | IDRiD grading test, once (new Stage 3 and 4 v2 maps for the 103 images) | about 1 h | reported once |

Total if every gate passes is about 28–31 T4-h. Early failure stops at about 3–7 h.

## 11. Corrections to §36 and §37

1. **§36 — "pretrained-encoder U-Net":** the encoder was unspecified. Now SE-ResNet-101 U-Net, at
   1,536 px with 512 px patches, on evidence.
2. **§36 — Retinal-Lesions "free on request, no APTOS overlap" as if usable:** it is not on Drive,
   access has not been requested, and the licence is unverified. MAPLES-DR images also need a MESSIDOR
   request. **Only IDRiD is verified available.**
3. **§36 — "each lesion map modulates the stage matching its lesion scale":** unsupported. Replaced by
   a shared prior encoder with multi-scale outputs.
4. **§36 E1 on legacy caches:** dropped (§37).
5. **§36 — "45–60 T4-h":** replaced by the gated sequence (about 3–7 h to the first decision).
6. **§37 — assumed Stage 4 v2 keeps the RGB + vessel input:** revised to RGB-only (pretrained 3-channel
   encoders; literature SOTA is RGB-only). Stage 3 now feeds only Stages 5–7, so §37's "Stage 3 maps
   for new Stage-4 training images" is unnecessary.
7. **§37 — "vessel depends only on LWNet":** refined. It depends on LWNet + Stage-2 native RGB +
   `VESSEL_SEG_TTA`, via a channel-independent joint resize. Still Stage-4-independent (verified).
8. **Missed before:** the old Stage 4 ran at **512 px input**, a likely major cause of the MA/SE failure.
   v2 uses 1,536.
9. **§37 — re-homing vessel/rgb via legacy-shard streaming:** replaced by a loose-file copy that never
   touches shards containing old lesion files.
10. **Not previously stated:** better Dice does not imply better grading. Gates C2 and D now test
    grading relevance directly.

## 12. GO / NO-GO

- **GO** to implement steps **B → C → C2 → D**: Stage 4 v2 (IDRiD-only v2-a), the v2 cache
  infrastructure and guards, and the Architecture-1 code, with a one-seed feasibility run.
- **NO-GO** on committing to E–G, or to multi-dataset Stage 4 training, until:
  - gates B, C2 and D pass;
  - dataset access is verified;
  - the DDR decision is made.
