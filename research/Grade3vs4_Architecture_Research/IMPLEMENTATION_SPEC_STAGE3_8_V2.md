# Implementation specification — Stage 3–8 v2 (2026-10-01)

This is the build spec. Nothing is implemented yet.

**Tags:**
- **[V]** verified in the repository, on Drive or on a dataset page.
- **[L]** literature.
- **[I]** decision / inference.

**Supersedes:** §39 where different (see §14).

## 1. Final architecture

```
Stage 2 canonical frame: full native image → direct resize (no crop, aspect squashed, as jtd._resize_rgb_01) [V]
 ├─ 512²×3   RGB  (cache Stage2/rgb-v1) ─────────────────────────────────────────► Stage 6 ConvNeXt-T (ImageNet)
 ├─ Stage 3 LWNet 91f0cada (frozen) → vessel 512²×1 (cache Stage3/s3-91f0cada) ──► Stage 5 vessel stem ─┐
 └─ 1536²×3 RGB → Stage 4 v2 U-Net (SE-ResNet encoder, multi-label K) → 1536²×K probs                   │
        → exact 3×3 block pooling → 512²×2K (mean + max per class) (cache Stage4/s4v2-…) → Stage 5 pathology stem
                                              Stage 5 → prior pyramid P1..P4 → Stage 7 gated injection into Stage 6 → GAP → LN → Stage 8 CORN
```

## 2. Stage 3 output and cache usage

- **[V] Model:** LWNet `best_model.pth`, sha `91f0cada4b26…18de1`, frozen.
- **[V] Cached value:** `APTOS_<id>_vessel_512x512.npy` = channel 3 of `racaf.prepare_stage4_input`, the
  full native frame resized directly to 512². Independent of Stage 4.
- **Reuse** after a parity check on 25 fixed-seed ids:
  - recompute `predict_vessel_mask(native_stage2_rgb)` with `VESSEL_SEG_TTA` = True and the sha-asserted
    LWNet, then `prepare_stage4_input(...)[0, ..., 3:4]`;
  - require max |Δ| ≤ 1e-4 for every id;
  - on failure, stop; regenerate Stage 3 only then.
- **Copy** from the loose Drive files (never the legacy shards) into `cache/Stage3/s3-91f0cada/vessel/`.
- **Path to Stage 5:** vessel 512²×1 float in [0, 1] → Stage 5 vessel stem (§8).

## 3. Stage 4 encoder / decoder

- **`segmentation_models_pytorch.Unet`**, encoder **SE-ResNet-101 (ImageNet)**, standard U-Net decoder
  (256, 128, 64, 32, 16), `classes = K`, **sigmoid per class** (multi-label). About 51 M params; the
  encoder is fine-tuned at 0.1× the decoder LR.
- **Why, over the realistic alternatives:**

  | candidate | IDRiD evidence [L] | cross-dataset evidence | multi-label / partial-label | T4 / engineering |
  |---|---|---|---|---|
  | **SE-ResNet-101 U-Net** | mAUPR 0.652 (IDRiD-only, 1,536 px / 512-px patches) | **won a 1,000-trial search optimised for cross-dataset generalisation** | trivial (K sigmoid heads) | smp, a few lines; fits in 16 GB with fp16 |
  | HRNet + HRDecoder | 0.713 | none reported | possible | mmsegmentation + custom decoder; 10 GB inference at 2,880×1,920 |
  | ConvNeXt (UPerNet, mmseg) | 0.699 | none reported | possible | mmsegmentation |
  | SegFormer (MiT) | below HRNet in the HRDecoder comparison | none | easy (smp) | easy |

  Our deployment is cross-dataset (APTOS is never segmented), and the in-domain margins rest on 27 test
  images. **SE-ResNet-101 is kept.**
- **Pre-coding check:** smp's `se_resnet101` ImageNet weights are hosted on the Cadene URL
  (`data.lip6.fr`), whose availability is uncertain. **Fallback, fixed now:** `tu-seresnext101_32x4d`
  (timm, Hugging Face-hosted).
- **Geometry (fixed by alignment):** full native Stage-2 image → **direct resize to 1,536² (the same
  squash as the canonical 512 frame)**.
  - Training: random 512² patches, with class-aware sampling: p = 0.5 a patch centred on a positive pixel
    of a random annotated class.
  - Inference: the full 1,536² image (fp16, batch 1).
  - Output 1,536²×K → **exact 3×3 block pooling** to 512² (mean and max).

## 4. Stage-4 channels to implement first (K = 4)

| class | data now [V] | masks / quality | learnable at 1,536 → 512 | grading relevance | include first? |
|---|---|---|---|---|---|
| **MA** | IDRiD 54 (+ TJDR) | pixel masks; tiny | shape lost at 512 — **presence kept via the max channel** | mild vs none | **yes** |
| **HE** | IDRiD 53/54 (+ TJDR) | pixel masks | yes | NPDR severity (4-2-1) | **yes** — defined as **any haemorrhage** (superset; §7) |
| **EX** | IDRiD 54 (+ TJDR) | pixel masks | yes | exudation | **yes** |
| **SE** (= CWS) | IDRiD 26 (+ TJDR) | pixel masks | yes | ischaemia | **yes** |
| OD | IDRiD 81 | pixel masks | yes | not a DR finding; the RGB trunk sees the disc directly; the value as an anchor is unproven | **no** — optional (§5) |

**Annotation completeness [V/I]:** IDRiD annotates all 4 lesions for every image; a missing mask file
means *absent* (a negative), not unannotated. TJDR (Labelme polygons, 4 classes) is assumed complete —
**verify on download.**

## 5. Optional later channels — admission conditions

| class | only sources | verified count | issues |
|---|---|---|---|
| NV | FGADR (49; agreement pending), MAPLES-DR (count unverified; MESSIDOR request), Retinal-Lesions (count unverified; request) | none ≥ 40 verified | fine, sparse; at 512 only via max |
| IRMA | FGADR only (159) | pending | agreement / clause 6 |
| Preretinal HE | Retinal-Lesions only | unverified (≤ about 62 DR4 images) | keep **separate** from VH unless counts force a pre-registered merge |
| Vitreous HE | Retinal-Lesions only | unverified | often diffuse haze; the boundary definition differs from focal lesions |
| Fibrous proliferation | Retinal-Lesions only | unverified | rare |
| Laser scars | Zhongshan 70 (attachments; licence unverified) | 70 | APTOS labelling convention for treated eyes unknown; domain (Chinese clinic cameras) |
| OD | IDRiD 81 | 81 | usefulness unproven |

**Admission rule (pre-registered, per class):**
1. Access and licence verified.
2. ≥ 40 training images with the class.
3. Held-out segmentation AUPR ≥ 0.30, or image-level presence AUROC ≥ 0.80.
4. A C2 incremental gain (§10) with the class added vs without, CI > 0.

Each admission produces a new K, a new Stage-4 generation and a new cache.

## 6. Partial-label loss (exact)

Per training image i from dataset d(i), with annotated class set A_d ⊆ {1…K}, logits z_{i,k} (pixels p),
targets y_{i,k,p} ∈ {0, 1}, and field-of-view mask f_{i,p} (the loss ignores the black border):

- **Masked BCE (per image, per class):**
  BCE_{i,k} = Σ_p f_{i,p} · [ −w⁺_k y log σ(z) − (1 − y) log(1 − σ(z)) ] / Σ_p f_{i,p}.
- **Pooled soft Dice per class over the batch, only across images that annotate k:**
  D_k = 1 − (2 Σ_{i∈B_k} Σ_p f σ(z) y + ε) / (Σ_{i∈B_k} Σ_p f (σ(z) + y) + ε),
  with B_k = {i ∈ batch : k ∈ A_{d(i)}} and ε = 1.
- **Total:**
  L = Σ_k λ_k · [ (1 / |B_k|) Σ_{i∈B_k} BCE_{i,k} + D_k ] / Σ_k 1[|B_k| > 0].
  A class with |B_k| = 0 contributes nothing. **Unannotated classes never contribute — neither negatives
  nor gradients.**
- **Imbalance:**
  - w⁺_k = clip(√(neg_k / pos_k), 1, 20) from training-pixel counts;
  - λ_k = 1 for the first 4 classes; λ_k = 0.5 for any admitted sparse class (fixed in advance);
  - class-aware patch sampling.
- **Stability for sparse classes:**
  - dataset-balanced batches (≥ 1 image from each dataset annotating a sparse class per 4 batches);
  - per-class loss normalised by |B_k|;
  - λ capped at 0.5;
  - gradient clipping at 1.0;
  - EMA of weights (decay 0.999); the EMA model is exported.
- **Definition harmonisation (required for combining datasets):**
  - **HE = any haemorrhage.** For Retinal-Lesions, HE := iHE ∪ pHE ∪ vHE, and PRH/VH are additional
    sub-classes, because IDRiD/TJDR "haemorrhage" does not separate them.
  - **SE ≡ CWS.**
  - **EX = hard exudates only.**
  - Datasets whose class definitions cannot be mapped are excluded for that class (A_d omits it).

## 7. Stage-4 datasets

- **Now:**
  - **IDRiD seg.** 54 train (10 held as internal validation, stratified by SE presence) — all 4 classes.
    The 27 test images are **used once** for the gate.
  - **TJDR** (561; Apache-2.0; direct download) — all 4 classes; its official split (train → training,
    test → a second validation). About an order of magnitude more core-lesion data. Pixel-annotation
    style (Labelme polygons) is coarser than IDRiD: a known mismatch, accepted.
- **Later (§5):** Retinal-Lesions (full-resolution originals available from local
  `datasets/EyePACS/raw` (moved from `D:\Projects\EyePACS` on 2026-10-01), but its masks are 896² and would be upscaled), MAPLES-DR, Zhongshan laser marks,
  FGADR if signed.
- **Never:** DDR (reserved external set); APTOS (no masks); the IDRiD test set for training.
- **Pre-flight:** SHA-256 of every Stage-4 training image vs the 3,662 APTOS images and the IDRiD grading
  test set → require 0 overlap.

## 8. Stage 5 — prior encoder (K-agnostic)

- **Inputs:** vessel V ∈ [0, 1]^{512²×1}; pathology Q ∈ [0, 1]^{512²×2K}, channels ordered from the
  Stage-4 manifest as [c:mean, c:max for each class c].
- **Vessel stem:** Conv3×3(1→16, stride 2) + GN(4) + GELU → 256².
- **Pathology stem:** Conv3×3(2K→32, stride 2) + GN(8) + GELU → 256². **The only K-dependent layer:**
  it is built from `len(manifest.channels)`.
- **Concat** → 48 ch @ 256².
- **Four residual down-blocks.** Each: Conv3×3 stride 2 + GN + GELU, then Conv3×3 + GN, plus a 1×1
  stride-2 shortcut, then GELU. Outputs:
  - P1 128²×32
  - P2 64²×64
  - P3 32²×128
  - P4 16²×256
- **Size:** about 1.3 M params.
- **Config:** `stage4_generation_id` → the manifest → channel list → K. No hard-coded lesion list anywhere.

## 9. Stage 6/7 — fusion

- **Stage 6:** ImageNet ConvNeXt-Tiny (the pinned P weights `d547c096…`), RGB 512², drop_path 0 (as P).
  Stage outputs:
  - F1 128²×96
  - F2 64²×192
  - F3 32²×384
  - F4 16²×768
- **Stage 7:** for s = 1..4, immediately after stage s and **before** downsampling layer s:
  F_s ← F_s ⊙ (1 + α_s ⊙ tanh(Conv1×1_a(P_s))) + γ_s ⊙ Conv1×1_b(P_s).
  - α_s and γ_s are per-channel and **zero-initialised**, so the initial output is exactly P. This is
    asserted by a unit test.
  - Modified features propagate into later stages.
  - Then GAP → LN (the backbone head norm) → CORN Dense(768→4).
- **Implementation:** a Keras functional ConvNeXt-T built block-by-block, with pinned weights copied
  positionally (reusing `pl_convnext.copy_pretrained`), so the fusion layers can sit between stages.
- **Size:** about 30 M params in total.
- **Contribution tests** (pre-specified, at inference): permute (a) V, (b) Q, (c) both, across validation
  images.

## 10. C2 — CPU information screen (before any grading run)

**Question:** does the NEW Stage-4 pathology representation (and separately the Stage-3 vessel map) carry
DR-grading information beyond what frozen RGB ConvNeXt already contains?

- **Features (all label-free):**
  - **R** = frozen ImageNet ConvNeXt-T `pooled_final` at 512 (the existing §31 `f512`, 2,921 training
    images);
  - **Q-pyr** = the *actual Stage-5 inputs* reduced by a fixed spatial pyramid: mean and max over
    1×1 + 2×2 + 4×4 grids of each of the 2K channels = 2K·2·21 features;
  - **V-pyr** = the same for the vessel map.
- **Probe:** the §34 capacity-matched protocol (block-wise PCA, total k = 64, LogisticRegressionCV,
  10 × 5-fold CV, identical folds).
  - Arms: R_64, Q_64, R_32 ⊕ Q_32, and R_32 ⊕ V_32.
  - Endpoints: the **mean AUROC over the 4 cumulative cuts (≥ 1, ≥ 2, ≥ 3, ≥ 4)** (primary, overall
    5-class), plus ≥ 3 vs ≤ 2 (secondary).
- **Gate:** R⊕Q − R ≥ +0.005 with paired CI > 0, **and** R⊕Q ≥ Q (complementarity, not substitution).
- **Descriptive:** V's increment; leave-one-class-out increments (used later for channel admission).
- **Cost:** pyramid features are computed on Colab during cache generation (a few MB) and probed on the
  laptop in about 1 h. If the gate fails, D is still allowed only as a clearly low-prior test (the CNN can
  use spatial detail that pyramids miss). PL remains the fallback.

## 11. Cache lineage

```
exported_models/LesionSegmentation_v2/<stamp>/  model.pt (EMA weights), stage4_model_manifest.json
   {sha256, encoder, weights source + sha, classes [ordered], datasets {name: count, sha of file list},
    annotated-class map A_d, geometry "full-frame→1536²", loss spec, IDRiD-test metrics, git commit,
    legacy deny-list 64b3c046…}
cache/Stage2/rgb-v1/                      manifest {preproc Stage-2 profile DR, frame full-direct-512, split bc80fd45…, ids, per-file sha}
cache/Stage3/s3-91f0cada/                 manifest {lwnet sha, VESSEL_SEG_TTA, parity result, split, ids, per-file sha}
cache/Stage4/s4v2-<model sha12>-K<k>/     manifest {stage4 sha, channels [c:mean, c:max…], pooling "3x3 block from 1536²",
                                           dtype uint8 (p·255), preproc version, split, ids, per-file sha, gen_id}
       APTOS_<id>_pathology-s4v2_512x512.npz  {maps uint8 (512,512,2K), channels, stage4_sha256, gen_id}
cache/Bundle/<rgb-v1>__<s3-…>__<s4v2-…>/  bundle_manifest {the three generation ids + shas (incl. stage3 sha), population}
                                           + optional tar shards (members from these generations only)
```

- The Stage-3 sha is recorded in the **bundle** (Stage 4 does not consume Stage 3).
- Any change of model or K produces a new Stage-4 generation.
- The Stage-3 generation is reused across all Stage-4 generations.
- **Storage:** uint8, 2K = 8 channels, about 2.1 MB per image, about 7.7 GB for 3,651 images.
- **All §37 guards apply:** no-default sha-checked loaders; the deny-list; directory guards; no legacy
  fallback; the freshness canary (recompute 8 images, max |Δ| ≤ 2/255); completeness; the legacy
  fingerprint.

## 12. Implementation / training sequence

1. **Code (laptop, CPU tests):**
   - `stage34_cache_v2.py`: manifests, npz I/O, guards, pyramid features;
   - `stage4_v2.py`: smp model, dataset specs with A_d, partial-label loss, patch sampler, 1,536
     inference, block pooling;
   - `arch1_model.py`: prior encoder + ConvNeXt-T with injection; zero-init ≡ P test;
   - `arch1_data.py`;
   - tests.
2. **Colab (download + pre-flight, about 0.5 h):** TJDR → Drive; hash-overlap checks; verify smp
   weight availability (or the fallback).
3. **Train Stage 4 v2 (K = 4), about 3 T4-h.**
   - Validate on IDRiD-val + TJDR-test.
   - **Gate once on the IDRiD test:** per-class Dice (same protocol as the documented old Stage-4
     report) beats 0.357 / 0.127 / 0.024 / 0.017 in all 4 classes, **and** mean AUPR ≥ 0.55.
   - The old checkpoint is never loaded; the documented numbers are the comparator.
4. **Colab (caches), about 1 T4-h + 1–2 h CPU:** Stage-3 parity + copies; RGB copies; Stage-4 inference
   on 3,651 images; manifests; canaries; pyramid features.
5. **Laptop: C2**, about 1 h.
6. **D:** Architecture 1, **one seed** (42), EMA shadow, compared with P-42 (legacy protocol), about
   4 T4-h. Gates: Q-permutation ΔQWK ≤ −0.01; non-inferiority (QWK ≥ P-42 − 0.02; AUROC ≥ P-42 − 0.01);
   guardrails.
7. **E:** +2 seeds; **F:** soft ordinal targets; **G:** IDRiD grading test, once. Each after its gate.

**T4 to the first meaningful decision (after C2):** about 4–4.5 T4-h. To the first grading result (D):
about 8–9 T4-h.

## 13. Decisions to resolve before coding

1. **Encoder weights:** Cadene `se_resnet101` availability vs the fixed fallback `tu-seresnext101_32x4d`
   (checked in step 2; the fallback is pre-approved).
2. **First channel set K = 4 (MA, HE, EX, SE), OD excluded:** confirm.
3. **TJDR verification on download:** image count, split, resolution, annotation completeness,
   class-name mapping.
4. **Framework split:** Stage 4 in PyTorch (as Stage 3 already is) vs Keras Stages 5–8. Confirm
   acceptable.
5. **Pathology cache as uint8 mean+max (2K channels) at 512:** confirm, vs float16 mean-only.
6. **C2 thresholds:** +0.005 with CI > 0, and R⊕Q ≥ Q — confirm.

## 14. Corrections to §39

- **"FOV-diameter normalised ~1,536 px":** wrong for fusion. The RGB/vessel caches use the **full-frame
  direct 512 resize** [V], so an FOV crop would misalign the maps. Replaced by a full-frame direct resize
  to 1,536² (exactly 3× the canonical frame).
- **"Output downsampled to 512² (legacy anti-aliased bilinear)":** would average MA/NV/IRMA away.
  Replaced by **exact 3×3 block mean + max pooling** (uint8).
- **"v2-a = MA, HE, EX, SE, OD":** OD is unproven for grading. **Excluded** from the first set;
  admission-rule candidate.
- **"PRH/VH merged":** premature. Kept as separate candidates; merged only by a pre-registered rule if
  counts force it. Plus the **HE = any-haemorrhage superset** definition, needed for consistent
  multi-dataset training (§39 missed it).
- **"OD Dice ≥ 0.85" gate:** removed with OD.
- **C2 "area / count / max / per-quadrant relative to OD" statistics:** replaced by pyramid pooling of
  the *actual Stage-5 inputs*, capacity-matched, with an overall 4-cut AUROC endpoint and a
  complementarity (R⊕Q ≥ Q) condition.
- **"Stage-4 gate beats old Stage 4 re-scored":** would require loading the deny-listed checkpoint.
  Replaced by the **documented** old Dice values under the same protocol.
- **"TJDR ~400–450 training images":** unverified; verify on download.
- **Not considered in §39:** the Cadene weight-hosting risk for `se_resnet101`, and the need for
  Stage-3 lineage in the **bundle** (Stage 4 is RGB-only).
- **Partial-label training:** needed only once new-class datasets are added. IDRiD and TJDR are fully
  annotated for the 4 core classes (TJDR to be confirmed).
