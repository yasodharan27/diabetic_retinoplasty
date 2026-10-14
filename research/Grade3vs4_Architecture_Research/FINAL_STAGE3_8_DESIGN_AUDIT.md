# Final Stage 3–8 design audit before implementation (2026-10-01)

Audit only: nothing was trained, regenerated or modified.

**Tags:**
- **[V]** verified in the repository or on Drive, or on a dataset page.
- **[L]** literature.
- **[I]** inference or recommendation.

**Supersedes:** where it differs, this document supersedes §36 / §37 / §38.

---

## Dataset evidence (drives the Stage 4 design)

| dataset | classes with pixel masks | images | resolution | access / licence | overlap / external-evaluation impact | usable now? |
|---|---|---|---|---|---|---|
| **IDRiD seg.** [V] | MA, HE, EX, SE, **optic disc** | 81 (54 train / 27 test; SE 26 / 14) | 4288×2848 | on Drive; already in use | 0 of 81 in the IDRiD grading test [V]; APTOS hash check pending | **yes** |
| **TJDR** [V: repository] | MA, HE, EX, SE | 561 (official train/test) | high-resolution (Topcon TRC-50DX, Zeiss CLARUS) | **Apache-2.0; direct OneDrive/Google Drive download** | Tongji Hospital (China): not APTOS / IDRiD / DDR | **yes (download)** |
| Retinal-Lesions [V: repository] | MA, iHE, EX, CWS, **vitreous HE, preretinal HE, NV, fibrous proliferation** | 1,593 (DR0 166 / DR1 337 / DR2 929 / **DR3 99 / DR4 62**) | 896×896 (pre-resized) | Google-form request; **no licence stated** | EyePACS-sourced | after request. **PDR-class mask counts in the public release are unpublished and at most tens** (the paper's 8-class results used a 12K internal set) |
| MAPLES-DR [V: docs] | MA, HE, exudates, CWS, **neo-vessels**, drusen + **optic disc/cup, macula, vessels** | 198 (138 / 60) | Messidor | annotations free (academic, cite); **images via MESSIDOR (ADCIS) request** | Messidor (France) | after the MESSIDOR request |
| Laser-marks, Zhongshan [V: paper] | **laser scars** | 70 with gold standard (154 total) | 1116×1080 – 4928×3264 | stated publicly available as journal attachments; licence to confirm on download | China | probably (verify the attachment) |
| FGADR Seg-set | MA, HE, EX, SE, **IRMA (159), NV (49)** | 1,842 | — | signed agreement; "no derivative works" clause | — | **pending user** |
| DDR lesion subset | MA, HE, EX, SE | 757 | — | CC BY 4.0 | **Training Stage 4 on DDR removes DDR's independence as the reserved external grading set** | **not recommended** |

**[I] Consequences.**
- The four core classes can be trained **now** on IDRiD + TJDR: 54 + about 400–450 training images
  (TJDR's official split to be read on download). That is roughly **8–10× today's training data**.
- **PDR-specific masks** (NV, pre-retinal/vitreous haemorrhage, fibrous proliferation, laser, IRMA) exist
  only in small numbers, spread across datasets that need requests. They are clinically the most relevant
  to the §26 failure set (low-burden PDR with haze / laser-like / haemorrhage appearance). Whether they are
  **sufficient and useful** cannot be verified until access is granted. They must be **gated, not assumed**.
- The **optic disc** is available now (IDRiD 81) and is typically easy to segment. It gives the prior
  encoder an anatomical reference (NVD at the disc, quadrant layout) at almost no cost.

---

## 1. Final Stage 3–8 architecture

```
Stage 2 RGB (canonical 512², Stage-2 processed; native res kept for Stage 3/4)
 ├─► Stage 3  LWNet (frozen, sha 91f0cada…)  ─► vessel prob 512²×1 ───────────────────────┐
 ├─► Stage 4  v2: SE-ResNet-101 U-Net, RGB-only, multi-label, partial-label trained        │
 │            ─► 512²×K maps: MA, HE, EX, SE, OD  [+ gated PDR channels: NV, PRH/VH, FP, LASER]
 │                                                                                         ▼
 └───────────────────────► Stage 6  ConvNeXt-Tiny (ImageNet) on RGB ◄── Stage 5 prior encoder (vessel stem + lesion stem)
                                            │  Stage 7: zero-init gated injection after each of the 4 stages
                                            ▼
                                     GAP → LN → Stage 8 CORN (5 grades)
```

## 2. Stage 3

- **[V]** LWNet `wnet`, `best_model.pth`, sha256 `91f0cada4b26ece63464b05be60f9b3a51f1bcd1f764081d07d3a35961118de1`.
  **Frozen.**
- **Output:** native probability map, then the cached 512² value = channel 3 of the joint 4-channel resize
  in `racaf.prepare_stage4_input`.
- **Independence [V]:** computed before, and independent of, any Stage-4 call.
- **Reuse [V + I]:** yes — the 3,651 `*_vessel_512x512.npy` files, after a K = 25 parity check
  (recompute; require max |Δ| ≤ 1e-4 with `VESSEL_SEG_TTA` = True). No regeneration.
- **[I] Consumer:** only Stage 5. It is no longer an input to Stage 4.

## 3. Stage 4 architecture

**Decision: option B+C** — one expandable **multi-label** segmentation model (sigmoid per class; classes may
overlap). It has a **shared encoder** and a per-class 1×1 output head, trained with **partial-label
supervision**: each dataset contributes loss only for the classes it annotates, and unannotated classes are
masked out of the loss [L: marginal / partial-label multi-dataset segmentation is established — e.g. DoDNet
(CVPR 2021) and the marginal loss for partially labelled data]. This is what makes combining IDRiD, TJDR,
Retinal-Lesions, MAPLES-DR and the laser set possible without false negatives for classes a dataset does not
label.

- **Model:** U-Net, **ImageNet SE-ResNet-101 encoder** (`segmentation_models_pytorch`, MIT), fine-tuned.
  About 51 M params (confirm at build).
- **Why SE-ResNet-101 [L]:** a published IDRiD-only mean AUPR of 0.652 with exactly this simple recipe;
  selected by a 1,000-trial search *for cross-dataset generalisation* (our deployment domain, APTOS, is
  unseen by Stage 4); trivially extensible to K outputs.
  - HRNet + HRDecoder (0.713) and ConvNeXt (0.699) score higher in-domain, but need mmsegmentation and
    custom decoders, and rest on 27 test images. [I] Not worth the engineering for days-scale work.
- **Input:** RGB only (Stage-2 processed), FOV-cropped.
  - Resolution: **FOV-diameter normalised to about 1,536 px** (Playout's tuned value for IDRiD-scale
    images). Images natively smaller (e.g. Retinal-Lesions at 896) are upscaled, with no gain.
  - Training: random 512 px patches with class-aware patch sampling for rare classes. Inference on the full
    image, then downsampled to 512² for the cache.
- **Loss:** masked (partial-label) BCE + Dice per class, with rare-class weighting fixed in advance.
- **Encoder fine-tuned** with a lower encoder LR. Weight EMA for Stage 4 too [L: weight averaging helps
  cross-dataset lesion segmentation, Playout 2024].
- **Stage-4 validation:** IDRiD internal validation (10 of 54) + TJDR's own validation split. **The IDRiD
  test set (27) is used once, for the gate.**

## 4. Stage 4 output classes

- **v2-a (data in hand):** **MA, HE, EX, SE, OD** (5 channels).
- **v2-b candidates (added only after access and gates):**
  - **NV** (MAPLES-DR, Retinal-Lesions, FGADR if the agreement is signed);
  - **PRH/VH merged** into one PDR-haemorrhage channel (Retinal-Lesions). Both are PDR-defining, and each
    alone is too rare;
  - **FP** (Retinal-Lesions);
  - **LASER** (Zhongshan);
  - **IRMA** (FGADR only).
- **Per-channel admission rule (pre-specified):**
  - ≥ 40 training images with the class;
  - a held-out segmentation AUPR ≥ 0.30 (image-level detection AUROC ≥ 0.80 as fallback);
  - it passes the C2 information screen (§10).
- CWS is SE (a naming alias). Drusen and the cup are excluded (not DR-grading findings).
- Vessels come from Stage 3.

## 5. Stage 4 training data

- **v2-a:** IDRiD train (54) + **TJDR train** (Apache-2.0, direct download; about 8–10× the data for the
  four core classes); OD from IDRiD.
- **v2-b (request now; add when verified):**
  - Retinal-Lesions (Google form; licence to confirm);
  - MAPLES-DR (figshare labels + MESSIDOR image request);
  - Zhongshan laser marks (attachment download);
  - FGADR only if the user signs.
- **Not used:** DDR — it keeps its role as an independent external set. Using it would make any later DDR
  grading evaluation non-independent for the full pipeline.

## 6. Stage 5–7 architecture and information flow

The alternatives re-considered:
- **Tokens:** heavier; S1 showed no pooling bottleneck.
- **Morphology or handcrafted statistics:** weak evidence; the vessel-geometry route is closed.
- **Stage-4 decoder features instead of probabilities:** richer but couples the stages. Possible later.
- **Anatomy-aware encoding:** provided by the OD channel.
- **First-layer concatenation (PL):** tested and weak.

**Retained: Architecture 1 with an expandable prior encoder.**
- **Stage 5 (about 1.5 M params, GroupNorm):**
  - a **vessel stem** (1 → 16);
  - a **lesion/pathology stem** (K → 32; K = 5 in v2-a);
  - concatenated, then 4 stages → 128²×32, 64²×64, 32²×128, 16²×256.
- **Stage 6:** ImageNet ConvNeXt-Tiny on RGB (the proven P trunk; no heavier backbone — §34 showed frozen
  DINOv2 is substitution, not complementarity, and P already overfits).
- **Stage 7:** after each ConvNeXt stage s, before downsampling:
  F_s ← F_s ⊙ (1 + α_s ⊙ tanh(W_a Pr_s)) + γ_s ⊙ W_b Pr_s.
  - α and γ are per-channel and zero-initialised (the model starts ≡ P).
  - Then GAP → LN.
- **Stage 8:** CORN.
- **Size and compute:** about 29.5 M params; about +10 % compute.

## 7. Role of each input

- **RGB:** the pretrained trunk (all grading appearance cues).
- **Vessel (Stage 3):** its own stem, then modulation at all 4 scales (vascular context; grade-0/1
  separation was its measured role in the old pipeline).
- **MA:** early-DR evidence; mostly the 128²/64² injections.
- **HE:** severity/extent — NPDR grading.
- **EX:** exudation/maculopathy extent.
- **SE:** ischaemia (CWS).
- **OD:** anatomical anchor — disc location for NVD and quadrant layout.
- **NV / PRH-VH / FP / LASER (v2-b, if admitted):** explicit PDR / treated-PDR evidence that none of the
  4 core maps encodes.
- **Every channel group's contribution is measured** by pre-specified inference-time permutation
  (vessel / core lesions / OD / PDR group).

## 8. CORN, EMA, soft targets

- **CORN retained** (decoding is faithful, §25).
- **Weight EMA as a shadow from the first grading run.** Primary comparison: raw weights at BEST (the P
  protocol). EMA at the BEST and final epochs is descriptive, with the rule fixed in advance.
- **Soft ordinal targets only after the architecture is validated.** No evidence supports bundling them.

## 9. Cache and dependency structure

```
cache/Stage2/rgb-v1/                         APTOS_<id>_rgb_512x512.npy   (copied from loose legacy files; parity Δ 0.0)
cache/Stage3/s3-91f0cada/                    stage3_manifest.json + vessel/APTOS_<id>_vessel_512x512.npy
exported_models/LesionSegmentation_v2/<stamp>/  model.pt + stage4_v2_manifest.json
cache/Stage4/s4v2-<sha12>/                   stage4_manifest.json + lesion/APTOS_<id>_pathology-s4v2_512x512.npz
       npz keys: maps (512,512,K) float16, channels [ordered names], stage4_sha256, stage3_sha256 (n/a → "none"
                 since RGB-only; kept for schema), gen_id, preproc_version
cache/Bundle/<s2-id>__<s3-id>__<s4-id>/      optional archive + bundle_manifest.json (members only from these generations)
```

- **Every manifest records:** its own sha; its upstream generation ids; the **channel list and order**;
  the split sha `bc80fd45…`; the population and exclusions; the preprocessing version (Stage-2 profile,
  FOV normalisation, 1,536 → 512 resize); per-file sha256; the legacy deny-list sha `64b3c046…`.
- **A change to K (v2-a → v2-b) yields a new Stage-4 sha, hence a new namespace.** Readers require an
  exact channel-list match.
- **Never reused:** the old Stage 4 checkpoint; old lesion `.npy` files; RACAF reliability; the legacy
  archive shards; `/content/cache/*`.
- All §37 guards apply: sha-checked loaders with no defaults; a deny-list; directory guards; no legacy
  fallback; freshness, non-legacy and completeness canaries; a legacy-fingerprint invariance check.

## 10. Cheapest gated sequence

| step | work | cost | gate |
|---|---|---|---|
| A | request Retinal-Lesions, MAPLES-DR (+ MESSIDOR), Zhongshan laser; download TJDR; hash-check all sets vs APTOS / IDRiD grading | 0 T4 | 0 overlap |
| B | implement the v2 infrastructure + guards; train **Stage 4 v2-a** (IDRiD + TJDR; 5 channels) | about 3–4 T4-h | **Segmentation gate:** on the IDRiD test (once), per-class AUPR beats the old Stage 4 re-scored with the same protocol, in all 4 core classes; mean ≥ 0.55; OD Dice ≥ 0.85 |
| C | Stage-3 parity + copies; Stage-4 v2-a inference on 3,651 APTOS images; manifests; canaries | about 1 T4-h + 1–2 CPU-h | canaries pass |
| **C2** | **CPU information screen** on the training split, label-free extractors: per-channel statistics (area, count, max, per-quadrant relative to OD) with CV probes. Two comparisons: (i) v2 vs legacy statistics; (ii) **frozen ImageNet ConvNeXt (§31 f512) ⊕ v2 statistics vs f512 alone, at matched capacity** (the §34 method) | about 1–2 CPU-h | (i) v2 ≥ legacy; (ii) a complementary gain with CI > 0 on the grade ≥ 3 vs ≤ 2 AUROC or QWK-proxy. If (ii) fails, D is still allowed (the fused CNN can use spatial detail that statistics miss), but it becomes a low-prior test |
| D | Architecture 1, **one seed**, EMA shadow, vs P-42 | about 4 T4-h | the model uses the lesion group (permutation ΔQWK ≤ −0.01); non-inferior; guardrails |
| E | 3-seed confirmation | about 8 T4-h | pre-registered effect rule vs the 3 P runs |
| B′ | *(parallel, when data arrives)* Stage 4 v2-b with PDR channels → C/C2 → admit channels | about 3–5 T4-h | per-channel admission rule (§4) |
| F | soft ordinal targets on the final architecture | about 12 T4-h | pre-registered |
| G | IDRiD grading test, once | about 1 T4-h | — |

## 11. T4 cost to the first meaningful decision

About **4–5 T4-h + 2–4 CPU-h** (B + C + C2) to decide whether the new Stage 4 carries grading information
beyond RGB. About **8–9 T4-h** to the first grading result (D).

## 12. Why this is preferable to the §38 plan

1. **About 8–10× more core-lesion training data now** (TJDR, Apache-2.0, direct download). §38 relied on
   IDRiD alone, or on unverified requests.
2. **Stage 4 is expandable.** Multi-label + partial-label training adds PDR / treated-PDR channels (NV,
   PRH/VH, FP, laser, IRMA) as their data becomes available. §38 fixed Stage 4 at 4 classes.
3. **Honest data realism for PDR signs.** The public Retinal-Lesions release has only 62 DR4 images, so PDR
   channels are admitted by a pre-specified rule, not assumed.
4. **An OD channel now**, at almost no cost, gives the fusion an anatomical anchor.
5. **A stronger pre-grading gate.** C2 tests *complementarity to RGB features* at matched capacity (the
   §34 lesson), not just "v2 better than legacy" — before any multi-seed spend.

## 13. Remaining major uncertainties

- Whether better lesion maps add grading information **beyond a strong pretrained RGB CNN**. The PL result
  says the old maps did not; C2 and D test this.
- **Access and actual counts** of the PDR-class masks. They may be too few to learn. The laser-set licence
  is unverified.
- **Domain shift** from Stage-4 training sets to APTOS; there is no APTOS lesion ground truth to measure it
  (only indirectly, via C2 and D).
- One-seed D can only screen for failure or non-use. A real effect needs E.
- The APTOS labelling convention for treated eyes (laser) is unknown, so a LASER channel's grading value is
  uncertain.

**GO:** A → B → C → C2 → D (with B′ in parallel as data arrives).
**NO-GO:** E–G until the gates pass.

## Addendum (2026-10-01): local EyePACS

- **Location:** `datasets/EyePACS/raw` (moved from `D:\Projects\EyePACS\raw` on 2026-10-01) — 88,702 Kaggle EyePACS images and `trainLabels.csv` (35,126 graded). No lesion masks.
- **Stage 4:** full-resolution originals for Retinal-Lesions masks; an unlabelled pool for semi-supervision later.
- **Grading:** a potential about-12× data increase for Stages 5–8 — a user scope decision, pre-registered after gate D.
- **Leakage:** none with APTOS, IDRiD or DDR.
