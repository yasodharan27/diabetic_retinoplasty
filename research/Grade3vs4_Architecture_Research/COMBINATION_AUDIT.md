# Combination audit — existing mechanisms combined with P (2026-09-30)

**Design audit only.** Nothing was implemented, trained or launched.

**Objective (as changed by the user on 2026-09-30).** Find a combination of two or more existing
mechanisms that:
- has not already been combined in this pipeline;
- has a concrete reason to improve **overall** DR grading, not just grade 4;
- can be tested in days.

Novelty is optional and is not claimed anywhere below.

**Evidence base.**
- Research record §§1–32.
- Every document in this folder.
- The reports under `docs/experiments/`.
- Drive manifests of the reusable frozen features (read with rclone).
- A fresh literature search, run after the project files were read.

---

## A. What the project has already closed or made redundant

| Direction | Where | Why it is closed |
|---|---|---|
| RACAF and any reliability-gated or κ fusion (C1-gate, C3); D6 reliability as an error signal | §§1–5, Multiseed §12 | 6-run INCONCLUSIVE; gate never learned; C3 engaged but gave +0.002; D6 0/6 above 0.5 |
| Generic vessel map or vessel statistics as extra information | §§6–7, Multiseed §12.4 | Increment over lesion +0.0006 (grade 4 vs 0–3) and −0.028 (grade 4 vs 3); vessel permutation mostly costs grade-0 recall |
| Threshold / decision-rule changes as a fix | §§8–10 | A dial on a poor curve; grade-3 recall −61 % |
| Lesion-burden pathway (Architecture A) | §13 | Gate INCONCLUSIVE; the premise was weakened |
| Objective pressure on 3 vs 4 (marginal aux BCE, task-3 weighting) | §§14, 16 | NOT_SUPPORTIVE / NONSPECIFIC |
| Any 3-vs-4 training experiment judged on APTOS validation | §17 | Minimum detectable effect 0.065 at 3 seeds, so it cannot be resolved |
| Output structure (two-route head); CORN decoding | §20, §25 | Null with a ±0.005 bound; decoding is faithful |
| Pathway rebalancing / ModDrop | §21 | G0 FAIL |
| Stage 3/4 priors as early fusion into P | §22 | +0.0076 < +0.01, and grade-3 recall −0.22 |
| Readout redesign on P (grid, max, multi-scale, MIL / lesion-query pooling) | §23 | `pooled_final` is already the best linear readout |
| Grade-4 failure diagnostics | §§24–27.1 | Closed by the user; reopen only with external evidence |
| Global higher input resolution (1024 px), frozen | §31 | Δ +0.0003 |
| Retinal-domain foundation model (RETFound-DINOv2) | §32 | Adds nothing beyond P, and is weaker than generalist DINOv2 |

**What P is limited by.** These are the limitations that survive the closures above; all are
measured, not assumed.

- **L1 — label memorisation and over-fitting.**
  - Training QWK 0.99 against validation 0.91.
  - Validation loss rises after epochs 5–8, and LAST is below BEST in 6/6 runs.
  - Late-backbone features degrade BEST→LAST (§23).
  - The head fits training grades 3/4 at 0.999.
  - The train-CV − validation gap grows with severity: 0.007 / 0.036 / 0.082 / 0.103 for 0v1 … 3v4
    (§24).
- **L2 — documented label noise.**
  - 32 of 134 byte-identical duplicate pairs disagree (24 %).
  - Disagreement concentrates in pairs touching grade 3/4, mostly 4 vs ≤ 2 (p = 0.020) (§15).
- **L3 — a representation prior that another pretraining can complement.**
  - Frozen DINOv2-L adds +0.019 AUROC at PCA-16 and +0.010 at PCA-64 over frozen ImageNet
    ConvNeXt-T for grade ≥ 3 vs ≤ 2 (CI excludes 0).
  - The gain is concentrated at 2 vs 3 (+0.051, CI +0.025 to +0.076) (§32).
  - **This was measured against frozen ImageNet ConvNeXt, not against the fine-tuned P.**
- **L4 — checkpoint-selection variance.**
  - BEST epochs range from 3 to 18.
  - `p_cond_3` moves up to 0.14 within a run (§16.1).
  - This is a protocol issue, not an information limit.

---

## B/C. Candidate combinations

"P" = ImageNet ConvNeXt-Tiny, fine-tuned on RGB at 512 px with a CORN head (the §22 baseline).

### 1. P + frozen DINOv2 ViT-L/14 (late fusion) — **STRONG PRACTICAL CANDIDATE**

1. **Component A:** P, a supervised ImageNet CNN fine-tuned on APTOS labels.
2. **Component B:** frozen DINOv2-L (self-distillation SSL, ViT, never exposed to APTOS labels), read
   by a small head fitted on training features.
3. **Contribution of each.**
   - A: a strong, label-fitted representation (4 vs 0–2 AUROC 0.95, QWK 0.92).
   - B: a representation from a different prior (SSL rather than supervised; global attention rather
     than local convolution) that cannot memorise APTOS labels.
4. **Why complementary.** Two separate reasons, each tied to a measured limitation:
   - (i) L3: the only label-free complementarity test the project ran found DINOv2 information
     beyond a ConvNeXt of the same kind, located at 2 vs 3. That is exactly where P's
     generalisation gap is large (L1).
   - (ii) L1/L2: P's features fit training labels almost perfectly, noise included. A frozen branch
     fitted only by a low-capacity head cannot fit those labels, so it acts as a label-free anchor.
   - This is not "two models ensemble better". The test below controls for plain ensembling.
5. **Evidence each component works alone.**
   - P: §22.
   - Frozen DINOv2 on DR: QWK 0.882 frozen vs 0.904 fine-tuned (IOVS abstract).
   - Fine-tuned DINOv2-L APTOS AUROC 0.952 (Candidate #2 research).
   - In this project: frozen DINOv2 alone scores 0.9245 on ≥ 3 vs ≤ 2 (training CV, §32).
6. **Literature status: established at application level.**
   - CNN + ViT fusion for DR grading is common (e.g. REFA-DINO, EfficientNet-B0 + DINOv2 with
     cross-attention; an evidential CNN–ViT hybrid, arXiv 2510.26315).
   - None found that tests complementarity against a strong fine-tuned CNN with an ensembling
     control. That is the only defensible contribution, and it is empirical, not methodological.
7. **Equivalent tested in the project?** No.
   - §32 compared DINOv2 with *frozen* ConvNeXt, training split only.
   - Candidate #2 Stage B (fusion with P) was never run, because the retinal arm failed.
   - RACAF's Ĝ fused a random-init Swin trained on the same labels. That is a different mechanism.
8. **Data / labels.** None new. DINOv2 CLS features already exist for all 2,921 training and 730
   validation images (`experiments/Cand2_FrozenFM/v1/cand2_features.npz`, sha `7d1566b6…`). P's saved
   validation logits exist for 3 seeds.
9. **Complexity.**
   - Late fusion: a probe plus a fixed-weight average of CORN cumulative probabilities. Low.
   - Trainable fusion: a zero-init residual on P's logits, as in the RECOMMENDED_DESIGN §3.1 pattern.
     Low to moderate.
10. **Colab cost.**
    - Late fusion: **no GPU** (CPU minutes). The optional frozen-ImageNet-ConvNeXt control needs its
      730 validation features: CPU ≈ 1 h, or T4 ≈ 10 min.
    - Trainable fusion: DINOv2 features for the 8 dihedral views (≈ 1 T4-h), then 6 runs at
      ≈ 3–5 T4-h each. The per-run figure is estimated from the §22 wall-clock, not measured.
11. **Overfitting risk.**
    - Late fusion: low (a probe on 1,024-d with 2,921 samples, L2 fixed in advance).
    - Trainable fusion: moderate. P can learn to ignore the branch.
12. **Clean ablation (late fusion).**
    - Arms: P_s alone | P_s ⊕ DINO | P_s ⊕ P_s′ (a second P seed: the **ensembling control**) |
      P_s ⊕ frozen-ImageNet-ConvNeXt probe (the **"any frozen probe" control**).
    - Same fitter, same fixed weights.
    - Only P ⊕ DINO beating both controls counts as complementarity.
13. **Completable quickly?** Yes: late fusion in under a day; trainable fusion in about 2 days of
    Colab.
14. **Worth implementing?** **Yes. First in the queue.**

### 2. P + ordinal soft-label targets inside CORN (Gaussian / N-ULS-style) — **STRONG PRACTICAL CANDIDATE**

1. **Component A:** P.
2. **Component B:** soft ordinal targets.
   - Each training label y becomes q(j) ∝ exp(−(j − y)² / 2σ²), with σ fixed in advance (≈ 0.5,
     keeping ~95 % of the mass within ±1 grade, as in N-ULS).
   - CORN is trained on its **expected** negative log-likelihood under q. Task k's target is
     P_q(Y > k | Y ≥ k), weighted by P_q(Y ≥ k).
   - With σ → 0 this reduces exactly to the current weighted CORN.
3. **Contribution of each.** A supplies the representation. B regularises the *targets* so that
   adjacent-grade label disagreement is not memorised as certainty.
4. **Why complementary.**
   - B acts on a **label** limitation (L1/L2), not a representation limitation.
   - The evidence: memorisation (training 0.99 vs validation 0.91), a severity-graded
     generalisation gap, and 24 % same-image label disagreement.
   - It does not overlap with candidate 1's mechanism.
5. **Evidence alone.**
   - Galdran et al. 2020 (TVST), non-uniform label smoothing for DR: QWK 73.17 → 77.69 over CE
     (p < 0.025), with no loss of AUROC. Plain label smoothing hurt.
   - Soft ordinal labels (SORD, CVPR 2019).
   - Ordinal noise-correction with label distributions (ORDAC 2025, including DR; synthetic noise).
6. **Literature status.** Established for softmax / DR. Its use inside CORN's conditional likelihood
   was not found. It is a trivial adaptation and is not claimed as novel.
7. **Equivalent tested in the project?** No.
   - §§12–16 changed the *pressure* on the 3/4 boundary (aux BCE, task weighting). They did not
     regularise the targets.
   - NEXT_STEP C9 (noise-robust training) was listed and never run.
   - SORD was earlier argued to be the "wrong direction" **for sharpening 3 vs 4**. That argument
     does not apply to the overall objective, but it is a real guardrail risk for grade 3/4.
8. **Data / labels.** None new.
9. **Complexity.** Low: one loss function plus tests. The σ → 0 reduction to `weighted_corn` is an
   exact unit test.
10. **Colab cost.** 3 new runs (seeds 42/123/2026) at ≈ 3–5 T4-h each. The 3 existing P runs are
    reused, because the protocol is identical.
11. **Overfitting risk.** It is itself an anti-over-fitting mechanism. The main risk is
    under-confidence at the severe end.
12. **Clean ablation.** P vs P + soft targets: same seeds, schedule, data order and head.
    - Mechanism endpoints: the train − validation QWK gap and ECE should fall.
13. **Completable quickly?** Yes: 1 day to implement and test, 1–1.5 days of Colab.
14. **Worth implementing?** **Yes. Run in parallel with candidate 1.**

### 3. P + DINOv2 + soft ordinal targets (three components) — **STRONG, CONDITIONAL**

- **Rationale.** The two added mechanisms act on different limitations: the representation prior
  (L3) and label memorisation (L1/L2). This is the only three-way combination with a separate reason
  for each part.
- **Status.**
  - Literature: application-level only.
  - Project: untested.
- **Ablation.** A 2×2 factorial (± DINOv2, ± soft targets) that reuses the arms of 1 and 2, so only
  one new cell is needed (3 runs).
- **Pre-condition.** Build this only if 1 **and** 2 each pass on their own. Otherwise it cannot be
  attributed.

### 4. P + frozen-FM-guided noisy-sample down-weighting (CUFIT / confident-learning style) — **POSSIBLE BUT WEAK**

- **Mechanism.** Frozen DINOv2 is probed with cross-fitted out-of-fold predictions on the training
  split. Training images whose given label the probe finds implausible (ordinal distance ≥ 2) are
  down-weighted in P's CORN loss.
- **Literature: essentially exists.** CUFIT (NeurIPS 2024) uses frozen-VFM linear probing to pick
  clean samples, then curriculum fine-tuning. It was tested on APTOS, with synthetic 40 % noise.
  Confident learning (Northcutt 2021) is the same idea.
- **Why weak here.**
  - APTOS noise concentrates at 4 vs ≤ 2, which is exactly where the genuine but hard low-burden PDR
    images sit (§§26–27). A detector cannot tell "mislabelled" from "hard". Down-weighting would
    probably cement the grade-4 failure.
  - Only about 12 % of the failure set has any duplicate-label evidence to validate the detector.
- **Verdict.** Candidate 2 handles the same limitation softly, without choosing samples to discard.

### 5. P + image-quality state from Stage 1 (EyeQ IQA) — **POSSIBLE BUT WEAK; cheap gate first**

- **Mechanism.** The frozen EfficientNet-B0 IQA (Good / Usable / Reject) is used as a loss weight, or
  as a stratification or conditioning input.
- **Complementarity.** It carries acquisition-quality information, not disease information.
  - In EyeQ, referable-DR prevalence rises from 17 % in Good images to 28 % in Reject images, so
    quality and grade are confounded.
- **Literature: established.** Quality-guided DR grading (MICCAI 2018); QGDR (Frontiers 2026); QMix
  (TMI; quality plus label noise).
- **Against.**
  - S4 found no quality difference in the recurrent failures (13 measures, Holm p = 1). Those were
    simple image statistics, not the IQA model.
  - Scalar reliability gating failed as RACAF.
  - IQA outputs are not cached for APTOS.
- **Gate (CPU, read-only).** Run IQA on the 730 validation images. Continue only if P's error rate
  in Usable+Reject is materially above Good (e.g. ≥ 1.5×, fixed before running).

### 6. P + anatomy-registered regional pooling (optic disc / fovea frame) — **POSSIBLE BUT WEAK**

- **Mechanism.**
  - A small OD/fovea heat-map regressor, trained on the IDRiD localisation labels. These are
    present locally at `datasets/IDRiD/localization/`.
  - It defines macular / ETDRS-zone or quadrant regions.
  - P's stage-3 map is pooled per region and passed through a zero-init residual to CORN.
- **Complementarity claimed.** Macular involvement and the 4-2-1 quadrant rule are anatomical.
  P's global pooling has no explicit anatomical frame.
- **Literature: published.** Anatomy-anchored pooling (PARAGraph 2026); ETDRS-overlay pipelines.
- **Against.**
  - S1: image-frame 4×4 and 2×2 grids added nothing (Δ −0.003 / −0.008), and APTOS fundus images
    are roughly position-standardised.
  - The Stage-4 maps are weak, and lesion priors add < 0.01 (§22).
  - Training on IDRiD localisation images also uses up IDRiD as an independent external set.

### 7. P + feature distillation from frozen DINOv2 — **POSSIBLE; conditional on 1**

- **Mechanism.** An auxiliary loss aligns P's `pooled_final` (via a projection) with DINOv2 CLS
  during fine-tuning.
- **What it adds.** It regularises P toward the SSL prior with no second model at inference.
- **Literature.** Established (foundation-model-to-CNN distillation).
- **When it matters.** Only as the deployment form of candidate 1, after 1 shows the information is
  complementary.

### 8. P + lesion-guided native-resolution crops (Zoom-in-Net-style MIL) — **TOO EXPENSIVE for the evidence**

- **Mechanism.** Stage-4 maps pick native-resolution patches, which a local encoder reads with MIL
  pooling.
- **Against.**
  - The frozen 1024-px screen was null, and 0 vs 1 got slightly *worse* (§31).
  - The Stage-4 guidance is weak.
  - Crop pipeline, MIL and about 4× compute make this a week-scale job.

### Rejected

| Combination | Class | Reason |
|---|---|---|
| P + vessel geometry (tortuosity, fractal dimension, calibre from LWNet) | NOT ACTUALLY COMPLEMENTARY | Same Stage-3 source as the closed vessel route; PL vessel prior null; P already sees vessels in RGB; biomarker associations are population-level, not grading-level |
| P + lesion multi-task / auxiliary segmentation on Stage-4 pseudo-masks | ALREADY CLOSED / REDUNDANT | §22 shows Stage-4 information is redundant with P when given directly. Supervising P toward the same weak maps (HE Dice 0.13) adds no information and reinforces the burden axis the grade-4 failures lack |
| P + lesion statistics (late fusion) | REDUNDANT | Architecture A gate; §22; matched 10-statistic probe 0.63 (§27) |
| P + RETFound (any variant) | ALREADY CLOSED | §32 |
| P + spatial / multi-scale / MIL readout | ALREADY CLOSED | §23 |
| P + wavelet / FFT / LBP texture branch | NOT ACTUALLY COMPLEMENTARY | A pretrained CNN already encodes texture; no project evidence of a gap |
| P + SWA / EMA / TTA / seed ensemble | NOT AN INFORMATION COMBINATION | Addresses L4 variance only. Could be adopted uniformly as protocol hygiene, but cannot be a contribution |
| P + supervised / ordinal contrastive loss | TOO EXPENSIVE (protocol) | Batch size 2 makes in-batch contrast infeasible without a memory bank |
| P + reliability / uncertainty-gated fusion | ALREADY CLOSED | §§1–5 |
| P + PDR-sign channels (NV / PRH / laser) | TOO MUCH NEW DATA | Needs external annotated data (FGADR / Retinal-Lesions); blocked on licence |

---

## D. Worth testing

1. **P + frozen DINOv2 late fusion** (candidate 1).
2. **P + soft ordinal CORN targets** (candidate 2).
3. **Their combination** (candidate 3), only if both pass individually.

The IQA gate (candidate 5) is a free read-only side check. It is not a third experiment.

## E. Minimum experiments

Common to all:
- APTOS split `bc80fd45…`; validation (730) scored once per arm; seeds 42/123/2026.
- **Primary endpoint:** the mean of the four CORN cumulative-threshold AUROCs (P(Y > k), for
  true > k vs ≤ k, k = 0–3). It is threshold-free and covers the whole scale.
- **Key secondaries:**
  - ≥ 3 vs ≤ 2 and 2 vs 3 AUROC;
  - QWK and MAE;
  - per-grade recall;
  - grade-4 vs 0–2 AUROC.
- **Guardrails (3-seed mean):**
  - QWK ≥ −0.01;
  - grade-3 recall ≥ −0.10 (the §10 lesson);
  - grade-4-vs-(0–2) AUROC ≥ −0.01.
- APTOS validation has been reused many times. Any positive result is a development result, not
  confirmation.

**E1 — DINOv2 complementarity (CPU; no training).**
- Fit a CORN head (the C1 fitter: L2 1e-4, sqrt-inverse-frequency weights) on the **training** DINOv2
  features.
- Score validation.
- Fuse by averaging P(Y > k) with P_s's **saved** validation outputs, at a fixed weight of 0.5. No
  tuning, and P's training features are never used (§27 contamination).
- Controls:
  - P_s ⊕ P_s′, the ensembling control (for each seed pair);
  - P_s ⊕ frozen-ImageNet-ConvNeXt probe, which needs the 730 validation features extracted once.
- **GO** if P ⊕ DINO beats **both** controls in 3/3 seeds, with mean Δ ≥ +0.005 on the primary, the
  seed-pooled paired bootstrap CI excluding 0, and the guardrails holding. Otherwise **DROP**.

**E1b (only if E1 is GO) — trainable fusion.**
- Design: P plus a zero-init residual `Dense(1024→4)` on LN(DINOv2 CLS), with the DINOv2 features
  cached for the 8 dihedral augmentation views.
- Control: the same architecture with the DINOv2 rows **deranged** (Sattolo), which keeps capacity
  identical and destroys the information.
- 6 runs.
- Decides whether joint training adds anything beyond late fusion. **Optional:** late fusion alone
  is already a usable system.

**E2 — soft ordinal targets.**
- 3 runs of P with expected-CORN-NLL under Gaussian q (σ fixed before training), against the 3
  existing P runs.
- **SUPPORTIVE** if the primary Δ is > 0 in 3/3 seeds, the mean is ≥ +0.005 and the guardrails hold.
- **Mechanism check (reported, not gating):** the train − validation QWK gap and ECE fall.

**E3 (only if E1 GO and E2 SUPPORTIVE) — combined arm.**
- 3 runs of P + soft targets + DINOv2 fusion, in the form that passed E1 / E1b.
- Read as a 2×2 factorial: additivity vs redundancy of the two mechanisms.

**Side gate — IQA.** Frozen Stage-1 inference on the 730 validation images (CPU), then P's error rate
by quality class. Read-only.

## F. Order

1. **Day 0–1 (no GPU):**
   - E1, on existing features plus a small validation extraction for the ConvNeXt control;
   - the IQA side gate;
   - implement and unit-test the E2 loss.
2. **Day 1–2 (Colab):** E2's 3 runs. These are independent of E1 and can start as soon as the loss is
   tested.
3. **Day 2–3:** if E1 is GO, E1b (6 runs) or go straight to E3. If E1 is DROP, stop the DINOv2 line;
   E2 stands alone.
4. **Day 3–4:** E3, only if both passed. Record every result in the research record.

**Stop rule.** If E1 is DROP and E2 is not SUPPORTIVE, this direction is closed. Do not try further
fusion weights, σ values or seeds.

---

## Sources

- DINOv2 for DR staging (IOVS 2024, ARVO abstract): https://iovs.arvojournals.org/article.aspx?articleid=2800509
- REFA-DINO (EfficientNet-B0 + DINOv2 fusion): https://pmc.ncbi.nlm.nih.gov/articles/PMC13507030/
- CNN–ViT evidential hybrid for DR grading: https://arxiv.org/abs/2510.26315
- Galdran et al. 2020, non-uniform label smoothing for DR (TVST): https://pubmed.ncbi.nlm.nih.gov/32832207/
- ORDAC, ordinal adaptive correction under label noise: https://arxiv.org/abs/2509.02351
- CUFIT, curriculum fine-tuning of a VFM under label noise (NeurIPS 2024): https://arxiv.org/abs/2412.00150
- QMix, quality-aware learning with mixed noise (TMI): https://arxiv.org/abs/2404.05169
- Quality-aware collaborative learning / QGDR (Frontiers in Medicine 2026): https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1795484/full
- Fundus image quality-guided DR grading (MICCAI workshop 2018): https://link.springer.com/chapter/10.1007/978-3-030-00949-6_29
- Vessel tortuosity / fractal dimension in DR: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12164056/
- Retinal landmark / ETDRS-zone pipelines: https://link.springer.com/article/10.1007/s10278-026-02259-6
- Earlier sources for this folder: `SOURCES.md`, `NEXT_STEP_SOURCES.md`, `FINAL_DIRECTION_DECISION.md` §5 / A.6, `CANDIDATE2_PRETRAINED_BRANCH_RESEARCH.md`.
