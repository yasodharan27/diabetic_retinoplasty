# P/PL — pretrained ConvNeXt-Tiny with vs without frozen Stage 3/4 priors

> **STATUS: PROTOCOL-LOCKED, IMPLEMENTED, NOT YET RUN.** No model has been trained and **no result exists**.
> All training runs on Google Colab (T4) through
> `colab/notebooks/pl_convnext_priors_experiment.ipynb`. The notebook writes its full report (`REPORT.md`,
> `results.json`, per-run CSVs) to `experiments/PL_ConvNeXtPriors/<timestamp>/` on Drive. The results
> sections below are to be transcribed from that run, never estimated.

**Code**
- `pl_convnext.py`: model builder, pinned weights, first-layer adaptation, head initialisation,
  optimiser exemption, metrics, decision rule.
- `tests/test_pl_convnext.py`: CPU tests.
- The notebook reuses the existing multiseed / Trainer infrastructure: resume, two-slot BEST, locks,
  stop decisions, `evaluate_arm_from_disk`.

**Source of truth:** the P/PL protocol-lock report. Nothing below changes it.

## Question

- **A (primary, controlled): PL − P.** Do frozen vessel/lesion segmentation-derived priors add
  useful information when supplied to the **same** strong pretrained CNN?
- **B: PL − the frozen `improved_multiseed_2026_09` NO_RACAF BEST reference.** A system-level
  comparison only. It does **not** isolate pretraining, and does not show that random initialisation
  caused earlier failures.

**P/PL is not a novel architecture.**

## Locked design

| | P | PL |
|---|---|---|
| Backbone | `keras.applications.ConvNeXtTiny`, ImageNet-1k, `include_top=False`, `include_preprocessing=False`, `pooling="avg"` | identical |
| Input | channels 0–2 of `stage5_input` (RGB) | channels 0–7: R, G, B, vessel, MA, HE, EX, SE |
| Normalisation | RGB: fixed ImageNet mean/std (equal to Keras PreStem on x·255) | same RGB; priors unchanged in [0,1] (no statistics) |
| First-layer kernel | pretrained 4×4×3×96 | 4×4×8×96 = [pretrained RGB \| zeros]; bias copied unchanged |
| Head | 768 → CORN Dense(768→4); GlorotUniform(seed = run seed); bias 0 | identical |
| Parameters | 27,820,128 + 3,076 | 27,827,808 + 3,076 (difference 7,680) |

**Pretrained weights**
- Source: `https://storage.googleapis.com/tensorflow/keras-applications/convnext/convnext_tiny_notop.h5`
- SHA-256: `d547c096cabd03329d7be5562c5e14798aa39ed24b474157cef5e85ab9e49ef1`
- Downloaded automatically once, SHA-verified, and cached on Drive under
  `experiments/PL_ConvNeXtPriors/pretrained_weights/`.
- Copied **by position** with shape checks. Names are unsafe because ConvNeXt LayerScale weights are
  `variable_N`, numbered by build order.

**Training (identical for P and PL)**
- Loss: weighted CORN with `PREREGISTERED_CLASS_WEIGHTS`.
- Optimiser: AdamW, learning rate 1e-4, weight decay 0.05. **No decay on any 1-D parameter**
  (biases, LayerNorm, LayerScale).
- Schedule: ReduceLROnPlateau (4, 0.5, 1e-6); EarlyStopping (12); at most 50 epochs; batch 2;
  mixed_float16.
- Checkpoints: BEST by val_QWK; LAST also evaluated.
- Augmentation unchanged.
- Split SHA `bc80fd45…`; 2,921 train / 730 validation.
- **Runs:** P-42, PL-42, P-123, PL-123, P-2026, PL-2026. Each is run once; none is selected or
  discarded.

**Pre-flight (all must pass before any epoch)**
1. GPU present.
2. Drive paths exist.
3. The pinned weights load.
4. P weights equal the reference, and P matches Keras' own preprocessed path.
5. PL's RGB first-layer slice is exact.
6. PL's five extra first-layer channel kernels are zero.
7. Initial PL(x8) equals P(xRGB).
8. Channel order is correct.
9. Cached channels lie in [0,1].
10. The split hash and population match.
11. Stage 3/4 models and caches are unchanged (fingerprint).
12. The CORN head initialisation is identical for P and PL.
13. The weight-decay exclusions are correct.

**Endpoints and decision rule**
- **Primary:** per-seed Δ AUROC of P(grade ≥ 3), grade 4 vs 0–2, BEST checkpoint, for A and B.
- **Secondary:**
  - bootstrap 95% CI (C1 procedure: 2,000 stratified paired resamples, seed 20260927);
  - grade-4 sensitivity at 95% specificity among grades 0–2;
  - QWK, MAE, grade-3 recall, false-urgent rate;
  - BEST vs LAST.
- **Guardrails** (3-seed mean): QWK ≥ −0.02, grade-3 recall ≥ −0.10, false-urgent rate ≤ +0.02.

| Verdict | Condition |
|---|---|
| **SUPPORTIVE** | mean Δ ≥ +0.03, and Δ > 0 in 3/3 seeds, and guardrails hold |
| **NOT_SUPPORTIVE** | mean Δ < +0.01, or Δ ≤ 0 in ≥ 2/3 seeds |
| **INCONCLUSIVE** | otherwise |

Applied independently to A and B.

**Leakage**
- ImageNet-1k pretraining only.
- Validation is used for nothing except scoring and the standard BEST rule.
- Stage 3/4 are read-only and fingerprinted.
- IDRiD: the §18 exclusions apply to both arms.
- DDR is not used; it is external only, after its own audit.
- The persistent-21 set is not an endpoint.

## Results

**Not yet run.** To be transcribed from the notebook's `REPORT.md`.

## What can and cannot be claimed

- **If A is SUPPORTIVE:** "Frozen segmentation-derived vessel/lesion priors improve grade-4-vs-(0–2)
  discrimination of a pretrained ConvNeXt-T on APTOS development data", pending IDRiD/DDR
  confirmation.
- **Not claimable:**
  - novelty;
  - that B isolates pretraining;
  - that random initialisation caused previous failures;
  - anything about grade 3 vs 4 (descriptive only);
  - clinical PDR signs.
