# P/PL — pretrained ConvNeXt-Tiny with vs without frozen Stage 3/4 priors

> **STATUS: COMPLETED (2026-09-28/29). Pre-registered verdicts: A (PL − P) NOT_SUPPORTIVE;
> B (PL − frozen NO_RACAF) INCONCLUSIVE.**
> Run: `experiments/PL_ConvNeXtPriors/2026-09-28_05-22-44/` on Drive, commit `33b1ce4`. All training ran on
> Colab (T4). The results below are transcribed from that run's `REPORT.md` and verified against its
> `results.json` and per-epoch histories. Research record: §22.

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

**Integrity.** All 13 pre-flight assertions passed (P = the pretrained reference to 3.6e-6; initial
PL = P exactly; channel order and [0,1] ranges; split 2,921 / 730; Stage 3/4 fingerprint unchanged; the
1-D weight-decay exemption including all 18 LayerScale weights; mixed precision with loss scaling).
6/6 runs completed, all by early stopping (16–31 epochs). **No protocol deviations.**

**Six runs (BEST checkpoint)**

| run | AUROC P(≥3), grade 4 vs 0–2 | sens @ 95 % spec | QWK | MAE | grade-3 recall | false-urgent | BEST epoch (0-based) |
|---|---|---|---|---|---|---|---|
| P-42 | 0.9588 | 0.741 | 0.9184 | 0.178 | 0.615 | 0.038 | 18 |
| PL-42 | 0.9617 | 0.810 | 0.9215 | 0.185 | 0.462 | 0.032 | 12 |
| P-123 | 0.9489 | 0.776 | 0.9148 | 0.201 | 0.513 | 0.032 | 3 |
| PL-123 | 0.9585 | 0.776 | 0.9138 | 0.189 | **0.103** | 0.006 | 6 |
| P-2026 | 0.9410 | 0.741 | 0.9176 | 0.190 | 0.513 | 0.036 | 6 |
| PL-2026 | 0.9515 | 0.793 | 0.9177 | 0.181 | 0.410 | 0.025 | 17 |
| frozen NO_RACAF 42 / 123 / 2026 | 0.9019 / 0.8907 / 0.9059 | 0.483 / 0.466 / 0.483 | 0.830 / 0.807 / 0.827 | 0.311 / 0.344 / 0.396 | 0.385 / 0.410 / 0.513 | 0.021 / 0.028 / 0.100 | — |

**3-seed means:** P AUROC 0.9496 ± 0.0089; PL 0.9572 ± 0.0052; frozen reference 0.8995.

**A. PL − P (primary, controlled)**
- Δ AUROC per seed: +0.0029 (95 % CI −0.006 to +0.012), +0.0095 (−0.004 to +0.025), +0.0105
  (−0.004 to +0.027).
- Mean **+0.0076** (SD 0.0041); seed-mean bootstrap 95 % CI −0.0003 to +0.0164; positive in 3/3 seeds.
- Secondary mean Δ:
  - AUROC P(4), 4 vs 0–2: +0.0065;
  - grade-4 sensitivity at 95 % specificity: +0.040;
  - AUROC P(≥3), grade 3 vs 0–2: −0.0175;
  - MAE −0.005; grade-4 recall +0.035; grade-4 precision +0.029;
  - grade-4 images decoded ≤ 2: +2.3.
- Guardrails: QWK +0.0007 (holds); **grade-3 recall −0.222 (fails; bound −0.10)**; false-urgent
  rate −0.014 (holds).
- **Pre-registered verdict A: NOT_SUPPORTIVE** (mean Δ < +0.01).

**B. PL − frozen NO_RACAF (system-level only)**
- Δ AUROC per seed: +0.0598, +0.0677, +0.0455; mean **+0.0577**; seed-mean bootstrap 95 % CI +0.033 to +0.086.
- Guardrails: QWK +0.0965 (holds); **grade-3 recall −0.111 (fails)**; false-urgent rate −0.028 (holds).
- **Pre-registered verdict B: INCONCLUSIVE** (primary met, a guardrail failed).

**BEST vs LAST (AUROC).** LAST is below BEST in 6/6 runs.
- P: 0.9588 → 0.9474, 0.9489 → 0.9348, 0.9410 → 0.9362.
- PL: 0.9617 → 0.9371, 0.9585 → 0.9499, 0.9515 → 0.9507.
- PL − P at LAST: −0.0103 / +0.0150 / +0.0145 (mean +0.0064, descriptive).

**Interpretation (labelled; the verdicts are unchanged by it)**
1. The frozen vessel/lesion priors add at most a small, sign-consistent AUROC gain to a pretrained
   ConvNeXt-Tiny, below the pre-registered +0.01 floor, with a CI that touches zero.
   - At the default decode, PL calls fewer images grade 3 (recall −0.222, driven by seed 123 at 0.103).
   - Its grade 3-vs-(0–2) ranking is slightly worse.
   - The priors do not improve the severe-NPDR boundary.
2. **Descriptive and not pre-registered:** the RGB-only pretrained P exceeds the frozen NO_RACAF
   reference by AUROC +0.050, QWK +0.096, MAE −0.160, sensitivity at 95 % specificity +0.276,
   grade-3 recall +0.111 and false-urgent rate −0.014. This is the strong-baseline control the Stage
   5–8 re-audit found missing. It does **not** isolate pretraining, because backbone, pretraining and
   input all differ, and the validation set has been used many times before.
3. Training QWK reaches about 0.99 against about 0.91 on validation, so the models are heavily over-fitted.
4. Several BEST checkpoints are very early (P-123 at epoch index 3; P-2026 and PL-123 at 6), so
   checkpoint selection matters, as in §16.1.

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
