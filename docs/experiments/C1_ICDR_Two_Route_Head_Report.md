# C1 — ICDR two-route Stage-8 head vs CORN refit on the frozen representation

> **STATUS: COMPLETED (2026-09-27). Pre-registered verdict: NOT_SUPPORTIVE.**
> Run: `experiments/C1_ICDRTwoRouteHead/2026-09-27_16-37-56/`, repo commit `1f6644e`. The results
> below are transcribed from that run's `REPORT.md` and `results.json`. The full detailed analysis
> (integrity, fit diagnostics, confusion matrices, per-image persistent-21 table) is in that
> `REPORT.md`. Research record: §20.

**Code**
- `icdr_two_route_head.py`: the heads, the fitter and the probability construction.
- `icdr_two_route_experiment.py`: the resumable pipeline steps, the analysis and the decision rule.
- `colab/notebooks/stage08_icdr_two_route_head.ipynb`: the Colab execution interface. It freezes the
  pre-registration, re-runs the test suite on the runtime, resumes per-seed E extraction after a
  disconnect, and prints the full report with a detailed analysis appendix. The run is single-use.
- `tests/test_icdr_two_route_head.py`: 36 tests.

**Naming.** The files are called `icdr_two_route_*` because `racaf_c1_control_model.py` already
uses "C1" for the unrelated RACAF gate control.

**Design record:** `research/Grade3vs4_Architecture_Research/FINAL_DIRECTION_DECISION.md`
(Addendum A), and research record §19–§19.1.

---

## 1. Research question

Does the sequential CORN output structure itself contribute to PDR under-triage? Specifically: are
grade-4 images decoded ≤ 2 when their NPDR lesion burden is low, even though PDR evidence is present
in the representation?

## 2. Hypothesis

The CORN continuation-ratio chain gates grade-4/PDR predictions. Changing **only** the Stage-8
output structure, on the identical frozen Stage-7 representation E, reduces this gating.

## 3. Why C1 was selected

- The 3-vs-4 signal is present in E (probe AUROC 0.71, §13) and is lost downstream of E (§13.3).
- p_gt_2 is inverted for grade 4 vs grade 3 (§13.2, §14.1).
- 21/58 grade-4 validation images are decoded ≤ 2 by all three NO_RACAF seeds, with low p_gt_2 but
  high p_cond_3 (§15).
- Thresholding only trades one error for another (§10).
- Objective manipulations were non-specific (§14, §16).
- The 3-vs-4 endpoint on 97 images is unresolvable at feasible cost (§17).

**C1 is NOT claimed as a new ordinal model family.** Splitting off one category first (a hurdle, or a
sequential split tree) is an established statistical and modelling idea. The contribution is the
mechanism-driven DR application:
- identifying CORN chain gating;
- restructuring the output according to the ICDR distinction between PDR and NPDR severity;
- testing it on an identical frozen representation against a capacity-matched control.

## 4. Architecture (Stage 8 only)

| Head | Architecture | Parameters |
|---|---|---|
| H0 | the original jointly trained CORN head (saved outputs) | 1,028 |
| H1 | CORN refit: Dense(256→4), four CORN tasks over grades 0–4 | **1,028** |
| H2 | two-route: PDR Dense(256→1) → q; NPDR CORN chain Dense(256→3) over grades 0–3 | **1,028** |

- **H2 probabilities:** P(4) = q; P(k) = (1 − q)·P_NPDR(k) for k = 0–3. The distribution sums to 1.
- **H2 decode:** grade 4 if q > 0.5 (fixed, not tuned); otherwise the CORN decode over grades 0–3.
- **Scores used by the endpoints:**
  - P(grade ≥ 3): p_gt_2 for H0/H1; P(3) + P(4) for H2.
  - P(grade 4): p_gt_3 for H0/H1; q for H2.

## 5. Parameter matching

`icdr_two_route_head.parameter_parity_report()` builds all three Keras heads and raises unless each
has exactly 1,028 trainable parameters.
- Measured: original CORN 1,028 (2 tensors), H1 1,028 (2 tensors), H2 1,028 (4 tensors).
- Difference: **0**.
- The runner re-checks this before any fit.

## 6. Training population and fitting

- **Training population.** Only the authoritative APTOS training split: 2,929 split entries
  (1444/296/799/154/236), split sha256 `bc80fd45…`.
  - As cached, the split excludes the empty field-of-view images that have no cached
    representation: 11 over the whole split, 3 of them in validation (733 → 730). The images the
    frozen backbones were trained on are the split minus these pinned ids. The run takes that
    population from the six-run manifest (`n_train_yielded`, `empty_fov_ids`) rather than from a
    hard-coded number, and stops on any other mismatch.
  - H1: all cached training images.
  - H2 PDR route: all cached training images.
  - H2 NPDR route: the grade 0–3 cached training images only. Grade-4 images give no NPDR
    supervision.
  - The exact counts are printed and recorded by the run.
- **Correction.** An earlier draft of this report stated 2,929 / 2,693. Those are split counts,
  not the cached population. This was corrected before any data were touched.
- **One shared fitter for every binary task of both heads.**
  - L2-regularised, class-weighted logistic regression.
  - Loss = Σ wᵢ·BCEᵢ / n_task + (L2/2)·‖β‖², with the bias unpenalised.
  - scipy L-BFGS-B from a zero start: convex and deterministic.
  - With a linear head the CORN tasks share no parameters, so per-task fitting *is* the CORN
    likelihood decomposition.
- **Fixed values, no tuning:**
  - L2 = **1e-4**, fixed in advance (NEXT_STEP_RECOMMENDATION.md, Step cRT).
  - Class weights: `weighted_corn.PREREGISTERED_CLASS_WEIGHTS` (square-root inverse frequency,
    training counts).
  - No oversampling, undersampling or batch balancing.
- **Standardisation.** Training-split mean/SD only, identical for H1 and H2, folded back into the
  Dense kernel and bias so both heads act on raw E. The standardiser is unsupervised
  preprocessing shared by both heads; it carries no labels.

## 7. Leakage controls

- **Validation is never used for fitting.** It is not used for scalers, class weights, L2, the q
  threshold or model choice. It is scored once.
- **The backbones are frozen and unchanged.** Each BEST checkpoint:
  - is validated against the SHA-256 in its sealed manifest;
  - is loaded read-only, frozen, and confirmed to have zero trainable variables;
  - is checked after extraction: weight-file SHA-256 and per-stage weight fingerprints must be
    unchanged.
- **E is proven to be the original head's input.** The original head applied to E must reproduce
  the model logits, and the model logits must reproduce the saved logits (tolerance 0.05, the
  Phase-0 value).
- **No caches are regenerated.** A population mismatch stops the run.
- **DDR and IDRiD are not used.**
- **The run is single-use.** The runner refuses to start if a completed run already exists.

## 8. H0 / H1 / H2

- **H0:** the saved original results, kept for continuity.
- **H1:** the control. It accounts for refitting the head at all.
- **H2:** the treatment.
- **Scientific comparison:** H2 − H1.

## 9. Primary endpoint

Per seed (42 / 123 / 2026), Δ = AUROC(H2) − AUROC(H1) for **grade 4 vs grades 0–2**, where each head
is scored by its own P(grade ≥ 3).
- Reported: mean, SD, and a paired stratified bootstrap 95% CI of the seed-averaged Δ (2,000
  resamples, seed 20260927).

## 10. Secondary endpoints

- AUROC of P(4) (q vs p_gt_3), grade 4 vs 0–2.
- AUROC of P(≥3), grade 3 vs 0–2.
- AUROC of P(4), grade 4 vs rest and grade 4 vs 3 (descriptive).
- Grade-4 decoded-grade distribution; grade-4 recall and precision; grade-3 recall.
- False urgent calls on grades 0–2.
- QWK, MAE, confusion matrices.
- The persistent-21 images still decoded ≤ 2.
- Per-seed values, and the mean/SD across seeds.

## 11. Decision rule (pre-registered; not to be changed after results)

| Verdict | Condition |
|---|---|
| **SUPPORTIVE** | mean Δ ≥ +0.03, Δ > 0 in 3/3 seeds, and all guardrails hold |
| **NOT_SUPPORTIVE** | mean Δ < +0.01, or Δ ≤ 0 in ≥ 2/3 seeds |
| **TRADE_OFF** | SUPPORTIVE primary criteria met, but a guardrail fails |
| **INCONCLUSIVE** | anything else |

Guardrails are applied to the 3-seed mean of (H2 − H1), with inclusive bounds:
- grade-3 recall ≥ −0.10;
- grade-3-vs-(0–2) AUROC ≥ −0.03;
- QWK ≥ −0.02;
- false urgent rate ≤ +0.02.

## 12–14. Per-seed results, aggregate results, guardrails

**Population and integrity**
- Population: 2,921 train (grades 1444/296/791/154/236) and 730 validation; NPDR route 2,685; PDR
  route 2,921.
- Backbone weights and Stage 5–7 + CORN fingerprints unchanged in 3/3 seeds.
- Logit parity within tolerance: ≤ 7.8e-3 vs saved, ≤ 1.8e-2 for head(E) vs graph.
- All 24 task fits converged.

**Primary endpoint** (AUROC of each head's P(grade ≥ 3), grade 4 vs 0–2)

| seed | H0 | H1 | H2 | Δ (H2 − H1) | 95 % CI |
|---|---|---|---|---|---|
| 42 | 0.9019 | 0.9110 | 0.9120 | +0.0011 | −0.006 to +0.008 |
| 123 | 0.8907 | 0.9054 | 0.9031 | −0.0022 | −0.010 to +0.005 |
| 2026 | 0.9059 | 0.9077 | 0.9104 | +0.0027 | −0.002 to +0.007 |

Mean Δ **+0.0005** (SD 0.0025); paired bootstrap 95 % CI of the mean **−0.0044 to +0.0052**;
positive in 2/3 seeds.

**Secondary endpoints** (3-seed means)

| metric | H0 | H1 | H2 | mean H2 − H1 |
|---|---|---|---|---|
| AUROC P(4), grade 4 vs 0–2 | 0.904 | 0.891 | 0.897 | +0.0055 |
| AUROC P(≥3), grade 3 vs 0–2 | 0.906 | 0.943 | 0.938 | −0.0042 |
| AUROC P(4), grade 4 vs 3 (descriptive) | 0.577 | 0.591 | 0.596 | +0.0053 |
| Grade-4 recall | 0.161 | 0.316 | 0.270 | −0.0460 |
| Grade-4 precision | 0.473 | 0.515 | 0.517 | +0.0013 |
| Grade-3 recall | 0.436 | 0.479 | 0.385 | −0.0940 |
| False-urgent rate, grades 0–2 | 0.049 | 0.059 | 0.032 | −0.0274 |
| QWK | 0.821 | 0.846 | 0.836 | −0.0100 |
| MAE | 0.350 | 0.317 | 0.312 | −0.0055 |
| PDR decoded ≤ 2 (of 58) | 31.7 | 25.3 | 33.0 | +7.67 |
| Persistent-21 still ≤ 2 | 21 | 18.7 | 19.7 | +1.0 |

**Guardrails** (3-seed mean of H2 − H1; all hold):
- grade-3 recall −0.0940 (bound −0.10; seeds 0.000 / −0.128 / −0.154);
- grade-3-vs-(0–2) AUROC −0.0042;
- QWK −0.0100;
- false-urgent rate −0.0274.

**What changed between H1 and H2.**
- Grade-4 images moved from ≤ 2 to ≥ 3: **0 / 0 / 0**.
- Grade-4 images moved from ≥ 3 to ≤ 2: 8 / 8 / 7.
- Grades 0–2 no longer called urgent: 20 / 20 / 13.

## 15. Scientific interpretation

**The experiment does not support the hypothesis that the output structure is the main limiting
factor on the frozen representation.**
- **The null is precise.** The CI on the mean is about ±0.005, far narrower than the 3-vs-4
  endpoint's noise (research record §17).
- **The mechanism did not act as hypothesised.**
  - Removing grade-4 supervision from the NPDR chain and deciding PDR on its own route rescued no
    gated PDR image.
  - The dedicated PDR score q, a 4-vs-all classifier fitted directly on E, stays below 0.30 in every
    seed for 16 of the persistent-21 images.
  - Only `b3819a805dca` and `6cfb7b44ef6f` exceed 0.5 in any seed. These are the two cases §15.2
    flagged as having clear laser-scar-like spots.
- **The failure is therefore located in the representation E, not in the ordinal output structure.**
  E does not carry recognisable PDR evidence for these eyes.
- **Refit effect (descriptive, not pre-registered).** H1 − H0: QWK +0.025, grade-4 recall +0.155.
  This is recorded, not adopted.

## 15a. Negative-result contribution

A pre-registered, capacity-matched, representation-controlled test with a precise bound (±0.005).
It relocates PDR under-triage from the CORN output structure to the representation. Whether
end-to-end chain supervision shaped E (Level C) is not tested and remains future work.

**Interpretation correction (2026-09-27; numbers and verdict unchanged).** Where §15 and §15a say the
failure is "located in the representation E" or that the result "relocates" it there, read instead:
C1 showed that modifying the Stage 8 output structure did not recover the observed Grade-4
under-triage from the frozen Stage 5–7 representation. This motivates investigation of the upstream
representation pathway but does not establish that Stage 5–7 is definitively the bottleneck.
(Research record §20.1.)

## 16. Limitations

- **Development set, not a test set.** APTOS validation (58 grade-4 images) was used for this
  pre-registered development decision and in earlier analyses of these models. It is **not an
  untouched external test set**. External validation (DDR if it passes its audit, otherwise
  IDRiD) remains future work.
- **Possible over-confidence.** The heads are fitted on training-image E from backbones trained on
  those images. Both heads share this, so the contrast is fair but absolute levels may be optimistic.
- **Head-only scope.** A head-only change cannot test whether E itself was shaped by chain
  supervision during end-to-end training. A negative result does not rule that out.
- **Unlabelled failure set.** The persistent-21 images are not established as treated PDR or as
  label noise.
- **Nothing else changed.** Stages 5–7 were not changed, the representation was frozen, and DDR was
  not used.

## 17. Final verdict

**NOT_SUPPORTIVE** under the pre-registered rule: mean Δ +0.0005 < +0.01. The guardrails held.

C1 (the output-structure / chain-gating hypothesis on the frozen representation) is **closed**. It
is not rerun with another head, threshold, L2 or class weights.

DDR was not used in this experiment and remains reserved for later external evaluation.
