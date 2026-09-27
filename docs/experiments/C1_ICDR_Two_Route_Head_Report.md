# C1 — ICDR two-route Stage-8 head vs CORN refit on the frozen representation

> **STATUS: PRE-REGISTERED, NOT YET RUN.** The implementation and unit tests are complete. No head
> has been fitted on real data and **no result exists**. The run needs a Colab T4 with Drive
> mounted, because the frozen backbones and caches are on Drive. It writes its own full report
> (`REPORT.md`) and all artifacts to `experiments/C1_ICDRTwoRouteHead/<timestamp>/`. The results
> sections below are to be transcribed from that run, never estimated.

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

**Not yet run.** To be transcribed from the run's `REPORT.md` and `results.json`.

## 15. Scientific interpretation

**Not yet run.** The permitted wording is fixed in advance:
- **SUPPORTIVE:** "The experiment supports the hypothesis that the CORN chain structure contributes
  to PDR under-triage on the frozen representation."
- **NOT_SUPPORTIVE:** "The experiment does not support the hypothesis that the output structure is
  the main limiting factor on the frozen representation."
- **TRADE_OFF / INCONCLUSIVE:** state exactly which endpoint moved and which guardrail or bar
  prevented a conclusion.

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

**Not yet run.**
