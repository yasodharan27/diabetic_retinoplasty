# Next-Step Recommendation — candidates, protocols, closed pathways

**Design only.** Nothing here is implemented or pre-registered on Drive. Each protocol below must be
frozen in its own `PREREGISTRATION.json` before it runs, and approved first. The pending task-3 vs
task-1 placebo experiment is running; **its outcome is unknown**, and every outcome-dependent choice
is taken from NEXT_STEP_DECISION_TREE.md.

---

## 1. Candidate directions (Part 6)

Classification uses only: **directly supported** / **plausible but unproven** / **weakly supported**
/ **currently unjustified**.

| # | Candidate | Mechanism | Supporting evidence | Evidence against | Addresses PDR → ≤ 2? | Addresses 3-vs-4? | Params | Compute | Data | Leakage risk | Overfit risk | Controls | Primary endpoint | External needed? | Confirms / falsifies | Class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | **Label-noise and ceiling audit (Step LN)** | quantify grade disagreement on identical images; bound the observable 3-vs-4 AUROC | 8/41 byte-identical pairs disagree; 6/12 at grade 3/4; Krause 2018; INSPIRED | small n; duplicates may be unrepresentative | **yes** (tests whether failing PDR images are label disagreements) | yes (ceiling) | 0 | CPU, ~1–2 h | existing APTOS files | none (no training, no label change) | none | exact-only vs near-duplicate estimates reported separately | 3/4 disagreement rate + implied ceiling | no | confirms/refutes that the plateau is noise-limited | **directly supported** |
| C2 | **External evaluation on IDRiD (Step EXT)** | one-shot test of APTOS findings on untouched data | validation reused §8–§14; literature norm | IDRiD 3/4 counts modest; same clinic as Stage 04 training | measures it externally | yes | 0 | T4 ~1–2 h | IDRiD grading (on Drive) | must exclude Stage-04 training images | none | frozen models only | outcome-specific (see Step EXT) | it *is* external | confirms/refutes generalisation | **directly supported** (as a need) |
| C3 | **`p_cond_3` variance / power audit (Step VAR)** | between-run spread of `p_cond_3`, minimum detectable effect | RACAF lifts `p_cond_3` +0.075 over B; aux runs spread 0.62–0.74 | — | no | yes (power) | 0 | CPU, minutes | saved per-sample files | none | none | — | SD of paired differences; MDE | no | decides whether APTOS 3-vs-4 experiments are resolvable | **directly supported** |
| C4 | **Decoupled CORN-head re-estimation on frozen E (Step cRT)** | refit only the CORN head on frozen E (Kang et al. 2020 cRT) | E probe 0.710 vs head 0.639 (3/3 seeds); BEST chosen by QWK | post-hoc probe; E on training images is over-fitted; gap ≈ 1 SE | no (tasks 0–2 refit too, but no new information) | yes | 0 new (refit 1,028) | T4 ~1 h + CPU | existing caches | E must come from the model's own training images only | moderate (390 boundary examples) | original jointly trained head on the same E | Δ AUROC `p_cond_3` | yes, if positive | head under-use | **plausible but unproven** |
| C5 | **PDR-specific lesion channels (NV/IRMA/VH/FP)** | new segmenter on FGADR / Retinal-Lesions, burden into the head | PDR defined by NV/VH; channels absent | heavy domain shift; NV hard; only 62 DR4 in Retinal-Lesions; premise unverified | yes, if failures are genuine PDR | indirectly | segmenter + ~2k | days | external pixel labels (public) | must keep IDRiD clean | high | capacity-matched + permuted-channel controls | 4-vs-rest recall at fixed spec. | yes | missing-feature hypothesis | **weakly supported** (pending C1) |
| C6 | **Non-chained grade-4 head** (4-vs-rest BCE head with its own decision) | grade-4 score not gated by tasks 0–2 | PDR → ≤ 2 failure | 4-vs-rest ranking already 0.88; aux 4-vs-rest pressure null; re-packages closed threshold route; loses rank consistency | yes, but as calibration | no | ~260 | full retrain | — | low | moderate | aux / threshold results | grade-4 recall with grade-3 guardrail | yes | — | **currently unjustified** |
| C7 | **Alternative ordinal family** (CORAL, CLM, softmax, unimodal, hybrid) | different output structure | general DR ordinal literature | none addresses this failure; CORAL/CLM worse for non-monotonic burden | no | no | small | full retrain | — | low | moderate | — | — | yes | — | **currently unjustified** |
| C8 | **Decode change** (report / decide with `p_cond_3`; sequential decode) | use the correct 3-vs-4 estimand at decision time | E10 (5/6), 9/9 aux runs | still requires `p_gt_2 > 0.5`; decision-rule family closed as a *solution* (§10) | no | reporting only | 0 | none | — | none | none | — | — | yes | — | **directly supported as a reporting convention; unjustified as a fix** |
| C9 | **Noise-robust training** (e.g. confident-learning reweighting) | train around suspected label errors | depends on C1 | would change the training signal; must not alter labels | possibly | possibly | 0 | full retrain | — | high if validation predictions feed training | moderate | — | — | yes | — | **plausible but unproven** (only after C1) |
| C10 | Pretrained foundation backbone | replace the random-init Swin | RETFound APTOS AUROC 0.943 | global confound (92 % of parameters); not a 3-vs-4 mechanism | unknown | unknown | ~300M | large | public weights | low | low | full ablation | — | yes | — | **currently unjustified for this question** |

**Reporting convention (no experiment needed):** from now on, report `p_cond_3` as the model's 3-vs-4
score, with `p_gt_3` and the 4-vs-rest AUROC alongside. This is mathematics, not a result
(NEXT_STEP_AUDIT §3).

---

## 2. Selected next step per pending outcome (Part 7)

| Pending outcome | Next step | Type |
|---|---|---|
| SUPPORTIVE | **Step EXT** — IDRiD external evaluation of frozen T/P/B | external evaluation |
| NONSPECIFIC | **Step VAR** — `p_cond_3` variance/power audit; close task weighting | read-only audit |
| NOT_SUPPORTIVE | **Step cRT** — decoupled CORN-head re-estimation on frozen E | small experiment (head only) |
| TRADE_OFF | **Step EXT** with identical guardrails | external evaluation |
| INCONCLUSIVE | **Step VAR** → pre-registered power decision | read-only audit |
| **any outcome, now** | **Step LN** — label-noise and ceiling audit | read-only audit |

Why these and not another variant of E14: every selected step either uses **no training** (LN, VAR,
EXT) or trains **only the 1,028-parameter head** (cRT). Each tests a *different* mechanism from E14
(noise ceiling, external generalisation, variance, head optimisation). None re-tunes β or adds seeds.

---

### Step LN — read-only label-noise and ceiling audit (run now, whatever the E14 outcome)

- **Hypothesis.** Disagreement between APTOS grade labels at the grade-3/4 boundary is large enough
  (symmetric flip ≥ 0.2) that the maximum observable 3-vs-4 AUROC on APTOS labels is ≤ 0.80.
- **Intervention.** None. Read-only analysis.
- **Control.** Agreement at other grade boundaries (0/1, 1/2, 2/3) from the same duplicate groups.
- **Seeds.** n/a. Bootstrap seed 20260926 for any CI.
- **Data.** `dataset_splits/aptos2019_train_val_split.csv`; raw APTOS images (Drive, read-only); the
  pinned 41-pair list; every saved `per_sample_best.csv` (B ×3, RACAF ×3, aux ×9, and T/P ×6 once
  complete); Phase-0 `frozen_logits_grade34.csv`. **No label is changed. No image is written.**
- **Analyses.**
  1. Exact duplicates across the **whole** APTOS set (train–train, val–val, train–val) by MD5 of the
     file bytes, confirmed by full byte compare, extending the existing train–val audit.
  2. Near-duplicates: 64-bit perceptual hash on the raw image, pre-registered Hamming distance ≤ 6.
     Reported **separately** and labelled "unconfirmed"; never merged with exact duplicates.
  3. Grade-agreement matrix per duplicate group; disagreement rates with Clopper–Pearson CIs, overall
     and for groups touching grade 3 or 4.
  4. Noise ceiling for 3-vs-4 AUROC under the estimated flip rates, using the formula in
     NEXT_STEP_AUDIT §5, at the point estimate and both CI bounds. Contamination variant: "4" labels
     whose duplicate is ≤ 2.
  5. Validation-only model consensus (all predictions there are out-of-sample): images labelled 4
     that ≥ 80 % of saved models decode ≤ 2, and images labelled 3 with mean `p_cond_3` > 0.7. Output
     a review sheet (id, label, model votes, duplicate status) for **visual** inspection by the user
     for laser scars, regressed PDR or poor quality. Counts only; no relabelling.
  6. Descriptive confident-learning estimate (Northcutt et al. 2021) of the validation noise matrix
     from the ensemble's out-of-sample class probabilities. Descriptive only.
- **Primary endpoint.** Exact-duplicate disagreement rate among groups touching grade 3/4, and the
  implied symmetric-flip ceiling on 3-vs-4 AUROC (point and CI).
- **Secondary endpoints.**
  - Rates at other boundaries.
  - Near-duplicate rates.
  - The consensus-≤ 2 PDR count.
  - The confident-learning matrix.
  - Overlap between consensus-≤ 2 PDR images and label conflicts.
- **Pre-registration rule.**
  - **CEILING_LIMITED** if the point estimate of the 3/4 flip rate is ≥ 0.20 (ceiling ≤ 0.80).
  - **NOT_CEILING_LIMITED** if the CI upper bound is < 0.10 (ceiling ≥ 0.90).
  - Otherwise **UNRESOLVED**.
- **Success / failure.** No success/failure: this is a measurement. Its classification governs how
  every later 3-vs-4 number is read.
- **Stopping rule.** Single pass; no re-analysis with different thresholds.
- **Expected compute.** CPU only, ~1–2 h (hashing 3,662 files on Drive).
- **IDRiD.** Not used.
- **Pathway closure.** If CEILING_LIMITED, APTOS-only 3-vs-4 training experiments are closed as a
  research instrument. Future 3-vs-4 claims then require an adjudicated external set (DDR or IDRiD).

### Step EXT — IDRiD external evaluation (SUPPORTIVE / TRADE_OFF)

- **Hypothesis.** T's specific `p_cond_3` gain over P on APTOS replicates on IDRiD.
- **Intervention.** Inference only: the nine frozen BEST checkpoints (B, T, P × seeds 42/123/2026).
- **Control.** P (primary) and B (secondary), same images.
- **Seeds.** The three existing seeds; no new training.
- **Data.**
  - IDRiD disease-grading **train + test** (516 images).
  - **Exclude** any image byte- or perceptual-hash-identical to the 81 Stage-04 lesion-training images.
  - Resolve IDRiD's own train/test duplicate by keeping the test copy.
  - Grade counts are asserted at run time. If, after exclusion, fewer than 60 grade-3 or 40 grade-4
    images remain, the result is reported as **UNDERPOWERED** and cannot confirm.
  - Frozen Stage 02–04 on the raw images into a **new, dedicated IDRiD cache**. This creates caches for
    a new dataset; APTOS caches are untouched.
- **Primary endpoint.** Per-seed Δ AUROC `p_cond_3` (T − P), IDRiD grade 4 vs 3.
- **Secondary endpoints.**
  - T − B and P − B.
  - `p_gt_3` 4-vs-3.
  - Replication of the `p_gt_2` inversion and of conditional > marginal (B models).
  - 4-vs-rest AUROC, QWK, MAE, grade-3 and grade-4 recall.
  - PDR decoded ≤ 2.
  - Paired stratified bootstrap CIs (2,000).
- **Pre-registration rule.** **CONFIRMED** if mean(T − P) ≥ +0.03 and T > P in 3/3 seeds. **NOT
  CONFIRMED** otherwise. For TRADE_OFF, the same guardrails as APTOS are applied to T vs B.
- **Success.** CONFIRMED with guardrails preserved.
- **Failure.** Mean(T − P) ≤ 0, or NOT CONFIRMED.
- **Stopping rule.** One evaluation pass; no thresholds, no refitting, no checkpoint choice using
  IDRiD labels.
- **Expected compute.** T4, ~1–2 h (Stage 03/04 on ≤ 516 images plus nine model passes).
- **IDRiD.** This *is* the IDRiD use. It consumes IDRiD as the one-shot confirmatory set for this
  question. The separate RACAF IDRiD protocol is untouched.
- **Pathway closure.** NOT CONFIRMED closes task-3 weighting. For TRADE_OFF, if the guardrail failure
  replicates, abandon it.

### Step VAR — `p_cond_3` variance and power audit (NONSPECIFIC / INCONCLUSIVE)

- **Hypothesis.** Between-run variation of `p_cond_3` AUROC under loss or architecture perturbations
  is large relative to 0.03, making 3-seed APTOS experiments on this endpoint unable to resolve
  effects of that size.
- **Intervention / control.** None; read-only over every saved run (B, RACAF, aux, T, P; C1/C3 if
  per-sample files exist).
- **Data.** APTOS validation per-sample files only.
- **Primary endpoint.** SD of within-seed paired differences between independently perturbed runs
  (P − B, aux − B, RACAF − B, T − P). From it, the minimum detectable mean effect for a 3-seed
  paired design (80 % power, α = 0.05, one-sided sign-consistent rule simulated by bootstrap).
- **Secondary endpoints.** The same for `p_gt_3`, and the validation-sample SE (bootstrap) vs the
  between-run SD.
- **Pre-registration rule.**
  - **NOT_RESOLVABLE_AT_FEASIBLE_COST** if the MDE at 3 seeds exceeds 0.05 **and** more than 10 seeds
    per arm would be needed for 80 % power at 0.03.
  - Otherwise **RESOLVABLE**, with the required seed count stated.
- **Success / failure.** A measurement. It decides whether an INCONCLUSIVE verdict may be followed
  by a confirmatory run.
- **Stopping rule.** Single pass.
- **Expected compute.** CPU, minutes.
- **IDRiD.** Not used.
- **Pathway closure.** NOT_RESOLVABLE closes APTOS-validation training experiments on 3-vs-4 until
  an external or larger labelled set is available.

### Step cRT — decoupled CORN-head re-estimation on frozen E (NOT_SUPPORTIVE)

- **Hypothesis.** The jointly trained CORN head under-uses E at the conditional 3/4 boundary.
  Re-estimating only the head on frozen E raises `p_cond_3` 4-vs-3 AUROC by ≥ 0.03 without degrading
  the ordinal task.
- **Intervention.**
  - Freeze each B BEST checkpoint (seeds 42/123/2026).
  - Extract E (256-d) for every cached training image (2,921) and validation image (730),
    unaugmented, inference only.
  - Re-fit the CORN head Dense(256→4) on **training E only**, with the unchanged weighted-CORN loss.
    Full-batch L-BFGS, initialised at the trained head, L2 = 1e-4 fixed in advance, stopped at
    |Δloss| < 1e-7 or 2,000 iterations. No tuning.
- **Control.** The original jointly trained head on the same E (identical to B's outputs).
- **Seeds.** 42, 123, 2026 (backbones). The head fit is deterministic.
- **Data.** Authoritative APTOS split; the validation set is scored once.
- **Primary endpoint.** Δ AUROC `p_cond_3` (re-estimated − original), grade 4 vs 3, APTOS validation.
- **Secondary endpoints.**
  - The E14 guardrails (QWK, MAE, grade-3 recall, 4-vs-rest AUROC).
  - `p_gt_2` inversion (does it persist?).
  - PDR decoded ≤ 2.
  - A comparison with the Phase-0 PCA-16 probe level (0.710).
- **Pre-registration rule.** **SUPPORTIVE** if mean Δ ≥ +0.03, Δ > 0 in 3/3 seeds, and the guardrails
  hold. **NOT_SUPPORTIVE** if mean Δ < +0.01 or Δ ≤ 0 in ≥ 2/3 seeds. **INCONCLUSIVE** otherwise.
- **Success.** SUPPORTIVE → then Step EXT for the re-estimated heads.
- **Failure.** NOT_SUPPORTIVE closes head-level explanations. The loss is then located at the label
  or feature level (mechanisms 7/8), and the next question belongs to Step LN's outcome.
- **Stopping rule.** One fit per seed; no hyperparameter variation.
- **Expected compute.** T4 ~1 h for E extraction (3 seeds × 3,651 images); CPU minutes for the fits.
- **IDRiD.** After, and only if SUPPORTIVE.
- **Pathway closure.** As above.
- **Known risk, stated in advance.** E on training images comes from a backbone that fit those
  images, so a head re-fit on it can be over-confident. That is why validation is the only scoring
  set and why the jointly trained head on the *same* E is the control.

---

## 3. Closed / Do Not Reopen Unless New Evidence Appears (Part 8)

| Pathway | Why closed | What would reopen it |
|---|---|---|
| RACAF / reliability fusion (incl. C1, C3, D6, shuffled-κ) | six-run INCONCLUSIVE; gate never learned; reliability contributes 0.002–0.017 QWK; D6 negative | a reliability signal shown to predict error out of sample |
| Generic-vessel PDR route | 4-vs-3 AUROC 0.572 (CI includes 0.5); negative increment | an NV-specific (not generic-vessel) representation |
| Threshold-only solution | moves along a poor curve; grade-3 recall −61 %; QWK blind to it | none: calibration finding only |
| Architecture A (lesion-burden pathway) | gate INCONCLUSIVE; the representation already exceeds the statistics | an independent resolution of its gate showing a representation-level loss |
| Aux loss as a solution (marginal grade-4 BCE) | NOT_SUPPORTIVE on its pre-registered endpoint; conditional lift confounded | none within this model family |
| Task-3 weighting | **pending E14** (closes under NONSPECIFIC / NOT_SUPPORTIVE, or if EXT does not confirm) | — |
| Indiscriminate hyperparameter tuning (β, λ, seeds) | validation set reused; multiplicity | a pre-registered, power-justified design (Step VAR) |
| Another generic lesion↔global cross-attention module | reproduces the suspected aggregation; RACAF/C1/C3 null at fusion points | evidence of a representation-level loss at Stage 07 (Phase 0: 1/3 seeds only) |
| Replacing CORN with CORAL / CLM / softmax / unimodal | none addresses the observed failure; single-direction families are expected to be worse | a controlled comparison showing a specific gain at the 3/4 boundary |

---

## 4. Research discipline (Part 9)

- No post-hoc endpoint switching. Every verdict comes from its pre-registered primary endpoint.
- No tuning without a pre-registered protocol. No repeated experiments because a result disappoints.
- **APTOS validation is no longer an untouched test set for 3-vs-4.** Any positive result needs
  external confirmation.
- **IDRiD stays separate and single-use.** No selection, threshold or tuning on IDRiD labels. It must
  never also be used for training.
- One mechanism per experiment unless a factorial design is pre-registered.
- No implementation before the design is approved. All frozen checkpoints and historical results stay
  untouched.

---

## 5. Final recommendation

**The next step should be a read-only audit: Step LN, the label-noise and ceiling audit.** Run it now,
independently of the pending experiment, on CPU, without training or relabelling.

Why this, after the literature review:
- **It is the only mechanism with direct, unexamined, same-image evidence.** Byte-identical images
  carry different grades in 8/41 pairs, and in 6/12 pairs touching grades 3 and 4.
- The literature treats reference-standard noise as decisive in DR evaluation (Krause 2018), and
  severe NPDR as the least reliably graded category.
- It bears on **both** open problems: the 3-vs-4 AUROC plateau (≈ 0.64–0.71 across every readout,
  inside the range a 20–30 % flip rate produces) and the PDR → ≤ 2 failure (4/11 grade-4 duplicate
  pairs have a twin graded ≤ 2).
- It costs no GPU, changes nothing, and sets the power and ceiling constraints that every later
  experiment must respect.

After the pending experiment reports, follow the table in §2: external evaluation (SUPPORTIVE /
TRADE_OFF), read-only variance audit (NONSPECIFIC / INCONCLUSIVE), or a head-only re-estimation
(NOT_SUPPORTIVE).

A PDR-feature data investigation (C5) is **not** recommended until Step LN shows that the failing PDR
images are genuine, untreated PDR rather than label disagreements. No new data acquisition is
justified by the current evidence.
