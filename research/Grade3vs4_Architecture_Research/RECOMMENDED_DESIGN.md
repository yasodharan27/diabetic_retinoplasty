# Recommended Design — Count-Preserving Lesion-Burden Pathway (LBP)

**Status: DESIGN ONLY. Nothing implemented, trained or pre-registered on Drive.** This document
is the draft protocol. Before any run, its frozen values must be copied into a
`PREREGISTRATION.json` that the experiment refuses to run without, following the pattern of
`grade4_aux_loss_experiment.ipynb`.

The running §12 auxiliary-loss experiment is independent of this design and must not be modified.
LBP trains at λ = 0 (weighted CORN only) whatever §13 finds, so that representation and
objective stay separable.

---

## 1. Hypothesis (one mechanism, stated so it can fail)

> The Grade-3/Grade-4 signal in the four frozen lesion channels is carried mainly by lesion
> **amount** (burden), which falls from severe NPDR to PDR. The current graph can pass lesion
> **presence** to the head but not amount. Stage 07's softmax cross-attention is
> cardinality-blind, and its queries come from an RGB-only encoder. Giving the CORN head a
> **count-preserving** path for image-level lesion burden will raise the grade 4 vs grade 3
> ranking of `p_gt_3`. It will do so *because of the information in that path, not because of
> the added parameters*, and without degrading the ordinal task.

It is falsified if the pathway does not beat an **identical pathway fed the burden statistics of a
different image**.

---

## 2. Phase 0 — read-only localisation (must run first; no training)

The architecture is only justified if the lost signal is (a) learnable from training data and (b)
not already present in E. Phase 0 answers both cheaply. Per the project's working style, it should
be **one standalone copy-paste script**, read-only, writing only to a new timestamped
`experiments/Grade3vs4_Phase0/<ts>/` directory.

| Probe | Question | Inputs | Compute |
|---|---|---|---|
| **P0.1** decode decomposition | Does the marginal product hurt? Is burden reversed 3→4 in *these* maps? | saved `per_sample_best.csv` (`p_gt_2`, `p_gt_3`) for the 6 runs; `per_image_variables.csv` | CPU, seconds |
| **P0.2** train-fitted lesion probe | Is the 0.670 learnable from the **training split**, not just within validation? | the 10 §11 statistics computed from cached lesion maps of the 390 training grade-3/4 images; the 97 validation images | CPU, reads ~1.6 GB of caches |
| **P0.3** representation probes | Where is the signal lost: Stage 05, Stage 07 or head/decode? | forward the frozen BEST NO_RACAF checkpoints (3 seeds) on train g3/g4 + val g3/g4; extract `E`, `GAP(L)`, `GAP_tokens(G)`, `z` | T4 recommended (~487 images × 3 seeds); CPU feasible, slow |
| **P0.4** stacked-head baseline | What does post-hoc stacking of `z` + lesion statistics achieve? | `z` from P0.3 + the statistics; fitted on train, scored on val | CPU |
| **P0.5** PRP-scar audit | Does EX-driven grade-4 signal come from laser scars? | 10 highest-EX grade-4 and 10 highest-EX grade-3 validation images, viewed by the user | manual |

Details:

- **P0.1.** Per run, AUROC (grade 4 vs 3) of `p_gt_2`, of the conditional `σ(z_3) = p_gt_3 / p_gt_2`,
  and of `p_gt_3`. Grade-wise medians (grades 2, 3, 4) of each of the 10 statistics.
- **P0.2.** Transform `log10(x + 1e-4)`, z-score with training-only constants, L2-logistic regression
  with C chosen by 5-fold CV *inside the training set*. Score once on validation. Report the 2,000×
  stratified bootstrap CI, and also the result with the 41 known duplicates excluded.
- **P0.3.** The same probe protocol on each representation. Every representation is 256- or
  1152-dimensional against 390 training samples, so use the same CV-chosen L2 penalty and
  also report a PCA-16 version to control for dimensionality.
- **P0.4.** The same probe on `[z (4) ‖ s (10)]`.

### Pre-registered Phase-0 gates

- **G1 (go/no-go).** P0.2 validation AUROC ≥ **0.62** and its bootstrap 95 % lower bound
  > **0.50**. If G1 fails, **stop**: the lesion-burden route is not learnable from training data,
  and Phase 1 is recorded as NOT JUSTIFIED. No architecture is built.
- **Interpretation of P0.3 (not a gate; it decides the secondary arm and how Phase 1 is read):**

| pattern (train-fitted val AUROC) | reading |
|---|---|
| probe(GAP(L)) ≈ P0.2 and probe(E) ≥ 0.05 lower | lost at **Stage 07 aggregation** → supports H1 → run secondary arm LBP-GAP |
| probe(GAP(L)) also ≥ 0.05 below P0.2 | lost in **Stage 05** (early fusion / BN at batch 2) or never learned → LBP bypasses it; LBP-GAP not run |
| probe(E) ≈ P0.2 but `p_gt_3` low | the representation has it and the **head/objective/decode** does not use it → §13 and P0.1 matter more than LBP; LBP still run, expected gain smaller |

- **P0.1 reading.** If the conditional `σ(z_3)` beats the marginal by ≥ 0.05 in ≥ 5/6 runs, the decode
  is a real loss point. It is recorded as its own finding (no threshold is proposed), and the
  conditional AUROC is promoted to co-primary in Phase 1. If burden does not fall from grade 3 to
  4 in these maps, the "reversal" part of the hypothesis is unsupported and only the "amount"
  part is tested.

---

## 3. Phase 1 — architecture (built only if G1 passes)

### 3.1 Exact graph

```
inputs: stage5_input (B,512,512,8), stage6_input (B,256,256,3), reliability (B,1),
        lesion_stats (B,10)                                      ← NEW 4th input, float32

— baseline graph, UNCHANGED and constructed FIRST, in the exact order of
  no_racaf_model.build_no_racaf_joint_model_matched_init() (incl. construct-and-discard RACAF) —
  L = Stage05(stage5_input)            (B,32,32,256)
  G = Stage06(stage6_input)            (B,64,1152)
  E = Stage07([L, G])                  (B,256)
  z = CORN(E)                          (B,4)

— NEW layers, constructed AFTER every baseline layer (so baseline init draws are bit-identical) —
  s  = lesion_stats                                 already transformed + standardised (see 3.2)
  u  = Dense(32, activation="gelu", dtype="float32", name="lbp_hidden")(s)
  u  = Dense(32, dtype="float32", name="lbp_embed")(u)
  Δz = Dense(4, use_bias=False, kernel_initializer="zeros", dtype="float32",
             name="lbp_logit_residual")(u)
  z' = z + cast(Δz, z.dtype)                        (B,4)
  z' = InertReliabilityConnection([z', reliability])   (unchanged, weight-free)

output: z'   — decoded by the UNCHANGED corn.decode_logits / CORNQuadraticWeightedKappa
```

| New tensor | Parameters |
|---|---|
| `lbp_hidden` 10→32 | 352 |
| `lbp_embed` 32→32 | 1,056 |
| `lbp_logit_residual` 32→4, no bias | 128 |
| **Total new** | **1,536** (43,043,336 → **43,044,872**, +0.0036 %) |

The residual is added to the *logits*, not concatenated into CORN's input. CORN's
`Dense(256→4)` therefore keeps its exact shape and initial draws, and each CORN task gets its own
signed burden weights (task 3 can learn the reversal). The residual is zero-initialised, so the
model at step 0 **is** the baseline function.

### 3.2 The statistics

- **Definition.** Exactly the 10 lesion variables of the §6/§11 diagnostic (`per_image_variables.csv`):
  per-channel `mean_prob` and `foreground_fraction` (> 0.5) for MA, HE, EX, SE, plus the two totals.
  The implementation must **reproduce the saved CSV's 10 columns for all 730 validation images to
  |Δ| ≤ 1e-6**. This ties the pathway to the exact features that produced E10.
- **Source.** The cached canonical (unaugmented) lesion maps. There is no cache regeneration.
  Image-global statistics are *exactly* invariant to the flip/rot90 augmentation (the pixel
  multiset is unchanged), so one precomputed table per image is exact for every epoch.
- **Transform.** `log10(x + 1e-4)`, then z-score with mean/SD from the **training split only**,
  frozen into the pre-registration. Validation statistics never enter the constants. The §11
  probe's coefficients (fitted inside validation) are never used.
- **Delivery.** An additive data module wraps `improved_training_data.make_epoch_dataset` and
  yields `((s5, s6, reliability, lesion_stats), grade)`. It consumes no RNG, so epoch order and
  augmentation are bit-identical to the baseline runs.

### 3.3 Arms

| Arm | Description | Runs |
|---|---|---|
| **BASE** | Frozen `improved_multiseed_2026_09/NO_RACAF/seed_{42,123,2026}`, reused read-only (identical to §12's reuse) | 0 new |
| **LBP** | §3.1 with each image's own statistics | 3 |
| **LBP-PERM** (capacity-identical, information-destroyed control) | Byte-identical architecture, parameters, init, optimiser and data order. `lesion_stats` for image *i* is the statistics row of a donor π(i). π is a fixed **Sattolo derangement** over the training ids (seed 20260923), and a separate fixed derangement over validation ids is used at evaluation. The donor's statistics have the same marginal distribution and carry no information about image *i*'s grade | 3 |
| **LBP-GAP** (secondary, mechanistic, *not in the primary verdict*) | Replaces `lesion_stats` with `GAP(L)` (256-d, learned, count-preserving over the fixed 32×32 grid) through the same MLP shape (256→32→32→4; 9,408 new params). Run **only if** P0.3 shows the Stage-07 loss pattern | 0 or 3 |

Why LBP-PERM is the right control: it matches parameter count, layer shapes, zero-init, and the
optimisation perturbation of a second input path exactly. Only the *image-specific
information* differs. A gain of LBP over BASE that LBP-PERM also shows is a parameter/optimisation
effect. A gain of LBP over LBP-PERM is the information.

### 3.4 Initialisation and verification (must pass before any run)

1. With `keras.utils.set_random_seed(seed)`, every baseline-layer weight of the LBP build is
   **bit-identical** to `build_no_racaf_joint_model_matched_init()` at the same seed. This works by
   constructing the new layers last, the same fix as C1's `global_projection`.
2. Initial `z'` equals the baseline's initial `z` **bit-for-bit** on random probes under
   mixed_float16 (the zero-init kernel gives an exact 0; `z + 0` is exact).
3. Trainable parameters = 43,044,872; tensors = 405 + 5.
4. Reliability is still inert (r = 0 vs r = 1 give bit-identical logits).
5. The statistics table reproduces `per_image_variables.csv` (≤ 1e-6), and the donor map is a
   derangement (no fixed point).
6. After an in-process rebuild following `clear_session()`, the initial weights match a
   fresh-process build (the property §12's auto-run loop relies on).

### 3.5 Optimisation and what stays frozen

- **Identical to `improved-multiseed-v1`:** weighted CORN (same weights), AdamW lr 1e-4 wd 0.05
  with bias/gamma/beta excluded (the new kernels *are* decayed, so the zero-init residual is pulled
  toward the baseline: conservative), batch 2, ReduceLROnPlateau(val_QWK ×0.5, patience 4, min
  1e-6), EarlyStopping patience 12, max 50 epochs, mixed_float16, BEST = max `val_QWK`,
  per-(seed, epoch, image) augmentation and order, same cache, same split, same 730-image evaluation.
- **Frozen / untouched:** Stage 01–04, vessel and lesion caches, preprocessing, the split, Stage
  05/06/07 architecture, CORN architecture and decode, RACAF (closed), reliability (inert), and
  the §12 experiment.
- **No auxiliary loss** (λ = 0), no threshold change, no checkpoint-rule change.
- **Seeds:** 42, 123, 2026 (matched to BASE).

---

## 4. Endpoints

**Primary.** AUROC of `p_gt_3` (= `p_cum[:,3]` of `z'`) on **grade 4 vs grade 3** (n = 97; 39 vs
58), BEST checkpoint. This is the same endpoint as §12, so the two experiments are directly comparable.
(If P0.1 triggers, the conditional `σ(z'_3)` AUROC becomes co-primary under the same rule.)

**Secondary** (always reported, whichever direction they fall):

1. Conditional `σ(z'_3)` AUROC on 4 vs 3.
2. Grade 4 vs rest AUROC of `p_gt_3`.
3. QWK, MAE, accuracy, balanced accuracy, macro-F1.
4. Grade-3 recall, grade-4 recall and precision at the default decode; the full confusion matrix.
5. The primary endpoint with the 41 duplicates excluded (n drops accordingly), and at the LAST checkpoint.
6. Per-seed paired stratified bootstrap 95 % CI of each ΔAUROC (2,000 resamples, seed 20260923).
7. **Mechanism engagement:** SD(Δz_k)/SD(z_k) per task on validation; the sign of the learned
   burden → Δz_3 association (e.g. Spearman of HE/EX burden with Δz_3). Does the head learn the
   3→4 reversal?
8. Train−val QWK gap (overfitting).
9. LBP vs P0.4's stacked head (context: does end-to-end add anything beyond stacking?).

---

## 5. Pre-registered decision rule

Per seed *i*: `dB_i = AUROC(LBP) − AUROC(BASE)`, `dP_i = AUROC(LBP) − AUROC(LBP-PERM)`.

- **Improved:** mean dB ≥ **+0.03** and dB > 0 in **3/3** seeds.
- **Information-specific:** mean dP ≥ **+0.03** and dP > 0 in **3/3** seeds.
- **Preserved** (LBP vs BASE): mean ΔQWK ≥ −0.02, mean ΔMAE ≤ +0.03, mean grade-3 recall change
  ≥ −0.10 (§10 lesson: the adjacent grade is part of the rule, not a footnote), and mean grade 4
  vs rest AUROC change ≥ −0.02.

| Verdict | Condition |
|---|---|
| **SUPPORTIVE** | Improved ∧ Information-specific ∧ Preserved |
| **TRADE_OFF** | Improved ∧ Information-specific ∧ ¬Preserved |
| **INCONCLUSIVE** | Improved but not Information-specific (the gain cannot be separated from parameters/perturbation); or mean dB ≥ +0.03 with only 2/3 signs; or +0.01 ≤ mean dB < +0.03 |
| **NOT_SUPPORTIVE** | mean dB < +0.01, or dB ≤ 0 in ≥ 2/3 seeds |

Sanity reporting (not in the verdict): mean AUROC(LBP-PERM) − AUROC(BASE). If |value| ≥ 0.03, a
perturbation effect exists and is reported as such.

**Why 0.03.** It matches §12 (so the two experiments are judged on one scale), and it is below the
plausible ceiling. §11 puts the lesion statistics +0.072 above `p_gt_3` and the combination at
0.683, so a real effect of a few hundredths is the realistic expectation. The single-AUROC
Hanley–McNeil SE at 39 vs 58 is ≈ 0.058. Paired comparison on identical images reduces it, but
**three seeds can only confirm a consistent effect of roughly ≥ 0.03–0.05**. A true effect near
0.02 will most likely come out INCONCLUSIVE, and that outcome must be reported as such, not
re-analysed.

**Multiplicity.** One architecture, one primary endpoint, one control. LBP-GAP and every secondary
endpoint are descriptive and cannot rescue the verdict.

---

## 6. Stopping, failure handling and cost

- Stopping is identical to BASE (early stopping on `val_QWK`, patience 12, cap 50). No arm gets
  extra epochs.
- A crashed run resumes through the existing two-slot/fingerprint infrastructure. Reruns are
  never selected among. A run that cannot complete is reported as missing, and the verdict is
  then INCONCLUSIVE by rule.
- **Cost:** 6 new runs (LBP, LBP-PERM × 3 seeds). Existing runs took 27–42 epochs at ~8–13
  min/epoch, so roughly **4–9 GPU-hours per run, ~25–55 GPU-hours total** on a T4, plus ~12–27 for
  the optional LBP-GAP. Do not share a runtime with the §12 experiment.

---

## 7. What each outcome would mean

| Outcome | Reading | Next step |
|---|---|---|
| SUPPORTIVE | Amount, not presence, was the missing ingredient, and the attention read-out was the bottleneck | E (spatial distribution) as the next arm on top of LBP. H (PDR-lesion channels via the same pathway) becomes justified |
| TRADE_OFF | Burden helps 3v4 but distorts other boundaries | Restrict Δz to task 3 only (a pre-registered follow-up) |
| INCONCLUSIVE (not information-specific) | Any gain is parameters/perturbation | Close LBP. The §11 gap is then not recoverable by exposing these statistics |
| NOT_SUPPORTIVE | Exposing burden end-to-end does not move 3v4 despite the probe | The probe gain is fragile or not trainable at n = 390. Revisit only with better lesion information (H) |

Together with §13 (objective pressure), the design separates the four layers:
information (P0.2), representation (P0.3 + LBP vs LBP-PERM), objective (§12), and decode (P0.1).
Calibration is already characterised.

---

## 8. Exactly what to implement next (when authorised)

1. **Phase-0 script** (one standalone, read-only, copy-paste Colab script): P0.1–P0.4, gates G1,
   the P0.3 interpretation table, and outputs to `experiments/Grade3vs4_Phase0/<timestamp>/`. P0.1 and P0.2
   need no GPU; P0.3 needs a T4 for reasonable speed.
2. Only if G1 passes, **additive** modules (no existing file edited):
   - `lesion_burden_statistics.py`: the 10 statistics, the transform, the training-only
     constants, the derangement map, and verification against `per_image_variables.csv`.
   - `lbp_model.py`: the builder (baseline-first construction order), `verify_lbp_model()` with
     checks 3.4.1–3.4.6.
   - `lbp_training_data.py`: the 4-input epoch dataset wrapper.
   - tests mirroring `tests/test_no_racaf_model.py` (parity, zero-init equality, inertness,
     derangement, statistics reproduction).
   - `colab/notebooks/lesion_burden_pathway_experiment.ipynb` on the §12 notebook's pattern:
     frozen `PREREGISTRATION.json`, a 6-run manifest (BASE reused), auto-run, verdict, `REPORT.md`.
3. Record Phase 0, and later Phase 1, in the research record as new sections. Earlier sections stay unchanged.
