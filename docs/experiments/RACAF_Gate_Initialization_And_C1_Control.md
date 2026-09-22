# DR grading research record — RACAF closure, PDR/grade-4 investigation, CORN decision rule

> **This file is the authoritative running research record.** Every experiment result is recorded
> here as it arrives. Sections are appended in chronological order and earlier sections are never
> rewritten; where a later result changes an earlier conclusion, the change and its reason are
> stated explicitly in the newer section.

Additive record. `docs/experiments/Multiseed_Improved_Training_Report.md` is **not** modified by
this document; the pre-registered six-run verdict (INCONCLUSIVE) is unchanged and is not
reinterpreted here. This file exists so the post-hoc results below survive outside chat and Drive
for future analysis.

Nothing here was trained beyond the C1 and C3 seed-42 runs named below. No existing checkpoint,
manifest, cache or experiment directory was modified by any of these diagnostics. §4 (D6) and §5
were added after the reliability line was closed on evidence; §§1–3 are unchanged.

**Chronology:** §1 C1 control → §2 gate-initialisation check → §3 C3 screening → §4 D6 selective
prediction → §5 direction transition → §6 PDR/grade-4 diagnostic (grade 4 vs 0–3) → §7 grade 3 vs
grade 4 vessel diagnostic → §8 CORN threshold diagnostic → §9 nested threshold selection → §10
grade-3/grade-4 trade-off → §11 model score vs lesion representation.

---

## 1. C1 decomposition control (RACAF's gate vs its Ĝ pathway)

**Motivation.** The pre-registered RACAF vs NO-RACAF comparison changes two things at once: the
2-parameter reliability-conditioned gate, and a 295,168-parameter global-readout pathway
`Ĝ = Dense(256)(GAP(G))` that NO-RACAF never had. Of RACAF's 295,170-parameter delta over
NO-RACAF, 99.99932% is Ĝ and 0.00068% is the gate. C1 removes only the gate's `r`-dependence:

```
RACAF: gate = sigmoid(w_g*r + b_g)      F = gate*E + (1-gate)*Ĝ
C1:    gate = sigmoid(b_g)              F = gate*E + (1-gate)*Ĝ      (r removed from the gate)
```

Model: `racaf_c3_kappa_fusion_model.py`'s predecessor `racaf_c1_control_model.py` (committed
`bd90748`), 43,338,505 trainable parameters — exactly RACAF minus the gate's `w_g` kernel.

**Run.** C1 / seed 42 only, identical protocol to the six-run experiment (protocol read directly
from that experiment's frozen `PREREGISTRATION.json`). Drive:
`experiments/ImprovedTrainingC1Control/racaf_c1_decomposition_control_2026_09/C1/seed_42`.
Stopped by early stopping at epoch 42.

| | BEST (epoch 29) | LAST (epoch 42) |
|---|---|---|
| QWK | 0.8322 | 0.8132 |
| Accuracy | 0.7479 | 0.7603 |
| Balanced accuracy | 0.5642 | 0.6238 |
| Macro F1 | 0.5513 | 0.6058 |
| Weighted F1 | 0.7498 | 0.7641 |
| MAE | 0.3342 | 0.3384 |
| Grade-3 recall | 0.4359 | 0.5641 |
| Grade-4 recall | 0.2241 | 0.2931 |
| Learned constant gate | 0.507457 | 0.508479 |

**Seed-42 three-way comparison (BEST QWK).**

| arm | BEST QWK | epoch |
|---|---|---|
| RACAF | 0.8298 | 23 |
| NO_RACAF | 0.8296735688 | 23 |
| C1 | 0.8322 | 29 |

Decomposition terms: `RACAF − C1 = −0.0024`, `C1 − NO_RACAF = +0.0025`.

**Interpretation.** Both terms are far inside the noise this design can resolve (across-seed SD
0.0125–0.0170; paired-Δ SD 0.0063). One seed cannot separate the gate's contribution from Ĝ's.
No winner declared; the six-run verdict is untouched.

**Caveat on C1 as a control.** C1's learned gate settled at ≈0.51 while RACAF/seed_42's effective
gate is ≈0.80, so `RACAF − C1` mixes r-dependence with a different blend level. BEST QWK was
nearly identical at E-weights 1.0 (NO_RACAF), ≈0.8 (RACAF) and ≈0.5 (C1), which suggests the blend
level is not what is driving the metric here.

---

## 2. Gate-initialisation check — was `w_g` ever learned?

Read-only. Trained nothing. Each seed's RACAF model was rebuilt through the exact construction
path training used (`multiseed_runs.build_arm_model("RACAF", seed)`, which calls
`keras.utils.set_random_seed(seed)` then `joint_training_model.build_joint_model()`); no
initializer value was reproduced by hand. BEST weights were read into the in-memory model and each
weights file's SHA256 re-verified afterwards (**unchanged: True** for all three seeds).

Drive artifact:
`experiments/ImprovedTraining/improved_multiseed_2026_09_posthoc_gate_initialization/`
(`results.json`, `SUMMARY.md`).

| seed | initial w_g | BEST w_g | Δw_g | initial b_g | BEST b_g | Δb_g | BEST/initial |
|---|---|---|---|---|---|---|---|
| 42 | +1.697090 | +1.560681 | −0.136409 | +0.000000 | +0.016006 | +0.016006 | 0.9196 |
| 123 | −1.699036 | −1.556498 | +0.142538 | +0.000000 | +0.019720 | +0.019720 | 0.9161 |
| 2026 | +0.562212 | +0.525406 | −0.036807 | +0.000000 | +0.022513 | +0.022513 | 0.9345 |

**Decomposition into weight decay vs gradient**, against each run's own decay-only trajectory
computed from its recorded per-epoch learning rates:

| seed | weight decay | gradient | gradient Δw_g | Δb_g | gradient Δw_g / Δb_g |
|---|---|---|---|---|---|
| 42 | −8.89% | +0.86% | +0.01452 | +0.01601 | 0.907 |
| 123 | −7.04% | +1.34% | +0.02285 | +0.01972 | 1.159 |
| 2026 | −9.72% | +3.17% | +0.01785 | +0.02251 | 0.793 |

**Weight decay, per the actual optimizer implementation.** `keras.optimizers.AdamW.
_use_weight_decay()` regex-matches `variable.name` against `("bias", "gamma", "beta")`;
`_apply_weight_decay()` then applies `variable -= variable * 0.05 * lr` once per step.

- `w_g` → `variable.name == "kernel"` → **weight decay applies**
- `b_g` → `variable.name == "bias"` → **excluded**

**Verdict: `w_g` is mostly unchanged from initialisation.**

1. Weight decay moved `w_g` 3–10× more than the gradient did, in every seed.
2. The BEST/initial ratios (0.9196 / 0.9161 / 0.9345) cluster tightly despite initial values
   differing in magnitude (1.70 vs 0.56) and sign — the signature of a multiplicative shrink, not
   of learning.
3. `b_g` is a clean, assumption-free channel: zeros-init (deterministic, no RNG, device-independent)
   and excluded from decay, so its entire movement is gradient. It moved only **+0.016 to +0.023**
   over ~24 epochs. That is the full size of the gradient signal reaching the gate.
4. The two agree mechanistically. For a sigmoid gate, gradient-attributable `Δw_g / Δb_g` should
   equal `mean(r) = 0.863`; observed 0.907 / 1.159 / 0.793. Two independently-derived quantities
   landing on the predicted constant across three seeds rules out coincidence.
5. The gradient pushed `w_g` **positive in all three seeds**, including seed 123 where `w_g` is
   negative — a uniform weak pressure from the shared `∂L/∂gate` term, not a seed-specific
   data-driven target.

**Consequence.** Each seed's realised gate at `mean(r) = 0.8631` — 0.796 / 0.210 / 0.617 — is
essentially its Glorot draw, shrunk ~8% by decay and nudged ~0.02 by learning. For a `(1,1)`
kernel the glorot-uniform limit is `sqrt(6/2) = 1.7321`; seeds 42 and 123 drew within 2% of that
extreme. The BEST gate's range across the full mathematical range of `r ∈ [0,1]` is 0.325 / 0.326 /
0.128, but across the empirical `r` (mean 0.8631, SD 0.04251) the realised modulation is only
≈0.043 / 0.044 / 0.021.

This means §12.1/§14's "co-adaptation" wording in the main report needs amending when that report
is next revised: co-adaptation still explains the gate-forcing collapse (CORN adapts to whatever
fixed blend it is handed), but the gate *level* is not a learned, data-driven quantity. **The main
report has not been modified.**

**Remaining caveat.** The initialisation reconstruction ran CPU/float32 while the original training
ran GPU/mixed_float16. Three seeds landing within 1–3% of independently-computed decay-only
trajectories makes a GPU/CPU RNG divergence implausible, but it is an assumption rather than a
measurement. Closing it would cost one T4 run of the build step alone. Deliberately not run.

---

## 3. C3 seed-42 screening (per-lesion-class κ, residual channel-wise fusion)

**Motivation.** Sections 1–2 established that RACAF's reliability hypothesis was never actually
tested: the mechanism carrying it was a 2-parameter scalar gate whose weight was decayed faster
than it was learned. C3 is a *corrected implementation* of the same hypothesis — per-lesion-class
`κ` (not the burden-collapsed scalar `r`) conditioning a 256-d channel-wise **residual**, not a
convex gate:

```
C3: gamma = Dense(4 -> 256)(kappa), linear, zero-init      F = E + gamma * Ghat
```

Model: `racaf_c3_kappa_fusion_model.py` (committed `fd1e8c5`), 43,339,784 trainable parameters —
RACAF + 1,278 (the gate's 2 params removed, γ's 1,280 added). `gamma` is zero-initialised in both
kernel and bias, so `F == E` exactly at step 0 — C3 starts at the NO-RACAF representation, verified
bit-exact under both float32 and the training's actual `mixed_float16` policy.

**Run.** C3 / seed 42 only, non-inferential screening, identical protocol to the six-run experiment
(read from its frozen `PREREGISTRATION.json`). Drive:
`experiments/ImprovedTrainingC3Kappa/racaf_c3_kappa_residual_2026_09/C3/seed_42`. Stopped by early
stopping at epoch 42; BEST at epoch 29.

| | BEST (epoch 29) | LAST (epoch 42) |
|---|---|---|
| QWK | 0.8315 | 0.7937 |
| Accuracy | 0.7397 | 0.7123 |
| Balanced accuracy | 0.5767 | 0.5631 |
| Macro F1 | 0.5609 | 0.5392 |
| MAE | 0.3507 | 0.4000 |
| Grade-3 recall (n=39) | 0.4615 | 0.4615 |
| Grade-4 recall (n=58) | 0.3276 | 0.2759 |
| Mechanism ratio, median (‖γ⊙Ĝ‖/‖E‖) | 0.0734 | 0.0769 |
| Mechanism ratio, mean (p25–p75) | 0.0724 (0.0611–0.0832) | 0.0751 (0.0624–0.0877) |
| mean｜γ｜ over full val (std) | 0.003631 (0.008591) | 0.003827 (0.008896) |
| Mechanism engaged (≥0.01)? | **True** | **True** |

`κ` over the full validation set, order (MA, HE, EX, SE): per-class SD = [0.0, 0.0810, 0.1330,
0.1453]; fraction exactly 1.0 = [1.000, 0.0027, 0.0575, 0.3192] — confirms the audit-time finding
that MA is a constant input and SE is partially degenerate (32% of validation images have no
detected soft exudate).

**Pre-registered screening verdict.**

| baseline | BEST QWK |
|---|---|
| NO_RACAF seed 42 (frozen) | 0.8296735688 |
| C3 seed 42 | 0.8315116763 |
| **Δ** | **+0.0018** |

Thresholds: advance ≥0.8360, abandon <0.8127, gray zone in between.

**VERDICT: GRAY ZONE — no automatic advance, per the pre-registered protocol. Not decided here.**

**Four-arm comparison at seed 42, BEST QWK.**

| arm | BEST QWK | mechanism |
|---|---|---|
| RACAF | 0.8298 | scalar gate, proven near-frozen (§2) |
| NO_RACAF | 0.8296735688 | none |
| C1 | 0.8322 | constant scalar gate (0.51), by construction untrainable by `r` |
| C3 | 0.8315 | 256-d residual, **demonstrably engaged** (mechanism ratio 7.3%, ~640× C1's parameter count) |

All four span **0.0025** — smaller than one paired-Δ SD (0.0063) from the six-run experiment.

**What is different about this result, and why it matters more than C1's.** C1's null was
ambiguous: its gate stayed near its 0.5 initialisation, so the null was consistent with either
"reliability doesn't help" or "the mechanism never got the chance to try." C3 closes that gap —
`mechanism_engaged = True` with a median ratio (7.3%) more than 7× the pre-registered 1% floor, and
γ's magnitude was already near this level by epoch 29 and barely grew by epoch 42 (mean｜γ｜ LAST/BEST
= 1.054×, ratio LAST/BEST = 1.048×). This is a mechanism that **did** try, with ~640× more capacity
and 256× the modulation bandwidth of C1's scalar gate, and it produced a QWK indistinguishable from
every arm that tried nothing (NO_RACAF) or that couldn't try (C1). That is a stronger piece of
evidence against the reliability hypothesis at this seed than either RACAF or C1 provided alone —
though still only one seed, and the gray-zone bar was deliberately calibrated so a result like C1's
would not auto-pass.

**Secondary observations, explicitly not part of the pre-registered decision.** C3's BEST grade-4
recall (0.3276, 19/58) is nominally the highest of the four arms (RACAF 0.2414, NO_RACAF 0.1034, C1
0.2241), and its grade-3 recall (0.4615, 18/39) matches C1's (0.4359). Both are secondary metrics on
small subsets (n=39, n=58) where a two- or three-sample swing moves the number several points; not
weighted into the screening verdict.

**BEST→LAST degradation is not unique to C3.** C3 drops 0.0378 from BEST to LAST; NO_RACAF drops an
almost identical 0.0378 over the same epoch range (36–42), while RACAF and C1 drop only ≈0.018–0.019.
This 2×-larger drop for C3/NO_RACAF than for RACAF/C1 does not track with mechanism presence — the
arm with no mechanism at all (NO_RACAF) shows the same magnitude as C3 — so it reads as shared
late-training instability rather than something C3's fusion specifically causes.

**No further runs executed.** Seeds 123/2026 and the shuffled-κ control remain unimplemented and
unrun pending an explicit decision, per the pre-registered protocol.

---

## 4. D6 — selective prediction: does reliability predict *grading error*?

**Research question.** *Can segmentation-derived uncertainty/reliability provide useful information
about the reliability of the DR grading prediction itself?*

Sections 1–3 tested reliability as a **fusion** input and found nothing. D6 asks the separate
question of whether the same signal is informative about *whether a prediction is wrong* — a
signal can in principle rank errors usefully without being useful as a feature, because the network
may already contain the information while a human triaging cases would still benefit from the rank.

**Method.** Entirely read-only. No training, no network inference, no cache regeneration. The
per-sample evaluation CSVs each experiment already wrote (`evaluation/per_sample_best.csv`, the
27-column `multiseed_runs.build_per_sample_rows` schema) were joined by `image_id` to the `kappa`/`r`
values already cached by Stage 04's frozen TTA pass. Outcome: `error = (predicted_grade !=
true_grade)`. Output directory:
`experiments/SelectivePrediction_D6/2026-09-22_13-00-03`.

**Pre-registered direction, fixed before any result was seen.** High `r` = higher reliability =
expected **lower** grading error. Every signal was expressed as a confidence (higher = expected
correct) and error-detection AUROC computed from `-confidence`, so AUROC > 0.5 means the signal
behaves as the project's semantics imply. The direction was not revised afterwards.

**Results.** Six matched runs, plus C1/C3 as supporting seed-42 analyses (not additional seeds):

| run | n | error rate | AUROC(r), pre-registered direction |
|---|---|---|---|
| RACAF / seed 42 | 730 | 0.2534 | 0.3956 |
| NO_RACAF / seed 42 | 730 | 0.2301 | 0.4281 |
| RACAF / seed 123 | 730 | 0.2616 | 0.3869 |
| NO_RACAF / seed 123 | 730 | 0.2507 | 0.4135 |
| RACAF / seed 2026 | 730 | 0.3082 | 0.3467 |
| NO_RACAF / seed 2026 | 730 | 0.3123 | 0.3736 |
| *C1 / seed 42 (supporting)* | 730 | 0.2521 | 0.3966 |
| *C3 / seed 42 (supporting)* | 730 | 0.2603 | 0.3914 |

- **Mean AUROC(r) over the six matched runs = 0.3907; 0/6 runs above 0.5.**
- Best incremental extension `plus_r_and_kappa_vector`: **mean ΔAUROC = +0.0009, positive in 2/6 runs.**
- `predicts_error = False`, `adds_information = False`.

**Under the pre-registered direction the hypothesis is rejected.**

**Post-hoc observation, recorded but not acted on.** All eight runs fall below 0.5 in a narrow band
(0.347–0.428), which is a consistent *inverse* association rather than an absence of association.
A plausible mechanism is the empty-foreground branch documented in §3: `κ = 1` when Stage 04
predicts no foreground, so high `r` partly encodes *segmentation silence* rather than segmentation
reliability. **This is a post-hoc observation only. It is not converted into a successful
reliability result, and it does not justify reopening the reliability pathway.** Acting on it would
require its own fresh pre-registration.

**Why the incremental analysis — not the direction — is decisive.** The incremental test fits a
logistic model with out-of-fold cross-validation, so it is free to learn **whichever sign of `r`
helps**; it was never constrained to the pre-registered direction. Given that freedom it gained
**+0.0009 mean AUROC**, positive in only 2 of 6 runs. Whichever way the signal is read, it carries
essentially nothing beyond the classifier's own confidence measures
(`predicted_class_probability`, `nearest_threshold_margin`, `class_entropy_nats`), all of which were
already present in the frozen artifacts.

**Conclusion.** *The current segmentation-derived reliability representation does not provide useful
incremental information for DR grading prediction in this pipeline.* This statement is scoped to
this representation and this pipeline. It is **not** a claim that segmentation uncertainty is
useless in general, which the evidence here does not support.

**Optional supporting tables.** The D6 output directory also contains `shuffled_control.csv` (real
vs one fixed Sattolo-derangement pairing) and `per_run_summary.csv` (per-channel κ AUROCs in the
authoritative order **MA, HE, EX, SE**, and the classifier-confidence baselines). Their values have
not been transcribed into this document. They are supporting closure evidence only — the D6
conclusion rests on the mean ΔAUROC of +0.0009 and does not depend on them.

---

## 5. Research-direction transition: reliability closed, PDR/grade-4 opened

**What the accumulated evidence establishes.**

1. RACAF did not produce a consistent grading improvement (six-run verdict: INCONCLUSIVE, unchanged
   and not reinterpreted).
2. C1 and C3 did not establish a meaningful improvement — both landed in the gray zone.
3. C3 **did** engage mechanistically (median residual ratio 0.073, ~7× the pre-registered floor) yet
   produced only a +0.0018 QWK difference, so *lack of mechanism activation is not a sufficient
   explanation* for RACAF's failure.
4. D6 tested the remaining possibility — that reliability predicts grading *error* rather than
   improving grading.
5. D6 failed under the pre-registered direction (0/6 runs above 0.5).
6. The consistently sub-0.5 values indicate an inverse association, but this is post-hoc and does
   **not** justify reopening the reliability pathway.
7. The incremental analysis is decisive: classifier confidence + `r`/κ produced mean ΔAUROC
   = **+0.0009**.
8. Therefore: **close RACAF; close the reliability-fusion pathway; do not build another reliability
   architecture; do not run C3 seeds 123/2026; do not run shuffled-κ as a new fusion experiment.**

**Why the next direction is PDR / grade-4.** Grade-4 performance is the project's largest remaining
measured weakness — grade-4 recall across the seed-42 arms spans roughly 0.10–0.33 (§3), meaning the
majority of sight-threatening cases are missed, an effect two orders of magnitude larger than
anything the reliability line could have delivered. The four lesion channels are **MA, HE, EX, SE**;
the clinical literature records that microaneurysm, haemorrhage and hard-exudate burden peak at
severe NPDR and *decline* at PDR, while the grade-4-defining features (IRMA, neovascularisation,
venous beading) have no channel in this pipeline. The existing vessel-permutation control (§12.4 of
the main report) shows the vessel channel contributes real but modest information (ΔQWK 0.042–0.075,
every CI excluding zero).

**A caveat that must not be lost.** That same §12.4 analysis reports the vessel contribution is
concentrated at **grade 0** — vessel permutation mainly costs grade-0 recall (0.931→0.809,
0.906→0.842, 0.828→0.734) "while leaving grades 3–4 roughly unchanged or slightly improved." On its
face this is evidence *against* the generic vessel representation carrying grade-4-specific
information. It is the single most important reason the next step must be **diagnostic rather than
architectural**.

**Critical limitation, preserved explicitly.** Stage 03 is a **generic retinal vessel segmenter**.
Generic vessel segmentation is **not** equivalent to neovascularisation, and neovascularisation is
what defines PDR. Nothing in this project has established that the cached vessel map encodes NV.

**Therefore the next experiment is a read-only diagnostic**, not a new model: does the existing
vessel representation contain image-specific information particularly relevant to distinguishing
grade 4 from non-grade-4, and is it sufficient to justify designing a dedicated PDR/grade-4 vascular
pathway? No claim is made here that a PDR-specific model will work, and none of this is presented as
a final solution.

---

## 6. PDR / grade-4 diagnostic (read-only) — Grade 4 vs Grades 0–3

**Method.** Read-only. No training, no network inference, no cache regeneration. Per-image summary
statistics were computed from the **already-cached** vessel maps (`APTOS_{id}_vessel_512x512.npy`,
`(512,512,1)`) and lesion maps (`(512,512,4)`, order **MA, HE, EX, SE**), joined to the frozen
per-sample grading artifacts. Output:
`experiments/PDR_Grade4_Diagnostic/2026-09-22_13-30-45`. Validation population verified **identical
across all six runs**: n = 730, **58 true grade-4 cases**.

**Grade-4 failure, characterised.**

| run | correct grade-4 | recall | predicted as grade 4 | precision |
|---|---|---|---|---|
| RACAF / 42 | 14/58 | 0.2414 | 29 | 0.4828 |
| NO_RACAF / 42 | 6/58 | 0.1034 | 11 | 0.5455 |
| RACAF / 123 | 5/58 | 0.0862 | 9 | 0.5556 |
| NO_RACAF / 123 | 3/58 | 0.0517 | 8 | 0.3750 |
| RACAF / 2026 | 19/58 | 0.3276 | 34 | 0.5588 |
| NO_RACAF / 2026 | 19/58 | 0.3276 | 38 | 0.5000 |

Mean recall **0.190** — roughly **81% of PDR cases are missed**. Recall spans 3/58 to 19/58, a
**6.3× spread across runs**, so grade-4 behaviour is not merely poor but highly unstable. Every run
**under-predicts** grade 4 (8–38 predictions against 58 true cases), consistent with the
grade-compression pattern recorded elsewhere.

**Discrimination results (Grade 4 vs Grades 0–3).**

| feature set | out-of-fold AUROC |
|---|---|
| vessel only | 0.6927 |
| lesion only | 0.7531 |
| lesion + vessel | 0.7537 |
| **vessel increment over lesion** | **+0.0006** |

Strongest single variable overall: `lesion_EX_foreground_fraction`, AUROC **0.7648** — a **lesion**
variable, higher than every vessel variable and higher than the entire vessel-only model. Strongest
vessel variable: `vessel_max`, AUROC **0.3213** (an **inverse** association).

**Pre-registered verdict: INCONCLUSIVE.** Vessel-only cleared the isolation bar (0.6927 ≥ 0.65) but
failed the incremental bar by roughly a factor of 33 (+0.0006 against a required +0.02).

**Reading.** The vessel representation carries real information about grade 4 *in isolation* and
almost none *beyond what the lesion representation already provides*. This is structurally the same
finding as D6 (§4): a signal that appears informative alone and contributes nothing incrementally.

**Post-hoc observation, not a finding.** `vessel_max` is inverted — a *higher* maximum vessel
probability makes grade 4 *less* likely. A max-probability statistic that falls on the most severe
cases is more consistent with the retina being obscured (vitreous/preretinal haemorrhage is itself a
PDR feature) or with image quality degrading, than with neovascularisation being detected. **This is
post-hoc and is explicitly not evidence that the vessel map encodes NV.** It also raises the
possibility that part of the isolated 0.69 is an image-quality proxy, which cannot be controlled
here because Stage 01 IQA is not part of the downstream graph. This is consistent with §12.4 of the
main report, where vessel permutation costs grade-0 recall rather than grade-3/4 recall.

**Limitations.** Only 58 grade-4 cases, so every grade-4 statistic is unstable. Generic vessel
segmentation is **not** neovascularisation and no NV annotation exists in this project. The
variables are global image-level summaries with no optic-disc reference, so they cannot localise
NVD/NVE. Single frozen APTOS split.

**Open methodological gap.** This contrast was Grade 4 vs Grades 0–3, which is dominated by grade
0/1 and therefore partly measures *diseased vs healthy* — something the lesion channels already do
well. The clinically relevant PDR discrimination is **Grade 4 vs Grade 3** (PDR vs severe NPDR),
precisely where the clinical literature records lesion burden declining. That contrast is
**unresolved** and is the subject of the follow-up diagnostic; every variable needed is already
saved in `per_image_variables.csv`, so it requires no new computation.

**Status.** A dedicated PDR/grade-4 pathway built on the **existing generic vessel representation**
is not justified by this evidence. That closes this particular route; it does not close the PDR
problem, which remains the project's largest measured weakness.

---

## 7. Grade 3 vs Grade 4 — the vessel representation on the clinically relevant contrast

Artifact: `experiments/PDR_Grade4_Diagnostic/grade3_vs_grade4_2026-09-22_13-56-17`. Read-only, from
the saved `per_image_variables.csv`. Population verified: **39 grade-3, 58 grade-4, n = 97**, no
other grade present. Positive = grade 4.

| feature set | OOF AUROC (Grade 4 vs Grades 0–3, §6) | **OOF AUROC (Grade 4 vs Grade 3)** | change |
|---|---|---|---|
| vessel only | 0.6927 | **0.5724** | −0.1203 |
| lesion only | 0.7531 | **0.6700** | −0.0831 |
| lesion + vessel | 0.7537 | **0.6416** | −0.1121 |
| **increment (vessel over lesion)** | +0.0006 | **−0.0284** | — |

**Category: NOT_SUPPORTIVE.** Vessel-only fell below the 0.60 bar and the increment was negative
(positive in only 1/5 repeats).

**The confound §6 flagged was real and large.** Removing grades 0–2 cost the vessel representation
0.1203 AUROC against lesion's 0.0831 — most of §6's apparent vessel signal was *diseased vs
healthy*, which the lesion channels already handle. Approximate Hanley–McNeil intervals (computed
from the reported AUROC and class counts): vessel-only ≈ [0.46, 0.69], which **includes 0.5**;
lesion-only ≈ [0.56, 0.78], which excludes it.

**The negative increment must not be over-read.** Going from 10 to 17 features on 97 samples
carries a real dimensionality cost that alone can produce a negative increment from uninformative
features. "Adds nothing" and "adds nothing while costing capacity" both fit; the stronger claim
that vessels are *anti*-informative is not supported.

**Why there was little to add.** Every vessel variable correlates with lesion variables at |ρ| ≈
0.30–0.43, and five of seven peak against `lesion_SE_mean_prob` (+0.412, +0.404, +0.388, +0.390,
+0.429), with `vessel_mean_prob_in_foreground` at −0.432 against `lesion_EX_foreground_fraction`.
The vessel segmenter's output partly tracks general retinal abnormality rather than
vasculature-specific pathology — consistent with §6's `vessel_max` inversion and with §12.4 of the
main report.

**Closes** the generic-vessel PDR route. Does **not** close the PDR problem, and does not show
PDR-specific vascular information is absent from fundus images — only that this frozen
representation does not expose it.

---

## 8. CORN decision-rule / threshold diagnostic (in-sample)

Artifact: `experiments/CORN_ThresholdDiagnostic/2026-09-22_14-07-33`. Read-only sweep over the
frozen `p_gt_0..3`; no retraining, no CORN modification, no threshold adopted.

**Threshold-free evidence — is the grade-4 information present at all?**

| run | AUROC(`p_gt_3`) grade 4 vs rest | AUROC grade 4 vs grade 3 | baseline recall |
|---|---|---|---|
| RACAF / 42 | 0.8878 | 0.6194 | 0.2414 |
| NO_RACAF / 42 | 0.8800 | 0.5199 | 0.1034 |
| RACAF / 123 | 0.8634 | 0.5889 | 0.0862 |
| NO_RACAF / 123 | 0.8806 | 0.5836 | 0.0517 |
| RACAF / 2026 | 0.8836 | 0.6472 | 0.3276 |
| NO_RACAF / 2026 | 0.8939 | 0.6286 | 0.3276 |
| **mean (six)** | **0.8815** | **0.5979** | 0.190 |

C1 (0.8699) and C3 (0.8921) sit in the same band — architecture did not change grade-4 ranking.

**Category: SUPPORTIVE.** Mean grade-4 recall gain **+0.4253** at QWK cost ≤0.02, in 6/6 runs.

**The decisive argument is a variance decomposition.** Ranking quality spans only **0.031** across
runs while realised recall spans **0.276** (0.0517–0.3276, a 6.3× ratio) — roughly nine times more
variance in the outcome than in the ranking that produces it. The models all identify grade-4 cases
similarly well; the fixed 0.5 rule is what varies and what discards the information. This also
retroactively explains the seed-to-seed grade-4 instability recorded in §6: it was never a
representation difference.

**Bounded by the second column.** At 0.5979, `p_gt_3` is near chance on grade 4 vs grade 3, so the
recoverable recall comes from separating PDR from grades 0–2, not from severe NPDR.

---

## 9. Nested CORN threshold selection — does the gain survive out-of-sample?

Artifact: `experiments/CORN_NestedThreshold/2026-09-22_14-14-00`. 5× repeated 5-fold stratified CV,
seed 20260922; `tau4` selected on the training portion only and applied to the held-out fold. The
in-sample figure was recomputed in the same run so the optimism is measured, not transcribed.

| run | baseline recall | in-sample | **OOF** | OOF QWK Δ | selected `tau4` range |
|---|---|---|---|---|---|
| RACAF / 42 | 0.2414 | 0.6552 | **0.6621** | −0.0165 | 0.18–0.24 |
| NO_RACAF / 42 | 0.1034 | 0.6552 | **0.6517** | −0.0149 | 0.14–0.18 |
| RACAF / 123 | 0.0862 | 0.5172 | **0.5000** | −0.0144 | 0.18–0.22 |
| NO_RACAF / 123 | 0.0517 | 0.6379 | **0.6103** | −0.0167 | 0.14–0.18 |
| RACAF / 2026 | 0.3276 | 0.5172 | **0.5000** | −0.0069 | 0.36–0.38 |
| NO_RACAF / 2026 | 0.3276 | 0.7069 | **0.6655** | −0.0196 | 0.36–0.38 |

**Category: SUPPORTIVE, 6/6.** Mean in-sample gain +0.4253 vs **out-of-sample +0.4086 — optimism
only 0.0167, about 4% of the effect.** Mean OOF QWK cost −0.0148, within the 0.02 tolerance in 6/6.

- The selected threshold is **far below CORN's 0.5** and is **run-specific** (≈0.14–0.24 for seeds
  42/123, ≈0.36–0.38 for seed 2026). There is no universal threshold; each model needs its own.
- Within a run the selection is **stable** (2–3 grid steps of 0.02), so it is not being fitted to
  fold noise — the main thing this experiment had to rule out.
- **The worst baselines gain most**: NO_RACAF/seed_123 goes 0.0517 → 0.6103, a 12× improvement,
  exactly as the decision-rule explanation predicts.
- Cross-run recall spread narrows from 0.276 (baseline) to 0.166 (OOF) against an AUROC spread of
  0.031 — thresholding moves realised performance toward the underlying ranking similarity.

---

## 10. What the recall recovery actually costs

Artifact: `experiments/CORN_NestedThreshold/tradeoff_analysis_2026-09-22_14-22-14`. Grade-3 metrics
derived from the saved OOF confusion matrix (repeat 0, the only one persisted; repeat-0 vs
across-repeat grade-4 recall differ negligibly). Baseline confusion matrices rebuilt from the
frozen `per_sample_best.csv` `predicted_grade` column.

| run | G4 recall | G4 precision | G3 recall | QWK Δ |
|---|---|---|---|---|
| RACAF / 42 | 0.2414 → 0.6552 | 0.4828 → 0.3551 | 0.3333 → 0.1026 | −0.0165 |
| NO_RACAF / 42 | 0.1034 → 0.6552 | 0.5455 → 0.3304 | 0.3846 → 0.1026 | −0.0149 |
| RACAF / 123 | 0.0862 → 0.5000 | 0.5556 → 0.3452 | 0.3590 → 0.1282 | −0.0144 |
| NO_RACAF / 123 | 0.0517 → 0.6207 | 0.3750 → 0.3303 | 0.4103 → 0.1282 | −0.0167 |
| RACAF / 2026 | 0.3276 → 0.5000 | 0.5588 → 0.4143 | 0.5128 → 0.3333 | −0.0069 |
| NO_RACAF / 2026 | 0.3276 → 0.6724 | 0.5000 → 0.3421 | 0.5128 → 0.1795 | −0.0196 |

- Grade-4 recall 0.190 → **0.601**; precision 0.5029 → **0.3529** (−30% relative).
- **Grade-3 recall 0.4188 → 0.1624 — a 61% relative collapse** (mean change −0.2564).
- **Only 30.4% of the additional grade-4 calls are correct** (~77 new PDR calls per run, ~24 right).
- QWK within tolerance in 6/6, mean −0.0148.

**Why QWK gave false comfort — methodological lesson.** QWK is quadratically weighted, so a 3↔4
confusion is a distance-1 error carrying **1/16** the weight of a distance-4 error. QWK is
therefore nearly blind to exactly the trade this threshold makes. "QWK within tolerance" is a weak
guardrail for this manipulation and should not be reused as one.

**Gap in the pre-registration, recorded honestly.** The "record as promising" criteria were recall
gain, QWK tolerance and no severe precision collapse — **none looked at the adjacent grade**, which
turned out to be the dominant cost. The script returned `record_as_promising = True` against those
bars; that verdict is arithmetically correct and substantively incomplete. Future decision-rule
criteria must include per-grade degradation on the adjacent grade.

**Revised status.** The threshold is a dial on a trade-off curve, not a fix; the curve's quality is
set by `p_gt_3`'s 0.5979 AUROC on grade 3 vs grade 4. Visible across runs: RACAF/seed_2026 with the
most conservative `tau4` (≈0.37) gained least and damaged least. Recorded as a **real but bounded
calibration finding, not a representation solution**. No clinical recommendation; no threshold
adopted.

---

## 11. Why is the model's PDR score weak? — model score vs lesion representation

Artifact: `experiments/Grade3vs4_ScoreComparison/2026-09-22_14-29-33`. Read-only. n = 97 (39
grade-3, 58 grade-4), identical folds across every feature set. QWK deliberately **not** used —
see §10 for why it cannot see this contrast.

| feature set | mean OOF AUROC |
|---|---|
| `nearest_threshold_margin` only | 0.4910 (chance) |
| `p_gt_3` only, OOF-fitted | 0.5546 |
| `p_gt_3` direct, unfitted | **0.5979** |
| lesion only (10 variables) | **0.6700** |
| `p_gt_3` + lesion | 0.6826 |

Per-run `p_gt_3` direct AUROC: 0.6194 / 0.5199 / 0.5889 / 0.5836 / 0.6472 / 0.6286.
Per-run combined: 0.6811 / 0.6520 / 0.6748 / 0.6713 / 0.6948 / 0.7217.

**Category: SUPPORTIVE, 6/6.** The asymmetry is the finding:

- lesion adds **+0.1280** to `p_gt_3` (6/6 runs, all ≥ 0.05)
- `p_gt_3` adds **+0.0126** to lesion

A ~10× asymmetry. **And the comparison is stacked against lesion**: `p_gt_3` is evaluated unfitted
with no CV penalty, while the lesion set pays a 10-features-on-97-samples penalty — and lesion
still wins by ~0.072.

Two consistency checks passed: `p_gt_3` direct mean 0.5979 reproduces §8's figure exactly from an
independently written script, and lesion-only is identical (0.6700) in all six runs as it must be
with shared features and fixed folds. `nearest_threshold_margin` at 0.4910 rules out the
alternative that the model encodes the distinction in confidence rather than in `p_gt_3`.

**Conclusion — this relocates the problem.** The grade-3-vs-4 information **is present in the
input**: the cached lesion maps enter Stage 05 as channels 4–7, and the features that beat the
model are literally **global means and foreground fractions of those same maps**. A 43M-parameter
network receives identical data and produces a worse score. This is not a data problem and not a
representation-availability problem — the information is being discarded between input and
`p_gt_3`.

**Hypothesis the evidence is consistent with — NOT established.** QWK and the CORN objective weight
a 3↔4 confusion at 1/16 of a distance-4 error (§10), so training applied almost no gradient
pressure to that boundary; the model never learned to separate the grades, and thresholding cannot
repair it because it only slides along an already-poor curve. This links §8, §10 and §11 but the
analysis is associational and does not establish causation. Testing it would require an objective
or auxiliary-head change — a **training** change, not implemented and not decided.

**Limitations.** 97 samples, 11 features in the combined set; bootstrap intervals in
`incremental_auroc.csv` are the honest read of the increment. One frozen APTOS split; the six runs
share it and the arms are matched pairs, so they are not six independent observations.

---

## 12. Grade-4 auxiliary loss — PRE-REGISTERED, launched (no results yet)

Recorded **before any auxiliary run was trained**, so the values below are verifiably not
post-hoc. Notebook: `colab/notebooks/grade4_aux_loss_experiment.ipynb`. Experiment root:
`experiments/Grade4AuxLoss/grade4_aux_loss_2026_09`, which also holds a frozen
`PREREGISTRATION.json`; the notebook refuses to run if that file and the notebook's constants ever
disagree.

**Hypothesis.** The CORN objective does not provide sufficient training pressure on the
Grade-3/Grade-4 boundary, so useful lesion information already present in the inputs is discarded
from the learned `p_gt_3` score. This is the hypothesis §11 flagged as *consistent with* the
evidence but explicitly **not established** — §12 is its first direct test.

**Design.** The only training change is an added term; CORN is neither replaced nor removed:

```
baseline:  weighted_corn_loss
auxiliary: weighted_corn_loss + lambda * BCE(p_gt_3, 1[grade == 4])
```

`p_gt_3` is the **marginal** `P(y=4) = prod_j sigmoid(logit_j)` — verified equal to
`corn.decode_logits(...)["p_cum"][:, 3]`, the column the per-sample artifacts store — not
`logit_3`, which is the conditional. The BCE is **unweighted**, so λ is the only knob; grade 4 is
~8% of training, making this a deliberately mild intervention.

**Frozen values.** λ ∈ {**0.05, 0.10, 0.20**}; seeds {**42, 123, 2026**}. 12-run manifest:
3 baseline + 9 auxiliary.

**Baseline is reused, not retrained.** The frozen NO_RACAF runs of `improved_multiseed_2026_09` at
the same seeds **are** the λ=0 arm by construction — same `build_no_racaf_joint_model_matched_init`
via `msr.build_arm_model`, same seed, data path, optimizer, schedule and `val_QWK` monitor. Nothing
in `TRAINING_BEHAVIOR_SOURCES` has changed since they were trained. They are read read-only.

**Unchanged:** Stage 01–07, lesion/vessel segmentation, RACAF, reliability, preprocessing, the
authoritative split, augmentation, batch size, optimizer, LR schedule, regularisation, backbone,
mixed precision, and checkpoint selection (`val_QWK`, max).

**Primary outcome: AUROC of `p_gt_3` on Grade 4 vs Grade 3** — *not* QWK, *not* Grade-4 recall,
*not* any tuned threshold. Reference points from §11: `p_gt_3` 0.5979, lesion-only probe 0.6700,
combined 0.6826.

**Decision rule, frozen in advance.**
- *Boundary improved* for a λ: mean ΔAUROC ≥ **+0.03** AND Δ > 0 in **3/3** seeds. The 0.03 is ~40%
  of §11's 0.072 gap; 3/3 sign consistency is required because the Hanley–McNeil SE of a single
  AUROC at 39 vs 58 cases is ≈0.058, and it mirrors the six-run experiment's own rule.
- *Performance preserved*: mean ΔQWK ≥ −0.02 AND mean ΔMAE ≤ +0.03.
- Per λ: SUPPORTIVE = improved and preserved; TRADE_OFF = improved, not preserved; NOT_SUPPORTIVE =
  not improved.
- Overall: SUPPORTIVE needs **≥2 of 3 λ** SUPPORTIVE. A single passing λ out of three is reported,
  never selected — guarding against multiplicity.
- Grade-3 recall change is reported with a caution flag below −0.10, applying §10's lesson that
  adjacent-grade degradation must be visible even when it does not drive the verdict.

**Verified before launch.** (a) `_p_gt_3` equals the project's own `p_cum[:, 3]`; (b) λ=0 reproduces
`weighted_corn` exactly; (c) the BCE matches an independent NumPy computation; (d) an in-process
model rebuild after `clear_session()` reproduces a fresh-process build's initial weights
**bit-for-bit**, which is what lets the notebook loop through runs in one runtime without breaking
matched initialisation against the frozen baselines.

**Status: launched, no results.** Results will be appended as §13 when runs complete.
