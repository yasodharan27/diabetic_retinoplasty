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

---

## 13. Phase 0 — where is the Grade-3/4 signal lost? (read-only; gate for Architecture A)

**Numbering note.** §12 said the auxiliary-loss results would be appended as §13. Phase 0 finished
first, so it takes §13, and the auxiliary-loss results will be §14. §12 itself is unchanged.

Artifact: `experiments/Grade3vs4_Phase0/2026-09-23_18-09-08`. Script:
`research/Grade3vs4_Architecture_Research/phase0_grade3vs4_localisation.py` (repo commit `b25da0f`).
The design and the pre-registered rules are in `research/Grade3vs4_Architecture_Research/RECOMMENDED_DESIGN.md`
§2 and in the script's configuration block, both written before the run. No network was trained.
Nothing was fitted on validation data, no cache was regenerated, and the frozen BEST weights were only read.

**Integrity.** Population as pinned: train 154 grade-3 / 236 grade-4, validation 39 / 58 (split sha256
`bc80fd45…`). The recomputed lesion statistics reproduce §6's saved CSV on all 97 validation images
(max |Δ| 9.7e-17). Frozen inference reproduces the saved validation logits (max |Δlogit| 3.9e-3,
7.8e-3, 2.0e-3; tolerance 0.05).

### 13.1 Train-fitted lesion statistics — the Architecture A gate

| variant | val AUROC (95% CI) | AUPRC (prevalence 0.598) | dup-excluded | train 5-fold |
|---|---|---|---|---|
| **primary** (log10, standardised) | **0.6202** (0.5000–0.7414) | 0.6767 | 0.6368 | 0.6053 |
| sensitivity (untransformed) | 0.6401 (0.5234–0.7560) | 0.6933 | 0.6547 | 0.6036 |

**Pre-registered verdict: INCONCLUSIVE.** The primary variant sits at the 0.62 bar, but the lower CI
bound is 0.5000, which does not exclude chance. The untransformed variant would have passed, but it
was pre-registered as a sensitivity check only. It is reported, and it is **not** substituted for the
primary result.

§11's 0.6700 was fitted and cross-validated *inside* validation. Fitted on the training split, the
same 10 statistics reach 0.62–0.64 on validation and 0.60 in training-internal CV. §11's figure was
therefore optimistic by about 0.03–0.05, and the lesion statistics on their own are a weak 3-vs-4 signal.

### 13.2 Frozen scores — is the signal lost in the marginal `p_gt_3`? (saved predictions, six runs)

| score (grade 4 vs 3) | mean AUROC (six runs) |
|---|---|
| `p_gt_3` = `p_gt_2 · p_cond_3` (marginal) | 0.5979 |
| `p_cond_3` = σ(logit_3) (conditional) | **0.6771** |
| `p_gt_2` | 0.4691 |

Conditional − marginal: +0.0376 / +0.1260 / +0.1353 / +0.0645 / +0.1158 / −0.0044 (RACAF 42,
NO_RACAF 42, RACAF 123, NO_RACAF 123, RACAF 2026, NO_RACAF 2026), mean **+0.0791**, positive in 5/6 runs.
**Pre-registered classification: SPECIFIC_LOSS_IN_MARGINAL.** Every per-run bootstrap CI includes 0.
The evidence is the cross-run consistency, and the six runs are matched pairs on one split, not
independent observations.

The mechanism predicted in the design audit (H2) is visible: `p_gt_2` ranks grade 4 slightly
**below** grade 3 (AUROC 0.469; 0.413 for NO_RACAF/42), so multiplying by it degrades the
conditional score. This is a decode-level finding. No threshold or decode change is proposed or adopted.

### 13.3 Stacked regression and internal feature probes (NO_RACAF, train-fitted, validation scored once)

Per seed (42 / 123 / 2026) and mean:

| readout | 42 | 123 | 2026 | **mean** |
|---|---|---|---|---|
| GAP(L), Stage 05 features (PCA-16 + LR) | 0.8134 | 0.7317 | 0.7277 | **0.7576** |
| E, Stage 07 output (PCA-16 + LR) | 0.7290 | 0.6830 | 0.7175 | **0.7098** |
| stacked: 4 logits + 10 lesion statistics (LR) | 0.6755 | 0.6622 | 0.7635 | 0.7004 |
| model-only: 4 logits (LR) | 0.6516 | 0.6446 | 0.6835 | 0.6599 |
| `p_cond_3` (saved, unfitted) | 0.6459 | 0.6481 | 0.6242 | 0.6394 |
| lesion statistics only (LR) | 0.6202 | 0.6202 | 0.6202 | 0.6202 |
| GAP(G), Stage 06 RGB Swin (PCA-16 + LR) | 0.5964 | 0.6004 | 0.5592 | 0.5853 |
| frozen `p_gt_3` (unfitted) | 0.5199 | 0.5840 | 0.6286 | 0.5775 |

- **Stacking (pre-registered rule): YES.** The stacked model gains +0.1229 over frozen `p_gt_3` and
  +0.0405 over the model-only regression, both 3/3 seeds positive. Per-seed CIs on the +0.04
  exclude 0 only at seed 2026. Most of the gain over `p_gt_3` (+0.082 of +0.123) comes from
  **re-reading the model's own four logits**. The lesion statistics add the remaining ~+0.04.
- **Localisation (descriptive, margin 0.05):**
  - The Stage 05 → E drop is +0.084 / +0.049 / +0.010 (mean +0.048). It meets the margin only at seed 42.
  - The E → frozen `p_gt_3` drop is +0.209 / +0.099 / +0.089 (mean **+0.132**), 3/3 seeds.
  - The per-seed label "present_in_E_not_used_by_head_or_decode" holds in 3/3 seeds.
  - Stage 06 (RGB-only Swin) carries little 3-vs-4 information (0.585).

### 13.4 What this changes (conclusion change, stated explicitly)

**§11's conclusion is refined.** §11 read "lesion statistics beat the model" as meaning the
information is discarded somewhere between the input and `p_gt_3`. Phase 0 locates that loss more
precisely:

1. The 3-vs-4 signal is present, and **stronger than in the handcrafted lesion statistics**, in the
   model's own learned features: Stage 05 about 0.76, E about 0.71, against 0.62 for the statistics.
   These are linear probes fitted on training images and scored on held-out validation.
2. The large, seed-consistent loss is **downstream of E**. It happens in the CORN head/objective
   (E 0.710 → 4-logit readout 0.660 → `p_cond_3` 0.639) and in the marginal decode (`p_cond_3` 0.639 →
   `p_gt_3` 0.578 for NO_RACAF; +0.079 across all six runs).
3. Loss at Stage 07 aggregation (hypothesis H1, the mechanism behind Architecture A) is **not
   established**: it meets the margin in 1/3 seeds, with a mean of 0.048 against a single-AUROC SE of about 0.058.

**Architecture A (count-preserving lesion-burden pathway): NOT CLEARED.** The pre-registered gate is
INCONCLUSIVE, so A is not implemented. Independently of the gate, Phase 0 weakens A's premise:
- the representation already carries more 3-vs-4 information than the statistics A would add;
- the main loss is not the aggregation step A was designed to bypass.

A is not formally CLOSED, because the gate did not return NO-GO. It is deprioritised, and it would
need new evidence of a representation-level (not head-level) loss before being reconsidered.

**Hypothesis status after Phase 0.**
- H2 (marginal decode penalises PDR): **supported descriptively** (SPECIFIC_LOSS_IN_MARGINAL, 5/6).
- H3 (insufficient objective pressure on the 3/4 boundary): **more plausible**, because a
  training-fitted linear readout of the frozen E beats the network's own task-3 head by about 0.07
  (0.710 vs 0.639). This is still associational. It is the subject of the running §12 experiment,
  whose primary endpoint is the marginal `p_gt_3`. Given 13.2, the conditional `p_cond_3` should be
  read alongside it when §14 is recorded, as a secondary and not as a change to §12's pre-registered rule.
- H1 (Stage 07 cardinality loss): **not supported as the main loss** (1/3 seeds).

**Label-noise evidence found on the way.** Of the 41 pinned train/validation duplicate pairs, 9 are
validation grade-3/4 images. Three pairs carry **different grades** for the same image: one
validation grade 3 whose training twin is grade 4, and two validation grade 4 whose training twins
are grade 2. This is direct evidence of label noise at exactly this boundary. It sets an unknown
ceiling on every 3-vs-4 AUROC in this record.

**Limitations.**
- 97 validation images, so each AUROC's 95% CI is about ±0.11.
- One split, shared by all runs.
- The probes are linear PCA-16 readouts with fixed hyperparameters, fitted on 390 training images whose
  features the network was itself trained on. Their held-out AUROCs are valid but may be conservative.
- Several readouts were compared. Only the gate and the analysis-2 and analysis-3 rules were
  pre-registered; the localisation labels are descriptive.
- The laser-scar (EX) confound is not assessed.

**Status.** Architecture A is not cleared. The evidence now points at the **head/objective and
decode** level, not at representation or aggregation. Nothing new is launched. The §12 auxiliary-loss
experiment continues unchanged, and its results will be recorded as §14.

---

## 14. Grade-4 auxiliary loss — results (the §12 pre-registered experiment)

Artifacts: `experiments/Grade4AuxLoss/grade4_aux_loss_2026_09/` (`REPORT.md`, `comparison.csv`,
`per_run_results.csv`, `results.json`). All **12/12** runs completed. The baselines are the frozen
NO_RACAF runs, reused as §12 specified. The decision rule is exactly as frozen in §12; nothing was
changed after the results were seen.

| λ | mean ΔAUROC `p_gt_3` (g4 vs g3) | seeds up | mean ΔQWK | mean ΔMAE | per-λ verdict |
|---|---|---|---|---|---|
| 0.05 | −0.0175 | 1/3 | −0.0038 | +0.0119 | NOT_SUPPORTIVE |
| 0.10 | −0.0113 | 1/3 | +0.0002 | +0.0037 | NOT_SUPPORTIVE |
| 0.20 | +0.0060 | 2/3 | −0.0059 | +0.0073 | NOT_SUPPORTIVE |

**Pre-registered overall verdict: NOT_SUPPORTIVE** (0 of 3 λ SUPPORTIVE; ≥2 were required).

**Reading.**
- No λ comes near the +0.03 bar, and no λ is sign-consistent. The largest mean change (+0.006 at
  λ = 0.20) is a fifth of the bar and far inside the noise (single-AUROC SE ≈ 0.058 at 39 vs 58).
- The overall task was preserved at every λ (|ΔQWK| ≤ 0.006, ΔMAE ≤ +0.012). The intervention was
  benign and ineffective, not harmful.
- The means rise with λ (−0.018 → −0.011 → +0.006). With three seeds and magnitudes this far below
  the bar, that is **not** interpreted as a dose-response. It does **not** justify a larger-λ rerun,
  which would be selecting on the result.

**What this closes.** The hypothesis §12 tested, that a mild, unweighted auxiliary BCE on the
marginal `p_gt_3` (grade 4 vs all) recovers the 3/4 boundary, is **CLOSED** at λ ≤ 0.20. Do not rerun
it at larger λ, with class weighting, or with more seeds without new mechanistic evidence.

**What it does not close (post-hoc, labelled as such — explanations of a null, not rescues).**
- The auxiliary target was **grade 4 vs all grades**. In training, grade 3 is 154 of the 2,693
  negatives (5.7 %). Most of the added gradient therefore separated grade 4 from grades 0–2, which
  `p_gt_3` already ranks at AUROC ≈ 0.88 (§8). Pressure on the 3/4 boundary *specifically* was
  small by construction. This is a design limitation of §12, which was written before §13 existed.
- The marginal-BCE gradient also pushes `σ(z_0..z_2)` down for grade-3 samples, against CORN
  (ARCHITECTURE_AUDIT L7 note). The auxiliary term acted on the one quantity §13 showed is degraded
  by the decode.
- §14 therefore shows that **this** objective change fails. It does **not** show that no
  boundary-specific objective could succeed. H3, stated narrowly as "the 3/4 conditional task gets
  too little pressure", remains open: §13 found a training-fitted linear readout of the frozen E at
  0.710 against the network's own conditional head at 0.639.

**Secondary reading promised in §13 (not yet available).** §13 said the conditional `p_cond_3`
AUROC would be read alongside the primary endpoint here. The notebook's summary does not report
it. It will be added below as a clearly labelled addendum once read from the saved per-sample
files, and it cannot change the pre-registered verdict above.

**State of the Grade-3/4 problem after §14.**
- **CLOSED:** the threshold as a representation fix (§10: a calibration dial only); the
  generic-vessel route (§7); the mild marginal grade-4 auxiliary BCE (§14).
- **NOT CLEARED:** Architecture A, the lesion-burden pathway (§13 gate INCONCLUSIVE, premise weakened).
- **ESTABLISHED (§13):** the 3-vs-4 signal is present in the learned features (Stage 05 ≈ 0.76,
  E ≈ 0.71). It is lost mainly in the head/objective (→ 0.64 conditional) and in the marginal
  decode (→ 0.58).
- **OPEN:** whether the CORN head can be made to use what E already carries. This is a head-level
  question, not a representation or threshold question. Nothing is launched or proposed here.

### 14.1 Addendum — secondary reading: the conditional `p_cond_3` (promised in §13; cannot change the verdict)

Read-only: the aux experiment's saved `per_sample_best.csv` files, with the frozen NO_RACAF files as
the λ = 0 reference. Paired stratified bootstrap, 2,000 resamples, seed 20260925.

**Consistency checks passed.**
- The baseline marginal AUROCs (0.5199 / 0.5836 / 0.6286) match `comparison.csv` and §8.
- The baseline conditional mean (0.6394) matches §13 exactly.
- The marginal deltas reproduce the notebook's summary (−0.0175 / −0.0113 / +0.0060).
- Image membership and grades are identical between each aux run and its baseline.

Grade 4 vs grade 3, validation (n = 97). Δ is aux − baseline at the same seed:

| λ | seed | Δ marginal `p_gt_3` (95% CI) | **Δ conditional `p_cond_3` (95% CI)** | AUROC `p_gt_2` |
|---|---|---|---|---|
| 0.05 | 42 | +0.018 (−0.046, +0.084) | **+0.092 (+0.027, +0.163)** | 0.401 |
| 0.05 | 123 | −0.021 (−0.095, +0.050) | **+0.065 (+0.012, +0.125)** | 0.396 |
| 0.05 | 2026 | −0.050 (−0.094, −0.004) | +0.000 (−0.073, +0.078) | 0.471 |
| 0.10 | 42 | +0.027 (−0.053, +0.113) | +0.053 (−0.050, +0.156) | 0.422 |
| 0.10 | 123 | −0.060 (−0.155, +0.033) | **+0.073 (+0.011, +0.144)** | 0.417 |
| 0.10 | 2026 | −0.001 (−0.056, +0.054) | +0.033 (−0.062, +0.123) | 0.499 |
| 0.20 | 42 | +0.038 (−0.016, +0.098) | +0.046 (−0.008, +0.105) | 0.430 |
| 0.20 | 123 | +0.013 (−0.019, +0.049) | +0.008 (−0.034, +0.050) | 0.466 |
| 0.20 | 2026 | −0.034 (−0.102, +0.032) | +0.029 (−0.092, +0.144) | 0.488 |

| λ | mean conditional AUROC | mean Δ conditional (seeds up) | mean Δ marginal (seeds up) | mean Δ grade-3 recall |
|---|---|---|---|---|
| 0 (baseline) | 0.6394 | — | — | — |
| 0.05 | 0.6919 | **+0.0525 (3/3)** | −0.0175 (1/3) | +0.086 |
| 0.10 | 0.6922 | **+0.0528 (3/3)** | −0.0113 (1/3) | +0.017 |
| 0.20 | 0.6670 | +0.0276 (3/3) | +0.0060 (2/3) | −0.043 |

**Reading (post-hoc, labelled).**
1. **The auxiliary loss did sharpen the 3/4 boundary — in the conditional component.** The Δ
   conditional is positive in **9/9** aux runs. It is positive in 3/3 seeds at every λ, and four
   of the nine per-run CIs exclude 0. The nine runs are not independent: the three λ share each
   seed's baseline and the same 97 images. The honest unit is "3/3 seeds at each of three λ".
2. **The marginal decode cancelled it.** `p_gt_2` still ranks grade 4 *below* grade 3 in every run
   (AUROC 0.40–0.50), and `p_gt_3 = p_gt_2 · p_cond_3`. The conditional gain therefore never reached
   the primary endpoint. This is the §13 finding (SPECIFIC_LOSS_IN_MARGINAL) reproduced in nine
   independently trained models.
3. The conditional reaches ≈ 0.69 at λ = 0.05 and 0.10, close to the frozen-E linear probe in §13
   (0.710). Those are different models, so this is a comparison of levels, not a matched measurement.
4. Grade-3 recall does not collapse (+0.086, +0.017, −0.043). λ = 0.10 / seed 42 predicts **no**
   grade-4 images at the default decode (grade-4 recall 0.000), an instability of the 0.5-rule
   decode noted here for completeness.

**What this does and does not change.**
- **The §14 verdict is unchanged: NOT_SUPPORTIVE.** The primary endpoint was the marginal `p_gt_3`.
  Swapping in the conditional after seeing results would be endpoint switching. Applying §12's rule
  to the conditional *post hoc*, λ = 0.05 and 0.10 would pass the "improved" bar (≥ +0.03, 3/3).
  That is recorded only to state how large the effect is, and **it is not a verdict**.
- **§14's reading is refined.** Mild objective pressure *does* move the 3/4 boundary that the CORN
  head learns (H3, narrowly stated, gains support). The marginal construction of `p_gt_3` prevents
  the gain from surfacing (H2, now seen in both frozen and newly trained models). The two
  hypotheses are linked: the bottleneck the evidence now isolates is **how the grade-4 score is
  built from the CORN outputs**, not the representation and not the amount of training pressure alone.
- **Nothing is reopened or adopted.** "Mild marginal aux BCE as a fix for `p_gt_3`" stays CLOSED. No
  decode change is adopted.
- **Any future experiment on this question must pre-register the conditional (or any alternative
  grade-4 score) as its primary endpoint before training.** This addendum is hypothesis-generating only.

> **Caveat (added 2026-09-25):** the 9/9 `p_cond_3` gain in §14.1 is confounded and is **not** causal evidence that the auxiliary Grade-4 pressure caused it. All nine runs were compared against the same three reused NO_RACAF baselines, and RACAF, a mechanistically inert change, also exceeded those baselines on `p_cond_3` (+0.011 / +0.076 / +0.139). The pre-registered task-3 vs mass-matched-placebo experiment is the specificity test.

---

## 15. Label-noise and Grade-3/4 ceiling audit (read-only)

Artifact: `experiments/Grade3vs4_LabelNoiseAudit/2026-09-27_09-39-11/label_noise_audit.json`.
Script: `research/Grade3vs4_Architecture_Research/label_noise_ceiling_audit.py`. The rules and the
0.85 threshold were fixed in the script before it ran. Nothing was trained, inferred, relabelled or
written except that JSON. The task-3 vs placebo experiment was **running** and its predictions were
deliberately not read.

**Duplicates (SHA-256 of all 3,662 images; split sha256 verified; 0 missing).**
- There are 3,534 unique images, in 123 duplicate groups covering 251 images. Group sizes: 119 × 2,
  3 × 3, 1 × 4.
- By split: 76 groups are train-only, 41 train+val, 6 val-only.
- The pinned 41-group train/validation list was reproduced exactly.
- **New:** 76 within-training duplicate groups exist. The earlier audit covered only train/validation.

**Label disagreement on byte-identical images.**

| Population | Disagreeing | Rate (95 % CI) |
|---|---|---|
| train/val pairs, all | 10 / 42 | 0.238 (0.121–0.395) |
| train/val pairs touching grade 3 or 4 | 6 / 12 | 0.500 (0.211–0.789) |
| train/val pairs with **both** labels in {3,4} | **1 / 7** | 0.143 (0.004–0.579) |
| all duplicate pairs incl. within-split | 32 / 134 | 0.239 (0.169–0.320) |

- Grade-4 images whose duplicate is graded ≤ 2: **9** (label-level r4 = 9/36 = 0.25).
- Grade-3 images whose duplicate carries a different grade: **13**.
- Train/val transitions (val → train): 3→4, 2→3, 1→4, 2→4, 4→2 ×2, 1→2 ×2, 0→1, 2→1.
- Disagreement is higher in pairs touching grade 3/4 than elsewhere (0.500 vs 0.133, one-sided
  Fisher p = 0.020). It is **mostly grade 4 vs ≤ 2, not 3 vs 4**.

**Correction to earlier figures.** The `NEXT_STEP_` design documents and my earlier count gave 8/41
disagreeing train/val pairs. The correct figure is **10/42 pairs (9/41 groups)**: one row of the
duplicate report lists two training twins (`51131b48f9d4`, val 1 vs train 2, 2) and was dropped by
a table parse. The grade-3/4 figures (6/12) are unchanged.

**Ceiling (a model-dependent sensitivity analysis, not an empirical ceiling).**
- M1 treats 3↔4 flips as independent of image features: the per-label flip e comes from d = 2e(1−e).
- M2 treats contamination: "4" labels whose true grade is ≤ 2.
- From the 1/7 pure-3/4 rate, e = 0.077, giving a perfect-model ceiling of 0.923.
- Combined with contamination c = r4/2 = 0.125 (neutral s = 0.5), the ceiling is **0.870**.
- The worst case (every CI upper bound, s = 0) is 0.289, which is not informative.
- Under the point estimate, the recorded readouts would correspond to true AUROCs about 0.02–0.05
  higher (E probe 0.710 → 0.748; `p_cond_3` 0.639 → 0.665).
- The three disputed validation grade-3/4 images can move any AUROC by at most ±0.060.

**Pre-registered plateau verdict: POSSIBLE BUT INSUFFICIENT EVIDENCE.** The point-estimate ceiling
(0.870) is above the 0.85 threshold, so noise at the observed rates would explain only a few
hundredths of the plateau. Only the worst case is binding.

Two limits must stay attached to this verdict:
- **The rule could not have returned "UNLIKELY" with this sample.** The CI upper bound of 1/7 maps
  to e = 0.5, so the worst case always falls below 0.85. This is a design weakness of the audit,
  recorded as such.
- **The M1 rate uses only the 7 train/val pairs with both labels in {3,4}.** The whole-dataset groups
  (134 pairs, 13 grade-3 images with a disagreeing twin) were not broken down by 3↔4 versus 3↔other
  grade. The 3↔4 flip rate is therefore estimated from very little data.

**PDR decoded ≤ 2 (validation grade 4, n = 58).**
- **21 (36 %) are decoded ≤ 2 by all three NO_RACAF seeds.** 16 of those are also ≤ 2 in all nine
  NO_RACAF-architecture aux runs.
- Only 2 of the 21 have a byte-identical duplicate:
  - `f03d3c4ce7fb` conflicts (its training twin is graded 2);
  - `65c958379680` agrees (its twin is also graded 4).
- For comparison, 1/37 of the other grade-4 images has a conflicting duplicate.
- **Duplicate coverage is too sparse (2/21) to test whether this failure is label noise.** The audit
  neither supports nor refutes that explanation, and at least one consistently-≤ 2 image carries a
  grade 4 confirmed by two gradings.

**A pattern in the review sheet (descriptive; the interpretation is a HYPOTHESIS).** The 21 images
have very low `p_gt_2` (0.03–0.47) but high `p_cond_3` (0.60–0.92). Four are decoded as grade 1 by
all three seeds.
- The conditional grade-3→4 task says "PDR rather than severe NPDR", while the chain of tasks 0–2
  says "not even grade 3".
- This fits the non-monotonic-burden mechanism: task 3 learned *lower burden → PDR* while tasks 0–2
  learned *lower burden → milder*. Low-burden PDR images are therefore gated out before task 3 is
  ever reached.
- Candidate explanations for the low burden are treated/regressed PDR (laser scars), PDR signs with
  no channel (NV, fibrous proliferation, vitreous haemorrhage), and poor image quality. Only a visual
  review can separate them.

**Answers (audit's pre-stated questions).**
1. Confirmed train/val label inconsistency: **yes** (10/42 pairs).
2. Concentrated around grade 3/4: **yes** (p = 0.020), but mostly 4 vs ≤ 2.
3. Large enough to materially limit 3-vs-4 AUROC: **possible but insufficient evidence**.
4. Numerical ceiling established: **no**.
5. More 3-vs-4 training now: **no**.
6. New PDR-specific annotations now: **no**.
7. Next: **another read-only step first**, a visual review of the 21-image sheet. External IDRiD
   evaluation stays tied to the pending task-3 vs placebo verdict.

**What changes.** Label noise is **confirmed to exist** and to sit mostly at the 4-vs-≤ 2 boundary.
It is **not shown** to explain the 3-vs-4 plateau: at the observed 3↔4 rate it accounts for only a
few hundredths. It remains **untested** as an explanation of the PDR → ≤ 2 failure, because
duplicate coverage there is too sparse. The PDR → ≤ 2 failure is now characterised: a consistent
set of 16–21 images with low `p_gt_2` and high `p_cond_3`, suited to visual review.

### 15.1 Visual review of the 21 PDR → ≤ 2 images (read-only, human, qualitative)

Material: the contact sheet from `research/Grade3vs4_Architecture_Research/pdr_visual_review_sheet.py`
(`experiments/Grade3vs4_VisualReview/<timestamp>/`). It shows the 21 Audit-5 images and a 10-image
comparison group, each as the full original fundus plus a fixed geometric central crop, unenhanced.
The review was done by eye on that sheet by the project reviewer. **It is not an adjudicated clinical
grading and no label was changed.** No automated image analysis was performed, and none will be.

**Observations (as reported; the categories are approximate and visual, not classifications).**
- **Heterogeneous, not one failure type.** The 21 cannot be collapsed into a single explanation.
- **Relatively low visible burden: about 15 cases** (`eaa0dfbd5024` … `887c26fc0e1f`). They show
  scattered small bright deposits and/or haemorrhage-like dots, with **no obvious large neovascular
  complex or unmistakable advanced PDR morphology** at contact-sheet resolution. This does not show
  that their labels are wrong.
- **Major dark/haemorrhagic obscuration: 3 cases** (`211518c46162`, `789434d095d1`,
  `84c663f39632`). Large dark regions obscure much of the fundus, around the disc/vessels in the
  first. The underlying morphology cannot be characterised from the sheet.
- **Appreciable lesion burden without an obvious neovascular complex:** several later cases
  (`6cfb7b44ef6f`, `4bd941611343`; `b3819a805dca` more subtle).
- **No obvious widespread panretinal laser pattern** anywhere in the failure set. This is *not* a
  finding that none are treated: subtle scars can be missed at this resolution and some fields are
  obscured.
- **The comparison group is itself heterogeneous.** Some images have heavy bright-lesion or
  haemorrhagic burden; others look much less dramatic. The models do not simply fail every visually
  subtle grade-4 image, and grade 4 is not one uniform phenotype here.
- **Terminology correction.** The comparison rule is "predicted 4 by at least one seed", so some
  members (e.g. `1c4f3aa4df06`, `3206171db5be`, predictions 2/2/4) are **not** correctly graded by
  all three seeds. The group is a *comparison group*, not "consistently correct" cases.

**Relation to the §15 hypotheses (interpretation, labelled).**
- *Treated / regressed PDR as the dominant explanation*: **not supported** by this review.
- *Label error as the dominant explanation*: **not established.** The review cannot test it, and
  the one known conflicting duplicate (`f03d3c4ce7fb`, train twin graded 2) falls in the low-burden
  group without being visually distinctive at this resolution.
- *Missing PDR-specific features*: **plausible for the obscured subset (HYPOTHESIS).** Large
  preretinal or vitreous haemorrhage is itself a PDR-defining sign under ICDR, and the pipeline has
  no channel for it. If full-resolution review confirms haemorrhagic obscuration, those three would
  be genuine PDR the model cannot represent. The dark regions are not identified as haemorrhage here.
- *Burden reversal / chain gating (§15 pattern)*: **consistent with the low-burden majority** (low
  `p_gt_2`, high `p_cond_3`). Subtle neovascularisation invisible at sheet resolution cannot be
  excluded.

**Status.** The §15 verdict is **unchanged**:
- label noise: confirmed, insufficient to explain the 3-vs-4 plateau;
- PDR → ≤ 2 failures: real and heterogeneous; cause unresolved;
- no new training or annotation justified now;
- the pending task-3 vs placebo verdict still decides the next step.

Four cases are flagged for individual full-resolution inspection:
- the three obscured cases, `211518c46162`, `789434d095d1` and `84c663f39632`;
- `f03d3c4ce7fb`, the known 4 ↔ 2 duplicate conflict.

Three more (`887c26fc0e1f`, `6cfb7b44ef6f`, `4bd941611343`) are secondary: visible burden despite a
uniform 2/2/2 prediction.

### 15.2 Second visual review, at near-native resolution (read-only, qualitative, AI-assisted)

The contact sheet was cut into per-case panels (temporary files, discarded) and all 31 cases were
viewed at about 1:1 sheet resolution, full fundus plus central crop. **The reviewer is an AI
assistant, not an ophthalmologist.** The morphological calls below are tentative descriptions of
appearance, not diagnoses. No label was changed and no automated image analysis was run.

**Correction to §15.1's case attribution.** At this resolution the three cases with a **large, dense,
dark haemorrhage** are `211518c46162`, `84c663f39632` and **`4bd941611343`**.
- `4bd941611343` has the largest: a dense, sharply bounded dark mass covering much of the inferior
  macula and temporal fundus.
- `789434d095d1`, listed in §15.1 as obscured, shows scattered blot haemorrhages and peripheral round
  spots, not a large obscuring region.
§15.1's list most likely shifted by position. §15.1 is left unchanged, and this is the corrected
reading.

**Failure cases (21): observed appearance, grouped; groups overlap.**

| Appearance | Cases | Note |
|---|---|---|
| Large, dense, dark haemorrhage with sharp margins (boat-shaped / sub-hyaloid-like) | `211518c46162` (peripapillary, with irregular fronds), `84c663f39632`, `4bd941611343`; smaller or possible: `b37aae3c8fe1`, `63b4d030b016`, `8bed09514c3b` | Morphology *compatible with* preretinal haemorrhage, a PDR-defining sign. Not confirmed |
| Round, regularly spaced pale/greenish spots in the mid-periphery | clear: `b3819a805dca` (a ring along the temporal edge), `6cfb7b44ef6f` (superior); possible: `789434d095d1`, `1bf30c84bbad`, `82bb8a01935f`, `d1a24527a15d` | Morphology *compatible with* laser photocoagulation scars; an optical artefact cannot be excluded |
| Marked haze / low contrast (media opacity) | `6cd606dc52e9`, `1bf30c84bbad`, `cd54d022e37d`; partial: `eaa0dfbd5024`, `8bed09514c3b` | Could be vitreous haemorrhage, cataract or capture quality. Not distinguishable here |
| Tortuous / looping vessel segments, dense flame haemorrhages, cotton-wool spots | `e3ab63dc9a60` (superotemporal loops), `fce93caa4758`, `887c26fc0e1f` | Features at least severe-NPDR-like. IRMA vs NVE not distinguishable |
| Exudates and haemorrhages without the above | `d48178e4a49b`, `65c958379680`, `f03d3c4ce7fb` | `65c958379680` and `f03d3c4ce7fb` show similar posterised dark inferior bands (likely capture/compression artefact) |

**Comparison group (10).**
- Florid proliferative appearance in `29bc0e721cfe` (extensive haemorrhage, disc changes) and
  `2fde69f20585` (fibrovascular/tractional membranes).
- Dense scattered haemorrhages in `3ac3fbfca7d4`, the only 4/4/4 case.
- **Laser-scar-like peripheral spots are also common here:** `3f752fcccec0` (a ring), `2017cd92c63d`
  (many, some pigmented), `3810040096cb`, `1c4f3aa4df06`, `3b185ac445d0`. **All of these are called 4
  by only one seed**, mostly seed 2026 (3/3/4 or 2/2/4).
- `3b185ac445d0` is a small image, visibly upsampled.

**Reading (interpretation, labelled; not established).**
1. **§15.1's "no obvious laser pattern" does not hold at this resolution.** Laser-scar-like spots
   appear in at least 2 failure cases clearly and 4 possibly, and in about 5 of 10 comparison cases.
   Where they occur in the comparison group, only one seed calls grade 4. **Treated-looking eyes
   appear systematically under-graded in both groups.**
2. Most failure cases show a feature that plausibly belongs to PDR or treated PDR: preretinal-type
   haemorrhage, laser-like scars, or haze possibly from vitreous haemorrhage. **None of these has a
   representation in the four Stage-04 channels (MA, HE, EX, SE).** This makes the PDR → ≤ 2 failure
   look more like **genuine PDR the pipeline cannot represent (mechanism 8)** than like label error
   (mechanism 7). It fits the chain pattern in §15: treated or haemorrhage-dominated eyes carry few
   *intraretinal* MA/HE/EX lesions, which gives low `p_gt_2`.
3. **The model does not treat a large preretinal-type haemorrhage as severe.** `4bd941611343` has the
   largest haemorrhage in the set and is called 2/2/2. Stage 04's HE channel was trained on IDRiD
   intraretinal haemorrhage and may not represent it.
4. Florid proliferative cases (`29bc0e721cfe`, `2fde69f20585`) are recognised at least partly, so
   the failure is concentrated in treated, haemorrhage-dominated and hazy PDR, not in florid PDR.

**Consequence for the open questions.**
- The PDR-specific-feature direction (NEXT_STEP_RECOMMENDATION C5) moves from *weakly supported* to
  **plausible but unproven** for the PDR → ≤ 2 failure.
- The relevant features are narrower than NV. They are **preretinal/vitreous haemorrhage and laser
  scars**, for which public labels exist: Retinal-Lesions has preretinal and vitreous haemorrhage
  masks, and RFMiD has image-level laser-scar labels.
- Label noise remains confirmed but is **less likely** to be the main driver of this failure.
- These are single-reviewer, AI-assisted visual impressions on downsampled images. **They require
  confirmation by a qualified grader before any design decision rests on them.**
- **Nothing is launched.** The pending task-3 vs placebo verdict still governs the next step.

---

## 16. Task-3 weighting vs mass-matched placebo — results (pre-registered)

Artifacts: `experiments/Task3WeightingPlacebo/2026-09-26_02-59-26/` (`PREREGISTRATION.json`, frozen
before training; `REPORT.md`, `comparison.csv`, `bootstrap_results.csv`, `per_run_results.csv`,
`results.json`, `run_status.json`). Notebook `colab/notebooks/task3_weighting_placebo_experiment.ipynb`
(commit `3de1fa3`), loss module `task_weighted_corn.py`.

Arms: B = frozen NO_RACAF (read-only); T = CORN task 3 × 2.0; P = task 1 × 1.3792 (equal added loss
mass, 731.2). Seeds 42 / 123 / 2026, all six trained runs complete. Primary endpoint: AUROC of
`p_cond_3 = σ(z₃)`, grade 4 vs grade 3, APTOS validation. Primary contrast T − P.

| Contrast | seed 42 | seed 123 | seed 2026 | mean | rule | result |
|---|---|---|---|---|---|---|
| **T − P** (primary) | +0.0588 | +0.0137 | −0.0186 | **+0.0180** | ≥ +0.03 and 3/3 | **SPECIFIC = False** |
| T − B | +0.0469 | +0.0550 | +0.0153 | +0.0391 | ≥ +0.03 and 3/3 | IMPROVED = True |
| P − B | −0.0119 | +0.0413 | +0.0338 | +0.0211 | descriptive, \|mean\| ≥ 0.03 and same sign | NO_CLEAR_PERTURBATION_EFFECT |

Guardrails, T vs B (all within bounds): ΔQWK −0.0109 (≥ −0.02), ΔMAE +0.0178 (≤ +0.03),
Δ grade-3 recall **+0.0855** (≥ −0.10), Δ AUROC grade 4 vs rest −0.0139 (≥ −0.02). PRESERVED = True.

**Pre-registered verdict: NONSPECIFIC.** T improves `p_cond_3` over the frozen baselines in 3/3
seeds, but not specifically beyond the equal-mass placebo: T > P in only 2/3 seeds, mean +0.018
against the +0.03 bar.

**Reading.**
- The improvement over B (+0.039) splits roughly into a generic part (P − B ≈ +0.021) and a
  task-3-specific remainder (T − P ≈ +0.018). Neither part is separable from noise with three seeds.
  A single-AUROC SE is about 0.058, and T − P changes sign across seeds.
- **Every retrained perturbation of this model has now scored above the three frozen NO_RACAF
  baselines on `p_cond_3`:** RACAF +0.075, marginal aux +0.028 to +0.053 (§14.1), P +0.021, T +0.039.
  This is consistent with the §14.1 caveat: the frozen baselines sit low in the `p_cond_3`
  distribution, so gains measured against them overstate any intervention effect.
- Task-3 weighting did not harm the ordinal task. Grade-3 recall rose (+0.085), so the trade-off
  pattern of §10 did not appear.

**What this closes (per NEXT_STEP_DECISION_TREE.md, branch B, fixed before the result).**
- **Task-3 loss weighting as a way to improve the conditional 3-vs-4 boundary: CLOSED** (at β₃ = 2).
  It is not rerun with another β or more seeds.
- **§14's 9/9 `p_cond_3` gain is most plausibly the same non-specific effect.** Training-objective
  manipulation for this boundary (marginal aux and conditional weighting) is closed.
- Mechanism 2 (the head under-using E) is **weakened, not closed**. The E-probe gap (0.71 vs 0.64)
  may partly be the same between-run variance.
- Still standing: the estimand finding (`p_cond_3` is the correct 3-vs-4 score), and the §15.2
  observation that PDR → ≤ 2 failures concentrate in treated, haemorrhage-dominated and hazy eyes
  (plausible but unproven).

**Next step (pre-specified for NONSPECIFIC): Step VAR**, a read-only `p_cond_3` variance and power
audit over every saved run (B ×3, RACAF ×3, aux ×9, T ×3, P ×3). It quantifies the between-run
spread and the minimum detectable effect before any further 3-vs-4 experiment is considered. Not
launched.

### 16.1 Addendum — secondary endpoints, LAST checkpoint and per-seed detail (descriptive; verdict unchanged)

Source: the experiment's saved `per_run_results.csv`, `bootstrap_results.csv` and `comparison.csv`,
read with a read-only script. Nothing here can change the §16 verdict.

**`p_cond_3` AUROC (grade 4 vs 3), BEST vs LAST checkpoint, per seed:**

| Arm | BEST 42 / 123 / 2026 | BEST mean | LAST 42 / 123 / 2026 | LAST mean |
|---|---|---|---|---|
| B | 0.646 / 0.648 / 0.624 | 0.639 | 0.685 / 0.630 / **0.766** | **0.693** |
| P | 0.634 / 0.689 / 0.658 | 0.661 | 0.707 / 0.679 / 0.698 | 0.695 |
| T | 0.693 / 0.703 / 0.640 | 0.679 | 0.671 / 0.701 / 0.632 | 0.668 |

**The ordering reverses at LAST.** Mean T − B is +0.039 at BEST but **−0.025 at LAST**; T − P goes
from +0.018 to −0.027. B's own `p_cond_3` moves by up to +0.142 between its BEST and LAST checkpoints
(seed 2026: 0.624 → 0.766).

**Consequence (interpretation, labelled).** The "frozen baselines sit low" effect in §14.1 and §16
is at least partly a **checkpoint-selection effect**. BEST is chosen by val_QWK, which is nearly
blind to 3↔4, so the epoch it picks can land anywhere in the run's `p_cond_3` trajectory, and for B
it landed low. Within-run, epoch-to-epoch variation of `p_cond_3` (up to ~0.14) is **larger than
every intervention effect tested on this boundary** (0.02–0.05). All 3-vs-4 effects measured at a
QWK-selected checkpoint carry this noise. This strengthens the NONSPECIFIC reading and makes Step
VAR the necessary next step. Only BEST and LAST are evaluated, so the full within-run trajectory of
`p_cond_3` is not observable from saved artifacts.

**Other secondary endpoints (BEST, arm means; B / P / T).**
- `p_gt_3` 4-vs-3: 0.577 / 0.564 / 0.601. `p_gt_2` 4-vs-3: 0.446 / 0.461 / 0.431, still inverted in
  every arm.
- Grade-4-vs-rest AUROC: 0.885 / 0.861 / 0.871. Both new arms are lower; the per-seed CIs exclude 0
  for P at seeds 42 and 2026 and for T at seed 2026.
- Grade-4 recall: 0.161 / 0.126 / 0.115, lower in both arms. Grade-3 recall: 0.436 / 0.521 / 0.521,
  higher in both. PDR decoded ≤ 2: 31.7 / 25.0 / 27.7 of 58, fewer in both. The shift of PDR images
  from ≤ 2 into grade 3 is **non-specific** (P shows it as strongly as T).
- QWK: 0.821 / 0.812 / 0.810. The mean guardrail change (−0.011) hides **seed 42**: both T and P
  early-stopped at 22 epochs with BEST at epoch 9 (QWK 0.788 / 0.776 vs B 0.830), and **both predict
  no grade-4 image at all** (grade-4 recall 0.000). Seed 42 was unstable under both loss
  perturbations. It did not breach the pre-registered mean guardrails.
- Brier of `p_cond_3` on grade 3/4: 0.232 / 0.228 / 0.225, essentially unchanged.
- At LAST, T and P lose more QWK than B (T 0.756, P 0.781, B 0.800), and grade-3 recall falls in all
  arms.

**Summary.** The secondary detail agrees with NONSPECIFIC and adds one finding with consequences for
the programme: **`p_cond_3` at a QWK-selected checkpoint is dominated by checkpoint-selection
variance of the same size as, or larger than, the effects being tested.** The T − P and T − B signs
are not stable between BEST and LAST.

---

## 17. Step VAR — `p_cond_3` variance and power audit (read-only)

A read-only copy-paste script (no model, no GPU, nothing written) read every saved per-sample file:
**23 models** (B ×3, RACAF ×3, aux ×9, T ×3, P ×3, C1, C3; none missing). The rules were fixed in the
script before any number was seen (NEXT_STEP_RECOMMENDATION.md, Step VAR). Population: 39 grade-3,
58 grade-4 validation images.

**Spread of `p_cond_3` AUROC (grade 4 vs 3).**
- Between runs (BEST): SD **0.0376** across all 23; **0.0348** across the 18 NO_RACAF-architecture
  runs (range 0.624–0.738). Per seed: 0.038 / 0.030 / 0.016.
- Within runs, LAST − BEST (n = 14): mean **+0.023**, SD **0.044**, max 0.141 (B seed 2026).
- Validation-sample SE of one AUROC (bootstrap): **0.059**.
- **The frozen B BEST values sit at the 22nd, 28th and 0th percentiles** of the NO_RACAF-architecture
  distribution. The baselines every comparison since §12 was measured against are among the lowest
  runs in the family, which confirms the §14.1 caveat quantitatively.

**Noise that a paired intervention must overcome** (45 within-seed pairs among B, aux ×3, T, P; BEST):

| Component | SD |
|---|---|
| paired difference, observed (σ_pair) | **0.0421** |
| validation-sample part (paired bootstrap SE, rms) | **0.0410** |
| training / checkpoint part, √(σ_pair² − SE²) | **0.0098** |

**Power (paired normal model, σ = σ_pair).**
- A null intervention passes the project rule (mean ≥ 0.03, 3/3 seeds) with probability **0.069**.
- The minimum detectable effect at 3 seeds (80 % pass probability) is **0.065**; using checkpoint
  noise instead, 0.070.
- Seeds per arm for 80 % power at a true effect of 0.03 (one-sided paired t-test): **14**.
- Secondary: for `p_gt_3`, σ_pair = 0.0446.

**Pre-registered verdict: NOT_RESOLVABLE_AT_FEASIBLE_COST.** The MDE of 0.065 exceeds the 0.05
limit, and 14 seeds exceeds the 10-seed limit.

**Reading (interpretation, labelled).**
1. **Every effect tested on this boundary lies below the detectable range.** That includes aux
   (≈ 0.03–0.05 vs B), task-3 weighting (T − P 0.018) and RACAF (0.075 vs a low B). The NULL and
   NONSPECIFIC verdicts of §14 and §16 are therefore *also* consistent with real effects of about
   0.02–0.04 that the design could not see. They are not evidence of absence.
2. **The noise is almost entirely the evaluation set, not training.** Once the finite 97-image
   validation sample is accounted for, runs differ in population-level `p_cond_3` by only about 0.01
   (the training part). The bottleneck is the **number of labelled grade-3/4 evaluation images**, not
   the number of seeds. Even the 14-seed figure is optimistic: every seed is scored on the *same* 97
   images, so part of the validation-sample noise is shared across seeds and more seeds cannot remove it.
3. Projection (not a result): the validation-sample SE shrinks roughly with √(images). Several
   hundred labelled grade-3/4 images would bring the paired SD toward the training component, making
   effects of about 0.03 detectable with few seeds.

**What this closes (the closure rule pre-stated for Step VAR).** **APTOS-validation training
experiments on the 3-vs-4 boundary are CLOSED** until a larger labelled evaluation set is available.
This covers any objective, head or architecture change evaluated only on these 97 images. It does
not reopen or re-judge earlier verdicts; it bounds what they could have shown.

**What stays open.**
- A larger, independent, adjudicated evaluation set for grades 3 and 4 is the prerequisite for any
  further 3-vs-4 work. Candidates: IDRiD grading train+test with Stage-04 overlap excluded (small),
  and **DDR** (13,673 graded images, 7 graders with consensus; grade-3/4 counts to be verified).
  This is a **data/evaluation-set investigation**, not training.
- The §15.2 hypothesis (PDR → ≤ 2 failures concentrate in treated, haemorrhage-dominated and hazy
  eyes that the four lesion channels cannot represent) is a **different endpoint** (grade 4 vs rest,
  58 vs 672 images). Its noise floor is much lower (grade-4-vs-rest bootstrap CIs in §16 are about
  ±0.01–0.03) and it was not covered by this audit. It still needs qualified-grader confirmation
  first.

**Status.** No experiment is launched. The next step, if any, is a read-only feasibility check of a
larger external grade-3/4 evaluation set, and/or qualified confirmation of the §15.2 visual findings.

---

## 18. External grade-3/4 evaluation set — feasibility audit (read-only)

A read-only script (no file kept, nothing written, nothing downloaded) ran on the **project's local
copies** of IDRiD (grading + segmentation) and APTOS 2019. The Drive copies are assumed identical but
were not re-checked. Rules were fixed in the script before running.
- **Exact overlap:** SHA-256.
- **Near-duplicate:** RGB-thumbnail (96×64) mean absolute difference ≤ 3.0 on a 0–255 scale.
- **Projections:** the §17 noise model (paired validation SE 0.0410 at 39/58, scaled by the
  Hanley–McNeil ratio at AUROC 0.68; training part 0.0098).

A first version using a 64-bit perceptual hash was discarded before any conclusion was drawn:
fundus photographs collided en masse (over 1,000 false "duplicates").

**Calibration of the pixel method.**
- Confirmed copies score MAD 0.00–0.70.
- The median nearest *distinct* image scores 6.5 (IDRiD vs IDRiD) and 7.6 (IDRiD vs segmentation set).
- The nearest APTOS image to any IDRiD image scores at least 9.6.

**IDRiD (disease grading, from the label files).** 516 images: train 413 (0/1/2/3/4 = 134/20/136/74/49)
and test 103 (34/5/32/19/13). That is **93 grade-3 and 62 grade-4**. ICDR 0–4 coding; the files carry
expert grades with no per-grader or adjudication detail.
- **Six byte-identical duplicate pairs inside IDRiD, three with conflicting grades:**
  `IDRiD_021 = IDRiD_391` (1 vs 2), **`IDRiD_028 = IDRiD_327` (4 vs 3)**, and **test `IDRiD_064` =
  train `IDRiD_118` (3 vs 0)**. The 3 ↔ 4 boundary is noisy in IDRiD as well.
- **Stage-04 lesion-training images reappear in the grading set.** There are 3 exact matches and **62
  pixel matches** (MAD 0.09–0.70) among the 81 segmentation originals. The segmentation images are
  largely re-encoded copies of grading images.
  **Three of the matches are grading *test* images** (`IDRiD_088`, `IDRiD_089`, `IDRiD_091`; MAD
  0.66–0.70), all grade 2.
  - `IDRiD_External_Evaluation_Report.md` §5 states that 0 of 81 Stage-04 images appear in the
    grading test set, verified by content hash. That is correct for *exact* hashes but **misses
    these re-encoded copies**.
  - This matters for the pending RACAF IDRiD evaluation (leakage policy). It is not acted on here.
- APTOS vs IDRiD: 0 pixel matches. Exact comparison is impossible across formats, and re-cropped
  copies cannot be excluded.
- **Usable independent grade-3/4 population: 78 grade-3 / 53 grade-4** (24 removed).
- Projection: single-AUROC SE 0.0485 (APTOS 0.0539); MDE at 3 seeds 0.060 (APTOS 0.065).
- **Verdict: INSUFFICIENT.** Only marginally better than the 97-image APTOS subset.

**DDR.** Not present in the project or on Drive, and not downloaded.
- Published figures (secondary literature, **not verified against DDR files**): 13,673 images, six
  classes (0–4 plus ungradable = 6,266 / 630 / 4,477 / **236** / **913** / 1,151), ICDR scale, seven
  trained graders with specialist consultation.
- If those counts hold: single-AUROC SE 0.018 and MDE at 3 seeds 0.040, the floor that the
  mean ≥ 0.03 rule itself imposes.
- Overlap with APTOS or IDRiD cannot be checked.
- **Verdict: UNVERIFIED.** It is the only candidate that would be materially better (single SE about
  one-third of APTOS's).

**Planning projection (not measured power).** Bringing the paired validation SE to ≤ 0.015 needs
about **370 images per grade** in a balanced design. With three seeds the MDE floors near 0.04,
because the project rule requires mean ≥ 0.03.

**Assessment.**
1. IDRiD is not large enough, and it carries label conflicts of its own.
2. DDR would be large enough *if* the published counts hold. That is unverified.
3. Only DDR could provide a materially better 3-vs-4 evaluation than the current 97 images.
4. Before DDR is used:
   - obtain and verify its label files and counts;
   - exclude the ungradable class;
   - exact- and pixel-check it against APTOS and IDRiD;
   - confirm ICDR definitions and that the frozen Stage 02–04 pipeline processes its images;
   - use it once, never tuning on it.
   Acquiring DDR is a data decision for the user and is **not** taken here.

---

## 19. Final research direction — decision (design only, 2026-09-27)

Full record: `research/Grade3vs4_Architecture_Research/FINAL_DIRECTION_DECISION.md`. **Nothing is
implemented, pre-registered, trained or downloaded.**

**Why a decision now.** A research-direction audit was run under a 2–3-week deadline. It covered this
record §1–§18, the project spec, the Stage 9–11 status (templates only) and a fresh literature review.
It compared six candidates.

**Selected: an ICDR-structured two-route ordinal head, with PDR taken off the CORN chain.**
- *Hypothesis.* Part of PDR under-triage is caused by the output structure. In `corn.py`, task *k*
  trains only on grades ≥ *k*, so:
  - grade-4 labels supervise NPDR tasks 0–2;
  - grade 4 is reachable only through the chain.
  This conflicts with the disjunctive ICDR definition of PDR and with lesion burden falling from severe
  NPDR to PDR.
- *Motivating results.* §10, §13.2 (p_gt_2 inversion), §15 (21 PDR with low p_gt_2 / high p_cond_3),
  §16.1 (about 32/58 PDR decoded ≤ 2), §17 (the 3-vs-4 endpoint is unresolvable).
- *Design.* H2 (two-route head) vs H1 (CORN head refitted), both on the same frozen training E of the
  three NO_RACAF BEST backbones.
  - Both heads have 1,028 parameters and are convex, deterministic fits.
  - H0, the original head, is a reference only.
- *Primary endpoint.* Paired Δ AUROC of P(grade ≥ 3), grade 4 vs grades 0–2 (58 vs 635), APTOS
  validation, scored once. IDRiD is used once as confirmation, with the §18 leakage exclusions.
- *Rule.* SUPPORTIVE if mean Δ ≥ +0.03, 3/3 seeds, and all guardrails hold. The guardrails are grade-3
  recall, grade-3 referral AUROC, QWK, and false urgent calls.
- *Step 0 gate (read-only).* Baseline AUROC of p_gt_2 ≤ 0.95 and projected MDE ≤ 0.05.
- *Fallback.* The CORN chain-discordance flag, as a Stage-9 component with no training.

**Rejected.**
- PDR-sign concept channels from RFMiD / Retinal-Lesions: data not permitted; acquisition and time risk.
- Zero-shot FLAIR concepts: FLAIR was pretrained on APTOS and IDRiD, so it leaks.
- Grade-conditional conformal sets: already published on APTOS → IDRiD.
- IQA–severity confound: kept as a Stage-11 safety analysis.

**This does not reopen** any closed pathway. The §14/§16 objective manipulations, thresholding, and
3-vs-4 APTOS experiments stay closed.

**User decisions still required.**
- Revise `PROJECT_CODE.md`, which still names RACAF as the single approved innovation.
- Approve the pre-registration.

### 19.1 Re-audit after DDR was confirmed obtainable (design only, 2026-09-27)

Full record: `FINAL_DIRECTION_DECISION.md`, Addendum A. **Nothing is downloaded, trained or modified.**

**New fact.** The official OIA-DDR distribution is obtainable. It holds 13,673 images; grades
0/1/2/3/4/ungradable = 6,266 / 630 / 4,477 / 236 / 913 / 1,151. It is not yet downloaded.

**Outcome.**
- **C1 is kept, Stage-8 only.** Stages 5–7 are not changed. No experiment shows a representation-level
  PDR failure: §13.3 places the loss downstream of E in 3/3 seeds, and the Stage-7 aggregation loss
  appears in only 1/3 seeds.
- **Level comparison.** A Stage-8-only change gives the most science per day. Representation changes
  (Level B) have no demonstrated failure to target. End-to-end retraining (Level C) brings back the
  retraining and checkpoint-selection confounds of §14 and §16 at 2–3× the time.
- **Novelty narrowed.** The two-route head is an established sequential/hurdle ordinal variant (compare
  Tutz's sequential models and Ord2Seq). The contribution is:
  - the chain-gating diagnosis;
  - the clinically fixed split order;
  - a representation-controlled, capacity-matched test with external confirmation.
  It is claimed as an adaptation of an established technique plus a new mechanism-driven question,
  never as a new model family.
- **DDR's role.** External evaluation only, once, never used for fitting, thresholds, checkpoints or
  tuning. It serves as:
  - the confirmatory set (official test split, gradable only; about 275 grade-4 vs 3,400 grades 0–2,
    estimated and unverified; single-AUROC SE ≈ 0.014);
  - the host of powered replications of the p_gt_2 inversion, conditional > marginal, and the gating
    rate.
- **Before DDR is used:** SHA-256 and pixel-thumbnail checks against APTOS and IDRiD; ICDR coding
  confirmed; grade 5 excluded; Stage 2–4 smoke test; domain shift described.
- **Deadline rule.** If DDR has not passed its audit by the end of Day 2, the plan falls back
  automatically to IDRiD confirmation (scenario B), with the claim stated as weaker.
- **Primary endpoint unchanged:** paired Δ (H2 − H1) AUROC of P(grade ≥ 3), grade 4 vs 0–2.
- **Decision rule.**

  | Set | Verdict | Condition |
  |---|---|---|
  | APTOS | SUPPORTIVE | mean Δ ≥ +0.03, 3/3 seeds, guardrails hold |
  | DDR | CONFIRMED | mean Δ ≥ +0.02, 3/3 positive, 95 % CI excludes 0 |
  | IDRiD | CONFIRMED | ≥ +0.03, 3/3 positive |
  | IDRiD | DIRECTIONAL | 3/3 positive, below the bar |

  The Step 0 gate is unchanged.
- **Schedule.** About 4 days for the innovation; C1-C (end-to-end) stays future work.
- **Known caveats.**
  - The DDR ungradable class may hold PDR obscured by vitreous haemorrhage.
  - DDR's low grade-3 count points to different grading conventions.
  - Stage 04 domain shift affects both arms equally.

**This does not reopen** any closed pathway. The user decisions listed in §19 are still required,
now including adding DDR to `PROJECT_CODE.md` as an evaluation-only dataset.

---

## 20. C1 — ICDR two-route Stage-8 head vs CORN refit: results (pre-registered)

**Sources**
- Artifacts: `experiments/C1_ICDRTwoRouteHead/2026-09-27_16-37-56/` (`PREREGISTRATION.json`,
  `REPORT.md`, `results.json`, `per_seed_results.csv`, `h2_minus_h1.csv`,
  `per_sample_predictions.csv`, `E_seed_*.npz`, `heads_seed_*.npz`, `backbone_seed_*.json`).
- Code at commit `1f6644e`: `icdr_two_route_head.py`, `icdr_two_route_experiment.py`,
  `colab/notebooks/stage08_icdr_two_route_head.ipynb`.
- Report: `docs/experiments/C1_ICDR_Two_Route_Head_Report.md`.

**Design (as frozen).** Frozen NO_RACAF BEST backbones, seeds 42/123/2026; Stages 1–7 untouched.
- H0: the original head (saved outputs).
- H1: CORN Dense(256→4) refit on frozen training E.
- H2: PDR route Dense(256→1) plus an NPDR CORN route Dense(256→3) over grades 0–3, with grade 4
  excluded from NPDR supervision.
- Both refitted heads have 1,028 parameters and use one L2 logistic fitter (L2 = 1e-4, sqrt
  inverse-frequency class weights, L-BFGS from zero).
- Primary endpoint: per-seed Δ = AUROC(H2) − AUROC(H1), grade 4 vs 0–2, each head scored by its own
  P(grade ≥ 3). DDR and IDRiD were not used.

**Integrity (all passed).**
- Population: 2,921 train / 730 validation. This is the split minus the 11 pinned empty-FOV ids.
  Before the run, the population pin was corrected from the split count of 2,929; no data had been
  touched.
- Routes: NPDR route n = 2,685; PDR route n = 2,921.
- Backbones: weight-file SHA-256 and Stage 5–7 + CORN fingerprints unchanged in 3/3 seeds.
- Logit parity: vs saved ≤ 7.8e-3; eager vs graph ≤ 1.8e-2; head(E) vs graph ≤ 1.8e-2 (tolerance
  0.05); decoded-grade agreement with saved 1.0000.
- Fits: all 24 task fits converged (max |grad| ≤ 6e-8).
- Smoke checks passed.

**Primary result.**

| seed | H0 | H1 | H2 | Δ (H2 − H1) | 95 % CI | resamples > 0 |
|---|---|---|---|---|---|---|
| 42 | 0.9019 | 0.9110 | 0.9120 | +0.0011 | −0.006 to +0.008 | 59 % |
| 123 | 0.8907 | 0.9054 | 0.9031 | −0.0022 | −0.010 to +0.005 | 28 % |
| 2026 | 0.9059 | 0.9077 | 0.9104 | +0.0027 | −0.002 to +0.007 | 89 % |

Mean Δ **+0.0005** (SD 0.0025); paired bootstrap 95 % CI of the mean **−0.0044 to +0.0052**;
positive in 2/3 seeds.

**Guardrails** (3-seed mean H2 − H1; all hold):
- grade-3 recall −0.0940 (bound −0.10; seeds 0.000 / −0.128 / −0.154);
- grade-3-vs-(0–2) AUROC −0.0042;
- QWK −0.0100;
- false-urgent rate −0.0274.

**Pre-registered verdict: NOT_SUPPORTIVE** (mean Δ < +0.01). *The experiment does not support the
hypothesis that the output structure is the main limiting factor on the frozen representation.*

**Secondary endpoints** (3-seed means; H0 / H1 / H2).
- AUROC of P(4), grade 4 vs 0–2: 0.904 / 0.891 / 0.897.
- Grade-4 recall: 0.161 / 0.316 / 0.270.
- Grade-4 precision: 0.473 / 0.515 / 0.517.
- Images called grade 4: 19 / 36 / 30.
- Grade-3 recall: 0.436 / 0.479 / 0.385.
- False-urgent rate: 0.049 / 0.059 / 0.032.
- QWK: 0.821 / 0.846 / 0.836.
- MAE: 0.350 / 0.317 / 0.312.
- PDR decoded ≤ 2 (of 58): 31.7 / 25.3 / 33.0.
- Persistent-21 still ≤ 2: 21 / 18.7 / 19.7.

**Reading (interpretation; the verdict is unchanged by it).**
1. **This is a precise null, not an underpowered one.** The CI on the mean difference is about
   ±0.005, roughly an order of magnitude narrower than the 3-vs-4 endpoint's noise (§17). On this
   representation, the structural change moves the referral-boundary ranking of PDR by at most
   about half an AUROC point.
2. **The mechanism did not act as hypothesised.**
   - H2 moved **no** grade-4 image from ≤ 2 to ≥ 3 in any seed (0 / 0 / 0).
   - It moved 8 / 8 / 7 the other way. PDR decoded ≤ 2 rose from 25–26 (H1) to 33 (H2).
   - H2 is more conservative overall: fewer grade-4 calls, fewer false urgent calls, lower grade-3
     recall.
3. **The persistent-21 images are not recognised by the dedicated PDR route either.**
   - q, a 4-vs-all classifier fitted directly on E, scores 16 of the 21 below 0.30 in every seed.
   - Only `b3819a805dca` (q 0.83 / 0.55 / 0.27) and `6cfb7b44ef6f` (0.67 / 0.37 / 0.28) reach
     q > 0.5 in any seed. These are the two cases §15.2 flagged as having clear laser-scar-like spots.
   - Once the chain no longer gates the decision, these images are graded by their intraretinal
     lesions, which is exactly what ICDR ordering does when no PDR evidence is detected.
   - **The failure is therefore located in E, not in the output structure:** the representation does
     not carry recognisable PDR evidence for these eyes. This fits the §15.2 hypothesis (treated,
     haemorrhage-dominated and hazy PDR without a lesion channel) but does not confirm it.
4. **Score distributions** (validation, pooled over seeds): grade-3 P(≥3) median 0.67 vs grade-4
   0.57 under both heads, and q median 0.19 (grade 3) vs 0.29 (grade 4). The representation ranks
   severe NPDR as "more severe" than PDR on average, whatever head reads it.
5. **Refit effect (H1 − H0; descriptive, post hoc, not pre-registered as an endpoint).**
   - Re-estimating the CORN head on frozen training E, with nothing else changed, raised mean
     QWK +0.025, grade-4 recall +0.155 and grade-3-vs-(0–2) AUROC +0.037.
   - The primary endpoint moved +0.0085.
   - This is a decision-level and calibration change of the jointly trained head selected at the
     val_QWK-best epoch (compare §16.1). It is recorded, **not adopted**. Any use of it would need its
     own decision and must note that the head was fitted on training-image E.

**What this closes and what it leaves.**
- **CLOSED: the output-structure (chain-gating) hypothesis on the frozen representation.** Not to be
  rerun with another head, threshold, L2 or class weights.
- **Not tested and not launched:** whether chain supervision shaped E during end-to-end training
  (Level C). A head-only negative does not rule it out; it stays future work, per §19.1.
- **Standing:** PDR under-triage on this pipeline is **representational**. Stage 04 has no channel
  for the PDR-defining signs, and E does not otherwise encode them for these images. This is now the
  best-supported explanation, but it is not demonstrated directly.
- **Thesis contribution, as allowed by §19.1 item 25:** a controlled, pre-registered negative that
  relocates the failure from the ordinal output structure to the representation, with a precise
  (±0.005) bound.
- Next steps are a user decision. Nothing is launched.

### 20.1 Interpretation correction (2026-09-27; §20's numbers and verdict unchanged)

§20 overstated one conclusion. Where it says the failure is "located in E, not in the output
structure", that PDR under-triage "is **representational**", or that the negative "relocates the
failure … to the representation", read instead:

> **C1 showed that modifying the Stage 8 output structure did not recover the observed Grade-4
> under-triage from the frozen Stage 5–7 representation. This motivates investigation of the
> upstream representation pathway but does not establish that Stage 5–7 is definitively the
> bottleneck.**

**RACAF, correspondingly.** The RACAF null (six matched runs, C1/C3 controls) is evidence against the
specific tested RGB/Swin-to-E fusion intervention (Ĝ). It is not proof that RGB evidence cannot exist
upstream, or that representation fusion is not responsible.

**Follow-up.** A Stage 5–7 direction audit
(`research/Grade3vs4_Architecture_Research/STAGE5_7_DIRECTION_AUDIT.md`) found no Stage 5–7 innovation
currently justified for implementation. It leaves one conditional direction: pathway rebalancing
between the lesion-map and RGB pathways, an established method family and not claimed as novel. That
direction is gated by a read-only check, **G0**, whose conditions and verdict rule were fixed in that
file before it was run. Its result will be §21. No training is authorised by this entry.

---

## 21. G0 — read-only pathway-evidence gate for Candidate E (pre-specified): results

**Source.** A read-only Colab script given in chat; it wrote no artifact. It reused the C1 code (commit
`1f6644e`). The rules were fixed in
`research/Grade3vs4_Architecture_Research/STAGE5_7_DIRECTION_AUDIT.md` §6 before running.

**Method**
- From the three frozen NO_RACAF BEST backbones, extract for all 2,921 training and 730 validation
  images:
  - pooled Stage 5 features (mean over the 32×32 grid, 256-d);
  - pooled Stage 6 RGB features (mean over 64 tokens, 1,152-d);
  - E.
- Fit the identical C1 PDR-route fitter to each (all training images, target grade 4, `weighted_corn`
  class weights, training-only standardisation, L2 1e-4, L-BFGS).
- Score validation grade 4 vs grades 0–2 (58 vs 633); C1 bootstrap (2,000 resamples, seed 20260927).

**Integrity (all passed).**
- Weight-file SHA-256 and Stage 5–7 + CORN fingerprints unchanged in 3/3 seeds.
- Max Δlogit vs saved: 3.9e-3 / 7.8e-3 / 7.8e-3.
- Recomputed E matches the E stored by the C1 run to 3.9e-3 (mixed-precision rounding).

**G0-A: RGB weakness.**

| seed | stage 5 pooled | stage 6 pooled (RGB) | E | stage 5 − stage 6 (95 % CI) | G0-A |
|---|---|---|---|---|---|
| 42 | 0.8996 (0.860–0.934) | 0.7081 (0.647–0.764) | 0.8995 (0.862–0.932) | +0.1915 (+0.124 to +0.257) | holds |
| 123 | 0.8769 (0.837–0.915) | 0.7503 (0.689–0.807) | 0.8908 (0.851–0.927) | +0.1266 (+0.063 to +0.195) | holds |
| 2026 | 0.9022 (0.869–0.932) | 0.6959 (0.631–0.758) | 0.9001 (0.865–0.932) | +0.2063 (+0.142 to +0.274) | holds |

G0-A **holds in 3/3 seeds**; every CI on the difference excludes 0.

**G0-B: persistent-21 evidence absence.** The median q of the persistent-21 is above the grade-2 median
in **all 9** representation × seed cells, so G0-B is violated everywhere.

| seed | representation | persistent-21 median q | grade-2 median q | persistent-21 above the grade-2 median |
|---|---|---|---|---|
| 42 | stage 5 / stage 6 / E | 0.128 / 0.043 / 0.131 | 0.091 / 0.011 / 0.103 | 14 / 13 / 14 |
| 123 | stage 5 / stage 6 / E | 0.091 / 0.189 / 0.161 | 0.090 / 0.048 / 0.124 | 11 / 14 / 13 |
| 2026 | stage 5 / stage 6 / E | 0.125 / 0.007 / 0.121 | 0.087 / 0.002 / 0.111 | 12 / 12 / 11 |

**Pre-specified verdict: G0 FAIL — do not implement pathway rebalancing.**
- The rule was: FAIL if G0-A holds in ≤ 1/3 seeds, **or** any representation violates G0-B in 3/3 seeds.
- All three representations violate G0-B in 3/3 seeds.

**Reading (interpretation, labelled; the verdict is unchanged by it).**
1. **The RGB pathway is weak.** Stage 6 alone reaches 0.70–0.75 for grade 4 vs 0–2, against 0.88–0.90
   for pooled Stage 5.
2. **Stage 7 loses nothing on this endpoint.** Pooled Stage 5 and E are essentially equal (0.900 vs
   0.900, 0.877 vs 0.891, 0.902 vs 0.900). This differs from Phase 0's modest loss on grade 3 vs 4
   (§13).
3. **The "evidence absent everywhere" premise of Candidate E is false as operationalised.** Every
   representation, including E, ranks the persistent-21 slightly above a typical grade-2 image
   (11–14 of 21 above the grade-2 median).
4. **The margins are small.** For E: 0.121–0.161 vs 0.103–0.124. This is consistent with §20's
   finding that these images are not recognised as PDR in absolute terms (q < 0.30 for 16/21).
5. **Recorded as a design limitation of G0.** The grade-2 median is a lenient bar: it detects *any*
   elevation above moderate NPDR, not recognition as PDR. It was fixed in advance and is **not
   re-scored**.
6. **Stage-6 per-image scores (descriptive, post hoc, not acted on).** Stage-6 q for individual
   persistent images is highly seed-unstable (e.g. `1bf30c84bbad` 0.99 / 0.00 / 0.00; `84c663f39632`
   0.98 / 0.62 / 0.02; `65c958379680` 0.90 / 0.36 / 0.88). The RGB pathway occasionally scores some of
   these images high, but not reproducibly, and its overall discrimination is weak. This is
   hypothesis-generating only; single-image, seed-unstable scores cannot support a design decision.

**What this closes.**
- **Candidate E (pathway rebalancing / ModDrop) is CLOSED**, per the rule fixed in the audit ("If G0
  fails: Candidate E is closed, and the project moves on to the remaining stages").
- With A–D already rejected in that audit, **no Stage 5–7 innovation direction remains open for this
  project.**
- G0 does not show that Stage 5–7 is, or is not, the bottleneck. It shows only that the specific
  pathway-dominance hypothesis did not pass its gate.

**Standing after §20–§21.**
- C1: changing the Stage-8 output structure did not recover the grade-4 under-triage from the frozen
  representation.
- The RGB pathway is weak relative to the lesion pathway.
- Stage 7 does not lose grade-4-vs-(0–2) information relative to pooled Stage 5.
- The persistent PDR failures are only marginally elevated above moderate NPDR in every representation.
- **Next step:** the user's decision. DDR stays reserved for later external evaluation. Nothing is
  launched.

---

## 22. P/PL — pretrained ConvNeXt-Tiny with vs without frozen Stage 3/4 priors: results (pre-registered)

**Sources**
- Artifacts: `experiments/PL_ConvNeXtPriors/2026-09-28_05-22-44/` (`PREREGISTRATION.json`, `REPORT.md`,
  `results.json`, `per_run_results.csv`, six `preflight_*.json` from the resumed sessions, per-run
  `evaluation/` and `history/`).
- Verified via rclone against the pasted report.
- Code at commit `33b1ce4`: `pl_convnext.py`, `colab/notebooks/pl_convnext_priors_experiment.ipynb`.
- Report: `docs/experiments/PL_ConvNeXt_Priors_Report.md`.

**Design (protocol-locked).**
- **P:** ImageNet-1k ConvNeXt-Tiny (pinned `convnext_tiny_notop.h5`, SHA-256 `d547c096…`) on Stage-02
  RGB at 512 px.
- **PL:** identical, with 8 input channels (RGB + vessel + MA/HE/EX/SE). The first-layer kernel is
  [pretrained RGB | zeros].
- **Shared:** CORN Dense(768→4) with a seeded identical head; weighted CORN; AdamW 1e-4 / 0.05 with no
  decay on any 1-D parameter; the frozen six-run schedule; BEST by val_QWK. Seeds 42/123/2026, each run
  once.
- **Primary:** Δ AUROC of P(grade ≥ 3), grade 4 vs 0–2, for A (PL − P) and B (PL − frozen NO_RACAF).

**Integrity.**
- All 13 pre-flight assertions passed, including: P = reference (max 3.6e-6); initial PL = P (0.0);
  channel order and ranges; split 2,921 / 730; Stage 3/4 fingerprint unchanged before and after;
  weight-decay exemption; mixed precision with loss scaling.
- 6/6 runs completed, all by early stopping (16–31 epochs).
- "No protocol deviations."

**Per-seed results (BEST checkpoint)**

| run | AUROC P(≥3) 4 vs 0–2 | sens @ 95 % spec | QWK | MAE | grade-3 recall | false-urgent | BEST epoch (0-based) |
|---|---|---|---|---|---|---|---|
| P-42 / PL-42 | 0.9588 / 0.9617 | 0.741 / 0.810 | 0.9184 / 0.9215 | 0.178 / 0.185 | 0.615 / 0.462 | 0.038 / 0.032 | 18 / 12 |
| P-123 / PL-123 | 0.9489 / 0.9585 | 0.776 / 0.776 | 0.9148 / 0.9138 | 0.201 / 0.189 | 0.513 / **0.103** | 0.032 / 0.006 | 3 / 6 |
| P-2026 / PL-2026 | 0.9410 / 0.9515 | 0.741 / 0.793 | 0.9176 / 0.9177 | 0.190 / 0.181 | 0.513 / 0.410 | 0.036 / 0.025 | 6 / 17 |
| frozen NO_RACAF 42 / 123 / 2026 | 0.9019 / 0.8907 / 0.9059 | 0.483 / 0.466 / 0.483 | 0.830 / 0.807 / 0.827 | 0.311 / 0.344 / 0.396 | 0.385 / 0.410 / 0.513 | 0.021 / 0.028 / 0.100 | — |

**Means (3 seeds)**

| arm | AUROC | sens @ 95 % spec | QWK | MAE | grade-3 recall | false-urgent |
|---|---|---|---|---|---|---|
| P | 0.9496 ± 0.0089 | 0.753 | 0.917 | 0.190 | 0.547 | 0.035 |
| PL | 0.9572 ± 0.0052 | 0.793 | 0.918 | 0.185 | 0.325 | 0.021 |
| frozen reference | 0.8995 | 0.477 | 0.821 | 0.350 | 0.436 | 0.050 |

**A. PL − P (primary, controlled)**
- Δ AUROC: +0.0029 / +0.0095 / +0.0105.
- Mean **+0.0076** (SD 0.0041); seed-mean bootstrap 95 % CI −0.0003 to +0.0164; positive in 3/3 seeds.
- Guardrails: QWK +0.0007 (holds); **grade-3 recall −0.222 (fails)**; false-urgent rate −0.014 (holds).
- **Pre-registered verdict A: NOT_SUPPORTIVE** (mean Δ < +0.01).

**B. PL − frozen NO_RACAF (system-level only)**
- Δ AUROC: +0.0598 / +0.0677 / +0.0455.
- Mean **+0.0577**; CI +0.033 to +0.086; 3/3 seeds.
- Guardrails: QWK +0.0965 (holds); **grade-3 recall −0.111 (fails)**; false-urgent rate −0.028 (holds).
- **Pre-registered verdict B: INCONCLUSIVE.** The primary was met, but a guardrail failed.

**Reading (interpretation, labelled; the verdicts are unchanged by it).**
1. **The frozen Stage 3/4 priors add at most a small amount to a pretrained CNN.** The gain is
   consistent in sign (3/3), sits below the pre-registered +0.01 floor, and its CI touches zero.
   - At the default decode, PL calls fewer images grade 3: grade-3 recall falls, driven by seed 123 (0.103).
   - PL's grade 3-vs-(0–2) ranking is also slightly worse (−0.0175).
   - PL decodes slightly more grade-4 images as ≤ 2 (+2.3).
   - The priors do not improve the severe-NPDR boundary.
2. **Largest effect, descriptive and not pre-registered: P vs the frozen reference.** RGB-only pretrained
   ConvNeXt-Tiny exceeds the current Stage 5–8 pipeline on every headline metric:
   - AUROC +0.050, QWK +0.096, MAE −0.160, sensitivity at 95 % specificity +0.276;
   - grade-3 recall +0.111 and false-urgent rate −0.014.

   This is the strong-baseline control the Stage 5–8 re-audit found missing. It is **not** a
   pre-registered contrast, and **it does not isolate pretraining**: backbone, pretraining and input
   priors all differ at once, on a validation set used many times before.
3. **The fitted models are heavily over-fitted.** Training QWK reaches about 0.99 against about 0.91 on
   validation.
4. **BEST checkpoints are sometimes very early** (P-123 at epoch index 3; P-2026 and PL-123 at 6).
   - LAST AUROC is lower than BEST in 6/6 runs.
   - PL − P at LAST is −0.0103 / +0.0150 / +0.0145 (mean +0.0064, descriptive).
   - The same checkpoint-selection sensitivity was seen in §16.1.

**What this closes and what it leaves.**
- **CLOSED at the pre-registered bar:** supplying the frozen vessel/lesion probability maps as extra input
  channels to a pretrained ConvNeXt-Tiny (early fusion, zero-initialised) as a way to improve
  grade-4-vs-(0–2) discrimination.
- **Not tested:** other ways of using Stage 3/4 outputs with a pretrained encoder; external data (IDRiD,
  DDR). DDR is still reserved.
- **Not claimable:**
  - that P/PL is novel;
  - that B isolates pretraining;
  - that random initialisation caused earlier failures;
  - anything about grade 3 vs 4.
- **Next step:** the user's decision. Nothing is launched.

## 23. S1 — read-only representation / spatial-pooling audit of the trained P models: results (pre-specified)

**Run (2026-09-29, Colab T4; repo 33b1ce4).** The run was read-only: nothing was trained, and no
checkpoint, cache, split or preprocessing was modified. It covered the P arm of §22
(`PL_ConvNeXtPriors/2026-09-28_05-22-44`), seeds 42/123/2026, BEST and LAST. The first attempt was
stopped before producing results: eager calls re-traced Keras's XLA grouped-convolution wrapper on
every batch. The rerun used a compiled feature model; its outputs match the eager path within 5e-6 in
an offline check. The analysis and the decision rule were unchanged between attempts.

**Pre-specified protocol (fixed before any real result).**
- **Representations:** the average-pooled vector of each stage (stage0–3: 96/192/384/768),
  `pooled_final` (= LN(GAP(stage3)), the CORN head's input), and fixed, non-trained reductions of the
  16×16×768 stage-3 map:
  - 4×4 grid (12,288; primary spatial representation);
  - 2×2 grid, spatial max, spatial std, mean+max (secondary).
- **Probe:** Phase-0 pipeline (StandardScaler → PCA-16 primary / PCA-64 secondary → logistic regression
  C=1), fitted on train (2,921) and scored once on validation (730).
- **Contrasts:** A 4 vs 0–2 (58 vs 633); B 3 vs 0–2 (39 vs 633); C 3 vs 4 (descriptive only).
- **CIs:** C1 stratified bootstrap (2,000 resamples, seed 20260927), paired for differences.
- **Decision (BEST, PCA-16, contrasts A and B).** "Supported" = 3-seed mean Δ vs `pooled_final`
  ≥ +0.03, AND > 0 in 3/3 seeds, AND paired CI > 0 in ≥ 2/3 seeds.
  - A: the 4×4 grid is supported.
  - B: a stage 0–2 average is supported.
  - C: every Δ has mean < +0.01.
  - D: anything else.
  - The paired-CI condition was added after an offline noise test and before any real data was seen.

**Integrity: all pass.**
- Split sha `bc80fd45…`; population 2,921 / 730.
- Stage 3/4 fingerprint unchanged; ImageNet weights sha matches the pre-registered one.
- 6/6 checkpoints: weights sha unchanged; 27,823,204 parameters; 3 input channels.
- Max |Δlogit| vs saved logits 2.4e-4 to 1.6e-2; head(`pooled_final`) vs logits 4.9e-3 to 1.5e-2.
  Both are within the 0.05 tolerance, and the size is consistent with mixed-precision float16 rounding.

**Results (BEST, mean over seeds, PCA-16 AUROC).**

| representation | A: 4 vs 0–2 | B: 3 vs 0–2 | C: 3 vs 4 (descr.) |
|---|---|---|---|
| stage0_gap | 0.8675 | 0.8967 | 0.7778 |
| stage1_gap | 0.8884 | 0.9200 | 0.7943 |
| stage2_gap | 0.9422 | 0.9497 | 0.8168 |
| stage3_gap | 0.9458 | 0.9534 | 0.8594 |
| **pooled_final** | **0.9557** | **0.9600** | 0.8488 |
| stage3_grid4 | 0.9531 | 0.9518 | 0.8186 |
| stage3_grid2 | 0.9511 | 0.9509 | 0.8292 |
| stage3_max | 0.9440 | 0.9321 | 0.8126 |
| stage3_std | 0.9500 | 0.9387 | 0.8453 |
| stage3_meanmax | 0.9462 | 0.9561 | 0.8594 |
| trained CORN head | 0.9496 | 0.9584 | 0.8313 |

**Decision table (A and B, Δ vs `pooled_final`).**
- All 16 rows have a **negative** mean Δ, and no row has a paired CI above 0 in any seed.
- 4×4 grid: A −0.0027 (1/3 positive), B −0.0082 (0/3).
- Stage averages: stage0 A −0.088 / B −0.063; stage1 −0.067 / −0.040; stage2 −0.014 / −0.010.
- Flags: spatial False, stage False, secondary False, all_negligible True.
- **Verdict: C. NO CLEAR BOTTLENECK.**

**Interpretation.**
1. **Global average pooling is not discarding linearly recoverable grade information.**
   - `pooled_final` is the best linear read-out of the P backbone for both pre-specified contrasts.
   - Keeping stage-3 spatial layout (4×4 or 2×2 grid) or focal extremes (max, std) adds nothing.
   - The PCA-64 probes agree: grid4 0.940 / 0.934 vs pooled 0.949 / 0.952 for A / B.
2. **No intermediate-resolution bottleneck.**
   - Separability rises steadily with depth: stage0 < stage1 < stage2 < stage3 ≤ `pooled_final` in all
     three contrasts.
   - The high-resolution early stages (128 × 128 and 64 × 64 maps) are clearly worse, even with PCA-64:
     A 0.902 / 0.921 vs 0.949.
3. **The trained head uses what is available.**
   - Head vs `pooled_final` probe: A −0.006, B −0.002.
   - This is consistent with C1 (§20): output structure is not the lever.
   - Contrast C, descriptive only: the head (`p_cond_3`) is 0.8313 vs the probe's 0.8488, with the widest
     gap in seed 123 (0.784 vs 0.842).
   - CIs on n = 97 are about ±0.08, and no paired head-vs-probe CI was computed, so this is not evidence.
4. **BEST vs LAST.**
   - Late-stage features degrade with continued training: `pooled_final` A −0.023, B −0.015; stage-3
     reductions −0.02 to −0.04.
   - Stages 0–2 are unchanged (within ±0.01).
   - So P's over-fitting (§22 item 3) affects the late backbone as well as the head.
5. **Contrast C (3 vs 4), descriptive only.** Stage-3 average and mean+max reach 0.859, vs 0.849 for
   `pooled_final`. Only seed 42's stage3_gap has a paired CI above 0 (+0.024, +0.002 to +0.048). The grid
   reductions are lower (0.82–0.83). There is no spatial signal for 3 vs 4.
6. **Linear saturation.** Every stage-3-level read-out sits at about 0.95–0.96 for A and B, and at about
   0.82–0.86 for C. The limit is in what the backbone has learned, and in the data and labels. It is not
   in how the final map is read out.

**Limits.**
- These are linear probes on fixed reductions; a trained non-linear read-out was not tested, by design.
- The grids are in the image frame, not an anatomy frame (optic disc / fovea).
- The same validation set was used to select BEST.
- The result does not rule out that anatomy-anchored or lesion-level information exists; it finds no
  evidence of it.

**What this closes and what it leaves.**
- **Not supported by evidence:** a new read-out architecture on top of the P backbone — spatial / grid /
  quadrant pooling, max or extreme pooling, multi-scale (early-stage) read-out, lesion-query or MIL
  pooling motivated by "GAP loses information". The audit found no linearly recoverable information for
  such an architecture to recover.
- **Not tested:** anatomy-registered spatial statistics; non-linear read-outs; data, label or supervision
  levers. DDR is still reserved.
- **Next step:** the user's decision. Nothing is launched.

## 24. S2 — why grade 3 vs 4 is weaker than 3/4 vs 0–2 in the trained P models (read-only audit)

**Run (2026-09-29/30).** Nothing was trained; no checkpoint, cache, split or preprocessing was modified.
- **Colab T4 (the only GPU step):** frozen features were extracted for 9 checkpoints (P seeds
  42/123/2026 × EARLY/BEST/LAST). They were written once to the new folder
  `experiments/P_FrozenFeatures/2026-09-29_17-57-51/` (sha256-verified; the P/PL experiment folder
  was not touched).
  - EARLY is the inactive BEST slot: epochs 14 / 2 / 3.
  - BEST is at epochs 19 / 4 / 7; LAST at epochs 31 / 16 / 19.
- **Integrity: all pass.**
  - Split `bc80fd45…`; population 2,921 / 730; Stage 3/4 fingerprint unchanged; ImageNet weights sha
    matches the pre-registered one.
  - 9/9 checkpoints: weights sha unchanged; 27,823,204 parameters.
  - BEST/LAST logits match the saved logits within 1.6e-2; head(`pooled_final`) reproduces the logits
    within 1.5e-2.
- **Analysis:** done on the laptop from sha-verified rclone copies (`s2_analysis.py`,
  `s2_local_head.py`).
- **Consistency check:** the BEST 3-vs-4 probe reproduces S1 (0.8408 / 0.8422 / 0.8630 vs
  0.8413 / 0.8422 / 0.8630).

**Pre-specified rule (fixed before extraction).**
- The inputs to [B] (the S1 probe and the BEST head outputs) had already been seen and were disclosed.
- Primary comparison: BEST, `pooled_final`, mean over seeds.
- **Inputs:**
  - [B]: probe(3v4) − head `p_cond_3`(3v4) ≥ +0.03, and > 0 in 3/3 seeds.
  - [U]: balanced-154 probe(3v4) ≤ min(0v1, 1v2, 2v3) − 0.05.
  - [G]: R(3v4) is the smallest adjacent-pair R, and ≤ 0.5 × R(2v3).
  - [L]: learning curve 100 % − 50 % ≥ +0.02, and > 0 in 3/3 seeds.
- **Verdict:** B if [B]; A if not B ∧ U ∧ G ∧ ¬L; C if not B ∧ (¬U ∨ L); D otherwise.

**1. Geometry (validation, standardised full-dimensional space; BEST mean; `pooled_final`).**
- **Between/within-class ratio R:** 0v1 1.745, 1v2 0.732, 2v3 0.484, **3v4 0.306**; 2v4 0.633
  (PCA-32: 2.05 / 0.90 / 0.58 / 0.36).
  - Per seed, R(3v4) vs R(2v3): 0.353 vs 0.474; 0.261 vs 0.550; 0.304 vs 0.427.
- **kNN 3-vs-4 AUROC:** 0.823 (0v1 0.979, 1v2 0.917, 2v3 0.861).
- **Neighbour grades (10 nearest training images) of validation grades 3 and 4:**
  - true 3 → grades 0–4: 0.00 / 0.01 / 0.37 / 0.41 / 0.20;
  - true 4 → grades 0–4: 0.00 / 0.07 / 0.27 / 0.12 / 0.54.
  - Grade-4 images sit nearer to grade 2 than to grade 3.
- **Within-grade dispersion rises with severity:** W = 449 / 373 / 392 / 477 / 571 for grades 0–4.

**2. Ordinal structure (axis = PC1 of the 5 training class centroids).**
- `pooled_final`: PC1 holds 0.64 of the centroid variance. Validation class means along the axis:
  −14.8 / 2.3 / 15.3 / **23.4 / 20.7** — grade 4 sits *below* grade 3.
  - Monotonic in 0/3 seeds.
  - Adjacent AUROC along the axis: 0.967 / 0.909 / 0.839 / **0.378**.
  - Cosine of the 3→4 step with the axis: −0.12; with the 2→3 step: −0.15.
- **Bias-corrected distances between validation centroids:** d(4,0) = 37.0 < d(3,0) = 39.1;
  d(4,3) = 12.7; d(4,2) = 17.3.
- **stage2_gap:** also non-monotonic (4 below 3); 3→4 step to axis −0.50; d(4,2) 6.6 < d(4,3) 7.1.
- **stage3_gap (pre-LayerNorm):** monotonic in 3/3 seeds, but the 3-vs-4 AUROC along the axis is only
  0.539.
- **Reading:** the representation orders grades 0→3 along one severity direction. Grade 4 does *not*
  lie further along it; it departs in a different direction.

**3. Pairwise probes (PCA-16, BEST, `pooled_final`; mean, with per-seed 95 % CIs in the output).**

| pair | full training | balanced-154 | training 5-fold CV (in-sample features) |
|---|---|---|---|
| 0v1 | 0.9917 | 0.9898 | 0.9986 |
| 1v2 | 0.9348 | 0.9303 | 0.9704 |
| 2v3 | 0.8853 | 0.8754 | 0.9672 |
| 3v4 | 0.8487 | 0.8485 | 0.9516 |
| 012v3 | 0.9597 | — | — |
| 012v4 | 0.9557 | — | — |
| 2v4 (extra) | 0.8897 | 0.8829 | — |

- **Adjacent-grade difficulty rises smoothly with severity, and is unchanged at matched training
  size.** 3 vs 4 is the hardest pair, but it is the end of a gradient, not a unique break: 2v3 − 3v4
  is +0.027 at matched size.
- **2 vs 4, two grades apart, is as hard as 2 vs 3.**
- **The 0.95–0.96 of 012v3 and 012v4 is inflated by the easy grades 0/1.**
- **The 3-vs-4 learning curve (25 / 50 / 75 / 100 %) is flat:** mean +0.007 from 50 % to 100 %.
- **The generalisation gap grows with severity:** training CV − validation is 0.007 / 0.036 / 0.082 /
  0.103 for 0v1 / 1v2 / 2v3 / 3v4.

**4. CORN outputs (BEST, saved validation CSVs).**
- **Conditional task AUROC:** 0.998 / 0.950 / 0.870 / **0.831**.
- **Expected-grade score E = Σ p_gt_k:**
  - adjacent AUROC 0.992 / 0.919 / 0.846 / **0.670**; Cohen's d 4.6 / 1.9 / 1.4 / 0.47;
  - median E per grade 0.00 / 1.17 / 2.11 / 2.88 / 3.50;
  - mean E − grade for grade 4: −0.83.
- **Calibration in aggregate:**
  - Σ P(grade 4) = 57.7 vs n = 58, so grade 4 is not systematically underestimated in total but is
    per image (mean own-class probability 0.54);
  - grade 3 is over-predicted in total (48.4 vs 39).
- **Chain gating:**
  - 13–16 of the 58 grade-4 images have p_gt_2 < 0.5; 8–14 of those have p_cond_3 > 0.5.
  - Among images with p_gt_2 > 0.5, p_cond_3 3v4 is 0.90 / 0.79 / 0.88.
- **Errors (pooled over seeds):**
  - grade 4: 44/74 to ≤ 2, 30/74 to 3;
  - grade 3: 38/53 to ≤ 2, 15/53 to 4;
  - 3↔4 swaps are 35 % of grade-3/4 errors.

**5. Dynamics.**
- **Histories:** they hold only QWK, loss and learning rate per epoch.
  - Validation QWK plateaus by epochs 3–5 at about 0.90–0.91; training QWK reaches 0.99.
  - Minimum validation loss is at epochs 5 / 8 / 6, after which it rises to 0.38–0.45.
- **EARLY → BEST → LAST, 3-vs-4 probe on `pooled_final`:** 0.863 / 0.836 / 0.838 → 0.841 / 0.842 /
  0.863 → 0.834 / 0.831 / 0.835.
  - Already about 0.84 at epoch 2–3 (seeds 123 and 2026). It does not improve and declines slightly
    at LAST.
- **Head `p_cond_3` 3v4:**
  - training AUROC: 0.84–0.99 → 0.91–0.999 → 0.999 (memorised);
  - validation: noisy (0.77–0.90), no trend;
  - E 3v4: 0.58–0.69 throughout.
- **Recall:** grade-3 recall falls at LAST in 3/3 seeds (0.615→0.385, 0.513→0.410, 0.513→0.333).
- **Classification: C — largely unchanged.** The 3-vs-4 information is present from the start, the
  training fit becomes perfect, and validation does not follow.

**6. Documented label/data evidence** (§§14–16, duplicate audit, label-noise JSON; nothing new was
computed).
- **Sample size:** 154 / 236 training and 39 / 58 validation images for grades 3 / 4 (CI about ±0.08).
- **Cross-split duplicates:** 8 of 58 validation grade-4 images (14 %) have a training twin, vs 5.6 %
  overall.
- **Label disagreement on identical images:**
  - train/validation pairs touching 3/4: 6/12 (vs 0.133 elsewhere, p = 0.020), mostly 4 vs ≤ 2;
  - pure 3↔4 pairs: 1/7;
  - dataset-wide: 32 of 134 pairs; 9 grade-4 images have a twin graded ≤ 2; 13 grade-3 images have a
    differently graded twin (not broken down by partner grade in the metadata).
- **§15 ceiling (model-dependent):** point estimate 0.870 (0.923 counting 3↔4 flips only).
- **Persistent PDR failures:** 21 images decoded ≤ 2 by every NO_RACAF seed; heterogeneous; not
  established as label errors.

**Pre-specified decision.**
- [B] per-seed gap −0.006 / +0.058 / 0.000, mean +0.017 → False.
- [U] balanced-154 3v4 0.8485 vs 2v3 0.8754 → not uniquely hard.
- [G] R(3v4) 0.306 is the lowest, but > 0.5 × R(2v3) = 0.242 → False.
- [L] +0.007 → False.
- **Verdict: C. DATA/LABEL LIMITATION.**

**Caveats on the verdict (post hoc; they do not change it).**
1. The C branch fired because 3 vs 4 is **not uniquely hard** (¬U), not because of positive
   sample-size evidence. For the linear read-out, training-set size is actually **ruled out** as the
   limiter: the learning curve is flat and the balanced-154 result is unchanged.
   The positive data/label evidence is:
   - the severity-graded generalisation gap (training CV 0.95 vs validation 0.85; the head fits
     training 3/4 at 0.999);
   - the documented 3/4 label disagreement;
   - the validation AUROC (0.85) sitting at the §15 point-estimate ceiling (0.87);
   - the ±0.08 CIs.

   Backbone-level data limitation (would more or cleaner 3/4 images change the *features*?) cannot be
   tested without training.
2. **The representation is not ordinal at 3→4.** Grade 4 lies off, or even behind, grade 3 on the
   principal severity direction. CORN's cumulative decoding therefore compresses 3/4 at the decision
   level: E 3v4 is 0.67, against 0.83–0.85 for the conditional readouts.
   This is an ordinal-structure/objective mechanism, but [B] tested only the conditional task
   (`p_cond_3` ≈ probe), where the head uses what is there.

   A stricter reading of all the evidence is **D (mixed: data/label + non-ordinal grade-4
   geometry)**. The pre-specified verdict stays C.

**Distinctions.**
- **Information absent from the representation:** no. About 0.85 is linearly recoverable, from
  epoch 2.
- **Present but overlapping:** yes. R is lowest for 3v4; grade-3 neighbours are 37 % grade 2 and 20 %
  grade 4.
- **Present but not used by the head:** no for the conditional 3-vs-4 task (head ≈ probe). At the
  decoded level, CORN's chain loses it (0.67; post hoc).
- **Limited by labels/data:** consistent with it (gap, label disagreement, ceiling, n). The size of
  the probe-level training set is not the limiter.
- **Affected by overfitting:**
  - late-stage features: 3v4 −0.007 / −0.011 / −0.028 BEST→LAST;
  - grade-3 recall collapses at LAST;
  - the head memorises the training 3/4 labels.

**What this closes and what it leaves.**
- **Not justified:** a new architecture to recover 3/4 information. The information is not absent,
  and the pre-specified verdict is C.
- **Not claimable:**
  - that CORN is defective;
  - that 3/4 is clinically ambiguous;
  - that any label is wrong;
  - that more APTOS 3/4 data would help (backbone-level effect untested).
- **What a future experiment would need to show (pre-registered):**
  - (a) on independent, reliably labelled grade-3/4 data, whether the train–validation gap is a data
    or a label effect — external data remains reserved (DDR) / single-use (IDRiD);
  - (b) for any ordinal/objective change, that decision-level 3v4 ranking rises toward the conditional
    ~0.85 without lowering 4-vs-(0–2). C1 (§20) found no such gain with a two-route head on the
    NO_RACAF E; it has not been tested on P.
- **Next step:** the user's decision. Nothing is launched.

## 25. S3 — is the conditional → expected-grade 3-vs-4 drop caused by CORN's cumulative decoding? (read-only)

**Run (2026-09-30).** CPU only; nothing trained; no threshold changed.
- **Inputs:** the saved frozen features (`P_FrozenFeatures/2026-09-29_17-57-51`, sha-verified), the
  saved per-sample CSVs, and the label-noise audit JSON.
- **Script:** `s3_decoder_audit.py`. Its decision rule was fixed in the script header before it ran.
  The p_cond_3, E and probe AUROCs from §24 were already known and were disclosed.

**Pre-specified rule.**
- **[D1]** p_cond_3 − E ≥ 0.05 (3v4 AUROC) in 3/3 BEST seeds.
- **[D3]** A frozen linear probe on `pooled_final` for {0,1,2} vs {3,4} beats CORN's P(Y≥3) on 3v4 by
  ≥ +0.05 in 3/3 seeds, and is not worse at 012 vs 34 (margin 0.01).
- **Verdict:** B if D1 ∧ D3; C if ¬D1; D if D1 ∧ ¬D3.
- A is not an outcome of this rule.
- The objective-experiment gate opens only under B.

**1. Reconstruction (BEST and LAST, all seeds).**
- The model's validation logits reproduce the saved CSVs: p_cum / p_cond / class prob within 1.0e-3;
  decoded grade identical in 730/730.
- There are 0 p_cum ordering violations.
- E = Σ P(Y≥k) = Σ k·P(Y=k) (to 1e-15). Decoded grade = #(P(Y≥k) > 0.5); argmax agrees in 711–729/730.
- Identity: P(Y=4) / (P(Y=3) + P(Y=4)) = p_cond_3 exactly (1e-16). The conditional signal *is* CORN's
  own class posterior for "4 rather than 3".
- The 3v4 E AUROC reproduces §24: 0.6795 / 0.6658 / 0.6645.

**2. Threshold by threshold, grade 3 vs 4 (BEST).**

| threshold | seed 42 | seed 123 | seed 2026 |
|---|---|---|---|
| P(Y≥1) | 0.42 | 0.38 | 0.35 |
| P(Y≥2) | 0.32 | 0.58 | 0.44 |
| P(Y≥3) | 0.545 | 0.637 | 0.575 |
| P(Y≥4) | 0.791 | 0.746 | 0.716 |
| p_cond_3 | 0.847 | 0.784 | 0.863 |

- P(Y≥1) and P(Y≥2) are about 1.0 for both grades and carry ≤ 6 % of Var(E) among true 3/4.
- P(Y≥3) carries 22–23 % of Var(E), with a 3v4 AUROC of only 0.55–0.64. The grade-3 and grade-4 IQRs
  overlap heavily (seed 42: 0.50–1.00 vs 0.24–1.00).
- **Hold-out decomposition:** setting P(Y≥1) and P(Y≥2) to 1 leaves E unchanged (0.690 / 0.672 /
  0.672). Also setting P(Y≥3) to 1 restores the marginal P(Y≥4) (0.791 / 0.746 / 0.716).
  P(Y≥4) = P(Y≥3) · p_cond_3.
- **The entire p_cond_3 → P(Y≥4) → E drop (about 0.83 → 0.75 → 0.67) is carried by the P(Y≥3)
  threshold.**
- The pattern reproduces at EARLY and LAST: p_cond_3 − E is 0.14–0.22 in all 9 checkpoints.

**3. Three views of the same prediction (BEST):**

| view | seed 42 | seed 123 | seed 2026 |
|---|---|---|---|
| A: p_cond_3 | 0.847 | 0.784 | 0.863 |
| B: E | 0.680 | 0.666 | 0.665 |
| C: decoded grade | 0.698 | 0.655 | 0.677 |

**4. Gating (BEST, thresholds unchanged).**
- **Grade 4 with P(Y≥3) ≤ 0.5:** 15 / 13 / 16 of 58.
  - Of those, p_cond_3 > 0.5: 12 / 8 / 14.
  - Grade 3 failing P(Y≥3) > 0.5: 10 / 15 / 13 of 39.
- **Grade 3 promoted to 4:** 5 / 4 / 6 of 39.
- **Grade-4 errors:** to grade 3 4 / 21 / 5; to ≤ 2 15 / 13 / 16.
- **The conditional signal carries no information below the ≥3 gate.** Among all images with
  P(Y≥3) ≤ 0.5, p_cond_3 > 0.5 occurs for:
  - 71–91 % of true grade 0;
  - 84–97 % of true grade 1;
  - 55–85 % of true grade 2.

  In that region, p_cond_3 separates true 4 from true ≤ 2 at 0.47 / 0.38 / 0.40 (below chance), while
  P(Y≥3) does so at 0.90 / 0.81 / 0.83 and the frozen 012v4 probe at 0.90 / 0.87 / 0.86.
  Task 3 is trained only on grade ≥ 3 images, so p_cond_3 below the gate is extrapolation. "Would be
  grade 4 under the conditional signal" is **not a valid counterfactual** for gated images.
- **No systematic suppression of grade 4 in aggregate.**
  - ΣP(Y=4) is 55.4 / 51.8 / 65.9 vs 58.
  - Decoded grade-4 counts are 53 / 29 / 51. Seed 123 (BEST at epoch 4) sends 21 grade-4 images to
    grade 3 (ΣP(Y=3) 64.2 vs 39); this is seed-specific.
  - Decoded ≥ 3 counts are 96 / 89 / 91 vs 97.
- **The same images are gated across architectures.** 20 grade-4 images are gated in at least one
  seed and 9 in all three. 7 of those 9 are in the NO_RACAF persistent-21 set (a different, randomly
  initialised architecture). 1 of the 9 has a documented duplicate graded ≤ 2.
- Gated grade-4 counts: EARLY 17 / 24 / 12; LAST 21 / 21 / 21.

**5. Representation vs decoder (BEST, AUROC).**

| seed | ≥3-cut probe 3v4 | CORN P(Y≥3) 3v4 | ≥3 probe 012v34 | CORN 012v34 | ≥4-cut probe 3v4 | CORN P(Y≥4) 3v4 |
|---|---|---|---|---|---|---|
| 42 | 0.538 | 0.545 | 0.953 | 0.961 | 0.752 | 0.791 |
| 123 | 0.616 | 0.637 | 0.956 | 0.950 | 0.759 | 0.746 |
| 2026 | 0.596 | 0.575 | 0.951 | 0.948 | 0.745 | 0.716 |

- The frozen representation's own ≥3 read-out ranks grade 4 against grade 3 **as poorly as CORN's
  P(Y≥3)** (mean difference −0.002), with equal 012-vs-34 performance.
- The same holds at EARLY and LAST (differences −0.06 to +0.04).

**Pre-specified decision.**
- [D1] p_cond_3 − E = +0.168 / +0.118 / +0.199 → True.
- [D3] probe − CORN P(Y≥3) = −0.007 / −0.021 / +0.021 → False.
- **Verdict: D. MIXED/UNRESOLVED.** The loss is upstream of the decoder, in the representation's own
  ≥3 ranking of a subset of grade-4 images.
- **NO OBJECTIVE EXPERIMENT JUSTIFIED BY THIS AUDIT.**

**Interpretation.**
1. **The cumulative decoder introduces no loss of its own.**
   - E and the decoded grade inherit exactly the P(Y≥3) signal.
   - P(Y≥3) is a faithful read-out of what the frozen representation says about "grade ≥ 3".
   - CORN's own class posterior gives the conditional 3v4 answer (p_cond_3 = P4 / (P3 + P4)).
2. **E is not an appropriate grade-3-vs-4 statistic here.**
   - Among true 3/4, E ≈ 2 + P(Y≥3) + P(Y≥4). It therefore mixes "is it ≥ 3?" (a 4-vs-≤ 2 question)
     with "3 or 4?".
   - Its low 3v4 AUROC is the 3v4 ranking contaminated by 13–16 grade-4 images that the model and the
     frozen representation place with grades ≤ 2.
   - These are 4-vs-(0–2) boundary failures (overall AUROC about 0.95), not 3-vs-4 failures.
3. **The residual problem is a fixed set of grade-4 images that look ≤ 2.**
   - It is reproduced across seeds, checkpoints and two unrelated architectures (7 of 9 overlap with the
     NO_RACAF persistent-21).
   - It is not established as label error (§§15–16: heterogeneous — treated / haemorrhage-dominated /
     hazy PDR).
   - Whether it is data/label or a representation-learning limit cannot be separated without training
     or independent labels.

**Correction to §24 caveat 2 (§24's numbers and verdict unchanged).**
- §24 wrote that "CORN's cumulative decoding therefore compresses 3/4 at the decision level" and
  classed this as an ordinal-structure/objective mechanism.
- S3 shows the decoding is faithful. The decision-level drop comes from the P(Y≥3) signal, which the
  frozen representation reproduces on its own.
- The "grade 4 lies off the severity direction" observation stands. It is a property of the
  representation (driven by the gated grade-4 subset), not of CORN's decoding.

**Not claimable:**
- that CORN is defective;
- that the gated images are mislabelled;
- that a different objective would recover them.

**Next step:** the user's decision. Nothing is launched.

## 26. S4 — failure-mechanism audit of the recurrent grade-4 failures (read-only)

**Run (2026-09-30).** CPU only, on the laptop; no GPU was needed. Nothing was trained or modified.
- **Read from Drive via rclone:**
  - P and NO_RACAF per-sample predictions; the P frozen features (§24);
  - the label-noise JSON and the preprocessing log;
  - for the 58 validation grade-4 images only: the cached 512 RGB, vessel and 4-channel lesion maps
    (MA / HE / EX / SE), the RACAF reliability files, and the raw PNGs.
- **No IQA outputs exist for APTOS** (only the inference code), so none was run.
- **Scripts:** `s4_groups.py` (groups fixed from predictions before any characteristic was computed)
  and `s4_mechanism_audit.py`. The criteria in the header were fixed before the characteristics were
  computed.

**Groups (validation grade 4, n = 58; fail = decoded ≤ 2).**
- **A** — P-persistent (fail in all 3 P BEST): **9**.
- **B** — P-recurrent (fail in ≥ 6 of 9 P checkpoints, EARLY/BEST/LAST × 3 seeds): **17** (A ⊂ B).
- **C** — cross-architecture (B and fail in all 3 NO_RACAF BEST): **12**.
- **D** — recognised controls (decoded ≥ 3 in all 9 P checkpoints): **29**.
- Intermediate: 12.
- Fail counts out of 9 are bimodal: 29 images at 0, 6 images at 9.
- The NO_RACAF all-seed failure count reproduces the documented 21.

**Pre-specified criteria.**
- **Tests:** Mann–Whitney B vs D, effect = AUROC; Holm correction within each family. "Material" =
  Holm p < 0.05 and AUROC ≥ 0.75 or ≤ 0.25.
- **H1:** documented conflict rate B − D ≥ 20 points and Fisher p < 0.05.
- **H2 / H3:** at least one material difference in the quality / Stage 3/4 family.
- **H4:** |C| / |B| ≥ 0.5, and ≥ 75 % of B nearer the grade-2 than the grade-4 `pooled_final`
  centroid in each seed.
- **Gate:**
  - A if H4 ∧ ¬H1 ∧ ¬H2;
  - B if (H1 ∨ H2) ∧ ¬H4;
  - C if H4 ∧ (H1 ∨ H2), or H3 alone;
  - D if none.

**Results.**
- **Duplicate / label (documented only).**
  - Train/validation twin: A 1/9, B 2/17, C 1/12, D 5/29.
  - Twin graded ≠ 4: A 1/9 (11 %), B 2/17 (12 %), C 1/12 (8 %), D 0/29.
  - The two conflicts are `f03d3c4ce7fb` [A] and `80964d8e0863` [B], both with a twin graded 2.
  - Fisher p = 0.131 → **H1 not material**.
  - Membership of the 6 validation-only and 76 within-training duplicate groups is not documented per
    image; label coverage is therefore about 12 % of B.
- **Image quality / preprocessing (13 measures).**
  - All 58 were processed; none has an empty or near-empty FOV.
  - No measure reaches the "material" bar: all Holm p = 1; AUROC 0.46–0.63 (raw size, aspect,
    brightness, contrast, saturation, green channel, dark/bright fractions, processing time).
  - **H2 not material.**
- **Stage 3/4 outputs (25 measures; model predictions, not ground truth). B has markedly lower segmented
  haemorrhage/exudate extent:**

  | measure | B | D | AUROC | Holm p |
  |---|---|---|---|---|
  | HE area > 0.5 | 0.016 | 0.041 | 0.14 | 0.0014 |
  | HE mean prob | — | — | 0.17 | 0.005 |
  | HE components | 56 | 84 | 0.20 | 0.015 |
  | EX components | 57 | 92 | 0.21 | 0.029 |
  | any-lesion area > 0.5 | 0.037 | 0.101 | 0.19 | 0.011 |

  - Vessel map is higher in B (AUROC 0.73–0.74, raw p 0.007–0.011) but not significant after Holm.
  - RACAF reliability r: 0.73, not significant.
  - MA and SE: not material.
  - **H3 material.**
- **Prediction / representation (descriptive; circular with the group definition).**
  - B medians: P(Y ≥ 3) 0.15, E 2.19, p_cond_3 0.75.
  - B sits nearer the grade-2 than the grade-4 centroid in 88 / 94 / 100 % of images; majority
    nearest-centroid grades: grade 2 in 11, grade 1 in 4, grade 3 in 2.
  - Cross-seed SD of P(Y ≥ 3) is higher in B (0.13 vs 0.007).
- **Group A across architectures.**
  - All 9 fail in 8–9 of 9 P checkpoints.
  - 7/9 fail in all 3 NO_RACAF seeds, and 2/9 in 2 of 3.
  - Direction is consistent: of all 108 A predictions, 75 are grade 2, 28 grade 1, 3 grade 3 and
    2 grade 4. Grade 0 never occurs.
  - 1 has a documented conflict.
- **Existing AI-assisted, non-clinical visual review (§15.2), covering 8 of 9 A images.** The
  appearances are heterogeneous:
  - haze / low contrast: `1bf30c84bbad`, `cd54d022e37d`, partial `eaa0dfbd5024`;
  - spots compatible with laser scars: possible `1bf30c84bbad`; `1c4f3aa4df06` (§15.2 comparison
    group);
  - a possible dense dark haemorrhage: `b37aae3c8fe1`;
  - severe-NPDR-like tortuous vessels and flame haemorrhages: `fce93caa4758`;
  - haemorrhages and exudates with a capture-artefact band: `d48178e4a49b`, `f03d3c4ce7fb`.

  `8fd7ad26e691` has not been reviewed. No new visual review was done.

**Pre-specified flags:** H1 False, H2 False, H3 True, H4 True (|C|/|B| = 0.71; grade-2-nearer shares
0.88 / 0.94 / 1.00). **Pre-specified gate output: A. ARCHITECTURAL MECHANISM IDENTIFIED.**

**The rule's A output is not accepted — the H4 operationalisation is invalid (post hoc correction,
stated openly):**
1. **Its representation clause is circular.** B is defined by decoded grade ≤ 2, and the CORN head is
   linear in `pooled_final`, so B lying nearer the grade-2 centroid follows almost by construction.
2. **Its cross-architecture clause does not discriminate between the explanations.** Recurrence across
   two unrelated architectures (random-init CNN+Swin with Stage 3/4 priors vs ImageNet ConvNeXt, RGB
   only) is what an *image-level* property predicts, whether that property is label, data or atypical
   presentation. An architecture-specific limitation predicts the opposite.
3. **A requires "sufficiently distinct from data/label artefacts".** With documented label coverage of
   about 12 %, H1's null is not evidence of clean labels.

**Corrected gate: C. MIXED — MORE EVIDENCE REQUIRED.**

**What is established (reproducible and measurable).**
- The recurrent failures are a stable image set, failing in the same direction (to grade 1–2) across
  seeds, checkpoints and two unrelated architectures.
- Independently trained Stage 4 lesion models, trained on IDRiD and not on APTOS grades, see these
  images as having about 2.5× lower haemorrhage/exudate extent than recognised grade-4 images.
- The failures are grade-4-labelled images with low segmented NPDR-lesion burden, which the models
  place with moderate NPDR.
- Image quality and preprocessing statistics do not distinguish them.

**What is not established.** Whether grade 4 is carried by signs that none of the models represents
(neovascularisation, treated/laser-scarred eyes, preretinal/vitreous haemorrhage, media haze), or by
label error. Both remain plausible. The appearance review is heterogeneous, AI-assisted and
non-clinical.

**Future-architecture gate: not justified now.**
- An architecture aimed at "grade 4 without high lesion burden" would be justified only if those
  images carry learnable, correctly labelled grade-4 evidence.
- Evidence that would separate the explanations:
  - (a) independent clinical adjudication of the fixed B set;
  - (b) a burden-matched read-only test: does any frozen representation separate *training* grade-4
    images with low Stage-4 burden from burden-matched grade-2 images (cross-validated)? This needs
    Stage-4 caches for the training set, a CPU Colab job with no training;
  - (c) external data with PDR-specific annotations, used only for evaluation.

**Not claimable:**
- that any image is mislabelled or clinically abnormal;
- that the segmentation maps show real lesions;
- that a new architecture would help.

**Next step:** the user's decision. Nothing is launched.

## 27. S5 — burden-matched grade 4 vs grade 2 representation audit (read-only)

**Run (2026-09-30).** CPU on the laptop; runtime 7,893 s. Nothing was trained, regenerated or modified.
- **Inputs:** existing Drive files via rclone. The Stage-4 lesion cache for all 1,283 grade-2/4
  images was streamed and deleted after use. P BEST frozen features from §24.
- **Scripts:** `s5_burden.py`, `s5_analysis.py`. Every rule was in the analysis script's header
  before any burden value or representation result was seen.
- **Integrity:**
  - The split sha256 (LF-normalised) is `bc80fd45…`. The working-tree hash differs only by
    Windows CRLF; git reports the file unmodified.
  - Feature files are sha-verified; train ids and grades are identical across seeds (2,921 / 730).
  - Burden values reproduce the Phase-0 saved statistics for all 294 overlapping grade-4 images
    (max |Δ| 1.2e-7).

**Design (pre-specified).**
- **Burden:** the documented Phase-0 scalar `lesion_total_foreground_fraction` — the unweighted
  fraction of 4-channel lesion-map entries > 0.5. Matched on log10(x + 1e-4).
- **Low burden:** ≤ the 25th percentile of *training* grade-2 burden (bound 0.00795); secondary bounds
  at the 10th and 50th percentiles.
- **Matching:** greedy 1:1 nearest neighbour, without replacement; caliper 0.2 SD of log-burden.
- **Probe:** Scaler → PCA-16 → LogisticRegressionCV (C ∈ {0.01, 0.1, 1, 10}), with inner CV fitted
  in-fold. Outer 5-fold (grouped by pair for matched data), 10 repeats.
- **Controls:**
  - A: unmatched;
  - B: matched (primary);
  - C: burden as the predictor;
  - D: 1,000 label permutations.
- **Gate:** fewer than 20 pairs → INCONCLUSIVE. REMAINS if permutation p < 0.01 and CI lower bound
  > 0.55 in 3/3 seeds. COLLAPSES if p ≥ 0.05 in 3/3 seeds and mean AUROC < 0.60.
- **Validation:** used once, only if there are ≥ 15 matched pairs.

**Results.**
- **Counts:**
  - Training: 791 grade 2 / 236 grade 4.
  - Burden median [IQR]: grade 2 0.0135 [0.0080–0.0257], grade 4 0.0203 [0.0121–0.0339].
- **Primary match:**
  - 198 low-burden grade 2 vs 30 low-burden grade 4; 30 pairs, none excluded.
  - Log-burden SMD +0.221 → +0.004; median burden 0.00545 vs 0.00546.
  - **Per-channel composition stays imbalanced:** HE foreground +0.47, EX foreground −0.44,
    EX mean −0.50 SMD.
- **Secondary matches:** 10th percentile 11 pairs; 50th percentile 72 pairs.
- **Probe AUROC, repeat-averaged OOF, 95 % CI:**

  | seed | pooled_final A (unmatched) | pooled_final B (matched) | stage3_gap B | stage2_gap B |
  |---|---|---|---|---|
  | 42 | 0.9994 | 0.9944 (0.976–1.000) | 0.994 | 0.803 |
  | 123 | 0.9487 | 0.8244 (0.711–0.919) | 0.826 | 0.794 |
  | 2026 | 0.9712 | 0.8822 (0.776–0.963) | 0.893 | 0.801 |
  | mean | 0.973 | 0.900 | 0.904 | 0.800 |

  - Matched AUPRC 0.85–0.99; balanced accuracy 0.78–0.97.
  - Secondary bounds (pooled_final): 10th percentile 0.979 / 0.707 / 0.624; 50th percentile
    0.997 / 0.910 / 0.950.
- **Control C:**
  - Burden scalar AUROC: 0.613 unmatched; 0.503 matched.
  - The 10 lesion statistics through the same probe: 0.698 unmatched; 0.633 matched
    (0.520–0.744) — composition differences remain after matching.
- **Control D:** observed first-repeat AUROC 0.994 / 0.811 / 0.889, against a null mean of about
  0.49 (99th percentile 0.70–0.72); p = 0.001 in 3/3 seeds (the minimum reachable).
- **Validation:** 52 low-burden grade 2, 12 grade 4, 12 matched pairs. This is below the
  pre-specified minimum of 15, so no validation AUROC is reported.
- **S4 groups (descriptive):** 8/17 P-recurrent failures fall below the training low-burden bound
  (median burden 0.0084), vs 2/29 recognised controls (median 0.0225).

**Pre-specified gate output: REPRESENTATION_SIGNAL_REMAINS.**

**Not accepted — a design flaw, identified after the results (my error at the design stage).**
The P backbones were fine-tuned end to end on these same 2,921 training images, with these same grade
labels. A probe on training-set features therefore recovers, in part, the labels the backbone was
trained to fit. Evidence of that contamination:
1. **The matched AUROC ranks exactly with training length:**
   - seed 42 (BEST at epoch 19, training QWK 0.99): 0.994;
   - seed 2026 (epoch 7): 0.882;
   - seed 123 (epoch 4): 0.824.
2. **Unmatched training AUROCs exceed held-out values:** 0.95–0.999 on training, against about 0.956
   for 012v4 on validation (§23), and a train-CV > validation gap already seen in §24.
3. **The permutation null does not control for it:** it shuffles only the probe's labels, while the
   backbone was trained on the true ones.

The only uncontaminated test in this design is validation, and its 12 matched pairs fall below the
pre-specified minimum.

**Corrected gate: INCONCLUSIVE.**
- The existing P representations *do* separate low-burden, burden-matched grade 4 from grade 2 on the
  images they were trained on.
- This design cannot tell whether that reflects image information beyond Stage-4 burden or fitting
  of the training labels.
- A secondary caveat: matching on total burden leaves per-channel composition imbalanced (HE and EX
  SMD about ±0.45; 10-variable lesion probe 0.63 after matching).

**Missing evidence (precise).** A burden-matched test on images whose labels the representation was
never trained on:
- **(a)** the same matched training-split design on a frozen representation with no APTOS
  supervision — the pinned ImageNet ConvNeXt-Tiny initialisation of P. Its weights are already cached
  on Drive; extracting features is a GPU inference step, not training, and would need the user's
  approval;
- **(b)** a larger held-out low-burden grade-4 sample than APTOS validation provides (12 pairs);
- **(c)** cross-fitted P models, which would require training, so this is excluded now.

**Not claimable:**
- that grade-4 information beyond lesion burden exists;
- that it does not;
- anything about label correctness or PDR-specific signal.

No architecture is proposed. **Next step:** the user's decision. Nothing is launched.

### 27.1 S5 follow-up — ImageNet-only ConvNeXt-Tiny on the S5 matched sample (final grade-4 diagnostic)

**Run (2026-09-30).** CPU on the laptop; runtime 1,212 s. Nothing was trained.
- **Representation:** `pooled_final` of the P architecture built from the pinned ImageNet weights
  (sha `d547c096…`, verified against the pre-registration), with **no P checkpoint loaded**. It has
  never seen APTOS labels.
- **Inputs:** the cached Stage-2 RGB for the 60 matched images (existing cache); fixed ImageNet
  normalisation in the P adapter; float32.
- **Checks:**
  - the S5 primary sample was rebuilt exactly (30 pairs, bound 0.00795, log-burden SMD +0.004;
    asserted);
  - the adapter ignores the prior channels (|Δ| 0);
  - the features match Keras' own ImageNet ConvNeXt-Tiny to 4.2e-6.
- **Unchanged from S5:** the probe, the 10-repeat grouped CV, the metrics and the 1,000 permutation
  label sets.
- **Gate (same bars as S5, fixed before running):**
  - SIGNAL REMAINS if p < 0.01 and CI lower bound > 0.55;
  - SIGNAL DISAPPEARS if p ≥ 0.05 and AUROC < 0.60;
  - otherwise INCONCLUSIVE.

**Result.**
- **Matched AUROC 0.719 (95 % CI 0.572–0.838).** AUPRC 0.712; balanced accuracy 0.667;
  sensitivity at 95 % specificity 0.167.
- **Permutation:** observed first-repeat AUROC 0.704, against a null mean of 0.486, 95th percentile
  0.648 and 99th percentile 0.717; **p = 0.016**.
- **Comparison with S5:** the label-exposed P features scored 0.994 / 0.824 / 0.882 (mean 0.900). The
  ImageNet-only features score 0.719.

**Verdict: INCONCLUSIVE.**
- The CI lower bound passes (0.572 > 0.55), but p = 0.016 misses the pre-set 0.01 bar. The result is
  above the 95th but not the 99th null percentile.
- This is weak, underpowered evidence of some grade-4/grade-2 separation in label-free features after
  matching on total burden. It is **not** attributable to information beyond Stage-4 burden: the
  matched pairs still differ in lesion *composition* (HE +0.47, EX −0.44 SMD), and the 10 lesion
  statistics alone reach 0.633 on the same sample.
- About 0.18 of the P matched AUROC (0.90 vs 0.72) sits in representations trained on these labels,
  consistent with the §27 contamination concern.

**Grade-4 failure-mechanism audit: CLOSED (user instruction, 2026-09-30).**
- No further diagnostic experiments unless a genuinely new external evidence source is available.
- **Standing summary (§§23–27.1):**
  - no pooling or decoding bottleneck;
  - the recurrent grade-4 failures are a stable, cross-architecture set with low segmented
    NPDR-lesion burden and normal image quality;
  - whether they carry learnable grade-4 evidence beyond lesion burden, or reflect label issues, is
    **unresolved** with APTOS alone;
  - no architecture is justified on this evidence.

## 28. Post-S5 research direction gate (2026-09-30; decision document, no experiment)

Full text: `research/Grade3vs4_Architecture_Research/POST_S5_RESEARCH_DIRECTION_GATE.md`.

- **Remaining problem:** a reproducible low-NPDR-burden PDR subset. With APTOS alone it cannot be told apart as missed PDR-defining evidence or unreliable labels.
- **Viable questions:**
  - RQ1: external replication with PDR-sign annotations (FGADR Seg-set / Retinal-Lesions). Underexplored as an analysis, not as a method.
  - RQ2: clinician adjudication of the fixed failure set. Established; data curation.
  - RQ3: sign-level PDR evidence in grading. Incremental, and only if RQ1 is positive.
- **Single next investigation:** RQ1 on the FGADR Seg-set, read-only, frozen inference, with a pre-registered within-dataset test.
- **Blocked on:** the user signing the FGADR research-use agreement.
- **Status:** nothing launched.

### 28.1 FGADR access/setup check (2026-09-30; no data downloaded, nothing run)

- **Access: BLOCKED on the user.** The individual research-use agreement must be signed and emailed by the user.
  - Fields: email, name, phone, organisation, role, signature, date.
  - Terms: non-commercial research use; no redistribution; no link sharing; no re-identification; UAE law; indemnity.
  - Clause 6 forbids "derivative works" — whether preprocessing or cached outputs count as one must be confirmed with the owner.
  - Only the Seg-set is released; the Grade-set is unreleased.
- **Released Seg-set (secondary sources — FGADR paper; arXiv:2410.03188 Table 1; unverified until download):**
  - 1,842 images, graded by 3 ophthalmologists: grades 0–4 = 101 / 212 / 595 / 647 / 287.
  - Lesion masks: MA 1,424; HE 1,456; SE 627; EX 1,279; **IRMA 159; NV 49** (grade split unknown).
  - **No released laser-mark or proliferative-membrane labels are documented** (the paper mentions image-level LM/PM annotation; neither the release page nor third-party users list them).
  - PRH/VH/FP are not annotated.
  - The Seg-set was *pre-selected by a pre-trained grading model* — a possible selection bias against low-burden PDR.
- **Pipeline compatibility:** the IDRiD external-evaluation pattern applies unchanged (raw read → Stage 02 `preprocess_array(profile="DR")` → 512 RGB → frozen Stage 03/04 → P); only the dataset reader and namespaced IDs are new. Unknowns: image format/resolution and the behaviour of the Stage 03 FOV heuristic (APTOS had 3 empty-FOV exclusions).
- **RQ1 as proposed:** Test 1 (P failure vs NPDR burden among 287 PDR) is executable; FGADR's ground-truth MA/HE/EX/SE masks allow burden from annotations as well as from Stage 04. **Test 2 is not executable as proposed** (no LM/PM; NV on ≤ 49 images; NV-mask absence ≠ NV absence). It must be narrowed to NV-only, and/or supplemented with Retinal-Lesions (NV/FP/PRH/VH masks; 62 DR4; free on request) before pre-registration.

## 29. Stage 5–8 complementary-branch audit on top of P (2026-09-30; design audit, no experiment)

Full text: `research/Grade3vs4_Architecture_Research/STAGE5_8_COMPLEMENTARY_BRANCH_AUDIT.md`.

- **Candidates assessed (7):**
  - high-resolution local-detail branch;
  - complementary pretraining branch;
  - anatomy-registered lesion distribution;
  - vessel-geometry descriptors;
  - image-quality-aware conditioning;
  - frequency/texture branch;
  - consistency/ensemble (baseline hygiene).
- **Worth deeper investigation:**
  - #1 high resolution (adds information physically removed at 512 px);
  - #2 complementary pretraining (cheap and cleanly ablatable);
  - #3 anatomy-registered lesion distribution (conditional; weakest).
- **Novelty:** none is a novel mechanism; any contribution would come from a rigorous, label-free, leakage-controlled complementarity result.
- **Status:** nothing trained or launched.

## 30. Candidate #2 (complementary pretrained branch) — research only (2026-09-30)

Full text: `research/Grade3vs4_Architecture_Research/CANDIDATE2_PRETRAINED_BRANCH_RESEARCH.md`.

- **Candidates assessed:**
  - RETFound-DINOv2 (MEH/AlzEye; CC BY-NC; gated);
  - DINOv2-L (Apache-2.0);
  - RETFound-MAE (frozen features weak);
  - RETFound-Green (pretrained on DDR → DDR leakage);
  - MedSigLIP (undisclosed ophthalmology data).
- **Excluded for leakage:** FLAIR / RET-CLIP / RetiZero-type models (pretrained on APTOS/IDRiD/DDR); DINORET (DDR).
- **Literature gap:** complementarity with an ImageNet CNN has not been measured under leakage control (FusionFM fused only eye foundation models).
- **Next frozen test (pre-specify first):** RETFound-DINOv2, with DINOv2-L as the generalist control; label-free training-split CV against the Candidate #1 512 px ImageNet ConvNeXt features.
- **Blocked on:** the user accepting the RETFound-DINOv2 gated access.
- **Status:** nothing run.

## 31. Candidate #1 screen — frozen ImageNet ConvNeXt-Tiny, 512 vs 1024 px (label-free; pre-specified): DROP

**Run (2026-09-30).**
- **Colab T4 extraction:** the pipeline's own Stage-2 native RGB, resized with `jtd._resize_rgb_01` to 512 or 1024 (the only difference between arms); pinned ImageNet weights; no APTOS checkpoint.
- **Checks:** recomputed 512 RGB = cached RGB (max |Δ| 0.0); features = Keras ImageNet reference (5.3e-6 / 1.7e-6).
- **Output:** `experiments/HR_Screen_ImageNet/v1/hr_screen_features.npz` (sha `0f8addb8…`, verified locally).
- **Analysis:** on the laptop.
- **Population:** training split (2,921); validation untouched.
- **Native size:** median 2144 × 1536; minimum side < 1024 px in 327 images (11 %), < 512 px in 30.

**Pre-specified rule.**
- Primary endpoint: grade ≥ 3 vs ≤ 2 (390 vs 2,531).
- Probe: S5 probe (PCA-16, LogisticRegressionCV), identical 10 × 5-fold CV for both resolutions.
- GO only if ΔAUROC ≥ +0.02, paired CI lower bound > 0, score-swap p < 0.01, and PCA-64 Δ > 0; otherwise DROP.

**Results (primary).**
- 512 px AUROC 0.9116 (0.899–0.924), AUPRC 0.612.
- 1024 px AUROC 0.9119 (0.899–0.924), AUPRC 0.614.
- **ΔAUROC +0.0003 (paired 95 % CI −0.008 to +0.008); swap p = 0.94.**
- Label-permutation nulls: p = 0.005 at both resolutions (minimum reachable; both far above chance).

**Secondary (descriptive).**
- PCA-64: Δ +0.003 (−0.003 to +0.009), p 0.32.
- Concatenation [512 + 1024] vs 512: Δ +0.002 (−0.002 to +0.007), p 0.33.
- Adjacent pairs, 1024 − 512:
  - 0v1 −0.005 (−0.009 to −0.001), p 0.03;
  - 1v2 −0.013 (−0.029 to +0.003);
  - 2v3 +0.011 (−0.016 to +0.039);
  - 3v4 +0.020 (−0.016 to +0.055), p 0.28.

**Decision: DROP.** Higher input resolution adds no measurable grade information to frozen ImageNet ConvNeXt-Tiny features. The fine-detail-sensitive 0-vs-1 boundary is marginally *worse* at 1024.

**Limit.** This is a frozen, label-free screen. ImageNet features at 1024 px see structures at an unfamiliar scale, so a *fine-tuned* high-resolution model is not tested. That was the pre-agreed scope, and a weak or null screen stops the direction.

**Status:** Candidate #1 closed. Nothing launched.

## 32. Candidate #2 Stage A — frozen RETFound-DINOv2 (MEH) vs DINOv2 control, complementarity beyond P (pre-specified): DROP

**Run (2026-09-30).**
- **Colab T4 extraction:** timm `vit_large_patch14_dinov2.lvd142m` at 224 px for both encoders; only the weights differ.
  - RETFound checkpoint sha `a3feeaf6…`, loaded with RETFound's own key handling: 342/342 model tensors filled, 1 unexpected key (`mask_token`, unused at inference).
  - Mean |RET − DINO| per feature 1.51.
  - Input: canonical Stage-2 512 RGB → 224 (antialiased), ImageNet normalisation, final-norm CLS token (1,024-d).
  - Output: `experiments/Cand2_FrozenFM/v1/cand2_features.npz` (sha `7d1566b6…`, verified locally; timm 1.0.29).
- **P arm:** the 512 px ImageNet ConvNeXt-Tiny features from §31 (same ids and order).
- **Analysis:** on the laptop.
- **Population:** training split only (2,921); validation features not read.

**Pre-specified rule.**
- Same probe and CV as §31.
- Primary endpoint: grade ≥ 3 vs ≤ 2.
- DROP if ΔRET < +0.01 or its CI lower bound ≤ 0.
- GO if ΔRET ≥ +0.02, CI lower > 0, swap p < 0.01, Δdomain CI lower > 0 and PCA-64 ΔRET > 0.
- INCONCLUSIVE otherwise.

**Results (primary, AUROC with 95 % CI; every label-permutation p = 0.005, the minimum reachable).**

| arm | AUROC (95 % CI) |
|---|---|
| P | 0.9116 (0.899–0.924) |
| RET | 0.8888 (0.875–0.902) |
| DINO | 0.9245 (0.912–0.935) |
| P+RET | 0.9079 (0.895–0.919) |
| P+DINO | 0.9308 (0.920–0.941) |

- **ΔRET −0.0037 (−0.011 to +0.003), swap p 0.31.**
- **ΔDINO +0.0192 (+0.012 to +0.026), p 0.0001.**
- **Δdomain −0.0229 (−0.032 to −0.015), p 0.0001.**
- PCA-64: P 0.9447; ΔRET +0.0017 (−0.003 to +0.007); ΔDINO +0.0102 (+0.005 to +0.016).

**Adjacent pairs (descriptive).**

| pair | ΔRET | ΔDINO |
|---|---|---|
| 0v1 | +0.001 | +0.004 |
| 1v2 | −0.003 | −0.003 |
| 2v3 | +0.013 (−0.005 to +0.030) | **+0.051 (+0.025 to +0.076)** |
| 3v4 | −0.001 (−0.027 to +0.026) | −0.012 (−0.040 to +0.015) |

**Decision: DROP — retinal-domain complementarity is not supported.**
- RETFound-DINOv2 adds nothing beyond P.
- It is weaker than P alone (−0.023) and weaker than the generalist DINOv2 (Δdomain negative).
- Candidate #2 stops here, as instructed; Stage B is not run.

**Observation (pre-specified control quantity, not a decision endpoint; hypothesis-generating only).**
- The generalist DINOv2 ViT-L/14 adds measurable information beyond P: +0.019 at PCA-16, +0.010 at PCA-64. It is concentrated at 2 vs 3 (+0.051).
- Part of the PCA-16 gain may be capacity (DINO alone > P at PCA-16); the PCA-64 gain is half as large.
- Nothing helps grade 3 vs 4.
- Acting on this would be a new, separately pre-registered question; nothing launched.

## 33. Objective change and combination audit (2026-09-30; design audit, no experiment)

Full text: `research/Grade3vs4_Architecture_Research/COMBINATION_AUDIT.md`.

**Objective change (user, 2026-09-30).**
- Novelty is no longer required.
- The target is a combination of existing mechanisms, not yet combined here, that:
  - improves **overall** DR grading;
  - is testable in days.
- Grade 4 is no longer the sole target.

**Limitations of P that remain after §§1–32:**
- label memorisation and over-fitting (training QWK 0.99 vs validation 0.91; severity-graded
  generalisation gap);
- documented label noise (24 % same-image disagreement);
- a representation prior that frozen DINOv2 complements against frozen ConvNeXt (§32:
  +0.019 / +0.010; +0.051 at 2 vs 3);
- checkpoint-selection variance.

**Classification.**
- **Strong practical candidates:**
  - P + frozen DINOv2 late fusion;
  - P + soft ordinal targets inside CORN (N-ULS-style expected CORN NLL);
  - their combination, conditional on both passing.
- **Possible but weak:**
  - frozen-FM-guided noisy-sample down-weighting (CUFIT-like; risks discarding hard genuine PDR);
  - Stage-1 IQA quality state (a read-only gate first);
  - OD/fovea anatomy-registered pooling;
  - DINOv2 distillation (conditional on the fusion result).
- **Too expensive:** lesion-guided native-resolution crops.
- **Closed, redundant or not complementary:** vessel geometry, lesion multi-task, lesion statistics,
  RETFound, readout redesign, texture branch, SWA/TTA as a contribution, contrastive at batch 2,
  reliability fusion, PDR-sign channels.
- None of these is claimed as novel.

**Proposed minimum experiments (none launched).**
- **E1:** CPU late fusion on existing DINOv2 features and P's saved validation outputs, against an
  ensembling control (P ⊕ P′) and a frozen-ConvNeXt-probe control.
- **E2:** 3 soft-target P runs against the existing P runs.
- **E3:** the combined arm, only if both pass.
- **Primary endpoint:** mean CORN cumulative-threshold AUROC. Guardrails: QWK, grade-3 recall, and
  grade-4-vs-(0–2) AUROC.

**Status:** nothing launched; next step is the user's decision.

## 34. Capacity-controlled DINOv2 complementarity audit (pre-specified): INCONCLUSIVE — substitution, not complementarity

**Run (2026-09-30/10-01).**
- Laptop, CPU; the user ran `c3_capacity.py` in a terminal after the tool permission check failed transiently.
- Same frozen features as §31/§32; training split only (2,921); validation not read.

**Design (pre-specified).**
- Every arm has the same total probe dimensionality k:
  - P_k = PCA-k of P;
  - D_k = PCA-k of DINOv2;
  - PD_k = block-wise PCA-(k/2) of P ‖ PCA-(k/2) of DINOv2.
- Then Scale → LogisticRegressionCV, with identical 10 × 5-fold CV.
- Primary: k = 64, grade ≥ 3 vs ≤ 2, ΔDINO = PD − P.
- GO if Δ ≥ +0.01, CI lower > 0, swap p < 0.01 and the k = 128 CI lower > 0.
- DROP if Δ < +0.005 or the CI includes 0.
- INCONCLUSIVE otherwise.

**Sweep (AUROC).**

| k | P | D | PD | PD − P (95 % CI) | swap p | PD − D (95 % CI) |
|---|---|---|---|---|---|---|
| 16 | 0.912 | 0.924 | 0.919 | +0.008 (+0.001 to +0.015) | 0.02 | −0.005 (−0.012 to +0.002) |
| 32 | 0.941 | 0.946 | 0.933 | **−0.008** (−0.015 to −0.001) | — | −0.013 (−0.019 to −0.007) |
| **64** | **0.942** | **0.951** | **0.950** | **+0.008 (+0.003 to +0.014)** | **0.027** | **−0.001 (−0.006 to +0.004)** |
| 128 | 0.944 | 0.954 | 0.955 | +0.011 (+0.006 to +0.017) | 0.0001 | +0.001 (−0.005 to +0.006) |

- All label-permutation p = 0.005.
- **Adjacent pairs at k = 64:**
  - 0v1 +0.002 (−0.001 to +0.005);
  - 1v2 +0.010 (−0.002 to +0.023);
  - **2v3 +0.010 (−0.007 to +0.028)**;
  - 3v4 +0.001 (−0.026 to +0.029).
- The 2-vs-3 concentration criterion is False.

**Decision: INCONCLUSIVE** (Δ +0.008 < +0.01; swap p 0.027 > 0.01).

**Interpretation.**
1. After capacity matching, the +0.019 of §32 shrinks to +0.008 and flips sign at k = 32 — not a stable effect.
2. **The fused arm is never better than DINOv2 alone:** PD − D ≈ 0 at k = 16, 64 and 128, and negative at k = 32.
   - P adds nothing to DINOv2; the apparent "gain over P" is **substitution** (frozen DINOv2 ≈ +0.009 better than frozen ImageNet ConvNeXt), not complementarity.
   - Genuine complementarity would need PD > both P and D.
3. The §32 concentration at 2 vs 3 does not survive (CI includes 0). Grade 3 vs 4 is unchanged.
4. **Limit:** P here is the *frozen* ImageNet ConvNeXt. A frozen-feature advantage of DINOv2 says nothing about a fine-tuned backbone swap, which earlier audits rejected as "no hypothesis".

**Status:** DINOv2 is not supported as a complementary branch to P. Stop; nothing launched.

**Bearing on §33 (added in a separate session).** §33 lists "P + frozen DINOv2 late fusion" as a strong practical candidate, citing §32's +0.019 / +0.010. Under capacity matching (§34):
- the frozen gain over frozen ConvNeXt is +0.008 at k = 64 and unstable across k;
- the fused arm never exceeds DINOv2 alone.
§33's E1, the late-fusion test on P's saved validation outputs, would therefore start from a substitution effect, not demonstrated complementarity. §33 itself is left unchanged.

## 35. Companion mechanism for soft ordinal CORN targets — audit (2026-10-01; design audit, no experiment)

Full text: `research/Grade3vs4_Architecture_Research/SOFT_TARGET_COMPANION_AUDIT.md`.

**P's recipe lacks standard fine-tuning regularisers:** stochastic depth 0.0, no weight EMA, batch size 2, constant LR with plateau decay, mild augmentation, early stopping on validation QWK.

**Candidates:** weight EMA; SAM; stochastic depth; ordinal-aware mixup; noisy-label training.

**Stage 3/4 replacement (option B) is not recommended:** P does not consume Stage 3/4 outputs, and every route into P is closed.

**Strongest two:**
1. Weight EMA — targets checkpoint instability, overfitting and label noise; about zero compute; within-run paired ablation.
2. SAM — about 2× compute.

**Recommended next experiment (to pre-register):** a 2 × 2 factorial of soft targets × EMA-as-shadow — 6 runs (P and P+soft, 3 seeds each, EMA tracked), yielding all four arms.

**Status:** nothing launched.

## 36. Full Stage 3–8 pipeline architecture audit (2026-10-01; design audit, no experiment)

Full text: `research/Grade3vs4_Architecture_Research/FULL_PIPELINE_STAGE3_8_ARCHITECTURE_AUDIT.md`.

**New binding requirement (user):** Stage 3/4 outputs must be consumed by the classifier; RGB must be kept; Stages 3–7 may be replaced.

**Key constraint found:** Stage 4 is weak (IDRiD test Dice: MA 0.017, HE 0.127, EX 0.357, SE 0.024; trained from scratch on 54 images) — a genuine upstream limit.

**Three architectures:**
1. Pretrained ConvNeXt-T RGB trunk + small prior encoder + zero-init multi-scale gated injection — **recommended first**.
2. Dual pretrained encoders + cross-attention.
3. Joint multi-task heads.

**Sequence:**
- E1: Architecture 1 with the current Stage 3/4 (EMA shadow).
- E2: Stage-4 retrain (pretrained-encoder U-Net on IDRiD + Retinal-Lesions ± MAPLES-DR; gated on the IDRiD test Dice).
- E3: Architecture 1 with the upgraded Stage 4.
- E4: soft targets.
- E5: IDRiD external evaluation, single use.

**Retain:** Stage 2, LWNet (Stage 3), CORN, the ConvNeXt-T trunk. **Replace:** Stage-4 weights, Stages 5–7, RACAF.

**Status:** nothing launched.

## 37. Stage 4 replacement — cache dependency and isolation audit (2026-10-01; audit only)

Full text: `research/Grade3vs4_Architecture_Research/STAGE4_REPLACEMENT_CACHE_AUDIT.md`.

**Hazards:**
- Caches are keyed by name only (`APTOS_<id>_<kind>_512x512.npy`), with no model identity.
- Every loader defaults to the OLD Stage 4 checkpoint (sha256 `64b3c0468bdc…ce76`, deny-listed).
- A silent persistent Drive fallback fills missing local files.
- All 8 legacy archive shards (28.5 GB) mix OLD lesion and reliability files with vessel and RGB.
- `/content/cache` is shared by every notebook.

**Design:**
- A separate generation namespace `cache/Stage34_v2/s4v2-<sha12>/`.
- Lesion files as `.npz` with embedded `stage4_sha256`/`stage3_sha256`/`gen_id`.
- No reliability artefact.
- Vessel (LWNet `91f0cada…`) and RGB re-homed via filtered shard streaming, after a parity spot-check.
- Freshness and non-legacy canaries; a legacy-fingerprint invariance check.
- Legacy caches kept read-only.

**Consequence:** §36's E1 (old caches) is not part of the final pipeline.

**Status:** nothing implemented.

## 38. Stage 3–8 redesign re-audit before implementation (2026-10-01; audit only)

Full text: `research/Grade3vs4_Architecture_Research/STAGE3_8_REAUDIT.md`.

**Verified:**
- Stage 3 = LWNet sha `91f0cada…`.
- The cached vessel map is independent of Stage 4 → reuse it after a K = 25 parity check.
- Drive holds only APTOS / EyeQ / IDRiD. IDRiD segmentation is 54 training images (SE 26) and 27 test images (SE 14).
- The old Stage 4 ran at 512 px.

**Stage 4 v2 (evidence-based):**
- SE-ResNet-101 ImageNet U-Net (smp), RGB-only, 1,536 px with 512 px patches.
- Published IDRiD-only mean AUPR 0.652 (Playout & Cheriet 2024); HRNet/HRDecoder 0.713 is heavier.
- The user allows new datasets: request Retinal-Lesions and MAPLES-DR. DDR is a user decision (it costs its external independence).

**Caches:** separate Stage-3 and Stage-4 generations; old lesion, RACAF and archive shards never touched.

**Architecture 1 re-confirmed,** with separate vessel and lesion stems and per-stream contribution tests.

**Gated sequence:** B (Stage 4 v2 + segmentation gate) → C (caches) → C2 (CPU grading-relevance check) → D (one seed) → E (3 seeds) → F (soft targets) → G (IDRiD once).

**Decision:** GO for B–D; NO-GO for E–G until the gates pass.

## 39. Final Stage 3–8 design audit before implementation (2026-10-01; audit only; supersedes §36–38 where different)

Full text: `research/Grade3vs4_Architecture_Research/FINAL_STAGE3_8_DESIGN_AUDIT.md`.

**Datasets verified:**
- **TJDR:** 561 images, MA/HE/EX/SE, Apache-2.0, direct download — usable now.
- **Retinal-Lesions:** 1,593 images, 8 classes incl. vitreous/preretinal HE, NV, fibrous proliferation; request, no licence stated. Public release has only 62 DR4 images; the paper's 8-class results used a 12K internal set.
- **MAPLES-DR:** 198 images, including neo-vessels and optic disc/macula; needs a MESSIDOR image request.
- **Zhongshan laser marks:** 70 masked images; public attachments.
- **FGADR:** NV 49, IRMA 159; agreement pending.
- **DDR:** not used (would cost external independence).

**Stage 4 v2:** SE-ResNet-101 U-Net, RGB-only, multi-label with partial-label training.
- v2-a: MA/HE/EX/SE + optic disc from IDRiD + TJDR.
- v2-b: PDR channels (NV, PRH/VH merged, FP, laser, IRMA), admitted only by a pre-specified rule.

**Stage 3:** LWNet `91f0cada…` reused after parity. **Stages 5–7:** Architecture 1 with separate vessel and pathology stems, expandable K. **CORN** retained; EMA shadow; soft targets later.

**New gate C2:** frozen ImageNet ConvNeXt ⊕ v2 statistics vs ConvNeXt alone, at matched capacity, before any grading run.

**Cost to first decision:** about 4–5 T4-h.

**Decision:** GO for A–D; NO-GO for E–G until the gates pass.

### 39.1 Addendum — local EyePACS dataset (2026-10-01; found after §39 was written)

**[V]** `D:\Projects\EyePACS\raw` holds the Kaggle EyePACS 2015 dataset: 88,702 images (train + test) and `trainLabels.csv` (DR grades for 35,126 training images). `processed/` is empty. There are no lesion masks.

**Consequences:**
1. **Stage 4 targets unchanged** — EyePACS has no masks.
2. **Retinal-Lesions** (released pre-resized to 896²) is an EyePACS subset. Once its masks are obtained they can be matched to these full-resolution originals, which matters most for small PDR lesions.
3. **Semi-supervised option:** EyePACS is a large unlabelled pool for later semi-supervised Stage 4 training (pseudo-labels).
4. **Grading data:** 35,126 graded images (about 12× APTOS) could strongly address P's overfitting.
   - This is a scope change: `PROJECT_CODE.md` limits EyePACS to its historical EyeQ role.
   - EyePACS labels are known to be noisy.
   - It needs Stage 3/4 caches for about 35k images (about 4–8 T4-h).
   - It is the user's decision, and a separate pre-registered step after gate D — not part of B–D.
5. **Leakage:** no overlap with APTOS, IDRiD or DDR; Retinal-Lesions and EyeQ are EyePACS subsets.

The §39 design and gates are otherwise unchanged.

## 40. Implementation specification — Stage 3–8 v2 (2026-10-01; spec only; supersedes §39 where different)

Full text: `research/Grade3vs4_Architecture_Research/IMPLEMENTATION_SPEC_STAGE3_8_V2.md`.

**Key fixes vs §39:**
- **[V] Frame:** the canonical RGB/vessel frame is a full-frame direct 512 resize. Stage 4 therefore runs on a full-frame direct 1,536² resize, with exact 3×3 block **mean + max** pooling to 512 (uint8, 2K channels) so MA/NV presence survives.
- **Channels:** OD excluded from the first set; PRH and VH kept separate as candidates; HE = any-haemorrhage superset.
- **Partial-label loss:** masked BCE + batch-pooled Dice over annotating images only.
- **C2:** pyramid-pooled actual Stage-5 inputs vs frozen ConvNeXt at matched capacity; overall 4-cut AUROC; requires R⊕Q > R and R⊕Q ≥ Q.
- **Stage-4 gate:** against the documented old Dice values — the deny-listed checkpoint is never loaded.

**First implementation:** K = 4 (MA, HE, EX, SE) from IDRiD + TJDR.

**Six decisions to confirm before coding.**

## 41. Stage 3–8 v2 — Step 1 implemented (code + CPU tests; 2026-10-01; no training, no caches)

**§40 decisions confirmed by the user:**
1. Encoder: ImageNet SE-ResNet-101 only. No silent fallback; if unavailable, STOP. The `tu-seresnext101_32x4d` fallback in spec §3/§13 is withdrawn.
2. K = 4 (MA, HE, EX, SE); OD excluded; K-configurable.
3. TJDR is verified on download (nothing assumed).
4. Stage 3/4 in PyTorch, Stage 5–8 in Keras, with the cache as the boundary.
5. The cache holds uint8 512×512×2K mean+max maps plus the ordered channels, Stage-4 SHA, **Stage-3 SHA** and generation ID in every file.
6. The C2 gate is unchanged.

**[V] Weight availability:** smp 0.5.0 maps `se_resnet101`/imagenet to HF `smp-hub/se_resnet101.imagenet` @ `71fe95cc0a27f444cf83671f354de02dc741b18b`.
- `model.safetensors`: 197,790,168 bytes, sha256 `f4ff9b3dbf8e7bf3929b02918d64d09ceb76ce04ea95e0ddc937879c00851fc6`.
- Public, not gated. Licence field "other" (re-hosted Cadene weights, BSD-3 upstream).
- `stage4_v2.py` fetches it itself (smp is built with `encoder_weights=None`, so smp's own original-URL retry can never run), verifies the size and SHA, and loads strictly. On failure it raises `Stage4WeightsUnavailableError` with the exact error. Not downloaded on the laptop.

**Files:**
- `pipeline_v2_config.py`
- `stage34_cache_v2.py` (numpy only: guards, deny-list, npz and manifest I/O, block pooling, canary/parity, C2 pyramid features)
- `stage4_v2.py`
- `arch1_model.py`
- `arch1_data.py`
- `tests/test_stage34_cache_v2.py` (22), `tests/test_stage4_v2.py` (19), `tests/test_arch1_model.py` (8), `tests/test_arch1_data.py` (5)
- `requirements.txt` (+ `segmentation-models-pytorch==0.5.0`)

No legacy file was modified.

**Tests:** 54/54 v2 tests pass on CPU; the regression tests `test_pl_convnext` + `test_environment_package_verification` pass 29/29. Verified points:
- Zero-initialised Architecture 1 matches P at seed 42 within 1e-5, and the priors have exactly no influence at init.
- The rebuilt ConvNeXt-T matches Keras `ConvNeXtTiny` within 1e-4, with the weights copied by layer name.
- The α/γ gates receive gradient.
- Unannotated classes get zero loss and exactly zero gradient.
- The loss equals a numpy implementation of the §6 formula.
- A 1-pixel lesion survives 3×3 pooling in the max channel.
- Deny-listed SHA, legacy directories/names and `.keras` checkpoints are refused.
- Readers have no defaults.

**Implementation choices not fixed by the spec:**
- The Stage-5 residual blocks use GN(8).
- Stage-4 mask resize is area-weighted and re-binarised at 0.5.
  - **Open:** measure MA survival at 1,536 when the data are prepared; adjust the threshold only with that evidence.
- The FOV mask is any channel > 10/255.
- `DatasetSpec("TJDR").verified = False`; training refuses it until it is verified.

**Not done:** no Stage-4 training, no cache generation, no TJDR download, no commit.

## 42. TJDR download and pre-training audit — Step 2 (2026-10-01; no training, no caches)

Full report: `research/Grade3vs4_Architecture_Research/TJDR_PRETRAINING_AUDIT.md`. Script: `tjdr_preflight_audit.py` (same folder). Outputs: `datasets/TJDR/audit/`.

**[V] Download:**
- From the authors' Google Drive folder, linked from github.com/NekoPii/TJDR.
- 1,122 files, 4.85 GiB; `rclone check` md5: 0 differences.

**[V] Structure:**
- 561 images: train 448, test 113. 257 are Topcon TRC-50DX 2048², 304 are Zeiss CLARUS 500 3912².
- One palette index-mask PNG per image, full size; no orphans, no decode errors.
- Codes EX = 1, HE = 2, MA = 3, SE = 4 (paper; confirmed by lesion colour and size).
- Images with each class: MA 174, HE 318, EX 322, SE 192. 57 images are lesion-free (all TRC-50DX).

**[V] Licence:**
- The GitHub repo LICENSE is Apache-2.0 (added 2025-02-27). The CC BY-NC-ND 4.0 on arXiv is the article's licence. The data folder has no licence file.
- Use for non-commercial research with citation; no redistribution. Not a blocker.

**[V] Overlap:**
- Exact sha plus a calibrated high-pass texture correlation (positive controls 0.83–0.98; APTOS null median 0.01).
- **0** exact or near duplicates against APTOS train, APTOS test, IDRiD grading and IDRiD segmentation (which includes the old Stage-4 training data). Maximum r = 0.50 / 0.46 / 0.40 / 0.31.
- **Inside TJDR:**
  - 6 byte-identical pairs with different masks; 2 of them cross the official split (train_101 = test_023, train_092 = test_021).
  - 2 same-eye near-duplicates: train_100 ≈ 105, test_002 ≈ 003.
- Dice between the twice-annotated copies: MA 0.36–0.55, HE 0.62–0.78, EX 0.69–0.83, SE 0.88–0.94, plus one SE omission.

**[V] Tiny-lesion survival:** measured under the current rule, which is unchanged (1536, threshold 0.5, 3×3 mean + max → 512).
- TJDR loses 16 of 8,566 components, all 1–12 native pixels. IDRiD loses 0 of 17,189.
- No image loses a class. MA survival: IDRiD 100 %, CLARUS 100 %, TRC 99.7 %.
- **The MA threshold is not changed.**

**Geometry:**
- TJDR is square, so the resize is isotropic; APTOS and IDRiD are squashed by up to 1.5×.
- CLARUS is wide-field: optic disc about 100 px at 1536, vs about 140–225 px for TRC, IDRiD and APTOS. So CLARUS lesions are about 0.5–0.6× scale.
- A known, non-blocking mismatch; report validation per camera.

**Loss compatibility:**
- A_TJDR = all four classes.
- Index masks are single-label; IDRiD multi-label overlap is < 0.1 %, so nothing material is lost.
- Label noise (especially MA) is a limitation, not a blocker.

**Required in training preparation (not implemented in this step):**
- R1: pinned exclusions — train −{041, 042, 091, 174, 105} → 443; test −{021, 023, 003} → 110.
- R2: TJDR mask reader — no `.convert()`, assert mode P and values 0–4, map MA 3 / HE 2 / EX 1 / SE 4, binarise per class **before** `resize_mask_full_frame`.
- R3: `DatasetSpec("TJDR").verified = True`, with codes, counts, exclusions and source-listing sha.
- R4: stage TJDR to Drive (md5-checked).

**Decision: TJDR GO for v2-a**, conditional on R1–R4.

**Dataset relocation (user request, 2026-10-01):**
- Every local dataset now lives in the git-ignored `datasets/`: `EyePACS/`, `EyeQ/official_repo`, `IDRiD/original_download`, `TJDR/`, `_archives/`.
- The `D:\Projects\EyePACS\raw` path in §39.1 is now `datasets/EyePACS/raw`. Earlier sections are left as written; this is the path note.

### 42.1 Dataset deduplication (user request, 2026-10-01)

Removed copies were verified identical to the kept `datasets/*/raw` files (name + size for every file, CRC32 on samples, zip CRCs for archives). About 329 GB freed:
- the Kaggle EyePACS zip parts plus a second extracted copy (and a truncated partial extraction);
- the original IDRiD download (its licence files are kept as `datasets/IDRiD/LICENSE.txt` and `CC-BY-4.0.txt`);
- the original EyeQ reconstruction output (== `EyeQ/raw`);
- the 137 GB `datasets.zip` backup.

Renames:
- `EyePACS/raw/{trainLabels,sampleSubmission}.csv` were folders; they are now plain files.
- EyeQ reconstruction scripts are now in `EyeQ/reconstruction/`.
- TJDR audit outputs and the source md5 listing are now in `TJDR/audit/`.

Current layout: `PROJECT_CODE.md` → Datasets → Local layout.

## 43. TJDR training preparation — R1–R4 applied (2026-10-01; no training, no caches)

**Code:**
- `pipeline_v2_config.py`: TJDR constants — codes, exclusions with reasons, counts, source-listing sha, Drive dir.
- `tjdr_dataset.py` (new).
- `stage4_v2.py`: `resize_mask_full_frame` now refuses any non-binary or non-2-D mask; `DatasetSpec` gains read-only `metadata`; TJDR is `verified=True`.
- `tests/test_tjdr_dataset.py` (new); `tests/test_stage4_v2.py` updated.

**R1:** pinned exclusions — train −{041, 042, 091, 174, 105}, test −{021, 023, 003}.
- `usable_ids` enforces **443 / 110**.
- `load_pair` refuses excluded ids.

**R2:** masks are read with `np.asarray(Image.open())` and never `.convert()`-ed.
- Mode P and values {0..4} are required.
- Explicit map MA 3 / HE 2 / EX 1 / SE 4.
- Split into (K, H, W) binary masks before `resize_mask_full_frame`. The threshold is unchanged at 0.5.

**R3:** `verify_tjdr` passed on all 561 real pairs (`datasets/TJDR/audit/tjdr_v2_verification.json`).
- Official class-image counts match §42.
- Usable class-image counts:

  | split | MA | HE | EX | SE |
  |---|---|---|---|---|
  | train | 135 | 245 | 251 | 147 |
  | test | 37 | 67 | 64 | 39 |

**R4:** server-side copy to `gdrive:DiabeticRetinopathy/datasets/TJDR/raw`.
- `rclone check` against the local verified files: 1,122 / 1,122 md5 match, 0 differences.
- The verification JSON and source listing are in `TJDR/audit/`.
- The full set is copied; exclusions are enforced in code.

**Tests:** 66/66 pass — the new TJDR tests plus the Stage-4, cache, arch1-data and arch1-model suites.

## 44. Stage-4 v2 K=4 training preparation (2026-10-01; code only — no training, no test evaluation, no caches)

**Data:**
- IDRiD v2 uses the 54 lesion-training images only, split into a pinned **44 train / 10 val**.
  - The split is stratified by SE presence (26/54 SE+ → 5 SE+ and 5 SE−), drawn with `default_rng(20261001)`.
  - Val: IDRiD_05, 10, 11, 14, 15, 18, 24, 30, 35, 48 (SE+: 14, 18, 30, 35, 48).
  - Split sha256 `2e5bf1c36f6126e68d0356c6252665bbb6871aab0d725bff08efb11439142f30`.
- Test images (IDRiD_55–81) are refused everywhere except `stage4_v2_gate`. Grading ids and paths are refused.

**Stage 2:** the canonical DR profile.
- IDRiD reuses its existing Drive `processed/` output (JPEG; the notebook checks parity to PSNR ≥ 30).
- TJDR gets `datasets/TJDR/processed` once, via `preprocess_image` (lossless PNG; the notebook checks it is bit-exact).
- Geometry is unchanged; the only resize is the full-frame direct resize to 1536.

**Training-input cache `cache/Stage4Train/s4train-v1`:**
- Built once by `colab/notebooks/stage4_v2_training_cache.ipynb`: Stage-2 image at 1536² uint8, plus the 4 binary masks under the unchanged 0.5 rule.
- Contents: IDRiD 44/10 and TJDR 443/110 (TJDR's official test set is the validation set).
- Manifest with per-file sha and a fingerprint. Roles are train/val only. It is not a prediction cache.

**Training (`stage4_v2_train.py`; notebook `stage4_v2_training.ipynb`):**
- Model and loss as fixed in §40: pinned SE-ResNet-101 smp U-Net (no fallback), K = 4, partial-label loss, w⁺ = clip(√(neg/pos), 1, 20) from FOV training-pixel counts, λ = 1.
- Encoder LR = 0.1 × decoder LR, gradient clip 1.0, EMA 0.999, AMP.
- 512 patches with class-aware p = 0.5 centring and flips.
- Dataset-balanced batches: 4 IDRiD + 4 TJDR. Sampling is counter-seeded, so a resume reproduces it.
- Chosen here (not fixed by §40):

  | setting | value |
  |---|---|
  | optimiser | AdamW, weight decay 1e-4 (none on 1-D parameters) |
  | decoder LR | 3e-4 |
  | warmup | 500 steps, then cosine down to 1 % |
  | total | 12,000 iterations, batch 8 |
  | validation | every 1,000 steps |
  | checkpoint | every 500 steps |

- Checkpoints hold: model, optimizer, scheduler, scaler, EMA, the sha of each weight set, config, loss weights, data fingerprints (cache, IDRiD split, TJDR exclusions + source sha, encoder sha), RNG state, environment, and history.
- **Validation** (EMA model, full 1536 frame):
  - per class, per group: IDRiD-val, TJDR-val-TRC50DX, TJDR-val-CLARUS500, plus an explicitly labelled both-camera TJDR row;
  - pooled soft Dice (the `training.metrics.dice_coefficient` definition) and pixel AUPR;
  - **model selection** = the mean over those three groups of the 4-class mean AUPR.

**One-time gate (`stage4_v2_gate.py`):**
- Runs only with the explicit confirmation token. A lock file makes it single-use.
- **Criteria:**
  - old-protocol Dice (pooled soft, 512², old `> 0` target resize, prediction = 3×3 block mean) must beat the documented reference in every class;
  - mean AUPR over the 4 classes must be ≥ 0.55.
- **Reference (verified against `SEGMENTATION_ARCHITECTURE.md` and the stage-04 notebook output): MA 0.0165, HE 0.1273, EX 0.3574, SE 0.0244.**
  - The request listed "MA 0.357, HE 0.127, EX 0.024, SE 0.017", which assigns the numbers to the wrong classes. The documented values are pinned.

**Colab:** `colab/common/stage4_v2_setup.py` does the project setup, then:
- points `TJDR_RAW_DIR` / `TJDR_PROCESSED_DIR` at Drive;
- checks smp == 0.5.0;
- fetches the pinned weights into a Drive cache and verifies size + SHA, raising on failure;
- checks the TJDR 443/110 counts and the IDRiD split.

## 45. Stage-4 v2 final pre-training step (2026-10-01; no training, no test access, no prediction caches)

**Commit:** `72f2557c45ab3ec65870e131f85322f40f178209`, pushed to `main`. It contains the v2 code, tests, notebooks, Colab setup and the smp pin. It does not contain PROJECT_CODE.md, the docs or `research/`.
- 78/78 v2/TJDR tests pass.
- The training notebook's cell 4 is now a T4 memory smoke test:
  - 3 steps at batch 8, falling back to batch 4 on OOM or if peak reserved memory exceeds 85 % of the card;
  - the result is written to `experiments/Stage4V2/smoke_test.json`;
  - the run id records the selected batch.

**Training-input cache `s4train-v1`:** built locally with the committed module functions. The same steps as `stage4_v2_training_cache.ipynb`, on CPU.
- **Stage-2 parity:**
  - TJDR `processed/` PNGs are bit-exact against a fresh Stage-2 run (max |Δ| = 0; 553 usable images written to `datasets/TJDR/processed`).
  - IDRiD's existing Drive `processed/` JPEGs match at PSNR 45–47 dB (JPEG loss only).
- **Result:** 607 files, 2.7 GB, all SHAs verified.
  - Counts: IDRiD 44/10, TJDR 443/110.
  - Cameras: TJDR train CLARUS 241 / TRC 202; val CLARUS 55 / TRC 55.
  - Fingerprint `642d87561d94977b4bba1e7f28bbb8da6d6e28b4765c10a99286a378d1fe3682`.
  - Split sha matches.
  - No IDRiD test id and no excluded TJDR id is present.
  - Images with positive pixels at 1536: MA 226, HE 365, EX 369, SE 212. These equal the native class counts, so no class was lost.
- **Drive upload** (`cache/Stage4Train/s4train-v1`): interrupted by local network failures (DNS "no such host" against googleapis). 157/607 files (0.7 GB) were uploaded when recorded.

**Not done:** the T4 smoke test (needs a Colab GPU) and any training.

### 45.1 Drive cache completed on Colab (2026-10-01)

The laptop upload was too slow and was stopped at 404/607 files.
- `progress.json` was uploaded to Drive, then the cache notebook was run on Colab. It skipped the 411 files whose SHA matched and rebuilt the rest. The notebook's TJDR Stage-2 step was changed to run on demand (commit `efd46eff34a9c9fcecaab8535c8c3a2d3bca7305`).
- **Colab result:** IDRiD 44/10, TJDR 443/110, complete.
  - Drive fingerprint: `7fe87712dbd284bfa4f3fe9283d9c844792f85fd8f512290b8a05d6a17e50a21`.
  - This **is the canonical fingerprint for training**; the local copy's `642d8756…` is superseded.
- **Why the fingerprints differ:** the 196 rebuilt `.npz` files have different byte hashes (zip timestamps). Content is identical:
  - same file names;
  - identical metadata and per-class positive-pixel counts at 1536 for all 607 files;
  - 8 randomly sampled rebuilt files compared array by array: RGB max difference 0, 0 differing mask pixels, zip integrity OK.
- TJDR Stage-2 parity on Colab: bit-exact (max |Δ| = 0).

### 45.2 T4 memory smoke test (2026-10-01)

Cells 1–4 of `stage4_v2_training.ipynb` on a Tesla T4 (14.56 GiB), commit `046259e`.
- The pinned SE-ResNet-101 weights were downloaded from HF smp-hub and verified by size and SHA in setup, which raises on mismatch.
- Training cache staged with the remount-safe copier.
- **Batch 8:** peak allocated 5.58 GiB, peak reserved 5.64 GiB (39 % of the card), fits.
- **Selected batch size 8** (4 IDRiD + 4 TJDR); batch 4 was not needed.
- Run dir: `experiments/Stage4V2/stage4v2_k4_seed42_bs8`.
- Only the 3-step throw-away smoke run has executed. The real training run has not started.

## 46. Downstream implementation after Stage 4 — gate audit, APTOS cache, C2, Architecture 1 (2026-10-01; code + CPU tests only)

**Status:** nothing downstream has run. No Stage-4 model existed at the time of writing: Stage-4 K = 4 training was in progress in Colab. No APTOS Stage-4 cache, C2 result or Architecture-1 result exists. Every step below needs the trained, exported and gated Stage-4 model.

**Execution order** (§40 step 3 onward):
1. `stage4_v2_training.ipynb`: train, export the best EMA model (cell 6), then run the one-time IDRiD gate (cell 7).
2. `stage4_v2_aptos_cache.ipynb`: Stage-3 parity + copy, RGB parity + copy, Stage-4 v2 inference, bundle.
3. Laptop C2: `python stage4_v2_c2.py --r … --q … --v … --stage4-sha …`.
4. `architecture1_training.ipynb`: seed 42 only, then evaluation against P-42.
5. Multi-seed work only after the one-seed result is recorded.

**Gate audit (`stage4_v2_gate.py`).** It was already correct on all of the following:
- explicit token required;
- only reader of IDRiD_55–81;
- old 512² pooled soft-Dice protocol with the old `> 0` target resize;
- reference MA 0.0165 / HE 0.1273 / EX 0.3574 / SE 0.0244, mean AUPR ≥ 0.55;
- lock written before evaluation.

Fixes:
- **The lock is now keyed by model SHA**, not by export folder. Previously, re-exporting the same weights would have reopened the gate.
- The report records the model SHA and path, and is also written to `experiments/Stage4V2/<run>/gate/`.
- `export_best` is idempotent and writes `exported_model.json` (path + SHA), so no model path is typed by hand.

**APTOS cache (`stage4_v2_aptos_cache.py`).**
- **Population:** the verified split minus the 11 pinned empty-FOV ids (from the six-run manifest) = 2,921 + 730 = **3,651**; population SHA recorded. Only 12-hex APTOS ids are accepted.
- **Stage 3:** LWNet SHA asserted. A 25-id fixed-seed parity recomputes exactly the reused cache's producer (LWNet with TTA on the native Stage-2 RGB → channel 3 of `racaf.prepare_stage4_input`); tol 1e-4; a failure stops the run. The loose `_vessel_` files are copied into `cache/Stage3/s3-91f0cada/` with per-file SHA and validation, plus the C2 V-pyramid.
- **RGB:** 25-id parity against `jtd._resize_rgb_01` of the Stage-2 output (tol 1e-6); the loose `_rgb_` files are copied into `cache/Stage2/rgb-v1/`. This is the second, and only other, permitted legacy read, through the new `assert_legacy_rgb_source` guard.
- **Stage 4:**
  - the model is loaded by SHA (deny-list enforced);
  - cuDNN deterministic, AMP, full-frame 1536 → exact 3×3 mean+max → uint8 512×512×8 with embedded Stage-4/Stage-3 SHA, generation id and channel order;
  - output goes to `cache/Stage4/s4v2-<sha12>-K4/`.
- **Generation rules:**
  - the directory holds one model only, recorded in `progress.json` before the first write and flushed on any crash;
  - a different model's partial directory is refused;
  - a completed generation is immutable.
- **Checks:** an 8-image freshness canary (≤ 2/255); a full verification pass over all 3,651 files, which also yields the C2 Q-pyramid; a final count of 3,651.
- **Bundle:** `cache/Bundle/rgb-v1__s3-91f0cada__s4v2-<sha12>-K4/`. It binds both generation SHAs, the population SHA, train/val ids, per-generation file fingerprints, K/channels, preprocessing version, dims, dtype, pooling, the gate record and a bundle fingerprint.
- **Gate requirement:** generation needs a gate report for this model SHA. A FAIL proceeds only with an explicit override token, which is recorded.

**Downstream safety (`arch1_data.Arch1Bundle`).** Opening a bundle refuses:
- a missing or mismatched Stage-4 model SHA;
- a non-v2 generation name;
- failed Stage-3 or RGB parity;
- the wrong split or population;
- train/val overlap or non-APTOS ids;
- a fingerprint that does not recompute;
- missing files.

Every sample is read through the SHA-verified readers.

**C2 (`stage4_v2_c2.py`, laptop CPU).** It reuses the §34 machinery unchanged. Pre-registered here before any Stage-4 map exists:
- R = frozen ImageNet ConvNeXt-T f512 (`HR_Screen_ImageNet/v1`, SHA pinned);
- Q = 336-d pyramid of the 8 Stage-4 channels; V = 42-d vessel pyramid;
- arms R_64, Q_64, R_32+Q_32, R_32+V_32: block-wise Scale→PCA fitted in-fold, then Scale→LogisticRegressionCV;
- 10 × 5-fold with identical folds; training split only (2,921), validation ids refused;
- endpoint: mean AUROC over the cuts ≥ 1, ≥ 2, ≥ 3, ≥ 4;
- uncertainty: 2,000 paired grade-stratified bootstraps;
- **PASS iff R+Q − R ≥ +0.005 AND CI lower > 0 AND R+Q ≥ Q**;
- descriptive only: ≥ 3 vs ≤ 2, the V increment, and leave-one-class-out increments.

Output goes to `results/C2/c2_<sha12>_<timestamp>/` (config.json, results.json, report.md); directories are never overwritten.

**Architecture 1 (`arch1_train.py`, `architecture1_training.ipynb`).**
- The P protocol is copied exactly, so the run is comparable with P-42: AdamW 1e-4 / wd 0.05 (no decay on 1-D), weighted CORN with the pre-registered class weights, batch 2, ≤ 50 epochs, val-QWK early stopping (12), ReduceLROnPlateau (4, 0.5, 1e-6), mixed precision.
- It uses P's epoch order and per-image augmentation (spatial on all channels together, intensity on RGB only) and P's `Trainer` / BEST / LAST machinery.
- Pre-flight: the zero-initialised model must equal P on real samples (≤ 1e-5).
- Evaluation: BEST and LAST; QWK, MAE, recalls and AUROCs; contribution tests permuting Q, V and both (spec §9).
- Pre-specified one-seed checks vs P-42 (§40 step 6):
  - Q-permutation ΔQWK ≤ −0.01;
  - QWK ≥ P-42 − 0.02;
  - AUROC(P≥3, grade 4 vs 0–2) ≥ P-42 − 0.01;
  - guardrails grade-3 recall and false-urgent rate.
- The notebook requires the C2 results file. If C2 FAILED, it requires an explicit acknowledgement, because §40 makes Architecture 1 then a low-prior test.
- Run directory: `experiments/Architecture1/arch1_<sha12>_seed42/`. The config hash is checked on every resume; a changed configuration is refused.

**Layout.** Caches stay in the §40/§11 `cache/` namespace (Stage2/Stage3/Stage4/Bundle), which `arch1_data` reads. Experiments go to `experiments/{Stage4V2, Architecture1}/<run>/` on Drive and `results/C2/<run>/` on the laptop. Every config records the git commit, model SHA, split SHA, population SHA, cache/bundle fingerprint, Stage-3 SHA, Stage-4 generation, seed and hyperparameters.

## 47. Stage-4 v2 K=4 run 1 — training, export, validation review (2026-10-02; IDRiD test NOT accessed)

**Run:** `experiments/Stage4V2/stage4v2_k4_seed42_bs8` on a T4.
- Commit `046259e`; seed 42; 12,000 steps at batch 8; decoder LR 3e-4.
- Positive weights `[20.0, 16.1, 20.0, 20.0]` (MA, HE, EX, SE): three classes at the cap.

**Export:** `exported_models/LesionSegmentation_v2/2026-10-01_18-38-32/model.pt`.
- **SHA `cb5fc7a8d370af7d2ae191cadaebdde858f71820d147fb76f852b757766f8ad8`**, from the best EMA at **step 6,000** (selection score 0.4738).
- Checkpoint hash verified, after fixes `d9e4999`, `20a9c13` and `4e32e0a`: the original EMA hash depended on the tensor device, and setup cached the encoder weights on Drive.

**Validation, selection score (mean over IDRiD-val, TJDR-TRC50DX, TJDR-CLARUS500 of the 4-class mean AUPR):**

| step | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 | 10000 | 11000 | 12000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| score | 0.012 | 0.302 | 0.423 | 0.451 | 0.454 | **0.474** | 0.454 | 0.450 | 0.464 | 0.448 | 0.463 | 0.444 |

- Step 1000 reflects EMA lag from the initial weights.
- Soft Dice keeps rising after step 6000 while AUPR is flat. IDRiD-val 1536 mean Dice: 0.125 at 6000, 0.215 at 11000. The probabilities become better calibrated late in training.

**Best step (6000), AUPR / pooled soft Dice at 1536:**

| group | MA | HE | EX | SE | mean AUPR |
|---|---|---|---|---|---|
| IDRiD-val (10) | 0.255 / 0.028 | 0.506 / 0.126 | 0.725 / 0.308 | 0.592 / 0.037 | 0.520 |
| TJDR-val TRC50DX (55) | 0.235 / 0.003 | 0.712 / 0.032 | 0.673 / 0.044 | 0.527 / 0.014 | 0.537 |
| TJDR-val CLARUS500 (55) | **0.024** / 0.000 | 0.626 / 0.026 | 0.340 / 0.014 | 0.469 / 0.007 | 0.365 |

Wide-field CLARUS is much weaker, and its MA is near chance: the scale risk from §42.

**Gate protocol applied to IDRiD-val** (10 pinned validation images, model `cb5fc7a8…`, CPU). Old-protocol 512 pooled soft Dice:

| class | v2 Dice | reference | beats reference |
|---|---|---|---|
| MA | 0.035 | 0.0165 | yes |
| HE | 0.147 | 0.1273 | yes |
| EX | **0.321** | 0.3574 | **no** |
| SE | 0.043 | 0.0244 | yes |

- Mean AUPR at 1536 is **0.522 < 0.55**. **Would FAIL** (EX Dice and mean AUPR).
- Diagnosis:
  - **EX is a calibration shortfall.** Its AUPR is 0.73; the soft-Dice deficit comes from probability mass spread over background under w⁺ = 20.
  - **Mean AUPR is limited mainly by MA** (0.26), and by CLARUS overall.
- The 10-image validation estimate is not the test result.

**Decision pending (user):**
- **A.** Run the one-time gate as pre-registered. A FAIL proceeds only with the recorded override; C2 then decides the information question. This is the assistant's recommendation.
- **B.** One improvement run first: calibration / positive-weight cap, CLARUS scale augmentation, Dice-aware selection. This is a documented deviation, decided on validation only. Only the model finally committed to is gated.

## 48. One-time IDRiD test gate — Stage-4 v2 `cb5fc7a8…`: **PASS** (2026-10-01T19:00 UTC; run once, locked)

**Decision §47 A:** the gate was run as pre-registered on the exported model.
- Model SHA `cb5fc7a8d370af7d2ae191cadaebdde858f71820d147fb76f852b757766f8ad8`; 27 IDRiD test images.
- Lock `exported_models/LesionSegmentation_v2/idrid_test_gate_locks/cb5fc7a8….json`.
- Report in `experiments/Stage4V2/stage4v2_k4_seed42_bs8/gate/idrid_test_gate.json`.

**Pass criterion:** old-protocol pooled soft Dice must beat the documented old Stage 4 in every class, and mean AUPR ≥ 0.55.

| class | v2 Dice (512, old protocol) | old Stage 4 | v2 AUPR (1536) |
|---|---|---|---|
| MA | **0.0835** | 0.0165 | 0.284 |
| HE | **0.3817** | 0.1273 | 0.567 |
| EX | **0.5068** | 0.3574 | 0.868 |
| SE | **0.1628** | 0.0244 | 0.506 |
| mean | 0.2837 | 0.1314 | **0.5562** (≥ 0.55) |

**PASS.**
- Every class beats the old Stage 4: Dice ×5.1 MA, ×3.0 HE, ×1.4 EX, ×6.7 SE.
- The AUPR criterion is met by a narrow margin (+0.006).
- The 10-image validation estimate (§47) had predicted a fail on EX and AUPR. The test set scored higher; small-sample variance between 10 and 27 images.
- For reference only, the literature IDRiD-only SE-ResNet-101 mAUPR is 0.652 (§40). Not a criterion.

**Use:**
- The test result is final for this model and is not used for any further selection or tuning.
- The IDRiD test set is now consumed for Stage 4.

**Next:** APTOS cache generation (`stage4_v2_aptos_cache.ipynb`) for this model, then C2.

## 49. Post-Stage-4 audit fixes (2026-10-02; code only — no training, no cache generation, no Colab)

The read-only audit found one CRITICAL defect and several smaller items. Fixed here:

- **CRITICAL — unbounded image prefetch in `stage4_v2_aptos_cache.generate_stage4_maps`.**
  - `ThreadPoolExecutor.map` submits every load at once (verified: all tasks start before the first result is consumed).
  - The reader threads would have decoded all 3,651 native APTOS images (about 3–21 MB each) far faster than the GPU consumes them, exhausting Colab's ~12 GB RAM.
  - Replaced by `bounded_prefetch`: a sliding window in which a new read starts only after the consumer has finished with the previous image.
  - At most `STAGE4_CACHE_PREFETCH = 4` native images are resident at once (loading, waiting, or in use), about 84 MB worst case, with `STAGE4_CACHE_READERS = 2` threads.
  - Order, image-id association, in-order exception propagation, and cancellation on early exit are preserved.
- **Second queue in the same path.** Finished map arrays waiting for Drive writes were only drained every 100 images, so up to about 100 × 2 MB could queue.
  - Writes now use back-pressure: at most `STAGE4_CACHE_MAX_PENDING_WRITES = 8` pending, with `STAGE4_CACHE_WRITERS = 4`.
  - The native image and maps are released as soon as they are used.
- **MEDIUM — slow bundle staging.** `colab/common/stage4_v2_setup.stage_files` now copies with 8 threads, keeping per-file SHA checks, resume, and remount-and-retry of only the failed files.
- **MINOR:**
  - The training notebook header now describes cells 4/5/6 correctly.
  - The last cell of the APTOS cache notebook prints the exact laptop C2 commands and the Drive upload for `C2_RESULTS`.
  - The spec has a superseded-items note (no fallback encoder; Stage-3 SHA in the npz; EMA for Architecture 1 not implemented and still open).

**Unchanged:** the Stage-4 model and checkpoint, loss, preprocessing, pooling, cache format, research protocol, C2 and Architecture 1.

**Left as is:**
- The EMA decision (open).
- `PROJECT_CODE.md`, which still describes the old Stage 04; the user asked for it not to be modified.
- A Stage-3 regeneration path: the code stops if parity fails, as specified.

### 49.1 APTOS cache run — Stage 3 done; RGB copy completed server-side (2026-10-02)

`stage4_v2_aptos_cache.ipynb`, first run:
- **Stage 3:** parity passed. 3,651 vessel maps, the manifest and `c2_v_pyramid.npz` are on Drive (`cache/Stage3/s3-91f0cada/`).
- **RGB:** parity max |Δ| = 0.0 (tol 1e-6).
- **Stall:** the Drive→Drive copy through Colab's FUSE mount stalled after about 1,000 files were recorded (586 actually uploaded; no progress for about 19 minutes). The cause was upload backlog or throttling after several thousand rapid file creations. The runtime was deleted.
- **Completion:** the RGB copy was finished from the laptop with an rclone **server-side** copy (`cache/LocalFeatureExtraction/APTOS_<id>_rgb_512x512.npy` → `cache/Stage2/rgb-v1/rgb/`) for exactly the 3,651 population ids. The legacy folder holds 3,662 RGB files; the 11 pinned empty-FOV ids are excluded.
- **Verification from Drive metadata:** 3,651 files, 0 duplicates, 0 missing, 0 extra, and every SHA-256 and size equal to its source.
- `cache/Stage2/rgb-v1/copy_progress.json` was written with those SHA-256 values, so the notebook skips the copy and still runs its own full verification pass (array validation + SHA) and writes the Stage-2 manifest.

No legacy file was modified. No Stage-4 map exists yet.

## 50. APTOS Stage-4 v2 cache complete; C2 information screen — **FAIL** (2026-10-02)

**Cache.** `stage4_v2_aptos_cache.ipynb` completed on Colab.
- Bundle `rgb-v1__s3-91f0cada__s4v2-cb5fc7a8d370-K4`, fingerprint `5069adc0fd7ae0774ef775ae64786931c7aed6e54d7ef90f6dc6ac25c73cba8d`.
- 3,651 images; Stage-4 SHA `cb5fc7a8…`; Stage-3 SHA `91f0cada…`; split `bc80fd45…`; population SHA `fc55cdde…`.
- Channels: MA / HE / EX / SE, each as mean and max.

**C2.** `stage4_v2_c2.py` on the laptop CPU; 1,210 s; run once.
- Inputs:
  - R = `hr_screen_features.npz` (SHA `0f8addb8…`, verified);
  - Q = `c2_q_pyramid.npz` (3,651 × 336; provenance model `cb5fc7a8…`);
  - V = `c2_v_pyramid.npz` (3,651 × 42).
- APTOS training split only: n = 2,921, grades [1444, 296, 791, 154, 236]. Validation not read.
- k = 64; 10 × 5-fold; 2,000 grade-stratified paired bootstraps.
- Output: `results/C2/c2_cb5fc7a8d370_2026-10-02_08-46-52/` (results.json SHA `2fb64280…`), copied to `gdrive:DiabeticRetinopathy/experiments/C2/`.

| arm | mean AUROC (95 % CI) | ≥1 | ≥2 | ≥3 | ≥4 |
|---|---|---|---|---|---|
| R_64 | 0.9581 (0.9512–0.9642) | 0.9970 | 0.9804 | 0.9421 | 0.9130 |
| Q_64 | 0.9268 (0.9191–0.9339) | 0.9974 | 0.9758 | 0.8991 | 0.8348 |
| R_32 ⊕ Q_32 | 0.9577 (0.9511–0.9636) | 0.9985 | 0.9812 | 0.9424 | 0.9086 |
| R_32 ⊕ V_32 | 0.9545 (0.9480–0.9608) | 0.9968 | 0.9778 | 0.9375 | 0.9061 |

**Primary (pre-registered):** R⊕Q − R = **−0.0005** (95 % CI −0.0042 to +0.0033).

**Checks:**
- Δ ≥ +0.005: **false**.
- CI lower > 0: **false**.
- R⊕Q ≥ Q: true (+0.0309, CI +0.0240 to +0.0378).

**→ C2 FAIL.** The CI upper bound (+0.0033) lies below the pre-registered minimum effect (+0.005).

**Descriptive:**
- V increment: R⊕V − R = −0.0036 (−0.0075 to +0.0002).
- Leave-one-class-out, R⊕Q minus R⊕Q without the class; every CI includes 0:

  | class | Δ |
  |---|---|
  | MA | −0.0011 |
  | HE | +0.0021 |
  | EX | −0.0005 |
  | SE | −0.0010 |

**Reading:**
- The Stage-4 v2 maps do carry grading information on their own: Q_64 scores 0.927, and equals R at ≥1 and ≥2. They are clearly weaker at ≥3 and ≥4.
- At matched capacity, that information is already contained in the frozen ImageNet ConvNeXt features. Replacing half of R's capacity with Q neither helps nor hurts. This is **substitution, not complementarity** — the same pattern as §34.

**Limits of the screen** (stated in §40 before it ran):
- pyramid-pooled mean/max features discard fine spatial detail;
- the probe is linear on frozen features;
- training-split cross-validation only.

**Consequence under §40:** Architecture 1 ("D") is allowed only as a clearly low-prior test; the notebook requires `ACKNOWLEDGE_C2_FAIL`. PL remains the completed fallback that satisfies the binding Stage 3/4 requirement. Decision pending (user).


## 51. Architecture 1 — seed-42 exploratory run: pre-run record (2026-10-02; written before training)

**Status.** C2 failed its pre-registered criterion (§50). Architecture 1 seed 42 is therefore a low-prior exploratory / falsification test, not a confirmatory experiment. One seed only; no further seeds run automatically. Training is run manually in Colab (`architecture1_training.ipynb`).

**C2 diagnostic audit (read-only; stored C2 outputs only, nothing refitted against grades).**
- Per-cut R⊕Q − R: ≥1 +0.0014, ≥2 +0.0008, ≥3 +0.0003, ≥4 −0.0045. Q − R: +0.0004, −0.0046, −0.0430, −0.0782.
- The information loss in Q is in the pooling (2,097,152 cache values → 336 grid statistics; finest cell 128 × 128 px), not in the probe's PCA (PCA-32 keeps 85 % of Q's variance, PCA-64 94 %; label-free check on the 2,921 training rows).
- Capacity matching masks little: R_32 ≈ R_64 at ≥3 (0.941 vs 0.942, §34).
- Image-level peak probability ≥ 0.9 in 53 % (MA), 58 % (HE), 81 % (EX) of training images, so the "max" half of Q is weak. Architecture 1 receives the same maps.
- Classification: C2 is evidence of redundancy for pooled, linearly-read lesion statistics; it cannot test pixel-aligned, multi-scale, non-linear, end-to-end use of the maps. That untested capability is the only reason this run is made.

**EMA.** Not used. The implementation follows the P protocol, which has no EMA. This is an explicit deviation from the §40 planning note ("EMA shadow"). EMA is not evaluated separately. `config.json` records `"ema": "none"`.

**Pre-run audit (laptop; Drive manifests read via rclone).**

| check | result |
|---|---|
| Stage-4 model SHA in bundle and Stage-4 manifest | `cb5fc7a8d370af7d2ae191cadaebdde858f71820d147fb76f852b757766f8ad8` |
| Stage-4 generation | `s4v2-cb5fc7a8d370-K4`; channels MA/HE/EX/SE × mean/max |
| Stage-3 SHA | `91f0cada…` (generation `s3-91f0cada`); parity passed, max \|Δ\| 8.3e-7 |
| RGB parity | passed, max \|Δ\| 0.0 |
| split SHA | `bc80fd45…`; population SHA `fc55cdde…`; bundle fingerprint `5069adc0…` |
| training ids | 2,921 = authoritative training split minus the 11 pinned empty-FOV ids (same order) |
| validation ids | 730 = authoritative validation split minus those ids (same order); no overlap with training |
| P-42 comparison set | P-42 `per_sample_best.csv` has the same 730 ids in the same order |
| IDRiD | the bundle accepts only 12-hex APTOS ids; no IDRiD path is read by `arch1_*` |
| legacy Stage-4 | deny-listed SHA, legacy-path and non-v2-generation refusals unchanged |
| C2 | read only for its PASS/FAIL flag; not a target, weight or hyper-parameter |
| mixed precision | forward + 3 optimizer steps under `mixed_float16` on CPU at 64 px: finite, loss-scaled AdamW |

**Protocol (unchanged).** Model, prior encoder, gated injection, CORN, AdamW 1e-4 / wd 0.05, ReduceLROnPlateau, augmentation, batch 2, BEST-by-val-QWK selection, seed 42 — as committed in `4c35b9a`.

**Pre-set criteria (§40 step 6), with the P-42 BEST values they refer to.**

| criterion | P-42 | Architecture 1 must reach |
|---|---|---|
| QWK | 0.9184 | ≥ 0.8984 |
| AUROC (≥3; grade 4 vs 0–2) | 0.9588 | ≥ 0.9488 |
| lesion-shuffle QWK drop (Stage-4 maps permuted across validation images; RGB and vessel unchanged) | — | ≥ 0.01 |
| grade-3 recall | 0.6154 | ≥ 0.5154 |
| false-urgent rate | 0.0379 | ≤ 0.0579 |

**Interpretation, fixed before the run.**
- Lesion-shuffle passing shows the model uses the lesion maps. It does not show they improve grading.
- All five passing: the route stays open for further investigation.
- Any one failing: Architecture 1 is closed as a downstream route for this pipeline.
- No criterion is a superiority test. One seed is not statistically conclusive.

**Code changes for this run (no model, data or training-loop change).**
- `arch1_train.py`: `config.json` records `ema`; BEST/LAST checkpoint metadata include epoch and monitor value; `write_verdict` writes `verdict.json` / `verdict.md` (criteria, P-42 comparison, identity, history, statements).
- `architecture1_training.ipynb`: pins the approved model SHA and the §50 C2 results file; `ACKNOWLEDGE_C2_FAIL = True`; asserts the split, population and P-42 validation ids; uses the base Colab setup (no TJDR / IDRiD / Stage-4 weight checks); writes the verdict.


## 52. Architecture 1 — extended to three sequential seeds (42 → 123 → 2026): pre-run record (2026-10-02; written before any Architecture-1 training)

**What changed.** The §51 seed-42 run becomes a three-seed sequential evaluation: seeds 42, 123, 2026, in that order, in one notebook. Only the orchestration changed. When this was written no Architecture-1 run existed (`gdrive:DiabeticRetinopathy/experiments/Architecture1` did not exist), so the plan is fixed before any result.

**Unchanged.** Architecture, prior encoder, gated injection, CORN, optimizer, learning rate and schedule, augmentation, batch size, BEST-by-val-QWK selection, split, bundle, Stage-3 and Stage-4 models, the lesion-shuffle protocol (one fixed derangement, permutation seed 20261001, the same for every seed), no EMA. C2 is not rerun. IDRiD is not read. Nothing is tuned between seeds, and no seed's result changes another seed.

**Per-seed closure criteria — the §51 five, against P-42 BEST, for every seed.**

| criterion | P-42 | each seed must reach |
|---|---|---|
| QWK | 0.9184 | ≥ 0.8984 |
| AUROC (≥3; grade 4 vs 0–2) | 0.9588 | ≥ 0.9488 |
| lesion-shuffle QWK drop | — | ≥ 0.01 |
| grade-3 recall | 0.6154 | ≥ 0.5154 |
| false-urgent rate | 0.0379 | ≤ 0.0579 |

Seeds 123 and 2026 are judged against the same P-42 values, as specified; the matched-seed P runs (P-123, P-2026) are not used as references.

**Route-level reading over the three seeds, fixed before the run.** This replaces §51's single-seed rule ("any one failing closes the route"), which was written when only seed 42 was planned.
- All three seeds pass all five checks: the route remains viable, and lesion-map use is supported by the shuffle tests.
- Every seed fails the checks: Architecture 1 is closed.
- Mixed: the inconsistency is reported; nothing is tuned and no seed is rerun.
- This is not a pre-registered superiority experiment. No superiority threshold exists, and none is claimed from the means.

**Orchestration (`arch1_train.py`).**
- `run_sequence` runs `SEEDS = (42, 123, 2026)` in order. Run directories: `experiments/Architecture1/arch1_cb5fc7a8d370_seed{42,123,2026}/`.
- Fresh initialisation: each seed's model is built inside `train_seed` from the frozen pretrained ConvNeXt weights and that seed's own initialisation; nothing learned passes between seeds. `initialization.json` records the hash of the weights each seed started from.
- A run directory is bound to its seed: `config.json` carries the seed in its hash, and a directory with another configuration is refused.
- Resume: rerunning keeps completed seeds as recorded, resumes the interrupted seed from its own checkpoint, and starts later seeds fresh. A fresh lock left by a dead runtime is waited out (as in the P/PL notebook) instead of failing.
- Failure: any exception stops the sequence; no later seed starts and no summary is written. A seed that returns without a stop decision is an error, not a skip.
- Per seed: `config.json`, `initialization.json`, checkpoints (BEST and LAST), `history/`, `metrics/`, `verdict.json`, `verdict.md`.
- Summary, only after all three verdicts exist: `experiments/Architecture1/arch1_cb5fc7a8d370_3seed_summary/summary.{json,md}` — per-seed values, mean, SD (n−1), per-seed pass/fail, seeds passing each check, shuffle results, and the route reading above.

**Tests (CPU; no real training).** Seed order and separate directories; fresh initialisation (a model built after another seed has trained equals a pristine build; the pretrained reference is never modified); resume of the real loop from its checkpoint without re-initialisation; a failed or unfinished seed stops the sequence with no summary; aggregate mean / SD / pass counts and the three route readings.
