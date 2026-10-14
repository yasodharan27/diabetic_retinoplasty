# Next-Step Audit — the Grade-3/Grade-4 bottleneck after §13, §14 and the pending T-vs-P experiment

Written 2026-09-26. Read-only audit of the research record (§1–§14.1), the code (`corn.py`,
`weighted_corn.py`, `task_weighted_corn.py`, `multiseed_runs.py`, the aux and T-vs-P notebooks),
the duplicate audit, the IDRiD external-evaluation protocol, and this folder's earlier documents.
Nothing was trained or run. **The task-3 vs task-1 placebo experiment is running and its outcome is
unknown; nothing here assumes a result.**

Evidence labels: **ESTABLISHED** (shown by a saved experiment, by the code, or by mathematics),
**OBSERVATIONAL** (a real measurement, not a controlled comparison), **HYPOTHESIS**.

---

## 1. Evidence table

| # | Hypothesis | Intervention | Control | Primary endpoint | Result | Pre-reg.? | Evidence type | Conclusion | Status | Unanswered |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | RACAF's reliability gate improves grading | RACAF fusion | matched NO_RACAF, 3 seeds | paired ΔQWK | −0.0072, 1/3 seeds +, all CIs span 0 | yes | causal (controlled) | no effect shown | **closed** | — |
| E2 | the gate (not Ĝ) carries RACAF's effect | C1 constant gate | RACAF, NO_RACAF (seed 42) | BEST QWK | ±0.0025 | yes (1 seed) | causal, underpowered | gray zone | **closed** | — |
| E3 | per-lesion κ fusion helps | C3 residual κ | NO_RACAF (seed 42) | BEST QWK | +0.0018 despite engaged mechanism | yes (screen) | causal, 1 seed | no effect | **closed** | — |
| E4 | reliability predicts grading error | D6 selective prediction | classifier confidence | AUROC(r) for error | 0.39 mean, 0/6 > 0.5; incr. +0.0009 | yes | observational | negative | **closed** | — |
| E5 | lesion channels carry image-specific signal | permutation at inference | vessel permutation | ΔQWK | lesion 0.63–0.74 vs vessel 0.04–0.07 | post-hoc | observational (off-manifold) | lesion maps matter | established | whether *segmentation quality* matters |
| E6 | generic vessel map carries PDR information | read-only probe | lesion probe | OOF AUROC 4 vs 3 | vessel 0.572 (CI incl. 0.5); increment −0.028 | yes | observational | not supportive | **closed** | NV is not represented at all |
| E7 | grade-4 info lost by the 0.5 decode rule | threshold sweep / nested | fixed 0.5 rule | grade-4 recall | 0.19 → 0.60 (OOF), but grade-3 recall −61 %, precision 0.50 → 0.35 | yes | observational (in-sample + nested CV) | calibration dial, not a fix | **closed as solution** | — |
| E8 | `p_gt_3` under-uses lesion information | score comparison | lesion-stat probe (in-val CV) | 4-vs-3 AUROC | `p_gt_3` 0.598 vs lesion 0.670 | yes | observational | superseded by E9–E11 | refined | — |
| E9 | lesion burden is learnable from TRAIN (Architecture A gate) | train-fitted LR | pre-registered bar 0.62 | val AUROC | 0.6202, CI [0.500, 0.741] | yes | observational | **INCONCLUSIVE**; §11 was optimistic | A **not cleared** | — |
| E10 | signal lost in the marginal `p_gt_3` | read-only, 6 runs | `p_gt_3` | AUROC `p_cond_3` − `p_gt_3` | +0.079, 5/6 + | yes | observational | SPECIFIC_LOSS_IN_MARGINAL | established | — |
| E11 | where is the signal lost? | linear probes on frozen features | lesion stats | train-fit / val AUROC | GAP(L) 0.758 > E 0.710 > 4 logits 0.660 > `p_cond_3` 0.639 > `p_gt_3` 0.578 | partly (probes descriptive) | observational, post-hoc | loss is downstream of E | established (descriptive) | whether a head can recover it |
| E12 | marginal grade-4 aux BCE recovers `p_gt_3` | aux λ ∈ {0.05, 0.10, 0.20} | frozen B | Δ AUROC `p_gt_3` 4 vs 3 | −0.018 / −0.011 / +0.006 | yes | causal (controlled) | **NOT_SUPPORTIVE** | **closed** | — |
| E13 | aux raises `p_cond_3` | same runs, secondary | same B (reused ×3) | Δ `p_cond_3` | +0.053 / +0.053 / +0.028, 9/9 + | no (post-hoc) | **confounded**: RACAF also beats B by +0.075 | not causal | hypothesis only | is the lift specific? |
| E14 | task-3 pressure specifically raises `p_cond_3` | T: task 3 × 2.0 | **P: task 1 × 1.3792 (mass-matched)**, B | Δ AUROC `p_cond_3` (T − P) | **pending** | yes | causal, specificity-controlled | — | **running** | everything in NEXT_STEP_DECISION_TREE.md |
| E15 | APTOS labels are noisy at the severe end | byte-identical duplicate audit (existing) | — | label agreement | 8/41 pairs disagree; 6/12 pairs touching grade 3/4 disagree | no (existing audit, re-tabulated here) | observational | under-examined | **open** | noise rate on the whole set; effect on every 3-vs-4 number |
| E16 | the model generalises externally | IDRiD test-set evaluation | — | QWK etc. | **not run** (protocol written for the RACAF model only) | yes | — | — | pending | nothing external exists for 3-vs-4 |

---

## 2. The eight mechanisms, kept separate

| # | Mechanism | Evidence for | Evidence against | Status |
|---|---|---|---|---|
| 1 | **Representation** lacks the 3/4 signal | Stage 06 (RGB Swin) probe only 0.585 | Stage 05 GAP(L) 0.758, E 0.710 exceed the handcrafted lesion statistics (0.620) | not the primary bottleneck (descriptive, linear probes) |
| 2 | **Head capacity / optimisation** | a train-fitted linear readout of frozen E (0.710) beats the jointly trained task-3 head (0.639); BEST is selected by val_QWK, which barely weighs 3↔4 | capacity is not the issue (a *linear* PCA-16 probe suffices); the gap is post-hoc and within ~1 SE | open, plausible |
| 3 | **Ordinal loss weighting** | E13 direction | E13 confounded (RACAF lift); E12 null on its endpoint | **pending E14** |
| 4 | **CORN conditional decomposition** | task 3 is fit on only 390 training images | the decomposition is a proper likelihood (§3 below); not a defect | expected property |
| 5 | **Decode / estimand mismatch** | E10 (+0.079, 5/6) and all 9 aux runs; mathematics (§3) | — | **established** |
| 6 | **PDR pushed below grade 3** | `p_gt_2` ranks grade 4 below grade 3 in 14/15 models; 21–37 of 58 PDR images decoded ≤ 2 in the NO_RACAF baselines (mean 55 %) | cause unknown | established observation, **cause open** |
| 7 | **Label noise** | byte-identical conflicts: 3↔4, 2↔3, 1↔4, 2↔4, 4↔2 ×2 (§5) | small n | **open, under-examined** |
| 8 | **Missing PDR-specific features** | no NV / IRMA / venous beading / VH / fibrous-proliferation / laser-scar channel exists; lesion burden falls at PDR (Esmaeilkhanian 2023) | 4-vs-rest ranking is already 0.88 | established by construction; consequence unquantified |

Mechanisms 6, 7 and 8 compete to explain the same observation (PDR scored as ≤ 2). **Nothing so far
distinguishes them.** That is the most important unresolved point in the whole programme.

---

## 3. The CORN formulation, audited

**Model.** `z = W E + b`, W ∈ ℝ^{4×256} (one row per task, no weight sharing between rows).
`p_cond_k = σ(z_k) = P(y > k | y ≥ k, x)`; `P(y > k | x) = Π_{j≤k} σ(z_j)`.

**What CORN is, formally.** This is the **sequential (continuation-ratio) ordinal model** (Tutz 1991;
taxonomy in Tutz 2022) with **category-specific slopes** on a learned representation. It is not a
cumulative-link model and not CORAL. Its log-likelihood factorises exactly:

`log P(y | x) = Σ_{k<y} log σ(z_k) + 1[y < 4] · log(1 − σ(z_y))`

— every term is the binary cross-entropy of task k on the subset `y ≥ k`. **`corn.corn_loss` is the
negative log-likelihood of this model, pooled over included pairs**; `weighted_corn` is a
class-weighted likelihood; `task_weighted_corn` re-weights individual factors.

**The 3-vs-4 estimand.** `P(y = 4 | x, y ∈ {3,4}) = P(y=4|x) / P(y≥3|x) = p_cum_3 / p_cum_2 = σ(z_3)`.
`p_cond_3` is the model's own posterior for this contrast; `p_gt_3` is its posterior for 4 vs all.
Ranking true grade-3/4 images by `p_gt_3` equals ranking by `p_gt_2 · p_cond_3`, which is harmless only
if `p_gt_2` is constant on those images. It is not: it is lower for grade 4 than grade 3 in 14/15
models (AUROC 0.40–0.50).

| Candidate explanation | Verdict | Basis |
|---|---|---|
| implementation defect | **no** | loss, decode and `p_cum` identity verified by tests (`tests/test_corn.py`, `test_weighted_corn.py`, `test_task_weighted_corn.py`); Phase-0 reproduced saved logits to ≤ 7.8e-3 |
| expected property of CORN | **yes** | the grade-4 marginal depends on all earlier conditionals by construction |
| estimand mismatch | **yes** (the §12 endpoint) | `p_gt_3` answers 4-vs-rest, not 4-vs-3 |
| training-objective mismatch | **no, in the strict sense** | the loss is the correct likelihood; the objective is fine, the *fit* of tasks 0–2 on PDR images is poor |
| calibration problem | **yes, for 4-vs-rest decisions** | ranking 0.88 but 0.5-rule recall 0.19 (E7) |
| combination | **yes** | the defect the evidence isolates is that tasks 0–2 score PDR images low — which is mechanism 6, whose cause is 7 and/or 8 |

**Comparison with other ordinal families — does any address the observed failure?**

| Family | Structure | Addresses "PDR scored low by tasks 0–2"? | Addresses 3-vs-4? |
|---|---|---|---|
| **CORN** (current) | sequential, task-specific slopes, rank-consistent | — | yes, via `p_cond_3` |
| CORAL (Cao et al. 2020) | shared weight vector, per-threshold bias | **worse**: one direction cannot reverse sign between thresholds, and burden reverses at 3→4 | worse |
| Cumulative-link / proportional odds (McCullagh 1980; Vargas et al. 2020) | one latent projection, ordered cut-points | **worse**, same reason | worse |
| Unimodal constraints (Beckham & Pal 2017) | Poisson/binomial output | no; imposes unimodality, unrelated to the failure | no |
| Soft ordinal labels / CDW-CE / kappa loss (Díaz & Marathe 2019; Polat et al. 2025; de la Torre et al. 2018) | distance-aware losses | no; make adjacent (3↔4) errors *cheaper* | no |
| Independent binary thresholds (Frank & Hall 2001; Niu et al. 2016) | each P(y>k) fit on all data, not chained | **partly**: P(y>3) is learned directly as 4-vs-rest, so PDR is not gated by tasks 0–2 — but rank consistency is lost, and 4-vs-rest ranking is already 0.88 (the problem is calibration, E7) | no better than `p_cond_3` |
| Hierarchical classification (e.g. CF-DRNet, Wu et al. 2020) | coarse-to-fine | CORN is already a conditional hierarchy | same as CORN |
| Nominal softmax | no order | grade-4 logit independent of the chain; loses ordinal structure; no DR evidence of better severe-end performance | no evidence |
| Hybrid ordinal + nominal heads (published for DR, e.g. Applied Intelligence 2023) | two heads | as nominal, plus extra parameters | no evidence |

**Conclusion:** no alternative family specifically addresses the observed failure better than CORN
read through the correct estimand. The families that impose a single latent direction (CORAL, CLM)
would be *expected* to do worse given the non-monotonic burden. Replacing CORN is **currently
unjustified**.

---

## 4. The dominant clinical failure: PDR decoded ≤ 2

In the NO_RACAF baselines, 37 / 37 / 21 of 58 validation PDR images are decoded as grade ≤ 2 (seeds
42 / 123 / 2026; mean **54.6 %**). Only 15–18 are decoded as grade 3. So the bulk of the clinical
failure is **not** a 3-vs-4 confusion; it happens at thresholds 0–2.

**Should the programme target PDR-vs-non-PDR instead of 3-vs-4?** The 4-vs-rest ranking is already
AUROC ≈ 0.88 (E7). Its failure is a decision/calibration failure (closed as a threshold solution) or
a fit failure on a subset of PDR images. Which images, and why, is unknown. A PDR-vs-non-PDR
*architecture* is justified only if that subset is genuine PDR lacking representable features. It is
**not** justified if they are label disagreements or treated/regressed eyes.

**What the existing pipeline can represent.** Stage 04 channels: MA, HE, EX, SE only (IDRiD Dice
0.017 / 0.127 / 0.357 / 0.024). None of NVD, NVE, IRMA, venous beading, preretinal or vitreous
haemorrhage, fibrous proliferation or laser scars has a channel.

**Datasets that could supply PDR-specific annotation (none acquired; none needed yet):**

| Dataset | PDR-relevant labels | Suitability |
|---|---|---|
| FGADR Seg-set (Zhou et al. 2021) | pixel masks incl. **IRMA, NV** (1,842 images) | best public NV/IRMA source; domain shift to APTOS |
| Retinal-Lesions (Wei et al. 2020) | **NV, fibrous proliferation, preretinal + vitreous haemorrhage** (1,593 images; only 62 DR4) | small PDR count |
| DDR (Li et al. 2019) | 13,673 graded images (7 graders + consensus); 757 lesion masks — MA/HE/EX/SE only | large **external grading** set; no NV masks |
| IDRiD (Porwal et al. 2018) | 516 graded images; MA/HE/EX/SE/OD masks | already used by Stage 04; see contamination note |
| RFMiD (Pachade et al. 2021) | image-level multi-disease labels incl. **laser scars** (3,200 images; 2-expert consensus) | could audit the laser-scar confound, not PDR grading |

**Contamination rules if any external set is ever used.** Training a new segmenter on FGADR /
Retinal-Lesions does not touch the APTOS split. **IDRiD must not be used both to train anything and
as the external test.** Stage 04 was trained on the IDRiD segmentation subset. The IDRiD protocol
verified 0/81 overlap with the grading *test* set, but the grading *train* set has not been checked.

---

## 5. Label-noise audit (existing evidence, re-tabulated)

The 41 train/validation duplicates were confirmed **byte-identical** (size prefilter → MD5 → full
byte compare, reproduced twice). Any grade disagreement within a pair is therefore genuine label
noise on the *same image*.

| Quantity | Count | Rate (95 % Clopper–Pearson) |
|---|---|---|
| pairs with different grades | 8 / 41 | 19.5 % (8.8–34.9 %) |
| pairs touching grade 3 or 4 that disagree | 6 / 12 | 50 % (21–79 %) |
| pairs containing a grade-4 label whose twin is graded ≤ 2 | 4 / 11 | 36 % (11–69 %) |
| validation grade-3/4 images whose twin disagrees | 3 / 9 | 33 % (7.5–70 %) |

The conflicting pairs are (validation grade, training twin grade): **3↔4**, **2↔3**, **1↔4**,
**2↔4**, **4↔2** (×2), plus 0↔1 and 2↔1. (§13 counted only the three with a grade-3/4 *validation*
image; including the training side, **six** conflicting pairs touch grade 3 or 4.)

**How much could this constrain the 3-vs-4 AUROC?**

- *Direct effect of the three disputed validation images:* each grade-4 image enters 39 of the 2,262
  grade-3×grade-4 pairs, each grade-3 image 58. Together they can move any AUROC by at most
  **±0.060**.
- *Ceiling if the disagreement rate generalises (illustrative, not an estimate).* With a fraction `a`
  of "4" labels truly 3, and `b` of "3" labels truly 4, a classifier with true AUROC A is observed at
  `(1−a)(1−b)A + ½[(1−a)b + a(1−b)] + ab(1−A)`:

| symmetric flip rate | observed AUROC of a *perfect* model | observed AUROC of a model with true A = 0.80 |
|---|---|---|
| 10 % | 0.90 | 0.74 |
| 20 % | 0.80 | 0.68 |
| 30 % | 0.70 | 0.62 |

  If instead a fraction c of "4" labels are images another grader called ≤ 2 (4/11 of grade-4 pairs
  here), and the model scores them below grade 3, the ceiling is ≈ 1 − c (c = 0.36 → 0.64).
- Every observed 3-vs-4 number (0.58–0.76 across all readouts) lies **inside** the range these
  scenarios produce. **Label noise is sufficient to explain the plateau.** It is not shown to be the
  explanation: n is small and the duplicate subset may not be representative.

The **read-only label-noise audit** that would quantify this is specified in
NEXT_STEP_RECOMMENDATION.md (Step LN). It changes no label.

---

## 6. Research-discipline state

- The same 97 APTOS validation grade-3/4 images have been scored in every analysis from §8 to §14.1 and
  will be again in E14. They are **no longer an untouched test set** for this question. Any positive
  finding needs an external confirmation.
- IDRiD is still untouched for classification. The external protocol exists but targets the RACAF
  model only, and was never run. It must stay a one-shot confirmatory set: no selection, no tuning,
  no repeated use for decisions.
- One mechanism per experiment has been respected (E12 λ, E14 β). This must continue.
