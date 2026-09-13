# RACAF vs NO-RACAF — Post-Run Ablation Analysis

**Reference:** `2026-09-12_02-45-05` (finalized RACAF model) · **Ablation:** `2026-09-13_04-11-07`
(NO-RACAF control) · Analysis date 2026-09-13.

Every number below was read from the saved, executed outputs of
[stage08_corn_classifier_racaf_ablation.ipynb](../../colab/notebooks/stage08_corn_classifier_racaf_ablation.ipynb)
and [stage08_corn_classifier.ipynb](../../colab/notebooks/stage08_corn_classifier.ipynb), including
their embedded diagnostic plots. Values not present in either notebook are marked **not available**.
Where this document computes something new, the computation is described and labelled as such.

The finalized RACAF notebook, its report and the NO-RACAF results are unmodified.

---

## 1. Executive summary

**Verdict: D — under this experiment, the NO-RACAF control performed better than the finalized
RACAF model on the pre-registered primary metric.** This is reported plainly, with the qualifiers
that determine how far it can be trusted.

| | RACAF | NO-RACAF | Δ (NO − RACAF) |
|---|---|---|---|
| **BEST val_QWK** (primary) | 0.7632750 | **0.8461052** | **+0.0828302** (+10.85%) |
| QWK bootstrap 95% CI (recorded) | [0.7230, 0.8023] | [0.8175, 0.8732] | intervals do not overlap |
| BEST val_loss (recorded) | **0.3024397** | 0.6215262 | +0.3190865 |
| Balanced accuracy | 0.489899 | **0.603883** | +0.113984 |
| Grade 3 / grade 4 recall | 0.2308 / 0.0690 | **0.5385 / 0.2414** | +0.3077 / +0.1724 |
| ECE (decode) | **0.0524** | 0.1652 | +0.1128 |

What supports the result: a large QWK gap; non-overlapping recorded confidence intervals; a
conservative bootstrap of the difference that excludes zero; an advantage visible from epoch 8 and in
26 of 30 matched epochs, not only at a noisy peak; and gains concentrated where the reference was
weakest — severe-grade recognition and large ordinal errors.

What limits it: **one run per arm with unseeded weight initialisation**; and **RACAF is better
calibrated and has a far lower validation loss**, because NO-RACAF heavily overfit the training set.
The **41 train/validation duplicate images have since been excluded exactly** — see
[RACAF_Duplicate_Contamination_Audit.md](RACAF_Duplicate_Contamination_Audit.md), which reads both
experiments' per-image prediction CSVs directly: the exact duplicate-excluded ΔQWK is **+0.0828904**
(n = 689, paired bootstrap 95% CI [+0.0453, +0.1232]), essentially identical to the all-image
+0.0828302 — the advantage does not depend on, and is not inflated by, the duplicates.

**What can be claimed:** under these controlled conditions, removing RACAF changed BEST val_QWK from
0.7633 to 0.8461, on both the full validation set and the exact duplicate-excluded subset.
**What cannot:** that RACAF is harmful in general, that NO-RACAF generalizes better,
or that the gap is architectural rather than partly run-to-run variance. RACAF's contribution as an
innovation is **not supported** by this experiment.

**Recommended next step:** a multi-seed RACAF vs NO-RACAF repeat before any architectural claim.
Proceed with IDRiD external evaluation, reporting RACAF as non-superior. No redesign or retuning is
justified by this result.

---

## 2. Experimental setup

Verified from the executed ablation notebook:

| Control | Evidence in the notebook | Status |
|---|---|---|
| RACAF absent from the forward path | `[7]`: "NO racaf_fusion layer", "NO trainable variable belongs to RACAF", "NO reliability_gate / global_projection variable" — all PASS | **verified** |
| Reliability does not alter logits | `[7]`: exactly one inert connection, **no weights**, r=0 vs r=1 **max \|diff\| = 0.0** | **verified** |
| Parameter delta is exactly RACAF | 43,338,506 → **43,043,336**; 409 → **405**; Δ **295,170 / 4**, equal to RACAF alone | **verified** |
| Stage 05, Stage 06, Stage 07, CORN present | `[7]` PASS; per-layer counts identical in the pre-run audit | **verified** |
| Stage 03/04 frozen, cache unchanged | `[6]`: 3,651 fully local, 11 empty-FOV; `[D13]` cache inventory unchanged | **verified** |
| Split and preprocessing | `[5]`/`[D2]`: split SHA-256 `f512a7a0…`, seed 42, fraction 0.2; 2929/733; 730 evaluated | **verified** |
| Loss / optimizer / precision | `joint_corn_loss`; `LossScaleOptimizer(inner=Adam)`; `mixed_float16` | **verified** |
| LR, batch, epochs, callbacks | `[8]`/`[P2]`/`[R1]`: LR 1e-4, batch 2, cap 50, ES patience 12, RLROP 4 / 0.5 / 1e-6, monitor `val_QWK` max | **verified** |
| Config identity | hash `adf02bb3f8aa6ab93843036bad6b339e` in metadata and every checkpoint | **verified** |
| Seed parity, no global init seed | `[7]` SEED PARITY PASS; "global weight-init seed NOT SET, exactly as in the reference run" | **verified** |
| No IDRiD data in training | training entries are the APTOS split only (`[5]`, `[9]`, `[D2]`) | **verified** |
| Runtime | Python 3.13.15, TF 2.20.0, Tesla T4, CUDA 12.5.1 (`[2]`) | recorded |

**Deviations found:**

1. **`[P2]` was edited during the run.** The executed notebook differs from the committed ablation
   notebook in `[P2]` (besides the expected `[S]` post-run switches): its gate checks were changed
   from the reference's hard-coded `409 trainable tensors` / `43,338,506 parameters` to `405` /
   `43,043,336`. The committed notebook would have refused to launch. This edit changes a launch-gate
   constant only — not model, data or training behaviour — and is the correct value. It is recorded
   because it is a divergence from committed code and was missed when the ablation notebook was built.
2. **In-memory weight restore at the end of the final training session.** `[T]` printed "Restoring
   model weights from the end of the best epoch: 43." — Keras EarlyStopping's session-local best in
   the resumed final session, not the global best (epoch 41). This affected only the in-memory model
   after `fit()`. All evaluation reloaded `checkpoints/best` from disk with SHA-256 verification
   (`[R2]`), and `[D13]` reproduced the per-sample predictions from the archive, so **no reported
   result is affected**.
3. **Training length differed.** RACAF stopped at epoch 30 by EarlyStopping; NO-RACAF reached the
   50-epoch cap (EarlyStopping wait 9/12). Same protocol, different trajectory — analysed in §6.

No deviation affecting the controlled architectural difference was found.

---

## 3. RACAF reference result

`2026-09-12_02-45-05`: BEST **epoch 18**, val_QWK **0.7632750272750854**, val_loss
**0.30243971943855286** (recorded; recomputed 0.302438), 30 epochs (EarlyStopping), LAST epoch 30
val_QWK 0.7192390 / val_loss 0.3549938. 43,338,506 trainable parameters in 409 tensors. Archive
SHA-256 `7b780c230e4ce83ae9e9b1c0bdb9ae73c707b262e5a9c759748343f9417b8b9a`. QWK bootstrap 95% CI
[0.7230, 0.8023].

## 4. NO-RACAF result

| Item | Value |
|---|---|
| Experiment | `experiments/FinalClassification/2026-09-13_04-11-07` |
| Archive | `exported_models/FinalClassification/2026-09-13_04-11-07_NO_RACAF_BEST` |
| Archive SHA-256 | `61619656c0d66995f3e6668cdb30d21b080513faad4354f4913ad98fe5d2e773` |
| Git commit | `f9dbc52647ffddd0aecdf92aacd9330b2d967623` |
| Epochs completed | 50 (cap reached; EarlyStopping wait 9/12) |
| **BEST epoch** | **41** (`best/` from `gen_00041`) |
| **BEST val_QWK** | **0.8461052179336548** (recomputed 0.846105; sklearn 0.846105) |
| **BEST val_loss** | **0.6215262413024902** recorded; 0.621483 recomputed |
| LAST (epoch 50) val_QWK / val_loss | 0.8318959474563599 / 0.7943733334541321 |
| QWK bootstrap 95% CI | [0.8175, 0.8732] |
| Accuracy / balanced accuracy | 0.763014 / 0.603883 |
| Macro F1 / weighted F1 | 0.585919 / 0.763182 |
| MAE | 0.310959 |
| Evaluated / excluded | 730 / 3 (`262ad704319c`, `26453eb7e989`, `3a122851e526` — empty field of view) |
| Ground-truth histogram | [361, 74, 198, 39, 58] |
| Prediction histogram | [354, 93, 193, 63, 27] — **all five grades predicted** |
| Optimizer iterations | 73,020 of 73,050 (30 loss-scale skips) |
| Technical validity | `[R1]`–`[R4]` all PASS; `[D2]`–`[D13]` **62/62** PASS |

Full training history (train loss, train QWK, val_loss, val_QWK, LR for all 50 epochs) is in `[R1]`;
§6 summarises it.

---

## 5. Direct comparison table

| Metric | RACAF | NO-RACAF | Δ | Rel. | Better |
|---|---|---|---|---|---|
| Trainable parameters | 43,338,506 | 43,043,336 | −295,170 | −0.68% | — (design) |
| Trainable tensors | 409 | 405 | −4 | — | — (design) |
| BEST epoch | 18 | 41 | +23 | — | — |
| **BEST val_QWK** | 0.7632750 | 0.8461052 | **+0.0828302** | +10.85% | **NO-RACAF** |
| BEST val_loss (recorded) | 0.3024397 | 0.6215262 | +0.3190865 | +105.50% | **RACAF** |
| LAST val_QWK | 0.7192390 | 0.8318959 | +0.1126570 | +15.66% | NO-RACAF |
| LAST val_loss | 0.3549938 | 0.7943733 | +0.4393796 | +123.77% | RACAF |
| Accuracy | 0.738356 | 0.763014 | +0.024658 | +3.34% | NO-RACAF |
| Balanced accuracy | 0.489899 | 0.603883 | +0.113984 | +23.27% | NO-RACAF |
| Macro F1 | 0.484207 | 0.585919 | +0.101712 | +21.01% | NO-RACAF |
| Weighted F1 | 0.718715 | 0.763182 | +0.044467 | +6.19% | NO-RACAF |
| MAE | 0.380822 | 0.310959 | −0.069863 | −18.35% | NO-RACAF |
| Brier | 0.342009 | 0.402815 | +0.060806 | +17.78% | **RACAF** |
| ECE (decode) | 0.052427 | 0.165210 | +0.112783 | +215% | **RACAF** |
| Errors ≥ 2 grades | 67 | 47 | −20 | — | NO-RACAF |
| Max error distance | 4 | 3 | −1 | — | NO-RACAF |
| Evaluated / excluded | 730 / 3 | 730 / 3 | identical | — | — |

Per-class:

| Grade (support) | Recall RACAF → NO | F1 RACAF → NO | TP |
|---|---|---|---|
| 0 (361) | 0.9446 → 0.9446 (0) | 0.9407 → 0.9538 | 341 → 341 |
| 1 (74) | 0.4324 → **0.6081** | 0.4444 → 0.5389 | 32 → 45 |
| 2 (198) | **0.7727** → 0.6869 | 0.6892 → 0.6957 | 153 → 136 |
| 3 (39) | 0.2308 → **0.5385** | 0.2308 → 0.4118 | 9 → 21 |
| 4 (58) | 0.0690 → **0.2414** | 0.1159 → 0.3294 | 4 → 14 |

---

## 6. Training dynamics

From the logged histories (`[R1]` in both notebooks):

- **Faster and higher.** NO-RACAF's val_QWK first exceeded RACAF's best (0.7633) at **epoch 8**
  (0.7649), and was higher in **26 of the 30 matched epochs** — RACAF led only at epochs 1, 2, 3 and 5.
  Mean val_QWK: epochs 1–10 RACAF 0.6006 vs 0.6812; 11–20 0.7359 vs 0.7732; 21–30 0.7255 vs 0.8155.
- **Not a single noisy peak.** Within RACAF's own 30-epoch horizon NO-RACAF already reached 0.8340
  (epoch 28). Over epochs 31–50 its val_QWK averaged 0.8291 (SD 0.0134, minimum 0.7877), and even its
  LAST checkpoint (0.8319) exceeds RACAF's BEST by 0.069. The advantage therefore **does not depend
  on the 20 extra epochs** or on the epoch-41 peak (+0.017 above the late plateau).
- **Heavy overfitting in NO-RACAF.** Training loss fell to 0.0118 (0.0052 at epoch 47) with training
  QWK 0.997; val_loss rose from its minimum 0.3277 (epoch 13) to 0.7944. RACAF's training loss ended
  at 0.1891 and its val_loss minimum coincided with its best QWK (epoch 18).
- **Loss and QWK decoupled in NO-RACAF.** Correlation of val_loss with val_QWK over the run was
  +0.474 for NO-RACAF (both rising) vs −0.867 for RACAF (moving together as expected). NO-RACAF kept
  improving its ordinal decisions while its probabilities became overconfident — which inflates the
  CORN loss on the errors it still makes.
- **LR schedule diverged as an outcome, not a setting.** Both used identical callbacks. RACAF
  plateaued, so ReduceLROnPlateau fired at epochs 8, 12, 18, 23, 27. NO-RACAF kept improving and
  stayed at 2.5e-5 for 24 epochs (13–36) before reductions at 37, 41, 46, 50.
- **Stopping.** EarlyStopping stopped RACAF (patience-12 replay confirmed); NO-RACAF was stopped by
  the cap. At each model's own minimum-val_loss epoch the comparison is RACAF 0.3024 / QWK 0.7633 vs
  NO-RACAF 0.3277 / QWK 0.7829 — so the loss gap at the BEST checkpoint is largely a consequence of
  NO-RACAF training on well past its loss minimum.

---

## 7. Confusion-matrix analysis

| | RACAF BEST (ep 18) | NO-RACAF BEST (ep 41) |
|---|---|---|
| true 0 | 341 · 6 · 6 · 8 · 0 | 341 · 15 · 3 · 2 · 0 |
| true 1 | 13 · 32 · 28 · 1 · 0 | 9 · 45 · 19 · 1 · 0 |
| true 2 | 6 · 27 · 153 · 10 · 2 | 4 · 27 · 136 · 22 · 9 |
| true 3 | 1 · 0 · 24 · 9 · 5 | 0 · 1 · 13 · 21 · 4 |
| true 4 | 3 · 5 · 35 · 11 · 4 | 0 · 5 · 22 · 17 · 14 |

(columns: predicted 0 · 1 · 2 · 3 · 4)

**Severe-grade compression toward grade 2 is substantially reduced, not removed.** True grade 3
predicted as 2: 24/39 → 13/39. True grade 4 predicted as 2: 35/58 → 22/58. All severe cases graded
≤ 2: **68 → 41**. The row-normalised plot shows grade 4 still spread across 2 (0.38), 3 (0.29) and 4
(0.24) — the weakest class remains grade 4.

**Part of the compression is traded for over-grading.** True grades 0–2 predicted as 3 or 4: 21 → 34,
driven by grade 2 (2→3: 10 → 22; 2→4: 2 → 9). Grade-2 recall falls 0.7727 → 0.6869, and grade-3
precision is only 0.3333 (63 grade-3 predictions for 39 true cases). The `[D8]` grade 2 → 3 panel shows
images with extensive exudates, predicted grade 3 at confidence 0.82–0.98.

**Large ordinal errors fall.** Error distance [0,1,2,3,4]: RACAF [539, 124, 50, 14, 3] vs NO-RACAF
[557, 126, 40, 7, 0]. The three grade 4 → 0 errors are gone; distance-3 errors halve. Grade 0 is
unchanged in recall (341), with fewer far errors (0→3: 8 → 2) but more adjacent ones (0→1: 6 → 15).

**Where the QWK gain comes from** (computed from the matrices). QWK = 1 − O/E, with O the
squared-distance-weighted disagreement. O fell 31.125 → 21.812. By distance: d=1 7.750 → 7.875,
d=2 12.500 → 10.000, **d=3 7.875 → 3.938, d=4 3.000 → 0.000**. By true grade: grades 3–4 contributed
17.625 → 10.688, grades 0–2 13.500 → 11.125. **About three quarters of the reduction in weighted
disagreement comes from true grades 3–4**, and almost all of it from errors of distance ≥ 2. The QWK
advantage is driven by the clinically more important severe grades and by fewer gross ordinal errors,
not by a change in the common grade-0 class. (E also rose 131.48 → 141.74 because the prediction
histogram is closer to the true one, which contributes too.)

---

## 8. Per-class and class-imbalance analysis

The validation population is half grade 0 (361/730) with only 74 grade 1, 39 grade 3 and 58 grade 4.

- **Raw accuracy barely moves (+3.34%)** because grade 0 dominates and its TP count is identical.
- **Balanced accuracy (+23.27%) and macro F1 (+21.01%) move a lot**, because the gains are in the rare
  classes: grade 3 TP 9 → 21, grade 4 TP 4 → 14, grade 1 TP 32 → 45.
- These are small counts. Grade 3's recall change is 12 images; grade 4's is 10. They are consistent
  with the QWK decomposition, but per-class estimates on 39 and 58 images carry wide uncertainty and
  should not be read individually as precise.
- Grade 2 pays for it: 17 fewer correct grade-2 images, mostly re-graded upward.

---

## 9. Duplicate-contamination — exact analysis

**Superseded by an exact per-image audit.** The scenarios below were computed from the recorded
confusion matrices when neither model's per-image predictions were available in the repository.
Both experiments' `evaluation/per_sample_corn_predictions.csv` files have since been read directly
from Drive (read-only, no training). Full detail, methodology, the reproduced 41-ID list with each
model's per-image prediction, and the exact statistics are in
[RACAF_Duplicate_Contamination_Audit.md](RACAF_Duplicate_Contamination_Audit.md). Headline:

| | Value |
|---|---|
| Duplicate images (independently reproduced, byte-size prefilter → MD5 → full byte compare) | 41, identical ID-for-ID to the earlier content-hash audit |
| True-grade distribution | {0: 5, 1: 6, 2: 21, 3: 1, 4: 8} — matches expectation |
| Duplicate-subset QWK (n = 41) | RACAF 0.5208, NO-RACAF 0.6737 (Δ +0.1529) |
| Duplicate-excluded QWK (n = 689) | RACAF 0.7656318, NO-RACAF 0.8485222 (**Δ +0.0828904**) |
| Duplicate-excluded paired bootstrap 95% CI | **[+0.0453, +0.1232]**, P(Δ ≤ 0) = 0.0000 (5000 resamples, seed 20260913) |
| Shift vs all-image ΔQWK (+0.0828302) | **+0.0000602** — negligible |

**Both models perform markedly worse on the 41 duplicates than on the rest of the population**
(accuracy 56% / 66% vs 75% / 77%), consistent with the subset being 51% grade 2 (the most
confusable class) and 22% carrying a training twin with a *different* label than the validation
image's own ground truth — not with either model having memorised the duplicated training images.

**The exact result supersedes every scenario below.** The previously-flagged "adversarial
allocation [that] can erase the gap" does not correspond to the actual per-image data: excluding the
same 41 images from both models leaves the advantage statistically unchanged (95% CI still excludes
zero). The scenarios are retained here only as a record of the sensitivity analysis performed before
the exact CSVs were read.

Original scenarios, computed from the recorded confusion matrices, removing 5/6/21/1/8 images from
the grade-0/1/2/3/4 rows (689 images remain):

| Scenario | RACAF | NO-RACAF | Δ |
|---|---|---|---|
| Duplicates removed proportionally within each true grade | 0.7691 | 0.8507 | +0.0815 |
| Both models had every duplicate correct | 0.7480 | 0.8318 | +0.0837 |
| NO-RACAF memorised them all, RACAF proportional | 0.7691 | 0.8318 | +0.0626 |
| Feasible extremes found by search (per model, independently) | 0.7431 – 0.8474 | 0.8318 – 0.9086 | as low as −0.0156 |

The actual exact-exclusion result (+0.0829) falls within the range these scenarios anticipated, and
close to the "removed proportionally" scenario (+0.0815) — the closest of the four to what the
per-image data show.

---

## 10. Statistical and uncertainty analysis

- **Recorded marginal bootstrap CIs** (`[R4]`, 2000 resamples, seed 20260911): RACAF [0.7230, 0.8023],
  NO-RACAF [0.8175, 0.8732]. They **do not overlap**.
- **Exact paired bootstrap of ΔQWK** (computed from the two per-image CSVs; see
  [RACAF_Duplicate_Contamination_Audit.md](RACAF_Duplicate_Contamination_Audit.md)): 5000 resamples,
  seed 20260913, the **same** resampled indices applied to both models each replicate (as the
  correctness of the two models on the same image is correlated, this is the correct design — not
  the conservative independent-resampling approximation used before the CSVs were available). All
  730 matched images: mean **+0.0819**, 95% CI **[+0.0460, +0.1202]**, P(Δ ≤ 0) = **0.0000** (0/5000
  replicates). As expected, this paired interval is narrower than the earlier conservative unpaired
  approximation ([+0.0355, +0.1331]), and both exclude zero.
- **Exact duplicate-excluded paired bootstrap** (n = 689, same method): mean +0.0828, 95% CI
  **[+0.0453, +0.1232]**, P(Δ ≤ 0) = 0.0000. The advantage is unchanged by excluding the 41
  duplicates (§9).
- **Paired discordance (exact-grade correctness, not QWK):** of 730 images, 488 both models get
  right, 122 both get wrong, 69 only NO-RACAF gets right, 51 only RACAF gets right — net +18 images
  (+2.47pp accuracy) in NO-RACAF's favour. Exact two-sided binomial McNemar test on the 120
  discordant pairs: **p = 0.1203** — not significant at α = 0.05. The QWK advantage is therefore
  driven more by the *size* of ordinal errors (large errors becoming small ones; §7) than by a
  significant net swing in how many images are graded exactly right.
- **What no statistic here addresses:** variance across training runs. The intervals above say the
  gap is unlikely to be an artefact of *which 730 images* were evaluated. They say nothing about how
  much a second seed of either model would move. See §12.
- Both models were selected on this same validation split, so both BEST values are optimistic; the
  selection bias applies to both, but NO-RACAF chose among 50 epochs versus 30.

---

## 11. Interpretation of RACAF's contribution

1. **No evidence that RACAF improves discrimination.** On the primary metric, and on balanced accuracy,
   macro F1, MAE, severe-grade recall and gross ordinal error, the model without RACAF did better.
2. **RACAF's clearest measurable effect here is regularisation-like.** With RACAF, training loss
   stayed higher (0.19 vs 0.01), validation loss stayed low, probabilities stayed calibrated (ECE 0.052
   vs 0.165; mean confidence 0.790 vs 0.926; confidence on errors 0.587 vs 0.834), and training ended
   earlier. The `[D6]` confidence histograms show this directly: NO-RACAF concentrates nearly all
   predictions — including most of its errors — at confidence ≈ 1.0, and its threshold-reliability
   plots show poorly calibrated upper thresholds (P(y>3) predicted 0.043 vs observed 0.080). Whether
   RACAF *causes* this, or this run simply trained differently, cannot be separated with one run each.
3. **The RACAF run's severe-grade compression is not an intrinsic property of the pipeline.** The same
   Stage 05/06/07 + CORN stack without RACAF recognised grades 3 and 4 far better. That reference
   weakness should no longer be attributed to data or task difficulty alone.
4. **Reliability-weighted fusion behaviour is not observable.** Neither notebook records RACAF gate
   values or reliability-conditioned diagnostics, and the ablation by construction has none. Whether
   RACAF's gate learned to down-weight `E` in a way that blurred severe cases is **not available** from
   the saved outputs.
5. **No implementation defect was identified** that explains the gap. The ablation removes exactly
   RACAF's 295,170 parameters; everything else verified identical.

---

## 12. Initialization limitation

The production RACAF run did not fix a global weight-initialisation seed, and the ablation preserved
that condition deliberately (`[7]`: "global weight-init seed NOT SET"). This is **one run per arm**.
The experiment therefore cannot separate RACAF's architectural effect from random initialisation and
training-trajectory variance.

The difference is large — +0.083 QWK, sustained across most of training, with non-overlapping
evaluation-set intervals — which makes it unlikely to be entirely run-to-run noise. But the two runs
also took visibly different trajectories (different LR-reduction timing, 30 vs 50 epochs), and the
size of seed-to-seed variation for this 43M-parameter model on 2,929 images is unknown. **No
definitive causal statement about RACAF is justified until repeated runs exist.**

---

## 13. Scientific verdict

**D — NO-RACAF performs better than RACAF under this experiment.**

- **Magnitude:** +0.0828 QWK (+10.85%); the largest gains are in grades 3 and 4 and in errors of two
  or more grades — practically meaningful for DR grading.
- **Uncertainty:** recorded CIs do not overlap; the exact paired bootstrap Δ interval is
  [+0.0460, +0.1202], P(Δ ≤ 0) = 0.0000. Uncertainty across training runs is unquantified.
- **Duplicates:** exact exclusion computed from both experiments' per-image CSVs
  ([audit](RACAF_Duplicate_Contamination_Audit.md)) — Δ moves from +0.0828302 (n=730) to +0.0828904
  (n=689), a shift of +0.00006. The advantage does not depend on, and is not inflated by, the 41
  duplicates.
- **Class-specific behaviour explains it:** reduced severe-grade compression and fewer gross errors,
  partly offset by grade-2 over-grading.
- **Against NO-RACAF:** worse validation loss (0.6215 vs 0.3024), worse calibration (ECE 0.165 vs
  0.052, Brier 0.403 vs 0.342), strong overfitting. Anyone using predicted probabilities rather than
  decoded grades would prefer RACAF's behaviour.
- **What can be claimed:** under these controlled conditions, removing RACAF changed BEST val_QWK from
  0.7633 to 0.8461, balanced accuracy from 0.4899 to 0.6039, and grade-4 recall from 0.0690 to 0.2414,
  while increasing validation loss from 0.3024 to 0.6215 and ECE from 0.052 to 0.165.
- **What cannot:** that RACAF is a beneficial innovation; that RACAF is harmful in general; that
  NO-RACAF generalizes better beyond this split; any clinical statement.

---

## 14. Recommended next step

1. **~~Exact paired and duplicate-excluded analysis~~ — done.** Both experiments' per-image CSVs
   were read directly from Drive (read-only, no training); see
   [RACAF_Duplicate_Contamination_Audit.md](RACAF_Duplicate_Contamination_Audit.md). This settled §9
   and tightened §10: the exact paired bootstrap and the exact duplicate-excluded comparison both
   confirm the advantage.
2. **Before any innovation claim about RACAF — a controlled multi-seed repeat** of RACAF vs NO-RACAF
   under the identical protocol, with per-run initialisation recorded, to measure run-to-run variance
   and whether the gap persists. Pre-register the number of seeds and the comparison before running.
   This is now the primary open item.
3. **Proceed with IDRiD external evaluation of the finalized RACAF model** as pre-registered, and report
   RACAF as **not demonstrated superior** — the ablation does not invalidate that experiment, which
   asks a different question. Evaluating the NO-RACAF archive on IDRiD as well would be informative, but
   only as a separate, explicitly labelled secondary analysis added to the protocol before it runs.
4. **Do not redesign or retune RACAF on the basis of this result.** No technical defect was found, and
   changing the architecture after seeing this validation outcome would itself be selection on the
   validation split.
5. **Do not silently replace the delivery model.** NO-RACAF was selected on the same validation split
   (duplicate contamination now shown not to be the explanation), over more epochs, and is poorly
   calibrated. Any change of delivery model is a decision to make explicitly after step 2.

---

## 15. Reproducibility and provenance

| | RACAF | NO-RACAF |
|---|---|---|
| Notebook | `colab/notebooks/stage08_corn_classifier.ipynb` | `colab/notebooks/stage08_corn_classifier_racaf_ablation.ipynb` (executed working copy) |
| Experiment | `2026-09-12_02-45-05` | `2026-09-13_04-11-07` |
| Config hash | `3f549e1638d9409f7862f1e799bd7052` | `adf02bb3f8aa6ab93843036bad6b339e` |
| Checkpoint git commit | `87d56719726c82004b3561d8e14f101ab8e85964` | `f9dbc52647ffddd0aecdf92aacd9330b2d967623` |
| BEST SHA-256 | `7b780c23…f9417b8b9a` | `61619656…fe5d2e773` |
| Archive | `…/2026-09-12_02-45-05_BEST` | `…/2026-09-13_04-11-07_NO_RACAF_BEST` |
| Split SHA-256 | `f512a7a0…d965837` | same |
| Diagnostic checks | 62/62 | 62/62 |

**Computations added by this analysis** (not in either notebook): QWK decomposition by error distance
and true grade; a conservative unpaired bootstrap of ΔQWK (5000 resamples, seed 20260913, from the
recorded confusion matrices, superseded by the exact paired result in §9–§10); duplicate-exclusion
scenarios and per-model feasible extremes (superseded by the exact result); matched-epoch and window
statistics from the recorded histories.

**Superseded by the exact per-image audit** (see
[RACAF_Duplicate_Contamination_Audit.md](RACAF_Duplicate_Contamination_Audit.md), which read both
experiments' `evaluation/per_sample_corn_predictions.csv` directly from Drive, read-only, no
training): the exact paired bootstrap of ΔQWK; the exact duplicate-excluded metrics (n = 689); the
exact McNemar discordance test; and the independently-reproduced 41 duplicate IDs (MD5 + full byte
comparison over `datasets/APTOS2019/raw/train_images` against
`dataset_splits/aptos2019_train_val_split.csv`, confirmed identical to the earlier content-hash
audit's list).

**Still not available:** RACAF gate or reliability-conditioned diagnostics (neither notebook records
them, and the ablation has none by construction); run-to-run variance (one run per architecture).
